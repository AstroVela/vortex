// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An experimental structural layout for list-typed columns.

mod expr;
mod reader;
pub mod writer;

use std::sync::Arc;

use reader::ListReader;
use vortex_array::ProstMetadata;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::Layout;
use crate::LayoutChildType;
use crate::LayoutDeserializeArgs;
use crate::LayoutId;
use crate::LayoutParts;
use crate::LayoutReaderContext;
use crate::LayoutReaderRef;
use crate::LayoutRef;
use crate::VTable;
use crate::children::OwnedLayoutChildren;
use crate::segments::SegmentSource;

/// Child index of the elements layout.
pub const ELEMENTS_CHILD_INDEX: usize = 0;
/// Child index of the offsets layout.
pub const OFFSETS_CHILD_INDEX: usize = 1;
/// Child index of the optional validity layout.
pub const VALIDITY_CHILD_INDEX: usize = 2;
/// Number of children for a non-nullable list.
pub const NUM_CHILDREN_NON_NULLABLE: usize = 2;

/// List layout vtable.
#[derive(Clone, Debug)]
pub struct List;

/// Backwards-compatible name for the list layout plugin.
pub use List as ListLayoutEncoding;

/// List-layout-specific data.
#[derive(Clone, Debug)]
pub struct ListData {
    offsets_ptype: PType,
    block_boundaries: Arc<[ListBlockBoundary]>,
}

/// Cumulative row boundaries for one input block of a structural list layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ListBlockBoundary {
    outer_row_end: u64,
    element_row_end: u64,
}

impl ListBlockBoundary {
    pub(super) fn new(outer_row_end: u64, element_row_end: u64) -> Self {
        Self {
            outer_row_end,
            element_row_end,
        }
    }

    pub(super) fn outer_row_end(self) -> u64 {
        self.outer_row_end
    }

    pub(super) fn element_row_end(self) -> u64 {
        self.element_row_end
    }
}

/// A list layout shredded into elements, offsets, and optional validity children.
pub type ListLayout = Layout<List>;

impl VTable for List {
    type LayoutData = ListData;
    type Metadata = ProstMetadata<ListLayoutMetadata>;

    fn id(&self) -> LayoutId {
        static ID: CachedId = CachedId::new("vortex.list");
        *ID
    }

    fn metadata(layout: &Layout<Self>) -> Self::Metadata {
        ProstMetadata(ListLayoutMetadata::new_with_boundaries(
            layout.offsets_ptype,
            &layout.block_boundaries,
        ))
    }

    fn deserialize(
        &self,
        args: &LayoutDeserializeArgs<'_>,
        metadata: &ListLayoutMetadata,
    ) -> VortexResult<Self::LayoutData> {
        ListLayout::validate_children(args.dtype, args.children.nchildren())?;
        let elements_dtype = args
            .dtype
            .as_list_element_opt()
            .ok_or_else(|| vortex_err!("ListLayout requires a List dtype, got {}", args.dtype))?;
        let elements = args.children.child(ELEMENTS_CHILD_INDEX, elements_dtype)?;
        let offsets_dtype = DType::Primitive(metadata.offsets_ptype(), Nullability::NonNullable);
        let offsets = args.children.child(OFFSETS_CHILD_INDEX, &offsets_dtype)?;
        vortex_error::vortex_ensure!(
            offsets.row_count().saturating_sub(1) == args.row_count,
            "List offsets row count does not match parent"
        );
        if args.dtype.is_nullable() {
            let validity = args
                .children
                .child(VALIDITY_CHILD_INDEX, &DType::Bool(Nullability::NonNullable))?;
            vortex_error::vortex_ensure!(
                validity.row_count() == args.row_count,
                "List validity row count does not match parent"
            );
        }
        let block_boundaries = metadata
            .block_boundaries
            .iter()
            .map(|boundary| {
                ListBlockBoundary::new(boundary.outer_row_end, boundary.element_row_end)
            })
            .collect::<Vec<_>>();
        validate_block_boundaries(args.row_count, elements.row_count(), &block_boundaries)?;

        Ok(ListData {
            offsets_ptype: metadata.offsets_ptype(),
            block_boundaries: block_boundaries.into(),
        })
    }

    fn nslots(_layout: &Layout<Self>) -> usize {
        // Elements, offsets, and an always-slotted (optionally present) validity child.
        VALIDITY_CHILD_INDEX + 1
    }

    fn slot_to_child(layout: &Layout<Self>, slot: usize) -> Option<usize> {
        match slot {
            ELEMENTS_CHILD_INDEX | OFFSETS_CHILD_INDEX => Some(slot),
            VALIDITY_CHILD_INDEX => layout.dtype().is_nullable().then_some(VALIDITY_CHILD_INDEX),
            _ => None,
        }
    }

    fn child_dtype(layout: &Layout<Self>, idx: usize) -> VortexResult<DType> {
        match idx {
            ELEMENTS_CHILD_INDEX => layout
                .dtype()
                .as_list_element_opt()
                .map(|dtype| dtype.as_ref().clone())
                .ok_or_else(|| vortex_err!("ListLayout requires a List dtype")),
            OFFSETS_CHILD_INDEX => Ok(DType::Primitive(
                layout.offsets_ptype,
                Nullability::NonNullable,
            )),
            VALIDITY_CHILD_INDEX if layout.dtype().is_nullable() => {
                Ok(DType::Bool(Nullability::NonNullable))
            }
            _ => vortex_bail!("Invalid child index {idx} for ListLayout"),
        }
    }

    fn child_type(layout: &Layout<Self>, idx: usize) -> LayoutChildType {
        match idx {
            ELEMENTS_CHILD_INDEX => LayoutChildType::Auxiliary("elements".into()),
            OFFSETS_CHILD_INDEX => LayoutChildType::Auxiliary("offsets".into()),
            VALIDITY_CHILD_INDEX if layout.dtype().is_nullable() => {
                LayoutChildType::Auxiliary("validity".into())
            }
            _ => vortex_panic!("Invalid child index {idx} for ListLayout"),
        }
    }

    fn new_reader(
        layout: &Layout<Self>,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(ListReader::try_new(
            layout.clone(),
            name,
            segment_source,
            session.clone(),
            ctx,
        )?))
    }
}

impl Layout<List> {
    /// Construct a list layout from its children.
    pub fn new(
        dtype: DType,
        elements: LayoutRef,
        offsets: LayoutRef,
        validity: Option<LayoutRef>,
    ) -> Self {
        Self::new_with_boundaries(dtype, elements, offsets, validity, Vec::new())
    }

    pub(super) fn new_with_boundaries(
        dtype: DType,
        elements: LayoutRef,
        offsets: LayoutRef,
        validity: Option<LayoutRef>,
        block_boundaries: Vec<ListBlockBoundary>,
    ) -> Self {
        let row_count = offsets.row_count().saturating_sub(1);
        let offsets_ptype = offsets.dtype().as_ptype();
        validate_block_boundaries(row_count, elements.row_count(), &block_boundaries)
            .vortex_expect("invalid list block boundaries");
        let mut children = vec![elements, offsets];
        children.extend(validity);
        Self::validate_children(&dtype, children.len()).vortex_expect("invalid list children");
        LayoutParts::new(
            List,
            dtype,
            row_count,
            Vec::new(),
            OwnedLayoutChildren::layout_children(children),
            ListData {
                offsets_ptype,
                block_boundaries: block_boundaries.into(),
            },
        )
        .into_typed()
    }

    /// Returns the elements child.
    pub fn elements(&self) -> VortexResult<LayoutRef> {
        self.slot(ELEMENTS_CHILD_INDEX)?
            .ok_or_else(|| vortex_err!("ListLayout elements slot is absent"))
    }

    /// Returns the offsets child.
    pub fn offsets(&self) -> VortexResult<LayoutRef> {
        self.slot(OFFSETS_CHILD_INDEX)?
            .ok_or_else(|| vortex_err!("ListLayout offsets slot is absent"))
    }

    /// Returns the optional validity child.
    pub fn validity(&self) -> VortexResult<Option<LayoutRef>> {
        self.slot(VALIDITY_CHILD_INDEX)
    }

    /// Returns the integer ptype used by offsets.
    pub fn offsets_ptype(&self) -> PType {
        self.offsets_ptype
    }

    pub(super) fn block_boundaries(&self) -> &[ListBlockBoundary] {
        &self.block_boundaries
    }

    /// Returns the list element dtype.
    pub fn elements_dtype(&self) -> &DType {
        self.dtype()
            .as_list_element_opt()
            .vortex_expect("ListLayout dtype must be a List")
    }

    fn validate_children(dtype: &DType, nchildren: usize) -> VortexResult<()> {
        let expected = NUM_CHILDREN_NON_NULLABLE + usize::from(dtype.is_nullable());
        vortex_ensure_eq!(nchildren, expected);
        Ok(())
    }
}

#[derive(prost::Message)]
pub struct ListLayoutMetadata {
    #[prost(enumeration = "PType", tag = "1")]
    offsets_ptype: i32,
    #[prost(message, repeated, tag = "2")]
    block_boundaries: Vec<ListBlockBoundaryMetadata>,
}

#[derive(Clone, PartialEq, Eq, prost::Message)]
struct ListBlockBoundaryMetadata {
    #[prost(uint64, tag = "1")]
    outer_row_end: u64,
    #[prost(uint64, tag = "2")]
    element_row_end: u64,
}

impl ListLayoutMetadata {
    pub fn new(offsets_ptype: PType) -> Self {
        Self::new_with_boundaries(offsets_ptype, &[])
    }

    fn new_with_boundaries(offsets_ptype: PType, block_boundaries: &[ListBlockBoundary]) -> Self {
        let mut metadata = Self::default();
        metadata.set_offsets_ptype(offsets_ptype);
        metadata.block_boundaries = block_boundaries
            .iter()
            .map(|boundary| ListBlockBoundaryMetadata {
                outer_row_end: boundary.outer_row_end(),
                element_row_end: boundary.element_row_end(),
            })
            .collect();
        metadata
    }
}

fn validate_block_boundaries(
    outer_row_count: u64,
    element_row_count: u64,
    block_boundaries: &[ListBlockBoundary],
) -> VortexResult<()> {
    let Some(last) = block_boundaries.last() else {
        return Ok(());
    };

    vortex_error::vortex_ensure!(
        block_boundaries[0].outer_row_end != 0,
        "List block outer-row boundaries must not contain zero"
    );
    vortex_error::vortex_ensure!(
        block_boundaries
            .windows(2)
            .all(|pair| pair[0].outer_row_end < pair[1].outer_row_end),
        "List block outer-row boundaries must be strictly increasing"
    );
    vortex_error::vortex_ensure!(
        block_boundaries
            .windows(2)
            .all(|pair| pair[0].element_row_end <= pair[1].element_row_end),
        "List block element-row boundaries must be non-decreasing"
    );
    vortex_error::vortex_ensure_eq!(last.outer_row_end, outer_row_count);
    vortex_error::vortex_ensure_eq!(last.element_row_end, element_row_count);
    Ok(())
}

#[cfg(test)]
mod tests {
    use vortex_array::DeserializeMetadata;
    use vortex_array::SerializeMetadata;

    use super::*;

    #[test]
    fn block_boundaries_round_trip_through_metadata() -> VortexResult<()> {
        let boundaries = [
            ListBlockBoundary::new(8, 21),
            ListBlockBoundary::new(16, 50),
        ];
        let encoded = ProstMetadata(ListLayoutMetadata::new_with_boundaries(
            PType::U64,
            &boundaries,
        ))
        .serialize();
        let decoded =
            <ProstMetadata<ListLayoutMetadata> as DeserializeMetadata>::deserialize(&encoded)?;

        assert_eq!(decoded.offsets_ptype(), PType::U64);
        assert_eq!(
            decoded
                .block_boundaries
                .iter()
                .map(|boundary| {
                    ListBlockBoundary::new(boundary.outer_row_end, boundary.element_row_end)
                })
                .collect::<Vec<_>>(),
            boundaries
        );
        Ok(())
    }

    #[test]
    fn legacy_metadata_has_no_block_boundaries() -> VortexResult<()> {
        let encoded = ProstMetadata(ListLayoutMetadata::new(PType::U32)).serialize();
        let decoded =
            <ProstMetadata<ListLayoutMetadata> as DeserializeMetadata>::deserialize(&encoded)?;

        assert_eq!(decoded.offsets_ptype(), PType::U32);
        assert!(decoded.block_boundaries.is_empty());
        Ok(())
    }
}
