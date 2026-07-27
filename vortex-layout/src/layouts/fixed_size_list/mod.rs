// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An experimental structural layout for fixed-size-list columns.
//!
//! [`FixedSizeListLayout`] decomposes a fixed-size list into independently configurable
//! `elements` and optional `validity` child layouts. The fixed list size maps outer rows directly
//! into element ranges, so projections can skip elements belonging exclusively to unselected
//! leading and trailing rows without storing an offsets child.

mod expr;
mod reader;
pub mod writer;

use std::sync::Arc;

use reader::FixedSizeListReader;
use vortex_array::EmptyMetadata;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
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
use crate::children::LayoutChildren;
use crate::children::OwnedLayoutChildren;
use crate::segments::SegmentSource;

/// Child index of the `elements` layout.
pub const ELEMENTS_CHILD_INDEX: usize = 0;
/// Child index of the `validity` layout, only present when the fixed-size list dtype is nullable.
pub const VALIDITY_CHILD_INDEX: usize = 1;

/// Fixed-size-list layout vtable.
#[derive(Clone, Debug)]
pub struct FixedSizeList;

/// Backwards-compatible name for the fixed-size-list layout plugin.
pub use FixedSizeList as FixedSizeListLayoutEncoding;

/// A fixed-size-list layout shredded into elements and optional validity children.
pub type FixedSizeListLayout = Layout<FixedSizeList>;

impl VTable for FixedSizeList {
    type LayoutData = ();
    type Metadata = EmptyMetadata;

    fn id(&self) -> LayoutId {
        static ID: CachedId = CachedId::new("vortex.fixed_size_list");
        *ID
    }

    fn metadata(_layout: &Layout<Self>) -> Self::Metadata {
        EmptyMetadata
    }

    fn deserialize(
        &self,
        args: &LayoutDeserializeArgs<'_>,
        _metadata: &EmptyMetadata,
    ) -> VortexResult<Self::LayoutData> {
        FixedSizeListLayout::validate_children(args.dtype, args.row_count, args.children)?;

        let element_dtype = args
            .dtype
            .as_fixed_size_list_element_opt()
            .ok_or_else(|| vortex_err!("FixedSizeListLayout requires a FixedSizeList dtype"))?;
        args.children.child(ELEMENTS_CHILD_INDEX, element_dtype)?;
        if args.dtype.is_nullable() {
            args.children
                .child(VALIDITY_CHILD_INDEX, &DType::Bool(Nullability::NonNullable))?;
        }
        Ok(())
    }

    fn child_dtype(layout: &Layout<Self>, idx: usize) -> VortexResult<DType> {
        match idx {
            ELEMENTS_CHILD_INDEX => layout
                .dtype()
                .as_fixed_size_list_element_opt()
                .map(|dtype| dtype.as_ref().clone())
                .ok_or_else(|| vortex_err!("FixedSizeListLayout requires a FixedSizeList dtype")),
            VALIDITY_CHILD_INDEX if layout.dtype().is_nullable() => {
                Ok(DType::Bool(Nullability::NonNullable))
            }
            _ => vortex_bail!("Invalid child index {idx} for FixedSizeListLayout"),
        }
    }

    fn child_type(layout: &Layout<Self>, idx: usize) -> LayoutChildType {
        match idx {
            ELEMENTS_CHILD_INDEX => LayoutChildType::Auxiliary("elements".into()),
            VALIDITY_CHILD_INDEX if layout.dtype().is_nullable() => {
                LayoutChildType::Auxiliary("validity".into())
            }
            _ => vortex_panic!("Invalid child index {idx} for FixedSizeListLayout"),
        }
    }

    fn new_reader(
        layout: &Layout<Self>,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(FixedSizeListReader::try_new(
            layout.clone(),
            name,
            segment_source,
            session.clone(),
            ctx,
        )?))
    }
}

impl Layout<FixedSizeList> {
    /// Construct a fixed-size-list layout from its components.
    ///
    /// # Invariants
    ///
    /// - `dtype` must be a [`DType::FixedSizeList`].
    /// - `elements.row_count() == row_count * list_size`.
    /// - `validity` is present iff `dtype.is_nullable()`.
    pub fn new(
        row_count: u64,
        dtype: DType,
        elements: LayoutRef,
        validity: Option<LayoutRef>,
    ) -> Self {
        let mut child_layouts = vec![elements];
        child_layouts.extend(validity);
        let children = OwnedLayoutChildren::layout_children(child_layouts);
        Self::validate_children(&dtype, row_count, children.as_ref())
            .vortex_expect("invalid fixed-size-list children");
        LayoutParts::new(FixedSizeList, dtype, row_count, Vec::new(), children, ()).into_typed()
    }

    /// Returns the elements child.
    pub fn elements(&self) -> VortexResult<LayoutRef> {
        self.child(ELEMENTS_CHILD_INDEX)
    }

    /// Returns the optional validity child.
    pub fn validity(&self) -> VortexResult<Option<LayoutRef>> {
        self.dtype()
            .is_nullable()
            .then(|| self.child(VALIDITY_CHILD_INDEX))
            .transpose()
    }

    /// The fixed number of elements in each list row.
    #[inline]
    pub fn list_size(&self) -> u32 {
        match self.dtype() {
            DType::FixedSizeList(_, list_size, _) => *list_size,
            _ => vortex_panic!("FixedSizeListLayout dtype must be FixedSizeList"),
        }
    }

    /// The dtype of the inner elements column.
    pub fn elements_dtype(&self) -> &DType {
        self.dtype()
            .as_fixed_size_list_element_opt()
            .vortex_expect("FixedSizeListLayout dtype must be FixedSizeList")
    }

    fn validate_child_count(dtype: &DType, nchildren: usize) -> VortexResult<()> {
        let expected = 1 + usize::from(dtype.is_nullable());
        vortex_ensure!(
            nchildren == expected,
            "FixedSizeListLayout expects {expected} children, got {nchildren}"
        );
        Ok(())
    }

    fn validate_children(
        dtype: &DType,
        row_count: u64,
        children: &dyn LayoutChildren,
    ) -> VortexResult<()> {
        Self::validate_child_count(dtype, children.nchildren())?;
        let DType::FixedSizeList(element_dtype, list_size, _) = dtype else {
            vortex_bail!("FixedSizeListLayout requires a FixedSizeList dtype, got {dtype}");
        };
        children.child(ELEMENTS_CHILD_INDEX, element_dtype)?;
        let expected_elements = row_count
            .checked_mul(u64::from(*list_size))
            .ok_or_else(|| vortex_err!("fixed-size list elements row count overflow"))?;
        let actual_elements = children.child_row_count(ELEMENTS_CHILD_INDEX);
        vortex_ensure!(
            actual_elements == expected_elements,
            "FixedSizeListLayout elements row count {actual_elements} does not match expected {expected_elements}"
        );
        if dtype.is_nullable() {
            children.child(VALIDITY_CHILD_INDEX, &DType::Bool(Nullability::NonNullable))?;
            let validity_rows = children.child_row_count(VALIDITY_CHILD_INDEX);
            vortex_ensure!(
                validity_rows == row_count,
                "FixedSizeListLayout validity row count {validity_rows} does not match row count {row_count}"
            );
        }
        Ok(())
    }
}
