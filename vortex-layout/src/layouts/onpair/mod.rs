// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A structural layout for OnPair-compressed string columns.
//!
//! The OnPair array encoding is per-chunk and self-contained, so every chunk
//! carries the dictionary it was trained on. At the default 8192-row block size
//! that dictionary dominates the column's size on disk.
//!
//! This layout hoists the dictionary out of the chunks: it holds one dictionary
//! as two auxiliary children written once, and the code stream plus per-row
//! bookkeeping as chunked children. A reader reassembles an ordinary
//! [`OnPairArray`] for any row range from the shared dictionary and that range's
//! codes, so every existing OnPair kernel applies unchanged.
//!
//! `codes` lives in *token* space while the other children live in row space —
//! the same split [`ListLayout`] has between its `elements` and `offsets`.
//!
//! [`OnPairArray`]: vortex_onpair::OnPairArray
//! [`ListLayout`]: crate::layouts::list::ListLayout

mod expr;
pub mod reader;
#[cfg(test)]
mod tests;
pub mod writer;

use std::sync::Arc;

use vortex_array::ProstMetadata;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
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
use crate::children::OwnedLayoutChildren;
use crate::layouts::onpair::reader::OnPairReader;
use crate::segments::SegmentSource;

/// Child index of the read-padded dictionary blob.
pub const DICT_BYTES_CHILD_INDEX: usize = 0;
/// Child index of the dictionary token offsets.
pub const DICT_OFFSETS_CHILD_INDEX: usize = 1;
/// Child index of the token code stream.
pub const CODES_CHILD_INDEX: usize = 2;
/// Child index of the per-row token boundaries.
pub const CODES_OFFSETS_CHILD_INDEX: usize = 3;
/// Child index of the per-row decoded byte lengths.
pub const UNCOMPRESSED_LENGTHS_CHILD_INDEX: usize = 4;
/// Child index of the optional validity child.
pub const VALIDITY_CHILD_INDEX: usize = 5;
/// Number of children for a non-nullable column.
pub const NUM_CHILDREN_NON_NULLABLE: usize = 5;

/// The dictionary blob is a byte buffer, held as a primitive child so it reads
/// back zero-copy with its trailing read-padding intact.
const DICT_BYTES_PTYPE: PType = PType::U8;
/// `onpair::CompactDictionary` offsets are `u32`.
const DICT_OFFSETS_PTYPE: PType = PType::U32;
/// `onpair::Token` is `u16`.
const CODES_PTYPE: PType = PType::U16;

/// OnPair layout vtable.
#[derive(Clone, Debug)]
pub struct OnPair;

/// OnPair-layout-specific data.
///
/// Only the two child ptypes a writer may legitimately choose are recorded; the
/// rest are fixed by the OnPair format (see [`DICT_BYTES_PTYPE`],
/// [`DICT_OFFSETS_PTYPE`], [`CODES_PTYPE`]) or implied by the column's dtype.
#[derive(Clone, Debug)]
pub struct OnPairLayoutData {
    codes_offsets_ptype: PType,
    uncompressed_lengths_ptype: PType,
}

/// A string column shredded into one shared OnPair dictionary plus a chunked
/// code stream.
pub type OnPairLayout = Layout<OnPair>;

impl VTable for OnPair {
    type LayoutData = OnPairLayoutData;
    type Metadata = ProstMetadata<OnPairLayoutMetadata>;

    fn id(&self) -> LayoutId {
        static ID: CachedId = CachedId::new("vortex.onpair");
        *ID
    }

    fn metadata(layout: &Layout<Self>) -> Self::Metadata {
        ProstMetadata(OnPairLayoutMetadata::new(
            layout.codes_offsets_ptype,
            layout.uncompressed_lengths_ptype,
        ))
    }

    fn deserialize(
        &self,
        args: &LayoutDeserializeArgs<'_>,
        metadata: &OnPairLayoutMetadata,
    ) -> VortexResult<Self::LayoutData> {
        vortex_ensure!(
            matches!(args.dtype, DType::Utf8(_) | DType::Binary(_)),
            "OnPairLayout requires a Utf8 or Binary dtype, got {}",
            args.dtype
        );
        OnPairLayout::validate_children(args.dtype, args.children.nchildren())?;

        let codes_offsets_ptype = metadata.codes_offsets_ptype();
        let uncompressed_lengths_ptype = metadata.uncompressed_lengths_ptype();

        args.children
            .child(DICT_BYTES_CHILD_INDEX, &non_nullable(DICT_BYTES_PTYPE))?;
        let dict_offsets = args
            .children
            .child(DICT_OFFSETS_CHILD_INDEX, &non_nullable(DICT_OFFSETS_PTYPE))?;
        vortex_ensure!(
            dict_offsets.row_count() >= 1,
            "OnPair dict_offsets must have at least one entry"
        );

        // `codes` is in token space, so its row count is bounded by the offsets
        // rather than by the parent. The decode path checks that bound.
        args.children
            .child(CODES_CHILD_INDEX, &non_nullable(CODES_PTYPE))?;

        let codes_offsets = args.children.child(
            CODES_OFFSETS_CHILD_INDEX,
            &non_nullable(codes_offsets_ptype),
        )?;
        vortex_ensure!(
            codes_offsets.row_count() == args.row_count + 1,
            "OnPair codes_offsets row count {} does not match parent row count + 1 ({})",
            codes_offsets.row_count(),
            args.row_count + 1
        );

        let uncompressed_lengths = args.children.child(
            UNCOMPRESSED_LENGTHS_CHILD_INDEX,
            &non_nullable(uncompressed_lengths_ptype),
        )?;
        vortex_ensure!(
            uncompressed_lengths.row_count() == args.row_count,
            "OnPair uncompressed_lengths row count does not match parent"
        );

        if args.dtype.is_nullable() {
            let validity = args
                .children
                .child(VALIDITY_CHILD_INDEX, &DType::Bool(Nullability::NonNullable))?;
            vortex_ensure!(
                validity.row_count() == args.row_count,
                "OnPair validity row count does not match parent"
            );
        }

        Ok(OnPairLayoutData {
            codes_offsets_ptype,
            uncompressed_lengths_ptype,
        })
    }

    fn nslots(_layout: &Layout<Self>) -> usize {
        // Every child is always slotted; validity is present only for a nullable column.
        VALIDITY_CHILD_INDEX + 1
    }

    fn slot_to_child(layout: &Layout<Self>, slot: usize) -> Option<usize> {
        match slot {
            DICT_BYTES_CHILD_INDEX
            | DICT_OFFSETS_CHILD_INDEX
            | CODES_CHILD_INDEX
            | CODES_OFFSETS_CHILD_INDEX
            | UNCOMPRESSED_LENGTHS_CHILD_INDEX => Some(slot),
            VALIDITY_CHILD_INDEX => layout.dtype().is_nullable().then_some(VALIDITY_CHILD_INDEX),
            _ => None,
        }
    }

    fn child_dtype(layout: &Layout<Self>, slot: usize) -> VortexResult<DType> {
        match slot {
            DICT_BYTES_CHILD_INDEX => Ok(non_nullable(DICT_BYTES_PTYPE)),
            DICT_OFFSETS_CHILD_INDEX => Ok(non_nullable(DICT_OFFSETS_PTYPE)),
            CODES_CHILD_INDEX => Ok(non_nullable(CODES_PTYPE)),
            CODES_OFFSETS_CHILD_INDEX => Ok(non_nullable(layout.codes_offsets_ptype)),
            UNCOMPRESSED_LENGTHS_CHILD_INDEX => Ok(non_nullable(layout.uncompressed_lengths_ptype)),
            VALIDITY_CHILD_INDEX if layout.dtype().is_nullable() => {
                Ok(DType::Bool(Nullability::NonNullable))
            }
            _ => vortex_bail!("Invalid child index {slot} for OnPairLayout"),
        }
    }

    fn child_type(layout: &Layout<Self>, slot: usize) -> LayoutChildType {
        match slot {
            DICT_BYTES_CHILD_INDEX => LayoutChildType::Auxiliary("dict_bytes".into()),
            DICT_OFFSETS_CHILD_INDEX => LayoutChildType::Auxiliary("dict_offsets".into()),
            CODES_CHILD_INDEX => LayoutChildType::Auxiliary("codes".into()),
            CODES_OFFSETS_CHILD_INDEX => LayoutChildType::Auxiliary("codes_offsets".into()),
            // Row-aligned with the parent, and the child splits are registered from.
            UNCOMPRESSED_LENGTHS_CHILD_INDEX => {
                LayoutChildType::Transparent("uncompressed_lengths".into())
            }
            VALIDITY_CHILD_INDEX if layout.dtype().is_nullable() => {
                LayoutChildType::Transparent("validity".into())
            }
            _ => vortex_panic!("Invalid child index {slot} for OnPairLayout"),
        }
    }

    fn new_reader(
        layout: &Layout<Self>,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(OnPairReader::try_new(
            layout.clone(),
            name,
            segment_source,
            session.clone(),
            ctx,
        )?))
    }
}

impl Layout<OnPair> {
    /// Construct an OnPair layout from its children.
    ///
    /// `codes_offsets` are cumulative over the whole node, indexing into the
    /// concatenated `codes` child, and carry the usual extra trailing entry.
    pub fn new(
        dtype: DType,
        dict_bytes: LayoutRef,
        dict_offsets: LayoutRef,
        codes: LayoutRef,
        codes_offsets: LayoutRef,
        uncompressed_lengths: LayoutRef,
        validity: Option<LayoutRef>,
    ) -> Self {
        let row_count = uncompressed_lengths.row_count();
        let codes_offsets_ptype = codes_offsets.dtype().as_ptype();
        let uncompressed_lengths_ptype = uncompressed_lengths.dtype().as_ptype();
        let mut children = vec![
            dict_bytes,
            dict_offsets,
            codes,
            codes_offsets,
            uncompressed_lengths,
        ];
        children.extend(validity);
        Self::validate_children(&dtype, children.len())
            .vortex_expect("invalid OnPair layout children");
        LayoutParts::new(
            OnPair,
            dtype,
            row_count,
            Vec::new(),
            OwnedLayoutChildren::layout_children(children),
            OnPairLayoutData {
                codes_offsets_ptype,
                uncompressed_lengths_ptype,
            },
        )
        .into_typed()
    }

    /// Returns the read-padded dictionary blob child.
    pub fn dict_bytes(&self) -> VortexResult<LayoutRef> {
        self.required_slot(DICT_BYTES_CHILD_INDEX, "dict_bytes")
    }

    /// Returns the dictionary offsets child.
    pub fn dict_offsets(&self) -> VortexResult<LayoutRef> {
        self.required_slot(DICT_OFFSETS_CHILD_INDEX, "dict_offsets")
    }

    /// Returns the token code stream child.
    pub fn codes(&self) -> VortexResult<LayoutRef> {
        self.required_slot(CODES_CHILD_INDEX, "codes")
    }

    /// Returns the per-row token boundaries child.
    pub fn codes_offsets(&self) -> VortexResult<LayoutRef> {
        self.required_slot(CODES_OFFSETS_CHILD_INDEX, "codes_offsets")
    }

    /// Returns the per-row decoded byte lengths child.
    pub fn uncompressed_lengths(&self) -> VortexResult<LayoutRef> {
        self.required_slot(UNCOMPRESSED_LENGTHS_CHILD_INDEX, "uncompressed_lengths")
    }

    /// Returns the optional validity child, present exactly when the column is
    /// nullable.
    pub fn validity(&self) -> VortexResult<Option<LayoutRef>> {
        self.slot(VALIDITY_CHILD_INDEX)
    }

    /// Returns the integer ptype used by `codes_offsets`.
    pub fn codes_offsets_ptype(&self) -> PType {
        self.codes_offsets_ptype
    }

    /// Returns the integer ptype used by `uncompressed_lengths`.
    pub fn uncompressed_lengths_ptype(&self) -> PType {
        self.uncompressed_lengths_ptype
    }

    fn required_slot(&self, slot: usize, name: &str) -> VortexResult<LayoutRef> {
        self.slot(slot)?
            .ok_or_else(|| vortex_err!("OnPairLayout {name} slot is absent"))
    }

    fn validate_children(dtype: &DType, nchildren: usize) -> VortexResult<()> {
        let expected = NUM_CHILDREN_NON_NULLABLE + usize::from(dtype.is_nullable());
        vortex_ensure!(
            nchildren == expected,
            "OnPairLayout expects {expected} children, got {nchildren}"
        );
        Ok(())
    }
}

fn non_nullable(ptype: PType) -> DType {
    DType::Primitive(ptype, Nullability::NonNullable)
}

#[derive(prost::Message)]
pub struct OnPairLayoutMetadata {
    #[prost(enumeration = "PType", tag = "1")]
    codes_offsets_ptype: i32,
    #[prost(enumeration = "PType", tag = "2")]
    uncompressed_lengths_ptype: i32,
}

impl OnPairLayoutMetadata {
    pub fn new(codes_offsets_ptype: PType, uncompressed_lengths_ptype: PType) -> Self {
        let mut metadata = Self::default();
        metadata.set_codes_offsets_ptype(codes_offsets_ptype);
        metadata.set_uncompressed_lengths_ptype(uncompressed_lengths_ptype);
        metadata
    }
}
