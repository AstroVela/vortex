// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod expr;
mod reader;
pub mod writer;

use std::sync::Arc;

use reader::ListReader;
use vortex_array::DeserializeMetadata;
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

use crate::LayoutBuildContext;
use crate::LayoutChildType;
use crate::LayoutEncodingRef;
use crate::LayoutId;
use crate::LayoutReaderContext;
use crate::LayoutReaderRef;
use crate::LayoutRef;
use crate::VTable;
use crate::children::LayoutChildren;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;
use crate::vtable;

/// Child index of the `elements` layout.
pub const ELEMENTS_CHILD_INDEX: usize = 0;
/// Child index of the `offsets` layout.
pub const OFFSETS_CHILD_INDEX: usize = 1;
/// Child index of the `validity` layout (only present when the list dtype is nullable).
pub const VALIDITY_CHILD_INDEX: usize = 2;

/// Number of children when the list dtype is non-nullable.
pub const NUM_CHILDREN_NON_NULLABLE: usize = 2;

vtable!(List);

impl VTable for List {
    type Layout = ListLayout;
    type Encoding = ListLayoutEncoding;
    type Metadata = ProstMetadata<ListLayoutMetadata>;

    fn id(_encoding: &Self::Encoding) -> LayoutId {
        static ID: CachedId = CachedId::new("vortex.list");
        *ID
    }

    fn encoding(_layout: &Self::Layout) -> LayoutEncodingRef {
        LayoutEncodingRef::new_ref(ListLayoutEncoding.as_ref())
    }

    fn row_count(layout: &Self::Layout) -> u64 {
        layout.row_count()
    }

    fn dtype(layout: &Self::Layout) -> &DType {
        &layout.dtype
    }

    fn metadata(layout: &Self::Layout) -> Self::Metadata {
        ProstMetadata(ListLayoutMetadata::new(
            layout.offsets_ptype(),
            layout.sample_stride(),
            layout.offset_samples().to_vec(),
        ))
    }

    fn segment_ids(_layout: &Self::Layout) -> Vec<SegmentId> {
        vec![]
    }

    fn nchildren(layout: &Self::Layout) -> usize {
        if layout.dtype.is_nullable() {
            NUM_CHILDREN_NON_NULLABLE + 1
        } else {
            NUM_CHILDREN_NON_NULLABLE
        }
    }

    fn child(layout: &Self::Layout, idx: usize) -> VortexResult<LayoutRef> {
        match (idx, layout.validity.as_ref()) {
            (ELEMENTS_CHILD_INDEX, _) => Ok(Arc::clone(&layout.elements)),
            (OFFSETS_CHILD_INDEX, _) => Ok(Arc::clone(&layout.offsets)),
            (VALIDITY_CHILD_INDEX, Some(validity)) => Ok(Arc::clone(validity)),
            _ => vortex_bail!("Invalid child index {idx} for ListLayout"),
        }
    }

    fn child_type(layout: &Self::Layout, idx: usize) -> LayoutChildType {
        match (idx, layout.validity.is_some()) {
            (ELEMENTS_CHILD_INDEX, _) => LayoutChildType::Auxiliary("elements".into()),
            (OFFSETS_CHILD_INDEX, _) => LayoutChildType::Auxiliary("offsets".into()),
            (VALIDITY_CHILD_INDEX, true) => LayoutChildType::Auxiliary("validity".into()),
            _ => vortex_panic!("Invalid child index {idx} for ListLayout"),
        }
    }

    fn new_reader(
        layout: &Self::Layout,
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

    fn build(
        _encoding: &Self::Encoding,
        dtype: &DType,
        _row_count: u64,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        _segment_ids: Vec<SegmentId>,
        children: &dyn LayoutChildren,
        _ctx: &LayoutBuildContext<'_>,
    ) -> VortexResult<Self::Layout> {
        validate_children(dtype, children.nchildren())?;

        let elements_dtype = dtype
            .as_list_element_opt()
            .ok_or_else(|| vortex_err!("ListLayout requires a List dtype, got {dtype}"))?;
        let elements = children.child(ELEMENTS_CHILD_INDEX, elements_dtype.as_ref())?;

        let offsets_dtype = DType::Primitive(metadata.offsets_ptype(), Nullability::NonNullable);
        let offsets = children.child(OFFSETS_CHILD_INDEX, &offsets_dtype)?;

        let validity = dtype
            .is_nullable()
            .then(|| children.child(VALIDITY_CHILD_INDEX, &DType::Bool(Nullability::NonNullable)))
            .transpose()?;

        Ok(ListLayout::new_with_samples(
            dtype.clone(),
            elements,
            offsets,
            validity,
            metadata.sample_stride,
            Arc::from(metadata.offset_samples.as_slice()),
        ))
    }

    fn with_children(layout: &mut Self::Layout, children: Vec<LayoutRef>) -> VortexResult<()> {
        validate_children(layout.dtype(), children.len())?;

        let mut iter = children.into_iter();
        layout.elements = iter
            .next()
            .ok_or_else(|| vortex_err!("missing elements child"))?;
        layout.offsets = iter
            .next()
            .ok_or_else(|| vortex_err!("missing offsets child"))?;
        layout.validity = layout
            .dtype
            .is_nullable()
            .then(|| {
                iter.next()
                    .ok_or_else(|| vortex_err!("missing validity child"))
            })
            .transpose()?;
        Ok(())
    }
}

/// Validates expected number of children based on `dtype`
fn validate_children(dtype: &DType, n_children: usize) -> VortexResult<()> {
    let expected = if dtype.is_nullable() {
        NUM_CHILDREN_NON_NULLABLE + 1
    } else {
        NUM_CHILDREN_NON_NULLABLE
    };

    vortex_ensure_eq!(n_children, expected);
    Ok(())
}

#[derive(Debug)]
pub struct ListLayoutEncoding;

/// Stores a list-typed array by shredding `elements`, `offsets`, and optional `validity` children.
#[derive(Clone, Debug)]
pub struct ListLayout {
    dtype: DType,
    elements: LayoutRef,
    offsets: LayoutRef,
    validity: Option<LayoutRef>,
    /// Distance, in rows, between adjacent offset samples. `0` means no samples are stored.
    sample_stride: u32,
    /// `offset_samples[k]` equals `offsets[k * sample_stride]`, always promoted to `u64`. Lets a
    /// reader bracket the elements range for a row window from resident metadata, so the elements
    /// fetch runs in parallel with the offsets fetch instead of waiting on it. Empty when sampling
    /// is disabled or the list is too small to be worth indexing.
    offset_samples: Arc<[u64]>,
}

impl ListLayout {
    /// Construct a new `ListLayout` from its components, without an offset-sample index.
    ///
    /// # Invariants
    ///
    /// - `dtype` must be a [`DType::List`].
    /// - `validity` must be `Some` iff `dtype.is_nullable()`.
    /// - `offsets.dtype()` must be a non-nullable integer.
    /// - `offsets.row_count()` is the Arrow-canonical `n+1` for `n` lists (or `0` for empty).
    /// - When present, `validity.row_count() == offsets.row_count().saturating_sub(1)`.
    pub fn new(
        dtype: DType,
        elements: LayoutRef,
        offsets: LayoutRef,
        validity: Option<LayoutRef>,
    ) -> Self {
        Self::new_with_samples(dtype, elements, offsets, validity, 0, Arc::from([]))
    }

    /// Like [`Self::new`] but carrying a pre-computed offset-sample index.
    ///
    /// `sample_stride` is the row distance between adjacent samples; pass `0` (and an empty slice)
    /// to disable sampling. When enabled, `offset_samples[k]` must equal `offsets[k * sample_stride]`.
    pub fn new_with_samples(
        dtype: DType,
        elements: LayoutRef,
        offsets: LayoutRef,
        validity: Option<LayoutRef>,
        sample_stride: u32,
        offset_samples: Arc<[u64]>,
    ) -> Self {
        Self {
            dtype,
            elements,
            offsets,
            validity,
            sample_stride,
            offset_samples,
        }
    }

    /// Number of lists in this layout.
    #[inline]
    pub fn row_count(&self) -> u64 {
        self.offsets.row_count().saturating_sub(1)
    }

    #[inline]
    pub fn elements(&self) -> &LayoutRef {
        &self.elements
    }

    #[inline]
    pub fn offsets(&self) -> &LayoutRef {
        &self.offsets
    }

    #[inline]
    pub fn validity(&self) -> Option<&LayoutRef> {
        self.validity.as_ref()
    }

    /// The integer type used for the `offsets` child layout.
    #[inline]
    pub fn offsets_ptype(&self) -> PType {
        self.offsets.dtype().as_ptype()
    }

    /// The dtype of the inner elements column.
    pub fn elements_dtype(&self) -> &DType {
        self.dtype
            .as_list_element_opt()
            .vortex_expect("ListLayout dtype must be a List")
    }

    /// Row distance between adjacent offset samples, or `0` when sampling is disabled.
    #[inline]
    pub fn sample_stride(&self) -> u32 {
        self.sample_stride
    }

    /// Cached offset samples. The `k`-th entry equals `offsets[k * sample_stride]`.
    #[inline]
    pub fn offset_samples(&self) -> &[u64] {
        &self.offset_samples
    }
}

#[derive(prost::Message)]
pub struct ListLayoutMetadata {
    #[prost(enumeration = "PType", tag = "1")]
    offsets_ptype: i32,
    /// Row distance between offset samples. `0` indicates samples are not present, in which case
    /// readers fall back to fetching offsets before elements.
    #[prost(uint32, tag = "2")]
    pub sample_stride: u32,
    /// `offset_samples[k]` is the value of the underlying `offsets` array at position
    /// `k * sample_stride`, always promoted to `u64` regardless of the offset `PType` so the
    /// reader can treat samples uniformly.
    #[prost(uint64, repeated, tag = "3")]
    pub offset_samples: Vec<u64>,
}

impl ListLayoutMetadata {
    pub fn new(offsets_ptype: PType, sample_stride: u32, offset_samples: Vec<u64>) -> Self {
        let mut metadata = Self::default();
        metadata.set_offsets_ptype(offsets_ptype);
        metadata.sample_stride = sample_stride;
        metadata.offset_samples = offset_samples;
        metadata
    }
}
