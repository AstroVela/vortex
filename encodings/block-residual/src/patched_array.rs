// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A prototype block-residual layout with child arrays for its patch and base components.
//!
//! `PatchedBlockResidual` stores the same information as [`crate::BlockResidual`], but the
//! per-block bases and the outlier high bits are real child arrays instead of buffer sections:
//!
//! - `bases`: one `u64` per block, recursively compressible.
//! - patches: the standard [`Patches`] triple (global sorted indices, high-bit values, and
//!   1024-granularity chunk offsets), so slicing and index search reuse the shared machinery.
//! - the packed low residuals stay in a single buffer because their per-block bit widths are
//!   not expressible by any existing encoding.
//!
//! The prototype supports unsigned 64-bit data, no re-slicing, and exists to compare sizes and
//! composability against the fused buffer layout.

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;

use fastlanes::BitPacking;
use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::EqMode;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::TypedArrayRef;
use vortex_array::array_slots;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::dtype::PType;
use vortex_array::patches::Patches;
use vortex_array::scalar::Scalar;
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTable;
use vortex_array::vtable::child_to_validity;
use vortex_array::vtable::validity_to_child;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::BlockResidualCodec;
use crate::codec::read_wide_bits;

const BLOCK_LEN: usize = 1024;
const WORDS_PER_WIDTH_BIT: usize = WORDS_PER_WIDTH_BIT_U32 as usize;
const WORDS_PER_WIDTH_BIT_U32: u32 = 16;

/// Block residuals with child arrays for bases and high-bit patches.
pub type PatchedBlockResidualArray = Array<PatchedBlockResidual>;

#[array_slots(PatchedBlockResidual)]
pub struct PatchedBlockResidualSlots {
    #[slot(0)]
    pub validity: Option<ArrayRef>,
    /// One `u64` base per 1024-value block.
    #[slot(1)]
    pub bases: ArrayRef,
    /// Sorted global row indices of patched values.
    #[slot(2)]
    pub patch_indices: Option<ArrayRef>,
    /// The high bits of each patched residual.
    #[slot(3)]
    pub patch_values: Option<ArrayRef>,
    /// Per-block prefix counts into the patch children.
    #[slot(4)]
    pub patch_chunk_offsets: Option<ArrayRef>,
}

#[derive(Clone, Debug)]
pub struct PatchedBlockResidualData {
    len: usize,
    payload: ByteBuffer,
    residual_widths: Buffer<u8>,
    residual_words: Buffer<u64>,
    // Derived at construction, not serialized: prefix word offsets per block.
    word_starts: Vec<u32>,
}

impl Display for PatchedBlockResidualData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "blocks: {}", self.residual_widths.len())
    }
}

impl ArrayHash for PatchedBlockResidualData {
    fn array_hash<H: Hasher>(&self, state: &mut H, accuracy: EqMode) {
        self.len.hash(state);
        self.payload.array_hash(state, accuracy);
    }
}

impl ArrayEq for PatchedBlockResidualData {
    fn array_eq(&self, other: &Self, accuracy: EqMode) -> bool {
        self.len == other.len && self.payload.array_eq(&other.payload, accuracy)
    }
}

impl PatchedBlockResidualData {
    /// Reconstruct the typed payload views from one contiguous buffer.
    fn try_new(len: usize, payload: ByteBuffer) -> VortexResult<Self> {
        let block_count = len.div_ceil(BLOCK_LEN);
        let widths_padded = block_count.next_multiple_of(8);
        vortex_ensure!(
            payload.len() >= widths_padded,
            "patched block residual payload is too short"
        );
        let residual_widths = Buffer::from_byte_buffer(
            payload.slice_with_alignment(0..block_count, Alignment::of::<u8>()),
        );
        let word_count: usize = residual_widths
            .iter()
            .map(|&width| usize::from(width) * WORDS_PER_WIDTH_BIT)
            .sum();
        vortex_ensure!(
            payload.len() == widths_padded + word_count * size_of::<u64>(),
            "patched block residual payload length is invalid"
        );
        let residual_words = Buffer::from_byte_buffer(
            payload.slice_with_alignment(widths_padded..payload.len(), Alignment::of::<u64>()),
        );
        let mut word_starts = Vec::with_capacity(block_count + 1);
        let mut start = 0u32;
        word_starts.push(start);
        for &width in residual_widths.iter() {
            start += u32::from(width) * WORDS_PER_WIDTH_BIT_U32;
            word_starts.push(start);
        }
        Ok(Self {
            len,
            payload,
            residual_widths,
            residual_words,
            word_starts,
        })
    }

    /// Return the word range of one block's packed residuals.
    fn block_words(&self, block_index: usize) -> std::ops::Range<usize> {
        self.word_starts[block_index] as usize..self.word_starts[block_index + 1] as usize
    }
}

#[derive(Clone, Debug)]
pub struct PatchedBlockResidual;

pub trait PatchedBlockResidualArrayExt:
    TypedArrayRef<PatchedBlockResidual> + PatchedBlockResidualArraySlotsExt
{
    /// Reassemble the standard [`Patches`] view over the patch children.
    fn patches(&self) -> Option<Patches> {
        let indices = self.patch_indices()?.clone();
        let values = self.patch_values()?.clone();
        let chunk_offsets = self.patch_chunk_offsets().cloned();
        // SAFETY: Validation checked sorted non-nullable indices within the array length.
        Some(unsafe {
            Patches::new_unchecked(
                self.as_ref().len(),
                0,
                indices,
                values,
                chunk_offsets.clone(),
                chunk_offsets.map(|_| 0),
            )
        })
    }
}

impl<T: TypedArrayRef<PatchedBlockResidual>> PatchedBlockResidualArrayExt for T {}

impl PatchedBlockResidual {
    /// Encode an unsigned 64-bit primitive array with child-array patches and bases.
    pub fn from_primitive(
        array: ArrayView<'_, Primitive>,
    ) -> VortexResult<PatchedBlockResidualArray> {
        vortex_ensure!(
            array.ptype() == PType::U64,
            "PatchedBlockResidual prototype requires u64, got {}",
            array.ptype()
        );
        let validity = array.validity()?;
        let values = array.as_slice::<u64>();
        let codec = BlockResidualCodec::encode_with_word_width(values, 64)?;

        let block_count = codec.blocks.len();
        let mut bases = Vec::with_capacity(block_count);
        let mut widths = Vec::with_capacity(block_count);
        let mut words = Vec::new();
        let mut patch_indices = Vec::new();
        let mut patch_values = Vec::new();
        let mut chunk_offsets = Vec::with_capacity(block_count + 1);
        chunk_offsets.push(0u32);
        for (block_index, block) in codec.blocks.iter().enumerate() {
            bases.push(block.base);
            widths.push(block.residual_width);
            words.extend_from_slice(&block.residuals);
            for (patch_index, &position) in block.patch_positions.iter().enumerate() {
                patch_indices.push(u32::try_from(
                    block_index * BLOCK_LEN + usize::from(position),
                )?);
                // SAFETY: The encoder writes fifteen readable padding bytes after the highs.
                patch_values.push(unsafe {
                    read_wide_bits(
                        &block.patch_highs,
                        patch_index * usize::from(block.high_width),
                        block.high_width,
                    )
                });
            }
            chunk_offsets.push(u32::try_from(patch_indices.len())?);
        }

        let widths_padded = block_count.next_multiple_of(8);
        let mut payload = ByteBufferMut::with_capacity_aligned(
            widths_padded + words.len() * size_of::<u64>(),
            Alignment::of::<u64>(),
        );
        payload.extend_from_slice(&widths);
        payload.extend_from_slice(&vec![0u8; widths_padded - block_count]);
        // SAFETY: u64 values contain no padding bytes.
        payload.extend_from_slice(unsafe {
            std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), words.len() * 8)
        });

        let has_patches = !patch_indices.is_empty();
        let slots = PatchedBlockResidualSlots {
            validity: validity_to_child(&validity, values.len()),
            bases: PrimitiveArray::new(Buffer::from(bases), NonNullable.into()).into_array(),
            patch_indices: has_patches.then(|| {
                PrimitiveArray::new(Buffer::from(patch_indices), NonNullable.into()).into_array()
            }),
            patch_values: has_patches.then(|| {
                PrimitiveArray::new(Buffer::from(patch_values), NonNullable.into()).into_array()
            }),
            patch_chunk_offsets: has_patches.then(|| {
                PrimitiveArray::new(Buffer::from(chunk_offsets), NonNullable.into()).into_array()
            }),
        }
        .into_slots();
        let data = PatchedBlockResidualData::try_new(values.len(), payload.freeze())?;
        let dtype = DType::Primitive(PType::U64, array.dtype().nullability());
        Array::try_from_parts(
            ArrayParts::new(PatchedBlockResidual, dtype, values.len(), data).with_slots(slots),
        )
    }
}

impl VTable for PatchedBlockResidual {
    type TypedArrayData = PatchedBlockResidualData;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.patched_block_residual");
        *ID
    }

    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(
            matches!(dtype, DType::Primitive(PType::U64, _)),
            "PatchedBlockResidual prototype requires u64, got {dtype}"
        );
        vortex_ensure!(data.len == len, "PatchedBlockResidual length differs");
        let block_count = len.div_ceil(BLOCK_LEN);
        vortex_ensure!(
            data.residual_widths.len() == block_count
                && data.residual_widths.iter().all(|&width| width <= 64),
            "PatchedBlockResidual widths are invalid"
        );
        let slots = PatchedBlockResidualSlotsView::from_slots(slots);
        let bases_dtype = DType::Primitive(PType::U64, NonNullable);
        vortex_ensure!(
            slots.bases.dtype() == &bases_dtype && slots.bases.len() == block_count,
            "PatchedBlockResidual bases child is invalid"
        );
        match (
            slots.patch_indices,
            slots.patch_values,
            slots.patch_chunk_offsets,
        ) {
            (None, None, None) => {}
            (Some(indices), Some(values), Some(chunk_offsets)) => {
                vortex_ensure!(
                    indices.dtype() == &DType::Primitive(PType::U32, NonNullable)
                        && values.dtype() == &DType::Primitive(PType::U64, NonNullable)
                        && indices.len() == values.len()
                        && !indices.is_empty(),
                    "PatchedBlockResidual patch children are invalid"
                );
                vortex_ensure!(
                    chunk_offsets.dtype() == &DType::Primitive(PType::U32, NonNullable)
                        && chunk_offsets.len() == block_count + 1,
                    "PatchedBlockResidual patch chunk offsets are invalid"
                );
            }
            _ => vortex_bail!("PatchedBlockResidual patch children must be set together"),
        }
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        1
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        match idx {
            0 => BufferHandle::new_host(array.payload.clone()),
            _ => vortex_panic!("PatchedBlockResidualArray buffer index {idx} out of bounds"),
        }
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        (idx == 0).then(|| "packed_lows".to_string())
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_ensure!(
            buffers.len() == 1,
            "PatchedBlockResidualArray expects one buffer"
        );
        let data = PatchedBlockResidualData::try_new(array.len(), host_payload(&buffers[0])?)?;
        Ok(
            ArrayParts::new(self.clone(), array.dtype().clone(), array.len(), data)
                .with_slots(array.slots().iter().cloned().collect()),
        )
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        let patch_count = array.patch_indices().map(|child| child.len()).unwrap_or(0);
        Ok(Some(u64::try_from(patch_count)?.to_le_bytes().to_vec()))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        let patch_count = usize::try_from(u64::from_le_bytes(
            metadata
                .try_into()
                .map_err(|_| vortex_error::vortex_err!("invalid metadata length"))?,
        ))?;
        vortex_ensure!(
            buffers.len() == 1,
            "PatchedBlockResidualArray expects one buffer"
        );
        let block_count = len.div_ceil(BLOCK_LEN);
        let bases_dtype = DType::Primitive(PType::U64, NonNullable);
        let u32_dtype = DType::Primitive(PType::U32, NonNullable);
        let (has_validity, has_patches) = match children.len() {
            1 => (false, false),
            2 => (true, false),
            4 => (false, true),
            5 => (true, true),
            count => vortex_bail!("PatchedBlockResidualArray child count {count} is invalid"),
        };
        let mut next = 0;
        let mut take = |dtype: &DType, child_len: usize| {
            let child = children.get(next, dtype, child_len);
            next += 1;
            child
        };
        let validity = has_validity
            .then(|| take(&Validity::DTYPE, len))
            .transpose()?;
        let bases = take(&bases_dtype, block_count)?;
        let (patch_indices, patch_values, patch_chunk_offsets) = if has_patches {
            (
                Some(take(&u32_dtype, patch_count)?),
                Some(take(&bases_dtype, patch_count)?),
                Some(take(&u32_dtype, block_count + 1)?),
            )
        } else {
            (None, None, None)
        };
        let data = PatchedBlockResidualData::try_new(len, host_payload(&buffers[0])?)?;
        let slots = PatchedBlockResidualSlots {
            validity,
            bases,
            patch_indices,
            patch_values,
            patch_chunk_offsets,
        }
        .into_slots();
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        PatchedBlockResidualSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let view = array.as_view();
        let data = view.data();
        let bases = view.bases().clone().execute::<PrimitiveArray>(ctx)?;
        let bases = bases.as_slice::<u64>();
        let mut output = Vec::with_capacity(data.len);
        let mut chunk = [0u64; BLOCK_LEN];
        for (block_index, &base) in bases.iter().enumerate() {
            let width = data.residual_widths[block_index];
            let block_len = (data.len - block_index * BLOCK_LEN).min(BLOCK_LEN);
            if width == 0 {
                output.extend(std::iter::repeat_n(base, block_len));
                continue;
            }
            let words = &data.residual_words[data.block_words(block_index)];
            // SAFETY: The encoder writes one complete FastLanes chunk per block.
            unsafe { BitPacking::unchecked_unpack(usize::from(width), words, &mut chunk) };
            output.extend(chunk[..block_len].iter().map(|low| base.wrapping_add(*low)));
        }
        if let Some(patches) = view.patches() {
            let indices = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
            let highs = patches.values().clone().execute::<PrimitiveArray>(ctx)?;
            for (&index, &high) in indices
                .as_slice::<u32>()
                .iter()
                .zip(highs.as_slice::<u64>())
            {
                let index = usize::try_from(index)?;
                let width = data.residual_widths[index / BLOCK_LEN];
                let base = bases[index / BLOCK_LEN];
                let low = output[index].wrapping_sub(base);
                output[index] = base.wrapping_add(low | (high << width));
            }
        }
        let validity = view.array().validity()?;
        Ok(ExecutionResult::done(
            PrimitiveArray::new(Buffer::from(output), validity).into_array(),
        ))
    }
}

impl OperationsVTable<PatchedBlockResidual> for PatchedBlockResidual {
    fn scalar_at(
        array: ArrayView<'_, PatchedBlockResidual>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        if !array.as_ref().is_valid(index, ctx)? {
            return Ok(Scalar::null(array.dtype().clone()));
        }
        let data = array.data();
        let block_index = index / BLOCK_LEN;
        let width = data.residual_widths[block_index];
        let mut residual = if width == 0 {
            0
        } else {
            let words = &data.residual_words[data.block_words(block_index)];
            // SAFETY: The encoder writes one complete FastLanes chunk per block.
            unsafe {
                <u64 as BitPacking>::unchecked_unpack_single(
                    usize::from(width),
                    words,
                    index % BLOCK_LEN,
                )
            }
        };
        if let Some(high) = patched_high(&array, block_index, index)? {
            residual |= high << width;
        }
        let base = match array.bases().as_opt::<Primitive>() {
            Some(bases) => bases.as_slice::<u64>()[block_index],
            None => u64::try_from(&array.bases().execute_scalar(block_index, ctx)?)?,
        };
        Ok(Scalar::primitive(
            base.wrapping_add(residual),
            array.dtype().nullability(),
        ))
    }
}

/// Look up the high bits patched at `index`, downcasting canonical patch children.
///
/// Falls back to the generic [`Patches`] search when a patch child is a non-canonical encoding.
fn patched_high(
    array: &ArrayView<'_, PatchedBlockResidual>,
    block_index: usize,
    index: usize,
) -> VortexResult<Option<u64>> {
    let (Some(indices), Some(values), Some(chunk_offsets)) = (
        array.patch_indices(),
        array.patch_values(),
        array.patch_chunk_offsets(),
    ) else {
        return Ok(None);
    };
    if let (Some(indices), Some(values), Some(chunk_offsets)) = (
        indices.as_opt::<Primitive>(),
        values.as_opt::<Primitive>(),
        chunk_offsets.as_opt::<Primitive>(),
    ) {
        let chunk_offsets = chunk_offsets.as_slice::<u32>();
        let chunk = usize::try_from(chunk_offsets[block_index])?
            ..usize::try_from(chunk_offsets[block_index + 1])?;
        let block_indices = &indices.as_slice::<u32>()[chunk.clone()];
        return Ok(block_indices
            .binary_search(&u32::try_from(index)?)
            .ok()
            .map(|patch_index| values.as_slice::<u64>()[chunk.start + patch_index]));
    }
    array
        .patches()
        .vortex_expect("patch children are present")
        .get_patched(index)?
        .map(|high| u64::try_from(&high))
        .transpose()
}

impl ValidityVTable<PatchedBlockResidual> for PatchedBlockResidual {
    fn validity(array: ArrayView<'_, PatchedBlockResidual>) -> VortexResult<Validity> {
        let slots = PatchedBlockResidualSlotsView::from_slots(array.slots());
        Ok(child_to_validity(
            slots.validity,
            array.dtype().nullability(),
        ))
    }
}

fn host_payload(buffer: &BufferHandle) -> VortexResult<ByteBuffer> {
    buffer
        .clone()
        .ensure_aligned(Alignment::of::<u64>())?
        .try_into_host_sync()
}

/// Register the prototype encoding in one session.
pub fn initialize_patched(session: &VortexSession) {
    use vortex_array::session::ArraySessionExt;
    session.arrays().register(PatchedBlockResidual);
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::LazyLock;

    use vortex_array::ArrayContext;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::assert_arrays_eq;
    use vortex_array::assert_nth_scalar;
    use vortex_array::serde::SerializeOptions;
    use vortex_array::serde::SerializedArray;
    use vortex_session::registry::ReadContext;

    use super::*;
    use crate::BlockResidual;

    pub(crate) static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = array_session();
        crate::initialize(&session);
        initialize_patched(&session);
        session
    });

    /// Ordered-float-like values: a smooth walk with occasional large outliers.
    pub(crate) fn sample_values(len: usize) -> Vec<u64> {
        let mut state = 0x9E3779B97F4A7C15_u64;
        let mut walk = 1_u64 << 40;
        (0..len)
            .map(|index| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                walk = walk.wrapping_add(state >> 52);
                if index % 331 == 0 {
                    walk | (0xFFF_u64 << 44)
                } else {
                    walk
                }
            })
            .collect()
    }

    fn serialized_nbytes(array: ArrayRef) -> VortexResult<usize> {
        let array_context = ArrayContext::empty();
        let serialized = array.serialize(&array_context, &SESSION, &SerializeOptions::default())?;
        Ok(serialized.iter().map(|buffer| buffer.len()).sum())
    }

    #[test]
    fn roundtrips_values() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let original =
            PrimitiveArray::new(Buffer::from(sample_values(4000)), Validity::NonNullable);
        let encoded = PatchedBlockResidual::from_primitive(original.as_view())?;
        assert!(encoded.patch_indices().is_some());
        assert_arrays_eq!(encoded, original, &mut ctx);
        Ok(())
    }

    #[test]
    fn scalar_access() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = sample_values(3000);
        let original = PrimitiveArray::new(Buffer::from(values.clone()), Validity::NonNullable);
        let encoded = PatchedBlockResidual::from_primitive(original.as_view())?;
        for index in [0, 331, 1023, 1024, 2999] {
            assert_nth_scalar!(encoded, index, values[index], &mut ctx);
        }
        Ok(())
    }

    #[test]
    fn preserves_nulls() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let original = PrimitiveArray::from_option_iter([Some(5_u64), None, Some(1 << 60)]);
        let encoded = PatchedBlockResidual::from_primitive(original.as_view())?;
        assert_arrays_eq!(encoded, original, &mut ctx);
        Ok(())
    }

    #[test]
    fn serde_roundtrip() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let original =
            PrimitiveArray::new(Buffer::from(sample_values(2500)), Validity::NonNullable);
        let encoded = PatchedBlockResidual::from_primitive(original.as_view())?.into_array();
        let dtype = encoded.dtype().clone();
        let len = encoded.len();
        let array_context = ArrayContext::empty();
        let serialized =
            encoded.serialize(&array_context, &SESSION, &SerializeOptions::default())?;
        let mut bytes = ByteBufferMut::empty();
        for buffer in serialized {
            bytes.extend_from_slice(buffer.as_ref());
        }
        let decoded = SerializedArray::try_from(bytes.freeze())?.decode(
            &dtype,
            len,
            &ReadContext::new(array_context.to_ids()),
            &SESSION,
        )?;
        assert_arrays_eq!(decoded, original, &mut ctx);
        Ok(())
    }

    #[test]
    fn compare_size_with_buffer_layout() -> VortexResult<()> {
        for len in [1024_usize, 10_000, 100_000] {
            let original =
                PrimitiveArray::new(Buffer::from(sample_values(len)), Validity::NonNullable);
            let fused = BlockResidual::from_primitive(original.as_view())?.into_array();
            let patched = PatchedBlockResidual::from_primitive(original.as_view())?.into_array();
            let fused_nbytes = serialized_nbytes(fused)?;
            let patched_nbytes = serialized_nbytes(patched)?;
            println!(
                "len {len}: fused {fused_nbytes} B, patched-children {patched_nbytes} B ({:+.2}%)",
                (patched_nbytes as f64 / fused_nbytes as f64 - 1.0) * 100.0
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod perf_tests {
    use std::hint::black_box;
    use std::time::Instant;

    use vortex_array::VortexSessionExecute;
    use vortex_error::VortexResult;

    use super::tests::SESSION;
    use super::tests::sample_values;
    use super::*;
    use crate::BlockResidual;

    fn time_decodes(array: &ArrayRef, iterations: usize) -> VortexResult<f64> {
        let mut ctx = SESSION.create_execution_ctx();
        // Warm up once so first-touch costs are excluded.
        black_box(array.clone().execute::<PrimitiveArray>(&mut ctx)?);
        let start = Instant::now();
        for _ in 0..iterations {
            black_box(array.clone().execute::<PrimitiveArray>(&mut ctx)?);
        }
        Ok(start.elapsed().as_secs_f64() / iterations as f64)
    }

    fn time_scalars(array: &ArrayRef, len: usize) -> VortexResult<f64> {
        let mut ctx = SESSION.create_execution_ctx();
        let start = Instant::now();
        let mut checksum = 0u64;
        for index in (0..len).step_by(7) {
            checksum =
                checksum.wrapping_add(u64::try_from(&array.execute_scalar(index, &mut ctx)?)?);
        }
        black_box(checksum);
        Ok(start.elapsed().as_secs_f64())
    }

    #[test]
    #[ignore = "manual perf comparison; run with --release --ignored --nocapture"]
    fn compare_decode_perf() -> VortexResult<()> {
        let len = 1 << 20;
        let original = PrimitiveArray::new(Buffer::from(sample_values(len)), Validity::NonNullable);
        let fused = BlockResidual::from_primitive(original.as_view())?.into_array();
        let patched = PatchedBlockResidual::from_primitive(original.as_view())?.into_array();

        let iterations = 20;
        let fused_decode = time_decodes(&fused, iterations)?;
        let patched_decode = time_decodes(&patched, iterations)?;
        println!(
            "full decode ({len} values): fused {:.3} ms, patched-children {:.3} ms ({:+.1}%)",
            fused_decode * 1e3,
            patched_decode * 1e3,
            (patched_decode / fused_decode - 1.0) * 100.0
        );

        let fused_scalar = time_scalars(&fused, len)?;
        let patched_scalar = time_scalars(&patched, len)?;
        println!(
            "scalar sweep (every 7th): fused {:.3} ms, patched-children {:.3} ms ({:+.1}%)",
            fused_scalar * 1e3,
            patched_scalar * 1e3,
            (patched_scalar / fused_scalar - 1.0) * 100.0
        );
        Ok(())
    }
}
