// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Range;

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
use vortex_array::arrays::slice::SliceReduce;
use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::dtype::PType;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar::Scalar;
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTable;
use vortex_array::vtable::child_to_validity;
use vortex_array::vtable::validity_to_child;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::BlockResidualCodec;
use crate::BlockResidualParts;
use crate::codec::ResidualWord;
use crate::codec::packed_words_as_native;
use crate::codec::read_wide_bits;

const BLOCK_LEN: usize = 1024;
const METADATA_VERSION: u8 = 1;
const METADATA_LEN: usize = 41;

/// Ordered unsigned integers with one reference and packed residuals per block.
pub type BlockResidualArray = Array<BlockResidual>;

#[array_slots(BlockResidual)]
pub struct BlockResidualSlots {
    #[slot(0)]
    pub bases: ArrayRef,
    #[slot(1)]
    pub residual_widths: ArrayRef,
    #[slot(2)]
    pub high_widths: ArrayRef,
    #[slot(3)]
    pub residual_starts: ArrayRef,
    #[slot(4)]
    pub patch_starts: ArrayRef,
    #[slot(5)]
    pub high_starts: ArrayRef,
    #[slot(6)]
    pub residual_words: ArrayRef,
    #[slot(7)]
    pub patch_positions: ArrayRef,
    #[slot(8)]
    pub patch_highs: ArrayRef,
    #[slot(9)]
    pub validity: Option<ArrayRef>,
}

#[derive(Clone, Debug)]
pub struct BlockResidualData {
    unsliced_len: usize,
    slice_start: usize,
    slice_stop: usize,
    residual_word_count: usize,
    patch_count: usize,
    patch_high_count: usize,
}

impl Display for BlockResidualData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "blocks: {}, slice: {}..{}",
            self.unsliced_len.div_ceil(BLOCK_LEN),
            self.slice_start,
            self.slice_stop
        )
    }
}

impl ArrayHash for BlockResidualData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        self.unsliced_len.hash(state);
        self.slice_start.hash(state);
        self.slice_stop.hash(state);
        self.residual_word_count.hash(state);
        self.patch_count.hash(state);
        self.patch_high_count.hash(state);
    }
}

impl ArrayEq for BlockResidualData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.unsliced_len == other.unsliced_len
            && self.slice_start == other.slice_start
            && self.slice_stop == other.slice_stop
            && self.residual_word_count == other.residual_word_count
            && self.patch_count == other.patch_count
            && self.patch_high_count == other.patch_high_count
    }
}

#[derive(Clone, Debug)]
pub struct BlockResidual;

impl VTable for BlockResidual {
    type TypedArrayData = BlockResidualData;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.block_residual");
        *ID
    }

    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let slots = BlockResidualSlotsView::from_slots(slots);
        let validity = child_to_validity(slots.validity, dtype.nullability());
        data.validate(dtype, len, slots, &validity)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("BlockResidualArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        vortex_panic!("BlockResidualArray buffer_name {idx} out of bounds")
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_array::vtable::with_empty_buffers(self, array, buffers)
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            BlockResidualMetadata::from_data(array.data())?.encode(),
        ))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        let metadata = BlockResidualMetadata::decode(metadata)?;
        let unsliced_len = usize::try_from(metadata.unsliced_len)?;
        let slice_start = usize::try_from(metadata.slice_start)?;
        let slice_stop = slice_start
            .checked_add(len)
            .ok_or_else(|| vortex_error::vortex_err!("block residual slice length overflows"))?;
        let residual_word_count = usize::try_from(metadata.residual_word_count)?;
        let patch_count = usize::try_from(metadata.patch_count)?;
        let patch_high_count = usize::try_from(metadata.patch_high_count)?;
        let block_count = unsliced_len.div_ceil(BLOCK_LEN);
        let bases = children.get(0, &primitive_dtype(PType::U64), block_count)?;
        let residual_widths = children.get(1, &primitive_dtype(PType::U8), block_count)?;
        let high_widths = children.get(2, &primitive_dtype(PType::U8), block_count)?;
        let residual_starts = children.get(3, &primitive_dtype(PType::U32), block_count + 1)?;
        let patch_starts = children.get(4, &primitive_dtype(PType::U32), block_count + 1)?;
        let high_starts = children.get(5, &primitive_dtype(PType::U32), block_count + 1)?;
        let residual_words = children.get(6, &primitive_dtype(PType::U64), residual_word_count)?;
        let patch_positions = children.get(7, &primitive_dtype(PType::U16), patch_count)?;
        let patch_highs = children.get(8, &primitive_dtype(PType::U8), patch_high_count)?;
        let validity = match children.len() {
            9 => Validity::from(dtype.nullability()),
            10 => Validity::Array(children.get(9, &Validity::DTYPE, unsliced_len)?),
            count => vortex_bail!("BlockResidualArray expects nine or ten children, got {count}"),
        };
        let slots = BlockResidualSlots {
            bases,
            residual_widths,
            high_widths,
            residual_starts,
            patch_starts,
            high_starts,
            residual_words,
            patch_positions,
            patch_highs,
            validity: validity_to_child(&validity, unsliced_len),
        }
        .into_slots();
        let data = BlockResidualData {
            unsliced_len,
            slice_start,
            slice_stop,
            residual_word_count,
            patch_count,
            patch_high_count,
        };
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        BlockResidualSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(
            decompress_array(array.as_view(), ctx)?.into_array(),
        ))
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
    }
}

impl OperationsVTable<BlockResidual> for BlockResidual {
    fn scalar_at(
        array: ArrayView<'_, BlockResidual>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        if !array.as_ref().is_valid(index, ctx)? {
            return Ok(Scalar::null(array.dtype().clone()));
        }
        let value = scalar_from_array(array, index, ctx)?;
        let nullability = array.dtype().nullability();
        Ok(match array.dtype().as_ptype() {
            PType::U8 => Scalar::primitive(u8::try_from(value)?, nullability),
            PType::U16 => Scalar::primitive(u16::try_from(value)?, nullability),
            PType::U32 => Scalar::primitive(u32::try_from(value)?, nullability),
            PType::U64 => Scalar::primitive(value, nullability),
            PType::I8 => Scalar::primitive(
                i8::from_le_bytes([(u8::try_from(value)? ^ (1_u8 << 7))]),
                nullability,
            ),
            PType::I16 => Scalar::primitive(
                i16::from_le_bytes((u16::try_from(value)? ^ (1_u16 << 15)).to_le_bytes()),
                nullability,
            ),
            PType::I32 => Scalar::primitive(
                i32::from_le_bytes((u32::try_from(value)? ^ (1_u32 << 31)).to_le_bytes()),
                nullability,
            ),
            PType::I64 => Scalar::primitive(
                i64::from_le_bytes((value ^ (1_u64 << 63)).to_le_bytes()),
                nullability,
            ),
            ptype => vortex_bail!("BlockResidual scalar access does not support {ptype}"),
        })
    }
}

impl ValidityVTable<BlockResidual> for BlockResidual {
    fn validity(array: ArrayView<'_, BlockResidual>) -> VortexResult<Validity> {
        array
            .unsliced_validity()
            .slice(array.data().slice_start..array.data().slice_stop)
    }
}

impl SliceReduce for BlockResidual {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let data = array.data().slice(range);
        Ok(Some(
            Array::try_from_parts(
                ArrayParts::new(BlockResidual, array.dtype().clone(), data.len(), data)
                    .with_slots(array.slots().iter().cloned().collect()),
            )?
            .into_array(),
        ))
    }
}

static RULES: ParentRuleSet<BlockResidual> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&SliceReduceAdaptor(BlockResidual))]);

pub trait BlockResidualArrayExt: TypedArrayRef<BlockResidual> + BlockResidualArraySlotsExt {
    fn unsliced_validity(&self) -> Validity {
        child_to_validity(
            self.as_ref().slots()[BlockResidualSlots::VALIDITY].as_ref(),
            self.as_ref().dtype().nullability(),
        )
    }

    /// Decode the logical slice.
    fn decompress(&self, ctx: &mut ExecutionCtx) -> VortexResult<PrimitiveArray> {
        decompress_array(self.to_owned().as_view(), ctx)
    }
}

impl<T: TypedArrayRef<BlockResidual>> BlockResidualArrayExt for T {}

impl BlockResidual {
    /// Encode an integer array in independent blocks.
    pub fn from_primitive(array: ArrayView<'_, Primitive>) -> VortexResult<BlockResidualArray> {
        vortex_ensure!(
            array.ptype().is_int(),
            "BlockResidual requires integer values"
        );
        let validity = array.validity()?;
        let values = match array.ptype() {
            PType::U8 => array
                .as_slice::<u8>()
                .iter()
                .map(|&value| u64::from(value))
                .collect(),
            PType::U16 => array
                .as_slice::<u16>()
                .iter()
                .map(|&value| u64::from(value))
                .collect(),
            PType::U32 => array
                .as_slice::<u32>()
                .iter()
                .map(|&value| u64::from(value))
                .collect(),
            PType::U64 => array.as_slice::<u64>().to_vec(),
            PType::I8 => array
                .as_slice::<i8>()
                .iter()
                .map(|&value| u64::from((value as u8) ^ (1_u8 << 7)))
                .collect(),
            PType::I16 => array
                .as_slice::<i16>()
                .iter()
                .map(|&value| u64::from((value as u16) ^ (1_u16 << 15)))
                .collect(),
            PType::I32 => array
                .as_slice::<i32>()
                .iter()
                .map(|&value| u64::from((value as u32) ^ (1_u32 << 31)))
                .collect(),
            PType::I64 => array
                .as_slice::<i64>()
                .iter()
                .map(|&value| (value as u64) ^ (1_u64 << 63))
                .collect(),
            ptype => vortex_bail!("BlockResidual does not support {ptype}"),
        };
        let parts = BlockResidualCodec::encode_with_word_width(
            &values,
            u8::try_from(array.ptype().bit_width())?,
        )?
        .into_parts()?;
        Self::try_new(parts, validity, array.ptype())
    }

    fn try_new(
        parts: BlockResidualParts,
        validity: Validity,
        ptype: PType,
    ) -> VortexResult<BlockResidualArray> {
        let data = BlockResidualData {
            unsliced_len: parts.len,
            slice_start: 0,
            slice_stop: parts.len,
            residual_word_count: parts.residual_words.len(),
            patch_count: parts.patch_positions.len(),
            patch_high_count: parts.patch_highs.len(),
        };
        let slots = BlockResidualSlots {
            bases: PrimitiveArray::from_iter(parts.bases).into_array(),
            residual_widths: PrimitiveArray::from_iter(parts.residual_widths).into_array(),
            high_widths: PrimitiveArray::from_iter(parts.high_widths).into_array(),
            residual_starts: PrimitiveArray::from_iter(parts.residual_starts).into_array(),
            patch_starts: PrimitiveArray::from_iter(parts.patch_starts).into_array(),
            high_starts: PrimitiveArray::from_iter(parts.high_starts).into_array(),
            residual_words: PrimitiveArray::from_iter(parts.residual_words).into_array(),
            patch_positions: PrimitiveArray::from_iter(parts.patch_positions).into_array(),
            patch_highs: PrimitiveArray::from_iter(parts.patch_highs).into_array(),
            validity: validity_to_child(&validity, data.unsliced_len),
        }
        .into_slots();
        Array::try_from_parts(
            ArrayParts::new(
                BlockResidual,
                DType::Primitive(ptype, validity.nullability()),
                data.unsliced_len,
                data,
            )
            .with_slots(slots),
        )
    }
}

impl BlockResidualData {
    fn validate(
        &self,
        dtype: &DType,
        len: usize,
        slots: BlockResidualSlotsView<'_>,
        validity: &Validity,
    ) -> VortexResult<()> {
        vortex_ensure!(
            dtype.is_int(),
            "BlockResidualArray requires an integer dtype"
        );
        vortex_ensure!(
            self.slice_start <= self.slice_stop && self.slice_stop <= self.unsliced_len,
            "block residual slice exceeds its source length"
        );
        vortex_ensure!(len == self.len(), "block residual slice length is invalid");
        let block_count = self.unsliced_len.div_ceil(BLOCK_LEN);
        for (child, ptype, child_len) in [
            (slots.bases, PType::U64, block_count),
            (slots.residual_widths, PType::U8, block_count),
            (slots.high_widths, PType::U8, block_count),
            (slots.residual_starts, PType::U32, block_count + 1),
            (slots.patch_starts, PType::U32, block_count + 1),
            (slots.high_starts, PType::U32, block_count + 1),
            (slots.residual_words, PType::U64, self.residual_word_count),
            (slots.patch_positions, PType::U16, self.patch_count),
            (slots.patch_highs, PType::U8, self.patch_high_count),
        ] {
            vortex_ensure!(
                child.dtype() == &primitive_dtype(ptype) && child.len() == child_len,
                "block residual child has an invalid dtype or length"
            );
        }
        if let Some(validity_len) = validity.maybe_len() {
            vortex_ensure!(
                validity_len == self.unsliced_len,
                "block residual validity length is invalid"
            );
        }
        Ok(())
    }

    fn len(&self) -> usize {
        self.slice_stop - self.slice_start
    }

    fn slice(&self, range: Range<usize>) -> Self {
        Self {
            slice_start: self.slice_start + range.start,
            slice_stop: self.slice_start + range.end,
            ..self.clone()
        }
    }
}

struct ExecutedBlockResidual {
    bases: PrimitiveArray,
    residual_widths: PrimitiveArray,
    high_widths: PrimitiveArray,
    residual_starts: PrimitiveArray,
    patch_starts: PrimitiveArray,
    high_starts: PrimitiveArray,
    residual_words: PrimitiveArray,
    patch_positions: PrimitiveArray,
    patch_highs: PrimitiveArray,
}

impl ExecutedBlockResidual {
    fn new(array: ArrayView<'_, BlockResidual>, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        macro_rules! execute_child {
            ($accessor:ident) => {
                array.$accessor().clone().execute::<PrimitiveArray>(ctx)?
            };
        }
        Ok(Self {
            bases: execute_child!(bases),
            residual_widths: execute_child!(residual_widths),
            high_widths: execute_child!(high_widths),
            residual_starts: execute_child!(residual_starts),
            patch_starts: execute_child!(patch_starts),
            high_starts: execute_child!(high_starts),
            residual_words: execute_child!(residual_words),
            patch_positions: execute_child!(patch_positions),
            patch_highs: execute_child!(patch_highs),
        })
    }
}

fn decode_array_values<T: NativePType, U: ResidualWord>(
    array: ArrayView<'_, BlockResidual>,
    ctx: &mut ExecutionCtx,
    mut transform: impl FnMut(U) -> T,
) -> VortexResult<PrimitiveArray> {
    let children = ExecutedBlockResidual::new(array, ctx)?;
    let bases = children.bases.as_slice::<u64>();
    let residual_widths = children.residual_widths.as_slice::<u8>();
    let high_widths = children.high_widths.as_slice::<u8>();
    let residual_starts = children.residual_starts.as_slice::<u32>();
    let patch_starts = children.patch_starts.as_slice::<u32>();
    let high_starts = children.high_starts.as_slice::<u32>();
    let residual_words = children.residual_words.as_slice::<u64>();
    let patch_positions = children.patch_positions.as_slice::<u16>();
    let patch_highs = children.patch_highs.as_slice::<u8>();
    let logical_range = array.data().slice_start..array.data().slice_stop;
    let mut values = Vec::with_capacity(logical_range.len());
    let mut residuals = [U::default(); BLOCK_LEN];

    let first_block = logical_range.start / BLOCK_LEN;
    let last_block = logical_range.end.div_ceil(BLOCK_LEN);
    for block_index in first_block..last_block {
        let block_start = block_index * BLOCK_LEN;
        let block_len = (array.data().unsliced_len - block_start).min(BLOCK_LEN);
        let block_stop = block_start + block_len;
        if block_stop <= logical_range.start || block_start >= logical_range.end {
            continue;
        }

        let residual_width = residual_widths[block_index];
        let high_width = high_widths[block_index];
        let base = U::from_u64(bases[block_index]);
        vortex_ensure!(
            residual_width <= U::BITS
                && high_width <= U::BITS
                && u16::from(residual_width) + u16::from(high_width) <= u16::from(U::BITS),
            "block residual bit widths are invalid"
        );
        let residual_payload = payload_range(
            residual_starts,
            block_index,
            residual_words.len(),
            "residual",
        )?;
        vortex_ensure!(
            residual_payload.len() == BLOCK_LEN * usize::from(residual_width) / 64,
            "block residual word count is invalid"
        );
        let patch_payload =
            payload_range(patch_starts, block_index, patch_positions.len(), "patch")?;
        let high_payload =
            payload_range(high_starts, block_index, patch_highs.len(), "patch high")?;
        let positions = &patch_positions[patch_payload];
        validate_patch_header(
            residual_width,
            high_width,
            positions.len(),
            high_payload.len(),
        )?;
        let highs = &patch_highs[high_payload];
        let local_start = logical_range.start.saturating_sub(block_start);
        let local_stop = (logical_range.end - block_start).min(block_len);

        if residual_width == 0 {
            let output_start = values.len();
            values
                .extend(std::iter::repeat_with(|| transform(base)).take(local_stop - local_start));
            let mut previous_position = None;
            for (patch_index, &position) in positions.iter().enumerate() {
                validate_patch_position(block_len, previous_position, position)?;
                previous_position = Some(position);
                let position = usize::from(position);
                if position < local_start || position >= local_stop {
                    continue;
                }
                // SAFETY: The payload includes fifteen readable padding bytes.
                let high = unsafe {
                    read_wide_bits(highs, patch_index * usize::from(high_width), high_width)
                };
                values[output_start + position - local_start] =
                    transform(base.wrapping_add(U::from_u64(high)));
            }
            continue;
        }

        let packed = packed_words_as_native::<U>(&residual_words[residual_payload]);
        // SAFETY: The encoder writes one complete FastLanes chunk for each block.
        unsafe {
            U::unchecked_unpack(usize::from(residual_width), packed, &mut residuals);
        }
        let mut previous_position = None;
        for (patch_index, &position) in positions.iter().enumerate() {
            validate_patch_position(block_len, previous_position, position)?;
            previous_position = Some(position);
            // SAFETY: The payload includes fifteen readable padding bytes.
            let high =
                unsafe { read_wide_bits(highs, patch_index * usize::from(high_width), high_width) };
            residuals[usize::from(position)].apply_high(high, residual_width);
        }

        values.extend(
            residuals[local_start..local_stop]
                .iter()
                .map(|&residual| transform(residual.wrapping_add(base))),
        );
    }
    Ok(PrimitiveArray::new(Buffer::from(values), array.validity()?))
}

fn decompress_array(
    array: ArrayView<'_, BlockResidual>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    match array.dtype().as_ptype() {
        PType::U8 => decode_array_values(array, ctx, |value: u8| value),
        PType::U16 => decode_array_values(array, ctx, |value: u16| value),
        PType::U32 => decode_array_values(array, ctx, |value: u32| value),
        PType::U64 => decode_array_values(array, ctx, |value: u64| value),
        PType::I8 => decode_array_values(array, ctx, |value: u8| (value ^ (1_u8 << 7)) as i8),
        PType::I16 => decode_array_values(array, ctx, |value: u16| (value ^ (1_u16 << 15)) as i16),
        PType::I32 => decode_array_values(array, ctx, |value: u32| (value ^ (1_u32 << 31)) as i32),
        PType::I64 => decode_array_values(array, ctx, |value: u64| (value ^ (1_u64 << 63)) as i64),
        ptype => vortex_bail!("BlockResidual decode does not support {ptype}"),
    }
}

pub(crate) fn decompress_ordered_f64(
    array: ArrayView<'_, BlockResidual>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    decode_array_values(array, ctx, |ordered: u64| {
        let bits = if ordered & (1_u64 << 63) == 0 {
            !ordered
        } else {
            ordered ^ (1_u64 << 63)
        };
        f64::from_bits(bits)
    })
}

fn scalar_from_array(
    array: ArrayView<'_, BlockResidual>,
    index: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<u64> {
    let source_index = array.data().slice_start + index;
    let block_index = source_index / BLOCK_LEN;
    let index_in_block = source_index % BLOCK_LEN;
    let children = ExecutedBlockResidual::new(array, ctx)?;
    let residual_width = children.residual_widths.as_slice::<u8>()[block_index];
    let high_width = children.high_widths.as_slice::<u8>()[block_index];
    let logical_width = array.dtype().as_ptype().bit_width();
    vortex_ensure!(
        usize::from(residual_width) <= logical_width
            && usize::from(high_width) <= logical_width
            && usize::from(residual_width) + usize::from(high_width) <= logical_width,
        "block residual bit widths are invalid"
    );
    let residual_words = children.residual_words.as_slice::<u64>();
    let residual_payload = payload_range(
        children.residual_starts.as_slice::<u32>(),
        block_index,
        residual_words.len(),
        "residual",
    )?;
    vortex_ensure!(
        residual_payload.len() == BLOCK_LEN * usize::from(residual_width) / 64,
        "block residual word count is invalid"
    );
    let mut residual = match logical_width {
        8 => unpack_single_residual::<u8>(
            residual_width,
            &residual_words[residual_payload],
            index_in_block,
        ),
        16 => unpack_single_residual::<u16>(
            residual_width,
            &residual_words[residual_payload],
            index_in_block,
        ),
        32 => unpack_single_residual::<u32>(
            residual_width,
            &residual_words[residual_payload],
            index_in_block,
        ),
        64 => unpack_single_residual::<u64>(
            residual_width,
            &residual_words[residual_payload],
            index_in_block,
        ),
        _ => vortex_bail!("block residual logical bit width is invalid"),
    };

    let positions = children.patch_positions.as_slice::<u16>();
    let patch_payload = payload_range(
        children.patch_starts.as_slice::<u32>(),
        block_index,
        positions.len(),
        "patch",
    )?;
    let block_positions = &positions[patch_payload];
    let block_start = block_index * BLOCK_LEN;
    let block_len = (array.data().unsliced_len - block_start).min(BLOCK_LEN);
    let highs = children.patch_highs.as_slice::<u8>();
    let high_payload = payload_range(
        children.high_starts.as_slice::<u32>(),
        block_index,
        highs.len(),
        "patch high",
    )?;
    validate_patch_payload(
        block_len,
        residual_width,
        high_width,
        block_positions,
        high_payload.len(),
    )?;
    if let Ok(patch_index) = block_positions.binary_search(&u16::try_from(index_in_block)?) {
        let high_payload = &highs[high_payload];
        // SAFETY: The payload includes fifteen readable padding bytes.
        let high = unsafe {
            read_wide_bits(
                high_payload,
                patch_index * usize::from(high_width),
                high_width,
            )
        };
        residual |= high << residual_width;
    }
    Ok(children.bases.as_slice::<u64>()[block_index].wrapping_add(residual))
}

fn unpack_single_residual<T: ResidualWord>(width: u8, packed_words: &[u64], index: usize) -> u64 {
    if width == 0 {
        return 0;
    }
    let packed = packed_words_as_native::<T>(packed_words);
    // SAFETY: The encoder writes one complete FastLanes chunk for each block.
    unsafe { T::unchecked_unpack_single(usize::from(width), packed, index).to_u64() }
}

fn validate_patch_payload(
    block_len: usize,
    residual_width: u8,
    high_width: u8,
    positions: &[u16],
    high_payload_len: usize,
) -> VortexResult<()> {
    validate_patch_header(
        residual_width,
        high_width,
        positions.len(),
        high_payload_len,
    )?;
    let mut previous_position = None;
    for &position in positions {
        validate_patch_position(block_len, previous_position, position)?;
        previous_position = Some(position);
    }
    Ok(())
}

fn validate_patch_header(
    residual_width: u8,
    high_width: u8,
    patch_count: usize,
    high_payload_len: usize,
) -> VortexResult<()> {
    vortex_ensure!(
        patch_count == 0 || (high_width > 0 && residual_width < 64),
        "block residual patches require nonzero high bits"
    );
    let expected_high_len = if patch_count == 0 {
        0
    } else {
        (patch_count * usize::from(high_width)).div_ceil(8) + 15
    };
    vortex_ensure!(
        high_payload_len == expected_high_len,
        "block residual patch high payload is invalid"
    );
    Ok(())
}

fn validate_patch_position(
    block_len: usize,
    previous_position: Option<u16>,
    position: u16,
) -> VortexResult<()> {
    vortex_ensure!(
        usize::from(position) < block_len
            && previous_position.is_none_or(|previous| previous < position),
        "block residual patch positions are invalid"
    );
    Ok(())
}

fn payload_range(
    starts: &[u32],
    block_index: usize,
    payload_len: usize,
    name: &str,
) -> VortexResult<Range<usize>> {
    let start = usize::try_from(
        *starts
            .get(block_index)
            .ok_or_else(|| vortex_error::vortex_err!("block residual {name} start is missing"))?,
    )?;
    let stop = usize::try_from(
        *starts
            .get(block_index + 1)
            .ok_or_else(|| vortex_error::vortex_err!("block residual {name} stop is missing"))?,
    )?;
    vortex_ensure!(
        start <= stop && stop <= payload_len,
        "block residual {name} offsets are invalid"
    );
    Ok(start..stop)
}

#[derive(Clone, Copy)]
struct BlockResidualMetadata {
    unsliced_len: u64,
    slice_start: u64,
    residual_word_count: u64,
    patch_count: u64,
    patch_high_count: u64,
}

impl BlockResidualMetadata {
    fn from_data(data: &BlockResidualData) -> VortexResult<Self> {
        Ok(Self {
            unsliced_len: u64::try_from(data.unsliced_len)?,
            slice_start: u64::try_from(data.slice_start)?,
            residual_word_count: u64::try_from(data.residual_word_count)?,
            patch_count: u64::try_from(data.patch_count)?,
            patch_high_count: u64::try_from(data.patch_high_count)?,
        })
    }

    fn encode(self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(METADATA_LEN);
        bytes.push(METADATA_VERSION);
        for value in [
            self.unsliced_len,
            self.slice_start,
            self.residual_word_count,
            self.patch_count,
            self.patch_high_count,
        ] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn decode(bytes: &[u8]) -> VortexResult<Self> {
        vortex_ensure!(
            bytes.len() == METADATA_LEN,
            "BlockResidualArray metadata requires {METADATA_LEN} bytes"
        );
        vortex_ensure!(
            bytes[0] == METADATA_VERSION,
            "unsupported BlockResidualArray metadata version {}",
            bytes[0]
        );
        let read = |offset: usize| {
            u64::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
                bytes[offset + 4],
                bytes[offset + 5],
                bytes[offset + 6],
                bytes[offset + 7],
            ])
        };
        Ok(Self {
            unsliced_len: read(1),
            slice_start: read(9),
            residual_word_count: read(17),
            patch_count: read(25),
            patch_high_count: read(33),
        })
    }
}

fn primitive_dtype(ptype: PType) -> DType {
    DType::Primitive(ptype, NonNullable)
}

#[cfg(test)]
mod tests {
    use vortex_array::ArrayContext;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::serde::SerializeOptions;
    use vortex_array::serde::SerializedArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::ByteBufferMut;
    use vortex_error::VortexResult;
    use vortex_session::registry::ReadContext;

    use super::BlockResidual;
    use super::BlockResidualArraySlotsExt;

    #[test]
    fn roundtrip_and_scalar_access() -> VortexResult<()> {
        let values = (0..4_099)
            .map(|index| Ok(1_000_000_u64 + u64::try_from(index * index)?))
            .collect::<VortexResult<Vec<_>>>()?;
        let primitive = PrimitiveArray::from_iter(values.clone());
        let encoded = BlockResidual::from_primitive(primitive.as_view())?;
        let session = array_session();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();
        assert_arrays_eq!(encoded.clone(), primitive.into_array(), &mut ctx);
        for index in [0, 1, 1_023, 1_024, 4_098] {
            let scalar = encoded.execute_scalar(index, &mut ctx)?;
            assert_eq!(
                scalar.as_primitive().typed_value::<u64>(),
                Some(values[index])
            );
        }
        Ok(())
    }

    #[test]
    fn signed_roundtrip_and_scalar_access() -> VortexResult<()> {
        let values = (0..2_050)
            .map(|index| match index {
                0 => i64::MIN,
                1_023 => -1,
                1_024 => 0,
                2_049 => i64::MAX,
                _ => (index as i64 - 1_025) * 1_000_003,
            })
            .collect::<Vec<_>>();
        let primitive = PrimitiveArray::from_iter(values.clone());
        let encoded = BlockResidual::from_primitive(primitive.as_view())?;
        let session = array_session();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();

        assert_arrays_eq!(encoded.clone(), primitive.into_array(), &mut ctx);
        for index in [0, 1, 1_023, 1_024, 2_049] {
            let scalar = encoded.execute_scalar(index, &mut ctx)?;
            assert_eq!(
                scalar.as_primitive().typed_value::<i64>(),
                Some(values[index])
            );
        }
        Ok(())
    }

    #[test]
    fn narrow_integer_roundtrip() -> VortexResult<()> {
        let signed = PrimitiveArray::from_iter((0..2_050).map(|index| {
            let value = (index * 7919) % 65_521;
            value - 32_760
        }));
        let unsigned = PrimitiveArray::from_iter((0..2_050).map(|index| {
            let value = (index * 7907) % 65_521;
            value as u16
        }));
        let signed_encoded = BlockResidual::from_primitive(signed.as_view())?;
        let unsigned_encoded = BlockResidual::from_primitive(unsigned.as_view())?;
        let session = array_session();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();

        assert_arrays_eq!(signed_encoded, signed.into_array(), &mut ctx);
        assert_arrays_eq!(unsigned_encoded, unsigned.into_array(), &mut ctx);
        Ok(())
    }

    #[test]
    fn nullable_slice_and_scalar_access() -> VortexResult<()> {
        let values = (0..2_050)
            .map(|index| Ok(u64::try_from(index * index)?))
            .collect::<VortexResult<Vec<_>>>()?;
        let validity = Validity::from_iter((0..values.len()).map(|index| index != 1_024));
        let primitive = PrimitiveArray::new(Buffer::from(values), validity);
        let encoded = BlockResidual::from_primitive(primitive.as_view())?;
        let session = array_session();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();

        assert!(encoded.execute_scalar(1_024, &mut ctx)?.is_null());
        let sliced = encoded.into_array().slice(1_023..1_026)?;
        let expected = primitive.into_array().slice(1_023..1_026)?;
        assert_arrays_eq!(sliced, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn zero_width_patched_block_roundtrip() -> VortexResult<()> {
        let mut values = vec![42_u32; 2_050];
        values[1_023] = u32::MAX;
        let validity = Validity::from_iter((0..values.len()).map(|index| index != 1_024));
        let primitive = PrimitiveArray::new(Buffer::from(values.clone()), validity);
        let encoded = BlockResidual::from_primitive(primitive.as_view())?;
        let session = array_session();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();
        let widths = encoded
            .residual_widths()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;

        assert_eq!(widths.as_slice::<u8>()[0], 0);
        assert_eq!(
            encoded
                .execute_scalar(1_023, &mut ctx)?
                .as_primitive()
                .typed_value::<u32>(),
            Some(u32::MAX)
        );
        assert!(encoded.execute_scalar(1_024, &mut ctx)?.is_null());
        assert_arrays_eq!(
            encoded.into_array().slice(1_022..1_025)?,
            primitive.into_array().slice(1_022..1_025)?,
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn nullable_slice_serialization_roundtrip() -> VortexResult<()> {
        let values = (0..2_050)
            .map(|index| Ok(u64::try_from(index * index)?))
            .collect::<VortexResult<Vec<_>>>()?;
        let validity = Validity::from_iter((0..values.len()).map(|index| index != 1_024));
        let primitive = PrimitiveArray::new(Buffer::from(values), validity);
        let sliced = BlockResidual::from_primitive(primitive.as_view())?
            .into_array()
            .slice(1_023..1_026)?;
        let expected = primitive.into_array().slice(1_023..1_026)?;
        let dtype = sliced.dtype().clone();
        let len = sliced.len();
        let array_context = ArrayContext::empty();
        let session = array_session();
        crate::initialize(&session);
        let serialized =
            sliced.serialize(&array_context, &session, &SerializeOptions::default())?;
        let mut bytes = ByteBufferMut::empty();
        for buffer in serialized {
            bytes.extend_from_slice(buffer.as_ref());
        }
        let decoded = SerializedArray::try_from(bytes.freeze())?.decode(
            &dtype,
            len,
            &ReadContext::new(array_context.to_ids()),
            &session,
        )?;
        assert!(decoded.is::<BlockResidual>());
        assert_arrays_eq!(decoded, expected, &mut session.create_execution_ctx());
        Ok(())
    }
}
