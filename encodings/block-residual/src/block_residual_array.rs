// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Range;

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
        let value = scalar_from_array(array, index, ctx)?;
        Ok(Scalar::primitive(value, array.dtype().nullability()))
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
    /// Encode a non-negative `u64` array in independent blocks.
    pub fn from_primitive(array: ArrayView<'_, Primitive>) -> VortexResult<BlockResidualArray> {
        vortex_ensure!(
            array.ptype() == PType::U64,
            "BlockResidual requires u64 values"
        );
        let validity = array.validity()?;
        let parts = BlockResidualCodec::encode(array.as_slice::<u64>())?.into_parts()?;
        Self::try_new(parts, validity)
    }

    fn try_new(parts: BlockResidualParts, validity: Validity) -> VortexResult<BlockResidualArray> {
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
                DType::Primitive(PType::U64, validity.nullability()),
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
            dtype.as_ptype() == PType::U64,
            "BlockResidualArray requires a u64 dtype"
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

fn decode_array_values<T: NativePType>(
    array: ArrayView<'_, BlockResidual>,
    ctx: &mut ExecutionCtx,
    mut transform: impl FnMut(u64) -> T,
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
    let mut residuals = [0_u64; BLOCK_LEN];

    for block_index in 0..bases.len() {
        let block_start = block_index * BLOCK_LEN;
        let block_len = (array.data().unsliced_len - block_start).min(BLOCK_LEN);
        let block_stop = block_start + block_len;
        if block_stop <= logical_range.start || block_start >= logical_range.end {
            continue;
        }

        residuals.fill(0);
        let residual_width = residual_widths[block_index];
        let high_width = high_widths[block_index];
        vortex_ensure!(
            residual_width <= 64
                && high_width <= 64
                && u16::from(residual_width) + u16::from(high_width) <= 64,
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
        if residual_width > 0 {
            // SAFETY: The encoder writes one complete FastLanes chunk for each block.
            unsafe {
                u64::unchecked_unpack(
                    usize::from(residual_width),
                    &residual_words[residual_payload],
                    &mut residuals,
                );
            }
        }

        let patch_payload =
            payload_range(patch_starts, block_index, patch_positions.len(), "patch")?;
        let high_payload =
            payload_range(high_starts, block_index, patch_highs.len(), "patch high")?;
        let positions = &patch_positions[patch_payload];
        vortex_ensure!(
            positions
                .iter()
                .all(|&position| usize::from(position) < block_len)
                && positions.windows(2).all(|pair| pair[0] < pair[1]),
            "block residual patch positions are invalid"
        );
        let expected_high_len = if positions.is_empty() {
            0
        } else {
            (positions.len() * usize::from(high_width)).div_ceil(8) + 15
        };
        vortex_ensure!(
            high_payload.len() == expected_high_len,
            "block residual patch high payload is invalid"
        );
        let highs = &patch_highs[high_payload];
        for (patch_index, &position) in positions.iter().enumerate() {
            // SAFETY: The payload includes fifteen readable padding bytes.
            let high =
                unsafe { read_wide_bits(highs, patch_index * usize::from(high_width), high_width) };
            residuals[usize::from(position)] |= high << residual_width;
        }

        let local_start = logical_range.start.saturating_sub(block_start);
        let local_stop = (logical_range.end - block_start).min(block_len);
        values.extend(
            residuals[local_start..local_stop]
                .iter()
                .map(|&residual| transform(bases[block_index].wrapping_add(residual))),
        );
    }
    Ok(PrimitiveArray::new(Buffer::from(values), array.validity()?))
}

fn decompress_array(
    array: ArrayView<'_, BlockResidual>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    decode_array_values(array, ctx, |value| value)
}

pub(crate) fn decompress_ordered_f64(
    array: ArrayView<'_, BlockResidual>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    decode_array_values(array, ctx, |ordered| {
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
    vortex_ensure!(
        residual_width <= 64
            && high_width <= 64
            && u16::from(residual_width) + u16::from(high_width) <= 64,
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
    let mut residual = if residual_width == 0 {
        0
    } else {
        // SAFETY: The encoder writes one complete FastLanes chunk for each block.
        unsafe {
            u64::unchecked_unpack_single(
                usize::from(residual_width),
                &residual_words[residual_payload],
                index_in_block,
            )
        }
    };

    let positions = children.patch_positions.as_slice::<u16>();
    let patch_payload = payload_range(
        children.patch_starts.as_slice::<u32>(),
        block_index,
        positions.len(),
        "patch",
    )?;
    let block_positions = &positions[patch_payload];
    if let Ok(patch_index) = block_positions.binary_search(&u16::try_from(index_in_block)?) {
        let highs = children.patch_highs.as_slice::<u8>();
        let high_payload = payload_range(
            children.high_starts.as_slice::<u32>(),
            block_index,
            highs.len(),
            "patch high",
        )?;
        let high_payload = &highs[high_payload];
        vortex_ensure!(
            high_payload.len()
                >= (block_positions.len() * usize::from(high_width)).div_ceil(8) + 15,
            "block residual patch high payload is invalid"
        );
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
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_error::VortexResult;

    use super::BlockResidual;

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
}
