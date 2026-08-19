// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Ordering;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::mem::MaybeUninit;
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
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTable;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

const MAX_BINS: usize = 64;
const SPLIT_CANDIDATES: usize = 64;
const TRAINING_SAMPLE_SIZE: usize = 8_192;
const BIN_TABLE_COST_BITS: f64 = 128.0;
const SYMBOL_PADDING: usize = 8;
const OFFSET_PADDING: usize = 15;
const CHECKPOINT_INTERVAL: usize = 32;
const VALIDITY_CHECKPOINT_INTERVAL: usize = 256;
const MAX_BLOCK_LEN: usize = 1_024;

/// Fixed-width range identifiers with variable-width offsets and scalar checkpoints.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RangePackedCodec {
    len: usize,
    block_len: usize,
    symbol_width: u8,
    bins: Vec<Bin>,
    block_offsets: Vec<u32>,
    payload: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Bin {
    lower: u64,
    offset_bits: u8,
}

/// Fixed-bin range packing with bounded scalar access.
pub type RangePackedArray = Array<RangePacked>;

#[derive(Clone, Debug)]
pub struct RangePacked;

#[derive(Clone, Debug)]
pub struct RangePackedData {
    codec: RangePackedCodec,
    physical: ByteBuffer,
    ptype: PType,
    validity: Validity,
    dense_validity: DenseValidity,
    len: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct DenseValidity {
    words: Vec<u64>,
    checkpoints: Vec<u32>,
    valid_count: usize,
}

impl Display for RangePackedData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "bins: {}", self.codec.bin_count())
    }
}

impl ArrayHash for RangePackedData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        self.codec.hash(state);
        self.ptype.hash(state);
        self.validity.array_hash(state, _accuracy);
        self.dense_validity.hash(state);
        self.len.hash(state);
    }
}

impl ArrayEq for RangePackedData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.codec == other.codec
            && self.ptype == other.ptype
            && self.validity.array_eq(&other.validity, _accuracy)
            && self.dense_validity == other.dense_validity
            && self.len == other.len
    }
}

impl VTable for RangePacked {
    type TypedArrayData = RangePackedData;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.experimental.range_packed");
        *ID
    }

    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(slots.is_empty(), "RangePackedArray cannot have children");
        vortex_ensure!(
            dtype.is_primitive() && dtype.as_ptype() == data.ptype,
            "RangePackedArray dtype differs from its physical type"
        );
        vortex_ensure!(
            dtype.nullability() == data.validity.nullability(),
            "RangePackedArray dtype differs from its validity"
        );
        vortex_ensure!(len == data.len, "RangePackedArray length differs");
        vortex_ensure!(
            data.codec.len == data.dense_validity.valid_count,
            "RangePackedArray dense length differs"
        );
        vortex_ensure!(
            data.physical.len() == data.codec.encoded_size() + data.dense_validity.encoded_size(),
            "RangePackedArray physical size differs"
        );
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        1
    }

    fn buffer(array: ArrayView<'_, Self>, index: usize) -> BufferHandle {
        if index == 0 {
            BufferHandle::new_host(array.data().physical.clone())
        } else {
            vortex_panic!("RangePackedArray buffer index {index} is invalid")
        }
    }

    fn buffer_name(_array: ArrayView<'_, Self>, index: usize) -> Option<String> {
        (index == 0).then(|| "encoded".to_string())
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_ensure!(buffers.len() == 1, "RangePackedArray needs one buffer");
        let mut data = array.data().clone();
        data.physical = buffers[0].clone().try_to_host_sync()?;
        Ok(ArrayParts::new(
            self.clone(),
            array.dtype().clone(),
            array.len(),
            data,
        ))
    }

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        vortex_bail!("experimental RangePackedArray does not support serialization")
    }

    fn deserialize(
        &self,
        _dtype: &DType,
        _len: usize,
        _metadata: &[u8],
        _buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_bail!("experimental RangePackedArray does not support deserialization")
    }

    fn slot_name(_array: ArrayView<'_, Self>, index: usize) -> String {
        vortex_panic!("RangePackedArray slot index {index} is invalid")
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let decoded = decode_primitive(array.data())?
            .into_array()
            .cast(array.dtype().clone())?;
        Ok(ExecutionResult::done(decoded))
    }
}

impl OperationsVTable<RangePacked> for RangePacked {
    fn scalar_at(
        array: ArrayView<'_, RangePacked>,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        if !array.data().dense_validity.is_valid(index) {
            return Ok(Scalar::null(array.dtype().clone()));
        }
        let dense_index = array.data().dense_validity.rank(index);
        ordered_scalar(
            array.data().codec.scalar_at(dense_index)?,
            array.data().ptype,
            array.dtype().nullability(),
        )
    }
}

impl DenseValidity {
    fn new(
        len: usize,
        valid_count: usize,
        validity: impl Iterator<Item = bool>,
    ) -> VortexResult<Self> {
        if valid_count == len {
            return Ok(Self {
                words: Vec::new(),
                checkpoints: Vec::new(),
                valid_count,
            });
        }

        let mut words = vec![0_u64; len.div_ceil(64)];
        let mut observed_len = 0usize;
        for (index, valid) in validity.enumerate() {
            vortex_ensure!(index < len, "dense validity exceeds its length");
            if valid {
                words[index / 64] |= 1_u64 << (index % 64);
            }
            observed_len += 1;
        }
        vortex_ensure!(observed_len == len, "dense validity is too short");

        let words_per_checkpoint = VALIDITY_CHECKPOINT_INTERVAL / 64;
        let mut checkpoints = Vec::with_capacity(words.len().div_ceil(words_per_checkpoint));
        let mut rank = 0usize;
        for block in words.chunks(words_per_checkpoint) {
            checkpoints.push(u32::try_from(rank)?);
            rank += block
                .iter()
                .map(|word| word.count_ones() as usize)
                .sum::<usize>();
        }
        vortex_ensure!(rank == valid_count, "dense validity count differs");
        Ok(Self {
            words,
            checkpoints,
            valid_count,
        })
    }

    #[inline]
    fn is_valid(&self, index: usize) -> bool {
        self.words.is_empty() || self.words[index / 64] & (1_u64 << (index % 64)) != 0
    }

    #[inline]
    fn rank(&self, index: usize) -> usize {
        if self.words.is_empty() {
            return index;
        }
        let word_index = index / 64;
        let checkpoint_index = index / VALIDITY_CHECKPOINT_INTERVAL;
        let words_per_checkpoint = VALIDITY_CHECKPOINT_INTERVAL / 64;
        let word_start = checkpoint_index * words_per_checkpoint;
        let mut rank = self.checkpoints[checkpoint_index] as usize;
        for word in &self.words[word_start..word_index] {
            rank += word.count_ones() as usize;
        }
        let bits_before = index % 64;
        if bits_before != 0 {
            let mask = (1_u64 << bits_before) - 1;
            rank += (self.words[word_index] & mask).count_ones() as usize;
        }
        rank
    }

    fn encoded_size(&self) -> usize {
        self.words.len() * size_of::<u64>() + self.checkpoints.len() * size_of::<u32>()
    }
}

impl ValidityVTable<RangePacked> for RangePacked {
    fn validity(array: ArrayView<'_, RangePacked>) -> VortexResult<Validity> {
        Ok(array.data().validity.clone())
    }
}

impl RangePacked {
    pub fn from_primitive(
        array: ArrayView<'_, Primitive>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<RangePackedArray> {
        vortex_ensure!(
            array.ptype().is_int(),
            "RangePackedArray needs integer values"
        );
        let validity = array.validity()?;
        let mask = validity.execute_mask(array.len(), ctx)?;
        let dense_validity = DenseValidity::new(array.len(), mask.true_count(), mask.iter())?;
        let ordered = ordered_primitive(array)?;
        let dense_ordered = ordered
            .into_iter()
            .zip(mask.iter())
            .filter_map(|(value, valid)| valid.then_some(value))
            .collect::<Vec<_>>();
        let codec = RangePackedCodec::encode(&dense_ordered, MAX_BLOCK_LEN)?;
        let physical = ByteBuffer::zeroed(codec.encoded_size() + dense_validity.encoded_size());
        let data = RangePackedData {
            codec,
            physical,
            ptype: array.ptype(),
            validity,
            dense_validity,
            len: array.len(),
        };
        Array::try_from_parts(ArrayParts::new(
            RangePacked,
            array.dtype().clone(),
            array.len(),
            data,
        ))
    }

    pub fn decode_mapped<T, F>(
        array: ArrayView<'_, RangePacked>,
        map: F,
        null_value: T,
    ) -> VortexResult<Vec<T>>
    where
        T: Copy,
        F: FnMut(u64) -> T,
    {
        let dense_values = array.data().codec.decode_mapped(map)?;
        if array.data().dense_validity.words.is_empty() {
            return Ok(dense_values);
        }
        let mut dense = dense_values.into_iter();
        let mut values = Vec::with_capacity(array.len());
        for index in 0..array.len() {
            values.push(if array.data().dense_validity.is_valid(index) {
                dense
                    .next()
                    .ok_or_else(|| vortex_error::vortex_err!("dense values are too short"))?
            } else {
                null_value
            });
        }
        vortex_ensure!(dense.next().is_none(), "dense values are too long");
        Ok(values)
    }
}

fn ordered_primitive(array: ArrayView<'_, Primitive>) -> VortexResult<Vec<u64>> {
    Ok(match array.ptype() {
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
        ptype => vortex_bail!("RangePackedArray does not support {ptype}"),
    })
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the encoded values retain the source integer word width"
)]
fn decode_primitive(data: &RangePackedData) -> VortexResult<PrimitiveArray> {
    let dense_ordered = data.codec.decode()?;
    let ordered = if data.dense_validity.words.is_empty() {
        dense_ordered
    } else {
        let mut dense = dense_ordered.into_iter();
        let mut values = Vec::with_capacity(data.len);
        for index in 0..data.len {
            values.push(if data.dense_validity.is_valid(index) {
                dense
                    .next()
                    .ok_or_else(|| vortex_error::vortex_err!("dense values are too short"))?
            } else {
                0
            });
        }
        vortex_ensure!(dense.next().is_none(), "dense values are too long");
        values
    };
    let validity = data.validity.clone();
    Ok(match data.ptype {
        PType::U8 => PrimitiveArray::new::<u8>(
            ordered
                .into_iter()
                .map(u8::try_from)
                .collect::<Result<Vec<_>, _>>()?,
            validity,
        ),
        PType::U16 => PrimitiveArray::new::<u16>(
            ordered
                .into_iter()
                .map(u16::try_from)
                .collect::<Result<Vec<_>, _>>()?,
            validity,
        ),
        PType::U32 => PrimitiveArray::new::<u32>(
            ordered
                .into_iter()
                .map(u32::try_from)
                .collect::<Result<Vec<_>, _>>()?,
            validity,
        ),
        PType::U64 => PrimitiveArray::new::<u64>(ordered, validity),
        PType::I8 => PrimitiveArray::new::<i8>(
            ordered
                .into_iter()
                .map(|value| ((value as u8) ^ (1_u8 << 7)) as i8)
                .collect::<Vec<_>>(),
            validity,
        ),
        PType::I16 => PrimitiveArray::new::<i16>(
            ordered
                .into_iter()
                .map(|value| ((value as u16) ^ (1_u16 << 15)) as i16)
                .collect::<Vec<_>>(),
            validity,
        ),
        PType::I32 => PrimitiveArray::new::<i32>(
            ordered
                .into_iter()
                .map(|value| ((value as u32) ^ (1_u32 << 31)) as i32)
                .collect::<Vec<_>>(),
            validity,
        ),
        PType::I64 => PrimitiveArray::new::<i64>(
            ordered
                .into_iter()
                .map(|value| (value ^ (1_u64 << 63)) as i64)
                .collect::<Vec<_>>(),
            validity,
        ),
        ptype => vortex_bail!("RangePackedArray does not support {ptype}"),
    })
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the encoded value retains the source integer word width"
)]
fn ordered_scalar(value: u64, ptype: PType, nullability: Nullability) -> VortexResult<Scalar> {
    Ok(match ptype {
        PType::U8 => Scalar::primitive(u8::try_from(value)?, nullability),
        PType::U16 => Scalar::primitive(u16::try_from(value)?, nullability),
        PType::U32 => Scalar::primitive(u32::try_from(value)?, nullability),
        PType::U64 => Scalar::primitive(value, nullability),
        PType::I8 => Scalar::primitive(((value as u8) ^ (1_u8 << 7)) as i8, nullability),
        PType::I16 => Scalar::primitive(((value as u16) ^ (1_u16 << 15)) as i16, nullability),
        PType::I32 => Scalar::primitive(((value as u32) ^ (1_u32 << 31)) as i32, nullability),
        PType::I64 => Scalar::primitive((value ^ (1_u64 << 63)) as i64, nullability),
        ptype => vortex_bail!("RangePackedArray does not support {ptype}"),
    })
}

impl RangePackedCodec {
    pub fn encode(values: &[u64], block_len: usize) -> VortexResult<Self> {
        vortex_ensure!(
            block_len > 0 && block_len <= MAX_BLOCK_LEN,
            "range packed block length must be in 1..={MAX_BLOCK_LEN}"
        );
        if values.is_empty() {
            return Ok(Self {
                len: 0,
                block_len,
                symbol_width: 0,
                bins: Vec::new(),
                block_offsets: vec![0],
                payload: Vec::new(),
            });
        }

        let (sample, minimum, maximum) = training_sample(values);
        let bins = cover_domain(optimize_bins(&sample), minimum, maximum);
        vortex_ensure!(
            bins.len().is_power_of_two(),
            "range packed bin count must be a power of two"
        );
        let symbols = assign_bins(values, &bins)?;
        let symbol_width = bit_width(u64::try_from(bins.len() - 1)?);
        let block_count = values.len().div_ceil(block_len);
        let mut block_offsets = Vec::with_capacity(block_count + 1);
        let mut payload = Vec::new();
        block_offsets.push(0);
        for block_index in 0..block_count {
            let start = block_index * block_len;
            let stop = (start + block_len).min(values.len());
            encode_block(
                &values[start..stop],
                &symbols[start..stop],
                &bins,
                symbol_width,
                &mut payload,
            )?;
            block_offsets.push(u32::try_from(payload.len())?);
        }

        Ok(Self {
            len: values.len(),
            block_len,
            symbol_width,
            bins,
            block_offsets,
            payload,
        })
    }

    pub fn decode(&self) -> VortexResult<Vec<u64>> {
        self.decode_mapped(|value| value)
    }

    pub fn decode_mapped<T, F>(&self, map: F) -> VortexResult<Vec<T>>
    where
        F: FnMut(u64) -> T,
    {
        let max_offset_bits = self.max_offset_bits();
        if max_offset_bits <= 40 {
            self.decode_mapped_with_width::<false, false, T, F>(map)
        } else if max_offset_bits <= 57 {
            self.decode_mapped_with_width::<false, true, T, F>(map)
        } else {
            self.decode_mapped_with_width::<true, true, T, F>(map)
        }
    }

    fn decode_mapped_with_width<const WIDE: bool, const PARALLEL: bool, T, F>(
        &self,
        mut map: F,
    ) -> VortexResult<Vec<T>>
    where
        F: FnMut(u64) -> T,
    {
        let mut values = Vec::with_capacity(self.len);
        let table = DecodeTable::new(&self.bins);
        for block_index in 0..self.block_count() {
            let payload = self.block_payload(block_index)?;
            let value_count = self.block_value_count(block_index);
            let output_start = values.len();
            decode_block::<WIDE, PARALLEL, T, F>(
                payload,
                value_count,
                self.symbol_width,
                &table,
                &mut values.spare_capacity_mut()[..value_count],
                &mut map,
            )?;
            // SAFETY: decode_block initialized each output slot.
            unsafe { values.set_len(output_start + value_count) };
        }
        Ok(values)
    }

    pub fn scalar_at(&self, index: usize) -> VortexResult<u64> {
        vortex_ensure!(index < self.len, "range packed index exceeds array length");
        let block_index = index / self.block_len;
        let index_in_block = index % self.block_len;
        let value_count = self.block_value_count(block_index);
        let payload = self.block_payload(block_index)?;
        let layout = BlockLayout::new(value_count, self.symbol_width, payload.len())?;
        let symbol = read_symbol(payload, index_in_block, self.symbol_width)?;
        let node = *self
            .bins
            .get(symbol)
            .ok_or_else(|| vortex_error::vortex_err!("range packed symbol exceeds bin table"))?;
        let checkpoint_index = index_in_block / CHECKPOINT_INTERVAL;
        let checkpoint_start = layout.checkpoints_start + checkpoint_index * size_of::<u16>();
        let checkpoint =
            u16::from_le_bytes([payload[checkpoint_start], payload[checkpoint_start + 1]]);
        let mut offset_position = usize::from(checkpoint);
        let scan_start = checkpoint_index * CHECKPOINT_INTERVAL;
        for preceding in scan_start..index_in_block {
            let preceding_symbol = read_symbol(payload, preceding, self.symbol_width)?;
            offset_position += usize::from(
                self.bins
                    .get(preceding_symbol)
                    .ok_or_else(|| {
                        vortex_error::vortex_err!("range packed symbol exceeds bin table")
                    })?
                    .offset_bits,
            );
        }
        let offset = read_bits_padded(
            &payload[layout.offsets_start..],
            offset_position,
            node.offset_bits,
        );
        Ok(node.lower.wrapping_add(offset))
    }

    pub fn encoded_size(&self) -> usize {
        self.bins.len() * (size_of::<u64>() + size_of::<u8>())
            + self.block_offsets.len() * size_of::<u32>()
            + self.payload.len()
    }

    pub fn bin_count(&self) -> usize {
        self.bins.len()
    }

    pub fn max_offset_bits(&self) -> u8 {
        self.bins
            .iter()
            .map(|bin| bin.offset_bits)
            .max()
            .unwrap_or_default()
    }

    pub fn offset_widths(&self) -> String {
        self.bins
            .iter()
            .map(|bin| bin.offset_bits.to_string())
            .collect::<Vec<_>>()
            .join(",")
    }

    fn block_count(&self) -> usize {
        self.len.div_ceil(self.block_len)
    }

    fn block_value_count(&self, block_index: usize) -> usize {
        (self.len - block_index * self.block_len).min(self.block_len)
    }

    fn block_payload(&self, block_index: usize) -> VortexResult<&[u8]> {
        let start = usize::try_from(self.block_offsets[block_index])?;
        let stop = usize::try_from(self.block_offsets[block_index + 1])?;
        Ok(&self.payload[start..stop])
    }
}

struct DecodeTable {
    lowers: [u64; MAX_BINS],
    widths: [u8; MAX_BINS],
    bin_count: usize,
}

impl DecodeTable {
    fn new(bins: &[Bin]) -> Self {
        let mut lowers = [0_u64; MAX_BINS];
        let mut widths = [0_u8; MAX_BINS];
        for (index, bin) in bins.iter().enumerate() {
            lowers[index] = bin.lower;
            widths[index] = bin.offset_bits;
        }
        Self {
            lowers,
            widths,
            bin_count: bins.len(),
        }
    }
}

struct BlockLayout {
    checkpoints_start: usize,
    offsets_start: usize,
}

impl BlockLayout {
    fn new(value_count: usize, symbol_width: u8, payload_len: usize) -> VortexResult<Self> {
        let symbol_bytes = (value_count * usize::from(symbol_width)).div_ceil(8);
        let checkpoints_start = symbol_bytes + SYMBOL_PADDING;
        let checkpoint_bytes = value_count.div_ceil(CHECKPOINT_INTERVAL) * size_of::<u16>();
        let offsets_start = checkpoints_start + checkpoint_bytes;
        vortex_ensure!(
            payload_len >= offsets_start + OFFSET_PADDING,
            "range packed block is too short"
        );
        Ok(Self {
            checkpoints_start,
            offsets_start,
        })
    }
}

fn encode_block(
    values: &[u64],
    symbols: &[u8],
    bins: &[Bin],
    symbol_width: u8,
    payload: &mut Vec<u8>,
) -> VortexResult<()> {
    let mut symbol_writer = BitWriter::with_capacity(values.len());
    let mut offset_writer = BitWriter::with_capacity(size_of_val(values));
    let mut checkpoints = Vec::with_capacity(values.len().div_ceil(CHECKPOINT_INTERVAL));
    for (index, (&value, &symbol)) in values.iter().zip(symbols).enumerate() {
        if index.is_multiple_of(CHECKPOINT_INTERVAL) {
            checkpoints.push(u16::try_from(offset_writer.bit_len())?);
        }
        symbol_writer.write(u64::from(symbol), symbol_width);
        let bin = bins[usize::from(symbol)];
        offset_writer.write(value - bin.lower, bin.offset_bits);
    }
    payload.extend_from_slice(&symbol_writer.finish());
    payload.extend_from_slice(&[0; SYMBOL_PADDING]);
    for checkpoint in checkpoints {
        payload.extend_from_slice(&checkpoint.to_le_bytes());
    }
    payload.extend_from_slice(&offset_writer.finish());
    payload.extend_from_slice(&[0; OFFSET_PADDING]);
    Ok(())
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the symbol mask never exceeds 63"
)]
fn decode_block<const WIDE: bool, const PARALLEL: bool, T, F>(
    payload: &[u8],
    value_count: usize,
    symbol_width: u8,
    table: &DecodeTable,
    values: &mut [MaybeUninit<T>],
    map: &mut F,
) -> VortexResult<()>
where
    F: FnMut(u64) -> T,
{
    let layout = BlockLayout::new(value_count, symbol_width, payload.len())?;
    let offsets = &payload[layout.offsets_start..];
    let mut offset_position = 0usize;
    let full_len = value_count / 8 * 8;
    let symbol_mask = low_mask(symbol_width);
    for base in (0..full_len).step_by(8) {
        let symbol_position = base * usize::from(symbol_width);
        let packed = read_bits_padded(payload, symbol_position, symbol_width * 8);
        if PARALLEL {
            let mut widths = [0_u8; 8];
            let mut lowers = [0_u64; 8];
            let mut positions = [0_usize; 8];
            for lane in 0..8 {
                let symbol =
                    ((packed >> (lane * usize::from(symbol_width))) & symbol_mask) as usize;
                debug_assert!(symbol < table.bin_count);
                // SAFETY: A power-of-two bin count makes every symbol code valid.
                widths[lane] = unsafe { *table.widths.get_unchecked(symbol) };
                // SAFETY: A power-of-two bin count makes every symbol code valid.
                lowers[lane] = unsafe { *table.lowers.get_unchecked(symbol) };
                positions[lane] = offset_position;
                offset_position += usize::from(widths[lane]);
            }
            for lane in 0..8 {
                let offset = read_offset::<WIDE>(offsets, positions[lane], widths[lane]);
                // SAFETY: This lane is within a complete eight-value output batch.
                unsafe {
                    values
                        .get_unchecked_mut(base + lane)
                        .write(map(lowers[lane].wrapping_add(offset)))
                };
            }
        } else {
            for lane in 0..8 {
                let symbol =
                    ((packed >> (lane * usize::from(symbol_width))) & symbol_mask) as usize;
                debug_assert!(symbol < table.bin_count);
                // SAFETY: A power-of-two bin count makes every symbol code valid.
                let width = unsafe { *table.widths.get_unchecked(symbol) };
                let offset = read_offset::<WIDE>(offsets, offset_position, width);
                offset_position += usize::from(width);
                // SAFETY: A power-of-two bin count makes every symbol code valid.
                let lower = unsafe { *table.lowers.get_unchecked(symbol) };
                // SAFETY: This lane is within a complete eight-value output batch.
                unsafe {
                    values
                        .get_unchecked_mut(base + lane)
                        .write(map(lower.wrapping_add(offset)))
                };
            }
        }
    }
    for index in full_len..value_count {
        let symbol = read_symbol(payload, index, symbol_width)?;
        debug_assert!(symbol < table.bin_count);
        // SAFETY: A power-of-two bin count makes every symbol code valid.
        let width = unsafe { *table.widths.get_unchecked(symbol) };
        let offset = read_offset::<WIDE>(offsets, offset_position, width);
        offset_position += usize::from(width);
        // SAFETY: A power-of-two bin count makes every symbol code valid.
        let lower = unsafe { *table.lowers.get_unchecked(symbol) };
        // SAFETY: The trailing index is less than value_count.
        unsafe {
            values
                .get_unchecked_mut(index)
                .write(map(lower.wrapping_add(offset)))
        };
    }
    let offset_bytes = offset_position.div_ceil(8);
    vortex_ensure!(
        offsets.len() == offset_bytes + OFFSET_PADDING,
        "range packed offset stream has unused bytes"
    );
    Ok(())
}

fn read_symbol(payload: &[u8], index: usize, symbol_width: u8) -> VortexResult<usize> {
    usize::try_from(read_bits_padded(
        payload,
        index * usize::from(symbol_width),
        symbol_width,
    ))
    .map_err(Into::into)
}

#[inline(always)]
fn read_offset<const WIDE: bool>(offsets: &[u8], bit_position: usize, width: u8) -> u64 {
    if WIDE {
        read_bits_padded(offsets, bit_position, width)
    } else if width == 0 {
        0
    } else {
        let byte_position = bit_position / 8;
        let bits_past_byte = bit_position % 8;
        // SAFETY: Offset streams include fifteen readable padding bytes.
        let packed = unsafe { read_u64_unaligned(offsets.as_ptr().add(byte_position)) };
        (packed >> bits_past_byte) & low_mask(width)
    }
}

fn read_bits_padded(bytes: &[u8], bit_position: usize, width: u8) -> u64 {
    if width == 0 {
        return 0;
    }
    let byte_position = bit_position / 8;
    let bits_past_byte = bit_position % 8;
    // SAFETY: Every encoded stream includes at least eight readable padding bytes.
    let first = unsafe { read_u64_unaligned(bytes.as_ptr().add(byte_position)) };
    if usize::from(width) <= 64 - bits_past_byte {
        (first >> bits_past_byte) & low_mask(width)
    } else {
        // SAFETY: Offset streams include fifteen readable padding bytes.
        let second = unsafe { read_u64_unaligned(bytes.as_ptr().add(byte_position + 7)) };
        let processed = 56 - bits_past_byte;
        ((first >> bits_past_byte) | (second << processed)) & low_mask(width)
    }
}

unsafe fn read_u64_unaligned(pointer: *const u8) -> u64 {
    // SAFETY: The caller provides eight readable bytes at the pointer.
    u64::from_le(unsafe { pointer.cast::<u64>().read_unaligned() })
}

fn assign_bins(values: &[u64], bins: &[Bin]) -> VortexResult<Vec<u8>> {
    values
        .iter()
        .map(|&value| {
            let symbol = bins
                .partition_point(|bin| bin.lower <= value)
                .saturating_sub(1);
            let bin = bins[symbol];
            vortex_ensure!(
                value - bin.lower <= low_mask(bin.offset_bits),
                "range packed value exceeds its bin"
            );
            Ok(u8::try_from(symbol)?)
        })
        .collect()
}

fn optimize_bins(sorted: &[u64]) -> Vec<Bin> {
    let mut segments = Vec::with_capacity(MAX_BINS);
    segments.push(0..sorted.len());
    let mut best_segments = segments.clone();
    let mut best_cost = partition_cost(sorted, &segments);

    while segments.len() < MAX_BINS {
        let best = segments
            .iter()
            .enumerate()
            .filter_map(|(segment_index, segment)| {
                best_split(sorted, segment).map(|split| (segment_index, split))
            })
            .max_by(|(_, left), (_, right)| {
                left.gain
                    .partial_cmp(&right.gain)
                    .unwrap_or(Ordering::Equal)
            });
        let Some((segment_index, split)) = best else {
            break;
        };
        if split.gain <= 0.0 {
            break;
        }
        let segment = segments.remove(segment_index);
        segments.push(segment.start..split.at);
        segments.push(split.at..segment.end);
        segments.sort_unstable_by_key(|segment| segment.start);
        let cost = partition_cost(sorted, &segments);
        if segments.len().is_power_of_two() && cost < best_cost {
            best_cost = cost;
            best_segments.clone_from(&segments);
        }
    }

    best_segments
        .into_iter()
        .map(|segment| Bin::from_values(sorted[segment.start], sorted[segment.end - 1]))
        .collect()
}

fn partition_cost(sorted: &[u64], segments: &[Range<usize>]) -> f64 {
    let residual_cost = segments
        .iter()
        .map(|segment| {
            segment.len() as f64
                * f64::from(bit_width(sorted[segment.end - 1] - sorted[segment.start]))
        })
        .sum::<f64>();
    let symbol_cost = sorted.len() as f64 * f64::from(bit_width((segments.len() - 1) as u64));
    residual_cost + symbol_cost + segments.len() as f64 * BIN_TABLE_COST_BITS
}

fn training_sample(values: &[u64]) -> (Vec<u64>, u64, u64) {
    let mut minimum = u64::MAX;
    let mut maximum = 0;
    for &value in values {
        minimum = minimum.min(value);
        maximum = maximum.max(value);
    }
    let mut sample = if values.len() <= TRAINING_SAMPLE_SIZE {
        values.to_vec()
    } else {
        (0..TRAINING_SAMPLE_SIZE)
            .map(|index| values[index * values.len() / TRAINING_SAMPLE_SIZE])
            .collect()
    };
    sample.push(minimum);
    sample.push(maximum);
    sample.sort_unstable();
    (sample, minimum, maximum)
}

fn cover_domain(mut bins: Vec<Bin>, minimum: u64, maximum: u64) -> Vec<Bin> {
    bins[0].lower = minimum;
    for index in 0..bins.len() - 1 {
        let upper = bins[index + 1].lower - 1;
        bins[index].offset_bits = bit_width(upper - bins[index].lower);
    }
    let last = bins.len() - 1;
    bins[last].offset_bits = bit_width(maximum - bins[last].lower);
    bins
}

#[derive(Clone, Copy)]
struct Split {
    at: usize,
    gain: f64,
}

fn best_split(sorted: &[u64], segment: &Range<usize>) -> Option<Split> {
    if segment.len() < 2 || sorted[segment.start] == sorted[segment.end - 1] {
        return None;
    }
    let whole_cost = segment.len() as f64
        * f64::from(bit_width(sorted[segment.end - 1] - sorted[segment.start]));
    let mut best = None;
    for candidate in 1..SPLIT_CANDIDATES {
        let mut at = segment.start + segment.len() * candidate / SPLIT_CANDIDATES;
        if at <= segment.start || at >= segment.end {
            continue;
        }
        while at < segment.end && sorted[at - 1] == sorted[at] {
            at += 1;
        }
        if at >= segment.end {
            continue;
        }
        let left_cost = (at - segment.start) as f64
            * f64::from(bit_width(sorted[at - 1] - sorted[segment.start]));
        let right_cost =
            (segment.end - at) as f64 * f64::from(bit_width(sorted[segment.end - 1] - sorted[at]));
        let split = Split {
            at,
            gain: whole_cost - left_cost - right_cost,
        };
        if best.is_none_or(|current: Split| split.gain > current.gain) {
            best = Some(split);
        }
    }
    best
}

impl Bin {
    fn from_values(lower: u64, upper: u64) -> Self {
        Self {
            lower,
            offset_bits: bit_width(upper - lower),
        }
    }
}

fn bit_width(value: u64) -> u8 {
    u8::try_from(u64::BITS - value.leading_zeros()).unwrap_or(64)
}

fn low_mask(bits: u8) -> u64 {
    match bits {
        0 => 0,
        64 => u64::MAX,
        _ => (1_u64 << bits) - 1,
    }
}

struct BitWriter {
    bytes: Vec<u8>,
    pending: u64,
    pending_bits: u8,
    bit_len: usize,
}

impl BitWriter {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(capacity),
            pending: 0,
            pending_bits: 0,
            bit_len: 0,
        }
    }

    fn bit_len(&self) -> usize {
        self.bit_len
    }

    fn write(&mut self, value: u64, width: u8) {
        self.bit_len += usize::from(width);
        if width == 0 {
            return;
        }
        let available = 64 - self.pending_bits;
        self.pending |= value << self.pending_bits;
        if width < available {
            self.pending_bits += width;
            return;
        }
        self.bytes.extend_from_slice(&self.pending.to_le_bytes());
        let remaining = width - available;
        self.pending = if remaining == 0 {
            0
        } else {
            value >> available
        };
        self.pending_bits = remaining;
    }

    fn finish(mut self) -> Vec<u8> {
        if self.pending_bits > 0 {
            let byte_count = usize::from(self.pending_bits).div_ceil(8);
            self.bytes
                .extend_from_slice(&self.pending.to_le_bytes()[..byte_count]);
        }
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_error::VortexResult;

    use super::RangePacked;
    use super::RangePackedCodec;

    #[test]
    fn clustered_roundtrip_and_scalar_access() -> VortexResult<()> {
        let values = (0_u64..20_000)
            .map(|index| match index % 4 {
                0 => index % 17,
                1 => 1_000_000 + index % 31,
                2 => u64::MAX - index % 13,
                _ => 1_000 + index % 7,
            })
            .collect::<Vec<_>>();
        let encoded = RangePackedCodec::encode(&values, 1_024)?;
        assert_eq!(encoded.decode()?, values);
        for index in [0, 1, 31, 32, 33, 1_023, 1_024, 19_999] {
            assert_eq!(encoded.scalar_at(index)?, values[index]);
        }
        Ok(())
    }

    #[test]
    fn full_width_roundtrip() -> VortexResult<()> {
        let values = [0, u64::MAX, 1, u64::MAX - 1];
        let encoded = RangePackedCodec::encode(&values, 4)?;
        assert_eq!(encoded.decode()?, values);
        for (index, value) in values.into_iter().enumerate() {
            assert_eq!(encoded.scalar_at(index)?, value);
        }
        Ok(())
    }

    #[test]
    fn nullable_dense_roundtrip_and_rank_boundaries() -> VortexResult<()> {
        let expected = PrimitiveArray::from_option_iter((0_i64..600).map(|value| {
            (!matches!(value, 0 | 1 | 63 | 64 | 255 | 256 | 257 | 511)).then_some(value - 300)
        }));
        let session = array_session();
        let mut ctx = session.create_execution_ctx();
        let encoded = RangePacked::from_primitive(expected.as_view(), &mut ctx)?;
        assert_arrays_eq!(encoded, expected, &mut ctx);
        for index in [0, 1, 2, 63, 64, 65, 255, 256, 257, 258, 511, 512, 599] {
            let actual = encoded.execute_scalar(index, &mut ctx)?;
            let expected_scalar = expected
                .clone()
                .into_array()
                .execute_scalar(index, &mut ctx)?;
            assert_eq!(actual, expected_scalar);
        }
        Ok(())
    }
}
