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
use vortex_array::arrays::Bool;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::arrays::slice::SliceReduce;
use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
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
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::BinTableView;
use crate::MAX_BLOCK_LEN;
use crate::RangePackedCodec;
use crate::RangePackedDecoder;
use crate::VALIDITY_CHECKPOINT_INTERVAL;
use crate::low_mask;

const METADATA_VERSION: u8 = 1;
const METADATA_LEN: usize = 34;

/// Fixed-bin range packing with bounded scalar access.
pub type RangePackedArray = Array<RangePacked>;

#[array_slots(RangePacked)]
pub struct RangePackedSlots {
    #[slot(0)]
    pub bin_lowers: ArrayRef,
    #[slot(1)]
    pub offset_widths: ArrayRef,
    #[slot(2)]
    pub block_offsets: ArrayRef,
    #[slot(3)]
    pub payload: ArrayRef,
    #[slot(4)]
    pub validity_bits: Option<ArrayRef>,
    #[slot(5)]
    pub rank_checkpoints: Option<ArrayRef>,
}

#[derive(Clone, Debug)]
pub struct RangePackedData {
    unsliced_len: usize,
    dense_len: usize,
    slice_start: usize,
    slice_stop: usize,
    payload_len: usize,
    symbol_width: u8,
}

impl Display for RangePackedData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "bins: {}, slice: {}..{}",
            self.bin_count(),
            self.slice_start,
            self.slice_stop
        )
    }
}

impl ArrayHash for RangePackedData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        self.unsliced_len.hash(state);
        self.dense_len.hash(state);
        self.slice_start.hash(state);
        self.slice_stop.hash(state);
        self.payload_len.hash(state);
        self.symbol_width.hash(state);
    }
}

impl ArrayEq for RangePackedData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.unsliced_len == other.unsliced_len
            && self.dense_len == other.dense_len
            && self.slice_start == other.slice_start
            && self.slice_stop == other.slice_stop
            && self.payload_len == other.payload_len
            && self.symbol_width == other.symbol_width
    }
}

#[derive(Clone, Debug)]
pub struct RangePacked;

impl VTable for RangePacked {
    type TypedArrayData = RangePackedData;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.range_packed");
        *ID
    }

    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        data.validate(dtype, len, RangePackedSlotsView::from_slots(slots))
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, index: usize) -> BufferHandle {
        vortex_panic!("RangePackedArray buffer index {index} is invalid")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, index: usize) -> Option<String> {
        vortex_panic!("RangePackedArray buffer index {index} is invalid")
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
        Ok(Some(RangePackedMetadata::from_data(array.data())?.encode()))
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
        let metadata = RangePackedMetadata::decode(metadata)?;
        let unsliced_len = usize::try_from(metadata.unsliced_len)?;
        let dense_len = usize::try_from(metadata.dense_len)?;
        let slice_start = usize::try_from(metadata.slice_start)?;
        let slice_stop = slice_start
            .checked_add(len)
            .ok_or_else(|| vortex_error::vortex_err!("RangePacked slice length overflows"))?;
        let payload_len = usize::try_from(metadata.payload_len)?;
        let bin_count = bin_count(dense_len, metadata.symbol_width)?;
        let block_count = dense_len.div_ceil(MAX_BLOCK_LEN);
        vortex_ensure!(
            matches!(children.len(), 4 | 6),
            "RangePackedArray expects four or six children"
        );
        let bin_lowers = children.get(0, &primitive_dtype(PType::U64), bin_count)?;
        let offset_widths = children.get(1, &primitive_dtype(PType::U8), bin_count)?;
        let block_offsets = children.get(2, &primitive_dtype(PType::U32), block_count + 1)?;
        let payload = children.get(3, &primitive_dtype(PType::U8), payload_len)?;
        let (validity_bits, rank_checkpoints) = if children.len() == 6 {
            (
                Some(children.get(4, &DType::Bool(NonNullable), unsliced_len)?),
                Some(children.get(
                    5,
                    &primitive_dtype(PType::U32),
                    unsliced_len.div_ceil(VALIDITY_CHECKPOINT_INTERVAL),
                )?),
            )
        } else {
            (None, None)
        };
        let data = RangePackedData {
            unsliced_len,
            dense_len,
            slice_start,
            slice_stop,
            payload_len,
            symbol_width: metadata.symbol_width,
        };
        let slots = RangePackedSlots {
            bin_lowers,
            offset_widths,
            block_offsets,
            payload,
            validity_bits,
            rank_checkpoints,
        };
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots.into_slots()))
    }

    fn slot_name(_array: ArrayView<'_, Self>, index: usize) -> String {
        RangePackedSlots::NAMES[index].to_string()
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(
            decode_primitive(array.as_view())?.into_array(),
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

impl OperationsVTable<RangePacked> for RangePacked {
    fn scalar_at(
        array: ArrayView<'_, RangePacked>,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let global_index = array.data().slice_start + index;
        if !is_valid(array, global_index) {
            return Ok(Scalar::null(array.dtype().clone()));
        }
        let dense_index = dense_rank(array, global_index)?;
        ordered_scalar(
            with_decoder(array, |decoder| decoder.scalar_at(dense_index))?,
            array.dtype().as_ptype(),
            array.dtype().nullability(),
        )
    }
}

impl ValidityVTable<RangePacked> for RangePacked {
    fn validity(array: ArrayView<'_, RangePacked>) -> VortexResult<Validity> {
        array
            .unsliced_validity()
            .slice(array.data().slice_start..array.data().slice_stop)
    }
}

impl SliceReduce for RangePacked {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let data = array.data().slice(range);
        Ok(Some(
            Array::try_from_parts(
                ArrayParts::new(RangePacked, array.dtype().clone(), data.len(), data)
                    .with_slots(array.slots().iter().cloned().collect()),
            )?
            .into_array(),
        ))
    }
}

static RULES: ParentRuleSet<RangePacked> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&SliceReduceAdaptor(RangePacked))]);

pub trait RangePackedArrayExt: TypedArrayRef<RangePacked> + RangePackedArraySlotsExt {
    fn unsliced_validity(&self) -> Validity {
        child_to_validity(
            self.as_ref().slots()[RangePackedSlots::VALIDITY_BITS].as_ref(),
            self.as_ref().dtype().nullability(),
        )
    }
}

impl<T: TypedArrayRef<RangePacked>> RangePackedArrayExt for T {}

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
        let dense_len = mask.true_count();
        let dense_ordered = ordered_primitive(array)?
            .into_iter()
            .zip(mask.iter())
            .filter_map(|(value, valid)| valid.then_some(value))
            .collect::<Vec<_>>();
        let codec = RangePackedCodec::encode(&dense_ordered, MAX_BLOCK_LEN)?;
        let actual_nulls = dense_len != array.len();
        let validity_bits = actual_nulls.then(|| BoolArray::from_iter(mask.iter()).into_array());
        let rank_checkpoints = actual_nulls
            .then(|| rank_checkpoints(mask.iter(), dense_len))
            .transpose()?
            .map(|ranks| PrimitiveArray::from_iter(ranks).into_array());
        let data = RangePackedData {
            unsliced_len: array.len(),
            dense_len,
            slice_start: 0,
            slice_stop: array.len(),
            payload_len: codec.payload.len(),
            symbol_width: codec.symbol_width,
        };
        let slots = RangePackedSlots {
            bin_lowers: PrimitiveArray::from_iter(codec.bins.iter().map(|bin| bin.lower))
                .into_array(),
            offset_widths: PrimitiveArray::from_iter(codec.bins.iter().map(|bin| bin.offset_bits))
                .into_array(),
            block_offsets: PrimitiveArray::from_iter(codec.block_offsets).into_array(),
            payload: PrimitiveArray::from_iter(codec.payload).into_array(),
            validity_bits,
            rank_checkpoints,
        };
        Array::try_from_parts(
            ArrayParts::new(
                RangePacked,
                DType::Primitive(array.ptype(), validity.nullability()),
                array.len(),
                data,
            )
            .with_slots(slots.into_slots()),
        )
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
        let dense_start = dense_rank(array, array.data().slice_start)?;
        let dense_stop = dense_rank(array, array.data().slice_stop)?;
        let dense_values = with_decoder(array, |decoder| {
            decoder.decode_mapped_range(dense_start..dense_stop, map)
        })?;
        let Some(validity_bits) = array.validity_bits() else {
            return Ok(dense_values);
        };
        let validity_bits = validity_bits.as_::<Bool>();
        let bits = validity_bits.bit_buffer_view();
        let mut dense = dense_values.into_iter();
        let mut values = Vec::with_capacity(array.len());
        for valid in bits
            .slice(array.data().slice_start..array.data().slice_stop)
            .iter()
        {
            values.push(if valid {
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

impl RangePackedData {
    fn validate(
        &self,
        dtype: &DType,
        len: usize,
        slots: RangePackedSlotsView<'_>,
    ) -> VortexResult<()> {
        vortex_ensure!(dtype.is_int(), "RangePackedArray requires an integer dtype");
        vortex_ensure!(
            self.slice_start <= self.slice_stop && self.slice_stop <= self.unsliced_len,
            "RangePacked slice exceeds its source length"
        );
        vortex_ensure!(len == self.len(), "RangePacked slice length is invalid");
        vortex_ensure!(
            self.dense_len <= self.unsliced_len,
            "RangePacked dense length exceeds its source length"
        );
        let bin_count = self.bin_count();
        validate_primitive_child(slots.bin_lowers, PType::U64, bin_count, "bin lowers")?;
        validate_primitive_child(slots.offset_widths, PType::U8, bin_count, "offset widths")?;
        validate_primitive_child(
            slots.block_offsets,
            PType::U32,
            self.dense_len.div_ceil(MAX_BLOCK_LEN) + 1,
            "block offsets",
        )?;
        validate_primitive_child(slots.payload, PType::U8, self.payload_len, "payload")?;
        vortex_ensure!(
            slots.validity_bits.is_some() == slots.rank_checkpoints.is_some(),
            "RangePacked validity and rank children must occur together"
        );
        if let (Some(validity_bits), Some(ranks)) = (slots.validity_bits, slots.rank_checkpoints) {
            vortex_ensure!(
                dtype.is_nullable(),
                "RangePacked validity requires a nullable dtype"
            );
            vortex_ensure!(
                validity_bits.is::<Bool>()
                    && validity_bits.dtype() == &DType::Bool(NonNullable)
                    && validity_bits.len() == self.unsliced_len,
                "RangePacked validity child is invalid"
            );
            validate_primitive_child(
                ranks,
                PType::U32,
                self.unsliced_len.div_ceil(VALIDITY_CHECKPOINT_INTERVAL),
                "rank checkpoints",
            )?;
            validate_ranks(
                validity_bits.as_::<Bool>(),
                ranks.as_::<Primitive>().as_slice::<u32>(),
                self.dense_len,
            )?;
        } else {
            vortex_ensure!(
                self.dense_len == self.unsliced_len,
                "RangePacked without validity must store every value"
            );
        }
        validate_bins(
            slots.bin_lowers.as_::<Primitive>().as_slice::<u64>(),
            slots.offset_widths.as_::<Primitive>().as_slice::<u8>(),
            dtype.as_ptype(),
        )?;
        validate_block_offsets(
            slots.block_offsets.as_::<Primitive>().as_slice::<u32>(),
            self.payload_len,
        )?;
        Ok(())
    }

    fn bin_count(&self) -> usize {
        if self.dense_len == 0 {
            0
        } else {
            1_usize << self.symbol_width
        }
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

fn validate_primitive_child(
    child: &ArrayRef,
    ptype: PType,
    len: usize,
    name: &str,
) -> VortexResult<()> {
    vortex_ensure!(
        child.is::<Primitive>() && child.dtype() == &primitive_dtype(ptype) && child.len() == len,
        "RangePacked {name} child is invalid"
    );
    Ok(())
}

fn validate_bins(lowers: &[u64], widths: &[u8], ptype: PType) -> VortexResult<()> {
    let word_width = u8::try_from(ptype.bit_width())?;
    for (index, (&lower, &width)) in lowers.iter().zip(widths).enumerate() {
        vortex_ensure!(width <= word_width, "RangePacked offset width is invalid");
        if word_width < 64 {
            vortex_ensure!(
                lower <= low_mask(word_width),
                "RangePacked bin lower exceeds its integer width"
            );
        }
        if let Some(&next) = lowers.get(index + 1) {
            vortex_ensure!(next > lower, "RangePacked bin lowers are not ordered");
            vortex_ensure!(
                next - lower - 1 <= low_mask(width),
                "RangePacked bin does not cover the next boundary"
            );
        }
    }
    Ok(())
}

fn validate_block_offsets(offsets: &[u32], payload_len: usize) -> VortexResult<()> {
    vortex_ensure!(
        offsets.first() == Some(&0),
        "RangePacked offsets must start at zero"
    );
    vortex_ensure!(
        offsets.last().copied().map(usize::try_from).transpose()? == Some(payload_len),
        "RangePacked offsets must end at the payload length"
    );
    vortex_ensure!(
        offsets.windows(2).all(|pair| pair[0] <= pair[1]),
        "RangePacked offsets are not ordered"
    );
    Ok(())
}

fn validate_ranks(bits: ArrayView<'_, Bool>, ranks: &[u32], dense_len: usize) -> VortexResult<()> {
    let bits = bits.bit_buffer_view();
    let mut rank = 0usize;
    for (checkpoint, &stored) in ranks.iter().enumerate() {
        vortex_ensure!(
            usize::try_from(stored)? == rank,
            "RangePacked rank checkpoint is invalid"
        );
        let start = checkpoint * VALIDITY_CHECKPOINT_INTERVAL;
        let stop = (start + VALIDITY_CHECKPOINT_INTERVAL).min(bits.len());
        rank += bits.slice(start..stop).true_count();
    }
    vortex_ensure!(rank == dense_len, "RangePacked dense length is invalid");
    Ok(())
}

fn rank_checkpoints(
    validity: impl Iterator<Item = bool>,
    valid_count: usize,
) -> VortexResult<Vec<u32>> {
    let mut ranks = Vec::new();
    let mut rank = 0usize;
    for (index, valid) in validity.enumerate() {
        if index.is_multiple_of(VALIDITY_CHECKPOINT_INTERVAL) {
            ranks.push(u32::try_from(rank)?);
        }
        rank += usize::from(valid);
    }
    vortex_ensure!(rank == valid_count, "RangePacked validity count differs");
    Ok(ranks)
}

#[inline]
fn is_valid(array: ArrayView<'_, RangePacked>, global_index: usize) -> bool {
    array
        .validity_bits()
        .is_none_or(|validity| validity.as_::<Bool>().bit_buffer_view().value(global_index))
}

fn dense_rank(array: ArrayView<'_, RangePacked>, global_index: usize) -> VortexResult<usize> {
    let Some(validity) = array.validity_bits() else {
        return Ok(global_index);
    };
    if global_index == array.data().unsliced_len {
        return Ok(array.data().dense_len);
    }
    let checkpoint = global_index / VALIDITY_CHECKPOINT_INTERVAL;
    let checkpoint_start = checkpoint * VALIDITY_CHECKPOINT_INTERVAL;
    let base = usize::try_from(
        array
            .rank_checkpoints()
            .ok_or_else(|| vortex_error::vortex_err!("RangePacked rank child is missing"))?
            .as_::<Primitive>()
            .as_slice::<u32>()[checkpoint],
    )?;
    Ok(base
        + validity
            .as_::<Bool>()
            .bit_buffer_view()
            .slice(checkpoint_start..global_index)
            .true_count())
}

fn with_decoder<T>(
    array: ArrayView<'_, RangePacked>,
    function: impl FnOnce(RangePackedDecoder<'_>) -> VortexResult<T>,
) -> VortexResult<T> {
    let bin_lowers = array.bin_lowers().as_::<Primitive>();
    let offset_widths = array.offset_widths().as_::<Primitive>();
    let block_offsets = array.block_offsets().as_::<Primitive>();
    let payload = array.payload().as_::<Primitive>();
    let decoder = RangePackedDecoder::try_new(
        array.data().dense_len,
        MAX_BLOCK_LEN,
        array.data().symbol_width,
        BinTableView::Split {
            lowers: bin_lowers.as_slice::<u64>(),
            widths: offset_widths.as_slice::<u8>(),
        },
        block_offsets.as_slice::<u32>(),
        payload.as_slice::<u8>(),
    )?;
    function(decoder)
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

fn decode_primitive(array: ArrayView<'_, RangePacked>) -> VortexResult<PrimitiveArray> {
    let ordered = RangePacked::decode_mapped(array, |value| value, 0)?;
    let validity = array.validity()?;
    Ok(match array.dtype().as_ptype() {
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
                .map(|value| {
                    u8::try_from(value).map(|value| i8::from_le_bytes([value ^ (1_u8 << 7)]))
                })
                .collect::<Result<Vec<_>, _>>()?,
            validity,
        ),
        PType::I16 => PrimitiveArray::new::<i16>(
            ordered
                .into_iter()
                .map(|value| {
                    u16::try_from(value)
                        .map(|value| i16::from_le_bytes((value ^ (1_u16 << 15)).to_le_bytes()))
                })
                .collect::<Result<Vec<_>, _>>()?,
            validity,
        ),
        PType::I32 => PrimitiveArray::new::<i32>(
            ordered
                .into_iter()
                .map(|value| {
                    u32::try_from(value)
                        .map(|value| i32::from_le_bytes((value ^ (1_u32 << 31)).to_le_bytes()))
                })
                .collect::<Result<Vec<_>, _>>()?,
            validity,
        ),
        PType::I64 => PrimitiveArray::new::<i64>(
            ordered
                .into_iter()
                .map(|value| i64::from_le_bytes((value ^ (1_u64 << 63)).to_le_bytes()))
                .collect::<Vec<_>>(),
            validity,
        ),
        ptype => vortex_bail!("RangePackedArray does not support {ptype}"),
    })
}

fn ordered_scalar(value: u64, ptype: PType, nullability: Nullability) -> VortexResult<Scalar> {
    Ok(match ptype {
        PType::U8 => Scalar::primitive(u8::try_from(value)?, nullability),
        PType::U16 => Scalar::primitive(u16::try_from(value)?, nullability),
        PType::U32 => Scalar::primitive(u32::try_from(value)?, nullability),
        PType::U64 => Scalar::primitive(value, nullability),
        PType::I8 => Scalar::primitive(
            i8::from_le_bytes([u8::try_from(value)? ^ (1_u8 << 7)]),
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
        ptype => vortex_bail!("RangePackedArray does not support {ptype}"),
    })
}

fn primitive_dtype(ptype: PType) -> DType {
    DType::Primitive(ptype, NonNullable)
}

fn bin_count(dense_len: usize, symbol_width: u8) -> VortexResult<usize> {
    vortex_ensure!(
        symbol_width <= 6,
        "RangePacked symbol width exceeds six bits"
    );
    Ok(if dense_len == 0 {
        0
    } else {
        1_usize << symbol_width
    })
}

#[derive(Clone, Copy)]
struct RangePackedMetadata {
    unsliced_len: u64,
    dense_len: u64,
    slice_start: u64,
    payload_len: u64,
    symbol_width: u8,
}

impl RangePackedMetadata {
    fn from_data(data: &RangePackedData) -> VortexResult<Self> {
        Ok(Self {
            unsliced_len: u64::try_from(data.unsliced_len)?,
            dense_len: u64::try_from(data.dense_len)?,
            slice_start: u64::try_from(data.slice_start)?,
            payload_len: u64::try_from(data.payload_len)?,
            symbol_width: data.symbol_width,
        })
    }

    fn encode(self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(METADATA_LEN);
        bytes.push(METADATA_VERSION);
        bytes.extend_from_slice(&self.unsliced_len.to_le_bytes());
        bytes.extend_from_slice(&self.dense_len.to_le_bytes());
        bytes.extend_from_slice(&self.slice_start.to_le_bytes());
        bytes.extend_from_slice(&self.payload_len.to_le_bytes());
        bytes.push(self.symbol_width);
        bytes
    }

    fn decode(bytes: &[u8]) -> VortexResult<Self> {
        vortex_ensure!(
            bytes.len() == METADATA_LEN,
            "RangePacked metadata requires {METADATA_LEN} bytes"
        );
        vortex_ensure!(
            bytes[0] == METADATA_VERSION,
            "unsupported RangePacked metadata version {}",
            bytes[0]
        );
        Ok(Self {
            unsliced_len: read_u64(bytes, 1),
            dense_len: read_u64(bytes, 9),
            slice_start: read_u64(bytes, 17),
            payload_len: read_u64(bytes, 25),
            symbol_width: bytes[33],
        })
    }
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        bytes[offset..offset + size_of::<u64>()]
            .try_into()
            .unwrap_or_else(|_| unreachable!("validated RangePacked metadata length")),
    )
}
