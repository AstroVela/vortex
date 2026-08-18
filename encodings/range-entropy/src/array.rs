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
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::dtype::PType;
use vortex_array::dtype::half::f16;
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

use crate::RangeEntropyCodec;
use crate::RangeEntropyParts;
use crate::rules::RULES;

const METADATA_VERSION: u8 = 1;
const METADATA_LEN: usize = 23;
const MAX_SCALE_BITS: u8 = 10;
const MAX_BINS: usize = 64;

/// A Vortex array with range-bin identifiers and fixed-width offsets.
pub type RangeEntropyArray = Array<RangeEntropy>;

#[array_slots(RangeEntropy)]
pub struct RangeEntropySlots {
    /// Lower bound for each range bin.
    #[slot(0)]
    pub bin_lowers: ArrayRef,
    /// Fixed offset width for each range bin.
    #[slot(1)]
    pub offset_widths: ArrayRef,
    /// Quantized tANS weight for each range bin.
    #[slot(2)]
    pub weights: ArrayRef,
    /// Byte offset for each restart block.
    #[slot(3)]
    pub block_offsets: ArrayRef,
    /// Validity for the unsliced logical values.
    #[slot(4)]
    pub validity: Option<ArrayRef>,
}

#[derive(Clone, Debug)]
pub struct RangeEntropyData {
    payload: BufferHandle,
    block_len: usize,
    scale_bits: u8,
    unsliced_len: usize,
    slice_start: usize,
    slice_stop: usize,
}

impl Display for RangeEntropyData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "block_len: {}, scale_bits: {}, slice: {}..{}",
            self.block_len, self.scale_bits, self.slice_start, self.slice_stop
        )
    }
}

impl ArrayHash for RangeEntropyData {
    fn array_hash<H: Hasher>(&self, state: &mut H, accuracy: EqMode) {
        self.payload.array_hash(state, accuracy);
        self.block_len.hash(state);
        self.scale_bits.hash(state);
        self.unsliced_len.hash(state);
        self.slice_start.hash(state);
        self.slice_stop.hash(state);
    }
}

impl ArrayEq for RangeEntropyData {
    fn array_eq(&self, other: &Self, accuracy: EqMode) -> bool {
        self.payload.array_eq(&other.payload, accuracy)
            && self.block_len == other.block_len
            && self.scale_bits == other.scale_bits
            && self.unsliced_len == other.unsliced_len
            && self.slice_start == other.slice_start
            && self.slice_stop == other.slice_stop
    }
}

#[derive(Clone, Debug)]
pub struct RangeEntropy;

impl VTable for RangeEntropy {
    type TypedArrayData = RangeEntropyData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.range_entropy");
        *ID
    }

    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let slots = RangeEntropySlotsView::from_slots(slots);
        let validity = child_to_validity(slots.validity, dtype.nullability());
        data.validate(dtype, len, slots, &validity)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        1
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        match idx {
            0 => array.payload.clone(),
            _ => vortex_panic!("RangeEntropyArray buffer index {idx} out of bounds"),
        }
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        match idx {
            0 => Some("payload".to_string()),
            _ => vortex_panic!("RangeEntropyArray buffer index {idx} out of bounds"),
        }
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_ensure!(
            buffers.len() == 1,
            "RangeEntropyArray expects one buffer, got {}",
            buffers.len()
        );
        let mut data = array.data().clone();
        data.payload = buffers[0].clone();
        Ok(
            ArrayParts::new(self.clone(), array.dtype().clone(), array.len(), data)
                .with_slots(array.slots().iter().cloned().collect()),
        )
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        let bin_count = u8::try_from(array.bin_lowers().len())?;
        Ok(Some(
            RangeEntropyMetadata {
                scale_bits: array.scale_bits,
                bin_count,
                block_len: u32::try_from(array.block_len)?,
                unsliced_len: u64::try_from(array.unsliced_len)?,
                slice_start: u64::try_from(array.slice_start)?,
            }
            .encode(),
        ))
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
        vortex_ensure!(
            buffers.len() == 1,
            "RangeEntropyArray expects one buffer, got {}",
            buffers.len()
        );
        let metadata = RangeEntropyMetadata::decode(metadata)?;
        let unsliced_len = usize::try_from(metadata.unsliced_len)?;
        let slice_start = usize::try_from(metadata.slice_start)?;
        let slice_stop = slice_start
            .checked_add(len)
            .ok_or_else(|| vortex_error::vortex_err!("range entropy slice length overflows"))?;
        let block_len = usize::try_from(metadata.block_len)?;
        vortex_ensure!(block_len > 0, "range entropy block length must be positive");
        let block_offsets_len = unsliced_len.div_ceil(block_len) + 1;
        let bin_count = usize::from(metadata.bin_count);

        let bin_lowers = children.get(0, &primitive_dtype(PType::U64), bin_count)?;
        let offset_widths = children.get(1, &primitive_dtype(PType::U8), bin_count)?;
        let weights = children.get(2, &primitive_dtype(PType::U16), bin_count)?;
        let block_offsets = children.get(3, &primitive_dtype(PType::U32), block_offsets_len)?;
        let validity = match children.len() {
            4 => Validity::from(dtype.nullability()),
            5 => Validity::Array(children.get(4, &Validity::DTYPE, unsliced_len)?),
            count => vortex_bail!("RangeEntropyArray expects four or five children, got {count}"),
        };
        let slots = RangeEntropySlots {
            bin_lowers,
            offset_widths,
            weights,
            block_offsets,
            validity: validity_to_child(&validity, unsliced_len),
        }
        .into_slots();
        let data = RangeEntropyData {
            payload: buffers[0].clone(),
            block_len,
            scale_bits: metadata.scale_bits,
            unsliced_len,
            slice_start,
            slice_stop,
        };
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        RangeEntropySlots::NAMES[idx].to_string()
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

impl OperationsVTable<RangeEntropy> for RangeEntropy {
    fn scalar_at(
        array: ArrayView<'_, RangeEntropy>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let codec = codec_from_array(array, ctx)?;
        scalar_from_latent(
            codec.value(array.slice_start + index)?,
            array.dtype().as_ptype(),
            array.dtype().nullability(),
        )
    }
}

impl ValidityVTable<RangeEntropy> for RangeEntropy {
    fn validity(array: ArrayView<'_, RangeEntropy>) -> VortexResult<Validity> {
        array
            .unsliced_validity()
            .slice(array.slice_start..array.slice_stop)
    }
}

impl SliceReduce for RangeEntropy {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let data = array.data().slice(range);
        Ok(Some(
            Array::try_from_parts(
                ArrayParts::new(RangeEntropy, array.dtype().clone(), data.len(), data)
                    .with_slots(array.slots().iter().cloned().collect()),
            )?
            .into_array(),
        ))
    }
}

pub trait RangeEntropyArrayExt: TypedArrayRef<RangeEntropy> + RangeEntropyArraySlotsExt {
    /// Return the validity for all values before a logical slice.
    fn unsliced_validity(&self) -> Validity {
        child_to_validity(
            self.as_ref().slots()[RangeEntropySlots::VALIDITY].as_ref(),
            self.as_ref().dtype().nullability(),
        )
    }

    /// Decode the logical slice to a canonical primitive array.
    fn decompress(&self, ctx: &mut ExecutionCtx) -> VortexResult<PrimitiveArray> {
        decompress_array(self.to_owned().as_view(), ctx)
    }
}

impl<T: TypedArrayRef<RangeEntropy>> RangeEntropyArrayExt for T {}

impl RangeEntropy {
    /// Encode a canonical primitive array.
    pub fn from_primitive(
        array: ArrayView<'_, Primitive>,
        block_len: usize,
    ) -> VortexResult<RangeEntropyArray> {
        let dtype = array.dtype().clone();
        let validity = array.validity()?;
        let codec = RangeEntropyCodec::encode(&ordered_latents(array), block_len)?;
        let parts = codec.into_parts();
        let unsliced_len = parts.len;
        let data = RangeEntropyData {
            payload: BufferHandle::new_host(parts.payload),
            block_len: parts.block_len,
            scale_bits: parts.scale_bits,
            unsliced_len,
            slice_start: 0,
            slice_stop: unsliced_len,
        };
        let slots = RangeEntropySlots {
            bin_lowers: PrimitiveArray::from_iter(parts.bin_lowers).into_array(),
            offset_widths: PrimitiveArray::from_iter(parts.offset_widths).into_array(),
            weights: PrimitiveArray::from_iter(parts.weights).into_array(),
            block_offsets: PrimitiveArray::from_iter(parts.block_offsets).into_array(),
            validity: validity_to_child(&validity, unsliced_len),
        }
        .into_slots();
        Array::try_from_parts(
            ArrayParts::new(RangeEntropy, dtype, unsliced_len, data).with_slots(slots),
        )
    }
}

impl RangeEntropyData {
    fn validate(
        &self,
        dtype: &DType,
        len: usize,
        slots: RangeEntropySlotsView<'_>,
        validity: &Validity,
    ) -> VortexResult<()> {
        vortex_ensure!(
            matches!(dtype, DType::Primitive(..)),
            "RangeEntropyArray requires a primitive dtype"
        );
        vortex_ensure!(
            self.block_len > 0,
            "range entropy block length must be positive"
        );
        vortex_ensure!(
            self.scale_bits <= MAX_SCALE_BITS,
            "tANS table log {} exceeds {MAX_SCALE_BITS}",
            self.scale_bits,
        );
        vortex_ensure!(
            self.slice_start <= self.slice_stop && self.slice_stop <= self.unsliced_len,
            "range entropy slice exceeds its source length"
        );
        vortex_ensure!(
            len == self.len(),
            "range entropy length does not match its slice"
        );
        vortex_ensure!(
            slots.bin_lowers.dtype() == &primitive_dtype(PType::U64),
            "range entropy bin lowers require non-nullable u64 values"
        );
        vortex_ensure!(
            slots.offset_widths.dtype() == &primitive_dtype(PType::U8),
            "range entropy offset widths require non-nullable u8 values"
        );
        vortex_ensure!(
            slots.weights.dtype() == &primitive_dtype(PType::U16),
            "range entropy weights require non-nullable u16 values"
        );
        vortex_ensure!(
            slots.block_offsets.dtype() == &primitive_dtype(PType::U32),
            "range entropy block offsets require non-nullable u32 values"
        );
        let bin_count = slots.bin_lowers.len();
        vortex_ensure!(
            bin_count <= MAX_BINS,
            "range entropy bin count exceeds {MAX_BINS}"
        );
        vortex_ensure!(
            slots.offset_widths.len() == bin_count && slots.weights.len() == bin_count,
            "range entropy bin children have different lengths"
        );
        vortex_ensure!(
            slots.block_offsets.len() == self.unsliced_len.div_ceil(self.block_len) + 1,
            "range entropy block offset count is invalid"
        );
        if let Some(validity_len) = validity.maybe_len() {
            vortex_ensure!(
                validity_len == self.unsliced_len,
                "range entropy validity length is invalid"
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

fn codec_from_array(
    array: ArrayView<'_, RangeEntropy>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<RangeEntropyCodec> {
    let bin_lowers = array.bin_lowers().clone().execute::<PrimitiveArray>(ctx)?;
    let offset_widths = array
        .offset_widths()
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    let weights = array.weights().clone().execute::<PrimitiveArray>(ctx)?;
    let block_offsets = array
        .block_offsets()
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    let payload = array.payload.clone().try_to_host_sync()?;
    RangeEntropyCodec::try_from_parts(RangeEntropyParts {
        len: array.unsliced_len,
        block_len: array.block_len,
        scale_bits: array.scale_bits,
        bin_lowers: bin_lowers.as_slice::<u64>().to_vec(),
        offset_widths: offset_widths.as_slice::<u8>().to_vec(),
        weights: weights.as_slice::<u16>().to_vec(),
        block_offsets: block_offsets.as_slice::<u32>().to_vec(),
        payload,
    })
}

fn decompress_array(
    array: ArrayView<'_, RangeEntropy>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let latents =
        codec_from_array(array, ctx)?.decode_range(array.slice_start..array.slice_stop)?;
    primitive_from_latents(latents, array.dtype().as_ptype(), array.validity()?)
}

#[derive(Clone, Copy)]
struct RangeEntropyMetadata {
    scale_bits: u8,
    bin_count: u8,
    block_len: u32,
    unsliced_len: u64,
    slice_start: u64,
}

impl RangeEntropyMetadata {
    fn encode(self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(METADATA_LEN);
        bytes.push(METADATA_VERSION);
        bytes.push(self.scale_bits);
        bytes.push(self.bin_count);
        bytes.extend_from_slice(&self.block_len.to_le_bytes());
        bytes.extend_from_slice(&self.unsliced_len.to_le_bytes());
        bytes.extend_from_slice(&self.slice_start.to_le_bytes());
        bytes
    }

    fn decode(bytes: &[u8]) -> VortexResult<Self> {
        vortex_ensure!(
            bytes.len() == METADATA_LEN,
            "RangeEntropyArray metadata requires {METADATA_LEN} bytes"
        );
        vortex_ensure!(
            bytes[0] == METADATA_VERSION,
            "unsupported RangeEntropyArray metadata version {}",
            bytes[0]
        );
        Ok(Self {
            scale_bits: bytes[1],
            bin_count: bytes[2],
            block_len: u32::from_le_bytes([bytes[3], bytes[4], bytes[5], bytes[6]]),
            unsliced_len: u64::from_le_bytes([
                bytes[7], bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14],
            ]),
            slice_start: u64::from_le_bytes([
                bytes[15], bytes[16], bytes[17], bytes[18], bytes[19], bytes[20], bytes[21],
                bytes[22],
            ]),
        })
    }
}

fn primitive_dtype(ptype: PType) -> DType {
    DType::Primitive(ptype, NonNullable)
}

fn ordered_latents(array: ArrayView<'_, Primitive>) -> Vec<u64> {
    match array.ptype() {
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
            .map(|&value| u64::from(value.cast_unsigned() ^ (1_u8 << 7)))
            .collect(),
        PType::I16 => array
            .as_slice::<i16>()
            .iter()
            .map(|&value| u64::from(value.cast_unsigned() ^ (1_u16 << 15)))
            .collect(),
        PType::I32 => array
            .as_slice::<i32>()
            .iter()
            .map(|&value| u64::from(value.cast_unsigned() ^ (1_u32 << 31)))
            .collect(),
        PType::I64 => array
            .as_slice::<i64>()
            .iter()
            .map(|&value| value.cast_unsigned() ^ (1_u64 << 63))
            .collect(),
        PType::F16 => array
            .as_slice::<f16>()
            .iter()
            .map(|&value| ordered_float_bits(u64::from(value.to_bits()), 16))
            .collect(),
        PType::F32 => array
            .as_slice::<f32>()
            .iter()
            .map(|&value| ordered_float_bits(u64::from(value.to_bits()), 32))
            .collect(),
        PType::F64 => array
            .as_slice::<f64>()
            .iter()
            .map(|&value| ordered_float_bits(value.to_bits(), 64))
            .collect(),
    }
}

fn ordered_float_bits(bits: u64, width: u8) -> u64 {
    let sign = 1_u64 << (width - 1);
    if bits & sign == 0 {
        bits ^ sign
    } else {
        !bits & width_mask(width)
    }
}

fn float_bits_from_ordered(value: u64, width: u8) -> u64 {
    let sign = 1_u64 << (width - 1);
    if value & sign == 0 {
        !value & width_mask(width)
    } else {
        value ^ sign
    }
}

fn width_mask(width: u8) -> u64 {
    if width == 64 {
        u64::MAX
    } else {
        (1_u64 << width) - 1
    }
}

fn primitive_from_latents(
    latents: Vec<u64>,
    ptype: PType,
    validity: Validity,
) -> VortexResult<PrimitiveArray> {
    Ok(match ptype {
        PType::U8 => PrimitiveArray::new(
            Buffer::from(convert_latents(&latents, u8::try_from)?),
            validity,
        ),
        PType::U16 => PrimitiveArray::new(
            Buffer::from(convert_latents(&latents, u16::try_from)?),
            validity,
        ),
        PType::U32 => PrimitiveArray::new(
            Buffer::from(convert_latents(&latents, u32::try_from)?),
            validity,
        ),
        PType::U64 => PrimitiveArray::new(Buffer::from(latents), validity),
        PType::I8 => PrimitiveArray::new(
            Buffer::from(
                latents
                    .into_iter()
                    .map(|value| Ok((u8::try_from(value)? ^ (1_u8 << 7)).cast_signed()))
                    .collect::<VortexResult<Vec<_>>>()?,
            ),
            validity,
        ),
        PType::I16 => PrimitiveArray::new(
            Buffer::from(
                latents
                    .into_iter()
                    .map(|value| Ok((u16::try_from(value)? ^ (1_u16 << 15)).cast_signed()))
                    .collect::<VortexResult<Vec<_>>>()?,
            ),
            validity,
        ),
        PType::I32 => PrimitiveArray::new(
            Buffer::from(
                latents
                    .into_iter()
                    .map(|value| Ok((u32::try_from(value)? ^ (1_u32 << 31)).cast_signed()))
                    .collect::<VortexResult<Vec<_>>>()?,
            ),
            validity,
        ),
        PType::I64 => PrimitiveArray::new(
            Buffer::from(latents)
                .map_each_in_place(|value| (value ^ (1_u64 << 63)).cast_signed())
                .freeze(),
            validity,
        ),
        PType::F16 => PrimitiveArray::new(
            Buffer::from(
                latents
                    .into_iter()
                    .map(|value| {
                        Ok(f16::from_bits(u16::try_from(float_bits_from_ordered(
                            value, 16,
                        ))?))
                    })
                    .collect::<VortexResult<Vec<_>>>()?,
            ),
            validity,
        ),
        PType::F32 => PrimitiveArray::new(
            Buffer::from(
                latents
                    .into_iter()
                    .map(|value| {
                        Ok(f32::from_bits(u32::try_from(float_bits_from_ordered(
                            value, 32,
                        ))?))
                    })
                    .collect::<VortexResult<Vec<_>>>()?,
            ),
            validity,
        ),
        PType::F64 => PrimitiveArray::new(
            Buffer::from(latents)
                .map_each_in_place(|value| f64::from_bits(float_bits_from_ordered(value, 64)))
                .freeze(),
            validity,
        ),
    })
}

fn convert_latents<T>(
    latents: &[u64],
    convert: impl Fn(u64) -> Result<T, std::num::TryFromIntError>,
) -> VortexResult<Vec<T>> {
    latents
        .iter()
        .copied()
        .map(convert)
        .collect::<Result<Vec<_>, _>>()
        .map_err(Into::into)
}

fn scalar_from_latent(latent: u64, ptype: PType, nullability: Nullability) -> VortexResult<Scalar> {
    Ok(match ptype {
        PType::U8 => Scalar::primitive(u8::try_from(latent)?, nullability),
        PType::U16 => Scalar::primitive(u16::try_from(latent)?, nullability),
        PType::U32 => Scalar::primitive(u32::try_from(latent)?, nullability),
        PType::U64 => Scalar::primitive(latent, nullability),
        PType::I8 => Scalar::primitive(
            (u8::try_from(latent)? ^ (1_u8 << 7)).cast_signed(),
            nullability,
        ),
        PType::I16 => Scalar::primitive(
            (u16::try_from(latent)? ^ (1_u16 << 15)).cast_signed(),
            nullability,
        ),
        PType::I32 => Scalar::primitive(
            (u32::try_from(latent)? ^ (1_u32 << 31)).cast_signed(),
            nullability,
        ),
        PType::I64 => Scalar::primitive((latent ^ (1_u64 << 63)).cast_signed(), nullability),
        PType::F16 => Scalar::primitive(
            f16::from_bits(u16::try_from(float_bits_from_ordered(latent, 16))?),
            nullability,
        ),
        PType::F32 => Scalar::primitive(
            f32::from_bits(u32::try_from(float_bits_from_ordered(latent, 32))?),
            nullability,
        ),
        PType::F64 => Scalar::primitive(
            f64::from_bits(float_bits_from_ordered(latent, 64)),
            nullability,
        ),
    })
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::ArrayContext;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::assert_arrays_eq;
    use vortex_array::assert_nth_scalar;
    use vortex_array::serde::SerializeOptions;
    use vortex_array::serde::SerializedArray;
    use vortex_buffer::ByteBufferMut;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;
    use vortex_session::registry::ReadContext;

    use super::*;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    fn assert_roundtrip(array: PrimitiveArray) -> VortexResult<()> {
        let encoded = RangeEntropy::from_primitive(array.as_view(), 3)?;
        let mut ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(encoded, array, &mut ctx);

        let slice_stop = array.len() - 1;
        let slice = encoded.slice(1..slice_stop)?;
        assert!(slice.is::<RangeEntropy>());
        assert_arrays_eq!(slice, array.into_array().slice(1..slice_stop)?, &mut ctx);
        Ok(())
    }

    #[test]
    fn primitive_types_roundtrip() -> VortexResult<()> {
        for array in [
            PrimitiveArray::from_iter([0_u8, 1, u8::MAX, 7]),
            PrimitiveArray::from_iter([0_u16, 1, u16::MAX, 7]),
            PrimitiveArray::from_iter([0_u32, 1, u32::MAX, 7]),
            PrimitiveArray::from_iter([0_u64, 1, u64::MAX, 7]),
            PrimitiveArray::from_iter([i8::MIN, -1, 0, i8::MAX]),
            PrimitiveArray::from_iter([i16::MIN, -1, 0, i16::MAX]),
            PrimitiveArray::from_iter([i32::MIN, -1, 0, i32::MAX]),
            PrimitiveArray::from_iter([i64::MIN, -1, 0, i64::MAX]),
            PrimitiveArray::from_iter([
                f16::NEG_INFINITY,
                f16::from_f32(-0.0),
                f16::from_f32(1.5),
                f16::INFINITY,
            ]),
            PrimitiveArray::from_iter([f32::NEG_INFINITY, -0.0, 1.5, f32::INFINITY]),
            PrimitiveArray::from_iter([f64::NEG_INFINITY, -0.0, 1.5, f64::INFINITY]),
        ] {
            assert_roundtrip(array)?;
        }
        Ok(())
    }

    #[test]
    fn nullable_slice_and_scalar_access() -> VortexResult<()> {
        let array = PrimitiveArray::from_option_iter([
            Some(-10.5_f64),
            None,
            Some(-0.0),
            Some(42.25),
            None,
            Some(1_000.0),
        ]);
        let encoded = RangeEntropy::from_primitive(array.as_view(), 2)?;
        let mut ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(encoded, array, &mut ctx);
        assert_nth_scalar!(encoded, 3, 42.25_f64, &mut ctx);
        assert!(encoded.execute_scalar(1, &mut ctx)?.is_null());

        let sliced = encoded.slice(1..5)?;
        assert!(sliced.is::<RangeEntropy>());
        assert_arrays_eq!(sliced, array.into_array().slice(1..5)?, &mut ctx);
        Ok(())
    }

    #[test]
    fn serialization_roundtrip() -> VortexResult<()> {
        let original = PrimitiveArray::from_option_iter([
            Some(i64::MIN),
            None,
            Some(-1),
            Some(0),
            Some(i64::MAX),
        ]);
        let encoded = RangeEntropy::from_primitive(original.as_view(), 2)?;
        let sliced = encoded.slice(1..5)?;
        let dtype = sliced.dtype().clone();
        let len = sliced.len();
        let array_context = ArrayContext::empty();
        let serialized =
            sliced.serialize(&array_context, &SESSION, &SerializeOptions::default())?;
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
        assert!(decoded.is::<RangeEntropy>());
        assert_arrays_eq!(
            decoded,
            original.into_array().slice(1..5)?,
            &mut SESSION.create_execution_ctx()
        );
        Ok(())
    }
}
