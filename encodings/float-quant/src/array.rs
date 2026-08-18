// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;

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
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_array::serde::ArrayChildren;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityChild;
use vortex_array::vtable::ValidityVTableFromChild;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::rules::RULES;

const METADATA_VERSION: u8 = 1;
const METADATA_LEN: usize = 2;

/// A lossless float split with quantized high bits and low-bit adjustments.
pub type FloatQuantArray = Array<FloatQuant>;

#[array_slots(FloatQuant)]
pub struct FloatQuantSlots {
    /// Ordered float bits after the low `k` bits are removed.
    #[slot(0)]
    pub primary: ArrayRef,
    /// Sign-normalized low `k` bits.
    #[slot(1)]
    pub secondary: ArrayRef,
}

#[derive(Clone, Debug)]
pub struct FloatQuantData {
    pub(crate) k: u8,
}

impl Display for FloatQuantData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "k: {}", self.k)
    }
}

impl ArrayHash for FloatQuantData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        self.k.hash(state);
    }
}

impl ArrayEq for FloatQuantData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.k == other.k
    }
}

#[derive(Clone, Debug)]
pub struct FloatQuant;

impl VTable for FloatQuant {
    type TypedArrayData = FloatQuantData;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.float_quant");
        *ID
    }

    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let ptype = PType::try_from(dtype)?;
        let latent_ptype = latent_ptype(ptype)?;
        let precision_bits = precision_bits(ptype)?;
        vortex_ensure!(
            data.k > 0 && data.k <= precision_bits,
            "FloatQuant k {} exceeds {ptype} precision {precision_bits}",
            data.k
        );

        let slots = FloatQuantSlotsView::from_slots(slots);
        let expected_primary = DType::Primitive(latent_ptype, dtype.nullability());
        let expected_secondary = DType::Primitive(latent_ptype, NonNullable);
        vortex_ensure!(
            slots.primary.dtype() == &expected_primary,
            "expected primary dtype {expected_primary}, got {}",
            slots.primary.dtype()
        );
        vortex_ensure!(
            slots.secondary.dtype() == &expected_secondary,
            "expected secondary dtype {expected_secondary}, got {}",
            slots.secondary.dtype()
        );
        vortex_ensure!(
            slots.primary.len() == len && slots.secondary.len() == len,
            "FloatQuant child length differs from {len}"
        );
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("FloatQuantArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        vortex_panic!("FloatQuantArray buffer_name index {idx} out of bounds")
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
        Ok(Some(vec![METADATA_VERSION, array.data().k]))
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
        vortex_ensure!(
            metadata.len() == METADATA_LEN,
            "FloatQuant metadata requires {METADATA_LEN} bytes"
        );
        vortex_ensure!(
            metadata[0] == METADATA_VERSION,
            "unsupported FloatQuant metadata version {}",
            metadata[0]
        );
        vortex_ensure!(children.len() == 2, "FloatQuant requires two children");

        let ptype = PType::try_from(dtype)?;
        let latent_ptype = latent_ptype(ptype)?;
        let primary_dtype = DType::Primitive(latent_ptype, dtype.nullability());
        let secondary_dtype = DType::Primitive(latent_ptype, NonNullable);
        let primary = children.get(0, &primary_dtype, len)?;
        let secondary = children.get(1, &secondary_dtype, len)?;
        let slots = FloatQuantSlots { primary, secondary }.into_slots();
        Ok(ArrayParts::new(
            self.clone(),
            dtype.clone(),
            len,
            FloatQuantData { k: metadata[1] },
        )
        .with_slots(slots))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        FloatQuantSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(
            decode(array.as_view(), ctx)?.into_array(),
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

impl OperationsVTable<FloatQuant> for FloatQuant {
    fn scalar_at(
        array: ArrayView<'_, FloatQuant>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let primary = array.primary().execute_scalar(index, ctx)?;
        if primary.is_null() {
            return Ok(Scalar::null(array.dtype().clone()));
        }
        let secondary = array.secondary().execute_scalar(index, ctx)?;
        let k = array.data().k;
        Ok(match PType::try_from(array.dtype())? {
            PType::F32 => Scalar::primitive(
                join_f32(
                    primary
                        .as_primitive()
                        .typed_value::<u32>()
                        .vortex_expect("validated primary scalar"),
                    secondary
                        .as_primitive()
                        .typed_value::<u32>()
                        .vortex_expect("validated secondary scalar"),
                    k,
                ),
                array.dtype().nullability(),
            ),
            PType::F64 => Scalar::primitive(
                join_f64(
                    primary
                        .as_primitive()
                        .typed_value::<u64>()
                        .vortex_expect("validated primary scalar"),
                    secondary
                        .as_primitive()
                        .typed_value::<u64>()
                        .vortex_expect("validated secondary scalar"),
                    k,
                ),
                array.dtype().nullability(),
            ),
            ptype => vortex_panic!("unsupported FloatQuant ptype {ptype}"),
        })
    }
}

impl ValidityChild<FloatQuant> for FloatQuant {
    fn validity_child(array: ArrayView<'_, FloatQuant>) -> ArrayRef {
        array.primary().clone()
    }
}

pub trait FloatQuantArrayExt: TypedArrayRef<FloatQuant> + FloatQuantArraySlotsExt {
    /// Return the number of split low bits.
    fn k(&self) -> u8 {
        self.deref().k
    }
}

impl<T: TypedArrayRef<FloatQuant>> FloatQuantArrayExt for T {}

impl FloatQuant {
    /// Construct a float quantization array from two latent children.
    pub fn try_new(
        primary: ArrayRef,
        secondary: ArrayRef,
        float_ptype: PType,
        k: u8,
    ) -> VortexResult<FloatQuantArray> {
        let dtype = DType::Primitive(float_ptype, primary.dtype().nullability());
        let len = primary.len();
        let slots = FloatQuantSlots { primary, secondary }.into_slots();
        Array::try_from_parts(
            ArrayParts::new(FloatQuant, dtype, len, FloatQuantData { k }).with_slots(slots),
        )
    }

    /// Split a canonical float array into two unsigned latent children.
    pub fn from_primitive(array: ArrayView<'_, Primitive>, k: u8) -> VortexResult<FloatQuantArray> {
        let validity = array.validity()?;
        match array.ptype() {
            PType::F32 => {
                let (primary, secondary) = split_f32(array.as_slice::<f32>(), k)?;
                Self::try_new(
                    PrimitiveArray::new(Buffer::from(primary), validity).into_array(),
                    PrimitiveArray::new(Buffer::from(secondary), NonNullable.into()).into_array(),
                    PType::F32,
                    k,
                )
            }
            PType::F64 => {
                let (primary, secondary) = split_f64(array.as_slice::<f64>(), k)?;
                Self::try_new(
                    PrimitiveArray::new(Buffer::from(primary), validity).into_array(),
                    PrimitiveArray::new(Buffer::from(secondary), NonNullable.into()).into_array(),
                    PType::F64,
                    k,
                )
            }
            ptype => vortex_bail!("FloatQuant requires f32 or f64, got {ptype}"),
        }
    }

    /// Split floats whose lowest `k` fraction bits are zero.
    pub fn from_primitive_constant_secondary(
        array: ArrayView<'_, Primitive>,
        k: u8,
    ) -> VortexResult<FloatQuantArray> {
        let validity = array.validity()?;
        let len = array.len();
        match array.ptype() {
            PType::F32 => {
                let primary = split_primary_f32(array.as_slice::<f32>(), k)?;
                Self::try_new(
                    PrimitiveArray::new(Buffer::from(primary), validity).into_array(),
                    ConstantArray::new(Scalar::from(0u32), len).into_array(),
                    PType::F32,
                    k,
                )
            }
            PType::F64 => {
                let primary = split_primary_f64(array.as_slice::<f64>(), k)?;
                Self::try_new(
                    PrimitiveArray::new(Buffer::from(primary), validity).into_array(),
                    ConstantArray::new(Scalar::from(0u64), len).into_array(),
                    PType::F64,
                    k,
                )
            }
            ptype => vortex_bail!("FloatQuant requires f32 or f64, got {ptype}"),
        }
    }

    /// Split a constant-secondary float array into frame-of-reference primary values.
    pub fn primary_for_primitive(
        array: ArrayView<'_, Primitive>,
        k: u8,
        primary_min: u64,
    ) -> VortexResult<PrimitiveArray> {
        let validity = array.validity()?;
        match array.ptype() {
            PType::F32 => {
                let primary_min = u32::try_from(primary_min)?;
                let primary = split_primary_for_f32(array.as_slice::<f32>(), k, primary_min)?;
                Ok(PrimitiveArray::new(Buffer::from(primary), validity))
            }
            PType::F64 => {
                let primary = split_primary_for_f64(array.as_slice::<f64>(), k, primary_min)?;
                Ok(PrimitiveArray::new(Buffer::from(primary), validity))
            }
            ptype => vortex_bail!("FloatQuant requires f32 or f64, got {ptype}"),
        }
    }
}

/// Compression facts derived during FloatQuant split selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FloatQuantAnalysis {
    /// Selected low-bit width.
    pub k: u8,
    /// Bit width of the frame-of-reference primary values.
    pub primary_bit_width: u8,
    /// Minimum primary value before frame-of-reference subtraction.
    pub primary_min: u64,
    /// True when every secondary value is zero.
    pub secondary_is_constant: bool,
}

/// Estimate a useful low-bit split width for a canonical float array.
pub fn estimate_k(array: ArrayView<'_, Primitive>) -> Option<u8> {
    analyze_float_quant(array).map(|analysis| analysis.k)
}

/// Analyze a canonical float array for a FloatQuant split.
pub fn analyze_float_quant(array: ArrayView<'_, Primitive>) -> Option<FloatQuantAnalysis> {
    match array.ptype() {
        PType::F32 => analyze_f32(array.as_slice::<f32>()),
        PType::F64 => analyze_f64(array.as_slice::<f64>()),
        _ => None,
    }
}

fn analyze_histogram(
    mut histogram: Vec<usize>,
    precision_bits: u8,
    len: usize,
    primary_min: u64,
    primary_max: u64,
) -> Option<FloatQuantAnalysis> {
    if len == 0 {
        return None;
    }
    let mut cumulative = 0usize;
    for count in histogram.iter_mut().rev() {
        cumulative += *count;
        *count = cumulative;
    }

    let mut best_k = 0u8;
    let mut best_savings = 0.0;
    for k in 1..=precision_bits {
        let frequency = histogram[usize::from(k)] as f64 / len as f64;
        if frequency == 0.0 {
            continue;
        }
        let category_count = ((1_u64 << k) - 1) as f64;
        let entropy = category_entropy(frequency)
            + category_count * category_entropy((1.0 - frequency) / category_count);
        let savings = f64::from(k) - entropy;
        if savings > best_savings {
            best_k = k;
            best_savings = savings;
        } else {
            break;
        }
    }
    if best_savings <= 1.5 {
        return None;
    }
    let secondary_is_constant = histogram[usize::from(best_k)] == len;
    let primary_min = primary_min >> best_k;
    let primary_max = primary_max >> best_k;
    let primary_bit_width =
        u8::try_from(u64::BITS - (primary_max - primary_min).leading_zeros()).unwrap_or(u8::MAX);
    Some(FloatQuantAnalysis {
        k: best_k,
        primary_bit_width,
        primary_min,
        secondary_is_constant,
    })
}

fn analyze_f32(values: &[f32]) -> Option<FloatQuantAnalysis> {
    let mut minimum = u32::MAX;
    let mut maximum = u32::MIN;
    let mut histogram = vec![0usize; 24];
    for value in values {
        let bits = value.to_bits();
        let ordered = ordered_u32(bits);
        minimum = minimum.min(ordered);
        maximum = maximum.max(ordered);
        let zeros = bits.trailing_zeros().min(23);
        histogram[zeros as usize] += 1;
    }
    analyze_histogram(
        histogram,
        23,
        values.len(),
        u64::from(minimum),
        u64::from(maximum),
    )
}

fn analyze_f64(values: &[f64]) -> Option<FloatQuantAnalysis> {
    let mut minimum = u64::MAX;
    let mut maximum = u64::MIN;
    let mut histogram = vec![0usize; 53];
    for value in values {
        let bits = value.to_bits();
        let ordered = ordered_u64(bits);
        minimum = minimum.min(ordered);
        maximum = maximum.max(ordered);
        let zeros = bits.trailing_zeros().min(52);
        histogram[zeros as usize] += 1;
    }
    analyze_histogram(histogram, 52, values.len(), minimum, maximum)
}

fn category_entropy(probability: f64) -> f64 {
    if probability == 0.0 || probability == 1.0 {
        0.0
    } else {
        -probability * probability.log2()
    }
}

fn latent_ptype(ptype: PType) -> VortexResult<PType> {
    match ptype {
        PType::F32 => Ok(PType::U32),
        PType::F64 => Ok(PType::U64),
        _ => vortex_bail!("FloatQuant requires f32 or f64, got {ptype}"),
    }
}

fn precision_bits(ptype: PType) -> VortexResult<u8> {
    match ptype {
        PType::F32 => Ok(23),
        PType::F64 => Ok(52),
        _ => vortex_bail!("FloatQuant requires f32 or f64, got {ptype}"),
    }
}

fn ordered_u32(bits: u32) -> u32 {
    if bits & (1_u32 << 31) == 0 {
        bits ^ (1_u32 << 31)
    } else {
        !bits
    }
}

fn ordered_u64(bits: u64) -> u64 {
    if bits & (1_u64 << 63) == 0 {
        bits ^ (1_u64 << 63)
    } else {
        !bits
    }
}

fn split_f32(values: &[f32], k: u8) -> VortexResult<(Vec<u32>, Vec<u32>)> {
    vortex_ensure!(k > 0 && k <= 23, "FloatQuant f32 k must be in 1..=23");
    let low_mask = (1_u32 << k) - 1;
    let mut primary = Vec::with_capacity(values.len());
    let mut secondary = Vec::with_capacity(values.len());
    for &value in values {
        let bits = value.to_bits();
        let ordered = ordered_u32(bits);
        primary.push(ordered >> k);
        let low = ordered & low_mask;
        secondary.push(if bits & (1_u32 << 31) == 0 {
            low
        } else {
            low_mask - low
        });
    }
    Ok((primary, secondary))
}

fn split_f64(values: &[f64], k: u8) -> VortexResult<(Vec<u64>, Vec<u64>)> {
    vortex_ensure!(k > 0 && k <= 52, "FloatQuant f64 k must be in 1..=52");
    let low_mask = (1_u64 << k) - 1;
    let mut primary = Vec::with_capacity(values.len());
    let mut secondary = Vec::with_capacity(values.len());
    for &value in values {
        let bits = value.to_bits();
        let ordered = ordered_u64(bits);
        primary.push(ordered >> k);
        let low = ordered & low_mask;
        secondary.push(if bits & (1_u64 << 63) == 0 {
            low
        } else {
            low_mask - low
        });
    }
    Ok((primary, secondary))
}

fn split_primary_f32(values: &[f32], k: u8) -> VortexResult<Vec<u32>> {
    vortex_ensure!(k > 0 && k <= 23, "FloatQuant f32 k must be in 1..=23");
    let low_mask = (1_u32 << k) - 1;
    values
        .iter()
        .map(|value| {
            let bits = value.to_bits();
            vortex_ensure!(
                bits & low_mask == 0,
                "FloatQuant constant secondary requires zero low bits"
            );
            Ok(ordered_u32(bits) >> k)
        })
        .collect()
}

fn split_primary_f64(values: &[f64], k: u8) -> VortexResult<Vec<u64>> {
    vortex_ensure!(k > 0 && k <= 52, "FloatQuant f64 k must be in 1..=52");
    let low_mask = (1_u64 << k) - 1;
    values
        .iter()
        .map(|value| {
            let bits = value.to_bits();
            vortex_ensure!(
                bits & low_mask == 0,
                "FloatQuant constant secondary requires zero low bits"
            );
            Ok(ordered_u64(bits) >> k)
        })
        .collect()
}

fn split_primary_for_f32(values: &[f32], k: u8, primary_min: u32) -> VortexResult<Vec<u32>> {
    vortex_ensure!(k > 0 && k <= 23, "FloatQuant f32 k must be in 1..=23");
    let low_mask = (1_u32 << k) - 1;
    values
        .iter()
        .map(|value| {
            let bits = value.to_bits();
            vortex_ensure!(
                bits & low_mask == 0,
                "FloatQuant constant secondary requires zero low bits"
            );
            Ok((ordered_u32(bits) >> k) - primary_min)
        })
        .collect()
}

fn split_primary_for_f64(values: &[f64], k: u8, primary_min: u64) -> VortexResult<Vec<u64>> {
    vortex_ensure!(k > 0 && k <= 52, "FloatQuant f64 k must be in 1..=52");
    let low_mask = (1_u64 << k) - 1;
    values
        .iter()
        .map(|value| {
            let bits = value.to_bits();
            vortex_ensure!(
                bits & low_mask == 0,
                "FloatQuant constant secondary requires zero low bits"
            );
            Ok((ordered_u64(bits) >> k) - primary_min)
        })
        .collect()
}

fn join_f32(primary: u32, secondary: u32, k: u8) -> f32 {
    let low_mask = (1_u32 << k) - 1;
    let sign_cutoff = (1_u32 << 31) >> k;
    let low = if primary >= sign_cutoff {
        secondary
    } else {
        low_mask.wrapping_sub(secondary)
    };
    let ordered = (primary << k).wrapping_add(low);
    let bits = if ordered & (1_u32 << 31) == 0 {
        !ordered
    } else {
        ordered ^ (1_u32 << 31)
    };
    f32::from_bits(bits)
}

fn join_f64(primary: u64, secondary: u64, k: u8) -> f64 {
    let low_mask = (1_u64 << k) - 1;
    let sign_cutoff = (1_u64 << 63) >> k;
    let low = if primary >= sign_cutoff {
        secondary
    } else {
        low_mask.wrapping_sub(secondary)
    };
    let ordered = (primary << k).wrapping_add(low);
    let bits = if ordered & (1_u64 << 63) == 0 {
        !ordered
    } else {
        ordered ^ (1_u64 << 63)
    };
    f64::from_bits(bits)
}

fn decode(
    array: ArrayView<'_, FloatQuant>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let primary = array.primary().clone().execute::<PrimitiveArray>(ctx)?;
    let secondary = array.secondary().clone().execute::<PrimitiveArray>(ctx)?;
    let validity = primary.validity()?;
    let k = array.data().k;
    Ok(match PType::try_from(array.dtype())? {
        PType::F32 => {
            let secondary_values = secondary.as_slice::<u32>();
            let mut index = 0;
            let values = primary
                .into_buffer::<u32>()
                .map_each_in_place(|primary| {
                    let value = join_f32(primary, secondary_values[index], k);
                    index += 1;
                    value
                })
                .freeze();
            PrimitiveArray::new(values, validity)
        }
        PType::F64 => {
            let secondary_values = secondary.as_slice::<u64>();
            let mut index = 0;
            let values = primary
                .into_buffer::<u64>()
                .map_each_in_place(|primary| {
                    let value = join_f64(primary, secondary_values[index], k);
                    index += 1;
                    value
                })
                .freeze();
            PrimitiveArray::new(values, validity)
        }
        ptype => vortex_panic!("unsupported FloatQuant ptype {ptype}"),
    })
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::ArrayContext;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::assert_arrays_eq;
    use vortex_array::assert_nth_scalar;
    use vortex_array::serde::SerializeOptions;
    use vortex_array::serde::SerializedArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::ByteBufferMut;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;
    use vortex_session::registry::ReadContext;

    use super::*;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = array_session();
        crate::initialize(&session);
        session
    });

    #[test]
    fn float_bit_patterns_roundtrip() -> VortexResult<()> {
        let values = [
            f64::NEG_INFINITY,
            -1.5,
            -0.0,
            0.0,
            1.5,
            f64::INFINITY,
            f64::from_bits(0x7ff8_0000_0000_1234),
            f64::from_bits(0xfff8_0000_0000_5678),
        ];
        let array = PrimitiveArray::from_iter(values);
        let encoded = FloatQuant::from_primitive(array.as_view(), 29)?;
        let decoded = encoded
            .into_array()
            .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())?;
        assert_eq!(
            decoded
                .as_slice::<f64>()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    fn nullable_slice_and_scalar_access() -> VortexResult<()> {
        let array = PrimitiveArray::new(
            Buffer::from(vec![1.25_f32, 0.0, -0.0, 42.5, -10.0]),
            Validity::from_iter([true, false, true, true, false]),
        );
        let encoded = FloatQuant::from_primitive(array.as_view(), 8)?;
        let mut ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(encoded, array, &mut ctx);
        assert_nth_scalar!(encoded, 3, 42.5_f32, &mut ctx);
        assert!(encoded.execute_scalar(1, &mut ctx)?.is_null());

        let sliced = encoded.into_array().slice(1..4)?;
        assert!(sliced.is::<FloatQuant>());
        assert_arrays_eq!(sliced, array.into_array().slice(1..4)?, &mut ctx);
        Ok(())
    }

    #[test]
    fn serialization_roundtrip() -> VortexResult<()> {
        let original = PrimitiveArray::from_option_iter([
            Some(f64::NEG_INFINITY),
            None,
            Some(-0.0),
            Some(42.25),
            Some(f64::from_bits(0x7ff8_0000_0000_1234)),
        ]);
        let encoded = FloatQuant::from_primitive(original.as_view(), 29)?;
        let sliced = encoded.into_array().slice(1..5)?;
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
        assert!(decoded.is::<FloatQuant>());
        assert_arrays_eq!(
            decoded,
            original.into_array().slice(1..5)?,
            &mut SESSION.create_execution_ctx()
        );
        Ok(())
    }
}
