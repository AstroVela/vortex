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
use vortex_utils::aliases::hash_set::HashSet;

use crate::rules::FLOAT_MULT_RULES;

const METADATA_VERSION: u8 = 1;
const F32_METADATA_LEN: usize = 5;
const F64_METADATA_LEN: usize = 9;
const MIN_ANALYSIS_VALUES: usize = 32;

/// A lossless float split into approximate multiples and ULP adjustments.
pub type FloatMultArray = Array<FloatMult>;

#[array_slots(FloatMult)]
pub struct FloatMultSlots {
    /// Signed integer multiples encoded in float order.
    #[slot(0)]
    pub primary: ArrayRef,
    /// Signed ULP adjustments with zero moved to the unsigned midpoint.
    #[slot(1)]
    pub secondary: Option<ArrayRef>,
}

#[derive(Clone, Debug)]
pub struct FloatMultData {
    base_bits: u64,
}

impl Display for FloatMultData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "base_bits: {}", self.base_bits)
    }
}

impl ArrayHash for FloatMultData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        self.base_bits.hash(state);
    }
}

impl ArrayEq for FloatMultData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.base_bits == other.base_bits
    }
}

#[derive(Clone, Debug)]
pub struct FloatMult;

impl VTable for FloatMult {
    type TypedArrayData = FloatMultData;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.float_mult");
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
        let base = data.base(ptype)?;
        vortex_ensure!(
            base.is_finite() && base.abs() > 0.0,
            "FloatMult base must be finite and nonzero"
        );

        let slots = FloatMultSlotsView::from_slots(slots);
        let expected_primary = DType::Primitive(latent_ptype, dtype.nullability());
        vortex_ensure!(
            slots.primary.dtype() == &expected_primary,
            "expected primary dtype {expected_primary}, got {}",
            slots.primary.dtype()
        );
        vortex_ensure!(
            slots.primary.len() == len,
            "FloatMult child length differs from {len}"
        );
        if let Some(secondary) = slots.secondary {
            let expected_secondary = DType::Primitive(latent_ptype, NonNullable);
            vortex_ensure!(
                secondary.dtype() == &expected_secondary,
                "expected secondary dtype {expected_secondary}, got {}",
                secondary.dtype()
            );
            vortex_ensure!(
                secondary.len() == len,
                "FloatMult child length differs from {len}"
            );
        }
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("FloatMultArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        vortex_panic!("FloatMultArray buffer_name index {idx} out of bounds")
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
        let mut metadata = vec![METADATA_VERSION];
        match PType::try_from(array.dtype())? {
            PType::F32 => metadata.extend_from_slice(
                &u32::try_from(array.data().base_bits)
                    .vortex_expect("validated f32 FloatMult base bits")
                    .to_le_bytes(),
            ),
            PType::F64 => metadata.extend_from_slice(&array.data().base_bits.to_le_bytes()),
            ptype => vortex_bail!("FloatMult requires f32 or f64, got {ptype}"),
        }
        Ok(Some(metadata))
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
        let ptype = PType::try_from(dtype)?;
        let expected_metadata_len = match ptype {
            PType::F32 => F32_METADATA_LEN,
            PType::F64 => F64_METADATA_LEN,
            _ => vortex_bail!("FloatMult requires f32 or f64, got {ptype}"),
        };
        vortex_ensure!(
            metadata.len() == expected_metadata_len,
            "FloatMult metadata requires {expected_metadata_len} bytes"
        );
        vortex_ensure!(
            metadata[0] == METADATA_VERSION,
            "unsupported FloatMult metadata version {}",
            metadata[0]
        );
        vortex_ensure!(
            matches!(children.len(), 1 | 2),
            "FloatMult requires one or two children"
        );

        let base_bits = match ptype {
            PType::F32 => {
                let mut bytes = [0; 4];
                bytes.copy_from_slice(&metadata[1..5]);
                u64::from(u32::from_le_bytes(bytes))
            }
            PType::F64 => {
                let mut bytes = [0; 8];
                bytes.copy_from_slice(&metadata[1..9]);
                u64::from_le_bytes(bytes)
            }
            _ => unreachable!(),
        };
        let latent_ptype = latent_ptype(ptype)?;
        let primary_dtype = DType::Primitive(latent_ptype, dtype.nullability());
        let primary = children.get(0, &primary_dtype, len)?;
        let secondary = if children.len() == 2 {
            let secondary_dtype = DType::Primitive(latent_ptype, NonNullable);
            Some(children.get(1, &secondary_dtype, len)?)
        } else {
            None
        };
        let slots = FloatMultSlots { primary, secondary }.into_slots();
        Ok(ArrayParts::new(
            self.clone(),
            dtype.clone(),
            len,
            FloatMultData { base_bits },
        )
        .with_slots(slots))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        FloatMultSlots::NAMES[idx].to_string()
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
        FLOAT_MULT_RULES.evaluate(array, parent, child_idx)
    }
}

impl OperationsVTable<FloatMult> for FloatMult {
    fn scalar_at(
        array: ArrayView<'_, FloatMult>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let primary = array.primary().execute_scalar(index, ctx)?;
        if primary.is_null() {
            return Ok(Scalar::null(array.dtype().clone()));
        }
        Ok(match PType::try_from(array.dtype())? {
            PType::F32 => {
                let primary = primary
                    .as_primitive()
                    .typed_value::<u32>()
                    .vortex_expect("validated primary scalar");
                let value = if let Some(secondary) = array.secondary() {
                    let secondary = secondary.execute_scalar(index, ctx)?;
                    join_f32(
                        primary,
                        secondary
                            .as_primitive()
                            .typed_value::<u32>()
                            .vortex_expect("validated secondary scalar"),
                        array.data().base_f32()?,
                    )
                } else {
                    int_float_from_u32(primary) * array.data().base_f32()?
                };
                Scalar::primitive(value, array.dtype().nullability())
            }
            PType::F64 => {
                let primary = primary
                    .as_primitive()
                    .typed_value::<u64>()
                    .vortex_expect("validated primary scalar");
                let value = if let Some(secondary) = array.secondary() {
                    let secondary = secondary.execute_scalar(index, ctx)?;
                    join_f64(
                        primary,
                        secondary
                            .as_primitive()
                            .typed_value::<u64>()
                            .vortex_expect("validated secondary scalar"),
                        array.data().base_f64()?,
                    )
                } else {
                    int_float_from_u64(primary) * array.data().base_f64()?
                };
                Scalar::primitive(value, array.dtype().nullability())
            }
            ptype => vortex_panic!("unsupported FloatMult ptype {ptype}"),
        })
    }
}

impl ValidityChild<FloatMult> for FloatMult {
    fn validity_child(array: ArrayView<'_, FloatMult>) -> ArrayRef {
        array.primary().clone()
    }
}

impl FloatMultData {
    fn base(&self, ptype: PType) -> VortexResult<f64> {
        match ptype {
            PType::F32 => Ok(f64::from(self.base_f32()?)),
            PType::F64 => self.base_f64(),
            _ => vortex_bail!("FloatMult requires f32 or f64, got {ptype}"),
        }
    }

    fn base_f32(&self) -> VortexResult<f32> {
        Ok(f32::from_bits(u32::try_from(self.base_bits)?))
    }

    fn base_f64(&self) -> VortexResult<f64> {
        Ok(f64::from_bits(self.base_bits))
    }
}

pub trait FloatMultArrayExt: TypedArrayRef<FloatMult> + FloatMultArraySlotsExt {
    /// Return the multiplier base as an f64 value.
    fn base(&self) -> f64 {
        self.deref()
            .base(
                match PType::try_from(self.primary().dtype())
                    .vortex_expect("validated FloatMult primary dtype")
                {
                    PType::U32 => PType::F32,
                    PType::U64 => PType::F64,
                    _ => vortex_panic!("validated FloatMult primary dtype must be unsigned"),
                },
            )
            .vortex_expect("validated FloatMult base")
    }
}

impl<T: TypedArrayRef<FloatMult>> FloatMultArrayExt for T {}

impl FloatMult {
    /// Construct a FloatMult array from one or two latent children.
    pub fn try_new(
        primary: ArrayRef,
        secondary: Option<ArrayRef>,
        float_ptype: PType,
        base: f64,
    ) -> VortexResult<FloatMultArray> {
        let base_bits = match float_ptype {
            PType::F32 => u64::from(narrow_f32(base).to_bits()),
            PType::F64 => base.to_bits(),
            _ => vortex_bail!("FloatMult requires f32 or f64, got {float_ptype}"),
        };
        let dtype = DType::Primitive(float_ptype, primary.dtype().nullability());
        let len = primary.len();
        let slots = FloatMultSlots { primary, secondary }.into_slots();
        Array::try_from_parts(
            ArrayParts::new(FloatMult, dtype, len, FloatMultData { base_bits }).with_slots(slots),
        )
    }

    /// Split a canonical float array into two unsigned latent children.
    pub fn from_primitive(
        array: ArrayView<'_, Primitive>,
        base: f64,
    ) -> VortexResult<FloatMultArray> {
        let validity = array.validity()?;
        match array.ptype() {
            PType::F32 => {
                let base = narrow_f32(base);
                vortex_ensure!(
                    base.is_finite() && base.abs() > 0.0,
                    "FloatMult base must be finite and nonzero"
                );
                let (primary, secondary): (Vec<_>, Vec<_>) = array
                    .as_slice::<f32>()
                    .iter()
                    .copied()
                    .map(|value| split_f32(value, base))
                    .unzip();
                Self::try_new(
                    PrimitiveArray::new(Buffer::from(primary), validity).into_array(),
                    Some(
                        PrimitiveArray::new(Buffer::from(secondary), NonNullable.into())
                            .into_array(),
                    ),
                    PType::F32,
                    f64::from(base),
                )
            }
            PType::F64 => {
                vortex_ensure!(
                    base.is_finite() && base.abs() > 0.0,
                    "FloatMult base must be finite and nonzero"
                );
                let (primary, secondary): (Vec<_>, Vec<_>) = array
                    .as_slice::<f64>()
                    .iter()
                    .copied()
                    .map(|value| split_f64(value, base))
                    .unzip();
                Self::try_new(
                    PrimitiveArray::new(Buffer::from(primary), validity).into_array(),
                    Some(
                        PrimitiveArray::new(Buffer::from(secondary), NonNullable.into())
                            .into_array(),
                    ),
                    PType::F64,
                    base,
                )
            }
            ptype => vortex_bail!("FloatMult requires f32 or f64, got {ptype}"),
        }
    }

    /// Split floats when every ULP adjustment is zero.
    pub fn from_primitive_constant_secondary(
        array: ArrayView<'_, Primitive>,
        base: f64,
    ) -> VortexResult<Option<FloatMultArray>> {
        let validity = array.validity()?;
        let len = array.len();
        match array.ptype() {
            PType::F32 => {
                let base = narrow_f32(base);
                let mut primary = Vec::with_capacity(len);
                for value in array.as_slice::<f32>() {
                    let (multiple, adjustment) = split_f32(*value, base);
                    if adjustment != 1_u32 << 31 {
                        return Ok(None);
                    }
                    primary.push(multiple);
                }
                Ok(Some(Self::try_new(
                    PrimitiveArray::new(Buffer::from(primary), validity).into_array(),
                    None,
                    PType::F32,
                    f64::from(base),
                )?))
            }
            PType::F64 => {
                let mut primary = Vec::with_capacity(len);
                for value in array.as_slice::<f64>() {
                    let (multiple, adjustment) = split_f64(*value, base);
                    if adjustment != 1_u64 << 63 {
                        return Ok(None);
                    }
                    primary.push(multiple);
                }
                Ok(Some(Self::try_new(
                    PrimitiveArray::new(Buffer::from(primary), validity).into_array(),
                    None,
                    PType::F64,
                    base,
                )?))
            }
            ptype => vortex_bail!("FloatMult requires f32 or f64, got {ptype}"),
        }
    }
}

/// Estimate a common multiplier base from a bounded canonical float sample.
pub fn estimate_float_mult_base(array: ArrayView<'_, Primitive>) -> Option<f64> {
    match array.ptype() {
        PType::F32 => estimate_base(
            &array
                .as_slice::<f32>()
                .iter()
                .copied()
                .map(f64::from)
                .collect::<Vec<_>>(),
            u32::BITS,
        )
        .map(|base| f64::from(narrow_f32(base))),
        PType::F64 => estimate_base(array.as_slice::<f64>(), u64::BITS),
        _ => None,
    }
}

/// Estimate a base that produces zero ULP adjustments for every sampled value.
pub fn estimate_float_mult_constant_base(array: ArrayView<'_, Primitive>) -> Option<f64> {
    let values = match array.ptype() {
        PType::F32 => array
            .as_slice::<f32>()
            .iter()
            .copied()
            .map(f64::from)
            .collect::<Vec<_>>(),
        PType::F64 => array.as_slice::<f64>().to_vec(),
        _ => return None,
    };
    let finite = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if finite.len() < MIN_ANALYSIS_VALUES {
        return None;
    }
    match array.ptype() {
        PType::F32 => constant_candidate_bases_f32(&finite)
            .into_iter()
            .filter_map(|base| {
                constant_score_f32(&finite, narrow_f32(base)).map(|score| (score, base))
            })
            .filter(|(score, _)| *score + 2 < u32::BITS)
            .min_by(|(left_score, left_base), (right_score, right_base)| {
                left_score
                    .cmp(right_score)
                    .then_with(|| right_base.total_cmp(left_base))
            })
            .map(|(_, base)| f64::from(narrow_f32(base))),
        PType::F64 => constant_candidate_bases_f64(&finite)
            .into_iter()
            .filter_map(|base| constant_score_f64(&finite, base).map(|score| (score, base)))
            .filter(|(score, _)| *score + 2 < u64::BITS)
            .min_by(|(left_score, left_base), (right_score, right_base)| {
                left_score
                    .cmp(right_score)
                    .then_with(|| right_base.total_cmp(left_base))
            })
            .map(|(_, base)| base),
        _ => None,
    }
}

fn constant_candidate_bases_f32(values: &[f64]) -> Vec<f64> {
    let mut candidates = Vec::with_capacity(20);
    for exponent in -6..=6 {
        push_candidate(&mut candidates, 10.0_f64.powi(exponent));
    }
    let mut binary_exponent = i32::MAX;
    for value in values {
        let value = narrow_f32(*value);
        if value == 0.0 {
            continue;
        }
        let exponent = (value.abs().to_bits() >> 23) as i32 - 127;
        let trailing_zeros = value.to_bits().trailing_zeros();
        let divisor_exponent = exponent - 23_u32.saturating_sub(trailing_zeros) as i32;
        binary_exponent = binary_exponent.min(divisor_exponent);
    }
    if binary_exponent != i32::MAX {
        push_candidate(&mut candidates, 2.0_f64.powi(binary_exponent));
    }
    push_common_divisor_candidates(&mut candidates, values);
    candidates
}

fn constant_candidate_bases_f64(values: &[f64]) -> Vec<f64> {
    let mut candidates = Vec::with_capacity(32);
    for exponent in -12..=12 {
        push_candidate(&mut candidates, 10.0_f64.powi(exponent));
    }
    let mut binary_exponent = i32::MAX;
    for value in values {
        if *value == 0.0 {
            continue;
        }
        let exponent = (value.abs().to_bits() >> 52) as i32 - 1023;
        let trailing_zeros = value.to_bits().trailing_zeros();
        let divisor_exponent = exponent - 52_u32.saturating_sub(trailing_zeros) as i32;
        binary_exponent = binary_exponent.min(divisor_exponent);
    }
    if binary_exponent != i32::MAX {
        push_candidate(&mut candidates, 2.0_f64.powi(binary_exponent));
    }
    push_common_divisor_candidates(&mut candidates, values);
    candidates
}

fn push_common_divisor_candidates(candidates: &mut Vec<f64>, values: &[f64]) {
    if let Some(base) = approximate_common_divisor(values) {
        push_candidate(candidates, base);
        push_candidate(candidates, snap_base(base));
    }
}

fn estimate_base(values: &[f64], source_bits: u32) -> Option<f64> {
    let finite = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if finite.len() < MIN_ANALYSIS_VALUES {
        return None;
    }

    let candidates = candidate_bases(&finite);

    candidates
        .into_iter()
        .filter_map(|base| score_base(&finite, base).map(|score| (score, base)))
        .filter(|(score, _)| *score + 2.0 < f64::from(source_bits))
        .min_by(|(left, _), (right, _)| left.total_cmp(right))
        .map(|(_, base)| base)
}

fn candidate_bases(values: &[f64]) -> Vec<f64> {
    let mut candidates = Vec::with_capacity(112);
    for exponent in -12..=12 {
        push_candidate(&mut candidates, 10.0_f64.powi(exponent));
    }
    for exponent in -40..=40 {
        push_candidate(&mut candidates, 2.0_f64.powi(exponent));
    }
    if let Some(base) = approximate_common_divisor(values) {
        push_candidate(&mut candidates, base);
        push_candidate(&mut candidates, snap_base(base));
    }
    candidates
}

fn push_candidate(candidates: &mut Vec<f64>, base: f64) {
    if !base.is_finite() || base <= 0.0 {
        return;
    }
    if candidates
        .iter()
        .any(|candidate| ((candidate - base) / base).abs() < 1e-12)
    {
        return;
    }
    candidates.push(base);
}

fn score_base(values: &[f64], base: f64) -> Option<f64> {
    let mut primary_min = u64::MAX;
    let mut primary_max = u64::MIN;
    let mut secondary_min = u64::MAX;
    let mut secondary_max = u64::MIN;
    let mut primary_distinct = HashSet::new();
    let mut secondary_distinct = HashSet::new();
    let mut primary_runs = 0usize;
    let mut secondary_runs = 0usize;
    let mut previous_primary = None;
    let mut previous_secondary = None;
    for value in values {
        let (primary, secondary) = split_f64(*value, base);
        primary_min = primary_min.min(primary);
        primary_max = primary_max.max(primary);
        secondary_min = secondary_min.min(secondary);
        secondary_max = secondary_max.max(secondary);
        primary_distinct.insert(primary);
        secondary_distinct.insert(secondary);
        primary_runs += usize::from(previous_primary != Some(primary));
        secondary_runs += usize::from(previous_secondary != Some(secondary));
        previous_primary = Some(primary);
        previous_secondary = Some(secondary);
    }
    let primary_bits = u64::BITS - primary_max.wrapping_sub(primary_min).leading_zeros();
    let secondary_bits = u64::BITS - secondary_max.wrapping_sub(secondary_min).leading_zeros();
    Some(
        estimated_latent_cost(
            primary_bits,
            primary_distinct.len(),
            primary_runs,
            values.len(),
        ) + estimated_latent_cost(
            secondary_bits,
            secondary_distinct.len(),
            secondary_runs,
            values.len(),
        ),
    )
}

fn constant_score_f32(values: &[f64], base: f32) -> Option<u32> {
    let mut primary_min = u32::MAX;
    let mut primary_max = u32::MIN;
    for value in values {
        let (primary, secondary) = split_f32(narrow_f32(*value), base);
        if secondary != 1_u32 << 31 {
            return None;
        }
        primary_min = primary_min.min(primary);
        primary_max = primary_max.max(primary);
    }
    Some(u32::BITS - (primary_max - primary_min).leading_zeros())
}

fn constant_score_f64(values: &[f64], base: f64) -> Option<u32> {
    let mut primary_min = u64::MAX;
    let mut primary_max = u64::MIN;
    for value in values {
        let (primary, secondary) = split_f64(*value, base);
        if secondary != 1_u64 << 63 {
            return None;
        }
        primary_min = primary_min.min(primary);
        primary_max = primary_max.max(primary);
    }
    Some(u64::BITS - (primary_max - primary_min).leading_zeros())
}

fn estimated_latent_cost(
    span_bits: u32,
    distinct_count: usize,
    run_count: usize,
    len: usize,
) -> f64 {
    if span_bits == 0 {
        return 0.0;
    }
    let span_bits = f64::from(span_bits);
    let code_bits = usize::BITS - distinct_count.saturating_sub(1).leading_zeros();
    let dict_cost = f64::from(code_bits) + distinct_count as f64 / len as f64 * span_bits;
    let index_bits = usize::BITS - len.saturating_sub(1).leading_zeros();
    let run_cost = run_count as f64 / len as f64 * (span_bits + f64::from(index_bits));
    span_bits.min(dict_cost).min(run_cost)
}

fn approximate_common_divisor(values: &[f64]) -> Option<f64> {
    let mut divisors = values
        .chunks_exact(2)
        .filter_map(|pair| {
            let a = pair[0].abs();
            let b = pair[1].abs();
            approximate_pair_divisor(a.max(b), a.min(b))
        })
        .collect::<Vec<_>>();
    if divisors.len() < 2 {
        return None;
    }
    divisors.sort_unstable_by(f64::total_cmp);
    for percentile in [1usize, 3, 5] {
        let candidate = divisors[percentile * divisors.len() / 10];
        let similar = divisors
            .iter()
            .filter(|divisor| (**divisor - candidate).abs() < candidate * 0.01)
            .count();
        if similar >= 2 {
            return Some(candidate);
        }
    }
    None
}

fn approximate_pair_divisor(greater: f64, lesser: f64) -> Option<f64> {
    if lesser == 0.0 || lesser == greater || lesser <= greater * 2.0_f64.powi(-46) {
        return None;
    }
    let mut greater_value = greater;
    let mut greater_error = 0.0;
    let mut lesser_value = lesser;
    let mut lesser_error = 0.0;
    for _ in 0..64 {
        let ratio = (greater_value / lesser_value).round();
        greater_error += ratio * lesser_error + greater_value * f64::EPSILON;
        let previous = greater_value;
        greater_value = (greater_value - ratio * lesser_value).abs();
        if greater_value <= previous * 2.0_f64.powi(-16) || greater_value <= greater_error {
            return Some(lesser_value);
        }
        if greater_value <= greater * 2.0_f64.powi(-46)
            || greater_value <= greater_error * 2.0_f64.powi(6)
        {
            return None;
        }
        std::mem::swap(&mut greater_value, &mut lesser_value);
        std::mem::swap(&mut greater_error, &mut lesser_error);
    }
    None
}

fn snap_base(base: f64) -> f64 {
    let inverse = base.recip();
    let rounded = inverse.round();
    let decimal = 10.0_f64.powf(inverse.log10().round());
    if (inverse - rounded).abs() < 0.02 {
        rounded.recip()
    } else if (inverse - decimal).abs() / inverse < 0.01 {
        decimal.recip()
    } else {
        base
    }
}

fn latent_ptype(ptype: PType) -> VortexResult<PType> {
    match ptype {
        PType::F32 => Ok(PType::U32),
        PType::F64 => Ok(PType::U64),
        _ => vortex_bail!("FloatMult requires f32 or f64, got {ptype}"),
    }
}

#[allow(clippy::cast_possible_truncation)]
fn int_float_to_u32(value: f32) -> u32 {
    let absolute = value.abs();
    let greatest_precise_integer = 1_u32 << f32::MANTISSA_DIGITS;
    let greatest_precise_float = greatest_precise_integer as f32;
    let absolute_integer = if absolute < greatest_precise_float {
        absolute as u32
    } else {
        greatest_precise_integer + (absolute.to_bits() - greatest_precise_float.to_bits())
    };
    if value.is_sign_positive() {
        (1_u32 << 31) + absolute_integer
    } else {
        (1_u32 << 31) - 1 - absolute_integer
    }
}

fn int_float_from_u32(value: u32) -> f32 {
    let middle = 1_u32 << 31;
    let (absolute, positive) = if value >= middle {
        (value - middle, true)
    } else {
        (middle - 1 - value, false)
    };
    let greatest_precise_integer = 1_u32 << f32::MANTISSA_DIGITS;
    let greatest_precise_float = greatest_precise_integer as f32;
    let absolute_float = if absolute < greatest_precise_integer {
        absolute as f32
    } else {
        f32::from_bits(greatest_precise_float.to_bits() + absolute - greatest_precise_integer)
    };
    if positive {
        absolute_float
    } else {
        -absolute_float
    }
}

#[allow(clippy::cast_possible_truncation)]
fn int_float_to_u64(value: f64) -> u64 {
    let absolute = value.abs();
    let greatest_precise_integer = 1_u64 << f64::MANTISSA_DIGITS;
    let greatest_precise_float = greatest_precise_integer as f64;
    let absolute_integer = if absolute < greatest_precise_float {
        absolute as u64
    } else {
        greatest_precise_integer + (absolute.to_bits() - greatest_precise_float.to_bits())
    };
    if value.is_sign_positive() {
        (1_u64 << 63) + absolute_integer
    } else {
        (1_u64 << 63) - 1 - absolute_integer
    }
}

#[allow(clippy::cast_possible_truncation)]
fn narrow_f32(value: f64) -> f32 {
    value as f32
}

fn int_float_from_u64(value: u64) -> f64 {
    let middle = 1_u64 << 63;
    let (absolute, positive) = if value >= middle {
        (value - middle, true)
    } else {
        (middle - 1 - value, false)
    };
    let greatest_precise_integer = 1_u64 << f64::MANTISSA_DIGITS;
    let greatest_precise_float = greatest_precise_integer as f64;
    let absolute_float = if absolute < greatest_precise_integer {
        absolute as f64
    } else {
        f64::from_bits(greatest_precise_float.to_bits() + absolute - greatest_precise_integer)
    };
    if positive {
        absolute_float
    } else {
        -absolute_float
    }
}

fn ordered_u32(value: f32) -> u32 {
    let bits = value.to_bits();
    if bits & (1_u32 << 31) == 0 {
        bits ^ (1_u32 << 31)
    } else {
        !bits
    }
}

fn ordered_u64(value: f64) -> u64 {
    let bits = value.to_bits();
    if bits & (1_u64 << 63) == 0 {
        bits ^ (1_u64 << 63)
    } else {
        !bits
    }
}

fn from_ordered_u32(ordered: u32) -> f32 {
    let bits = if ordered & (1_u32 << 31) == 0 {
        !ordered
    } else {
        ordered ^ (1_u32 << 31)
    };
    f32::from_bits(bits)
}

fn from_ordered_u64(ordered: u64) -> f64 {
    let bits = if ordered & (1_u64 << 63) == 0 {
        !ordered
    } else {
        ordered ^ (1_u64 << 63)
    };
    f64::from_bits(bits)
}

fn split_f32(value: f32, base: f32) -> (u32, u32) {
    let multiple = (value / base).round();
    let primary = int_float_to_u32(multiple);
    let secondary = ordered_u32(value)
        .wrapping_sub(ordered_u32(multiple * base))
        .wrapping_add(1_u32 << 31);
    (primary, secondary)
}

fn split_f64(value: f64, base: f64) -> (u64, u64) {
    let multiple = (value / base).round();
    let primary = int_float_to_u64(multiple);
    let secondary = ordered_u64(value)
        .wrapping_sub(ordered_u64(multiple * base))
        .wrapping_add(1_u64 << 63);
    (primary, secondary)
}

fn join_f32(primary: u32, secondary: u32, base: f32) -> f32 {
    let approximate = int_float_from_u32(primary) * base;
    from_ordered_u32(ordered_u32(approximate).wrapping_add(secondary.wrapping_add(1_u32 << 31)))
}

fn join_f64(primary: u64, secondary: u64, base: f64) -> f64 {
    let approximate = int_float_from_u64(primary) * base;
    from_ordered_u64(ordered_u64(approximate).wrapping_add(secondary.wrapping_add(1_u64 << 63)))
}

fn decode(array: ArrayView<'_, FloatMult>, ctx: &mut ExecutionCtx) -> VortexResult<PrimitiveArray> {
    let primary = array.primary().clone().execute::<PrimitiveArray>(ctx)?;
    let validity = primary.validity()?;
    let Some(secondary) = array.secondary() else {
        return Ok(match PType::try_from(array.dtype())? {
            PType::F32 => {
                let base = array.data().base_f32()?;
                let values = primary
                    .into_buffer::<u32>()
                    .map_each_in_place(|primary| int_float_from_u32(primary) * base)
                    .freeze();
                PrimitiveArray::new(values, validity)
            }
            PType::F64 => {
                let base = array.data().base_f64()?;
                let values = primary
                    .into_buffer::<u64>()
                    .map_each_in_place(|primary| int_float_from_u64(primary) * base)
                    .freeze();
                PrimitiveArray::new(values, validity)
            }
            ptype => vortex_panic!("unsupported FloatMult ptype {ptype}"),
        });
    };

    let secondary = secondary.clone().execute::<PrimitiveArray>(ctx)?;
    Ok(match PType::try_from(array.dtype())? {
        PType::F32 => {
            let secondary_values = secondary.as_slice::<u32>();
            let base = array.data().base_f32()?;
            let mut index = 0;
            let values = primary
                .into_buffer::<u32>()
                .map_each_in_place(|primary| {
                    let value = join_f32(primary, secondary_values[index], base);
                    index += 1;
                    value
                })
                .freeze();
            PrimitiveArray::new(values, validity)
        }
        PType::F64 => {
            let secondary_values = secondary.as_slice::<u64>();
            let base = array.data().base_f64()?;
            let mut index = 0;
            let values = primary
                .into_buffer::<u64>()
                .map_each_in_place(|primary| {
                    let value = join_f64(primary, secondary_values[index], base);
                    index += 1;
                    value
                })
                .freeze();
            PrimitiveArray::new(values, validity)
        }
        ptype => vortex_panic!("unsupported FloatMult ptype {ptype}"),
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
            1.234_567_89,
            f64::INFINITY,
            f64::from_bits(0x7ff8_0000_0000_1234),
            f64::from_bits(0xfff8_0000_0000_5678),
        ];
        let original = PrimitiveArray::from_iter(values);
        let encoded = FloatMult::from_primitive(original.as_view(), 0.01)?;
        let secondary = encoded
            .secondary()
            .vortex_expect("general FloatMult must contain adjustments")
            .clone()
            .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())?;
        assert!(
            secondary
                .as_slice::<u64>()
                .iter()
                .any(|adjustment| *adjustment != 1_u64 << 63)
        );
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
    fn implicit_zero_adjustment_roundtrip() -> VortexResult<()> {
        let original = PrimitiveArray::from_iter([0.0_f32, 1.0, 42.0, 65_536.0]);
        let encoded = FloatMult::from_primitive_constant_secondary(original.as_view(), 1.0)?
            .vortex_expect("integer floats use an implicit zero adjustment");
        assert!(encoded.secondary().is_none());
        assert_eq!(encoded.as_ref().nchildren(), 1);
        assert_arrays_eq!(encoded, original, &mut SESSION.create_execution_ctx());

        let encoded = encoded.into_array();
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
        assert!(decoded.as_::<FloatMult>().secondary().is_none());
        assert_arrays_eq!(decoded, original, &mut SESSION.create_execution_ctx());
        Ok(())
    }

    #[test]
    fn nullable_slice_and_scalar_access() -> VortexResult<()> {
        let original = PrimitiveArray::new(
            Buffer::from(vec![1.25_f32, 0.0, -0.0, 42.5, -10.0]),
            Validity::from_iter([true, false, true, true, false]),
        );
        let encoded = FloatMult::from_primitive(original.as_view(), 0.25)?;
        let mut ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(encoded, original, &mut ctx);
        assert_nth_scalar!(encoded, 3, 42.5_f32, &mut ctx);
        assert!(encoded.execute_scalar(1, &mut ctx)?.is_null());

        let sliced = encoded.into_array().slice(1..4)?;
        assert!(sliced.is::<FloatMult>());
        assert_arrays_eq!(sliced, original.into_array().slice(1..4)?, &mut ctx);
        Ok(())
    }

    #[test]
    fn serialization_roundtrip() -> VortexResult<()> {
        let original = PrimitiveArray::from_option_iter([
            Some(-10.25_f64),
            None,
            Some(-0.0),
            Some(42.25),
            Some(100.5),
        ]);
        let encoded = FloatMult::from_primitive(original.as_view(), 0.25)?.into_array();
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
        assert!(decoded.is::<FloatMult>());
        assert_arrays_eq!(decoded, original, &mut SESSION.create_execution_ctx());
        Ok(())
    }

    #[test]
    fn estimates_decimal_and_binary_bases() {
        let decimals = PrimitiveArray::from_iter((0..1_000).map(|value| value as f64 * 0.01));
        assert_eq!(estimate_float_mult_base(decimals.as_view()), Some(0.01));

        let binary = PrimitiveArray::from_iter((0..1_000).map(|value| value as f64 * 0.125));
        assert_eq!(estimate_float_mult_base(binary.as_view()), Some(0.125));
    }
}
