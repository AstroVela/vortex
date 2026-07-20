// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBufferMut;
use vortex_buffer::BitBufferView;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::GroupRanges;
use crate::aggregate_fn::GroupedArray;
use crate::aggregate_fn::fns::max::Max;
use crate::aggregate_fn::fns::min::Min;
use crate::aggregate_fn::kernels::DynGroupedAggregateKernel;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::dtype::NativePType;
use crate::dtype::half::f16;
use crate::match_each_native_ptype;
use crate::validity::Validity;

/// Encoding-specific grouped [`Min`] and [`Max`] kernel for primitive element arrays.
///
/// Each outer list is one group over a shared primitive element buffer. The kernel consumes the
/// outer ranges and both validity levels in bulk, avoiding a per-list array slice and aggregate
/// state. Other element encodings may execute into [`Primitive`] before reaching this kernel.
#[derive(Debug)]
pub(crate) struct PrimitiveGroupedExtremaEncodingKernel;

impl DynGroupedAggregateKernel for PrimitiveGroupedExtremaEncodingKernel {
    fn grouped_aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        groups: &GroupedArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if let Some(options) = aggregate_fn.as_opt::<Min>() {
            return try_grouped_extreme(groups, ctx, options.skip_nans, Extreme::Min);
        }
        if let Some(options) = aggregate_fn.as_opt::<Max>() {
            return try_grouped_extreme(groups, ctx, options.skip_nans, Extreme::Max);
        }
        Ok(None)
    }
}

#[derive(Clone, Copy)]
enum Extreme {
    Min,
    Max,
}

/// Type-specific identities and combine operations used by the lane reducer.
///
/// Starting every lane at an identity eliminates a seed search. A separate participation flag is
/// still required because the identity may also be a legitimate result. Keeping `CAN_NAN` and the
/// combine operation type-specific lets monomorphization remove float-only work for integers and
/// expose native min/max instructions to the optimizer.
trait ExtremaIdentity: NativePType {
    const CAN_NAN: bool;
    const MIN_IDENTITY: Self;
    const MAX_IDENTITY: Self;

    fn minimum(candidate: Self, current: Self) -> Self;
    fn maximum(candidate: Self, current: Self) -> Self;
}

macro_rules! impl_integer_extrema_identity {
    ($($t:ty),* $(,)?) => {
        $(
            impl ExtremaIdentity for $t {
                const CAN_NAN: bool = false;
                const MIN_IDENTITY: Self = Self::MAX;
                const MAX_IDENTITY: Self = Self::MIN;

                #[inline(always)]
                fn minimum(candidate: Self, current: Self) -> Self {
                    candidate.min(current)
                }

                #[inline(always)]
                fn maximum(candidate: Self, current: Self) -> Self {
                    candidate.max(current)
                }
            }
        )*
    };
}

macro_rules! impl_float_extrema_identity {
    ($($t:ty),* $(,)?) => {
        $(
            impl ExtremaIdentity for $t {
                const CAN_NAN: bool = true;
                const MIN_IDENTITY: Self = Self::INFINITY;
                const MAX_IDENTITY: Self = Self::NEG_INFINITY;

                #[inline(always)]
                fn minimum(candidate: Self, current: Self) -> Self {
                    select(candidate.is_lt(current), candidate, current)
                }

                #[inline(always)]
                fn maximum(candidate: Self, current: Self) -> Self {
                    select(candidate.is_gt(current), candidate, current)
                }
            }
        )*
    };
}

impl_integer_extrema_identity!(u8, u16, u32, u64, i8, i16, i32, i64);
impl_float_extrema_identity!(f16, f32, f64);

/// Fully specialized operations shared by the scalar and lane reducers for one invocation.
#[derive(Clone, Copy)]
struct ExtremaOps<T, B, C> {
    skip_nans: bool,
    identity: T,
    is_better: B,
    combine: C,
}

/// Element validity materialized once for the complete shared element array.
///
/// The `Some` variant borrows the packed bitmap. Per-group reducers take zero-copy views or load a
/// word directly rather than constructing a new [`Mask`] and its cached valid-run representation.
#[derive(Clone, Copy)]
enum ElementValidity<'a> {
    All,
    None,
    Some(BitBufferView<'a>),
}

/// Run the encoding kernel when the shared elements are primitive.
///
/// Returning `None` declines the encoding-specific dispatch and allows the grouped accumulator to
/// execute the elements further or use its generic per-group fallback.
fn try_grouped_extreme(
    groups: &GroupedArray,
    ctx: &mut ExecutionCtx,
    skip_nans: bool,
    extreme: Extreme,
) -> VortexResult<Option<ArrayRef>> {
    if !groups.elements().is::<Primitive>() {
        return Ok(None);
    }
    let elements = groups.elements().clone().downcast::<Primitive>();
    let group_ranges = groups.group_ranges(ctx)?;
    let group_validity = groups.group_validity(ctx)?;
    Ok(Some(grouped_extreme(
        &elements,
        &group_ranges,
        &group_validity,
        ctx,
        skip_nans,
        extreme,
    )?))
}

/// Materialize element validity once, select the native physical type, and collect one result per
/// group.
///
/// Type dispatch happens outside the group loop, so the inner reducer is monomorphized for both the
/// physical type and the requested extremum.
fn grouped_extreme(
    elements: &PrimitiveArray,
    group_ranges: &GroupRanges,
    group_validity: &Mask,
    ctx: &mut ExecutionCtx,
    skip_nans: bool,
    extreme: Extreme,
) -> VortexResult<ArrayRef> {
    let element_validity = elements
        .as_ref()
        .validity()?
        .execute_mask(elements.as_ref().len(), ctx)?;

    let result = match_each_native_ptype!(elements.ptype(), |T| {
        let values = elements.as_slice::<T>();
        match extreme {
            Extreme::Min => collect_extrema(
                values,
                group_ranges,
                group_validity,
                &element_validity,
                ExtremaOps {
                    skip_nans,
                    identity: T::MIN_IDENTITY,
                    is_better: |candidate: T, current| candidate.is_lt(current),
                    combine: <T as ExtremaIdentity>::minimum,
                },
            ),
            Extreme::Max => collect_extrema(
                values,
                group_ranges,
                group_validity,
                &element_validity,
                ExtremaOps {
                    skip_nans,
                    identity: T::MAX_IDENTITY,
                    is_better: |candidate: T, current| candidate.is_gt(current),
                    combine: <T as ExtremaIdentity>::maximum,
                },
            ),
        }
    });
    Ok(result.into_array())
}

/// Reduce every valid group directly into a final-length output buffer.
///
/// Invalid outer lists are already represented by `group_validity`. `missing` records only valid
/// groups that have no participating element: empty lists, all-null lists, or float lists containing
/// only skipped NaNs.
fn collect_extrema<T, B, C>(
    values: &[T],
    group_ranges: &GroupRanges,
    group_validity: &Mask,
    element_validity: &Mask,
    ops: ExtremaOps<T, B, C>,
) -> PrimitiveArray
where
    T: ExtremaIdentity,
    B: Fn(T, T) -> bool + Copy,
    C: Fn(T, T) -> T + Copy,
{
    let element_validity = match element_validity.bit_buffer() {
        AllOr::All => ElementValidity::All,
        AllOr::None => ElementValidity::None,
        AllOr::Some(validity) => ElementValidity::Some(validity.as_view()),
    };
    let mut extrema = BufferMut::<T>::zeroed(group_ranges.len());
    let mut missing = Vec::new();

    for (index, ((offset, size), group_is_valid)) in
        group_ranges.iter().zip(group_validity.iter()).enumerate()
    {
        if !group_is_valid {
            continue;
        }
        match reduce_group(values, offset, size, element_validity, ops) {
            Some(extreme) => extrema.as_mut_slice()[index] = extreme,
            None => missing.push(index),
        }
    }

    PrimitiveArray::new(extrema.freeze(), output_validity(group_validity, &missing))
}

/// Combine outer-list validity with valid groups that did not produce an extremum.
///
/// The common case reuses the original bitmap. A mutable copy is allocated only when at least one
/// otherwise-valid group is empty, all-null, or all-skipped-NaN.
fn output_validity(group_validity: &Mask, missing: &[usize]) -> Validity {
    if missing.is_empty() {
        return match group_validity.bit_buffer() {
            AllOr::All => Validity::AllValid,
            AllOr::None => Validity::AllInvalid,
            AllOr::Some(validity) => Validity::from(validity.clone()),
        };
    }

    let mut validity = match group_validity.bit_buffer() {
        AllOr::All => BitBufferMut::new_set(group_validity.len()),
        AllOr::None => BitBufferMut::new_unset(group_validity.len()),
        AllOr::Some(validity) => BitBufferMut::copy_from(validity),
    };
    for &index in missing {
        validity.unset(index);
    }
    Validity::from(validity.freeze())
}

/// Reduce one `(offset, size)` range using a size-adaptive strategy.
///
/// Groups shorter than 32 values use the scalar bitmap-word path. Groups from 32 through 511 use
/// one 16-byte native vector's worth of independent accumulators. Larger groups use four vectors'
/// worth to shorten the loop-carried dependency chain and amortize validity-mask handling.
///
/// The range is independent of neighboring groups, which is required for overlapping and
/// out-of-order ListView values.
fn reduce_group<T, B, C>(
    values: &[T],
    offset: usize,
    size: usize,
    element_validity: ElementValidity<'_>,
    ops: ExtremaOps<T, B, C>,
) -> Option<T>
where
    T: ExtremaIdentity,
    B: Fn(T, T) -> bool + Copy,
    C: Fn(T, T) -> T + Copy,
{
    if size < 32 {
        return reduce_group_scalar(
            values,
            offset,
            size,
            element_validity,
            ops.skip_nans,
            ops.is_better,
        );
    }

    // Long groups use four native SIMD vectors of independent accumulators to shorten the
    // loop-carried dependency. Smaller groups use one vector to limit setup and reduction costs.
    if size >= 512 {
        return match size_of::<T>() {
            1 => reduce_group_lanes::<T, B, C, 64>(values, offset, size, element_validity, ops),
            2 => reduce_group_lanes::<T, B, C, 32>(values, offset, size, element_validity, ops),
            4 => reduce_group_lanes::<T, B, C, 16>(values, offset, size, element_validity, ops),
            8 => reduce_group_lanes::<T, B, C, 8>(values, offset, size, element_validity, ops),
            _ => reduce_group_lanes::<T, B, C, 4>(values, offset, size, element_validity, ops),
        };
    }

    match size_of::<T>() {
        1 => reduce_group_lanes::<T, B, C, 16>(values, offset, size, element_validity, ops),
        2 => reduce_group_lanes::<T, B, C, 8>(values, offset, size, element_validity, ops),
        4 => reduce_group_lanes::<T, B, C, 4>(values, offset, size, element_validity, ops),
        8 => reduce_group_lanes::<T, B, C, 2>(values, offset, size, element_validity, ops),
        _ => reduce_group_lanes::<T, B, C, 1>(values, offset, size, element_validity, ops),
    }
}

/// Reduce a group through `LANES` independent accumulators and merge them into one result.
///
/// `LANES` is selected to represent either 16 or 64 bytes of values. The element bitmap is sliced
/// as a borrowed [`BitBufferView`]; all-valid groups synthesize validity words without allocating a
/// bitmap. `found` distinguishes an empty reduction from a real result equal to `ops.identity`.
fn reduce_group_lanes<T, B, C, const LANES: usize>(
    values: &[T],
    offset: usize,
    size: usize,
    element_validity: ElementValidity<'_>,
    ops: ExtremaOps<T, B, C>,
) -> Option<T>
where
    T: ExtremaIdentity,
    B: Fn(T, T) -> bool + Copy,
    C: Fn(T, T) -> T + Copy,
{
    debug_assert!(LANES > 0 && LANES.is_power_of_two() && 64 % LANES == 0);
    let values = &values[offset..offset + size];
    let validity = match element_validity {
        ElementValidity::All => None,
        ElementValidity::None => return None,
        ElementValidity::Some(validity) => Some(validity.slice(offset..offset + size)),
    };
    let mut accumulators = [ops.identity; LANES];
    let mut found = false;
    let poisoned = match validity {
        Some(validity) => reduce_validity_words(
            &mut accumulators,
            values,
            validity.chunks().iter_padded(),
            ops.skip_nans,
            &ops.combine,
            &mut found,
        ),
        None => reduce_validity_words(
            &mut accumulators,
            values,
            std::iter::repeat_n(u64::MAX, values.len().div_ceil(64)),
            ops.skip_nans,
            &ops.combine,
            &mut found,
        ),
    };
    if let Some(nan) = poisoned {
        return Some(nan);
    }

    found.then(|| merge_accumulators(accumulators, &ops.is_better))
}

/// Reduce packed validity words without materializing per-group masks or valid-run vectors.
///
/// Each input word describes at most 64 consecutive values. Tail bits are explicitly masked, so a
/// padded final word cannot make an empty lane appear to participate. A returned value is a NaN that
/// poisoned the group under `include_nans`; ordinary completion returns `None` and leaves the lane
/// results in `accumulators`.
fn reduce_validity_words<T: ExtremaIdentity, const LANES: usize>(
    accumulators: &mut [T; LANES],
    values: &[T],
    validity_words: impl Iterator<Item = u64>,
    skip_nans: bool,
    combine: &impl Fn(T, T) -> T,
    found: &mut bool,
) -> Option<T> {
    let mut base = 0;
    for word in validity_words {
        let word_len = values.len().saturating_sub(base).min(64);
        let valid_mask = if word_len == 64 {
            u64::MAX
        } else {
            (1u64 << word_len) - 1
        };
        if !T::CAN_NAN || !skip_nans {
            *found |= word & valid_mask != 0;
        }
        let word_values = &values[base..base + word_len];
        let (chunks, remainder) = word_values.as_chunks::<LANES>();
        let mut validity = word;
        for lane_values in chunks {
            if let Some(nan) = reduce_lane_chunk(
                accumulators,
                lane_values,
                validity,
                skip_nans,
                combine,
                found,
            ) {
                return Some(nan);
            }
            if LANES == 64 {
                validity = 0;
            } else {
                validity >>= LANES;
            }
        }

        if !remainder.is_empty()
            && let Some(nan) =
                reduce_lane_remainder(accumulators, remainder, validity, skip_nans, combine, found)
        {
            return Some(nan);
        }
        base += word_len;
    }
    None
}

/// Reduce a short group without constructing a bitmap slice or lane accumulators.
///
/// Nullable groups load all relevant validity bits into one word. This path is restricted to fewer
/// than 32 values, which keeps the direct unaligned bitmap load simple even when the view begins at
/// a non-byte-aligned offset.
fn reduce_group_scalar<T: NativePType>(
    values: &[T],
    offset: usize,
    size: usize,
    element_validity: ElementValidity<'_>,
    skip_nans: bool,
    is_better: impl Fn(T, T) -> bool,
) -> Option<T> {
    let mut best = None;
    let group_values = &values[offset..offset + size];
    match element_validity {
        ElementValidity::All => {
            reduce_scalar_run(&mut best, group_values, skip_nans, &is_better);
        }
        ElementValidity::None => {}
        ElementValidity::Some(validity) => {
            reduce_nullable_word_scalar(
                &mut best,
                group_values,
                validity_word(validity, offset, size),
                skip_nans,
                &is_better,
            );
        }
    }
    best
}

/// Reduce the contiguous set-bit runs in one short group's validity word.
///
/// The all-valid fast path scans the entire value slice. Otherwise, zero and one bit counts locate
/// valid runs without a per-element validity lookup.
fn reduce_nullable_word_scalar<T: NativePType>(
    best: &mut Option<T>,
    values: &[T],
    mut word: u64,
    skip_nans: bool,
    is_better: &impl Fn(T, T) -> bool,
) -> bool {
    let valid_mask = if values.is_empty() {
        0
    } else {
        (1u64 << values.len()) - 1
    };
    word &= valid_mask;

    if word == valid_mask {
        return reduce_scalar_run(best, values, skip_nans, is_better);
    }

    while word != 0 {
        let start = word.trailing_zeros() as usize;
        let run_len = (word >> start).trailing_ones() as usize;
        if reduce_scalar_run(best, &values[start..start + run_len], skip_nans, is_better) {
            return true;
        }
        word &= !(((1u64 << run_len) - 1) << start);
    }
    false
}

#[inline]
/// Load and align the validity bits for a short group from the shared bitmap.
///
/// The checked eight-byte chunk load compiles to an inline word load in the common case. The fold is
/// used only near the end of the backing buffer where eight bytes are not available.
fn validity_word(validity: BitBufferView<'_>, offset: usize, len: usize) -> u64 {
    debug_assert!(len < 32);
    debug_assert!(offset + len <= validity.len());
    if len == 0 {
        return 0;
    }

    let bit_offset = validity.offset() + offset;
    let buffer = &validity.inner()[bit_offset / 8..];
    let word = buffer.first_chunk::<8>().map_or_else(
        || {
            buffer.iter().enumerate().fold(0, |word, (index, &byte)| {
                word | u64::from(byte) << (index * 8)
            })
        },
        |bytes| u64::from_le_bytes(*bytes),
    );
    (word >> (bit_offset % 8)) & ((1u64 << len) - 1)
}

/// Fold a contiguous run of valid values into `best`.
///
/// Returns `true` only when an included NaN poisons the group, allowing callers to stop scanning
/// that group immediately. Skipped NaNs do not initialize or update `best`.
fn reduce_scalar_run<T: NativePType>(
    best: &mut Option<T>,
    values: &[T],
    skip_nans: bool,
    is_better: &impl Fn(T, T) -> bool,
) -> bool {
    for &candidate in values {
        if candidate.is_nan() {
            if skip_nans {
                continue;
            }
            *best = Some(candidate);
            return true;
        }
        match best {
            Some(current) if is_better(candidate, *current) => *current = candidate,
            None => *best = Some(candidate),
            _ => {}
        }
    }
    false
}

#[inline(always)]
/// Combine one full lane chunk with its packed validity bits.
///
/// The branchless combine-and-select form is intentional: after monomorphization, LLVM can lower
/// integer min/max plus validity selection to packed SIMD instructions. Float-only NaN logic is
/// removed entirely for integer types through [`ExtremaIdentity::CAN_NAN`].
fn reduce_lane_chunk<T: ExtremaIdentity, const LANES: usize>(
    accumulators: &mut [T; LANES],
    values: &[T; LANES],
    validity: u64,
    skip_nans: bool,
    combine: &impl Fn(T, T) -> T,
    found: &mut bool,
) -> Option<T> {
    for lane in 0..LANES {
        let candidate = values[lane];
        let valid = validity & (1 << lane) != 0;
        let current = accumulators[lane];
        if T::CAN_NAN {
            let is_nan = candidate.is_nan();
            if skip_nans {
                *found |= valid & !is_nan;
            }
            if valid & is_nan & !skip_nans {
                return Some(candidate);
            }
            let reduced = combine(candidate, current);
            accumulators[lane] = select(valid & !is_nan, reduced, current);
        } else {
            let reduced = combine(candidate, current);
            accumulators[lane] = select(valid, reduced, current);
        }
    }
    None
}

/// Apply the same lane semantics to the final partial chunk of a validity word.
fn reduce_lane_remainder<T: ExtremaIdentity, const LANES: usize>(
    accumulators: &mut [T; LANES],
    values: &[T],
    validity: u64,
    skip_nans: bool,
    combine: &impl Fn(T, T) -> T,
    found: &mut bool,
) -> Option<T> {
    for (lane, &candidate) in values.iter().enumerate() {
        let valid = validity & (1 << lane) != 0;
        let current = accumulators[lane];
        if T::CAN_NAN {
            let is_nan = candidate.is_nan();
            if skip_nans {
                *found |= valid & !is_nan;
            }
            if valid & is_nan & !skip_nans {
                return Some(candidate);
            }
            let reduced = combine(candidate, current);
            accumulators[lane] = select(valid & !is_nan, reduced, current);
        } else {
            let reduced = combine(candidate, current);
            accumulators[lane] = select(valid, reduced, current);
        }
    }
    None
}

/// Tree-reduce independent lane accumulators into the group's scalar extremum.
///
/// A tree keeps the dependency depth logarithmic in `LANES`, which matters most for the four-vector
/// path used by long groups.
fn merge_accumulators<T: NativePType, const LANES: usize>(
    mut accumulators: [T; LANES],
    is_better: &impl Fn(T, T) -> bool,
) -> T {
    let mut len = LANES;
    while len >= 2 {
        let mid = len / 2;
        for lane in 0..mid {
            let candidate = accumulators[lane + mid];
            let current = accumulators[lane];
            accumulators[lane] = select(is_better(candidate, current), candidate, current);
        }
        len /= 2;
    }
    accumulators[0]
}

#[inline(always)]
/// Select a value in a form that the lane loop can lower to a SIMD mask operation.
fn select<T: Copy>(condition: bool, if_true: T, if_false: T) -> T {
    if condition { if_true } else { if_false }
}
