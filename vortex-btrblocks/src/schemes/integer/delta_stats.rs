// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cheap sampled statistics that predict whether Delta encoding pays off.
//!
//! Delta's benefit is a *local* property: what a later FoR / ZigZag + BitPacking layer packs is
//! the difference between neighbouring values, so a handful of short contiguous runs describe it
//! as well as the whole array does. That is what makes these stats cheap: they read
//! [`SAMPLE_RUN_LEN`] × [`SAMPLE_RUN_COUNT`] values regardless of array length, in a single pass,
//! accumulating two bit-width histograms and four counters.
//!
//! FastLanes Delta stores the lag-1 difference of the *original* order. The FastLanes transpose
//! permutes values within a 1024-element chunk, and the delta kernel walks each lane in
//! `FL_ORDER`, which together map back to consecutive original indices; only the `1024 / T` lane
//! heads per chunk are stored as bases instead. So sampling consecutive values models exactly the
//! residuals that get packed, and the bases cost exactly one bit per value
//! (`1024 / T` bases of `T` bits per 1024 values).
//!
//! See `scripts/delta-analysis/README.md` for the measurements behind the model, including why
//! delta-of-delta is exposed here as a statistic but not as a scheme.

use std::sync::Arc;

use rand::RngExt;
use rand::SeedableRng;
use rand::prelude::StdRng;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::PType;
use vortex_array::match_each_integer_ptype;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_fastlanes::bitpack_compress::find_best_bit_width;

use crate::ArrayAndStats;

/// Number of consecutive values in each sampled run.
///
/// Runs must be contiguous: a residual only exists between neighbouring values, so index-wise
/// random sampling would measure nothing about Delta.
pub const SAMPLE_RUN_LEN: usize = 64;

/// Number of sampled runs, spread evenly across the array.
pub const SAMPLE_RUN_COUNT: usize = 16;

/// Fixed seed, so that stats - and therefore compressed output - stay deterministic.
const SAMPLE_SEED: u64 = 1234567890;

/// The type's own bit width, as the `u8` that BitPacking widths are expressed in.
fn full_bit_width(ptype: PType) -> u8 {
    u8::try_from(ptype.bit_width()).vortex_expect("integer ptypes are at most 64 bits wide")
}

/// Bytes of overhead per BitPacking exception (the value plus a `u32` index), mirroring the cost
/// model in [`find_best_bit_width`].
fn bytes_per_exception(ptype: PType) -> usize {
    ptype.byte_width() + 4
}

/// Sampled statistics describing how compressible an integer array's differences are.
///
/// Both lag-1 (Delta) and lag-2 (delta-of-delta) residuals are described, each by a bit-width
/// histogram of the residual as the cascade below Delta would see it, plus the FoR span.
#[derive(Debug, Clone)]
pub struct DeltaStats {
    /// The primitive type of the array the stats were generated from.
    ptype: PType,
    /// Number of lag-1 residuals in the sample.
    delta_count: u32,
    /// Number of lag-2 residuals in the sample.
    dod_count: u32,
    /// Number of lag-1 residuals equal to zero.
    zero_count: u32,
    /// Number of lag-1 residuals below zero.
    decreasing_count: u32,
    /// Histogram of packed bit widths of the lag-1 residuals, indexed by bit width.
    delta_widths: Vec<usize>,
    /// Histogram of packed bit widths of the lag-2 residuals, indexed by bit width.
    dod_widths: Vec<usize>,
    /// `max - min` of the lag-1 residuals, saturating at the type's range.
    delta_span: u128,
    /// `max - min` of the lag-2 residuals, saturating at the type's range.
    dod_span: u128,
}

/// Returns the array's [`DeltaStats`], generating them on first access and caching them for the
/// rest of the compression site.
pub fn delta_stats(data: &ArrayAndStats, ctx: &mut ExecutionCtx) -> Arc<DeltaStats> {
    let primitive = data.array_as_primitive().into_owned();
    data.get_or_insert_with::<DeltaStats>(|| {
        DeltaStats::generate(&primitive, ctx).vortex_expect("DeltaStats should not fail")
    })
}

impl DeltaStats {
    /// Generates delta statistics from a sample of the array.
    pub fn generate(array: &PrimitiveArray, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        match_each_integer_ptype!(array.ptype(), |T| { typed_delta_stats::<T>(array, ctx) })
    }

    /// The bit width a BitPacking layer would choose for the lag-1 residuals.
    pub fn delta_bit_width(&self) -> u8 {
        self.bit_width(&self.delta_widths, self.delta_span)
    }

    /// The bit width a BitPacking layer would choose for the lag-2 residuals.
    pub fn delta_of_delta_bit_width(&self) -> u8 {
        self.bit_width(&self.dod_widths, self.dod_span)
    }

    /// Estimated bits per value to store the lag-1 residuals, exceptions included.
    ///
    /// This excludes the one bit per value that Delta's bases cost.
    pub fn delta_bits_per_value(&self) -> f64 {
        self.bits_per_value(&self.delta_widths, self.delta_span, self.delta_count)
    }

    /// Estimated bits per value to store the lag-2 residuals, exceptions included.
    ///
    /// This excludes the two bits per value that two layers of Delta bases cost.
    pub fn delta_of_delta_bits_per_value(&self) -> f64 {
        self.bits_per_value(&self.dod_widths, self.dod_span, self.dod_count)
    }

    /// Fraction of sampled residuals that are zero, i.e. repeats of the previous value.
    ///
    /// A high value means the array is run-heavy, which RunEnd or Dict usually encode better than
    /// Delta does.
    pub fn zero_fraction(&self) -> f64 {
        if self.delta_count == 0 {
            return 0.0;
        }
        f64::from(self.zero_count) / f64::from(self.delta_count)
    }

    /// Fraction of sampled residuals that decrease, i.e. where the array is not sorted.
    ///
    /// Decreases are not fatal - BitPacking stores them as exceptions - but on an unsigned array
    /// they wrap around, which is why [`DeltaStats::delta_bit_width`] falls back to the full width
    /// once FoR can no longer bridge the span.
    pub fn decreasing_fraction(&self) -> f64 {
        if self.delta_count == 0 {
            return 0.0;
        }
        f64::from(self.decreasing_count) / f64::from(self.delta_count)
    }

    /// Whether every sampled residual is identical, i.e. the array looks like an arithmetic
    /// sequence, which [`SequenceScheme`] encodes more cheaply than Delta.
    ///
    /// [`SequenceScheme`]: super::SequenceScheme
    pub fn is_constant_delta(&self) -> bool {
        self.delta_count > 0 && self.delta_span == 0
    }

    /// Number of lag-1 residuals the stats were computed from.
    pub fn sample_size(&self) -> u32 {
        self.delta_count
    }

    /// Width of the cheaper of the two layers the cascade can put under Delta: BitPacking with
    /// exceptions (the histogram), or FoR when the residuals sit in a narrow band away from zero.
    fn bit_width(&self, widths: &[usize], span: u128) -> u8 {
        let packed = find_best_bit_width(self.ptype, widths)
            .vortex_expect("histogram is sized to the ptype's bit width");
        u8::min(packed, span_bit_width(span, self.ptype))
    }

    /// Bits per value for `widths`, charging exceptions at the same rate BitPacking does.
    fn bits_per_value(&self, widths: &[usize], span: u128, count: u32) -> f64 {
        if count == 0 {
            return f64::from(full_bit_width(self.ptype));
        }
        let packed = find_best_bit_width(self.ptype, widths)
            .vortex_expect("histogram is sized to the ptype's bit width");
        let exceptions: usize = widths.iter().skip(usize::from(packed) + 1).sum();
        let exception_bits = (exceptions * bytes_per_exception(self.ptype) * 8) as f64;
        let packed_bits = f64::from(packed) + exception_bits / f64::from(count);

        // FoR pays no exceptions: it can only span the full range of the residuals.
        f64::min(packed_bits, f64::from(span_bit_width(span, self.ptype)))
    }
}

/// Bits needed to hold `span`, capped at the type's own width.
fn span_bit_width(span: u128, ptype: PType) -> u8 {
    let full = full_bit_width(ptype);
    span.checked_ilog2()
        .map_or(0, |log| u8::try_from(log + 1).unwrap_or(full))
        .min(full)
}

/// Picks `SAMPLE_RUN_COUNT` evenly spread runs of `SAMPLE_RUN_LEN` consecutive values.
///
/// Mirrors the stratified strategy of the compressor's sampler, but keeps the runs separate so
/// that residuals are never taken across a discontinuity.
fn sample_runs(len: usize) -> Vec<(usize, usize)> {
    if len <= SAMPLE_RUN_LEN * SAMPLE_RUN_COUNT {
        return vec![(0, len)];
    }

    let mut rng = StdRng::seed_from_u64(SAMPLE_SEED);
    (0..SAMPLE_RUN_COUNT)
        .map(|partition| {
            let start = len * partition / SAMPLE_RUN_COUNT;
            let stop = len * (partition + 1) / SAMPLE_RUN_COUNT;
            let offset = rng.random_range(start..=(stop - SAMPLE_RUN_LEN));
            (offset, offset + SAMPLE_RUN_LEN)
        })
        .collect()
}

/// Accumulator for one lag of residuals.
struct LagStats {
    /// Histogram of packed bit widths, indexed by bit width.
    widths: Vec<usize>,
    /// Smallest residual seen.
    min: i128,
    /// Largest residual seen.
    max: i128,
    /// Number of residuals seen.
    count: u32,
}

impl LagStats {
    fn new(ptype: PType) -> Self {
        Self {
            widths: vec![0; ptype.bit_width() + 1],
            min: i128::MAX,
            max: i128::MIN,
            count: 0,
        }
    }

    /// Records one residual, bucketing it by the width BitPacking would need for it.
    fn push(&mut self, residual: i128, ptype: PType) {
        self.min = self.min.min(residual);
        self.max = self.max.max(residual);
        self.count += 1;
        self.widths[usize::from(packed_width(residual, ptype))] += 1;
    }

    fn span(&self) -> u128 {
        if self.count == 0 {
            return 0;
        }
        self.max.abs_diff(self.min)
    }
}

/// Width BitPacking needs for a single residual, in the domain the cascade sees it in.
///
/// A signed array's residuals reach BitPacking through ZigZag, which costs a sign bit. An
/// unsigned array has no such path: Vortex subtracts in the unsigned domain, so a decrease wraps
/// to a near-maximal value and can only be stored as an exception.
fn packed_width(residual: i128, ptype: PType) -> u8 {
    let full = full_bit_width(ptype);
    if residual < 0 && !ptype.is_signed_int() {
        return full;
    }
    let magnitude = residual.unsigned_abs();
    let bits = magnitude
        .checked_ilog2()
        .map_or(0, |log| u8::try_from(log + 1).unwrap_or(full));
    let sign_bit = u8::from(ptype.is_signed_int() && bits > 0);
    bits.saturating_add(sign_bit).min(full)
}

/// Computes delta stats for a concrete integer type.
fn typed_delta_stats<T>(array: &PrimitiveArray, ctx: &mut ExecutionCtx) -> VortexResult<DeltaStats>
where
    T: IntegerPType,
    i128: From<T>,
{
    let ptype = array.ptype();
    let mut delta = LagStats::new(ptype);
    let mut dod = LagStats::new(ptype);
    let mut zero_count = 0;
    let mut decreasing_count = 0;

    let len = array.len();
    let buffer = array.to_buffer::<T>();
    let validity = array.as_ref().validity()?.execute_mask(len, ctx)?;
    if validity.all_false() {
        return Ok(finish(ptype, delta, dod, zero_count, decreasing_count));
    }
    // Materialize the "are there any nulls" decision once, rather than per sampled value.
    let check_validity = !validity.all_true();

    for (start, stop) in sample_runs(len) {
        // Nulls are fill-forwarded by Delta, so the residual sequence is over valid values only.
        let mut previous: Option<i128> = None;
        let mut previous_delta: Option<i128> = None;

        for idx in start..stop {
            if check_validity && !validity.value(idx) {
                continue;
            }
            let value = i128::from(buffer[idx]);
            let Some(prior) = previous.replace(value) else {
                continue;
            };

            let residual = value - prior;
            delta.push(residual, ptype);
            zero_count += u32::from(residual == 0);
            decreasing_count += u32::from(residual < 0);

            if let Some(prior_delta) = previous_delta.replace(residual) {
                dod.push(residual - prior_delta, ptype);
            }
        }
    }

    Ok(finish(ptype, delta, dod, zero_count, decreasing_count))
}

/// Folds the two lag accumulators into a [`DeltaStats`].
fn finish(
    ptype: PType,
    delta: LagStats,
    dod: LagStats,
    zero_count: u32,
    decreasing_count: u32,
) -> DeltaStats {
    DeltaStats {
        ptype,
        delta_count: delta.count,
        dod_count: dod.count,
        zero_count,
        decreasing_count,
        delta_span: delta.span(),
        dod_span: dod.span(),
        delta_widths: delta.widths,
        dod_widths: dod.widths,
    }
}

#[cfg(test)]
mod tests {
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rstest::rstest;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;

    use super::DeltaStats;

    fn stats<T: vortex_array::dtype::NativePType>(values: &[T]) -> VortexResult<DeltaStats> {
        let array = PrimitiveArray::new(Buffer::copy_from(values), Validity::NonNullable);
        DeltaStats::generate(&array, &mut array_session().create_execution_ctx())
    }

    #[test]
    fn constant_delta_is_a_sequence() -> VortexResult<()> {
        let values: Vec<i64> = (0..8192).map(|i| 1_000_000 + i * 7).collect();
        let stats = stats(&values)?;
        assert!(stats.is_constant_delta());
        assert_eq!(stats.delta_bit_width(), 0);
        Ok(())
    }

    /// A wide-ranging but smoothly increasing column: the values need the full width, the
    /// residuals only a handful of bits.
    #[test]
    fn smooth_ramp_has_narrow_residuals() -> VortexResult<()> {
        let mut rng = StdRng::seed_from_u64(7);
        let mut value = 1_600_000_000_000i64;
        let values: Vec<i64> = (0..1 << 16)
            .map(|_| {
                value += 1 + i64::from(rng.random::<u8>() % 8);
                value
            })
            .collect();

        let stats = stats(&values)?;
        assert!(!stats.is_constant_delta());
        assert!(
            stats.delta_bit_width() <= 6,
            "residuals need at most 3 magnitude bits plus a sign bit, got {}",
            stats.delta_bit_width()
        );
        assert_eq!(stats.decreasing_fraction(), 0.0);
        assert_eq!(stats.sample_size() as usize, super::SAMPLE_RUN_COUNT * 63);
        Ok(())
    }

    /// Second differencing doubles the span of jittery residuals, so delta-of-delta cannot pay for
    /// the extra layer of bases it costs - the measurement that keeps it a statistic rather than a
    /// scheme. On smoothly drifting data it can narrow the residuals by about a bit, which is why
    /// the bound here is the cost of that extra layer rather than zero.
    #[test]
    fn delta_of_delta_never_beats_delta_by_more_than_its_bases() -> VortexResult<()> {
        let mut rng = StdRng::seed_from_u64(11);
        let mut value = 0i64;
        let values: Vec<i64> = (0..1 << 16)
            .map(|_| {
                value += 60_000 + i64::from(rng.random::<u8>());
                value
            })
            .collect();

        let stats = stats(&values)?;
        assert!(
            stats.delta_of_delta_bits_per_value() + 1.0 >= stats.delta_bits_per_value(),
            "delta-of-delta ({}) beat delta ({}) by more than the extra layer of bases",
            stats.delta_of_delta_bits_per_value(),
            stats.delta_bits_per_value(),
        );
        Ok(())
    }

    /// Random values have no delta structure: residuals are as wide as the values themselves.
    #[test]
    fn random_values_have_full_width_residuals() -> VortexResult<()> {
        let mut rng = StdRng::seed_from_u64(13);
        let values: Vec<i32> = (0..1 << 16).map(|_| rng.random::<i32>()).collect();
        let stats = stats(&values)?;
        assert_eq!(stats.delta_bit_width(), 32);
        Ok(())
    }

    /// Vortex subtracts in the unsigned domain, so a decreasing unsigned column wraps around and
    /// Delta cannot help. The signed column with the same shape keeps narrow residuals.
    #[rstest]
    #[case::unsigned(false, 32)]
    #[case::signed(true, 5)]
    fn decreasing_columns(#[case] signed: bool, #[case] max_width: u8) -> VortexResult<()> {
        let stats = if signed {
            let values: Vec<i32> = (0..1 << 16).map(|i| 4_000_000i32 - i).collect();
            stats(&values)?
        } else {
            let values: Vec<u32> = (0..1 << 16).map(|i| 4_000_000u32 - i).collect();
            stats(&values)?
        };
        assert!(
            stats.delta_bit_width() <= max_width,
            "expected at most {max_width} bits, got {}",
            stats.delta_bit_width()
        );
        assert_eq!(stats.decreasing_fraction(), 1.0);
        Ok(())
    }

    /// Nulls are fill-forwarded by Delta, so the residual sequence skips them rather than
    /// differencing against whatever the null slot happens to hold.
    #[test]
    fn nulls_are_skipped() -> VortexResult<()> {
        let values: Vec<i64> = (0..1 << 16)
            .map(|i| if i % 4 == 0 { i64::MAX } else { 1_000_000 + i })
            .collect();
        let validity = Validity::from_iter((0..values.len()).map(|i| i % 4 != 0));
        let array = PrimitiveArray::new(Buffer::copy_from(&values), validity);
        let stats = DeltaStats::generate(&array, &mut array_session().create_execution_ctx())?;
        assert!(
            stats.delta_bit_width() <= 4,
            "residuals should stay small across nulls, got {}",
            stats.delta_bit_width()
        );
        Ok(())
    }

    /// The whole point of sampling: the stats are the same whether the array is 64 Ki or 16 Mi
    /// values long, and cost the same to compute.
    #[test]
    fn sample_size_is_independent_of_array_length() -> VortexResult<()> {
        let short: Vec<i64> = (0..1 << 16).map(|i| i * 3 + (i % 5)).collect();
        let long: Vec<i64> = (0..1 << 22).map(|i| i * 3 + (i % 5)).collect();
        assert_eq!(stats(&short)?.sample_size(), stats(&long)?.sample_size());
        Ok(())
    }
}
