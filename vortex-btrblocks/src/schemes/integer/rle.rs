// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Run-length integer encoding and shared RLE compression helpers.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_compressor::scheme::AncestorExclusion;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::DescendantExclusion;
use vortex_compressor::scheme::EstimateVerdict;
#[cfg(feature = "unstable_encodings")]
use vortex_compressor::scheme::SchemeId;
use vortex_error::VortexResult;
#[cfg(feature = "unstable_encodings")]
use vortex_fastlanes::Delta;
use vortex_fastlanes::RLE;
use vortex_fastlanes::RLEArrayExt;
use vortex_fastlanes::RLEArraySlotsExt;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;
use crate::schemes::rle_ancestor_exclusions;
use crate::schemes::rle_descendant_exclusions;

/// RLE scheme for integer arrays.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntRLEScheme;

/// Bits per element that RLE spends on positional information.
///
/// RLE keeps one chunk-local run index per element. Within a 1024-element FastLanes chunk those
/// indices are monotone and grow by at most one per element, so Delta turns them into a 0/1
/// bitmap — one bit per element — plus one rank sample per FastLanes lane. Measured over
/// ClickBench the two together come to ~1.35 bits per element, so 1.5 is a safe upper bound.
const RLE_POSITION_BITS_PER_ELEMENT: f64 = 1.5;

/// Returns `true` when RLE has a chance of beating a plain bit-packed layout.
///
/// RLE replaces `value_count` values with one dictionary entry per run plus a positional index
/// per element, so it costs roughly
///
/// ```text
/// bit_width / average_run_length + RLE_POSITION_BITS_PER_ELEMENT
/// ```
///
/// bits per element, against bit-packing's flat `bit_width`. This rejects the two cases where RLE
/// cannot win no matter how the cascade turns out: data with no runs to exploit, and data whose
/// values are already narrower than RLE's own positional index — a boolean-like column costs one
/// bit per element bit-packed, which run-length encoding can never undercut.
///
/// Everything that clears this bound is handed to the sampled estimate, which measures the real
/// cascaded size and compares it against the other schemes. Deliberately keep the bound loose:
/// long runs are not a precondition for RLE, because the positional index is paid per element
/// rather than per run. A 64-bit column with an average run of only 1.2 elements still packs into
/// well under half its bit-packed size.
pub(crate) fn rle_can_pay_for_itself(value_count: u32, run_count: u32, bit_width: u32) -> bool {
    if run_count == 0 || run_count >= value_count {
        return false;
    }
    let bit_width = f64::from(bit_width);
    let average_run_length = f64::from(value_count) / f64::from(run_count);
    bit_width / average_run_length + RLE_POSITION_BITS_PER_ELEMENT < bit_width
}

/// Shared compression logic for RLE schemes.
pub(crate) fn rle_compress(
    scheme: &dyn Scheme,
    compressor: &CascadingCompressor,
    data: &ArrayAndStats,
    compress_ctx: CompressorContext,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let rle_array = RLE::encode(data.array_as_primitive(), exec_ctx)?;

    let rle_values_primitive = rle_array
        .values()
        .clone()
        .execute::<PrimitiveArray>(exec_ctx)?;
    let compressed_values = compressor.compress_child(
        &rle_values_primitive.into_array(),
        &compress_ctx,
        scheme.id(),
        0,
        exec_ctx,
    )?;

    // Delta is an unstable encoding, once we deem it stable we can switch over to this always.
    //
    // Note the indices are deliberately *not* narrowed first. FastLanes Delta emits one base per
    // lane per 1024-element chunk and the lane count is `1024 / bit_width`, so narrowing u16
    // indices to u8 would double the bases (64 -> 128 per chunk) while leaving the delta payload
    // byte-identical, because bit-packing a chunk to `w` bits costs `1024 * w` bits whatever the
    // storage type. Worse, an index array that needs a full 8 bits is out of bit-packing's reach
    // once it is stored as u8: `bitpack_encode` rejects a width equal to the storage width, so
    // the bases stay unpacked.
    //
    // That reasoning assumes the cascade does pack the delta output. When it cannot — a cascade
    // configured without BitPacking, or an array too short for it to pay off — the wider storage
    // simply costs twice as much per element, so fall back to the narrowed encoding.
    #[cfg(feature = "unstable_encodings")]
    let compressed_indices = {
        let rle_indices_primitive = rle_array
            .indices()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?;
        let compressed = try_compress_delta(
            compressor,
            rle_indices_primitive.as_ref(),
            &compress_ctx,
            scheme.id(),
            1,
            exec_ctx,
        )?;

        if compressed.nbytes() < rle_indices_primitive.as_ref().nbytes() {
            compressed
        } else {
            try_compress_delta(
                compressor,
                &rle_indices_primitive.narrow(exec_ctx)?.into_array(),
                &compress_ctx,
                scheme.id(),
                1,
                exec_ctx,
            )?
        }
    };

    #[cfg(not(feature = "unstable_encodings"))]
    let compressed_indices = {
        let rle_indices_primitive = rle_array
            .indices()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?
            .narrow(exec_ctx)?;
        compressor.compress_child(
            &rle_indices_primitive.into_array(),
            &compress_ctx,
            scheme.id(),
            1,
            exec_ctx,
        )?
    };

    let rle_offsets_primitive = rle_array
        .values_idx_offsets()
        .clone()
        .execute::<PrimitiveArray>(exec_ctx)?
        .narrow(exec_ctx)?;
    let compressed_offsets = compressor.compress_child(
        &rle_offsets_primitive.into_array(),
        &compress_ctx,
        scheme.id(),
        2,
        exec_ctx,
    )?;

    // SAFETY: Recursive compression doesn't affect the invariants.
    unsafe {
        Ok(RLE::new_unchecked(
            compressed_values,
            compressed_indices,
            compressed_offsets,
            rle_array.offset(),
            rle_array.len(),
        )
        .into_array())
    }
}

#[cfg(feature = "unstable_encodings")]
pub(crate) fn try_compress_delta(
    compressor: &CascadingCompressor,
    child: &ArrayRef,
    parent_ctx: &CompressorContext,
    parent_id: SchemeId,
    child_index: usize,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let child_primitive = child.clone().execute::<PrimitiveArray>(exec_ctx)?;
    let (bases, deltas) = vortex_fastlanes::delta_compress(&child_primitive, exec_ctx)?;

    let compressed_bases = compressor.compress_child(
        &bases.into_array(),
        parent_ctx,
        parent_id,
        child_index,
        exec_ctx,
    )?;
    let compressed_deltas = compressor.compress_child(
        &deltas.into_array(),
        parent_ctx,
        parent_id,
        child_index,
        exec_ctx,
    )?;

    Delta::try_new(compressed_bases, compressed_deltas, 0, child.len()).map(IntoArray::into_array)
}

impl Scheme for IntRLEScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.rle"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![RLE.id()]
    }

    /// Children: values=0, indices=1, offsets=2.
    fn num_children(&self) -> usize {
        3
    }

    fn descendant_exclusions(&self) -> Vec<DescendantExclusion> {
        rle_descendant_exclusions()
    }

    fn ancestor_exclusions(&self) -> Vec<AncestorExclusion> {
        rle_ancestor_exclusions()
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        // RLE is only useful when we cascade it with another encoding.
        if compress_ctx.finished_cascading() {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }
        let stats = data.integer_stats(exec_ctx);
        // Compare against the width bit-packing would actually reach, which is the FoR span
        // rather than the declared type width.
        let bit_width = u64::BITS - stats.erased().max_minus_min().leading_zeros();
        if !rle_can_pay_for_itself(stats.value_count(), stats.run_count(), bit_width) {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        CompressionEstimate::Deferred(DeferredEstimate::Sample)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        rle_compress(self, compressor, data, compress_ctx, exec_ctx)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::rle_can_pay_for_itself;

    #[rstest]
    // No runs at all: every element is its own run, so the index array is pure overhead.
    #[case::all_distinct(1000, 1000, 64, false)]
    #[case::more_runs_than_values(1000, 1001, 64, false)]
    #[case::no_runs_counted(1000, 0, 64, false)]
    // Boolean-like columns bit-pack to one bit per element, which RLE's index can never undercut.
    #[case::one_bit_long_runs(1000, 10, 1, false)]
    // A wide column repays the index even when runs are barely longer than a single element.
    #[case::wide_short_runs(1000, 833, 64, true)]
    #[case::wide_pairs(1000, 500, 64, true)]
    // Narrow columns need proportionally longer runs before the index pays for itself.
    #[case::four_bit_pairs(1000, 500, 4, true)]
    #[case::four_bit_short_runs(1000, 800, 4, false)]
    fn can_pay_for_itself(
        #[case] value_count: u32,
        #[case] run_count: u32,
        #[case] bit_width: u32,
        #[case] expected: bool,
    ) {
        assert_eq!(
            rle_can_pay_for_itself(value_count, run_count, bit_width),
            expected
        );
    }

    /// The old fixed `average_run_length >= 4` gate rounded a 1.9-element average down to 1 and
    /// dropped it. Runs that short still pay off on wide values.
    #[test]
    fn short_runs_on_wide_values_are_considered() {
        assert!(rle_can_pay_for_itself(524_288, 264_792, 30));
        assert!(rle_can_pay_for_itself(524_288, 257_000, 64));
    }
}
