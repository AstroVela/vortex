// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! FastLanes Delta integer encoding.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::PrimitiveArray;
use vortex_compressor::builtins::BinaryDictScheme;
use vortex_compressor::builtins::FloatDictScheme;
use vortex_compressor::builtins::IntDictScheme;
use vortex_compressor::builtins::StringDictScheme;
use vortex_compressor::scheme::AncestorExclusion;
use vortex_compressor::scheme::ChildSelection;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DescendantExclusion;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexResult;
use vortex_fastlanes::Delta;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;
use crate::schemes::integer::delta_stats::delta_stats;

/// FastLanes Delta encoding for smooth / near-monotone integers.
///
/// Delta replaces each value with its difference from the preceding value, so a later cascade
/// layer (FoR / ZigZag / BitPacking) packs the smaller residuals. It only pays off when those
/// residuals span meaningfully fewer bits than the values themselves.
///
/// Selection is driven by [`DeltaStats`], a sampled statistic: the residual widths are a local
/// property of the data, so a few short contiguous runs describe them as accurately as the whole
/// array does, at a cost that does not grow with array length.
///
/// The minimum penalized compression ratio required for Delta to be selected is configurable via
/// [`DeltaScheme::new`]; [`DeltaScheme::default`] uses a ratio of `1.05`.
///
/// There is deliberately no delta-of-delta scheme: the layer below Delta is FoR (or ZigZag) plus
/// BitPacking, which already subtracts the mean rate, so a second Delta layer only doubles the
/// span of what is left while adding another bit per value of bases. Across 1966 real
/// (column, block) pairs it never once came out ahead - see `scripts/delta-analysis/README.md`.
///
/// [`DeltaStats`]: super::DeltaStats
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct DeltaScheme {
    min_ratio: f64,
}

impl DeltaScheme {
    /// Creates a Delta scheme requiring `min_ratio` after the delta penalty before it wins.
    ///
    /// Pass a higher ratio to make Delta more conservative, or a lower one to select it more
    /// eagerly. [`DeltaScheme::default`] uses a ratio of `1.05`.
    pub const fn new(min_ratio: f64) -> Self {
        Self { min_ratio }
    }
}

impl Default for DeltaScheme {
    fn default() -> Self {
        Self::new(1.05)
    }
}

/// Multiplicative penalty applied to Delta's estimated compression ratio.
///
/// Unlike FoR/BitPacking, Delta breaks random access and adds a prefix-sum decode pass, so we
/// require it to be slightly smaller than the best alternative rather than picking it for a
/// single-bit gain. The structural costs Delta pays - the bases and the residual sign bit - are
/// charged directly to the estimate instead, so this factor is only the "random access tax".
const DELTA_PENALTY: f64 = 0.98;

/// Bits per value that Delta's bases cost.
///
/// FastLanes stores `1024 / T` bases of `T` bits for every 1024 values, which is exactly one bit
/// per value, whatever the type. The bases are themselves compressed by the cascade, so this is
/// an upper bound.
const BASE_BITS_PER_VALUE: f64 = 1.0;

/// Minimum length before Delta is worth considering (one FastLanes chunk).
const MIN_DELTA_LEN: usize = 1024;

impl Scheme for DeltaScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.delta"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![Delta.id()]
    }

    fn num_children(&self) -> usize {
        2
    }

    /// Delta-encode the data at most once per path: exclude Delta from the subtrees of both the
    /// bases and the deltas children so we never delta-encode data that was already delta-encoded.
    fn descendant_exclusions(&self) -> Vec<DescendantExclusion> {
        vec![DescendantExclusion {
            excluded: self.id(),
            children: ChildSelection::All,
        }]
    }

    /// Delta over dictionary codes just adds indirection: codes are compact integers with no
    /// monotone structure, so (like FoR/Sequence) skip the codes child.
    fn ancestor_exclusions(&self) -> Vec<AncestorExclusion> {
        vec![
            AncestorExclusion {
                ancestor: IntDictScheme.id(),
                children: ChildSelection::One(1),
            },
            AncestorExclusion {
                ancestor: FloatDictScheme.id(),
                children: ChildSelection::One(1),
            },
            AncestorExclusion {
                ancestor: StringDictScheme.id(),
                children: ChildSelection::One(1),
            },
            AncestorExclusion {
                ancestor: BinaryDictScheme.id(),
                children: ChildSelection::One(1),
            },
        ]
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        // Delta only pays off if a later cascade layer (FoR/BitPacking) packs the residuals.
        if compress_ctx.finished_cascading() {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }
        // Too short to transpose into FastLanes chunks meaningfully.
        if data.array_len() < MIN_DELTA_LEN {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        let primitive = data.array_as_primitive();
        let full_width = primitive.ptype().bit_width() as f64;

        let stats = delta_stats(data, exec_ctx);

        // Constant deltas are an arithmetic sequence, which SequenceScheme stores in O(1).
        if stats.is_constant_delta() {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        // The residuals are packed by the layer below, and the bases cost one bit per value.
        let cost = stats.delta_bits_per_value() + BASE_BITS_PER_VALUE;
        let ratio = full_width / cost * DELTA_PENALTY;
        if ratio <= self.min_ratio {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }
        CompressionEstimate::Verdict(EstimateVerdict::Ratio(ratio))
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let primitive = data.array().clone().execute::<PrimitiveArray>(exec_ctx)?;
        let len = primitive.len();
        let (bases, deltas) = vortex_fastlanes::delta_compress(&primitive, exec_ctx)?;

        let compressed_bases = compressor.compress_child(
            &bases.into_array(),
            &compress_ctx,
            self.id(),
            0,
            exec_ctx,
        )?;
        let compressed_deltas = compressor.compress_child(
            &deltas.into_array(),
            &compress_ctx,
            self.id(),
            1,
            exec_ctx,
        )?;

        Delta::try_new(compressed_bases, compressed_deltas, 0, len).map(IntoArray::into_array)
    }
}
