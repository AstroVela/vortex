// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compression ratio estimation types and sampling-based estimation.

use std::fmt;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_error::VortexResult;

use crate::CascadingCompressor;
use crate::cost::Cost;
use crate::ctx::CompressorContext;
use crate::sample::SAMPLE_SIZE;
use crate::sample::sample;
use crate::sample::sample_count_approx_one_percent;
use crate::scheme::Scheme;
use crate::scheme::SchemeExt;
use crate::stats::ArrayAndStats;
use crate::trace;

/// Closure type for [`DeferredEstimate::Callback`].
///
/// The compressor calls this with the same arguments it would pass to sampling, plus a safe best
/// ratio when using the built-in default size model. Models installed through
/// [`CascadingCompressor::with_cost_model`] receive `None` because they may prefer a lower-ratio
/// candidate. The closure must resolve directly to a terminal [`EstimateVerdict`].
///
/// `best_ratio` is an early-exit hint. If your scheme's maximum achievable compression ratio is
/// not strictly greater than it, you should return [`EstimateVerdict::Skip`]. Returning an equal
/// ratio is permitted but will lose to the prior best due to strict tie-breaking in the selector.
/// Use the threshold only to avoid work, never to perform additional work.
#[rustfmt::skip]
pub type EstimateFn = dyn FnOnce(
        &CascadingCompressor,
        &ArrayAndStats,
        Option<f64>,
        CompressorContext,
        &mut ExecutionCtx,
    ) -> VortexResult<EstimateVerdict>
    + Send
    + Sync;

/// The result of a [`Scheme`]'s compression ratio estimation.
///
/// This type is returned by [`Scheme::expected_compression_ratio`] to tell the compressor how
/// promising this scheme is for a given array without performing any expensive work.
///
/// [`CompressionEstimate::Verdict`] means the scheme already knows the terminal answer.
/// [`CompressionEstimate::Deferred`] means the compressor must do extra work before the scheme can
/// produce a terminal answer.
#[derive(Debug)]
pub enum CompressionEstimate {
    /// The scheme already knows the terminal estimation verdict.
    Verdict(EstimateVerdict),

    /// The compressor must perform deferred work to resolve the terminal estimation verdict.
    Deferred(DeferredEstimate),
}

/// The terminal answer to a compression estimate request.
#[derive(Debug)]
pub enum EstimateVerdict {
    /// Do not use this scheme for this array.
    Skip,

    /// Always use this scheme, as it is definitively the best choice.
    ///
    /// Some examples include decimal byte parts and temporal decomposition.
    ///
    /// The compressor will select this scheme immediately without evaluating further candidates.
    /// Schemes that return `AlwaysUse` must be mutually exclusive per canonical type (enforced by
    /// [`Scheme::matches`]), otherwise the winner depends silently on registration order.
    ///
    /// [`Scheme::matches`]: crate::scheme::Scheme::matches
    AlwaysUse,

    /// The estimated compression ratio, interpreted by the configured cost model.
    ///
    /// The default size model requires a finite, non-subnormal ratio greater than `1.0` to beat
    /// the canonical encoding. Other models may price the same signal differently.
    Ratio(f64),
}

/// Deferred work that can resolve to a terminal [`EstimateVerdict`].
pub enum DeferredEstimate {
    /// The scheme cannot cheaply estimate its ratio, so the compressor should compress a small
    /// sample to determine effectiveness.
    Sample,

    /// A fallible estimation requiring a custom expensive computation.
    ///
    /// Use this only when the scheme needs to perform trial encoding or other costly checks to
    /// determine its compression ratio. The callback returns an [`EstimateVerdict`] directly, so
    /// it cannot request more sampling or another deferred callback.
    ///
    /// The compressor evaluates all immediate [`CompressionEstimate::Verdict`] results before
    /// invoking any deferred callback. With the default size model, it passes the best ratio
    /// observed so far to the callback. This lets the callback return [`EstimateVerdict::Skip`]
    /// without performing expensive work when its maximum achievable ratio cannot beat the
    /// current best. Custom models receive no ratio threshold. See [`EstimateFn`] for the full
    /// contract.
    Callback(Box<EstimateFn>),
}

/// Winner estimate carried from scheme selection into result tracing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum WinnerEstimate {
    /// The scheme must be used immediately.
    AlwaysUse,
    /// The scheme won after being priced by the compressor's cost model.
    Priced {
        /// The winning candidate's estimated compression ratio, if one exists.
        estimated_ratio: Option<f64>,
        /// The winning candidate's cost under the compressor's cost model.
        cost: Cost,
    },
}

impl WinnerEstimate {
    /// Returns the traceable numeric ratio for the winning estimate.
    pub(super) fn trace_ratio(self) -> Option<f64> {
        match self {
            Self::AlwaysUse => None,
            Self::Priced {
                estimated_ratio, ..
            } => estimated_ratio,
        }
    }

    /// Returns the traceable cost for the winning estimate.
    pub(super) fn trace_cost(self) -> Option<f64> {
        match self {
            Self::AlwaysUse => None,
            Self::Priced { cost, .. } => Some(cost.value()),
        }
    }
}

/// A sampling-based estimate: the optional measured ratio and compressed sample array.
pub(crate) struct SampledEstimate {
    /// The compression ratio measured on the sample, or `None` for a zero-byte output.
    pub(crate) estimated_ratio: Option<f64>,

    /// The compressed sample array. Its encoding tree is the best available prediction of
    /// the full-array encoding tree.
    pub(crate) sampled: ArrayRef,
}

/// Estimates compression ratio by compressing a ~1% sample of the data.
///
/// Creates a new [`ArrayAndStats`] for the sample so that stats are generated from the sample, not
/// the full array.
///
/// Returns the compressed sample alongside its measured ratio so the cost model can inspect its
/// encoding tree while pricing the candidate.
///
/// # Errors
///
/// Returns an error if sample compression fails.
pub(super) fn estimate_compression_ratio_with_sampling<S: Scheme + ?Sized>(
    compressor: &CascadingCompressor,
    scheme: &S,
    array: &ArrayRef,
    compress_ctx: CompressorContext,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<SampledEstimate> {
    let sample_array = if compress_ctx.is_sample() {
        array.clone()
    } else {
        let sample_count = sample_count_approx_one_percent(array.len());
        // `ArrayAndStats` expects a canonical array (so that it can easily compute lazy stats).
        let canonical: Canonical = sample(array, SAMPLE_SIZE, sample_count).execute(exec_ctx)?;
        canonical.into_array()
    };

    let sample_data = ArrayAndStats::new(sample_array, scheme.stats_options());
    let error_ctx = trace::enabled_error_context(&compress_ctx);
    let sample_ctx = compress_ctx.with_sampling();

    let compressed = match scheme.compress(compressor, &sample_data, sample_ctx, exec_ctx) {
        Ok(compressed) => compressed,
        Err(err) => {
            trace::sample_compress_failed(scheme.id(), error_ctx.as_ref(), &err);
            return Err(err);
        }
    };

    let after = compressed.nbytes();
    let before = sample_data.array().nbytes();

    let estimated_ratio = (after != 0).then(|| before as f64 / after as f64);

    if estimated_ratio.is_none() {
        trace::zero_byte_sample_result(scheme.id(), before);
    }

    Ok(SampledEstimate {
        estimated_ratio,
        sampled: compressed,
    })
}

impl fmt::Debug for DeferredEstimate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeferredEstimate::Sample => write!(f, "Sample"),
            DeferredEstimate::Callback(_) => write!(f, "Callback(..)"),
        }
    }
}
