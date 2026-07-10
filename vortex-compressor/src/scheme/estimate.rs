// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compression ratio estimation types returned by [`Scheme::expected_compression_ratio`].
//!
//! [`Scheme::expected_compression_ratio`]: crate::scheme::Scheme::expected_compression_ratio

use std::fmt;

use vortex_array::ExecutionCtx;
use vortex_error::VortexResult;

use crate::CascadingCompressor;
use crate::scheme::CompressorContext;
use crate::stats::ArrayAndStats;

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
///
/// [`Scheme`]: crate::scheme::Scheme
/// [`Scheme::expected_compression_ratio`]: crate::scheme::Scheme::expected_compression_ratio
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

impl fmt::Debug for DeferredEstimate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeferredEstimate::Sample => write!(f, "Sample"),
            DeferredEstimate::Callback(_) => write!(f, "Callback(..)"),
        }
    }
}
