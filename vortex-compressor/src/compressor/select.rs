// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scheme selection: pricing candidates with the cost model and choosing the winner.

use vortex_array::ExecutionCtx;
use vortex_error::VortexResult;

use super::ROOT_SCHEME_ID;
use super::sample::estimate_compression_ratio_with_sampling;
use crate::CascadingCompressor;
use crate::cost::Candidate;
use crate::cost::Cost;
use crate::scheme::CompressionEstimate;
use crate::scheme::CompressorContext;
use crate::scheme::DeferredEstimate;
use crate::scheme::EstimateVerdict;
use crate::scheme::Scheme;
use crate::scheme::SchemeExt;
use crate::stats::ArrayAndStats;
use crate::trace;

/// The selector state retained for the current winner.
///
/// Unlike [`Candidate`], this owns no sampled array or cascade history.
struct BestCandidate {
    /// The currently selected scheme.
    scheme: &'static dyn Scheme,
    /// The scheme's estimated ratio, retained for deferred callback thresholds and tracing.
    estimated_ratio: Option<f64>,
    /// The model-defined cost used for later comparisons and tracing.
    cost: Cost,
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

impl CascadingCompressor {
    /// Calls [`expected_compression_ratio`] on each candidate and returns the winning scheme along
    /// with its resolved winner estimate, or `None` if no scheme beats the canonical encoding.
    ///
    /// Each scored candidate is priced by the compressor's [`CostModel`]; the winner is the
    /// candidate with the minimum cost, and only candidates pricing strictly below
    /// [`CostModel::canonical_cost`] are eligible.
    ///
    /// Selection runs in two passes. Pass 1 evaluates every immediate
    /// [`CompressionEstimate::Verdict`] and tracks the running best. [`Scheme`]s returning
    /// [`CompressionEstimate::Deferred`] are stashed for pass 2 so that we do not make any
    /// expensive computations if we don't have to.
    ///
    /// Pass 2 evaluates the deferred work and, with the default size model, passes the current
    /// best ratio to each [`DeferredEstimate::Callback`] as an early-exit hint. Custom models
    /// receive no such hint because they may prefer a lower-ratio candidate.
    ///
    /// Ties are broken by registration order within each pass (displacing the running best
    /// requires a strictly lower cost).
    ///
    /// [`expected_compression_ratio`]: Scheme::expected_compression_ratio
    pub(super) fn choose_best_scheme(
        &self,
        schemes: &[&'static dyn Scheme],
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<(&'static dyn Scheme, WinnerEstimate)>> {
        let mut best: Option<BestCandidate> = None;
        let mut deferred: Vec<(&'static dyn Scheme, DeferredEstimate)> = Vec::new();

        let canonical_cost = self
            .cost_model
            .canonical_cost(data, data.array().len() as u64);

        let cascade = compress_ctx.cascade_history();

        // Pass 1: evaluate every immediate verdict. Stash deferred work for pass 2.
        {
            let _verdict_pass = trace::verdict_pass_span().entered();
            for &scheme in schemes {
                match scheme.expected_compression_ratio(data, compress_ctx.clone(), exec_ctx) {
                    CompressionEstimate::Verdict(EstimateVerdict::Skip) => {}
                    CompressionEstimate::Verdict(EstimateVerdict::AlwaysUse) => {
                        return Ok(Some((scheme, WinnerEstimate::AlwaysUse)));
                    }
                    CompressionEstimate::Verdict(EstimateVerdict::Ratio(ratio)) => {
                        let estimated_ratio = Some(ratio);
                        let candidate =
                            Candidate::new(scheme, estimated_ratio, data, None, cascade);

                        if let Some(cost) =
                            self.price(&candidate, canonical_cost, best.as_ref().map(|b| b.cost))
                        {
                            best = Some(BestCandidate {
                                scheme,
                                estimated_ratio,
                                cost,
                            });
                        }
                    }
                    CompressionEstimate::Deferred(deferred_estimate) => {
                        deferred.push((scheme, deferred_estimate));
                    }
                }
            }
        }

        // Pass 2: run deferred work. Under size ordering, callbacks receive the current best as a
        // ratio threshold so they can short-circuit with `Skip` when they cannot beat it. Custom
        // models receive no threshold because their ordering may differ.
        for (scheme, deferred_estimate) in deferred {
            let _span = trace::scheme_eval_span(scheme.id()).entered();
            let threshold = if self.use_ratio_threshold {
                best.as_ref()
                    .and_then(|candidate| candidate.estimated_ratio)
            } else {
                None
            };
            match deferred_estimate {
                DeferredEstimate::Sample => {
                    let estimate = estimate_compression_ratio_with_sampling(
                        self,
                        scheme,
                        data.array(),
                        compress_ctx.clone(),
                        exec_ctx,
                    )?;
                    let candidate = Candidate::new(
                        scheme,
                        estimate.estimated_ratio,
                        data,
                        Some(&estimate.sampled),
                        cascade,
                    );

                    if let Some(cost) =
                        self.price(&candidate, canonical_cost, best.as_ref().map(|b| b.cost))
                    {
                        best = Some(BestCandidate {
                            scheme,
                            estimated_ratio: estimate.estimated_ratio,
                            cost,
                        });
                    }
                }
                DeferredEstimate::Callback(callback) => {
                    match callback(self, data, threshold, compress_ctx.clone(), exec_ctx)? {
                        EstimateVerdict::Skip => {}
                        EstimateVerdict::AlwaysUse => {
                            return Ok(Some((scheme, WinnerEstimate::AlwaysUse)));
                        }
                        EstimateVerdict::Ratio(ratio) => {
                            let estimated_ratio = Some(ratio);
                            let candidate =
                                Candidate::new(scheme, estimated_ratio, data, None, cascade);

                            if let Some(cost) = self.price(
                                &candidate,
                                canonical_cost,
                                best.as_ref().map(|b| b.cost),
                            ) {
                                best = Some(BestCandidate {
                                    scheme,
                                    estimated_ratio,
                                    cost,
                                });
                            }
                        }
                    }
                }
            }
        }

        Ok(best.map(|candidate| {
            (
                candidate.scheme,
                WinnerEstimate::Priced {
                    estimated_ratio: candidate.estimated_ratio,
                    cost: candidate.cost,
                },
            )
        }))
    }

    /// Prices a candidate and returns its cost iff it becomes the new best: the model must
    /// price it (`Some`), and the cost must be strictly below both the canonical baseline and
    /// the best so far. Strict `<` preserves registration-order tie-breaking.
    fn price(
        &self,
        candidate: &Candidate<'_>,
        canonical_cost: Cost,
        best_cost: Option<Cost>,
    ) -> Option<Cost> {
        let cost = self.cost_model.cost(candidate)?;
        (cost < canonical_cost && best_cost.is_none_or(|best_cost| cost < best_cost))
            .then_some(cost)
    }

    // TODO(connor): Lots of room for optimization here.
    /// Returns `true` if the candidate scheme should be excluded based on the cascade history and
    /// exclusion rules.
    pub(super) fn is_excluded(&self, candidate: &dyn Scheme, ctx: &CompressorContext) -> bool {
        let id = candidate.id();
        let history = ctx.cascade_history();

        // Self-exclusion: no scheme appears twice in any chain.
        if history.iter().any(|&(sid, _)| sid == id) {
            return true;
        }

        let mut iter = history.iter().copied().peekable();

        // The root entry is always first in the history (if present). Check if the root has
        // excluded us.
        if let Some((_, child_idx)) = iter.next_if(|&(sid, _)| sid == ROOT_SCHEME_ID)
            && self
                .root_exclusions
                .iter()
                .any(|rule| rule.excluded == id && rule.children.contains(child_idx))
        {
            return true;
        }

        // Push rules: Check if any of our ancestors have excluded us.
        for (ancestor_id, child_idx) in iter {
            if let Some(ancestor) = self.schemes.iter().find(|s| s.id() == ancestor_id)
                && ancestor
                    .descendant_exclusions()
                    .iter()
                    .any(|rule| rule.excluded == id && rule.children.contains(child_idx))
            {
                return true;
            }
        }

        // Pull rules: Check if we have excluded ourselves because of our ancestors.
        for rule in candidate.ancestor_exclusions() {
            if history
                .iter()
                .any(|(sid, cidx)| *sid == rule.ancestor && rule.children.contains(*cidx))
            {
                return true;
            }
        }

        false
    }
}
