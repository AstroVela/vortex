// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pluggable cost models for scheme selection.
//!
//! A [`CostModel`] is the policy half of scheme selection: [`Scheme`]s produce mechanical
//! signals (estimated or sample-measured compression ratios, collected into a [`Candidate`]),
//! and the model prices each candidate. The compressor picks the candidate with the lowest
//! cost that (strictly) beats [`CostModel::canonical_cost`] — the price of leaving the array
//! in its canonical encoding.
//!
//! The default model is [`SizeCost`], which preserves the compressor's historical candidate
//! ordering and canonical-acceptance threshold.
//!
//! # What sits outside the model
//!
//! The initial cost-model boundary covers scored candidates inside
//! `CascadingCompressor::choose_best_scheme`. The following pre-existing decisions remain
//! outside it:
//!
//! - Constant-array handling occurs before scheme selection.
//! - [`EstimateVerdict::AlwaysUse`] is a forced-selection path that short-circuits priced
//!   candidates.
//! - The byte-acceptance gate: after the winning scheme compresses the full array, the result
//!   is kept only if it is byte-wise smaller than its input. This is an axiom for **all**
//!   models — compression never grows bytes. A model that prefers canonical (e.g. for speed)
//!   expresses that by pricing every candidate at or above `canonical_cost`, so selection
//!   returns no winner and the array stays canonical; the gate never forces a *bad* encoding,
//!   only "no encoding". (The gate's `AnyScalarFn` carve-out is likewise semantic
//!   denormalization, not cost.)
//! - Extension arrays separately compare scheme-based compression with compression of their
//!   storage array by byte size.
//!
//! # Determinism
//!
//! Cost models must be pure functions of the candidate and their own configuration: no
//! timing measurements, no I/O, no global state. The compressor's output must be a
//! deterministic function of its input and configuration.
//!
//! [`Scheme`]: crate::scheme::Scheme
//! [`EstimateVerdict::AlwaysUse`]: crate::estimate::EstimateVerdict::AlwaysUse

mod size;

use std::cmp::Ordering;
use std::fmt::Debug;

pub use size::SizeCost;

pub use crate::candidate::Candidate;
use crate::stats::ArrayAndStats;

/// An opaque, totally ordered cost. Lower is better.
///
/// Units are defined by each [`CostModel`], not by the framework: costs are only ever
/// compared *within* one model on one compressor instance, so cross-model comparability is
/// intentionally not provided.
///
/// Values must be finite. `NaN`/infinite prices are not representable orderings; a model
/// that cannot price a candidate rejects it by returning `None` from [`CostModel::cost`].
#[derive(Debug, Clone, Copy)]
pub struct Cost(f64);

impl Cost {
    /// Creates a cost from a finite value.
    ///
    /// # Panics
    ///
    /// Panics if `value` is not finite.
    pub fn new(value: f64) -> Self {
        assert!(value.is_finite(), "Cost must be finite, got {value}");
        Self(if value == 0.0 { 0.0 } else { value })
    }

    /// Returns the raw cost value.
    pub fn value(self) -> f64 {
        self.0
    }
}

impl PartialEq for Cost {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Cost {}

impl PartialOrd for Cost {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Cost {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Prices [`Candidate`]s during scheme selection.
///
/// Implementations must be deterministic pure functions of their inputs and cheap: `cost` is
/// called once per scored candidate per selection site, i.e. `O(schemes × cascade levels)`
/// times per compressed chunk.
///
/// Selection semantics: the winner is the candidate with the minimum cost, requiring
/// strictly lower cost both to displace the running best (ties break by scheme registration
/// order) and to beat [`canonical_cost`]. Returning `None` from [`cost`] rejects the
/// candidate outright.
///
/// Installing a custom model also disables compression-ratio early-exit hints for deferred
/// scheme callbacks. A custom ordering may prefer a candidate whose ratio is below the current
/// winner, so ratio-based pruning would be unsound.
///
/// See the [module docs](self) for the acceptance axioms that apply to all models.
///
/// [`canonical_cost`]: CostModel::canonical_cost
/// [`cost`]: CostModel::cost
pub trait CostModel: Debug + Send + Sync + 'static {
    /// Estimated cost of choosing this candidate. Lower is better; `None` rejects the
    /// candidate.
    fn cost(&self, candidate: &Candidate<'_>) -> Option<Cost>;

    /// Cost of leaving the array canonical — the baseline every candidate must strictly
    /// beat to be selected.
    fn canonical_cost(&self, data: &ArrayAndStats, n_values: u64) -> Cost;
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use super::Cost;

    #[test]
    fn rejects_non_finite_values() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                std::panic::catch_unwind(|| Cost::new(value)).is_err(),
                "non-finite cost {value} was accepted"
            );
        }
    }

    #[test]
    fn signed_zero_has_one_representation() {
        let negative = Cost::new(-0.0);
        let positive = Cost::new(0.0);

        assert_eq!(negative, positive);
        assert_eq!(negative.cmp(&positive), Ordering::Equal);
        assert_eq!(negative.value().to_bits(), 0.0f64.to_bits());
    }

    #[test]
    fn costs_have_a_total_order() {
        let mut costs = [Cost::new(3.0), Cost::new(-2.0), Cost::new(0.0)];
        costs.sort();

        assert_eq!(costs.map(Cost::value), [-2.0, 0.0, 3.0]);
    }
}
