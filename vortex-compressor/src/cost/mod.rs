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

mod candidate;
pub use candidate::Candidate;

mod model;
pub use model::Cost;
pub use model::CostModel;

mod size;
pub use size::SizeCost;
