// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! RaBitQ sidecars with original-vector reranking.
//!
//! Version 2 uses the one-bit Haar implementation below. The optional
//! `rabitq-official` backend uses pinned official two-bit encoding and FastScan,
//! per-posting means, and a fixed exact-rerank budget in version-3 bundles.
//!
//! The estimator follows Gao and Long, SIGMOD 2024, §3.2:
//! <https://arxiv.org/abs/2405.12497>. Full precision query components avoid the
//! additional query scalar-quantization error. Codes use exactly ceil(d / 8)
//! bytes; norms, correction factors, rotation and routing are additional storage.

mod codec;
#[cfg(all(unix, any(feature = "native", test)))]
mod distance;
#[cfg(feature = "rabitq-official")]
mod official;
#[cfg(all(unix, any(feature = "native", test)))]
mod search;
#[cfg(all(unix, any(feature = "native", test)))]
mod storage;
#[cfg(feature = "rabitq-official")]
pub(crate) use official::OfficialIndex;
#[cfg(feature = "rabitq-official")]
pub(crate) use official::build_official_sidecar;
pub(crate) const OFFICIAL_SIDECAR: &str = "spfresh/rabitq-official.bin";

pub use codec::DistanceEstimate;
pub use codec::EncodedVector;
pub use codec::RaBitQ;
pub use codec::RaBitQQuery;
use serde::Deserialize;
use serde::Serialize;
#[cfg(feature = "native")]
pub(crate) use storage::QuantizedIndex;
#[cfg(feature = "native")]
pub(crate) use storage::SIDECAR;
#[cfg(feature = "native")]
pub(crate) use storage::build_sidecar;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

/// Reproducible construction options. The actual rotation is persisted, so
/// readers do not depend on the PRNG implementation of the producing binary.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RaBitQBuildOptions {
    /// Seed for independent Gaussian Householder reflectors.
    pub seed: u64,
}

/// Official two-bit encoding with per-posting database means and persisted rotation.
///
/// Requires the `rabitq-official` feature, artifact version 3 and one posting replica.
/// Queries estimate every candidate, select a fixed rerank budget, then use original
/// Float32 vectors for exact squared-L2 distances. Approximate routing and shortlist
/// selection require workload-specific recall qualification.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OfficialRaBitQBuildOptions {
    /// Seed for the official rotation. Readers restore the saved rotation directly.
    pub seed: u32,
}

/// Error bound used when discarding candidates before full precision reranking.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RaBitQErrorBound {
    /// Cauchy-Schwarz bound, with numerical slack. No probabilistic pruning.
    /// ANN centroid navigation is still approximate.
    #[default]
    Deterministic,
    /// RaBitQ concentration bound: epsilon / sqrt(d - 1), capped at the
    /// deterministic bound. This is a per-comparison confidence interval,
    /// not a no-false-negative or whole-query recall guarantee. A typical
    /// experimental setting is epsilon=4; qualify recall on your own corpus.
    Probabilistic {
        /// Positive, finite confidence multiplier; larger is more conservative.
        epsilon: f64,
    },
}

impl RaBitQErrorBound {
    pub(crate) fn validate(self) -> VortexResult<()> {
        if let Self::Probabilistic { epsilon } = self
            && (!epsilon.is_finite() || epsilon <= 0.0)
        {
            vortex_bail!("RaBitQ epsilon must be positive and finite");
        }
        Ok(())
    }

    fn factor(self, dimension: usize) -> f64 {
        match self {
            Self::Deterministic => 1.0,
            Self::Probabilistic { epsilon } if dimension > 1 => {
                (epsilon / ((dimension - 1) as f64).sqrt()).min(1.0)
            }
            Self::Probabilistic { .. } => 1.0,
        }
    }
}

#[cfg(test)]
#[cfg(unix)]
mod tests;
