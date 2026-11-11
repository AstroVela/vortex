// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Experimental, immutable Float32 squared-L2 SPFresh indexes.
//!
//! Native support is opt-in. See the crate README for the pinned build and format contract.
#![deny(missing_docs)]

/// Exact upstream revision defining this backend's native file format.
pub const SPFRESH_REVISION: &str = "5893eb61ee3b18610b6b00f1939be7dae1af8904";

pub mod rabitq;

mod bundle;
pub use bundle::SpFreshBundle;
pub use bundle::import_bundle;

/// Registry identifier for this backend, not the SPFresh source revision.
pub const SPFRESH_ID: &str = "spfresh.static";
/// Legacy unquantized bundle version; the row mapping format remains version 1.
pub const SPFRESH_FORMAT_VERSION: u32 = 1;
/// Static bundle version with a RaBitQ sidecar and lazy original-vector reranking.
pub const SPFRESH_RABITQ_FORMAT_VERSION: u32 = 2;
/// Static bundle with official two-bit RaBitQ, grouped FastScan and exact reranking.
pub const SPFRESH_OFFICIAL_RABITQ_FORMAT_VERSION: u32 = 3;
/// Pinned official RaBitQ numerical-library revision used by the optional backend.
pub const RABITQ_OFFICIAL_REVISION: &str = "dabfdb8c200a6f155df4edd0b9c825e6728b14a1";

#[cfg(feature = "native")]
mod builder;
#[cfg(feature = "native")]
pub use builder::SpFreshBuildLimits;
#[cfg(feature = "native")]
pub use builder::SpFreshBuildOptions;
#[cfg(feature = "native")]
pub use builder::SpFreshIndexBuilder;
#[cfg(feature = "native")]
mod ffi;
#[cfg(feature = "native")]
mod provider;
#[cfg(feature = "native")]
mod validate;
#[cfg(feature = "native")]
pub use provider::SpFreshLimits;
#[cfg(feature = "native")]
pub use provider::SpFreshProvider;

#[cfg(test)]
mod tests;
