// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Experimental, immutable Float32 squared-L2 SPFresh indexes.
//!
//! Native support is opt-in. See the crate README for the pinned build and format contract.
#![deny(missing_docs)]

/// Exact upstream revision defining this backend's native file format.
pub const SPFRESH_REVISION: &str = "5893eb61ee3b18610b6b00f1939be7dae1af8904";

mod bundle;
pub use bundle::SpFreshBundle;
pub use bundle::import_bundle;

/// Registry identifier for this backend, not the SPFresh source revision.
pub const SPFRESH_ID: &str = "spfresh.static";
/// Version of the closed bundle, descriptor and row mapping format.
pub const SPFRESH_FORMAT_VERSION: u32 = 1;

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
