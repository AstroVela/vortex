// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Opt-in, immutable hnswlib Float32 squared-L2 indexes, independent of SQL.
#![deny(missing_docs)]

/// Explicit registry identifier, not an automatically selected default backend.
pub const HNSWLIB_ID: &str = "hnswlib.static";
/// Closed descriptor, native-file and physical-row-mapping format version.
pub const HNSWLIB_FORMAT_VERSION: u32 = 1;
/// Exact upstream revision defining the native serialization and search implementation.
pub const HNSWLIB_REVISION: &str = "d9b3608c83d83b46c96e25088cb1d729b29dcfe9";

mod bundle;
pub use bundle::HnswBundle;
pub use bundle::import_bundle;
#[cfg(feature = "native")]
mod builder;
#[cfg(feature = "native")]
pub use builder::HnswBuildLimits;
#[cfg(feature = "native")]
pub use builder::HnswBuildOptions;
#[cfg(feature = "native")]
pub use builder::HnswIndexBuilder;
#[cfg(feature = "native")]
mod ffi;
#[cfg(feature = "native")]
mod provider;
#[cfg(feature = "native")]
pub use provider::HnswLimits;
#[cfg(feature = "native")]
pub use provider::HnswProvider;
#[cfg(feature = "native")]
mod validate;

#[cfg(test)]
mod tests;
