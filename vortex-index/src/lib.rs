// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![deny(missing_docs)]
#![forbid(unsafe_code)]

//! Experimental, engine-independent secondary index contracts.
//!
//! Indexes are optional derived data, bound to an immutable source snapshot.
//! Neither a backend's numeric identifiers nor scan partition numbers are public
//! row identities. See [`RowAddress`] and [`Snapshot`] for the addressing contract.
//! The initial implementation requires an exact snapshot match; it does not
//! implement transactions, compaction remapping, or automatic SQL optimization.

#[cfg(feature = "file")]
pub mod file;
mod flat;
mod metadata;
mod query;
mod registry;
#[cfg(all(unix, feature = "local-store"))]
pub mod store;
mod traits;

pub use flat::FlatIndex;
pub use metadata::IndexArtifact;
pub use metadata::IndexMetadata;
pub use metadata::RowAddress;
pub use metadata::Snapshot;
pub use metadata::SourceFile;
pub use query::CandidateGuarantee;
pub use query::DistanceMetric;
pub use query::RowFilter;
pub use query::ScalarCandidates;
pub use query::SearchHit;
pub use query::SearchMode;
pub use query::VectorSearchOptions;
pub use query::VectorSpec;
pub use registry::IndexRegistry;
pub use traits::Index;
pub use traits::IndexBuildRequest;
pub use traits::IndexBuilder;
pub use traits::IndexProvider;
pub use traits::IndexSource;
pub use traits::IndexStore;
pub use traits::ScalarIndex;
pub use traits::SourceBatch;
pub use traits::VectorIndex;

#[cfg(test)]
mod tests;
