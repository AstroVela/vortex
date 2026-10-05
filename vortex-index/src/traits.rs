// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::BoxStream;
use vortex_array::ArrayRef;
use vortex_array::expr::Expression;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::IndexArtifact;
use crate::IndexMetadata;
use crate::RowAddress;
use crate::RowFilter;
use crate::ScalarCandidates;
use crate::SearchHit;
use crate::Snapshot;
use crate::VectorSearchOptions;
use crate::VectorSpec;

/// An opened, immutable generation. Implementations synchronize non-thread-safe native handles.
pub trait Index: Debug + Send + Sync {
    /// Validated metadata describing this exact generation.
    fn metadata(&self) -> &IndexMetadata;

    /// Vector-query capability, if implemented.
    fn as_vector(&self) -> Option<&dyn VectorIndex> {
        None
    }

    /// Scalar-query capability, if implemented.
    fn as_scalar(&self) -> Option<&dyn ScalarIndex> {
        None
    }
}

/// Vector search over one index's coverage, not necessarily the entire dataset.
#[async_trait]
pub trait VectorIndex: Index {
    /// Query dimensions and distance semantics.
    fn spec(&self) -> VectorSpec;

    /// Whether exact search is supported. False requires rejecting `SearchMode::Exact`.
    fn supports_exact(&self) -> bool;

    /// Whether predicate/visibility restrictions can be respected during retrieval.
    ///
    /// False requires rejecting restricted filters, not silently post-filtering top-k.
    fn supports_filter(&self) -> bool;

    /// Return at most k distinct, finite-distance hits ordered by (distance, row address).
    ///
    /// Implementations must validate the filter snapshot, dimensions, finite query
    /// values, and options before calling native code. Mandatory exclusions always
    /// apply, even for approximate queries. Unknown options must not be ignored.
    async fn search(
        &self,
        query: &[f32],
        options: &VectorSearchOptions,
        filter: &RowFilter,
    ) -> VortexResult<Vec<SearchHit>>;

    /// Search multiple vectors, preserving outer query order; empty batches are valid.
    ///
    /// Providers with a native batch API can override this sequential fallback.
    async fn search_batch(
        &self,
        queries: &[Buffer<f32>],
        options: &VectorSearchOptions,
        filter: &RowFilter,
    ) -> VortexResult<Vec<Vec<SearchHit>>> {
        let mut results = Vec::with_capacity(queries.len());
        for query in queries {
            results.push(self.search(query, options, filter).await?);
        }
        Ok(results)
    }
}

/// Predicate lookup, separate from ranked nearest-neighbor retrieval.
#[async_trait]
pub trait ScalarIndex: Index {
    /// Return an exact set or a no-false-negative superset within index coverage.
    ///
    /// Validate snapshot identity and apply visibility; reject unsupported expressions.
    /// The caller must scan uncovered files and evaluate any residual predicate.
    async fn search(
        &self,
        predicate: &Expression,
        filter: &RowFilter,
    ) -> VortexResult<ScalarCandidates>;
}

/// One projected source batch; rows and data must have identical lengths.
pub struct SourceBatch {
    /// Physical addresses before filtering, in the same order as the array rows.
    pub rows: Vec<RowAddress>,
    /// Projected Vortex data, normally a struct array.
    pub data: ArrayRef,
}

impl SourceBatch {
    /// Validate address alignment at the boundary between a source and an index.
    ///
    /// Duplicate addresses are permitted because ordered `take` must preserve them.
    pub fn try_new(
        snapshot: &Snapshot,
        rows: Vec<RowAddress>,
        data: ArrayRef,
    ) -> VortexResult<Self> {
        snapshot.validate()?;
        if rows.len() != data.len() {
            vortex_bail!("Source batch addresses and array rows must have identical lengths");
        }
        for row in &rows {
            snapshot.validate_row(*row)?;
        }
        Ok(Self { rows, data })
    }
}

/// Fixed-snapshot source adapter, independent of SQL and backend algorithms.
#[async_trait]
pub trait IndexSource: Send + Sync {
    /// Pinned source and visibility identity, unchanged during this adapter's lifetime.
    fn snapshot(&self) -> &Snapshot;

    /// Stream visible rows in the requested files, projecting the named fields.
    ///
    /// Do not renumber after filtering. File versions must be checked when opening
    /// data; missing/changed files are errors, not silently skipped rows.
    fn scan(
        &self,
        files: &[u64],
        fields: &[String],
    ) -> VortexResult<BoxStream<'static, VortexResult<SourceBatch>>>;

    /// Fetch projected visible rows in request order, preserving duplicate addresses.
    ///
    /// Reject missing/deleted rows instead of returning a shorter, misaligned batch.
    async fn take(&self, rows: &[RowAddress], fields: &[String]) -> VortexResult<SourceBatch>;
}

/// Storage scoped to a private or sealed index generation.
///
/// This stores arbitrary backend files, not just Vortex columns. Implementations
/// enforce path confinement (including symlinks for local files), checksums and
/// length validation. A native backend may initially require a local adapter.
#[async_trait]
pub trait IndexStore: Send + Sync {
    /// Read and verify an immutable artifact; a missing or corrupt object is an error.
    async fn read(&self, artifact: &IndexArtifact) -> VortexResult<Bytes>;

    /// Write a new private artifact without overwriting an existing name.
    ///
    /// Return only after the bytes are durable under the store's documented contract.
    /// This does not publish an index or commit a source snapshot.
    async fn write(&self, path: &str, data: Bytes) -> VortexResult<IndexArtifact>;

    /// Optional blocking file-path IO for native backends.
    ///
    /// Providers requiring paths must reject stores without this capability.
    fn as_local_files(&self) -> Option<&dyn LocalIndexFiles> {
        None
    }
}

/// Optional file-path access without loading whole artifacts into memory.
///
/// These operations are blocking and belong on a blocking worker, including when
/// called from an async builder or provider. They do not publish a generation.
pub trait LocalIndexFiles: Send + Sync {
    /// Stream a closed, regular source file into a new private artifact.
    ///
    /// The destination is a portable relative artifact path. The source must be
    /// absolute, contain no symlinks, and remain unchanged throughout the copy.
    /// The caller must quiesce native writers first; this does not establish a
    /// consistent snapshot of a running backend. Copy bytes, never adopt or hard
    /// link the source. Return a size/checksum identity only after durable IO.
    fn import_file(&self, path: &str, source: &Path) -> VortexResult<IndexArtifact>;

    /// Copy and verify selected artifacts from a sealed generation into a lease.
    ///
    /// Reject unknown identities and duplicate paths. Preserve relative paths
    /// beneath a fresh directory in `scratch_root`, which must be absolute,
    /// symlink-free and owner-managed. `max_bytes` bounds the sum of artifact
    /// lengths, not filesystem allocation or RSS; per-artifact limits also apply.
    /// Copies must not share writable inodes with the generation or other leases.
    /// A failed call must not expose a partially verified lease.
    /// A store opened through verified materialization may transfer its unused
    /// complete-inventory lease on the first matching call; the same isolation
    /// and byte-limit requirements apply.
    fn materialize(
        &self,
        artifacts: &[IndexArtifact],
        scratch_root: &Path,
        max_bytes: u64,
    ) -> VortexResult<Box<dyn LocalArtifactLease>>;
}

/// Owns private, verified local copies for a native reader's lifetime.
///
/// The provider must retain this lease until all native handles, background
/// tasks and mappings have been closed, and must not mutate its files. Paths
/// embedded in backend configuration still require backend-specific validation;
/// a lease is not a sandbox for native code or same-user filesystem mutation.
/// Dropping the lease attempts cleanup, not durable reclamation. Process crashes
/// can leave scratch files; their cleanup belongs to the scratch-root owner.
pub trait LocalArtifactLease: Debug + Send + Sync {
    /// Absolute directory containing exactly the requested relative file paths.
    ///
    /// The directory is process-local scratch space, not a persistent identity.
    fn path(&self) -> &Path;
}

/// A backend-owned build request for a new, unpublished generation.
pub struct IndexBuildRequest {
    /// Target metadata and coverage. The builder fills the artifact inventory.
    pub metadata: IndexMetadata,
    /// Versioned backend configuration; the common layer must not reinterpret it.
    pub backend_options: Bytes,
}

/// Optional initial-build capability. Incremental mutation is deliberately separate.
#[async_trait]
pub trait IndexBuilder: Send + Sync {
    /// Build durable artifacts from the pinned source and return complete artifact metadata.
    ///
    /// Validate metadata against the source before reading. A successful return is
    /// not a store seal or catalog commit; the owner must seal its generation and
    /// compare-and-swap the expected snapshot before publication.
    /// A failure may leave unreferenced artifacts but must not change reader visibility.
    async fn build(
        &self,
        request: IndexBuildRequest,
        source: Arc<dyn IndexSource>,
        store: Arc<dyn IndexStore>,
    ) -> VortexResult<IndexMetadata>;
}

/// Registry entry for an algorithm implementation, not a query engine.
#[async_trait]
pub trait IndexProvider: Send + Sync {
    /// Stable, case-sensitive backend identifier (for example, `vortex.spfresh`).
    fn id(&self) -> &str;

    /// Whether this implementation can read an artifact format version.
    fn supports_version(&self, version: u32) -> bool;

    /// Optional total byte limit for eagerly materializing a local generation.
    ///
    /// A caller may combine generation verification and private copying before
    /// [`Self::open`], allowing the provider's first complete-inventory
    /// [`LocalIndexFiles::materialize`] call to reuse that lease. This is an
    /// optimization hint: providers must still validate their inputs and limits,
    /// and must also work with stores that perform materialization on demand.
    fn local_materialization_limit(&self) -> Option<u64> {
        None
    }

    /// Open and verify the declared artifacts without mutating the generation.
    async fn open(
        &self,
        metadata: &IndexMetadata,
        store: Arc<dyn IndexStore>,
    ) -> VortexResult<Arc<dyn Index>>;

    /// Initial-build capability, if provided; no mandatory dummy build implementation.
    fn builder(&self) -> Option<&dyn IndexBuilder> {
        None
    }
}
