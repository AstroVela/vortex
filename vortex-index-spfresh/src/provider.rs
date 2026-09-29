// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeMap;
use std::fs;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_index::DistanceMetric;
use vortex_index::Index;
use vortex_index::IndexBuilder;
use vortex_index::IndexMetadata;
use vortex_index::IndexProvider;
use vortex_index::IndexStore;
use vortex_index::LocalArtifactLease;
use vortex_index::RowAddress;
use vortex_index::RowFilter;
use vortex_index::SearchHit;
use vortex_index::SearchMode;
use vortex_index::VectorIndex;
use vortex_index::VectorSearchOptions;
use vortex_index::VectorSpec;

use crate::SPFRESH_FORMAT_VERSION;
use crate::SPFRESH_ID;
use crate::SPFRESH_REVISION;
use crate::SpFreshBundle;
use crate::SpFreshIndexBuilder;
use crate::bundle::DESCRIPTOR;
use crate::bundle::Descriptor;
use crate::bundle::NATIVE_FILES;
use crate::bundle::ROWS;
use crate::bundle::validate_metadata;
use crate::bundle::validate_rows;
use crate::ffi::Native;
use crate::validate;

/// Explicit bounds on local copies, mapping and query work, not a total RSS limit.
#[derive(Debug, Clone, Copy)]
pub struct SpFreshLimits {
    /// Sum of all materialized artifact sizes; the store also enforces its own limits.
    pub max_materialized_bytes: u64,
    /// Maximum number of mapped native IDs. The row map occupies 16 bytes per row.
    pub max_rows: u32,
    /// Maximum vectors in one batch. Queries/results are buffered in memory.
    pub max_batch_vectors: u32,
    /// Maximum posting-buffer allocation for one native query (pages * probes * 4096).
    pub max_posting_buffer_bytes: u64,
}

impl Default for SpFreshLimits {
    fn default() -> Self {
        Self {
            max_materialized_bytes: 1024 * 1024 * 1024,
            max_rows: 1_000_000,
            max_batch_vectors: 256,
            max_posting_buffer_bytes: 256 * 1024 * 1024,
        }
    }
}

/// Read-only, blocking native provider for pinned SPFresh static bundles.
///
/// Requires the `native` feature and a local-file-capable sealed store. Call open
/// and search on a blocking worker; async signatures do not make native IO async.
/// All native operations in this bridge are serialized across handles, and native
/// thread-local workspaces are cleared at call boundaries. Initial construction
/// is opt-in via [`Self::with_builder`]; opened indexes have no mutation capability.
/// The owner must qualify recall for its workload.
#[derive(Debug)]
pub struct SpFreshProvider {
    scratch_root: PathBuf,
    limits: SpFreshLimits,
    builder: Option<SpFreshIndexBuilder>,
}

impl SpFreshProvider {
    /// Configure an absolute, owner-managed scratch root and resource bounds.
    ///
    /// The local store checks symlinks and directory safety during materialization.
    pub fn try_new(scratch_root: PathBuf, limits: SpFreshLimits) -> VortexResult<Self> {
        if !scratch_root.is_absolute()
            || limits.max_rows == 0
            || limits.max_rows > i32::MAX as u32
            || limits.max_materialized_bytes == 0
            || limits.max_batch_vectors == 0
            || limits.max_posting_buffer_bytes == 0
        {
            vortex_bail!("Invalid SPFresh scratch root or limits");
        }
        Ok(Self {
            scratch_root,
            limits,
            builder: None,
        })
    }

    /// Advertise an explicitly configured, initial-build-only capability.
    pub fn with_builder(mut self, builder: SpFreshIndexBuilder) -> Self {
        self.builder = Some(builder);
        self
    }
}

#[async_trait]
impl IndexProvider for SpFreshProvider {
    fn id(&self) -> &str {
        SPFRESH_ID
    }
    fn supports_version(&self, version: u32) -> bool {
        version == SPFRESH_FORMAT_VERSION
    }

    fn builder(&self) -> Option<&dyn IndexBuilder> {
        self.builder
            .as_ref()
            .map(|builder| builder as &dyn IndexBuilder)
    }

    async fn open(
        &self,
        metadata: &IndexMetadata,
        store: Arc<dyn IndexStore>,
    ) -> VortexResult<Arc<dyn Index>> {
        validate_metadata(metadata)?;
        validate_inventory(metadata, self.limits)?;
        let local = store
            .as_local_files()
            .ok_or_else(|| vortex_err!("SPFresh requires a local-file store"))?;
        let lease = local.materialize(
            &metadata.artifacts,
            &self.scratch_root,
            self.limits.max_materialized_bytes,
        )?;
        let descriptor: Descriptor =
            serde_json::from_slice(&fs::read(lease.path().join(DESCRIPTOR))?)
                .map_err(|err| vortex_err!("SPFresh descriptor: {err}"))?;
        if descriptor.format_version != SPFRESH_FORMAT_VERSION
            || descriptor.revision != SPFRESH_REVISION
            || descriptor.value_type != "float32"
            || descriptor.metric != "squared_l2"
            || descriptor.snapshot != metadata.snapshot
            || descriptor.fields != metadata.fields
            || descriptor.covered_files != metadata.covered_files
        {
            vortex_bail!("SPFresh descriptor does not match the native format or metadata binding");
        }
        let bundle = descriptor.bundle;
        bundle.validate()?;
        if bundle.rows > self.limits.max_rows {
            vortex_bail!("SPFresh row count exceeds provider limit");
        }
        let rows = validate::mapping(&lease.path().join(ROWS), bundle.rows)?;
        validate_rows(metadata, bundle, &rows)?;
        let root = lease.path().join("spfresh/native");
        validate::native_files(&root, bundle)?;
        let native = Native::open(&root, bundle)?;
        let dimension = NonZeroUsize::new(bundle.dimension as usize)
            .ok_or_else(|| vortex_err!("Zero SPFresh dimension"))?;
        Ok(Arc::new(SpFreshIndex {
            native,
            _lease: lease,
            metadata: metadata.clone(),
            bundle,
            rows,
            limits: self.limits,
            spec: VectorSpec {
                dimension,
                metric: DistanceMetric::SquaredL2,
            },
        }))
    }
}

fn validate_inventory(metadata: &IndexMetadata, limits: SpFreshLimits) -> VortexResult<()> {
    let mut expected = NATIVE_FILES
        .iter()
        .map(|name| format!("spfresh/native/{name}"))
        .collect::<Vec<_>>();
    expected.extend([DESCRIPTOR.to_owned(), ROWS.to_owned()]);
    expected.sort_unstable();
    let mut actual = metadata
        .artifacts
        .iter()
        .map(|artifact| artifact.path.clone())
        .collect::<Vec<_>>();
    actual.sort_unstable();
    if actual != expected {
        vortex_bail!("SPFresh requires exactly the closed native bundle artifact inventory");
    }
    let mut total = 0u64;
    for artifact in &metadata.artifacts {
        total = total
            .checked_add(artifact.size)
            .ok_or_else(|| vortex_err!("Artifact size overflow"))?;
        let limit = match artifact.path.as_str() {
            DESCRIPTOR => 1024 * 1024,
            ROWS => 16 + 16 * u64::from(limits.max_rows),
            _ => limits.max_materialized_bytes,
        };
        if artifact.size == 0 || artifact.size > limit {
            vortex_bail!("SPFresh artifact exceeds size limit: {}", artifact.path);
        }
    }
    if total > limits.max_materialized_bytes {
        vortex_bail!("SPFresh bundle exceeds materialization budget");
    }
    Ok(())
}

#[derive(Debug)]
struct SpFreshIndex {
    // Rust drops fields in declaration order: close native IO before releasing files.
    native: Native,
    _lease: Box<dyn LocalArtifactLease>,
    metadata: IndexMetadata,
    bundle: SpFreshBundle,
    rows: Vec<RowAddress>,
    limits: SpFreshLimits,
    spec: VectorSpec,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct QueryOptions {
    max_check: u32,
    internal_results: u32,
    search_pages: u32,
}

impl SpFreshIndex {
    fn options(
        &self,
        options: &VectorSearchOptions,
        filter: &RowFilter,
    ) -> VortexResult<(u32, QueryOptions)> {
        if filter.snapshot() != &self.metadata.snapshot {
            vortex_bail!("SPFresh filter snapshot mismatch");
        }
        if filter.is_restricted() || options.mode != SearchMode::Approximate {
            vortex_bail!("SPFresh supports only unfiltered approximate search");
        }
        if options.k.get() > 4096 {
            vortex_bail!("SPFresh k exceeds 4096");
        }
        let k = u32::try_from(options.k.get())
            .map_err(|err| vortex_err!("{err}"))?
            .min(self.bundle.rows);
        let native = if options.backend_options.is_empty() {
            QueryOptions {
                max_check: 4096,
                internal_results: k.max(64),
                search_pages: self.bundle.posting_page_limit,
            }
        } else {
            if options.backend_options.len() > 4096 {
                vortex_bail!("SPFresh query options exceed 4096 bytes");
            }
            serde_json::from_slice(&options.backend_options)
                .map_err(|err| vortex_err!("SPFresh query options: {err}"))?
        };
        if native.internal_results < k
            || native.internal_results > 4096
            || native.max_check < native.internal_results
            || native.max_check > 1_048_576
            || native.search_pages != self.bundle.posting_page_limit
        {
            vortex_bail!("Invalid SPFresh native query options");
        }
        // Posting reads are truncated by the bundle page limit when the index is
        // opened; search_pages must equal that limit (validated above), so each
        // native page buffer holds exactly posting_page_limit pages.
        let pages = u64::from(self.bundle.posting_page_limit);
        if pages * u64::from(native.internal_results) * 4096 > self.limits.max_posting_buffer_bytes
        {
            vortex_bail!("SPFresh query exceeds posting buffer budget");
        }
        Ok((k, native))
    }

    fn validate_query(&self, query: &[f32]) -> VortexResult<()> {
        if query.len() != self.spec.dimension.get() || query.iter().any(|value| !value.is_finite())
        {
            vortex_bail!("SPFresh query must have the declared dimension and finite components");
        }
        Ok(())
    }

    fn run(
        &self,
        queries: &[f32],
        k: u32,
        options: QueryOptions,
    ) -> VortexResult<Vec<Vec<SearchHit>>> {
        self.native
            .search(
                queries,
                k,
                options.max_check,
                options.internal_results,
                options.search_pages,
            )?
            .into_iter()
            .map(|hits| {
                let mut unique: BTreeMap<RowAddress, f32> = BTreeMap::new();
                for (id, distance) in hits {
                    let row = self
                        .rows
                        .get(id as usize)
                        .ok_or_else(|| vortex_err!("SPFresh ID outside mapping"))?;
                    unique
                        .entry(*row)
                        .and_modify(|best| *best = best.min(distance))
                        .or_insert(distance);
                }
                let mut hits = unique
                    .into_iter()
                    .map(|(row, distance)| SearchHit { row, distance })
                    .collect::<Vec<_>>();
                hits.sort_unstable_by(|left, right| {
                    left.distance
                        .total_cmp(&right.distance)
                        .then(left.row.cmp(&right.row))
                });
                Ok(hits)
            })
            .collect()
    }
}

impl Index for SpFreshIndex {
    fn metadata(&self) -> &IndexMetadata {
        &self.metadata
    }
    fn as_vector(&self) -> Option<&dyn VectorIndex> {
        Some(self)
    }
}

#[async_trait]
impl VectorIndex for SpFreshIndex {
    fn spec(&self) -> VectorSpec {
        self.spec
    }
    fn supports_exact(&self) -> bool {
        false
    }
    fn supports_filter(&self) -> bool {
        false
    }

    async fn search(
        &self,
        query: &[f32],
        options: &VectorSearchOptions,
        filter: &RowFilter,
    ) -> VortexResult<Vec<SearchHit>> {
        let (k, native_options) = self.options(options, filter)?;
        self.validate_query(query)?;
        self.run(query, k, native_options)?
            .pop()
            .ok_or_else(|| vortex_err!("Missing SPFresh result"))
    }

    async fn search_batch(
        &self,
        queries: &[Buffer<f32>],
        options: &VectorSearchOptions,
        filter: &RowFilter,
    ) -> VortexResult<Vec<Vec<SearchHit>>> {
        let (k, native_options) = self.options(options, filter)?;
        if queries.len() as u64 > u64::from(self.limits.max_batch_vectors) {
            vortex_bail!("SPFresh batch exceeds provider limit");
        }
        for query in queries {
            self.validate_query(query)?;
        }
        if queries.is_empty() {
            return Ok(Vec::new());
        }
        let flat = queries
            .iter()
            .flat_map(|query| query.iter().copied())
            .collect::<Vec<_>>();
        self.run(&flat, k, native_options)
    }
}
