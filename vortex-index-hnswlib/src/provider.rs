// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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

use crate::HNSWLIB_FORMAT_VERSION;
use crate::HNSWLIB_ID;
use crate::HNSWLIB_REVISION;
use crate::HnswIndexBuilder;
use crate::bundle::DESCRIPTOR;
use crate::bundle::Descriptor;
use crate::bundle::INDEX;
use crate::bundle::ROWS;
use crate::bundle::validate_metadata;
use crate::bundle::validate_rows;
use crate::ffi::Native;
use crate::validate;

/// Explicit bounds on private copies, mapped rows and query work, not total RSS.
#[derive(Debug, Clone, Copy)]
pub struct HnswLimits {
    /// Maximum total materialized artifact bytes.
    pub max_materialized_bytes: u64,
    /// Maximum number of mapped native labels; mapping uses 16 bytes per row.
    pub max_rows: u32,
    /// Maximum number of queries in a sequential batch.
    pub max_batch_vectors: u32,
    /// Maximum accepted ef search width.
    pub max_ef: u32,
}

impl Default for HnswLimits {
    fn default() -> Self {
        Self {
            max_materialized_bytes: 1024 * 1024 * 1024,
            max_rows: 1_000_000,
            max_batch_vectors: 256,
            max_ef: 1_048_576,
        }
    }
}

/// Opt-in immutable, local hnswlib provider; blocking operations require a blocking worker.
///
/// Each generation owns a private verified lease and one native handle. Searches
/// and ef changes are serialized per handle, not globally across all indexes.
/// Initial construction is explicitly added with [`Self::with_builder`].
#[derive(Debug)]
pub struct HnswProvider {
    scratch_root: PathBuf,
    limits: HnswLimits,
    builder: Option<HnswIndexBuilder>,
}

impl HnswProvider {
    /// Set an absolute, owner-managed scratch root and resource bounds.
    pub fn try_new(scratch_root: PathBuf, limits: HnswLimits) -> VortexResult<Self> {
        if !scratch_root.is_absolute()
            || limits.max_materialized_bytes == 0
            || !(1..=i32::MAX as u32).contains(&limits.max_rows)
            || limits.max_batch_vectors == 0
            || !(1..=1_048_576).contains(&limits.max_ef)
        {
            vortex_bail!("Invalid hnswlib scratch root or provider limits");
        }
        Ok(Self {
            scratch_root,
            limits,
            builder: None,
        })
    }

    /// Advertise explicitly configured static construction without incremental mutation.
    pub fn with_builder(mut self, builder: HnswIndexBuilder) -> Self {
        self.builder = Some(builder);
        self
    }
}

#[async_trait]
impl IndexProvider for HnswProvider {
    fn id(&self) -> &str {
        HNSWLIB_ID
    }
    fn supports_version(&self, version: u32) -> bool {
        version == HNSWLIB_FORMAT_VERSION
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
        let mut paths = metadata
            .artifacts
            .iter()
            .map(|artifact| artifact.path.as_str())
            .collect::<Vec<_>>();
        paths.sort_unstable();
        let mut expected = [DESCRIPTOR, INDEX, ROWS];
        expected.sort_unstable();
        if paths != expected {
            vortex_bail!("hnswlib requires exactly its closed artifact inventory");
        }
        let mut total = 0u64;
        for artifact in &metadata.artifacts {
            total = total
                .checked_add(artifact.size)
                .ok_or_else(|| vortex_err!("hnswlib artifact size overflow"))?;
            let limit = match artifact.path.as_str() {
                DESCRIPTOR => 1024 * 1024,
                ROWS => 16 + u64::from(self.limits.max_rows) * 16,
                _ => self.limits.max_materialized_bytes,
            };
            if artifact.size == 0 || artifact.size > limit {
                vortex_bail!("hnswlib artifact exceeds its byte limit");
            }
        }
        if total > self.limits.max_materialized_bytes {
            vortex_bail!("hnswlib artifacts exceed materialization budget");
        }
        let local = store
            .as_local_files()
            .ok_or_else(|| vortex_err!("hnswlib requires local-file support"))?;
        let lease = local.materialize(
            &metadata.artifacts,
            &self.scratch_root,
            self.limits.max_materialized_bytes,
        )?;
        let descriptor: Descriptor =
            serde_json::from_slice(&fs::read(lease.path().join(DESCRIPTOR))?)
                .map_err(|err| vortex_err!("hnswlib descriptor: {err}"))?;
        if descriptor.format_version != HNSWLIB_FORMAT_VERSION
            || descriptor.revision != HNSWLIB_REVISION
            || descriptor.value_type != "float32"
            || descriptor.metric != "squared_l2"
            || descriptor.snapshot != metadata.snapshot
            || descriptor.fields != metadata.fields
            || descriptor.covered_files != metadata.covered_files
            || descriptor.bundle.rows > self.limits.max_rows
        {
            vortex_bail!("hnswlib descriptor does not match the format, binding or limits");
        }
        descriptor.bundle.validate()?;
        let rows = validate::mapping(&lease.path().join(ROWS), descriptor.bundle.rows)?;
        validate_rows(metadata, descriptor.bundle, &rows)?;
        let native = Native::open(&lease.path().join(INDEX), descriptor.bundle)?;
        let spec = VectorSpec {
            dimension: NonZeroUsize::new(descriptor.bundle.dimension as usize)
                .ok_or_else(|| vortex_err!("Zero hnswlib dimension"))?,
            metric: DistanceMetric::SquaredL2,
        };
        Ok(Arc::new(HnswIndex {
            native,
            _lease: lease,
            metadata: metadata.clone(),
            rows,
            limits: self.limits,
            spec,
        }))
    }
}

#[derive(Debug)]
struct HnswIndex {
    // Drop the native handle before removing its private files.
    native: Native,
    _lease: Box<dyn LocalArtifactLease>,
    metadata: IndexMetadata,
    rows: Vec<RowAddress>,
    limits: HnswLimits,
    spec: VectorSpec,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct QueryOptions {
    ef: u32,
}

impl HnswIndex {
    fn options(
        &self,
        options: &VectorSearchOptions,
        filter: &RowFilter,
    ) -> VortexResult<(u32, u32)> {
        if filter.snapshot() != &self.metadata.snapshot
            || filter.is_restricted()
            || options.mode != SearchMode::Approximate
            || options.k.get() > 4096
        {
            vortex_bail!(
                "hnswlib supports only matching-snapshot, unfiltered approximate queries with k <= 4096"
            );
        }
        let requested = u32::try_from(options.k.get())?;
        let ef = if options.backend_options.is_empty() {
            requested.max(64)
        } else {
            if options.backend_options.len() > 4096 {
                vortex_bail!("hnswlib query options exceed 4096 bytes");
            }
            serde_json::from_slice::<QueryOptions>(&options.backend_options)
                .map_err(|err| vortex_err!("hnswlib query options: {err}"))?
                .ef
        };
        if ef < requested || ef > self.limits.max_ef {
            vortex_bail!("hnswlib ef must be at least k and within the configured limit");
        }
        Ok((requested.min(u32::try_from(self.rows.len())?), ef))
    }
}

impl Index for HnswIndex {
    fn metadata(&self) -> &IndexMetadata {
        &self.metadata
    }
    fn as_vector(&self) -> Option<&dyn VectorIndex> {
        Some(self)
    }
}

#[async_trait]
impl VectorIndex for HnswIndex {
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
        let (k, ef) = self.options(options, filter)?;
        if query.len() != self.spec.dimension.get() || query.iter().any(|value| !value.is_finite())
        {
            vortex_bail!("hnswlib query requires the declared dimension and finite components");
        }
        let mut hits = self
            .native
            .search(query, k, ef)?
            .into_iter()
            .map(|(id, distance)| {
                let row = *self
                    .rows
                    .get(id as usize)
                    .ok_or_else(|| vortex_err!("hnswlib label outside mapping"))?;
                Ok(SearchHit { row, distance })
            })
            .collect::<VortexResult<Vec<_>>>()?;
        hits.sort_unstable_by(|left, right| {
            left.distance
                .total_cmp(&right.distance)
                .then(left.row.cmp(&right.row))
        });
        let mut unique = hits.iter().map(|hit| hit.row).collect::<Vec<_>>();
        unique.sort_unstable();
        if unique.windows(2).any(|pair| pair[0] == pair[1]) {
            vortex_bail!("hnswlib returned duplicate labels");
        }
        Ok(hits)
    }
    async fn search_batch(
        &self,
        queries: &[Buffer<f32>],
        options: &VectorSearchOptions,
        filter: &RowFilter,
    ) -> VortexResult<Vec<Vec<SearchHit>>> {
        self.options(options, filter)?;
        if queries.len() as u64 > u64::from(self.limits.max_batch_vectors) {
            vortex_bail!("hnswlib batch exceeds provider limit");
        }
        for query in queries {
            if query.len() != self.spec.dimension.get()
                || query.iter().any(|value| !value.is_finite())
            {
                vortex_bail!("hnswlib batch contains invalid vectors");
            }
        }
        let mut results = Vec::with_capacity(queries.len());
        for query in queries {
            results.push(self.search(query, options, filter).await?);
        }
        Ok(results)
    }
}
