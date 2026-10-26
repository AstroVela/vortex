// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Query a table-owned snapshot using static base indexes and explicit delta scans.
//!
//! The table owns commits, source identity, deletion/version visibility and index
//! publication. Each base generation must cover unchanged files and all currently
//! visible rows in those files. The caller explicitly selects the delta files to
//! scan. Invalid coverage and unsupported queries are errors, with no scan fallback.
//! No stable primary key, transaction engine or mutable ANN state is introduced.
//!
//! ```no_run
//! use std::num::NonZeroUsize;
//! use std::sync::Arc;
//! use vortex_error::VortexResult;
//! use vortex_index::{IndexSource, RowFilter, SearchMode, VectorSpec};
//! use vortex_index::snapshot::{SnapshotIndex, SnapshotVectorSearch, VisibleSource};
//! use vortex_session::VortexSession;
//!
//! # async fn example(source: Arc<dyn IndexSource>, visibility: RowFilter,
//! #     predicate: RowFilter, base: Vec<SnapshotIndex>, delta_files: Vec<u64>,
//! #     spec: VectorSpec, query: &[f32], session: VortexSession) -> VortexResult<()> {
//! let search = SnapshotVectorSearch::try_new(
//!     VisibleSource::try_new(source, visibility)?,
//!     "embedding".into(), spec, base, delta_files, session,
//! )?;
//! let hits = search.search(query, NonZeroUsize::MIN, SearchMode::Approximate, &predicate).await?;
//! let rows: Vec<_> = hits.iter().map(|hit| hit.row).collect();
//! let payload = search.source().take(&rows, &["title".into()]).await?;
//! assert_eq!(payload.rows, rows);
//! # Ok(())
//! # }
//! ```

use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::collections::BinaryHeap;
use std::num::NonZeroUsize;
use std::sync::Arc;

use async_stream::try_stream;
use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::stream::BoxStream;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::DistanceMetric;
use crate::Index;
use crate::IndexSource;
use crate::RowAddress;
use crate::RowFilter;
use crate::SearchHit;
use crate::SearchMode;
use crate::Snapshot;
use crate::SourceBatch;
use crate::VectorSearchOptions;
use crate::VectorSpec;

/// A source with mandatory table-owned visibility, preserving physical row addresses.
///
/// The supplied mask must include deletion markers and superseded row versions for
/// this exact snapshot. Scans omit invisible rows; `take` rejects them. Wrapping
/// an already visibility-filtered source is allowed but cannot restore hidden rows.
pub struct VisibleSource {
    source: Arc<dyn IndexSource>,
    visibility: RowFilter,
}

impl VisibleSource {
    /// Bind a complete table visibility mask to an already pinned source.
    pub fn try_new(source: Arc<dyn IndexSource>, visibility: RowFilter) -> VortexResult<Self> {
        source.snapshot().validate()?;
        if source.snapshot() != visibility.snapshot() {
            vortex_bail!("Source and visibility snapshots differ");
        }
        Ok(Self { source, visibility })
    }

    /// Immutable table visibility used by scans, row retrieval and index reuse proofs.
    pub fn visibility(&self) -> &RowFilter {
        &self.visibility
    }
}

#[async_trait]
impl IndexSource for VisibleSource {
    fn snapshot(&self) -> &Snapshot {
        self.source.snapshot()
    }

    fn scan(
        &self,
        files: &[u64],
        fields: &[String],
    ) -> VortexResult<BoxStream<'static, VortexResult<SourceBatch>>> {
        let mut stream = self.source.scan(files, fields)?;
        let visibility = self.visibility.clone();
        let files: BTreeSet<_> = files.iter().copied().collect();
        Ok(try_stream! {
            while let Some(batch) = stream.try_next().await? {
                if batch.rows.len() != batch.data.len() {
                    Err(vortex_err!("Source row/data alignment mismatch"))?;
                }
                let mut rows = Vec::new();
                let mut positions = Vec::new();
                for (position, row) in batch.rows.iter().enumerate() {
                    visibility.snapshot().validate_row(*row)?;
                    if !files.contains(&row.file_id) {
                        Err(vortex_err!("Source scan returned a file outside the request"))?;
                    }
                    if visibility.includes(*row) {
                        rows.push(*row);
                        positions.push(position as u64);
                    }
                }
                if rows.is_empty() { continue; }
                let data = if rows.len() == batch.rows.len() { batch.data } else {
                    batch.data.take(Buffer::from(positions).into_array())?
                };
                yield SourceBatch::try_new(visibility.snapshot(), rows, data)?;
            }
        }
        .boxed())
    }

    async fn take(&self, rows: &[RowAddress], fields: &[String]) -> VortexResult<SourceBatch> {
        for row in rows {
            self.snapshot().validate_row(*row)?;
            if !self.visibility.includes(*row) {
                vortex_bail!("Cannot retrieve a row hidden by snapshot visibility");
            }
        }
        let batch = self.source.take(rows, fields).await?;
        if batch.rows != rows || batch.data.len() != rows.len() {
            vortex_bail!("Source take did not preserve requested row order and multiplicity");
        }
        Ok(batch)
    }
}

/// An opened static generation and its catalog-recorded build visibility.
///
/// `indexed_visibility` must be the complete visibility used when building the
/// index, not today's mask. It supplies the table's proof that newly visible rows
/// were not omitted at build time. A provider's whole-file coverage alone does
/// not prove that it indexed rows hidden in its original snapshot.
pub struct SnapshotIndex {
    index: Arc<dyn Index>,
    indexed_visibility: RowFilter,
    backend_options: Bytes,
}

impl SnapshotIndex {
    /// Validate a catalog-selected vector generation without changing its metadata.
    pub fn try_new(
        index: Arc<dyn Index>,
        indexed_visibility: RowFilter,
        backend_options: Bytes,
    ) -> VortexResult<Self> {
        index
            .metadata()
            .validate_for(indexed_visibility.snapshot())?;
        if index.as_vector().is_none() {
            vortex_bail!("Snapshot vector search requires a vector index");
        }
        Ok(Self {
            index,
            indexed_visibility,
            backend_options,
        })
    }

    fn validate_coverage(&self, visibility: &RowFilter) -> VortexResult<()> {
        let metadata = self.index.metadata();
        if !metadata
            .covered_files
            .iter()
            .all(|id| metadata.snapshot.file(*id) == visibility.snapshot().file(*id))
        {
            vortex_bail!("Base index covers changed or missing files; rebuild the generation");
        }
        if !visibility.covered_subset_of(&self.indexed_visibility, &metadata.covered_files) {
            vortex_bail!("Base index omits newly visible rows; rebuild the generation");
        }
        Ok(())
    }
}

/// Vector search over one pinned source using static base generations and delta files.
///
/// The first implementation supports non-null, finite FixedSizeList<Float32> and
/// squared L2. Other payload columns retain their ordinary Vortex types. Indexes
/// must match the dataset, schema, vector field and metric. Base coverage and the
/// explicit delta inventory must partition the snapshot's files. Stale generations,
/// missing/overlapping coverage and unsupported queries are errors. No generation
/// is silently skipped or replaced with a scan.
pub struct SnapshotVectorSearch {
    source: VisibleSource,
    field: String,
    spec: VectorSpec,
    indexes: Vec<SnapshotIndex>,
    delta_files: Vec<u64>,
    session: VortexSession,
}

impl SnapshotVectorSearch {
    /// Validate a complete base/delta query plan for a table-owned snapshot.
    ///
    /// Every file must occur exactly once, either in base index coverage or in
    /// `delta_files`. An initial delta-only snapshot is allowed when all its files
    /// are explicitly listed. Changed base files require a new generation or an
    /// explicit new plan; this constructor never inserts an automatic scan.
    pub fn try_new(
        source: VisibleSource,
        field: String,
        spec: VectorSpec,
        indexes: Vec<SnapshotIndex>,
        delta_files: Vec<u64>,
        session: VortexSession,
    ) -> VortexResult<Self> {
        if field.trim().is_empty()
            || field.contains('\0')
            || spec.metric != DistanceMetric::SquaredL2
        {
            vortex_bail!("Snapshot vector search requires a field and squared L2");
        }
        let mut covered = BTreeSet::new();
        for entry in &indexes {
            let metadata = entry.index.metadata();
            let vector = entry
                .index
                .as_vector()
                .ok_or_else(|| vortex_err!("Missing vector capability"))?;
            if metadata.snapshot.dataset_id != source.snapshot().dataset_id
                || metadata.snapshot.schema_fingerprint != source.snapshot().schema_fingerprint
                || metadata.fields != [field.clone()]
                || vector.spec() != spec
            {
                vortex_bail!("Index dataset, schema, field or vector specification mismatch");
            }
            entry.validate_coverage(&source.visibility)?;
            for id in &metadata.covered_files {
                if !covered.insert(*id) {
                    vortex_bail!("Snapshot indexes have overlapping file coverage");
                }
            }
        }
        for id in &delta_files {
            if source.snapshot().file(*id).is_none() {
                vortex_bail!("Delta inventory references an unknown file: {id}");
            }
            if !covered.insert(*id) {
                vortex_bail!("Delta inventory duplicates base coverage or another delta file");
            }
        }
        if covered.len() != source.snapshot().files.len() {
            vortex_bail!("Every snapshot file must have explicit base or delta coverage");
        }
        Ok(Self {
            source,
            field,
            spec,
            indexes,
            delta_files,
            session,
        })
    }

    /// Pinned, visibility-enforcing source for projected retrieval of ranked hits.
    pub fn source(&self) -> &VisibleSource {
        &self.source
    }

    /// Search all visible, predicate-matching rows in this snapshot.
    ///
    /// Every base index receives the filter before top-k. Delta vectors are
    /// streamed with only the vector column projected and a bounded top-k heap.
    /// Payload retrieval is separate, through [`Self::source`]. Exact queries
    /// require exact support from every base index; restricted queries require
    /// filtering support. Provider failures propagate without scan fallback.
    pub async fn search(
        &self,
        query: &[f32],
        k: NonZeroUsize,
        mode: SearchMode,
        predicate: &RowFilter,
    ) -> VortexResult<Vec<SearchHit>> {
        if query.len() != self.spec.dimension.get() || query.iter().any(|v| !v.is_finite()) {
            vortex_bail!("Query dimensions or components are invalid");
        }
        let filter = self.source.visibility.intersection(predicate)?;
        let mut best = BinaryHeap::new();
        for entry in &self.indexes {
            let metadata = entry.index.metadata();
            let vector = entry
                .index
                .as_vector()
                .ok_or_else(|| vortex_err!("Missing vector capability"))?;
            let native_filter =
                filter.for_index_snapshot(metadata.snapshot.clone(), &metadata.covered_files)?;
            if mode == SearchMode::Exact && !vector.supports_exact() {
                vortex_bail!("Base index does not support exact search");
            }
            if native_filter.is_restricted() && !vector.supports_filter() {
                vortex_bail!(
                    "Base index does not support the required visibility/predicate filter"
                );
            }
            let hits = vector
                .search(
                    query,
                    &VectorSearchOptions {
                        k,
                        mode,
                        backend_options: entry.backend_options.clone(),
                    },
                    &native_filter,
                )
                .await?;
            if hits.len() > k.get() {
                vortex_bail!("Index returned more than k hits");
            }
            let mut seen = BTreeSet::new();
            for hit in hits {
                self.source.snapshot().validate_row(hit.row)?;
                if !metadata.covers(hit.row.file_id)
                    || !filter.includes(hit.row)
                    || !seen.insert(hit.row)
                    || !hit.distance.is_finite()
                    || hit.distance < 0.0
                {
                    vortex_bail!("Index returned an invalid or invisible hit");
                }
                push_best(&mut best, k.get(), hit);
            }
        }
        if !self.delta_files.is_empty() {
            self.scan(query, k.get(), filter, &self.delta_files, &mut best)
                .await?;
        }
        Ok(best
            .into_sorted_vec()
            .into_iter()
            .map(|hit| hit.0)
            .collect())
    }

    async fn scan(
        &self,
        query: &[f32],
        k: usize,
        filter: RowFilter,
        files: &[u64],
        best: &mut BinaryHeap<Ranked>,
    ) -> VortexResult<()> {
        let source = VisibleSource::try_new(Arc::clone(&self.source.source), filter)?;
        let dimension = u32::try_from(self.spec.dimension.get())?;
        let expected = DType::Struct(
            StructFields::new(
                [self.field.as_str()].into(),
                vec![DType::FixedSizeList(
                    Arc::new(DType::Primitive(PType::F32, Nullability::NonNullable)),
                    dimension,
                    Nullability::NonNullable,
                )],
            ),
            Nullability::NonNullable,
        );
        let mut stream = source.scan(files, std::slice::from_ref(&self.field))?;
        let mut ctx = self.session.create_execution_ctx();
        while let Some(batch) = stream.try_next().await? {
            if batch.data.dtype() != &expected {
                vortex_bail!(
                    "Vector scan requires non-null FixedSizeList<Float32> of the declared dimension"
                );
            }
            let data = batch.data.execute::<StructArray>(&mut ctx)?;
            let vectors = data
                .unmasked_field(0)
                .clone()
                .execute::<FixedSizeListArray>(&mut ctx)?;
            let values = vectors
                .elements()
                .clone()
                .execute::<PrimitiveArray>(&mut ctx)?;
            let values = values.as_slice::<f32>();
            if batch.rows.len().checked_mul(self.spec.dimension.get()) != Some(values.len()) {
                vortex_bail!("Scanned vector dimensions do not match physical rows");
            }
            for (row, values) in batch
                .rows
                .iter()
                .zip(values.chunks_exact(self.spec.dimension.get()))
            {
                if values.iter().any(|v| !v.is_finite()) {
                    vortex_bail!("Vector scan encountered a non-finite component");
                }
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "Search scores are f32; overflow is rejected below"
                )]
                let distance = values
                    .iter()
                    .zip(query)
                    .map(|(a, b)| {
                        let difference = f64::from(*a) - f64::from(*b);
                        difference * difference
                    })
                    .sum::<f64>() as f32;
                if !distance.is_finite() {
                    vortex_bail!("Squared L2 distance exceeds finite f32");
                }
                push_best(
                    best,
                    k,
                    SearchHit {
                        row: *row,
                        distance,
                    },
                );
            }
        }
        Ok(())
    }
}

struct Ranked(SearchHit);
impl PartialEq for Ranked {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Ranked {}
impl PartialOrd for Ranked {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Ranked {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .distance
            .total_cmp(&other.0.distance)
            .then_with(|| self.0.row.cmp(&other.0.row))
    }
}

fn push_best(best: &mut BinaryHeap<Ranked>, k: usize, hit: SearchHit) {
    let hit = Ranked(hit);
    if best.len() < k {
        best.push(hit);
    } else if let Some(mut worst) = best.peek_mut()
        && hit < *worst
    {
        *worst = hit;
    }
}

#[cfg(test)]
#[cfg(feature = "file")]
mod tests;
