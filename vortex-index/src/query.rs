// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::num::NonZeroUsize;

use bytes::Bytes;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_scan::selection::Selection;
use vortex_scan::strict_sorted_buffer::StrictSortedBuffer;

use crate::RowAddress;
use crate::Snapshot;

/// Distance definition. Backends must reject unsupported metrics explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DistanceMetric {
    /// Sum of squared component differences; smaller is closer.
    SquaredL2,
    /// One minus cosine similarity; zero-norm vectors must be rejected.
    Cosine,
    /// Negated dot product; smaller is closer.
    NegativeDot,
}

/// Common properties of a vector index; native dtype/quantization are backend details.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VectorSpec {
    /// Number of components in each query vector.
    pub dimension: NonZeroUsize,
    /// Distance definition used for search and result merging.
    pub metric: DistanceMetric,
}

/// Whether approximate candidate retrieval is permitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    /// Require exact nearest neighbors within the covered and allowed rows.
    Exact,
    /// Permit approximate retrieval, but never relax visibility or filtering.
    Approximate,
}

/// Algorithm-independent options and an explicitly backend-owned options payload.
#[derive(Debug, Clone)]
pub struct VectorSearchOptions {
    /// Maximum number of visible, allowed hits per query.
    pub k: NonZeroUsize,
    /// Exact SQL top-k cannot be transparently changed into approximate search.
    pub mode: SearchMode,
    /// Backend-defined request payload, not interpreted by the common layer.
    pub backend_options: Bytes,
}

/// One ranked hit. Batch APIs retain query identity through their outer result order.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchHit {
    /// Physical address in the index snapshot, never a native ANN identifier.
    pub row: RowAddress,
    /// Finite distance under the index's declared metric.
    pub distance: f32,
}

/// Snapshot-bound allowed rows and mandatory exclusions (including tombstones).
///
/// `None` allows all rows, while an empty allow list allows none. Exclusions win.
/// The table/source adapter must supply the complete snapshot visibility mask.
#[derive(Debug, Clone)]
pub struct RowFilter {
    snapshot: Snapshot,
    allow: Option<BTreeSet<RowAddress>>,
    exclude: BTreeSet<RowAddress>,
}

impl RowFilter {
    /// Construct a filter, rejecting addresses outside the snapshot.
    pub fn try_new(
        snapshot: Snapshot,
        allow: Option<BTreeSet<RowAddress>>,
        exclude: BTreeSet<RowAddress>,
    ) -> VortexResult<Self> {
        snapshot.validate()?;
        for row in allow.iter().flatten().chain(&exclude) {
            snapshot.validate_row(*row)?;
        }
        Ok(Self {
            snapshot,
            allow,
            exclude,
        })
    }

    /// Snapshot defining both the physical addresses and row visibility.
    pub fn snapshot(&self) -> &Snapshot {
        &self.snapshot
    }

    /// Whether a valid physical address passes the predicate and visibility mask.
    pub fn includes(&self, row: RowAddress) -> bool {
        self.allow.as_ref().is_none_or(|allow| allow.contains(&row)) && !self.exclude.contains(&row)
    }

    /// Whether a backend needs filtered-search support for this request.
    pub fn is_restricted(&self) -> bool {
        self.allow.is_some() || !self.exclude.is_empty()
    }

    /// Intersect predicates/visibility from the same snapshot. Exclusions always win.
    pub fn intersection(&self, other: &Self) -> VortexResult<Self> {
        if self.snapshot != other.snapshot {
            vortex_bail!("Cannot intersect row filters from different snapshots");
        }
        let allow = match (&self.allow, &other.allow) {
            (Some(left), Some(right)) => Some(left.intersection(right).copied().collect()),
            (Some(rows), None) | (None, Some(rows)) => Some(rows.clone()),
            (None, None) => None,
        };
        Self::try_new(
            self.snapshot.clone(),
            allow,
            self.exclude.union(&other.exclude).copied().collect(),
        )
    }

    #[cfg(feature = "snapshot")]
    pub(crate) fn covered_subset_of(&self, old: &Self, files: &[u64]) -> bool {
        let covered = |row: &RowAddress| files.binary_search(&row.file_id).is_ok();
        if let Some(allow) = &self.allow {
            return allow
                .iter()
                .filter(|row| covered(row) && self.includes(**row))
                .all(|row| old.includes(*row));
        }
        if let Some(allow) = &old.allow {
            // Count visible rows rather than enumerating every physical row. The
            // caller has already proved that covered file identities are equal.
            return files.iter().all(|id| {
                let Some(file) = self.snapshot.file(*id) else {
                    return false;
                };
                let visible = file.row_count
                    - self.exclude.iter().filter(|row| row.file_id == *id).count() as u64;
                let shared = allow
                    .iter()
                    .filter(|row| row.file_id == *id && old.includes(**row) && self.includes(**row))
                    .count() as u64;
                visible == shared
            });
        }
        old.exclude
            .iter()
            .filter(|row| covered(row))
            .all(|row| !self.includes(*row))
    }

    #[cfg(feature = "snapshot")]
    pub(crate) fn for_index_snapshot(
        &self,
        snapshot: Snapshot,
        files: &[u64],
    ) -> VortexResult<Self> {
        let covered = |row: &&RowAddress| files.binary_search(&row.file_id).is_ok();
        Self::try_new(
            snapshot,
            self.allow
                .as_ref()
                .map(|rows| rows.iter().filter(covered).copied().collect()),
            self.exclude.iter().filter(covered).copied().collect(),
        )
    }
}

/// Meaning of a scalar index's candidate set within its declared coverage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateGuarantee {
    /// Every candidate satisfies the predicate and every matching row is included.
    Exact,
    /// No matching rows are omitted, but the predicate must be re-evaluated on candidates.
    Superset,
}

/// Scalar candidates in one snapshot. Coverage comes from the index metadata.
#[derive(Debug, Clone)]
pub struct ScalarCandidates {
    /// Snapshot against which the candidate addresses were obtained.
    pub snapshot: Snapshot,
    /// Candidate physical addresses, after mandatory visibility filtering.
    pub rows: Vec<RowAddress>,
    /// Whether residual predicate evaluation is required.
    pub guarantee: CandidateGuarantee,
}

impl ScalarCandidates {
    /// Group, sort, and deduplicate candidates for per-file Vortex scans.
    ///
    /// Selections are relative to the whole file and must be applied independently
    /// per file, not broadcast to every partition of a multi-file scan. This loses
    /// input ordering; ranked search must retain its hit list to restore rank later.
    pub fn selections(&self, snapshot: &Snapshot) -> VortexResult<BTreeMap<u64, Selection>> {
        snapshot.validate()?;
        if self.snapshot != *snapshot {
            vortex_bail!("Candidate snapshot does not match the scan snapshot");
        }
        let mut rows: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
        for row in &self.rows {
            snapshot.validate_row(*row)?;
            rows.entry(row.file_id).or_default().push(row.row_offset);
        }
        rows.into_iter()
            .map(|(file, mut offsets)| {
                offsets.sort_unstable();
                offsets.dedup();
                Ok((
                    file,
                    Selection::IncludeByIndex(StrictSortedBuffer::try_new(Buffer::from_iter(
                        offsets,
                    ))?),
                ))
            })
            .collect()
    }
}

/// A full-text hit. Scores are BM25 relevance, ordered descending.
#[derive(Debug, Clone, PartialEq)]
pub struct FullTextHit {
    /// Physical row address in the pinned source snapshot.
    pub row: RowAddress,
    /// Finite relevance score; higher is better.
    pub score: f32,
}
