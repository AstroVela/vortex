// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_index::RowAddress;
use vortex_index::RowFilter;
use vortex_index::SearchHit;

use super::RaBitQErrorBound;
use super::distance::ExactQuery;
use super::storage::QuantizedIndex;

#[derive(Debug, Default)]
pub(crate) struct SearchStats {
    pub scanned: usize,
    pub retained: usize,
    pub reranked: usize,
    pub original_bytes: usize,
    pub original_read_calls: usize,
    pub original_read_bytes: usize,
    pub original_cache_bytes: usize,
    pub exact_fallbacks: usize,
}

#[derive(PartialEq)]
struct Ranked(SearchHit);
impl Eq for Ranked {}
impl Ord for Ranked {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .distance
            .total_cmp(&other.0.distance)
            .then_with(|| self.0.row.cmp(&other.0.row))
    }
}
impl PartialOrd for Ranked {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// Scores use the common f32 API. Compare rounded lower bounds to rounded heap
// scores so a different f64 distance that rounds to a tie cannot be pruned.
#[expect(
    clippy::cast_possible_truncation,
    reason = "The shared search API returns f32 distances"
)]
pub(super) fn score(value: f64) -> f32 {
    value as f32
}

impl QuantizedIndex {
    pub(crate) fn search(
        &self,
        query: &[f32],
        heads: &[u32],
        rows: &[RowAddress],
        filter: &RowFilter,
        k: usize,
        bound: RaBitQErrorBound,
    ) -> VortexResult<(Vec<SearchHit>, SearchStats)> {
        if rows.len() != self.codes.len() || k == 0 {
            vortex_bail!("Invalid RaBitQ search shape");
        }
        let prepared = self.quantizer.query(query, bound)?;
        let mut seen = vec![0u8; self.codes.len().div_ceil(8)];
        let mut ids = Vec::new();
        let mut stats = SearchStats::default();
        let restricted = filter.is_restricted();
        for head in heads {
            let head = *head as usize;
            if head >= self.head_ids.len() {
                vortex_bail!("RaBitQ head outside routing table");
            }
            for id in std::iter::once(&self.head_ids[head]).chain(&self.postings[head]) {
                let index = *id as usize;
                let bit = 1 << (index % 8);
                if seen[index / 8] & bit != 0 || (restricted && !filter.includes(rows[index])) {
                    continue;
                }
                seen[index / 8] |= bit;
                stats.scanned += 1;
                ids.push(*id);
            }
        }
        let mut candidates = Vec::with_capacity(ids.len());
        let (batches, tail) = ids.as_chunks::<4>();
        for batch in batches {
            candidates.extend(
                batch
                    .iter()
                    .copied()
                    .zip(self.codes.estimate_four(&prepared, *batch)?),
            );
        }
        for id in tail {
            candidates.push((*id, self.codes.estimate(&prepared, *id as usize)?));
        }
        // An interval upper bound supplies a threshold before any original IO.
        // In confidence mode both this threshold and lower bounds are probabilistic.
        let limit = if candidates.len() >= k {
            candidates
                .select_nth_unstable_by(k - 1, |left, right| {
                    left.1.upper().total_cmp(&right.1.upper())
                })
                .1
                .1
                .upper()
        } else {
            f64::INFINITY
        };
        candidates.retain(|(_, interval)| score(interval.lower()) <= score(limit));
        stats.retained = candidates.len();
        candidates.sort_unstable_by(|left, right| {
            left.1
                .estimate
                .total_cmp(&right.1.estimate)
                .then_with(|| rows[left.0 as usize].cmp(&rows[right.0 as usize]))
        });
        if candidates.is_empty() {
            return Ok((Vec::new(), stats));
        }
        let mut best = BinaryHeap::<Ranked>::with_capacity(k.min(candidates.len()));
        // Dense survivors amortize chunk reads; sparse reranks avoid prefetching unused vectors.
        let cache_bytes = if stats.retained * 2 >= stats.scanned {
            64 * 1024 * 1024
        } else {
            0
        };
        let mut originals = self.original_reader(cache_bytes)?;
        let exact = ExactQuery::new(query)?;
        for (id, interval) in candidates {
            if best.len() == k
                && best
                    .peek()
                    .is_some_and(|worst| score(interval.lower()) > worst.0.distance)
            {
                continue;
            }
            let (distance, fallback) = originals.exact_score(id, &exact)?;
            stats.exact_fallbacks += usize::from(fallback);
            stats.reranked += 1;
            stats.original_bytes += query.len() * 4;
            if !distance.is_finite() {
                vortex_bail!("Squared L2 distance exceeds finite f32 range");
            }
            let hit = Ranked(SearchHit {
                row: rows[id as usize],
                distance,
            });
            if best.len() < k {
                best.push(hit);
            } else if best.peek().is_some_and(|worst| hit < *worst) {
                best.pop();
                best.push(hit);
            }
        }
        stats.original_read_calls = originals.read_calls;
        stats.original_read_bytes = originals.read_bytes;
        stats.original_cache_bytes = originals.cached_bytes;
        Ok((
            best.into_sorted_vec()
                .into_iter()
                .map(|hit| hit.0)
                .collect(),
            stats,
        ))
    }
}
