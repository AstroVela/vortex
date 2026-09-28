// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;

use async_trait::async_trait;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::DistanceMetric;
use crate::Index;
use crate::IndexMetadata;
use crate::RowAddress;
use crate::RowFilter;
use crate::SearchHit;
use crate::VectorIndex;
use crate::VectorSearchOptions;
use crate::VectorSpec;

/// In-memory exact squared-L2 reference for small, dense file coverage.
///
/// This is an oracle/conformance implementation, not a production ANN backend.
/// It materializes all rows and sorts all allowed distances on each search.
/// Scores accumulate in f64 and round to f32; distance overflow is an error.
#[derive(Debug)]
pub struct FlatIndex {
    metadata: IndexMetadata,
    spec: VectorSpec,
    rows: Vec<RowAddress>,
    values: Buffer<f32>,
}

impl FlatIndex {
    /// Backend identity used by this reference implementation.
    pub const ID: &'static str = "vortex.flat.reference";

    /// Construct from every physical row of each covered file, with no NULL vectors.
    ///
    /// Row input order is arbitrary, but duplicates, missing rows, wrong dimensions,
    /// non-finite values, unsupported metrics, and persisted artifacts are rejected.
    pub fn try_new(
        metadata: IndexMetadata,
        spec: VectorSpec,
        rows: Vec<RowAddress>,
        values: Buffer<f32>,
    ) -> VortexResult<Self> {
        metadata.validate_for(&metadata.snapshot)?;
        if metadata.backend != Self::ID
            || metadata.backend_version != 1
            || !metadata.artifacts.is_empty()
        {
            vortex_bail!("Flat reference requires its own backend ID, version 1, and no artifacts");
        }
        if metadata.fields.len() != 1 || spec.metric != DistanceMetric::SquaredL2 {
            vortex_bail!("Flat reference supports one vector field and squared L2 only");
        }
        if rows.len().checked_mul(spec.dimension.get()) != Some(values.len())
            || values.iter().any(|value| !value.is_finite())
        {
            vortex_bail!("Invalid flat vector dimensions or non-finite values");
        }
        let mut distinct = BTreeSet::new();
        for row in &rows {
            metadata.snapshot.validate_row(*row)?;
            if !metadata.covers(row.file_id) || !distinct.insert(*row) {
                vortex_bail!("Flat rows must be unique and inside index coverage");
            }
        }
        let expected = metadata
            .snapshot
            .files
            .iter()
            .filter(|file| metadata.covers(file.id))
            .try_fold(0u64, |count, file| count.checked_add(file.row_count));
        if expected != Some(u64::try_from(rows.len())?) {
            vortex_bail!("Flat reference requires every physical row of its covered files");
        }
        Ok(Self {
            metadata,
            spec,
            rows,
            values,
        })
    }
}

impl Index for FlatIndex {
    fn metadata(&self) -> &IndexMetadata {
        &self.metadata
    }

    fn as_vector(&self) -> Option<&dyn VectorIndex> {
        Some(self)
    }
}

#[async_trait]
impl VectorIndex for FlatIndex {
    fn spec(&self) -> VectorSpec {
        self.spec
    }

    fn supports_exact(&self) -> bool {
        true
    }

    fn supports_filter(&self) -> bool {
        true
    }

    async fn search(
        &self,
        query: &[f32],
        options: &VectorSearchOptions,
        filter: &RowFilter,
    ) -> VortexResult<Vec<SearchHit>> {
        self.metadata.validate_for(filter.snapshot())?;
        if query.len() != self.spec.dimension.get() || query.iter().any(|value| !value.is_finite())
        {
            vortex_bail!("Invalid query dimensions or non-finite values");
        }
        if !options.backend_options.is_empty() {
            vortex_bail!("Flat reference accepts no backend search options");
        }
        let mut hits = Vec::new();
        for (row, vector) in self
            .rows
            .iter()
            .zip(self.values.chunks_exact(self.spec.dimension.get()))
        {
            if !filter.includes(*row) {
                continue;
            }
            #[expect(
                clippy::cast_possible_truncation,
                reason = "The result API uses f32 scores; overflow is rejected immediately below"
            )]
            let distance = vector
                .iter()
                .zip(query)
                .map(|(left, right)| {
                    let difference = f64::from(*left) - f64::from(*right);
                    difference * difference
                })
                .sum::<f64>() as f32;
            if !distance.is_finite() {
                vortex_bail!("Squared L2 distance exceeds the finite f32 range");
            }
            hits.push(SearchHit {
                row: *row,
                distance,
            });
        }
        hits.sort_unstable_by(|left, right| {
            left.distance
                .total_cmp(&right.distance)
                .then_with(|| left.row.cmp(&right.row))
        });
        hits.truncate(options.k.get());
        Ok(hits)
    }
}
