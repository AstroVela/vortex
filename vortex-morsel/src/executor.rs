// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ScanBuilder`](vortex_layout::scan::scan_builder::ScanBuilder) integration.

use std::ops::Range;
use std::sync::Arc;

use futures::future::BoxFuture;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::Expression;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::LayoutRef;
use vortex_layout::scan::scan_builder::ScanExecutor;
use vortex_layout::scan::scan_builder::ScanRequest;
use vortex_layout::segments::SegmentSource;
use vortex_mask::AllOr;

use crate::MorselScan;
use crate::build::build_plan;
use crate::driver::morsels;
use crate::nodes::ConjunctMode;

/// Morsel-driven execution backend for a layout scan builder.
pub struct MorselScanExecutor {
    layout: LayoutRef,
    segments: Arc<dyn SegmentSource>,
    target_rows: u64,
    conjunct_mode: ConjunctMode,
}

impl MorselScanExecutor {
    /// Create an executor over a raw layout and its segment source.
    pub fn new(layout: LayoutRef, segments: Arc<dyn SegmentSource>) -> Self {
        Self {
            layout,
            segments,
            target_rows: 128 * 1024,
            conjunct_mode: ConjunctMode::Cascade,
        }
    }

    /// Set the target number of rows per morsel.
    pub fn with_target_rows(mut self, target_rows: u64) -> Self {
        self.target_rows = target_rows;
        self
    }

    /// Set the conjunct evaluation policy.
    pub fn with_conjunct_mode(mut self, conjunct_mode: ConjunctMode) -> Self {
        self.conjunct_mode = conjunct_mode;
        self
    }
}

impl ScanExecutor for MorselScanExecutor {
    fn build(
        &self,
        request: ScanRequest,
    ) -> VortexResult<Vec<BoxFuture<'static, VortexResult<Option<ArrayRef>>>>> {
        if request.limit.is_some() {
            vortex_bail!("the morsel scan executor does not support limits");
        }
        if request.row_offset != 0 {
            vortex_bail!("the morsel scan executor does not support row offsets");
        }

        let projection = unbind(&request.projection)?;
        let filter = request.filter.as_ref().map(unbind).transpose()?;
        let plan = Arc::new(build_plan(
            &self.layout,
            &projection,
            filter.as_ref(),
            self.conjunct_mode,
        )?);

        let full_range = request
            .row_range
            .clone()
            .unwrap_or_else(|| 0..plan.row_count());
        let morsels = selected_morsels(
            morsels(&plan, self.target_rows),
            &full_range,
            &request.selection,
        );

        morsels
            .into_iter()
            .map(|morsel| {
                let pruning = request.filter.as_ref().map(|filter| {
                    request.layout_reader.pruning_evaluation(
                        &morsel.range,
                        filter,
                        request.selection.row_mask(&morsel.range).mask().clone(),
                    )
                });
                let pruning = pruning.transpose()?;
                let plan = Arc::clone(&plan);
                let segments = Arc::clone(&self.segments);
                let session = request.session.clone();
                let handle = request.session.handle();

                Ok(Box::pin(async move {
                    if let Some(pruning) = pruning
                        && pruning.await?.all_false()
                    {
                        return Ok(None);
                    }

                    // The driver coordinates its own affinity workers while awaiting their
                    // completion. Keep that blocking coordinator off single-threaded async
                    // runtimes so file/object-store IO can continue making progress.
                    handle
                        .spawn_blocking(move || {
                            let (mut batches, _) = MorselScan::new(plan, segments, session)
                                .with_threads(1)
                                .with_morsels(morsel.selected_ranges)
                                .run()?;
                            match batches.len() {
                                0 => Ok(None),
                                1 => Ok(batches.pop()),
                                _ => {
                                    let dtype = batches[0].dtype().clone();
                                    Ok(Some(ChunkedArray::try_new(batches, dtype)?.into_array()))
                                }
                            }
                        })
                        .await
                })
                    as BoxFuture<'static, VortexResult<Option<ArrayRef>>>)
            })
            .collect()
    }
}

fn unbind(expr: &BoundExpression) -> VortexResult<Expression> {
    let Some(scalar_fn) = expr.as_scalar() else {
        return Ok(Expression::Root);
    };
    Expression::try_new(
        scalar_fn.clone(),
        expr.children()
            .iter()
            .map(unbind)
            .collect::<VortexResult<Vec<_>>>()?,
    )
}

struct SelectedMorsel {
    range: Range<u64>,
    selected_ranges: Vec<Range<u64>>,
}

fn selected_morsels(
    morsels: Vec<Range<u64>>,
    row_range: &Range<u64>,
    selection: &vortex_scan::selection::Selection,
) -> Vec<SelectedMorsel> {
    morsels
        .into_iter()
        .filter_map(|range| {
            let start = range.start.max(row_range.start);
            let end = range.end.min(row_range.end);
            (start < end).then_some(start..end)
        })
        .filter_map(|range| {
            let mask = selection.row_mask(&range);
            let selected_ranges = match mask.mask().slices() {
                AllOr::All => vec![range.clone()],
                AllOr::None => Vec::new(),
                AllOr::Some(slices) => slices
                    .iter()
                    .map(|&(start, end)| range.start + start as u64..range.start + end as u64)
                    .collect(),
            };
            (!selected_ranges.is_empty()).then_some(SelectedMorsel {
                range,
                selected_ranges,
            })
        })
        .collect()
}
