// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::StreamExt;
use futures::stream::BoxStream;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_array::expr::forms::conjuncts;
use vortex_error::VortexResult;
use vortex_io::session::RuntimeSessionExt;
use vortex_scan::selection::Selection;
use vortex_session::VortexSession;
use vortex_utils::parallelism::get_available_parallelism;

use crate::LayoutReaderContext;
use crate::LayoutReaderRef;
use crate::LayoutRef;
use crate::layouts::row_idx::RowIdx;
use crate::reader_plan::ExpressionPlan;
use crate::reader_plan::RowIdxPlan;
use crate::scan::filter::FilterExpr;
use crate::scan::scan_builder::ScanBuilder;
use crate::scan::tasks::ReaderPlanTaskContext;
use crate::scan::tasks::reader_plan_split_exec;
use crate::segments::SegmentSource;

/// A prepared layout-v27 scan using expression-specific reader trees.
pub struct ReaderPlanScan {
    session: VortexSession,
    context: Arc<ReaderPlanTaskContext>,
    splits: Arc<[u64]>,
    concurrency: usize,
    dtype: DType,
}

impl ReaderPlanScan {
    /// Plan a projection and optional filter over a layout using the layout-v27 rules.
    pub fn try_new(
        session: VortexSession,
        layout: &LayoutRef,
        legacy_reader: LayoutReaderRef,
        segment_source: Arc<dyn SegmentSource>,
        projection: &Expression,
        filter: Option<&Expression>,
    ) -> VortexResult<Self> {
        let projection = projection.optimize_recursive(layout.dtype())?;
        let filter = filter
            .map(|expression| expression.optimize_recursive(layout.dtype()))
            .transpose()?;
        let bound_projection = projection.bind(legacy_reader.dtype())?;
        let bound_filter = filter
            .as_ref()
            .map(|expression| expression.bind(legacy_reader.dtype()))
            .transpose()?;
        let split_builder = ScanBuilder::new(session.clone(), legacy_reader)
            .with_projection(bound_projection)
            .with_some_filter(bound_filter.clone());
        let mut splits = split_builder.full_file_splits()?;
        if splits.first().copied() != Some(0) {
            splits.insert(0, 0);
        }
        if splits.last().copied() != Some(layout.row_count()) {
            splits.push(layout.row_count());
        }
        let mut source = layout.new_reader_plan()?;
        let projection_uses_row_idx = projection.contains::<RowIdx>()?;
        let filter_uses_row_idx = filter
            .as_ref()
            .map(|expression| expression.contains::<RowIdx>())
            .transpose()?
            .unwrap_or(false);
        if projection_uses_row_idx || filter_uses_row_idx {
            source = RowIdxPlan::new_ref(0, source);
        }
        let projection_plan =
            ExpressionPlan::new_ref(projection, Arc::clone(&source))?.optimize()?;
        let predicate_plans = filter
            .as_ref()
            .map(conjuncts)
            .unwrap_or_default()
            .into_iter()
            .map(|expression| ExpressionPlan::new_ref(expression, Arc::clone(&source))?.optimize())
            .collect::<VortexResult<Vec<_>>>()?;
        let reader_context = LayoutReaderContext::default();
        let projection = projection_plan.new_reader(
            "layout-v27-projection".into(),
            Arc::clone(&segment_source),
            &session,
            &reader_context,
        )?;
        let predicates = predicate_plans
            .iter()
            .enumerate()
            .map(|(index, predicate)| {
                predicate.new_reader(
                    format!("layout-v27-predicate-{index}").into(),
                    Arc::clone(&segment_source),
                    &session,
                    &reader_context,
                )
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let predicate_roots = predicates
            .iter()
            .map(|predicate| {
                vortex_array::expr::BoundExpression::new_root(predicate.dtype().clone())
            })
            .collect();
        let projection_root =
            vortex_array::expr::BoundExpression::new_root(projection.dtype().clone());
        let dtype = projection.dtype().clone();

        Ok(Self {
            session,
            context: Arc::new(ReaderPlanTaskContext {
                filter: bound_filter.map(|filter| Arc::new(FilterExpr::new(filter))),
                predicates,
                predicate_roots,
                projection,
                projection_root,
            }),
            splits: splits.into(),
            concurrency: 4,
            dtype,
        })
    }

    /// Set the per-worker split concurrency.
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        assert!(concurrency > 0);
        self.concurrency = concurrency;
        self
    }

    /// Return the projected dtype.
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    /// Execute the planned scan as an ordered stream.
    pub fn into_stream(self) -> VortexResult<BoxStream<'static, VortexResult<ArrayRef>>> {
        let tasks = self
            .splits
            .windows(2)
            .filter(|window| window[0] < window[1])
            .map(|window| {
                let range = window[0]..window[1];
                reader_plan_split_exec(Arc::clone(&self.context), Selection::All.row_mask(&range))
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let concurrency = self.concurrency * get_available_parallelism().unwrap_or(1);
        let handle = self.session.handle();
        let stream = futures::stream::iter(tasks)
            .map(move |task| handle.spawn(task))
            .buffered(concurrency)
            .filter_map(|chunk| async move { chunk.transpose() })
            .boxed();
        Ok(stream)
    }
}
