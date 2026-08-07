// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;
use std::ops::Range;
use std::sync::Arc;

use bit_vec::BitVec;
use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_array::ArrayRef;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_error::VortexResult;
use vortex_layout::plan::PlanExecutionContext;
use vortex_layout::plan::PlanRef;
use vortex_layout::scan::filter::FilterExpr;
use vortex_mask::Mask;
use vortex_scan::row_mask::RowMask;

pub(crate) type TaskFuture<A> = BoxFuture<'static, VortexResult<A>>;

/// The mask density below which a conjunct is evaluated over only the selected rows.
///
/// Above the threshold, gathering the selected rows costs more than evaluating the predicate over
/// the whole split and intersecting the two masks afterwards. This mirrors the equivalent
/// threshold applied by the `LayoutReader` scan.
const SELECTIVE_FILTER_THRESHOLD: f64 = 0.2;

pub(crate) fn split_exec<A: 'static + Send>(
    ctx: Arc<TaskContext<A>>,
    read_mask: RowMask,
    limit: Option<&mut u64>,
) -> VortexResult<TaskFuture<Option<A>>> {
    let row_range = read_mask.row_range();
    let row_mask = read_mask.mask().clone();
    tracing::trace!(
        target: "vortex_scan_v2::execution",
        ?row_range,
        selected_rows = row_mask.true_count(),
        has_pruning = ctx.pruning.is_some(),
        has_filter = ctx.filter.is_some(),
        "executing a plan scan split"
    );

    let row_mask = match (&ctx.filter, limit) {
        (None, Some(limit)) if *limit == 0 => Mask::new_false(row_mask.len()),
        (None, Some(limit)) => {
            let true_count = row_mask.true_count();
            let mask_limit = usize::try_from(*limit)
                .map(|limit| limit.min(true_count))
                .unwrap_or(true_count);
            let row_mask = row_mask.limit(mask_limit);
            *limit -= mask_limit as u64;
            row_mask
        }
        _ => row_mask,
    };

    Ok(async move {
        let mut row_mask = row_mask;
        if let Some(pruning) = &ctx.pruning {
            let proof = pruning.execute(
                &ctx.execution,
                &row_range,
                MaskFuture::ready(row_mask.clone()),
            )?;
            let proof = proof.await?;
            let mut execution = ctx.execution.session().create_execution_ctx();
            let pruned: Mask = proof.null_as_false().execute(&mut execution)?;
            let pruned_rows = pruned.true_count();
            row_mask = row_mask.intersect_by_rank(&!pruned);
            tracing::trace!(
                target: "vortex_scan_v2::execution",
                ?row_range,
                pruned_rows,
                remaining_rows = row_mask.true_count(),
                "applied the plan pruning proof"
            );
        }

        if row_mask.all_false() {
            tracing::trace!(
                target: "vortex_scan_v2::execution",
                ?row_range,
                "plan pruning skipped the scan split"
            );
            return Ok(None);
        }

        let filter_mask = match &ctx.filter {
            None => MaskFuture::ready(row_mask),
            Some(filter) => {
                let filter = Arc::clone(filter);
                let execution = ctx.execution.clone();
                let filter_range = row_range.clone();
                MaskFuture::new(row_mask.len(), async move {
                    filter.execute(&execution, &filter_range, row_mask).await
                })
            }
        };

        // Register projection reads before resolving the filter mask so segments used by both
        // expressions can share the same in-flight request.
        let projection = ctx
            .projection
            .execute(&ctx.execution, &row_range, filter_mask.clone())?;
        let row_mask = filter_mask.await?;

        if row_mask.all_false() {
            tracing::trace!(
                target: "vortex_scan_v2::execution",
                ?row_range,
                "plan scan split produced no matching rows"
            );
            return Ok(None);
        }

        let array = projection.await?;
        tracing::trace!(
            target: "vortex_scan_v2::execution",
            ?row_range,
            output_rows = array.len(),
            dtype = %array.dtype(),
            "completed a plan scan split"
        );
        (ctx.mapper)(array).map(Some)
    }
    .boxed())
}

/// The conjuncts of a scan filter, each with its own physical plan.
///
/// Splitting the filter lets a split stop as soon as one conjunct rejects every remaining row, and
/// lets later conjuncts see the mask narrowed by the earlier ones. The evaluation order is chosen
/// from the selectivity measured by previous splits.
pub(crate) struct FilterPlan {
    expression: FilterExpr,
    /// One plan per conjunct, in the same order as [`FilterExpr::conjuncts`].
    conjuncts: Vec<PlanRef>,
}

impl FilterPlan {
    pub(crate) fn new(expression: FilterExpr, conjuncts: Vec<PlanRef>) -> Self {
        debug_assert_eq!(
            expression.conjuncts().len(),
            conjuncts.len(),
            "Every filter conjunct must have a plan"
        );
        Self {
            expression,
            conjuncts,
        }
    }

    /// Returns the physical plan of each conjunct.
    pub(crate) fn conjunct_plans(&self) -> &[PlanRef] {
        &self.conjuncts
    }

    /// Evaluates every conjunct against `row_mask`, returning the rows matching all of them.
    async fn execute(
        &self,
        execution: &PlanExecutionContext,
        row_range: &Range<u64>,
        row_mask: Mask,
    ) -> VortexResult<Mask> {
        let mut mask = row_mask;
        let mut remaining = BitVec::from_elem(self.conjuncts.len(), true);
        while let Some(index) = self.expression.next_conjunct(&remaining) {
            remaining.set(index, false);
            if mask.all_false() {
                return Ok(mask);
            }

            // A sparse mask is cheaper to gather than a full-width evaluation is to discard, so
            // push it into the conjunct's plan. A dense mask is cheaper to apply afterwards.
            let selective = mask.density() < SELECTIVE_FILTER_THRESHOLD;
            let input = if selective {
                MaskFuture::ready(mask.clone())
            } else {
                MaskFuture::new_true(mask.len())
            };

            let predicate = self.conjuncts[index]
                .execute(execution, row_range, input)?
                .await?;
            let mut ctx = execution.session().create_execution_ctx();
            let predicate: Mask = predicate.null_as_false().execute(&mut ctx)?;

            mask = if selective {
                mask.intersect_by_rank(&predicate)
            } else {
                mask.bitand(&predicate)
            };
            self.expression.report_selectivity(index, mask.density());
        }
        Ok(mask)
    }
}

pub(crate) struct TaskContext<A> {
    pub(crate) execution: PlanExecutionContext,
    pub(crate) pruning: Option<PlanRef>,
    pub(crate) filter: Option<Arc<FilterPlan>>,
    pub(crate) projection: PlanRef,
    pub(crate) mapper: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
}
