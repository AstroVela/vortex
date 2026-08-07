// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;
use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_array::ArrayRef;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_error::VortexResult;
use vortex_layout::plan::PlanExecutionContext;
use vortex_layout::plan::PlanRef;
use vortex_mask::Mask;
use vortex_scan::row_mask::RowMask;

pub(crate) type TaskFuture<A> = BoxFuture<'static, VortexResult<A>>;

/// The mask density below which a filter conjunct is evaluated only over the surviving rows,
/// letting leaves filter their arrays before expression evaluation.
const FILTER_EVAL_DENSITY_THRESHOLD: f64 = 0.2;

fn sparse_filter_input(mask: &Mask) -> bool {
    !mask.all_true() && mask.density() < FILTER_EVAL_DENSITY_THRESHOLD
}

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
        conjuncts = ctx.filter.len(),
        "executing a plan scan split"
    );

    let row_mask = match (ctx.filter.is_empty(), limit) {
        (true, Some(limit)) if *limit == 0 => Mask::new_false(row_mask.len()),
        (true, Some(limit)) => {
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
    let split_len = row_mask.len();

    // NOTE: every plan execution is registered OUTSIDE the returned future. Registering leaf
    // segment reads eagerly - before any split task is polled - lets the IO system coalesce
    // reads across expressions and across splits. Registered reads trigger no IO until their
    // futures are polled, and dropped futures are canceled when possible.
    let mut filter_mask = if let Some(pruning) = &ctx.pruning {
        let proof = pruning.execute(
            &ctx.execution,
            &row_range,
            MaskFuture::ready(row_mask.clone()),
        )?;
        let session = ctx.execution.session().clone();
        let pruning_range = row_range.clone();
        MaskFuture::new(split_len, async move {
            let proof = proof.await?;
            let mut execution = session.create_execution_ctx();
            let pruned: Mask = proof.null_as_false().execute(&mut execution)?;
            let pruned_rows = pruned.true_count();
            let row_mask = if pruned.all_false() {
                row_mask
            } else if pruned.all_true() {
                Mask::new_false(row_mask.len())
            } else {
                row_mask.intersect_by_rank(&!pruned)
            };
            tracing::trace!(
                target: "vortex_scan_v2::execution",
                row_range = ?pruning_range,
                pruned_rows,
                remaining_rows = row_mask.true_count(),
                "applied the plan pruning proof"
            );
            Ok(row_mask)
        })
    } else {
        MaskFuture::ready(row_mask)
    };

    // Evaluate each filter conjunct over the rows surviving the previous conjuncts. When the
    // surviving mask is sparse, the conjunct is evaluated only over the surviving rows so leaves
    // filter their arrays before the expression runs; otherwise it is evaluated over the full
    // split and intersected afterwards. The same density decision is recomputed from the same
    // resolved mask in both futures, so they always agree.
    for conjunct in &ctx.filter {
        let predicate_input = {
            let input_mask = filter_mask.clone();
            MaskFuture::new(split_len, async move {
                let row_mask = input_mask.await?;
                Ok(if sparse_filter_input(&row_mask) {
                    row_mask
                } else {
                    Mask::new_true(row_mask.len())
                })
            })
        };
        let predicate = conjunct.execute(&ctx.execution, &row_range, predicate_input)?;
        let session = ctx.execution.session().clone();
        let input_mask = filter_mask.clone();
        filter_mask = MaskFuture::new(split_len, async move {
            let row_mask = input_mask.await?;
            if row_mask.all_false() {
                return Ok(row_mask);
            }
            let predicate = predicate.await?;
            let mut execution = session.create_execution_ctx();
            let predicate: Mask = predicate.null_as_false().execute(&mut execution)?;
            if sparse_filter_input(&row_mask) {
                Ok(row_mask.intersect_by_rank(&predicate))
            } else {
                Ok(row_mask.bitand(&predicate))
            }
        });
    }

    // Register projection reads before any mask resolves so segments used by several
    // expressions can share the same in-flight request.
    let projection = ctx
        .projection
        .execute(&ctx.execution, &row_range, filter_mask.clone())?;

    let mapper = Arc::clone(&ctx.mapper);
    Ok(async move {
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
        mapper(array).map(Some)
    }
    .boxed())
}

pub(crate) struct TaskContext<A> {
    pub(crate) execution: PlanExecutionContext,
    pub(crate) pruning: Option<PlanRef>,
    pub(crate) filter: Vec<PlanRef>,
    pub(crate) projection: PlanRef,
    pub(crate) mapper: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
}
