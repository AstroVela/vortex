// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Split scanning task implementation.

use std::ops::BitAnd;
use std::sync::Arc;

use bit_vec::BitVec;
use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_array::ArrayRef;
use vortex_array::MaskFuture;
use vortex_array::expr::BoundExpression;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scan::row_mask::RowMask;

use crate::LayoutReader;
use crate::scan::filter::FilterExpr;

pub type TaskFuture<A> = BoxFuture<'static, VortexResult<A>>;

/// Logic for executing a single split reading task.
/// N.B. read_mask should be evaluated against all_false() before calling this
/// method to avoid creating an empty TaskFuture.
///
/// # Task execution flow
///
/// First, the task's row range (split) is intersected with the global file row-range requested,
/// if any.
///
/// The intersected row range is then further reduced via expression-based pruning. After pruning
/// has eliminated more blocks, the full filter is executed over the remainder of the split.
///
/// This mask is then provided to the reader to perform a filtered projection over the split data,
/// finally mapping the Vortex columnar record batches into some result type `A`.
pub fn split_exec<A: 'static + Send>(
    ctx: Arc<TaskContext<A>>,
    read_mask: RowMask,
    limit: Option<&mut u64>,
) -> VortexResult<TaskFuture<Option<A>>> {
    let row_range = read_mask.row_range();
    let filter_mask = filter_mask(Arc::clone(&ctx), &read_mask, limit)?;
    project_exec_with_mask(ctx, row_range, filter_mask)
}

/// Evaluate pruning and the exact filter for one physical filter range.
pub fn filter_exec<A: 'static + Send>(
    ctx: Arc<TaskContext<A>>,
    read_mask: RowMask,
) -> VortexResult<TaskFuture<RowMask>> {
    let row_offset = read_mask.row_range().start;
    let filter_mask = filter_mask(ctx, &read_mask, None)?;
    Ok(async move { Ok(RowMask::new(row_offset, filter_mask.await?)) }.boxed())
}

/// Project one physical projection range after its exact filter mask is known.
pub fn project_exec<A: 'static + Send>(
    ctx: Arc<TaskContext<A>>,
    read_mask: RowMask,
) -> VortexResult<TaskFuture<Option<A>>> {
    if read_mask.mask().all_false() {
        return Ok(async { Ok(None) }.boxed());
    }

    let row_range = read_mask.row_range();
    let mask = MaskFuture::ready(read_mask.mask().clone());
    project_exec_with_mask(ctx, row_range, mask)
}

fn filter_mask<A: 'static + Send>(
    ctx: Arc<TaskContext<A>>,
    read_mask: &RowMask,
    limit: Option<&mut u64>,
) -> VortexResult<MaskFuture> {
    let row_range = read_mask.row_range();
    let row_mask = read_mask.mask().clone();

    Ok(match ctx.filter.as_ref() {
        // No filter == immediate mask
        None => {
            let row_mask = match limit {
                Some(l) if *l == 0 => Mask::new_false(row_mask.len()),
                Some(l) => {
                    let true_count = row_mask.true_count();
                    let mask_limit = usize::try_from(*l)
                        .map(|l| l.min(true_count))
                        .unwrap_or(true_count);
                    let row_mask = row_mask.limit(mask_limit);
                    *l -= mask_limit as u64;
                    row_mask
                }
                None => row_mask,
            };

            MaskFuture::ready(row_mask)
        }
        Some(filter) => {
            // NOTE: it's very important that the pruning and filter evaluations are built OUTSIDE
            // the future. Registering these row ranges eagerly is a hint to the IO system that
            // we want to start prefetching the IO for this split.
            let reader = Arc::clone(&ctx.reader);
            let filter = Arc::clone(filter);
            MaskFuture::new(row_mask.len(), async move {
                let mut mask = row_mask;
                let mut dynamic_versions = vec![None; filter.conjuncts().len()];

                // TODO(ngates): we could use FuturedUnordered to intersect the masks in parallel.
                for (idx, conjunct) in filter.conjuncts().iter().enumerate() {
                    if mask.all_false() {
                        return Ok(mask);
                    }

                    // Store the latest version of the dynamic expression prior to pruning.
                    // We will re-run the pruning later if the version has changed in the meantime.
                    dynamic_versions[idx] = filter.dynamic_updates(idx).map(|du| du.version());

                    let conjunct_mask = reader
                        .pruning_evaluation(&row_range, conjunct, mask.clone())?
                        .await?;
                    mask = mask.bitand(&conjunct_mask);
                }

                // Now we loop through the conjuncts in the preferred order and evaluate them.
                let mut remaining = BitVec::from_elem(filter.conjuncts().len(), true);
                while let Some(idx) = filter.next_conjunct(&remaining) {
                    remaining.set(idx, false);
                    if mask.all_false() {
                        return Ok(mask);
                    }

                    let conjunct = &filter.conjuncts()[idx];

                    // If the dynamic expression has changed since pruning, re-run the pruning.
                    // Store the dynamic update once to avoid TOCTOU race condition
                    let current_version = filter.dynamic_updates(idx).map(|du| du.version());
                    if let Some(dv) = current_version
                        && dynamic_versions[idx].is_none_or(|v| v < dv)
                    {
                        // The dynamic expression has been updated, re-run the pruning.
                        dynamic_versions[idx] = Some(dv);
                        let conjunct_mask = reader
                            .pruning_evaluation(&row_range, conjunct, mask.clone())?
                            .await?;
                        mask = mask.bitand(&conjunct_mask);
                    }
                    if mask.all_false() {
                        return Ok(mask);
                    }

                    let input_true_count = mask.true_count();
                    let conjunct_mask = reader
                        .filter_evaluation(&row_range, conjunct, MaskFuture::ready(mask))?
                        .await?;
                    filter.report_selectivity(
                        idx,
                        conditional_selectivity(input_true_count, conjunct_mask.true_count()),
                    );

                    // Filter evaluations return a mask already intersected with the input mask.
                    mask = conjunct_mask;
                }

                Ok(mask)
            })
        }
    })
}

fn project_exec_with_mask<A: 'static + Send>(
    ctx: Arc<TaskContext<A>>,
    row_range: std::ops::Range<u64>,
    filter_mask: MaskFuture,
) -> VortexResult<TaskFuture<Option<A>>> {
    // Step 4: execute the projection, only at the mask for rows which match the filter
    let projection_future =
        ctx.reader
            .projection_evaluation(&row_range, &ctx.projection, filter_mask.clone())?;

    let mapper = Arc::clone(&ctx.mapper);
    let array_fut = async move {
        let mask = filter_mask.await?;
        if mask.all_false() {
            return Ok(None);
        }

        let array = projection_future.await?;
        mapper(array).map(Some)
    };

    Ok(array_fut.boxed())
}

fn conditional_selectivity(input_true_count: usize, output_true_count: usize) -> f64 {
    debug_assert!(input_true_count > 0);
    debug_assert!(output_true_count <= input_true_count);
    output_true_count as f64 / input_true_count as f64
}

/// Information needed to execute a single split task.
///
/// Row selection is evaluated before creating a split task so it's not included
pub struct TaskContext<A> {
    /// The shared filter expression.
    pub filter: Option<Arc<FilterExpr>>,
    /// The layout reader.
    pub reader: Arc<dyn LayoutReader>,
    /// The projection expression to apply to gather the scanned rows.
    pub projection: BoundExpression,
    /// Function that maps into an A.
    pub mapper: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
}

#[cfg(test)]
mod tests {
    use super::conditional_selectivity;

    #[test]
    fn selectivity_is_relative_to_the_input_mask() {
        assert_eq!(conditional_selectivity(20, 5), 0.25);
    }
}
