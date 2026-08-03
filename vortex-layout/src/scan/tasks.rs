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
use vortex_array::expr::Expression;
use vortex_array::expr::root;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;
use vortex_scan::row_mask::RowMask;

use crate::LayoutReaderRef;
use crate::scan::filter::FilterExpr;

pub type TaskFuture<A> = BoxFuture<'static, VortexResult<A>>;

/// Information needed to execute one split from materialized physical plans.
pub(crate) struct TaskContext<A> {
    filter: Option<Arc<FilterExpr>>,
    predicates: Vec<LayoutReaderRef>,
    projection: LayoutReaderRef,
    mapper: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
}

impl<A> TaskContext<A> {
    pub(crate) fn new(
        filter: Option<Expression>,
        predicates: Vec<LayoutReaderRef>,
        projection: LayoutReaderRef,
        mapper: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
    ) -> Self {
        Self {
            filter: filter.map(|filter| Arc::new(FilterExpr::new(filter))),
            predicates,
            projection,
            mapper,
        }
    }
}

/// Executes one split using expression-bound layout readers materialized from physical plans.
pub(crate) fn split_exec<A: 'static + Send>(
    ctx: Arc<TaskContext<A>>,
    read_mask: RowMask,
    limit: Option<&mut u64>,
) -> VortexResult<TaskFuture<Option<A>>> {
    let row_range = read_mask.row_range();
    let row_mask = read_mask.mask().clone();

    let filter_mask = match ctx.filter.as_ref() {
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
            if filter.conjuncts().len() != ctx.predicates.len() {
                vortex_bail!(
                    "physical predicate count {} does not match conjunct count {}",
                    ctx.predicates.len(),
                    filter.conjuncts().len()
                );
            }

            let ctx = Arc::clone(&ctx);
            let filter = Arc::clone(filter);
            let row_range = row_range.clone();

            MaskFuture::new(row_mask.len(), async move {
                let mut mask = row_mask;
                let mut dynamic_versions = vec![None; filter.conjuncts().len()];

                for (idx, predicate) in ctx.predicates.iter().enumerate() {
                    if mask.all_false() {
                        return Ok(mask);
                    }

                    dynamic_versions[idx] = filter.dynamic_updates(idx).map(|du| du.version());
                    let conjunct_mask = predicate
                        .pruning_evaluation(&row_range, &root(), mask.clone())?
                        .await?;
                    mask = mask.bitand(&conjunct_mask);
                }

                let mut remaining = BitVec::from_elem(filter.conjuncts().len(), true);
                while let Some(idx) = filter.next_conjunct(&remaining) {
                    remaining.set(idx, false);
                    if mask.all_false() {
                        return Ok(mask);
                    }

                    let current_version = filter.dynamic_updates(idx).map(|du| du.version());
                    if let Some(version) = current_version
                        && dynamic_versions[idx].is_none_or(|old| old < version)
                    {
                        dynamic_versions[idx] = Some(version);
                        let conjunct_mask = ctx.predicates[idx]
                            .pruning_evaluation(&row_range, &root(), mask.clone())?
                            .await?;
                        mask = mask.bitand(&conjunct_mask);
                    }
                    if mask.all_false() {
                        return Ok(mask);
                    }

                    let conjunct_mask = ctx.predicates[idx]
                        .filter_evaluation(&row_range, &root(), MaskFuture::ready(mask))?
                        .await?;
                    filter.report_selectivity(idx, conjunct_mask.density());
                    mask = conjunct_mask;
                }

                Ok(mask)
            })
        }
    };

    let projection_future =
        ctx.projection
            .projection_evaluation(&row_range, &root(), filter_mask.clone())?;

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
