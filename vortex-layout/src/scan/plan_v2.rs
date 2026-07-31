// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// See https://github.com/vortex-data/vortex/issues/9062

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
use vortex_session::VortexSession;

use crate::LayoutReaderRef;
use crate::plan::PlanRef;
use crate::scan::filter::FilterExpr;
use crate::segments::SegmentFuture;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;

pub(crate) struct PlanV2 {
    projection: LayoutReaderRef,
    predicates: Vec<LayoutReaderRef>,
    filter: Option<Expression>,
}

impl PlanV2 {
    pub(crate) fn try_new(
        projection: PlanRef,
        predicates: Vec<PlanRef>,
        filter: Option<Expression>,
        session: &VortexSession,
    ) -> VortexResult<Self> {
        let segment_source: Arc<dyn SegmentSource> = Arc::new(UnavailableSegmentSource);
        let ctx = Default::default();
        let projection =
            projection.new_reader(Arc::from(""), Arc::clone(&segment_source), session, &ctx)?;
        let predicates = predicates
            .iter()
            .map(|predicate| {
                predicate.new_reader(Arc::from(""), Arc::clone(&segment_source), session, &ctx)
            })
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(Self {
            projection,
            predicates,
            filter,
        })
    }

    pub(crate) fn task_context<A>(
        &self,
        mapper: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
    ) -> Arc<TaskContext<A>> {
        Arc::new(TaskContext {
            filter: self
                .filter
                .clone()
                .map(|filter| Arc::new(FilterExpr::new(filter))),
            predicates: self.predicates.clone(),
            projection: Arc::clone(&self.projection),
            mapper,
        })
    }
}

/// Environment variable selecting the scan planning implementation.
pub const SCAN_IMPL_ENV: &str = "VORTEX_SCAN_IMPL";

/// Returns whether V2 heap-allocated planning is enabled for this process.
///
/// The existing `plan` path remains the default on this extraction branch. Set
/// `VORTEX_SCAN_IMPL=planv2` to exercise the V2 path with the same execution implementation.
pub fn plan_v2_enabled() -> VortexResult<bool> {
    match std::env::var(SCAN_IMPL_ENV) {
        Ok(value) => parse_scan_impl(&value),
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(std::env::VarError::NotUnicode(value)) => {
            vortex_bail!("{SCAN_IMPL_ENV} must be valid unicode, got {value:?}")
        }
    }
}

fn parse_scan_impl(value: &str) -> VortexResult<bool> {
    match value {
        "" | "plan" | "v1" | "legacy" | "layout-reader" => Ok(false),
        "planv2" | "plan-v2" | "v2" | "planned" | "scan-plan" => Ok(true),
        other => vortex_bail!(
            "{SCAN_IMPL_ENV} must be one of plan, v1, legacy, layout-reader, planv2, plan-v2, v2, planned, or scan-plan, got {other:?}"
        ),
    }
}

/// Execute one split using a V2 physical scan plan.
///
/// The execution order intentionally mirrors [`crate::scan::plan::split_exec`]. Expressions were
/// consumed during planning, so execution selects a predicate or projection plan without passing
/// an expression.
pub(crate) fn split_exec<A: 'static + Send>(
    ctx: Arc<TaskContext<A>>,
    read_mask: RowMask,
    limit: Option<&mut u64>,
) -> VortexResult<BoxFuture<'static, VortexResult<Option<A>>>> {
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

/// Information needed to execute one split from a V2 physical scan plan.
pub(crate) struct TaskContext<A> {
    filter: Option<Arc<FilterExpr>>,
    predicates: Vec<LayoutReaderRef>,
    projection: LayoutReaderRef,
    mapper: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
}

struct UnavailableSegmentSource;

impl SegmentSource for UnavailableSegmentSource {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        async move {
            vortex_bail!(
                "Compatibility plan unexpectedly requested segment {id:?} while materializing an existing reader"
            )
        }
        .boxed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_impl_accepts_v1_and_v2_values() -> VortexResult<()> {
        for value in ["", "plan", "v1", "legacy", "layout-reader"] {
            assert!(!parse_scan_impl(value)?);
        }
        for value in ["planv2", "plan-v2", "v2", "planned", "scan-plan"] {
            assert!(parse_scan_impl(value)?);
        }
        Ok(())
    }

    #[test]
    fn scan_impl_rejects_unknown_value() {
        assert!(parse_scan_impl("unknown").is_err());
    }
}
