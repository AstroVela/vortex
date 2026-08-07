// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::ops::Range;
use std::sync::Arc;
use std::sync::OnceLock;

use futures::FutureExt;
use futures::TryFutureExt;
use futures::future::BoxFuture;
use futures::future::Shared;
use parking_lot::RwLock;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::dtype::DType;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::traversal::NodeExt;
use vortex_array::expr::traversal::Transformed;
use vortex_array::expr::traversal::TraversalOrder;
use vortex_array::scalar_fn::fns::dynamic::DynamicExprUpdates;
use vortex_array::scalar_fn::fns::stat::StatFn;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_error::SharedVortexResult;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;

use crate::LayoutRef;
use crate::layouts::zoned::LegacyStats;
use crate::layouts::zoned::Zoned;
use crate::layouts::zoned::zone_map::ZoneMap;
use crate::plan::ExpressionPlan;
use crate::plan::Plan;
use crate::plan::PlanArrayFuture;
use crate::plan::PlanExecutionContext;
use crate::plan::PlanRef;
use crate::plan::new_plan;
use crate::plan::optimize_child;
use crate::plan::optimizer::PlanParentReduceRule;

const DATA_CHILD_INDEX: usize = 0;
const ZONES_CHILD_INDEX: usize = 1;

type SharedZoneMap = Shared<BoxFuture<'static, SharedVortexResult<ZoneMap>>>;
type SharedPruningResult = Shared<BoxFuture<'static, SharedVortexResult<Arc<ZonedPruningResult>>>>;

#[derive(Clone)]
struct ZonedPruningState {
    expression: BoundExpression,
    zone_map: Arc<OnceLock<SharedZoneMap>>,
    pruning_result: Arc<OnceLock<SharedPruningResult>>,
}

impl ZonedPruningState {
    fn new(expression: BoundExpression) -> Self {
        Self {
            expression,
            zone_map: Arc::new(OnceLock::new()),
            pruning_result: Arc::new(OnceLock::new()),
        }
    }

    fn zone_map(
        &self,
        ctx: &PlanExecutionContext,
        zones: &PlanRef,
        column_dtype: &DType,
        aggregate_fns: &Arc<[AggregateFnRef]>,
        zone_len: u64,
        row_count: u64,
    ) -> VortexResult<SharedZoneMap> {
        let zone_count = zones.row_count();
        let zone_count_usize = usize::try_from(zone_count)?;
        Ok(self
            .zone_map
            .get_or_init(|| {
                let ctx = ctx.clone();
                let zones = Arc::clone(zones);
                let column_dtype = column_dtype.clone();
                let aggregate_fns = Arc::clone(aggregate_fns);
                async move {
                    let zones = zones.execute(
                        &ctx,
                        &(0..zone_count),
                        MaskFuture::new_true(zone_count_usize),
                    )?;
                    let mut execution = ctx.session().create_execution_ctx();
                    let zones = zones.await?.execute::<StructArray>(&mut execution)?;
                    // SAFETY: zoned layout construction validated that the auxiliary child was
                    // written from this column dtype and stats-table schema.
                    Ok(unsafe {
                        ZoneMap::new_unchecked(
                            column_dtype,
                            zones,
                            aggregate_fns,
                            zone_len,
                            row_count,
                        )
                    })
                }
                .map_err(Arc::new)
                .boxed()
                .shared()
            })
            .clone())
    }

    fn pruning_result(
        &self,
        ctx: &PlanExecutionContext,
        zones: &PlanRef,
        column_dtype: &DType,
        aggregate_fns: &Arc<[AggregateFnRef]>,
        zone_len: u64,
        row_count: u64,
    ) -> VortexResult<SharedPruningResult> {
        let zone_map =
            self.zone_map(ctx, zones, column_dtype, aggregate_fns, zone_len, row_count)?;
        let expression = self.expression.clone();
        let session = ctx.session().clone();
        Ok(self
            .pruning_result
            .get_or_init(|| {
                async move {
                    let zone_map = zone_map.await?;
                    let initial_result =
                        zone_map.evaluate(&expression, &session).map_err(Arc::new)?;
                    let dynamic_updates = DynamicExprUpdates::new(&expression.unbind());
                    Ok(Arc::new(ZonedPruningResult {
                        zone_map,
                        expression,
                        dynamic_updates,
                        latest_result: RwLock::new((0, initial_result)),
                        session,
                    }))
                }
                .boxed()
                .shared()
            })
            .clone())
    }
}

struct ZonedPruningResult {
    zone_map: ZoneMap,
    expression: BoundExpression,
    dynamic_updates: Option<DynamicExprUpdates>,
    latest_result: RwLock<(u64, BoolArray)>,
    session: VortexSession,
}

impl ZonedPruningResult {
    fn evaluate(&self) -> VortexResult<BoolArray> {
        let Some(dynamic_updates) = &self.dynamic_updates else {
            return Ok(self.latest_result.read().1.clone());
        };
        let version = dynamic_updates.version();
        {
            let result = self.latest_result.read();
            if result.0 >= version {
                return Ok(result.1.clone());
            }
        }

        let mut result = self.latest_result.write();
        if result.0 < version {
            result.1 = self.zone_map.evaluate(&self.expression, &self.session)?;
            result.0 = version;
        }
        Ok(result.1.clone())
    }
}

/// A physical zoned plan over either its transparent data child or its auxiliary zone map.
///
/// An expression containing abstract statistic functions can rewrite this node into pruning
/// state. The rewritten plan keeps the original row domain, drops the data child, and expands
/// zone-level proof values into row-aligned booleans during execution.
pub struct ZonedPlan {
    layout: LayoutRef,
    dtype: DType,
    data: Option<PlanRef>,
    zones: PlanRef,
    zone_len: u64,
    aggregate_fns: Arc<[AggregateFnRef]>,
    pruning: Option<ZonedPruningState>,
}

impl ZonedPlan {
    pub(crate) fn try_new(layout: &LayoutRef) -> VortexResult<Self> {
        let data = new_plan(
            &layout
                .slot(DATA_CHILD_INDEX)?
                .ok_or_else(|| vortex_error::vortex_err!("Zoned data child is absent"))?,
        )?;
        let zones = new_plan(
            &layout
                .slot(ZONES_CHILD_INDEX)?
                .ok_or_else(|| vortex_error::vortex_err!("Zoned zones child is absent"))?,
        )?;
        let metadata = if let Some(layout) = layout.as_opt::<Zoned>() {
            layout.data()
        } else if let Some(layout) = layout.as_opt::<LegacyStats>() {
            layout.data()
        } else {
            vortex_bail!("ZonedPlan requires a zoned layout")
        };

        Ok(Self {
            layout: Arc::clone(layout),
            dtype: layout.dtype().clone(),
            data: Some(data),
            zones,
            zone_len: u64::try_from(metadata.zone_len())?,
            aggregate_fns: metadata.aggregate_fns(),
            pruning: None,
        })
    }

    fn with_children(&self, data: Option<PlanRef>, zones: PlanRef) -> Self {
        Self {
            layout: Arc::clone(&self.layout),
            dtype: self.dtype.clone(),
            data,
            zones,
            zone_len: self.zone_len,
            aggregate_fns: Arc::clone(&self.aggregate_fns),
            pruning: self.pruning.clone(),
        }
    }

    fn with_pruning(&self, expression: BoundExpression) -> Option<Self> {
        if self.zone_len == 0 || self.pruning.is_some() {
            return None;
        }
        let dtype = expression.dtype().clone();
        Some(Self {
            layout: Arc::clone(&self.layout),
            dtype,
            data: None,
            zones: Arc::clone(&self.zones),
            zone_len: self.zone_len,
            aggregate_fns: Arc::clone(&self.aggregate_fns),
            pruning: Some(ZonedPruningState::new(expression)),
        })
    }

    /// Returns whether this plan evaluates a zone-backed pruning proof.
    pub fn is_pruning(&self) -> bool {
        self.pruning.is_some()
    }

    /// Returns the abstract pruning proof carried by this plan, when present.
    pub fn pruning_expression(&self) -> Option<&BoundExpression> {
        self.pruning.as_ref().map(|state| &state.expression)
    }

    fn execute_pruning(
        &self,
        ctx: &PlanExecutionContext,
        row_range: &Range<u64>,
        mask: MaskFuture,
    ) -> VortexResult<PlanArrayFuture> {
        vortex_ensure!(
            row_range.start <= row_range.end && row_range.end <= self.row_count(),
            "Zoned pruning row range {:?} is outside 0..{}",
            row_range,
            self.row_count()
        );
        let range_len = usize::try_from(row_range.end - row_range.start)?;
        vortex_ensure!(
            mask.len() == range_len,
            "Zoned pruning mask length mismatch"
        );

        let state = self
            .pruning
            .clone()
            .ok_or_else(|| vortex_error::vortex_err!("Zoned pruning state is absent"))?;
        let ctx = ctx.clone();
        let zones = Arc::clone(&self.zones);
        let column_dtype = self.layout.dtype().clone();
        let output_dtype = self.dtype.clone();
        let aggregate_fns = Arc::clone(&self.aggregate_fns);
        let zone_len = self.zone_len;
        let row_count = self.row_count();
        let row_range = row_range.clone();

        Ok(async move {
            let input_mask = mask.await?;
            if input_mask.all_false() {
                return Ok(BoolArray::new(
                    BitBuffer::new_unset(0),
                    Validity::from(output_dtype.nullability()),
                )
                .into_array());
            }

            let pruning_result = state.pruning_result(
                &ctx,
                &zones,
                &column_dtype,
                &aggregate_fns,
                zone_len,
                row_count,
            )?;
            let evaluated = pruning_result.await?.evaluate()?;
            let mut execution = ctx.session().create_execution_ctx();
            let zone_validity =
                BoolArrayExt::validity(&evaluated).execute_mask(evaluated.len(), &mut execution)?;
            let zone_values = evaluated.to_bit_buffer();

            let zone_start = row_range.start / zone_len;
            let zone_end = row_range.end.div_ceil(zone_len);
            let zone_start_usize = usize::try_from(zone_start)?;
            let zone_end_usize = usize::try_from(zone_end)?;
            vortex_ensure!(
                zone_end_usize <= evaluated.len(),
                "Zoned pruning requires zones {zone_start}..{zone_end}, but only {} exist",
                evaluated.len()
            );

            let mut values = BitBufferMut::with_capacity(range_len);
            let mut validity = BitBufferMut::with_capacity(range_len);
            let relevant_values = zone_values.slice(zone_start_usize..zone_end_usize);
            let relevant_validity = zone_validity.slice(zone_start_usize..zone_end_usize);
            for (offset, (value, valid)) in relevant_values
                .iter()
                .zip(relevant_validity.iter())
                .enumerate()
            {
                let zone_index = zone_start + u64::try_from(offset)?;
                let zone_row_start = zone_index.saturating_mul(zone_len).min(row_count);
                let zone_row_end = zone_index
                    .saturating_add(1)
                    .saturating_mul(zone_len)
                    .min(row_count);
                let start = zone_row_start.max(row_range.start);
                let end = zone_row_end.min(row_range.end);
                if start < end {
                    let len = usize::try_from(end - start)?;
                    values.append_n(value, len);
                    validity.append_n(valid, len);
                }
            }
            vortex_ensure!(
                values.len() == range_len && validity.len() == range_len,
                "Expanded zone proof length does not match row range"
            );

            let validity = if output_dtype.is_nullable() {
                Validity::from(validity.freeze())
            } else {
                vortex_ensure!(
                    validity.freeze().true_count() == range_len,
                    "Non-nullable zoned proof produced null values"
                );
                Validity::NonNullable
            };
            let output = BoolArray::new(values.freeze(), validity).into_array();
            if input_mask.all_true() {
                Ok(output)
            } else {
                output.filter(input_mask)
            }
        }
        .boxed())
    }
}

impl Plan for ZonedPlan {
    fn name(&self) -> &'static str {
        "ZonedPlan"
    }

    fn optimize(&self) -> VortexResult<PlanRef> {
        let data = self
            .data
            .as_ref()
            .map(|data| optimize_child(data))
            .transpose()?;
        let zones = optimize_child(&self.zones)?;
        Ok(Arc::new(self.with_children(data, zones)))
    }

    // Zoned children come from `new_plan` and no rule injects expressions below a zoned plan,
    // so the subtree is always layout-pure.
    fn needs_optimize(&self) -> bool {
        false
    }

    fn execute(
        &self,
        ctx: &PlanExecutionContext,
        row_range: &Range<u64>,
        mask: MaskFuture,
    ) -> VortexResult<PlanArrayFuture> {
        if self.pruning.is_some() {
            return self.execute_pruning(ctx, row_range, mask);
        }
        self.data
            .as_ref()
            .ok_or_else(|| vortex_error::vortex_err!("Zoned data child is absent"))?
            .execute(ctx, row_range, mask)
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count(&self) -> u64 {
        self.layout.row_count()
    }

    fn child_count(&self) -> usize {
        2
    }

    fn child(&self, index: usize) -> VortexResult<Option<PlanRef>> {
        match index {
            DATA_CHILD_INDEX => Ok(self.data.clone()),
            ZONES_CHILD_INDEX => Ok(Some(Arc::clone(&self.zones))),
            _ => vortex_bail!("Zoned plan has no child {index}"),
        }
    }

    fn child_name(&self, index: usize) -> Cow<'_, str> {
        match index {
            DATA_CHILD_INDEX => Cow::Borrowed("data"),
            ZONES_CHILD_INDEX => Cow::Borrowed("zones"),
            _ => Cow::Owned(format!("child[{index}]")),
        }
    }
}

/// Rewrites an abstract statistic expression over a zoned plan into its pruning state.
#[derive(Debug)]
pub(crate) struct ExpressionZonedRule;

impl PlanParentReduceRule<ZonedPlan> for ExpressionZonedRule {
    type Parent = ExpressionPlan;

    fn reduce_parent(
        &self,
        child: &ZonedPlan,
        parent: &ExpressionPlan,
        _child_idx: usize,
    ) -> VortexResult<Option<PlanRef>> {
        let mut contains_stat = false;
        let mut contains_root = false;
        parent.expression().clone().transform_down(|expression| {
            if expression
                .as_scalar()
                .is_some_and(|scalar_fn| scalar_fn.is::<StatFn>())
            {
                contains_stat = true;
                return Ok(Transformed {
                    value: expression,
                    order: TraversalOrder::Skip,
                    changed: false,
                });
            }
            contains_root |= expression.is_root();
            Ok(Transformed::no(expression))
        })?;
        if !parent.dtype().is_boolean() || !contains_stat || contains_root {
            return Ok(None);
        }

        Ok(child
            .with_pruning(parent.expression().clone())
            .map(|plan| Arc::new(plan) as PlanRef))
    }
}
