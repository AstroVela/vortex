// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::dtype::DType;
use vortex_array::expr::BoundExpression;
use vortex_error::VortexResult;

use crate::node::ExecCx;
use crate::node::ExecNode;
use crate::node::ExecPoll;
use crate::node::NodeId;
use crate::node::PlanCx;
use crate::node::PlanItem;
use crate::node::PlanPoll;
use crate::node::RetireCx;
use crate::node::Value;
use crate::node::ValueBatch;

/// The root of a morsel: refine the demand with the filter, then project under it.
pub struct FilterExec {
    predicate: Option<NodeId>,
    projection: NodeId,
    projection_expr: BoundExpression,
    output_dtype: DType,

    // Per-morsel state.
    range: Range<u64>,
    plan_stage: u8,
    plan_started: bool,
    done: bool,
    children: Vec<NodeId>,
}

impl FilterExec {
    /// Build a filter node.
    pub fn new(
        predicate: Option<NodeId>,
        projection: NodeId,
        projection_expr: BoundExpression,
        output_dtype: DType,
    ) -> Self {
        let children = predicate.into_iter().chain([projection]).collect();
        Self {
            predicate,
            projection,
            projection_expr,
            output_dtype,
            range: 0..0,
            plan_stage: 0,
            plan_started: false,
            done: false,
            children,
        }
    }
}

impl ExecNode for FilterExec {
    fn reset(&mut self, range: Range<u64>) {
        self.range = range;
        self.plan_stage = 0;
        self.plan_started = false;
        self.done = false;
    }

    fn next_plan(&mut self, cx: &mut PlanCx<'_>) -> VortexResult<PlanPoll> {
        loop {
            let child = match (self.plan_stage, self.predicate) {
                (0, Some(predicate)) => predicate,
                (0, None) | (1, _) => self.projection,
                _ => return Ok(PlanPoll::Complete),
            };
            if cx.out_of_budget() {
                return Ok(PlanPoll::Item(PlanItem::Plan));
            }
            let fresh = !self.plan_started;
            self.plan_started = true;
            if cx.plan_child(child, self.range.clone(), fresh)? {
                self.plan_stage += if self.plan_stage == 0 && self.predicate.is_none() {
                    2
                } else {
                    1
                };
                self.plan_started = false;
            } else {
                return Ok(PlanPoll::Item(PlanItem::Plan));
            }
        }
    }

    fn execute(&mut self, cx: &mut ExecCx<'_>) -> VortexResult<ExecPoll> {
        if self.done {
            return Ok(ExecPoll::Done);
        }
        self.done = true;

        let demand = cx.demand().clone();
        let mask = match self.predicate {
            Some(predicate) => cx.child_mask(predicate, demand)?,
            None => demand,
        };

        if mask.all_false() {
            cx.stats().morsels_empty += 1;
            return Ok(ExecPoll::Value(ValueBatch {
                coverage: self.range.clone(),
                value: Value::Array(Canonical::empty(&self.output_dtype).into_array()),
            }));
        }

        // The projection subtree reads only the surviving rows: the mask is the demand, so a
        // sealed-empty chunk under it costs no read at all.
        let array = cx.child_array(self.projection, mask)?;
        let array = array.apply_bound(&self.projection_expr)?;

        Ok(ExecPoll::Value(ValueBatch {
            coverage: self.range.clone(),
            value: Value::Array(array),
        }))
    }

    fn retire(&mut self, cx: &mut RetireCx<'_>) {
        for &child in &self.children {
            cx.retire_child(child);
        }
    }

    fn children(&self) -> &[NodeId] {
        &self.children
    }
}
