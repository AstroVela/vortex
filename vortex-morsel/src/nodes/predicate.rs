// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;
use std::ops::Range;
use std::time::Duration;

use vortex_array::expr::BoundExpression;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::node::ChildPoll;
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
use crate::nodes::EXPR_EVAL_THRESHOLD;
use crate::nodes::ordering::AdaptiveOrdering;
use crate::nodes::ordering::OrderingGoal;

/// All conjuncts backed by one input column subtree.
///
/// The input is decoded once under the group's incoming demand. Every expression then evaluates
/// against that same array before their masks are intersected. Sparse input may use an equivalent
/// normalized predicate list when it eliminates redundant expression passes.
pub struct PredicateExec {
    input: NodeId,
    predicates: Vec<BoundExpression>,
    sparse_predicates: Option<Vec<BoundExpression>>,
    ordering: AdaptiveOrdering,
    sparse_ordering: Option<AdaptiveOrdering>,
    children: [NodeId; 1],

    // Per-morsel state.
    range: Range<u64>,
    plan_started: bool,
    predicate_order: Vec<usize>,
    sparse_predicate_order: Option<Vec<usize>>,
    order_recorded: bool,
    done: bool,
}

impl PredicateExec {
    /// Create one grouped predicate node over `input`.
    pub fn new(
        input: NodeId,
        predicates: Vec<BoundExpression>,
        sparse_predicates: Option<Vec<BoundExpression>>,
    ) -> Self {
        debug_assert!(!predicates.is_empty());
        let ordering = AdaptiveOrdering::new(predicates.len(), OrderingGoal::SparseEvaluation);
        let sparse_ordering = sparse_predicates.as_ref().map(|predicates| {
            AdaptiveOrdering::new(predicates.len(), OrderingGoal::SparseEvaluation)
        });
        let sparse_predicate_order = sparse_predicates.as_ref().map(|_| Vec::new());
        Self {
            input,
            predicates,
            sparse_predicates,
            ordering,
            sparse_ordering,
            children: [input],
            range: 0..0,
            plan_started: false,
            predicate_order: Vec::new(),
            sparse_predicate_order,
            order_recorded: false,
            done: false,
        }
    }

    fn evaluate(
        array: vortex_array::ArrayRef,
        predicate: &BoundExpression,
        cx: &mut ExecCx<'_>,
    ) -> VortexResult<Mask> {
        let predicate = array.apply_bound(predicate)?;
        predicate.null_as_false().execute(cx.execution())
    }
}

impl ExecNode for PredicateExec {
    fn reset(&mut self, range: Range<u64>) {
        self.range = range;
        self.plan_started = false;
        self.ordering.update_order(&mut self.predicate_order);
        if let Some((ordering, predicate_order)) = self
            .sparse_ordering
            .as_ref()
            .zip(self.sparse_predicate_order.as_mut())
        {
            ordering.update_order(predicate_order);
        }
        self.order_recorded = false;
        self.done = false;
    }

    fn next_plan(&mut self, cx: &mut PlanCx<'_>) -> VortexResult<PlanPoll> {
        if cx.out_of_budget() {
            return Ok(PlanPoll::Item(PlanItem::Plan));
        }
        let fresh = !self.plan_started;
        self.plan_started = true;
        if cx.plan_child(self.input, self.range.clone(), fresh)? {
            Ok(PlanPoll::Complete)
        } else {
            Ok(PlanPoll::Item(PlanItem::Plan))
        }
    }

    fn execute(&mut self, cx: &mut ExecCx<'_>) -> VortexResult<ExecPoll> {
        if self.done {
            return Ok(ExecPoll::Done);
        }

        let incoming = cx.demand().clone();
        let sparse = incoming.density() < EXPR_EVAL_THRESHOLD;
        let child_demand = if sparse {
            incoming.clone()
        } else {
            Mask::new_true(incoming.len())
        };
        let array = match cx.child_array(self.input, child_demand)? {
            ChildPoll::Value(array) => array,
            ChildPoll::Blocked(waits) => return Ok(ExecPoll::Blocked(waits)),
            ChildPoll::Done => {
                return Err(vortex_err!(
                    "grouped predicate input {} produced no value",
                    self.input
                ));
            }
        };

        let mask = if sparse {
            // `array` is already filtered to the incoming mask. Keep a rank-space mask over that
            // one decoded input and progressively filter it before each later conjunct.
            let predicates = self
                .sparse_predicates
                .as_deref()
                .unwrap_or(&self.predicates);
            let predicate_order = self
                .sparse_predicate_order
                .as_deref()
                .unwrap_or(&self.predicate_order);
            if !self.order_recorded && !is_identity(predicate_order) {
                cx.stats().intra_group_reorders += 1;
            }
            self.order_recorded = true;
            let mut relative = Mask::new_true(array.len());
            for position in 0..predicate_order.len() {
                let predicate_index = predicate_order[position];
                if relative.all_false() {
                    cx.stats().conjuncts_short_circuited +=
                        (predicate_order.len() - position) as u64;
                    break;
                }
                let observe = self
                    .sparse_ordering
                    .as_ref()
                    .unwrap_or(&self.ordering)
                    .needs_observation(predicate_index);
                let input_rows = observe.then(|| relative.true_count());
                let active = if relative.all_true() {
                    array.clone()
                } else {
                    array.filter(relative.clone())?
                };
                let predicate_mask = Self::evaluate(active, &predicates[predicate_index], cx)?;
                let output_rows = observe.then(|| predicate_mask.true_count());
                relative = relative.intersect_by_rank(&predicate_mask);
                if let (Some(input_rows), Some(output_rows)) = (input_rows, output_rows) {
                    let ordering = self.sparse_ordering.as_mut().unwrap_or(&mut self.ordering);
                    ordering.observe(predicate_index, input_rows, output_rows, Duration::ZERO);
                }
            }
            incoming.intersect_by_rank(&relative)
        } else {
            // `array` covers the full range. Mirror the original cross-node cascade: evaluate
            // densely while the surviving mask is dense, then switch to filtered rank space.
            let mut refined = incoming;
            if !self.order_recorded && !is_identity(&self.predicate_order) {
                cx.stats().intra_group_reorders += 1;
            }
            self.order_recorded = true;
            for position in 0..self.predicate_order.len() {
                let predicate_index = self.predicate_order[position];
                if refined.all_false() {
                    cx.stats().conjuncts_short_circuited +=
                        (self.predicate_order.len() - position) as u64;
                    break;
                }
                let observe = self.ordering.needs_observation(predicate_index);
                let sparse = refined.density() < EXPR_EVAL_THRESHOLD;
                let observed_input_rows = observe.then(|| {
                    if sparse {
                        refined.true_count()
                    } else {
                        array.len()
                    }
                });
                let active = if sparse {
                    array.filter(refined.clone())?
                } else {
                    array.clone()
                };
                let predicate_mask = Self::evaluate(active, &self.predicates[predicate_index], cx)?;
                let observed_output_rows = observe.then(|| predicate_mask.true_count());
                refined = if sparse {
                    refined.intersect_by_rank(&predicate_mask)
                } else {
                    refined.bitand(&predicate_mask)
                };
                if let (Some(observed_input_rows), Some(observed_output_rows)) =
                    (observed_input_rows, observed_output_rows)
                {
                    self.ordering.observe(
                        predicate_index,
                        observed_input_rows,
                        observed_output_rows,
                        Duration::ZERO,
                    );
                }
            }
            refined
        };
        self.done = true;
        Ok(ExecPoll::Value(ValueBatch {
            coverage: self.range.clone(),
            value: Value::Mask(mask),
        }))
    }

    fn retire(&mut self, cx: &mut RetireCx<'_>) {
        cx.retire_child(self.input);
    }

    fn children(&self) -> &[NodeId] {
        &self.children
    }
}

fn is_identity(order: &[usize]) -> bool {
    order.iter().copied().eq(0..order.len())
}
