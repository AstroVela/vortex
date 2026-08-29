// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;
use std::ops::Range;

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

/// All conjuncts backed by one input column subtree.
///
/// The input is decoded once under the group's incoming demand. Every expression then evaluates
/// against that same array before their masks are intersected. Sparse input may use an equivalent
/// normalized predicate list when it eliminates redundant expression passes.
pub struct PredicateExec {
    input: NodeId,
    predicates: Vec<BoundExpression>,
    sparse_predicates: Option<Vec<BoundExpression>>,
    children: [NodeId; 1],

    // Per-morsel state.
    range: Range<u64>,
    plan_started: bool,
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
        Self {
            input,
            predicates,
            sparse_predicates,
            children: [input],
            range: 0..0,
            plan_started: false,
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
            let mut relative = Mask::new_true(array.len());
            for (index, predicate) in predicates.iter().enumerate() {
                if relative.all_false() {
                    cx.stats().conjuncts_short_circuited += (predicates.len() - index) as u64;
                    break;
                }
                let active = if relative.all_true() {
                    array.clone()
                } else {
                    array.filter(relative.clone())?
                };
                let predicate_mask = Self::evaluate(active, predicate, cx)?;
                relative = relative.intersect_by_rank(&predicate_mask);
            }
            incoming.intersect_by_rank(&relative)
        } else {
            // `array` covers the full range. Mirror the original cross-node cascade: evaluate
            // densely while the surviving mask is dense, then switch to filtered rank space.
            let mut refined = incoming;
            for (index, predicate) in self.predicates.iter().enumerate() {
                if refined.all_false() {
                    cx.stats().conjuncts_short_circuited += (self.predicates.len() - index) as u64;
                    break;
                }
                let sparse = refined.density() < EXPR_EVAL_THRESHOLD;
                let active = if sparse {
                    array.filter(refined.clone())?
                } else {
                    array.clone()
                };
                let predicate_mask = Self::evaluate(active, predicate, cx)?;
                refined = if sparse {
                    refined.intersect_by_rank(&predicate_mask)
                } else {
                    refined.bitand(&predicate_mask)
                };
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
