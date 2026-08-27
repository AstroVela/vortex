// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;
use std::ops::Range;

use vortex_array::VortexSessionExecute;
use vortex_array::expr::BoundExpression;
use vortex_error::VortexResult;
use vortex_mask::Mask;

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

/// One conjunct: the subtree producing its input, and the predicate applied to that input.
pub struct ConjunctSlot {
    /// The node producing the fields this predicate reads.
    pub input: NodeId,
    /// The predicate, bound to the input subtree's output dtype.
    pub predicate: BoundExpression,
}

/// How the conjuncts of one filter relate to each other.
///
/// This is the whole of the cascade-versus-parallel policy: the operators are identical, only
/// the demand each conjunct sees differs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConjunctMode {
    /// Each conjunct sees the mask the previous one produced, and an all-false mask ends the
    /// morsel early. Fewer rows read; a serial dependency between conjuncts.
    Cascade,
    /// Every conjunct sees the incoming mask, and the results are intersected. More rows read;
    /// no dependency between conjuncts.
    Parallel,
}

/// The demand spine: predicate evaluations feeding one intersection.
pub struct ConjunctExec {
    slots: Vec<ConjunctSlot>,
    mode: ConjunctMode,

    // Per-morsel state.
    range: Range<u64>,
    plan_cursor: usize,
    plan_started: bool,
    done: bool,
    children: Vec<NodeId>,
}

impl ConjunctExec {
    /// Build a conjunct node.
    pub fn new(slots: Vec<ConjunctSlot>, mode: ConjunctMode) -> Self {
        let children = slots.iter().map(|slot| slot.input).collect();
        Self {
            slots,
            mode,
            range: 0..0,
            plan_cursor: 0,
            plan_started: false,
            done: false,
            children,
        }
    }

    /// Evaluate one conjunct under `incoming`, returning the refined mask.
    fn eval(
        &self,
        idx: usize,
        incoming: &Mask,
        cx: &mut ExecCx<'_>,
    ) -> VortexResult<Mask> {
        let slot = &self.slots[idx];

        // The regime switch: over a sparse mask, filter first and correct by rank; over a dense
        // one, evaluate the whole range and intersect. Same choice the V1 flat reader makes.
        let sparse = incoming.density() < EXPR_EVAL_THRESHOLD;
        let child_demand = if sparse {
            incoming.clone()
        } else {
            Mask::new_true(incoming.len())
        };

        let array = cx.child_array(slot.input, child_demand)?;
        let array = array.apply_bound(&slot.predicate)?;
        let mut ctx = cx.session().create_execution_ctx();
        let predicate_mask = array.null_as_false().execute(&mut ctx)?;

        Ok(if sparse {
            incoming.intersect_by_rank(&predicate_mask)
        } else {
            incoming.bitand(&predicate_mask)
        })
    }
}

impl ExecNode for ConjunctExec {
    fn reset(&mut self, range: Range<u64>) {
        self.range = range;
        self.plan_cursor = 0;
        self.plan_started = false;
        self.done = false;
    }

    fn next_plan(&mut self, cx: &mut PlanCx<'_>) -> VortexResult<PlanPoll> {
        // Emit-once planning: every conjunct's IO is named up front, whatever the mode. Under
        // cascade a later conjunct may turn out not to be needed, but a use is named before its
        // demand is known — refining it after emission is P2's cancellation path, not a reason
        // to defer naming it here.
        while self.plan_cursor < self.slots.len() {
            if cx.out_of_budget() {
                return Ok(PlanPoll::Item(PlanItem::Plan));
            }
            let fresh = !self.plan_started;
            self.plan_started = true;
            if cx.plan_child(self.slots[self.plan_cursor].input, self.range.clone(), fresh)? {
                self.plan_cursor += 1;
                self.plan_started = false;
            } else {
                return Ok(PlanPoll::Item(PlanItem::Plan));
            }
        }
        Ok(PlanPoll::Complete)
    }

    fn execute(&mut self, cx: &mut ExecCx<'_>) -> VortexResult<ExecPoll> {
        if self.done {
            return Ok(ExecPoll::Done);
        }
        self.done = true;

        let incoming = cx.demand().clone();
        let mask = match self.mode {
            ConjunctMode::Cascade => {
                let mut mask = incoming;
                for idx in 0..self.slots.len() {
                    if mask.all_false() {
                        cx.stats().conjuncts_short_circuited += (self.slots.len() - idx) as u64;
                        break;
                    }
                    mask = self.eval(idx, &mask, cx)?;
                }
                mask
            }
            ConjunctMode::Parallel => {
                let mut mask = incoming.clone();
                for idx in 0..self.slots.len() {
                    let refined = self.eval(idx, &incoming, cx)?;
                    mask = mask.bitand(&refined);
                }
                mask
            }
        };

        Ok(ExecPoll::Value(ValueBatch {
            coverage: self.range.clone(),
            value: Value::Mask(mask),
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
