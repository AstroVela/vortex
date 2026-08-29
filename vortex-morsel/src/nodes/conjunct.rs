// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;
use std::ops::Range;
use std::time::Duration;
use std::time::Instant;

use vortex_error::VortexExpect;
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
use crate::nodes::ordering::AdaptiveOrdering;
use crate::nodes::ordering::OrderingGoal;

/// One independently schedulable group of same-column conjuncts.
pub struct ConjunctGroup {
    /// The grouped predicate node.
    pub predicate: NodeId,
    /// Number of logical conjuncts evaluated by that node.
    pub conjunct_count: usize,
}

/// How the conjuncts of one filter relate to each other.
///
/// This is the whole of the cascade-versus-parallel policy: the operators are identical, only
/// the demand each conjunct sees differs.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
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
    groups: Vec<ConjunctGroup>,
    mode: ConjunctMode,
    ordering: AdaptiveOrdering,
    adaptive: bool,

    // Per-morsel state.
    range: Range<u64>,
    group_order: Vec<usize>,
    plan_cursor: usize,
    plan_started: bool,
    exec_cursor: usize,
    active_elapsed: Duration,
    order_recorded: bool,
    incoming: Option<Mask>,
    mask: Option<Mask>,
    done: bool,
    children: Vec<NodeId>,
}

impl ConjunctExec {
    /// Build a conjunct node.
    pub fn new(groups: Vec<ConjunctGroup>, mode: ConjunctMode) -> Self {
        let children = groups.iter().map(|group| group.predicate).collect();
        let ordering = AdaptiveOrdering::new(groups.len(), OrderingGoal::CostPerRejectedRow);
        let adaptive = mode == ConjunctMode::Cascade && groups.len() > 1;
        Self {
            groups,
            mode,
            ordering,
            adaptive,
            range: 0..0,
            group_order: Vec::new(),
            plan_cursor: 0,
            plan_started: false,
            exec_cursor: 0,
            active_elapsed: Duration::ZERO,
            order_recorded: false,
            incoming: None,
            mask: None,
            done: false,
            children,
        }
    }

    fn remaining_conjuncts(&self) -> usize {
        self.group_order[self.exec_cursor..]
            .iter()
            .map(|&index| self.groups[index].conjunct_count)
            .sum()
    }
}

impl ExecNode for ConjunctExec {
    fn reset(&mut self, range: Range<u64>) {
        self.range = range;
        if self.adaptive {
            self.ordering.update_order(&mut self.group_order);
        } else {
            self.group_order.clear();
            self.group_order.extend(0..self.groups.len());
        }
        self.plan_cursor = 0;
        self.plan_started = false;
        self.exec_cursor = 0;
        self.active_elapsed = Duration::ZERO;
        self.order_recorded = false;
        self.incoming = None;
        self.mask = None;
        self.done = false;
    }

    fn next_plan(&mut self, cx: &mut PlanCx<'_>) -> VortexResult<PlanPoll> {
        // Emit-once planning: every conjunct's IO is named up front, whatever the mode. Under
        // cascade a later conjunct may turn out not to be needed, but a use is named before its
        // demand is known — refining it after emission is P2's cancellation path, not a reason
        // to defer naming it here.
        while self.plan_cursor < self.groups.len() {
            if cx.out_of_budget() {
                return Ok(PlanPoll::Item(PlanItem::Plan));
            }
            let fresh = !self.plan_started;
            self.plan_started = true;
            let group_index = self.group_order[self.plan_cursor];
            if cx.plan_child(
                self.groups[group_index].predicate,
                self.range.clone(),
                fresh,
            )? {
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
        if self.incoming.is_none() {
            let incoming = cx.demand().clone();
            self.mask = Some(incoming.clone());
            self.incoming = Some(incoming);
            if !self.order_recorded && !is_identity(&self.group_order) {
                cx.stats().inter_group_reorders += 1;
            }
            self.order_recorded = true;
        }

        while self.exec_cursor < self.groups.len() {
            let eval_demand = match self.mode {
                ConjunctMode::Cascade => self.mask.as_ref(),
                ConjunctMode::Parallel => self.incoming.as_ref(),
            }
            .vortex_expect("execution masks initialized")
            .clone();
            if self.mode == ConjunctMode::Cascade && eval_demand.all_false() {
                cx.stats().conjuncts_short_circuited += self.remaining_conjuncts() as u64;
                self.exec_cursor = self.groups.len();
                break;
            }

            let group_index = self.group_order[self.exec_cursor];
            let predicate = self.groups[group_index].predicate;
            let observe = self.adaptive && self.ordering.needs_observation(group_index);
            let input_rows = observe.then(|| eval_demand.true_count());
            let started = observe.then(Instant::now);
            let poll = cx.child_mask(predicate, eval_demand);
            if let Some(started) = started {
                self.active_elapsed += started.elapsed();
            }
            match poll? {
                ChildPoll::Value(refined) => {
                    if let Some(input_rows) = input_rows {
                        self.ordering.observe(
                            group_index,
                            input_rows,
                            refined.true_count(),
                            self.active_elapsed,
                        );
                    }
                    self.active_elapsed = Duration::ZERO;
                    if self.mode == ConjunctMode::Parallel {
                        self.mask = Some(
                            self.mask
                                .take()
                                .vortex_expect("execution mask initialized")
                                .bitand(&refined),
                        );
                    } else {
                        self.mask = Some(refined);
                    }
                    self.exec_cursor += 1;
                }
                ChildPoll::Blocked(waits) => return Ok(ExecPoll::Blocked(waits)),
                ChildPoll::Done => {
                    return Err(vortex_err!(
                        "conjunct {} produced no value",
                        self.exec_cursor
                    ));
                }
            }
        }

        let mask = self.mask.take().vortex_expect("execution mask initialized");
        self.incoming = None;
        self.done = true;

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

fn is_identity(order: &[usize]) -> bool {
    order.iter().copied().eq(0..order.len())
}
