// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Ordering;
use std::time::Duration;

use crate::nodes::EXPR_EVAL_THRESHOLD;

/// The benefit an adaptive order is trying to maximize.
pub(crate) enum OrderingGoal {
    /// Put the cheapest row eliminator first across independently decoded predicate groups.
    CostPerRejectedRow,
    /// Within one decoded column, reorder only when the winner unlocks sparse evaluation.
    SparseEvaluation,
}

/// Worker-local observations retained as its arena advances between morsels.
///
/// No ordering state crosses workers. It is a performance hint and never affects scan semantics.
pub(crate) struct AdaptiveOrdering {
    observations: Vec<Observation>,
    goal: OrderingGoal,
}

#[derive(Clone, Copy, Default)]
struct Observation {
    samples: u64,
    elapsed_ns: u64,
    input_rows: u64,
    rejected_rows: u64,
}

impl AdaptiveOrdering {
    pub(crate) fn new(len: usize, goal: OrderingGoal) -> Self {
        Self {
            observations: vec![Observation::default(); len],
            goal,
        }
    }

    pub(crate) fn update_order(&self, order: &mut Vec<usize>) {
        order.clear();
        order.extend(0..self.observations.len());
        match self.goal {
            OrderingGoal::CostPerRejectedRow => {
                if self
                    .observations
                    .iter()
                    .any(|observation| observation.samples == 0)
                {
                    return;
                }
                order.sort_by(|&left, &right| {
                    compare_cost(
                        left,
                        self.observations[left],
                        right,
                        self.observations[right],
                    )
                });
            }
            OrderingGoal::SparseEvaluation => {
                if self
                    .observations
                    .iter()
                    .any(|observation| observation.samples == 0)
                    || !self.observations.iter().any(unlocks_sparse_evaluation)
                {
                    return;
                }
                order.sort_by(|&left, &right| {
                    compare_selectivity(
                        left,
                        self.observations[left],
                        right,
                        self.observations[right],
                    )
                });
            }
        }
    }

    #[cfg(test)]
    fn order(&self) -> Vec<usize> {
        let mut order = Vec::new();
        self.update_order(&mut order);
        order
    }

    pub(crate) fn needs_observation(&self, index: usize) -> bool {
        self.observations[index].samples == 0
    }

    pub(crate) fn observe(
        &mut self,
        index: usize,
        input_rows: usize,
        output_rows: usize,
        elapsed: Duration,
    ) {
        let observation = &mut self.observations[index];
        observation.samples = observation.samples.saturating_add(1);
        observation.elapsed_ns = observation
            .elapsed_ns
            .saturating_add(u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX).max(1));
        let input_rows = u64::try_from(input_rows).unwrap_or(u64::MAX);
        let output_rows = u64::try_from(output_rows).unwrap_or(u64::MAX);
        observation.input_rows = observation.input_rows.saturating_add(input_rows);
        observation.rejected_rows = observation
            .rejected_rows
            .saturating_add(input_rows.saturating_sub(output_rows));
    }
}

fn compare_cost(left: usize, a: Observation, right: usize, b: Observation) -> Ordering {
    match (a.rejected_rows == 0, b.rejected_rows == 0) {
        (true, false) => return Ordering::Greater,
        (false, true) => return Ordering::Less,
        (true, true) => return left.cmp(&right),
        (false, false) => {}
    }

    (u128::from(a.elapsed_ns) * u128::from(b.rejected_rows))
        .cmp(&(u128::from(b.elapsed_ns) * u128::from(a.rejected_rows)))
        .then_with(|| left.cmp(&right))
}

fn compare_selectivity(left: usize, a: Observation, right: usize, b: Observation) -> Ordering {
    (u128::from(b.rejected_rows) * u128::from(a.input_rows))
        .cmp(&(u128::from(a.rejected_rows) * u128::from(b.input_rows)))
        .then_with(|| left.cmp(&right))
}

fn unlocks_sparse_evaluation(observation: &Observation) -> bool {
    observation.input_rows != 0
        && observation.rejected_rows as f64 / observation.input_rows as f64
            > 1.0 - EXPR_EVAL_THRESHOLD
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::AdaptiveOrdering;
    use super::OrderingGoal;

    #[test]
    fn cost_order_waits_for_all_entries_then_uses_rejection_efficiency() {
        let mut ordering = AdaptiveOrdering::new(3, OrderingGoal::CostPerRejectedRow);
        assert_eq!(ordering.order(), vec![0, 1, 2]);

        ordering.observe(0, 100, 99, Duration::from_nanos(10));
        ordering.observe(1, 100, 95, Duration::from_nanos(10));
        assert_eq!(ordering.order(), vec![0, 1, 2]);

        ordering.observe(2, 100, 100, Duration::from_nanos(1));
        assert_eq!(ordering.order(), vec![1, 0, 2]);
    }

    #[test]
    fn within_group_order_changes_only_to_unlock_sparse_evaluation() {
        let mut ordering = AdaptiveOrdering::new(2, OrderingGoal::SparseEvaluation);
        ordering.observe(0, 100, 60, Duration::ZERO);
        ordering.observe(1, 100, 30, Duration::ZERO);
        assert_eq!(ordering.order(), vec![0, 1]);

        ordering.observe(1, 100, 0, Duration::ZERO);
        assert_eq!(ordering.order(), vec![1, 0]);
    }
}
