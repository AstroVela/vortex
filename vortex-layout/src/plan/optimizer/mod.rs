// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Static parent-child rewrite rules for physical plans.

mod rules;

pub use rules::DynPlanParentReduceRule;
pub use rules::PlanParentReduceRule;
pub use rules::PlanParentReduceRuleAdapter;
pub use rules::PlanParentRuleSet;
use vortex_error::VortexResult;

use super::ChunkedPlan;
use super::DictPlan;
use super::PlanRef;
use super::RowIdxPlan;
use super::StructPlan;
use super::plans::ExpressionChunkedRule;
use super::plans::ExpressionDictRule;
use super::plans::ExpressionRowIdxRule;
use super::plans::ExpressionStructRule;

static EXPRESSION_CHUNKED_RULE: PlanParentReduceRuleAdapter<ChunkedPlan, ExpressionChunkedRule> =
    PlanParentReduceRuleAdapter::new(ExpressionChunkedRule);
static EXPRESSION_DICT_RULE: PlanParentReduceRuleAdapter<DictPlan, ExpressionDictRule> =
    PlanParentReduceRuleAdapter::new(ExpressionDictRule);
static EXPRESSION_ROW_IDX_RULE: PlanParentReduceRuleAdapter<RowIdxPlan, ExpressionRowIdxRule> =
    PlanParentReduceRuleAdapter::new(ExpressionRowIdxRule);
static EXPRESSION_STRUCT_RULE: PlanParentReduceRuleAdapter<StructPlan, ExpressionStructRule> =
    PlanParentReduceRuleAdapter::new(ExpressionStructRule);

static PARENT_RULES: PlanParentRuleSet = PlanParentRuleSet::new(&[
    &EXPRESSION_CHUNKED_RULE,
    &EXPRESSION_DICT_RULE,
    &EXPRESSION_ROW_IDX_RULE,
    &EXPRESSION_STRUCT_RULE,
]);

/// Attempts a static rewrite for `parent` and its child at `child_idx`.
pub(crate) fn reduce_parent(parent: &PlanRef, child_idx: usize) -> VortexResult<Option<PlanRef>> {
    let Some(child) = parent.child(child_idx)? else {
        return Ok(None);
    };
    PARENT_RULES.evaluate(&child, parent, child_idx)
}
