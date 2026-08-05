// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::sync::Arc;

use vortex_array::expr::ExactBoundExpr;
use vortex_array::expr::label_bound_tree;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::layouts::dict::DictLayout;
use crate::plan::ExpressionPlan;
use crate::plan::Plan;
use crate::plan::PlanRef;
use crate::plan::new_plan;
use crate::plan::optimizer::PlanParentReduceRule;

/// A physical dictionary plan with children ordered as `[codes, values]`.
pub struct DictPlan {
    layout: DictLayout,
    dtype: vortex_array::dtype::DType,
    codes: PlanRef,
    values: PlanRef,
}

impl DictPlan {
    pub(crate) fn try_new(layout: &DictLayout) -> VortexResult<Self> {
        // Dict serialization stores values before codes; the plan order is deliberately codes,
        // values because that is the optimizer-facing logical shape.
        let codes = new_plan(
            &layout
                .slot(1)?
                .ok_or_else(|| vortex_error::vortex_err!("Dictionary codes child is absent"))?,
        )?;
        let values = new_plan(
            &layout
                .slot(0)?
                .ok_or_else(|| vortex_error::vortex_err!("Dictionary values child is absent"))?,
        )?;
        Ok(Self {
            layout: layout.clone(),
            dtype: values.dtype().clone(),
            codes,
            values,
        })
    }

    fn with_children(&self, codes: PlanRef, values: PlanRef) -> Self {
        Self {
            layout: self.layout.clone(),
            dtype: values.dtype().clone(),
            codes,
            values,
        }
    }
}

impl Plan for DictPlan {
    fn name(&self) -> &'static str {
        "DictPlan"
    }

    fn optimize(&self) -> VortexResult<PlanRef> {
        let codes = self.codes.optimize()?;
        let values = self.values.optimize()?;
        Ok(Arc::new(self.with_children(codes, values)))
    }

    fn dtype(&self) -> &vortex_array::dtype::DType {
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
            0 => Ok(Some(Arc::clone(&self.codes))),
            1 => Ok(Some(Arc::clone(&self.values))),
            _ => vortex_bail!("Dictionary plan has no child {index}"),
        }
    }

    fn child_name(&self, index: usize) -> Cow<'_, str> {
        match index {
            0 => Cow::Borrowed("codes"),
            1 => Cow::Borrowed("values"),
            _ => Cow::Owned(format!("child[{index}]")),
        }
    }
}

/// Pushes a safe boolean expression into dictionary values.
#[derive(Debug)]
pub(crate) struct ExpressionDictRule;

impl PlanParentReduceRule<DictPlan> for ExpressionDictRule {
    type Parent = ExpressionPlan;

    fn reduce_parent(
        &self,
        child: &DictPlan,
        parent: &ExpressionPlan,
        _child_idx: usize,
    ) -> VortexResult<Option<PlanRef>> {
        let expression = parent.expression();
        if !expression.dtype().is_boolean() {
            return Ok(None);
        }
        let labels = label_bound_tree(
            expression,
            |node| match node.as_scalar() {
                Some(scalar_fn) => (
                    false,
                    scalar_fn.signature().is_strict(),
                    scalar_fn.signature().is_fallible(),
                ),
                None => (true, true, false),
            },
            |acc, &child| (acc.0 | child.0, acc.1 & child.1, acc.2 | child.2),
        );
        let (references_root, is_strict, is_fallible) = labels
            .get(&ExactBoundExpr(expression.clone()))
            .copied()
            .unwrap_or((false, false, true));
        if !references_root || !is_strict || is_fallible {
            return Ok(None);
        }

        let values =
            ExpressionPlan::new(expression.clone(), Arc::clone(&child.values)).optimize()?;
        Ok(Some(Arc::new(
            child.with_children(Arc::clone(&child.codes), values),
        )))
    }
}
