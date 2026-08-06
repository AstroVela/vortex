// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::traversal::NodeExt;
use vortex_array::expr::traversal::Transformed;
use vortex_array::expr::traversal::TraversalOrder;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::plan::Plan;
use crate::plan::PlanRef;

/// A physical plan that applies an expression to the output of `child`.
pub struct ExpressionPlan {
    expression: BoundExpression,
    child: PlanRef,
}

impl ExpressionPlan {
    /// Creates an expression plan from an expression bound to the child's dtype.
    pub fn new(expression: BoundExpression, child: PlanRef) -> Self {
        Self { expression, child }
    }

    /// Returns the expression evaluated by this plan.
    pub fn expression(&self) -> &BoundExpression {
        &self.expression
    }

    /// Returns the child plan supplying the expression root.
    pub fn child_plan(&self) -> &PlanRef {
        &self.child
    }
}

impl Plan for ExpressionPlan {
    fn name(&self) -> &'static str {
        "ExpressionPlan"
    }

    fn optimize(&self) -> VortexResult<PlanRef> {
        let child = self.child.optimize()?;
        if self.expression.is_root() {
            return Ok(child);
        }
        if let Some(inner) = child.downcast_ref::<Self>() {
            let expression = replace_root(self.expression.clone(), inner.expression.clone())?;
            return Ok(Arc::new(Self::new(expression, Arc::clone(&inner.child))));
        }
        Ok(Arc::new(Self::new(self.expression.clone(), child)))
    }

    fn dtype(&self) -> &DType {
        self.expression.dtype()
    }

    fn row_count(&self) -> u64 {
        self.child.row_count()
    }

    fn child_count(&self) -> usize {
        1
    }

    fn child(&self, index: usize) -> VortexResult<Option<PlanRef>> {
        if index != 0 {
            vortex_bail!("Expression plan has no child {index}")
        }
        Ok(Some(Arc::clone(&self.child)))
    }

    fn child_name(&self, index: usize) -> Cow<'_, str> {
        if index == 0 {
            Cow::Borrowed("child")
        } else {
            Cow::Owned(format!("child[{index}]"))
        }
    }
}

fn replace_root(
    expression: BoundExpression,
    replacement: BoundExpression,
) -> VortexResult<BoundExpression> {
    Ok(expression
        .transform_down(|node| {
            if node.is_root() {
                Ok(Transformed {
                    value: replacement.clone(),
                    order: TraversalOrder::Skip,
                    changed: true,
                })
            } else {
                Ok(Transformed::no(node))
            }
        })?
        .into_inner())
}
