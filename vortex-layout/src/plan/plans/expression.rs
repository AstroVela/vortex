// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_array::expr::is_root;
use vortex_array::expr::root;
use vortex_array::expr::transform::replace;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::plan::Plan;
use crate::plan::PlanRef;
use crate::plan::optimizer::reduce_parent;

/// A physical plan that applies an expression to the output of `child`.
pub struct ExpressionPlan {
    expression: Expression,
    child: PlanRef,
    dtype: DType,
}

impl ExpressionPlan {
    /// Creates an expression plan and validates its output dtype.
    pub fn try_new(expression: Expression, child: PlanRef) -> VortexResult<Self> {
        let dtype = expression.return_dtype(child.dtype())?;
        Ok(Self {
            expression,
            child,
            dtype,
        })
    }

    /// Returns the expression evaluated by this plan.
    pub fn expression(&self) -> &Expression {
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
        let expression = self.expression.optimize_recursive(child.dtype())?;
        if is_root(&expression) {
            return Ok(child);
        }
        if let Some(inner) = child.downcast_ref::<Self>() {
            let expression = replace(expression, &root(), inner.expression.clone());
            return Self::try_new(expression, Arc::clone(&inner.child))?.optimize();
        }
        let parent: PlanRef = Arc::new(Self::try_new(expression, child)?);
        if let Some(rewritten) = reduce_parent(&parent, 0)? {
            return Ok(rewritten);
        }
        Ok(parent)
    }

    fn dtype(&self) -> &DType {
        &self.dtype
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
