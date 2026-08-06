// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::TypeId;
use std::borrow::Cow;
use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use vortex_array::MaskFuture;
use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_array::expr::is_root;
use vortex_array::expr::root;
use vortex_array::expr::transform::replace;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::plan::Plan;
use crate::plan::PlanArrayFuture;
use crate::plan::PlanExecutionContext;
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

    pub(crate) fn try_new_ref(expression: Expression, child: PlanRef) -> VortexResult<PlanRef> {
        Ok(Arc::new(Self::try_new(expression, child)?))
    }

    fn optimize_top_down(&self, blocked_child_type: Option<TypeId>) -> VortexResult<PlanRef> {
        let expression = self.expression.optimize_recursive(self.child.dtype())?;
        if is_root(&expression) {
            return self.child.optimize();
        }
        if let Some(inner) = self.child.downcast_ref::<Self>() {
            let expression = replace(expression, &root(), inner.expression.clone());
            return Self::try_new(expression, Arc::clone(&inner.child))?.optimize_top_down(None);
        }

        let child_type = self.child.as_ref().type_id();
        let parent = Self::try_new_ref(expression.clone(), Arc::clone(&self.child))?;
        if blocked_child_type != Some(child_type)
            && let Some(rewritten) = reduce_parent(&parent, 0)?
        {
            return Self::optimize_rewrite(rewritten, child_type);
        }

        let child = self.child.optimize()?;
        let expression = expression.optimize_recursive(child.dtype())?;
        if is_root(&expression) {
            return Ok(child);
        }
        if let Some(inner) = child.downcast_ref::<Self>() {
            let expression = replace(expression, &root(), inner.expression.clone());
            return Self::try_new(expression, Arc::clone(&inner.child))?.optimize_top_down(None);
        }

        let child_type = child.as_ref().type_id();
        let parent = Self::try_new_ref(expression, child)?;
        if blocked_child_type != Some(child_type)
            && let Some(rewritten) = reduce_parent(&parent, 0)?
        {
            return Self::optimize_rewrite(rewritten, child_type);
        }
        Ok(parent)
    }

    fn optimize_rewrite(rewritten: PlanRef, previous_child_type: TypeId) -> VortexResult<PlanRef> {
        let Some(expression) = rewritten.downcast_ref::<Self>() else {
            return rewritten.optimize();
        };
        let child_type = expression.child.as_ref().type_id();
        // A residual expression may remain above the same child kind after a successful rewrite.
        // Do not immediately apply that rule again; recursively optimize only the retained child.
        let blocked_child_type = (child_type == previous_child_type).then_some(previous_child_type);
        expression.optimize_top_down(blocked_child_type)
    }
}

impl Plan for ExpressionPlan {
    fn name(&self) -> &'static str {
        "ExpressionPlan"
    }

    fn optimize(&self) -> VortexResult<PlanRef> {
        self.optimize_top_down(None)
    }

    fn execute(
        &self,
        ctx: &PlanExecutionContext,
        row_range: &Range<u64>,
        mask: MaskFuture,
    ) -> VortexResult<PlanArrayFuture> {
        let child = self.child.execute(ctx, row_range, mask)?;
        let expression = self.expression.clone();
        Ok(async move { child.await?.apply(&expression) }.boxed())
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
