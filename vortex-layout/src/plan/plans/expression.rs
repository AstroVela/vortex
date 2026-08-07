// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::TypeId;
use std::borrow::Cow;
use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use vortex_array::MaskFuture;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::traversal::NodeExt;
use vortex_array::expr::traversal::Transformed;
use vortex_array::expr::traversal::TraversalOrder;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::scalar_fn::fns::get_item::GetItem;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::plan::Plan;
use crate::plan::PlanArrayFuture;
use crate::plan::PlanExecutionContext;
use crate::plan::PlanRef;
use crate::plan::optimize_child;
use crate::plan::optimizer::reduce_parent;

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

    pub(crate) fn new_ref(expression: BoundExpression, child: PlanRef) -> PlanRef {
        Arc::new(Self::new(expression, child))
    }

    fn optimize_top_down(&self, blocked_child_type: Option<TypeId>) -> VortexResult<PlanRef> {
        if self.expression.is_root() {
            return optimize_child(&self.child);
        }
        if let Some(inner) = self.child.downcast_ref::<Self>() {
            let expression = replace_root(self.expression.clone(), inner.expression.clone())?;
            return Self::new(expression, Arc::clone(&inner.child)).optimize_top_down(None);
        }

        let child_type = self.child.as_ref().type_id();
        let parent = Self::new_ref(self.expression.clone(), Arc::clone(&self.child));
        if blocked_child_type != Some(child_type)
            && let Some(rewritten) = reduce_parent(&parent, 0)?
        {
            return Self::optimize_rewrite(rewritten, child_type);
        }

        let child = optimize_child(&self.child)?;
        if let Some(inner) = child.downcast_ref::<Self>() {
            let expression = replace_root(self.expression.clone(), inner.expression.clone())?;
            return Self::new(expression, Arc::clone(&inner.child)).optimize_top_down(None);
        }

        let child_type = child.as_ref().type_id();
        let parent = Self::new_ref(self.expression.clone(), child);
        if blocked_child_type != Some(child_type)
            && let Some(rewritten) = reduce_parent(&parent, 0)?
        {
            return Self::optimize_rewrite(rewritten, child_type);
        }
        Ok(parent)
    }

    fn optimize_rewrite(rewritten: PlanRef, previous_child_type: TypeId) -> VortexResult<PlanRef> {
        let Some(expression) = rewritten.downcast_ref::<Self>() else {
            if !rewritten.needs_optimize() {
                return Ok(rewritten);
            }
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
        Ok(async move { child.await?.apply_bound(&expression) }.boxed())
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

pub(super) fn rewrite_partition_root(
    expression: BoundExpression,
    root_dtype: DType,
    collapsed: &[(FieldName, FieldName)],
) -> VortexResult<BoundExpression> {
    Ok(expression
        .transform_down(|node| {
            if let Some(value_name) = node
                .as_scalar()
                .and_then(|scalar_fn| scalar_fn.as_opt::<GetItem>())
            {
                let partition_access = &node.children()[0];
                if let Some(partition_name) = partition_access
                    .as_scalar()
                    .and_then(|scalar_fn| scalar_fn.as_opt::<GetItem>())
                    && partition_access.children()[0].is_root()
                    && collapsed.iter().any(|(partition, value)| {
                        partition == partition_name && value == value_name
                    })
                {
                    return Ok(Transformed {
                        value: BoundExpression::try_new(
                            GetItem.bind(partition_name.clone()),
                            [BoundExpression::new_root(root_dtype.clone())],
                        )?,
                        changed: true,
                        order: TraversalOrder::Skip,
                    });
                }
            }

            if node.is_root() {
                Ok(Transformed {
                    value: BoundExpression::new_root(root_dtype.clone()),
                    changed: true,
                    order: TraversalOrder::Skip,
                })
            } else {
                Ok(Transformed::no(node))
            }
        })?
        .into_inner())
}
