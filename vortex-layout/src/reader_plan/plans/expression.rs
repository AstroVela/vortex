// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::borrow::Cow;
use std::ops::Range;
use std::sync::Arc;

use vortex_array::MaskFuture;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::Expression;
use vortex_array::expr::is_root;
use vortex_array::expr::root;
use vortex_array::expr::transform::replace;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::ArrayFuture;
use crate::LayoutReader;
use crate::LayoutReaderRef;
use crate::RowSplits;
use crate::SplitRange;
use crate::reader_plan::Plan;
use crate::reader_plan::PlanRef;
use crate::segments::SegmentSource;

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

    /// Creates a shared expression plan.
    pub fn new_ref(expression: Expression, child: PlanRef) -> VortexResult<PlanRef> {
        if let Some(inner) = child.as_any().downcast_ref::<Self>() {
            let expression = replace(expression, &root(), inner.expression.clone());
            return Ok(Arc::new(Self::try_new(
                expression,
                Arc::clone(&inner.child),
            )?));
        }
        Ok(Arc::new(Self::try_new(expression, child)?))
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
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &'static str {
        "ExpressionPlan"
    }

    fn optimize(&self) -> VortexResult<PlanRef> {
        let child = self.child.optimize()?;
        let expression = self.expression.optimize_recursive(child.dtype())?;
        if is_root(&expression) {
            return Ok(child);
        }
        Self::new_ref(expression, child)
    }

    fn new_reader(
        &self,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &crate::LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        ExpressionReader::try_new_ref(
            self.expression.clone(),
            self.child.new_reader(name, segment_source, session, ctx)?,
        )
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

/// A reader that binds an expression to the output of another reader.
pub(crate) struct ExpressionReader {
    expression: BoundExpression,
    child: LayoutReaderRef,
    dtype: DType,
}

impl ExpressionReader {
    fn try_new(expression: Expression, child: LayoutReaderRef) -> VortexResult<Self> {
        let expression = expression.bind(child.dtype())?;
        let dtype = expression.dtype().clone();
        Ok(Self {
            expression,
            child,
            dtype,
        })
    }

    pub(crate) fn try_new_ref(
        expression: Expression,
        child: LayoutReaderRef,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(Self::try_new(expression, child)?))
    }

    fn ensure_root(expression: &BoundExpression) -> VortexResult<()> {
        vortex_ensure!(
            expression.is_root(),
            "layout-v27 expression readers only accept root evaluation"
        );
        Ok(())
    }
}

impl LayoutReader for ExpressionReader {
    fn name(&self) -> &Arc<str> {
        self.child.name()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count(&self) -> u64 {
        self.child.row_count()
    }

    fn register_splits(
        &self,
        _field_mask: &[FieldMask],
        split_range: &SplitRange,
        splits: &mut RowSplits,
    ) -> VortexResult<()> {
        // Mapping output field masks back through an arbitrary expression is an optimization. All
        // child fields is conservative and preserves the child's natural split boundaries.
        self.child
            .register_splits(&[FieldMask::All], split_range, splits)
    }

    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expression: &BoundExpression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        Self::ensure_root(expression)?;
        self.child
            .pruning_evaluation(row_range, &self.expression, mask)
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expression: &BoundExpression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        Self::ensure_root(expression)?;
        self.child
            .filter_evaluation(row_range, &self.expression, mask)
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expression: &BoundExpression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        Self::ensure_root(expression)?;
        self.child
            .projection_evaluation(row_range, &self.expression, mask)
    }
}
