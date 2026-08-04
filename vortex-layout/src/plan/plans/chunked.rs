// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_array::expr::label_tree;
use vortex_error::VortexResult;

use crate::layouts::chunked::ChunkedLayout;
use crate::layouts::row_idx::RowIdx;
use crate::plan::ExpressionPlan;
use crate::plan::LazyPlanChildren;
use crate::plan::Plan;
use crate::plan::PlanRef;
use crate::plan::new_plan;

/// A physical plan with one child per row chunk.
pub struct ChunkedPlan {
    layout: ChunkedLayout,
    dtype: DType,
    chunks: LazyPlanChildren,
}

impl ChunkedPlan {
    pub(crate) fn new(layout: &ChunkedLayout) -> Self {
        let child_layout = layout.clone();
        let chunks = LazyPlanChildren::new(layout.nchildren(), move |index| {
            let child = child_layout
                .slot(index)?
                .ok_or_else(|| vortex_error::vortex_err!("Missing chunk {index}"))?;
            Ok(Some(new_plan(&child)?))
        });
        Self {
            layout: layout.clone(),
            dtype: layout.dtype().clone(),
            chunks,
        }
    }

    fn with_chunks(&self, dtype: DType, chunks: LazyPlanChildren) -> Self {
        Self {
            layout: self.layout.clone(),
            dtype,
            chunks,
        }
    }
}

impl Plan for ChunkedPlan {
    fn name(&self) -> &'static str {
        "ChunkedPlan"
    }

    fn optimize(&self) -> VortexResult<PlanRef> {
        let chunks = self.chunks.try_map(|_, chunk| chunk.optimize())?;
        Ok(Arc::new(self.with_chunks(self.dtype.clone(), chunks)))
    }

    fn optimize_expression(&self, expression: &Expression) -> VortexResult<Option<PlanRef>> {
        let references_row_idx = label_tree(
            expression,
            |node| node.is::<RowIdx>(),
            |acc, &child| acc | child,
        )
        .get(expression)
        .copied()
        .unwrap_or(false);
        if references_row_idx {
            return Ok(None);
        }

        let dtype = expression.return_dtype(&self.dtype)?;
        let chunks = self
            .chunks
            .try_map(|_, chunk| ExpressionPlan::try_new(expression.clone(), chunk)?.optimize())?;
        Ok(Some(Arc::new(self.with_chunks(dtype, chunks))))
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count(&self) -> u64 {
        self.layout.row_count()
    }

    fn child_count(&self) -> usize {
        self.chunks.len()
    }

    fn child(&self, index: usize) -> VortexResult<Option<PlanRef>> {
        if index >= self.chunks.len() {
            return Ok(None);
        }
        self.chunks.get(index)
    }

    fn child_name(&self, index: usize) -> Cow<'_, str> {
        Cow::Owned(format!("chunks[{index}]"))
    }
}
