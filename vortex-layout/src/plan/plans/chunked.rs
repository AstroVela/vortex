// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::future;
use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::TryStreamExt;
use futures::stream::FuturesOrdered;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::arrays::ChunkedArray;
use vortex_array::dtype::DType;
use vortex_array::expr::ExactBoundExpr;
use vortex_array::expr::label_bound_tree;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::layouts::chunked::ChunkedLayout;
use crate::layouts::row_idx::RowIdx;
use crate::plan::ExpressionPlan;
use crate::plan::LazyPlanChildren;
use crate::plan::Plan;
use crate::plan::PlanArrayFuture;
use crate::plan::PlanExecutionContext;
use crate::plan::PlanRef;
use crate::plan::new_plan;
use crate::plan::optimizer::PlanParentReduceRule;

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

    fn execute(
        &self,
        ctx: &PlanExecutionContext,
        row_range: &Range<u64>,
        mask: MaskFuture,
    ) -> VortexResult<PlanArrayFuture> {
        vortex_ensure!(
            row_range.start <= row_range.end && row_range.end <= self.row_count(),
            "Chunked plan row range {:?} is outside 0..{}",
            row_range,
            self.row_count()
        );
        vortex_ensure!(
            mask.len() == usize::try_from(row_range.end - row_range.start)?,
            "Chunked plan mask length mismatch"
        );
        if row_range.is_empty() {
            let empty = Canonical::empty(&self.dtype).into_array();
            return Ok(future::ready(Ok(empty)).boxed());
        }

        // Visit only the chunks overlapping the row range instead of scanning every chunk.
        let chunk_offsets = self.layout.data().chunk_offsets();
        let first_chunk = chunk_offsets
            .partition_point(|&offset| offset <= row_range.start)
            .saturating_sub(1);
        let mut chunk_futures = Vec::new();
        for chunk_index in first_chunk..self.chunks.len() {
            let chunk_offset = chunk_offsets[chunk_index];
            if chunk_offset >= row_range.end {
                break;
            }
            let chunk_end = chunk_offsets[chunk_index + 1];
            let start = row_range.start.max(chunk_offset);
            let end = row_range.end.min(chunk_end);
            if start >= end {
                continue;
            }
            let chunk = self
                .chunks
                .get(chunk_index)?
                .ok_or_else(|| vortex_error::vortex_err!("Chunk {chunk_index} has no plan"))?;
            let child_range = start - chunk_offset..end - chunk_offset;
            let mask_range =
                usize::try_from(start - row_range.start)?..usize::try_from(end - row_range.start)?;
            let child_mask = if mask_range.start == 0 && mask_range.end == mask.len() {
                mask.clone()
            } else {
                mask.slice(mask_range)
            };
            chunk_futures.push(chunk.execute(ctx, &child_range, child_mask)?);
        }

        Ok(async move {
            let chunks: Vec<_> = FuturesOrdered::from_iter(chunk_futures)
                .try_collect()
                .await?;
            vortex_ensure!(!chunks.is_empty(), "Non-empty row range selected no chunks");
            if chunks.len() == 1 {
                return Ok(chunks.into_iter().next().vortex_expect("one chunk"));
            }
            let dtype = chunks[0].dtype().clone();
            Ok(ChunkedArray::try_new(chunks, dtype)?.into_array())
        }
        .boxed())
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

/// Pushes an expression through every chunk of a chunked plan.
#[derive(Debug)]
pub(crate) struct ExpressionChunkedRule;

impl PlanParentReduceRule<ChunkedPlan> for ExpressionChunkedRule {
    type Parent = ExpressionPlan;

    fn reduce_parent(
        &self,
        child: &ChunkedPlan,
        parent: &ExpressionPlan,
        _child_idx: usize,
    ) -> VortexResult<Option<PlanRef>> {
        let expression = parent.expression();
        let references_row_idx = label_bound_tree(
            expression,
            |node| {
                node.as_scalar()
                    .is_some_and(|scalar_fn| scalar_fn.is::<RowIdx>())
            },
            |acc, &child| acc | child,
        )
        .get(&ExactBoundExpr(expression.clone()))
        .copied()
        .unwrap_or(false);
        if references_row_idx {
            return Ok(None);
        }

        let dtype = expression.dtype().clone();
        let chunks = child
            .chunks
            .try_map(|_, chunk| Ok(ExpressionPlan::new_ref(expression.clone(), chunk)))?;
        Ok(Some(Arc::new(child.with_chunks(dtype, chunks))))
    }
}
