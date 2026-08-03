// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_error::VortexResult;

use crate::layouts::chunked::ChunkedLayout;
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

    fn with_chunks(&self, chunks: LazyPlanChildren) -> Self {
        Self {
            layout: self.layout.clone(),
            dtype: self.dtype.clone(),
            chunks,
        }
    }
}

impl Plan for ChunkedPlan {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &'static str {
        "ChunkedPlan"
    }

    fn optimize(&self) -> VortexResult<PlanRef> {
        let chunks = self.chunks.map(|_, chunk| chunk.optimize());
        Ok(Arc::new(self.with_chunks(chunks)))
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
