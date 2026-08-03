// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;

use crate::LayoutReaderContext;
use crate::LayoutReaderRef;
use crate::layouts::chunked::ChunkedLayout;
use crate::layouts::chunked::reader::ChunkedReader;
use crate::plan::Plan;
use crate::plan::PlanRef;
use crate::segments::SegmentSource;

/// A physical plan with one child per row chunk.
pub struct ChunkedPlan {
    layout: ChunkedLayout,
    dtype: DType,
    chunks: Arc<[PlanRef]>,
}

impl ChunkedPlan {
    pub(crate) fn try_new(layout: &ChunkedLayout) -> VortexResult<Self> {
        let chunks = (0..layout.nchildren())
            .map(|index| {
                layout
                    .slot(index)?
                    .ok_or_else(|| vortex_error::vortex_err!("Missing chunk {index}"))?
                    .new_plan()
            })
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(Self {
            layout: layout.clone(),
            dtype: layout.dtype().clone(),
            chunks: chunks.into(),
        })
    }

    fn try_with_chunks(&self, chunks: Vec<PlanRef>) -> VortexResult<Self> {
        let dtype = chunks
            .first()
            .map(|chunk| chunk.dtype().clone())
            .unwrap_or_else(|| self.dtype.clone());
        for (index, chunk) in chunks.iter().enumerate() {
            vortex_ensure!(
                chunk.dtype() == &dtype,
                "Chunk {index} plan dtype {} does not match {}",
                chunk.dtype(),
                dtype
            );
        }
        Ok(Self {
            layout: self.layout.clone(),
            dtype,
            chunks: chunks.into(),
        })
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
        let chunks = self
            .chunks
            .iter()
            .map(|chunk| chunk.optimize())
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(Arc::new(self.try_with_chunks(chunks)?))
    }

    fn new_reader(
        &self,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        let chunks = self
            .chunks
            .iter()
            .enumerate()
            .map(|(index, chunk)| {
                chunk.new_reader(
                    format!("{name}.[{index}]").into(),
                    Arc::clone(&segment_source),
                    session,
                    ctx,
                )
            })
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(Arc::new(ChunkedReader::try_new_with_readers(
            self.layout.clone(),
            self.dtype.clone(),
            name,
            chunks,
        )?))
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
        Ok(self.chunks.get(index).cloned())
    }

    fn child_name(&self, index: usize) -> Cow<'_, str> {
        Cow::Owned(format!("chunks[{index}]"))
    }
}
