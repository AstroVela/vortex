// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use vortex_array::MaskFuture;
use vortex_array::serde::SerializedArray;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::layouts::flat::FlatLayout;
use crate::plan::Plan;
use crate::plan::PlanArrayFuture;
use crate::plan::PlanExecutionContext;
use crate::plan::PlanRef;

/// A physical plan for a flat layout.
pub struct FlatPlan {
    layout: FlatLayout,
}

impl FlatPlan {
    pub(crate) fn new(layout: &FlatLayout) -> Self {
        Self {
            layout: layout.clone(),
        }
    }
}

impl Plan for FlatPlan {
    fn name(&self) -> &'static str {
        "FlatPlan"
    }

    fn optimize(&self) -> VortexResult<PlanRef> {
        Ok(Arc::new(Self::new(&self.layout)))
    }

    fn needs_optimize(&self) -> bool {
        false
    }

    fn execute(
        &self,
        ctx: &PlanExecutionContext,
        row_range: &Range<u64>,
        mask: MaskFuture,
    ) -> VortexResult<PlanArrayFuture> {
        vortex_ensure!(
            row_range.start <= row_range.end && row_range.end <= self.layout.row_count(),
            "Flat plan row range {:?} is outside 0..{}",
            row_range,
            self.layout.row_count()
        );
        let row_count = usize::try_from(self.layout.row_count())?;
        let row_range = usize::try_from(row_range.start)?..usize::try_from(row_range.end)?;
        vortex_ensure!(
            mask.len() == row_range.len(),
            "Flat plan mask length mismatch"
        );

        let segment = ctx.segment_source().request(self.layout.segment_id());
        let array_ctx = self.layout.array_ctx().clone();
        let array_tree = self.layout.array_tree().cloned();
        let dtype = self.layout.dtype().clone();
        let session = ctx.session().clone();

        Ok(async move {
            let segment = segment.await?;
            let serialized = if let Some(array_tree) = array_tree {
                SerializedArray::from_flatbuffer_and_segment(array_tree, segment)?
            } else {
                SerializedArray::try_from(segment)?
            };
            let mut array = serialized.decode(&dtype, row_count, &array_ctx, &session)?;
            if row_range.start > 0 || row_range.end < array.len() {
                array = array.slice(row_range)?;
            }
            let mask = mask.await?;
            if !mask.all_true() {
                array = array.filter(mask)?;
            }
            Ok(array)
        }
        .boxed())
    }

    fn dtype(&self) -> &vortex_array::dtype::DType {
        self.layout.dtype()
    }

    fn row_count(&self) -> u64 {
        self.layout.row_count()
    }
}
