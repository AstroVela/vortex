// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::LayoutReaderContext;
use crate::LayoutReaderRef;
use crate::layouts::flat::FlatLayout;
use crate::plan::Plan;
use crate::plan::PlanRef;
use crate::segments::SegmentSource;

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
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn optimize(&self) -> VortexResult<PlanRef> {
        Ok(Arc::new(Self::new(&self.layout)))
    }

    fn new_reader(
        &self,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        self.layout.new_reader(name, segment_source, session, ctx)
    }

    fn dtype(&self) -> &vortex_array::dtype::DType {
        self.layout.dtype()
    }

    fn row_count(&self) -> u64 {
        self.layout.row_count()
    }
}
