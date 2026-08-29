// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::LayoutReaderContext;
use crate::LayoutReaderRef;
use crate::LayoutRef;
use crate::reader_plan::Plan;
use crate::reader_plan::PlanRef;
use crate::segments::SegmentSource;

/// An opaque plan for a layout without a specialized plan implementation.
///
/// This node retains reader-construction inputs and therefore does not construct its reader until
/// after plan optimization.
pub(crate) struct LayoutPlan {
    layout: LayoutRef,
}

impl LayoutPlan {
    pub(crate) fn new(layout: LayoutRef) -> Self {
        Self { layout }
    }
}

impl Plan for LayoutPlan {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &'static str {
        "LayoutPlan"
    }

    fn optimize(&self) -> VortexResult<PlanRef> {
        Ok(Arc::new(Self::new(Arc::clone(&self.layout))))
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

    fn dtype(&self) -> &DType {
        self.layout.dtype()
    }

    fn row_count(&self) -> u64 {
        self.layout.row_count()
    }
}
