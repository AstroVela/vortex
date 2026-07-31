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
use crate::plan::Plan;
use crate::plan::PlanRef;
use crate::segments::SegmentSource;

/// An opaque plan for a layout without a specialized plan implementation.
///
/// Unlike [`LayoutReaderPlan`], this node retains reader-construction inputs and therefore does not
/// construct its reader until after plan optimization.
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

/// A compatibility plan for an already-created layout reader.
///
/// New layout-backed entry points should use [`crate::Layout::new_plan`]. This node remains for
/// callers that only have a reader and cannot reconstruct its layout tree.
pub struct LayoutReaderPlan {
    reader: LayoutReaderRef,
}

impl LayoutReaderPlan {
    /// Creates a compatibility plan for `reader`.
    pub fn new(reader: LayoutReaderRef) -> Self {
        Self { reader }
    }
}

impl Plan for LayoutReaderPlan {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn optimize(&self) -> VortexResult<PlanRef> {
        Ok(Arc::new(Self::new(Arc::clone(&self.reader))))
    }

    fn new_reader(
        &self,
        _name: Arc<str>,
        _segment_source: Arc<dyn SegmentSource>,
        _session: &VortexSession,
        _ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::clone(&self.reader))
    }

    fn dtype(&self) -> &DType {
        self.reader.dtype()
    }

    fn row_count(&self) -> u64 {
        self.reader.row_count()
    }
}
