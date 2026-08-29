// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::borrow::Cow;
use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use crate::LayoutReaderContext;
use crate::LayoutReaderRef;
use crate::layouts::row_idx::RowIdxLayoutReader;
use crate::reader_plan::Plan;
use crate::reader_plan::PlanRef;
use crate::segments::SegmentSource;

/// A physical plan that adds row-index expression support to its child.
pub(crate) struct RowIdxPlan {
    row_offset: u64,
    child: PlanRef,
}

impl RowIdxPlan {
    pub(crate) fn new_ref(row_offset: u64, child: PlanRef) -> PlanRef {
        Arc::new(Self { row_offset, child })
    }
}

impl Plan for RowIdxPlan {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &'static str {
        "RowIdxPlan"
    }

    fn optimize(&self) -> VortexResult<PlanRef> {
        Ok(Self::new_ref(self.row_offset, self.child.optimize()?))
    }

    fn new_reader(
        &self,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        let child = self.child.new_reader(name, segment_source, session, ctx)?;
        Ok(Arc::new(RowIdxLayoutReader::new(
            self.row_offset,
            child,
            session.clone(),
        )))
    }

    fn dtype(&self) -> &DType {
        self.child.dtype()
    }

    fn row_count(&self) -> u64 {
        self.child.row_count()
    }

    fn child_count(&self) -> usize {
        1
    }

    fn child(&self, index: usize) -> VortexResult<Option<PlanRef>> {
        if index != 0 {
            vortex_bail!("Row-index plan has no child {index}")
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
