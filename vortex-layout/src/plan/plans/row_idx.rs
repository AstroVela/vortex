// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::plan::Plan;
use crate::plan::PlanRef;

/// A physical plan that adds row-index expression support to its child.
pub struct RowIdxPlan {
    row_offset: u64,
    child: PlanRef,
}

impl RowIdxPlan {
    /// Creates a shared row-index plan with `row_offset` applied to its child domain.
    pub fn new_ref(row_offset: u64, child: PlanRef) -> PlanRef {
        Arc::new(Self { row_offset, child })
    }
}

impl Plan for RowIdxPlan {
    fn name(&self) -> &'static str {
        "RowIdxPlan"
    }

    fn optimize(&self) -> VortexResult<PlanRef> {
        Ok(Self::new_ref(self.row_offset, self.child.optimize()?))
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
