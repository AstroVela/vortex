// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_error::VortexResult;

use crate::LayoutRef;
use crate::plan::Plan;
use crate::plan::PlanRef;

/// An opaque plan for a layout without a specialized plan implementation.
///
/// This node keeps an unsupported layout opaque while retaining its dtype and row domain.
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

    fn dtype(&self) -> &DType {
        self.layout.dtype()
    }

    fn row_count(&self) -> u64 {
        self.layout.row_count()
    }
}
