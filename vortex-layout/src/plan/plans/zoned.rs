// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::LayoutRef;
use crate::plan::Plan;
use crate::plan::PlanRef;
use crate::plan::new_plan;

const DATA_CHILD_INDEX: usize = 0;
const ZONES_CHILD_INDEX: usize = 1;

/// A physical zoned plan with a transparent data child and an auxiliary zones child.
///
/// This plan represents both current `vortex.zoned` layouts and legacy `vortex.stats` layouts,
/// which have the same physical child shape.
pub struct ZonedPlan {
    layout: LayoutRef,
    dtype: DType,
    data: PlanRef,
    zones: PlanRef,
}

impl ZonedPlan {
    pub(crate) fn try_new(layout: &LayoutRef) -> VortexResult<Self> {
        let data = new_plan(
            &layout
                .slot(DATA_CHILD_INDEX)?
                .ok_or_else(|| vortex_err!("Zoned data child is absent"))?,
        )?;
        let zones = new_plan(
            &layout
                .slot(ZONES_CHILD_INDEX)?
                .ok_or_else(|| vortex_err!("Zoned zones child is absent"))?,
        )?;
        Ok(Self {
            layout: Arc::clone(layout),
            dtype: layout.dtype().clone(),
            data,
            zones,
        })
    }

    fn with_children(&self, data: PlanRef, zones: PlanRef) -> Self {
        Self {
            layout: Arc::clone(&self.layout),
            dtype: self.dtype.clone(),
            data,
            zones,
        }
    }
}

impl Plan for ZonedPlan {
    fn name(&self) -> &'static str {
        "ZonedPlan"
    }

    fn optimize(&self) -> VortexResult<PlanRef> {
        let data = self.data.optimize()?;
        let zones = self.zones.optimize()?;
        Ok(Arc::new(self.with_children(data, zones)))
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count(&self) -> u64 {
        self.layout.row_count()
    }

    fn child_count(&self) -> usize {
        2
    }

    fn child(&self, index: usize) -> VortexResult<Option<PlanRef>> {
        match index {
            DATA_CHILD_INDEX => Ok(Some(Arc::clone(&self.data))),
            ZONES_CHILD_INDEX => Ok(Some(Arc::clone(&self.zones))),
            _ => vortex_bail!("Zoned plan has no child {index}"),
        }
    }

    fn child_name(&self, index: usize) -> Cow<'_, str> {
        match index {
            DATA_CHILD_INDEX => Cow::Borrowed("data"),
            ZONES_CHILD_INDEX => Cow::Borrowed("zones"),
            _ => Cow::Owned(format!("child[{index}]")),
        }
    }
}
