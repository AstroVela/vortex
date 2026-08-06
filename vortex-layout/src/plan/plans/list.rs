// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::layouts::list::ELEMENTS_CHILD_INDEX;
use crate::layouts::list::ListLayout;
use crate::layouts::list::OFFSETS_CHILD_INDEX;
use crate::layouts::list::VALIDITY_CHILD_INDEX;
use crate::plan::Plan;
use crate::plan::PlanRef;
use crate::plan::new_plan;

/// A physical list plan with children ordered as `[elements, offsets, validity?]`.
pub struct ListPlan {
    layout: ListLayout,
    dtype: DType,
    elements: PlanRef,
    offsets: PlanRef,
    validity: Option<PlanRef>,
}

impl ListPlan {
    pub(crate) fn try_new(layout: &ListLayout) -> VortexResult<Self> {
        let elements = new_plan(
            &layout
                .slot(ELEMENTS_CHILD_INDEX)?
                .ok_or_else(|| vortex_error::vortex_err!("List elements child is absent"))?,
        )?;
        let offsets = new_plan(
            &layout
                .slot(OFFSETS_CHILD_INDEX)?
                .ok_or_else(|| vortex_error::vortex_err!("List offsets child is absent"))?,
        )?;
        let validity = layout
            .slot(VALIDITY_CHILD_INDEX)?
            .map(|validity| new_plan(&validity))
            .transpose()?;
        Ok(Self {
            layout: layout.clone(),
            dtype: layout.dtype().clone(),
            elements,
            offsets,
            validity,
        })
    }

    fn with_children(
        &self,
        elements: PlanRef,
        offsets: PlanRef,
        validity: Option<PlanRef>,
    ) -> Self {
        Self {
            layout: self.layout.clone(),
            dtype: DType::List(
                Arc::new(elements.dtype().clone()),
                self.layout.dtype().nullability(),
            ),
            elements,
            offsets,
            validity,
        }
    }
}

impl Plan for ListPlan {
    fn name(&self) -> &'static str {
        "ListPlan"
    }

    fn optimize(&self) -> VortexResult<PlanRef> {
        let elements = self.elements.optimize()?;
        let offsets = self.offsets.optimize()?;
        let validity = self
            .validity
            .as_ref()
            .map(|validity| validity.optimize())
            .transpose()?;
        Ok(Arc::new(self.with_children(elements, offsets, validity)))
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count(&self) -> u64 {
        self.layout.row_count()
    }

    fn child_count(&self) -> usize {
        3
    }

    fn child(&self, index: usize) -> VortexResult<Option<PlanRef>> {
        match index {
            0 => Ok(Some(Arc::clone(&self.elements))),
            1 => Ok(Some(Arc::clone(&self.offsets))),
            2 => Ok(self.validity.clone()),
            _ => vortex_bail!("List plan has no child {index}"),
        }
    }

    fn child_name(&self, index: usize) -> Cow<'_, str> {
        match index {
            0 => Cow::Borrowed("elements"),
            1 => Cow::Borrowed("offsets"),
            2 => Cow::Borrowed("validity"),
            _ => Cow::Owned(format!("child[{index}]")),
        }
    }
}
