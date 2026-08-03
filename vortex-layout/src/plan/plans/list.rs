// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use crate::LayoutReaderContext;
use crate::LayoutReaderRef;
use crate::layouts::list::ELEMENTS_CHILD_INDEX;
use crate::layouts::list::ListLayout;
use crate::layouts::list::OFFSETS_CHILD_INDEX;
use crate::layouts::list::VALIDITY_CHILD_INDEX;
use crate::layouts::list::reader::ListReader;
use crate::plan::Plan;
use crate::plan::PlanRef;
use crate::segments::SegmentSource;

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
        let elements = layout
            .slot(ELEMENTS_CHILD_INDEX)?
            .ok_or_else(|| vortex_error::vortex_err!("List elements child is absent"))?
            .new_plan()?;
        let offsets = layout
            .slot(OFFSETS_CHILD_INDEX)?
            .ok_or_else(|| vortex_error::vortex_err!("List offsets child is absent"))?
            .new_plan()?;
        let validity = layout
            .slot(VALIDITY_CHILD_INDEX)?
            .map(|validity| validity.new_plan())
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
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

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

    fn new_reader(
        &self,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        let elements = self.elements.new_reader(
            format!("{name}.elements").into(),
            Arc::clone(&segment_source),
            session,
            ctx,
        )?;
        let offsets = self.offsets.new_reader(
            format!("{name}.offsets").into(),
            Arc::clone(&segment_source),
            session,
            ctx,
        )?;
        let validity = self
            .validity
            .as_ref()
            .map(|validity| {
                validity.new_reader(
                    format!("{name}.validity").into(),
                    Arc::clone(&segment_source),
                    session,
                    ctx,
                )
            })
            .transpose()?;
        Ok(Arc::new(ListReader::try_new_with_readers(
            self.layout.clone(),
            name,
            session.clone(),
            elements,
            offsets,
            validity,
        )?))
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
