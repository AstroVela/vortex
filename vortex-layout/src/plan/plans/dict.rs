// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use crate::LayoutReaderContext;
use crate::LayoutReaderRef;
use crate::layouts::dict::DictLayout;
use crate::layouts::dict::reader::DictReader;
use crate::plan::Plan;
use crate::plan::PlanRef;
use crate::segments::SegmentSource;

/// A physical dictionary plan with children ordered as `[codes, values]`.
pub struct DictPlan {
    layout: DictLayout,
    dtype: vortex_array::dtype::DType,
    codes: PlanRef,
    values: PlanRef,
}

impl DictPlan {
    pub(crate) fn try_new(layout: &DictLayout) -> VortexResult<Self> {
        // Dict serialization stores values before codes; the plan order is deliberately codes,
        // values because that is the optimizer-facing logical shape.
        let codes = layout
            .slot(1)?
            .ok_or_else(|| vortex_error::vortex_err!("Dictionary codes child is absent"))?
            .new_plan()?;
        let values = layout
            .slot(0)?
            .ok_or_else(|| vortex_error::vortex_err!("Dictionary values child is absent"))?
            .new_plan()?;
        Ok(Self {
            layout: layout.clone(),
            dtype: values.dtype().clone(),
            codes,
            values,
        })
    }

    fn with_children(&self, codes: PlanRef, values: PlanRef) -> Self {
        Self {
            layout: self.layout.clone(),
            dtype: values.dtype().clone(),
            codes,
            values,
        }
    }
}

impl Plan for DictPlan {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &'static str {
        "DictPlan"
    }

    fn optimize(&self) -> VortexResult<PlanRef> {
        let codes = self.codes.optimize()?;
        let values = self.values.optimize()?;
        Ok(Arc::new(self.with_children(codes, values)))
    }

    fn new_reader(
        &self,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        let codes = self.codes.new_reader(
            format!("{name}.codes").into(),
            Arc::clone(&segment_source),
            session,
            ctx,
        )?;
        let values = self.values.new_reader(
            format!("{name}.values").into(),
            segment_source,
            session,
            ctx,
        )?;
        Ok(Arc::new(DictReader::try_new_with_readers(
            self.layout.clone(),
            name,
            session.clone(),
            codes,
            values,
        )?))
    }

    fn dtype(&self) -> &vortex_array::dtype::DType {
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
            0 => Ok(Some(Arc::clone(&self.codes))),
            1 => Ok(Some(Arc::clone(&self.values))),
            _ => vortex_bail!("Dictionary plan has no child {index}"),
        }
    }

    fn child_name(&self, index: usize) -> Cow<'_, str> {
        match index {
            0 => Cow::Borrowed("codes"),
            1 => Cow::Borrowed("values"),
            _ => Cow::Owned(format!("child[{index}]")),
        }
    }
}
