// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::try_join;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ListArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use crate::layouts::list::ELEMENTS_CHILD_INDEX;
use crate::layouts::list::ListLayout;
use crate::layouts::list::OFFSETS_CHILD_INDEX;
use crate::layouts::list::VALIDITY_CHILD_INDEX;
use crate::plan::Plan;
use crate::plan::PlanArrayFuture;
use crate::plan::PlanExecutionContext;
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

    fn execute(
        &self,
        ctx: &PlanExecutionContext,
        row_range: &Range<u64>,
        mask: MaskFuture,
    ) -> VortexResult<PlanArrayFuture> {
        vortex_ensure!(
            row_range.start <= row_range.end && row_range.end <= self.row_count(),
            "List plan row range {:?} is outside 0..{}",
            row_range,
            self.row_count()
        );
        let row_count = usize::try_from(row_range.end - row_range.start)?;
        vortex_ensure!(mask.len() == row_count, "List plan mask length mismatch");

        let offsets_range = row_range.start
            ..row_range
                .end
                .checked_add(1)
                .ok_or_else(|| vortex_error::vortex_err!("List offsets range overflow"))?;
        let offsets = self.offsets.execute(
            ctx,
            &offsets_range,
            MaskFuture::new_true(row_count.saturating_add(1)),
        )?;
        let validity = self
            .validity
            .as_ref()
            .map(|validity| validity.execute(ctx, row_range, MaskFuture::new_true(row_count)))
            .transpose()?;
        let elements = Arc::clone(&self.elements);
        let execution = ctx.clone();
        let dtype = self.dtype.clone();
        let nullability = dtype.nullability();

        Ok(async move {
            let (offsets, mask) = try_join!(offsets, mask)?;
            if mask.all_false() {
                return Ok(Canonical::empty(&dtype).into_array());
            }

            let elements_range = elements_range_from_offsets(&offsets, execution.session())?;
            let elements_count = usize::try_from(elements_range.end - elements_range.start)?;
            let elements = elements
                .execute(
                    &execution,
                    &elements_range,
                    MaskFuture::new_true(elements_count),
                )?
                .await?;
            let validity = match validity {
                Some(validity) => Some(validity.await?),
                None => None,
            };
            let offsets = rebase_offsets(offsets, elements_range.start)?;
            // SAFETY: ListLayout validation guarantees compatible elements and monotonically
            // increasing offsets. Rebasing preserves the represented list lengths.
            let list = unsafe {
                ListArray::new_unchecked(elements, offsets, create_validity(validity, nullability))
            }
            .into_array();
            if mask.all_true() {
                Ok(list)
            } else {
                list.filter(mask)
            }
        }
        .boxed())
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

fn elements_range_from_offsets(
    offsets: &vortex_array::ArrayRef,
    session: &vortex_session::VortexSession,
) -> VortexResult<Range<u64>> {
    if offsets.is_empty() {
        return Ok(0..0);
    }
    let mut ctx = session.create_execution_ctx();
    let start = offsets
        .execute_scalar(0, &mut ctx)?
        .as_primitive()
        .as_::<u64>()
        .vortex_expect("offset value must fit in u64");
    let end = offsets
        .execute_scalar(offsets.len() - 1, &mut ctx)?
        .as_primitive()
        .as_::<u64>()
        .vortex_expect("offset value must fit in u64");
    Ok(start..end)
}

fn rebase_offsets(
    offsets: vortex_array::ArrayRef,
    first: u64,
) -> VortexResult<vortex_array::ArrayRef> {
    if first == 0 {
        return Ok(offsets);
    }
    let constant = ConstantArray::new(first, offsets.len())
        .into_array()
        .cast(offsets.dtype().clone())?;
    offsets.binary(constant, Operator::Sub)
}

fn create_validity(
    validity: Option<vortex_array::ArrayRef>,
    nullability: vortex_array::dtype::Nullability,
) -> Validity {
    match validity {
        Some(validity) => Validity::Array(validity),
        None if nullability.is_nullable() => Validity::AllValid,
        None => Validity::NonNullable,
    }
}
