// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_array::dtype::StructFields;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::layouts::struct_::StructLayout;
use crate::plan::Plan;
use crate::plan::PlanRef;
use crate::plan::new_plan;

/// A physical struct plan with children ordered as `[field(0), ..., field(n - 1), validity?]`.
pub struct StructPlan {
    layout: StructLayout,
    dtype: DType,
    fields: Arc<[PlanRef]>,
    validity: Option<PlanRef>,
}

impl StructPlan {
    pub(crate) fn try_new(layout: &StructLayout) -> VortexResult<Self> {
        // Struct layout slot 0 is validity and field i is slot i + 1. The plan puts validity last
        // so field indices are identical to their plan-child indices.
        let fields =
            (0..layout.struct_fields().nfields())
                .map(|index| {
                    new_plan(&layout.slot(index + 1)?.ok_or_else(|| {
                        vortex_error::vortex_err!("Struct field {index} is absent")
                    })?)
                })
                .collect::<VortexResult<Vec<_>>>()?;
        let validity = layout
            .slot(0)?
            .map(|validity| new_plan(&validity))
            .transpose()?;
        Ok(Self {
            layout: layout.clone(),
            dtype: layout.dtype().clone(),
            fields: fields.into(),
            validity,
        })
    }

    fn with_children(&self, fields: Vec<PlanRef>, validity: Option<PlanRef>) -> Self {
        let struct_fields = StructFields::from_iter(
            self.layout
                .struct_fields()
                .names()
                .iter()
                .zip(fields.iter())
                .map(|(name, field)| (name.clone(), field.dtype().clone())),
        );
        Self {
            layout: self.layout.clone(),
            dtype: DType::Struct(struct_fields, self.layout.dtype().nullability()),
            fields: fields.into(),
            validity,
        }
    }
}

impl Plan for StructPlan {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &'static str {
        "StructPlan"
    }

    fn optimize(&self) -> VortexResult<PlanRef> {
        let fields = self
            .fields
            .iter()
            .map(|field| field.optimize())
            .collect::<VortexResult<Vec<_>>>()?;
        let validity = self
            .validity
            .as_ref()
            .map(|validity| validity.optimize())
            .transpose()?;
        Ok(Arc::new(self.with_children(fields, validity)))
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count(&self) -> u64 {
        self.layout.row_count()
    }

    fn child_count(&self) -> usize {
        self.fields.len() + 1
    }

    fn child(&self, index: usize) -> VortexResult<Option<PlanRef>> {
        if let Some(field) = self.fields.get(index) {
            return Ok(Some(Arc::clone(field)));
        }
        if index == self.fields.len() {
            return Ok(self.validity.clone());
        }
        vortex_bail!("Struct plan has no child {index}")
    }

    fn child_name(&self, index: usize) -> Cow<'_, str> {
        if let Some(name) = self.layout.struct_fields().field_name(index) {
            return Cow::Borrowed(name.as_ref());
        }
        if index == self.fields.len() {
            return Cow::Borrowed("validity");
        }
        Cow::Owned(format!("child[{index}]"))
    }
}
