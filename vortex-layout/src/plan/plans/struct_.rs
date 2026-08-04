// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_array::expr::col;
use vortex_array::expr::make_free_field_annotator;
use vortex_array::expr::root;
use vortex_array::expr::transform::partition;
use vortex_array::expr::transform::replace;
use vortex_array::expr::transform::replace_root_fields;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::layouts::struct_::StructLayout;
use crate::plan::ExpressionPlan;
use crate::plan::LazyPlanChildren;
use crate::plan::Plan;
use crate::plan::PlanRef;
use crate::plan::new_plan;

/// A physical struct plan with children ordered as `[field(0), ..., field(n - 1), validity?]`.
pub struct StructPlan {
    layout: StructLayout,
    dtype: DType,
    children: LazyPlanChildren,
}

impl StructPlan {
    pub(crate) fn new(layout: &StructLayout) -> Self {
        // Struct layout slot 0 is validity and field i is slot i + 1. The plan puts validity last
        // so field indices are identical to their plan-child indices.
        let child_layout = layout.clone();
        let field_count = layout.struct_fields().nfields();
        let children = LazyPlanChildren::new(field_count + 1, move |index| {
            if index < field_count {
                let field = child_layout
                    .slot(index + 1)?
                    .ok_or_else(|| vortex_error::vortex_err!("Struct field {index} is absent"))?;
                return Ok(Some(new_plan(&field)?));
            }
            child_layout
                .slot(0)?
                .map(|validity| new_plan(&validity))
                .transpose()
        });
        Self {
            layout: layout.clone(),
            dtype: layout.dtype().clone(),
            children,
        }
    }

    fn with_children(&self, children: LazyPlanChildren) -> Self {
        Self {
            layout: self.layout.clone(),
            dtype: self.dtype.clone(),
            children,
        }
    }
}

impl Plan for StructPlan {
    fn name(&self) -> &'static str {
        "StructPlan"
    }

    fn optimize(&self) -> VortexResult<PlanRef> {
        let children = self.children.map(|_, child| child.optimize());
        Ok(Arc::new(self.with_children(children)))
    }

    fn optimize_expression(&self, expression: &Expression) -> VortexResult<Option<PlanRef>> {
        if self.dtype.is_nullable() {
            return Ok(None);
        }

        let fields = self.layout.struct_fields();
        let expanded =
            replace_root_fields(expression.clone(), fields).optimize_recursive(&self.dtype)?;
        let partitioned = partition(
            expanded.clone(),
            &self.dtype,
            make_free_field_annotator(fields),
        )?;
        if partitioned.partition_names.len() != 1 {
            return Ok(None);
        }

        let field_name = partitioned
            .partition_names
            .get(0)
            .ok_or_else(|| vortex_err!("Struct expression partition has no field"))?;
        let field_index = fields.find(field_name).ok_or_else(|| {
            vortex_err!("Struct expression references unknown field '{field_name}'")
        })?;
        let field = self
            .children
            .get(field_index)?
            .ok_or_else(|| vortex_err!("Struct field '{field_name}' has no plan"))?;
        let lowered = replace(expanded, &col(field_name.clone()), root());

        Ok(Some(ExpressionPlan::try_new(lowered, field)?.optimize()?))
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count(&self) -> u64 {
        self.layout.row_count()
    }

    fn child_count(&self) -> usize {
        self.children.len()
    }

    fn child(&self, index: usize) -> VortexResult<Option<PlanRef>> {
        self.children.get(index)
    }

    fn child_name(&self, index: usize) -> Cow<'_, str> {
        if let Some(name) = self.layout.struct_fields().field_name(index) {
            return Cow::Borrowed(name.as_ref());
        }
        if index == self.layout.struct_fields().nfields() {
            return Cow::Borrowed("validity");
        }
        Cow::Owned(format!("child[{index}]"))
    }
}
