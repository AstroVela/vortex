// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::StructFields;
use vortex_array::expr::col;
use vortex_array::expr::get_item;
use vortex_array::expr::immediate_scope_access;
use vortex_array::expr::make_free_field_annotator;
use vortex_array::expr::root;
use vortex_array::expr::transform::partition;
use vortex_array::expr::transform::replace;
use vortex_array::expr::transform::replace_root_fields;
use vortex_array::scalar_fn::fns::pack::Pack;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::layouts::struct_::StructLayout;
use crate::plan::ExpressionPlan;
use crate::plan::LazyPlanChildren;
use crate::plan::Plan;
use crate::plan::PlanRef;
use crate::plan::new_plan;
use crate::plan::optimizer::PlanParentReduceRule;

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

    fn struct_fields(&self) -> &StructFields {
        self.dtype
            .as_struct_fields_opt()
            .vortex_expect("StructPlan dtype must be a struct")
    }

    fn with_children(&self, children: LazyPlanChildren) -> VortexResult<Self> {
        let fields = self.struct_fields();
        let dtypes = (0..fields.nfields())
            .map(|index| {
                children
                    .get(index)?
                    .map(|child| child.dtype().clone())
                    .ok_or_else(|| vortex_err!("Struct field {index} has no plan"))
            })
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(Self {
            layout: self.layout.clone(),
            dtype: DType::Struct(
                StructFields::new(fields.names().clone(), dtypes),
                self.dtype.nullability(),
            ),
            children,
        })
    }

    fn with_pruned_fields(&self, fields: Vec<(FieldName, PlanRef)>) -> VortexResult<Self> {
        vortex_ensure!(
            !self.dtype.is_nullable(),
            "Cannot prune fields from a nullable StructPlan"
        );
        let dtype = DType::Struct(
            StructFields::from_iter(
                fields
                    .iter()
                    .map(|(name, plan)| (name.clone(), plan.dtype().clone())),
            ),
            self.dtype.nullability(),
        );
        let field_plans: Arc<[PlanRef]> = fields
            .into_iter()
            .map(|(_, plan)| plan)
            .collect::<Vec<_>>()
            .into();
        let field_count = field_plans.len();
        let children = LazyPlanChildren::new(field_count + 1, move |index| {
            Ok(field_plans.get(index).cloned())
        });
        Ok(Self {
            layout: self.layout.clone(),
            dtype,
            children,
        })
    }
}

impl Plan for StructPlan {
    fn name(&self) -> &'static str {
        "StructPlan"
    }

    fn optimize(&self) -> VortexResult<PlanRef> {
        let children = self.children.try_map(|_, child| child.optimize())?;
        Ok(Arc::new(self.with_children(children)?))
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
        if let Some(name) = self.struct_fields().field_name(index) {
            return Cow::Borrowed(name.as_ref());
        }
        if index == self.struct_fields().nfields() {
            return Cow::Borrowed("validity");
        }
        Cow::Owned(format!("child[{index}]"))
    }
}

/// Partitions an expression across the fields of a struct plan.
#[derive(Debug)]
pub(crate) struct ExpressionStructRule;

impl PlanParentReduceRule<StructPlan> for ExpressionStructRule {
    type Parent = ExpressionPlan;

    fn reduce_parent(
        &self,
        child: &StructPlan,
        parent: &ExpressionPlan,
        _child_idx: usize,
    ) -> VortexResult<Option<PlanRef>> {
        if child.dtype.is_nullable() {
            return Ok(None);
        }

        let expression = parent.expression();
        let fields = child.struct_fields();
        let referenced_fields = immediate_scope_access(expression, fields);
        let expanded =
            replace_root_fields(expression.clone(), fields).optimize_recursive(&child.dtype)?;
        let partitioned = partition(
            expanded.clone(),
            &child.dtype,
            make_free_field_annotator(fields),
        )?;
        if partitioned.partition_names.is_empty() {
            let selected_indices = fields
                .names()
                .iter()
                .enumerate()
                .filter_map(|(index, name)| referenced_fields.contains(name).then_some(index))
                .collect::<Vec<_>>();
            if selected_indices.len() == fields.nfields() {
                return Ok(None);
            }

            let pruned_fields = selected_indices
                .into_iter()
                .map(|field_index| {
                    let field_name = fields
                        .field_name(field_index)
                        .ok_or_else(|| vortex_err!("Struct field {field_index} has no name"))?
                        .clone();
                    let field = child
                        .children
                        .get(field_index)?
                        .ok_or_else(|| vortex_err!("Struct field '{field_name}' has no plan"))?;
                    Ok((field_name, field))
                })
                .collect::<VortexResult<Vec<_>>>()?;
            let rewritten: PlanRef = Arc::new(child.with_pruned_fields(pruned_fields)?);
            return Ok(Some(Arc::new(ExpressionPlan::try_new(
                expression.clone(),
                rewritten,
            )?)));
        }

        if partitioned.partition_names.len() == 1 {
            let field_name = partitioned
                .partition_names
                .get(0)
                .ok_or_else(|| vortex_err!("Struct expression partition has no field"))?;
            let field_index = fields.find(field_name).ok_or_else(|| {
                vortex_err!("Struct expression references unknown field '{field_name}'")
            })?;
            let field = child
                .children
                .get(field_index)?
                .ok_or_else(|| vortex_err!("Struct field '{field_name}' has no plan"))?;
            let lowered = replace(expanded, &col(field_name.clone()), root());

            return Ok(Some(ExpressionPlan::try_new(lowered, field)?.optimize()?));
        }

        let mut residual = partitioned.root;
        let mut field_expressions = vec![None; fields.nfields()];
        for index in 0..partitioned.partitions.len() {
            let field_name = &partitioned.partition_names[index];
            let partition = &partitioned.partitions[index];
            let field_index = fields.find(field_name).ok_or_else(|| {
                vortex_err!("Struct expression references unknown field '{field_name}'")
            })?;

            let lowered = if let Some(pack) = partition.as_opt::<Pack>()
                && partition.children().len() == 1
            {
                let value_name = pack
                    .names
                    .get(0)
                    .ok_or_else(|| vortex_err!("Struct expression partition pack is empty"))?;
                residual = replace(
                    residual,
                    &get_item(value_name.clone(), get_item(field_name.clone(), root())),
                    get_item(field_name.clone(), root()),
                );
                replace(partition.child(0).clone(), &col(field_name.clone()), root())
            } else {
                replace(partition.clone(), &col(field_name.clone()), root())
            };
            field_expressions[field_index] = Some(lowered);
        }

        let mut pruned_fields = Vec::with_capacity(partitioned.partition_names.len());
        for (field_index, expression) in field_expressions.into_iter().enumerate() {
            let Some(expression) = expression else {
                continue;
            };
            let field_name = fields
                .field_name(field_index)
                .ok_or_else(|| vortex_err!("Struct field {field_index} has no name"))?
                .clone();
            let field = child
                .children
                .get(field_index)?
                .ok_or_else(|| vortex_err!("Struct field '{field_name}' has no plan"))?;
            let field = ExpressionPlan::try_new(expression, field)?.optimize()?;
            pruned_fields.push((field_name, field));
        }
        let rewritten: PlanRef = Arc::new(child.with_pruned_fields(pruned_fields)?);

        Ok(Some(Arc::new(ExpressionPlan::try_new(
            residual, rewritten,
        )?)))
    }
}
