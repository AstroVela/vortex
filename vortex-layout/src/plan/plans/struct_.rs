// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::ExactBoundExpr;
use vortex_array::expr::descendent_bound_annotations;
use vortex_array::expr::make_bound_free_field_annotator;
use vortex_array::expr::transform::partition_bound;
use vortex_array::expr::traversal::NodeExt;
use vortex_array::expr::traversal::Transformed;
use vortex_array::expr::traversal::TraversalOrder;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::scalar_fn::fns::get_item::GetItem;
use vortex_array::scalar_fn::fns::pack::Pack;
use vortex_array::scalar_fn::fns::pack::PackOptions;
use vortex_array::scalar_fn::fns::select::Select;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use super::expression::rewrite_partition_root;
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
        let referenced_fields =
            descendent_bound_annotations(expression, make_bound_free_field_annotator(fields))
                .get(&ExactBoundExpr(expression.clone()))
                .vortex_expect("Bound expression missing free-field annotations")
                .clone();
        let expanded_root = expanded_struct_root(&child.dtype, fields)?;
        let expanded = expand_struct_root(expression.clone(), &expanded_root, fields)?;
        let partitioned =
            partition_bound(expanded.clone(), make_bound_free_field_annotator(fields))?;
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
            return Ok(Some(Arc::new(ExpressionPlan::new(
                expression.clone(),
                rewritten,
            ))));
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
            let lowered = step_into_struct_field(expanded, field_name, field.dtype().clone())?;

            return Ok(Some(ExpressionPlan::new(lowered, field).optimize()?));
        }

        let residual = partitioned.root;
        let mut collapsed = Vec::with_capacity(partitioned.partitions.len());
        let mut field_expressions = vec![None; fields.nfields()];
        for index in 0..partitioned.partitions.len() {
            let field_name = &partitioned.partition_names[index];
            let partition = &partitioned.partitions[index];
            let field_index = fields.find(field_name).ok_or_else(|| {
                vortex_err!("Struct expression references unknown field '{field_name}'")
            })?;

            let field = child
                .children
                .get(field_index)?
                .ok_or_else(|| vortex_err!("Struct field '{field_name}' has no plan"))?;
            let lowered = if let Some(pack) = partition
                .as_scalar()
                .and_then(|scalar_fn| scalar_fn.as_opt::<Pack>())
                && partition.children().len() == 1
            {
                let value_name = pack
                    .names
                    .get(0)
                    .ok_or_else(|| vortex_err!("Struct expression partition pack is empty"))?;
                collapsed.push((field_name.clone(), value_name.clone()));
                partition.children()[0].clone()
            } else {
                partition.clone()
            };
            let lowered = step_into_struct_field(lowered, field_name, field.dtype().clone())?;
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
            let field = ExpressionPlan::new(expression, field).optimize()?;
            pruned_fields.push((field_name, field));
        }
        let rewritten: PlanRef = Arc::new(child.with_pruned_fields(pruned_fields)?);
        let residual = rewrite_partition_root(residual, rewritten.dtype().clone(), &collapsed)?;

        Ok(Some(Arc::new(ExpressionPlan::new(residual, rewritten))))
    }
}

fn expanded_struct_root(
    root_dtype: &DType,
    fields: &StructFields,
) -> VortexResult<BoundExpression> {
    let root = BoundExpression::new_root(root_dtype.clone());
    let children = fields
        .names()
        .iter()
        .map(|name| BoundExpression::try_new(GetItem.bind(name.clone()), [root.clone()]))
        .collect::<VortexResult<Vec<_>>>()?;
    bound_pack(fields.names().clone(), children)
}

fn expand_struct_root(
    expression: BoundExpression,
    expanded_root: &BoundExpression,
    fields: &StructFields,
) -> VortexResult<BoundExpression> {
    Ok(expression
        .transform_down(|node| {
            if node.is_root() {
                return Ok(Transformed {
                    value: expanded_root.clone(),
                    changed: true,
                    order: TraversalOrder::Skip,
                });
            }

            let Some(scalar_fn) = node.as_scalar() else {
                return Ok(Transformed::no(node));
            };
            if !node
                .children()
                .first()
                .is_some_and(BoundExpression::is_root)
            {
                return Ok(Transformed::no(node));
            }

            if let Some(field_name) = scalar_fn.as_opt::<GetItem>() {
                let index = fields.find(field_name).ok_or_else(|| {
                    vortex_err!("Field {field_name} not found while expanding struct root")
                })?;
                return Ok(Transformed {
                    value: expanded_root.children()[index].clone(),
                    changed: true,
                    order: TraversalOrder::Skip,
                });
            }

            if let Some(selection) = scalar_fn.as_opt::<Select>() {
                let names = selection.normalize_to_included_fields(fields.names())?;
                let children = names
                    .iter()
                    .map(|name| {
                        let index = fields.find(name).vortex_expect(
                            "normalized selection fields must exist in the struct root",
                        );
                        expanded_root.children()[index].clone()
                    })
                    .collect();
                return Ok(Transformed {
                    value: bound_pack(names, children)?,
                    changed: true,
                    order: TraversalOrder::Skip,
                });
            }

            Ok(Transformed::no(node))
        })?
        .into_inner())
}

fn step_into_struct_field(
    expression: BoundExpression,
    field_name: &FieldName,
    field_dtype: DType,
) -> VortexResult<BoundExpression> {
    Ok(expression
        .transform_down(|node| {
            let is_field_access = node
                .as_scalar()
                .and_then(|scalar_fn| scalar_fn.as_opt::<GetItem>())
                .is_some_and(|name| name == field_name)
                && node.children()[0].is_root();

            if is_field_access {
                Ok(Transformed {
                    value: BoundExpression::new_root(field_dtype.clone()),
                    changed: true,
                    order: TraversalOrder::Skip,
                })
            } else {
                Ok(Transformed::no(node))
            }
        })?
        .into_inner())
}

fn bound_pack(names: FieldNames, children: Vec<BoundExpression>) -> VortexResult<BoundExpression> {
    BoundExpression::try_new(
        Pack.bind(PackOptions {
            names,
            nullability: Nullability::NonNullable,
        }),
        children,
    )
}
