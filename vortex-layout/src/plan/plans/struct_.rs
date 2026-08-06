// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::future;
use futures::try_join;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::arrays::StructArray;
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
use vortex_array::validity::Validity;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use super::expression::rewrite_partition_root;
use crate::layouts::struct_::StructLayout;
use crate::plan::ExpressionPlan;
use crate::plan::LazyPlanChildren;
use crate::plan::Plan;
use crate::plan::PlanArrayFuture;
use crate::plan::PlanExecutionContext;
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

    fn execute(
        &self,
        ctx: &PlanExecutionContext,
        row_range: &Range<u64>,
        mask: MaskFuture,
    ) -> VortexResult<PlanArrayFuture> {
        vortex_ensure!(
            row_range.start <= row_range.end && row_range.end <= self.row_count(),
            "Struct plan row range {:?} is outside 0..{}",
            row_range,
            self.row_count()
        );
        vortex_ensure!(
            mask.len() == usize::try_from(row_range.end - row_range.start)?,
            "Struct plan mask length mismatch"
        );
        let struct_fields = self.struct_fields();
        let names = struct_fields.names().clone();
        let field_count = struct_fields.nfields();
        let mut field_futures = Vec::with_capacity(field_count);
        for index in 0..field_count {
            let child = self
                .children
                .get(index)?
                .ok_or_else(|| vortex_err!("Struct field {index} has no plan"))?;
            field_futures.push(child.execute(ctx, row_range, mask.clone())?);
        }
        let validity = self
            .children
            .get(field_count)?
            .map(|validity| validity.execute(ctx, row_range, mask.clone()))
            .transpose()?;
        let output_mask = mask;

        Ok(async move {
            let fields = future::try_join_all(field_futures);
            let validity = async move {
                match validity {
                    Some(validity) => validity.await.map(Some),
                    None => Ok(None),
                }
            };
            let (fields, validity) = try_join!(fields, validity)?;
            let len = output_mask.await?.true_count();
            let validity = validity.map_or(Validity::NonNullable, Validity::Array);
            Ok(StructArray::try_new(names, fields, len, validity)?.into_array())
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
        let expanded = expand_struct_root(expression.clone(), fields)?;
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

            return Ok(Some(ExpressionPlan::new_ref(lowered, field)));
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
            let field = ExpressionPlan::new_ref(expression, field);
            pruned_fields.push((field_name, field));
        }
        let rewritten: PlanRef = Arc::new(child.with_pruned_fields(pruned_fields)?);
        let residual = rewrite_partition_root(residual, rewritten.dtype().clone(), &collapsed)?;

        Ok(Some(ExpressionPlan::new_ref(residual, rewritten)))
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
    fields: &StructFields,
) -> VortexResult<BoundExpression> {
    Ok(expression
        .transform_down(|node| {
            if node.is_root() {
                return Ok(Transformed {
                    value: expanded_struct_root(node.dtype(), fields)?,
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

            if scalar_fn.is::<GetItem>() {
                return Ok(Transformed {
                    value: node,
                    changed: false,
                    order: TraversalOrder::Skip,
                });
            }

            if let Some(selection) = scalar_fn.as_opt::<Select>() {
                let names = selection.normalize_to_included_fields(fields.names())?;
                let root = node.children()[0].clone();
                let children = names
                    .iter()
                    .map(|name| {
                        BoundExpression::try_new(GetItem.bind(name.clone()), [root.clone()])
                    })
                    .collect::<VortexResult<Vec<_>>>()?;
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_array::expr::get_item;
    use vortex_array::expr::root;
    use vortex_error::VortexResult;
    use vortex_session::registry::ReadContext;

    use super::StructPlan;
    use crate::LayoutRef;
    use crate::layouts::flat::FlatLayout;
    use crate::layouts::struct_::StructLayout;
    use crate::plan::ExpressionPlan;
    use crate::plan::LazyPlanChildren;
    use crate::plan::Plan;
    use crate::plan::PlanRef;
    use crate::plan::RowIdxPlan;
    use crate::segments::SegmentId;

    struct CountingPlan {
        dtype: DType,
        optimizations: Arc<AtomicUsize>,
    }

    impl CountingPlan {
        fn new_ref(dtype: DType, optimizations: Arc<AtomicUsize>) -> PlanRef {
            Arc::new(Self {
                dtype,
                optimizations,
            })
        }
    }

    impl Plan for CountingPlan {
        fn optimize(&self) -> VortexResult<PlanRef> {
            self.optimizations.fetch_add(1, Ordering::Relaxed);
            Ok(Self::new_ref(
                self.dtype.clone(),
                Arc::clone(&self.optimizations),
            ))
        }

        fn dtype(&self) -> &DType {
            &self.dtype
        }

        fn row_count(&self) -> u64 {
            1
        }
    }

    fn flat(dtype: DType, segment_id: u32) -> LayoutRef {
        FlatLayout::new(1, dtype, SegmentId::from(segment_id), ReadContext::new([])).into_layout()
    }

    #[test]
    fn expression_optimizes_only_referenced_struct_fields() -> VortexResult<()> {
        let field_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let struct_dtype = DType::Struct(
            StructFields::from_iter([("a", field_dtype.clone()), ("b", field_dtype.clone())]),
            Nullability::NonNullable,
        );
        let layout = StructLayout::new(
            1,
            struct_dtype.clone(),
            vec![flat(field_dtype.clone(), 0), flat(field_dtype.clone(), 1)],
        );
        let a_optimizations = Arc::new(AtomicUsize::new(0));
        let b_optimizations = Arc::new(AtomicUsize::new(0));
        let children: Arc<[Option<PlanRef>]> = [
            Some(CountingPlan::new_ref(
                field_dtype.clone(),
                Arc::clone(&a_optimizations),
            )),
            Some(CountingPlan::new_ref(
                field_dtype,
                Arc::clone(&b_optimizations),
            )),
            None,
        ]
        .into();
        let child_count = children.len();
        let struct_plan: PlanRef = Arc::new(StructPlan {
            layout,
            dtype: struct_dtype,
            children: LazyPlanChildren::new(child_count, move |index| {
                Ok(children.get(index).cloned().flatten())
            }),
        });
        let plan = RowIdxPlan::new_ref(0, struct_plan);

        let expression = get_item("a", root())
            .optimize_recursive(plan.dtype())?
            .bind(plan.dtype())?;
        ExpressionPlan::new(expression, plan).optimize()?;

        assert_eq!(a_optimizations.load(Ordering::Relaxed), 1);
        assert_eq!(b_optimizations.load(Ordering::Relaxed), 0);
        Ok(())
    }
}
