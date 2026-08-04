// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::transform::partition_bound;
use vortex_array::expr::traversal::NodeExt;
use vortex_array::expr::traversal::Transformed;
use vortex_array::expr::traversal::TraversalOrder;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::scalar_fn::fns::get_item::GetItem;
use vortex_array::scalar_fn::fns::pack::Pack;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::layouts::row_idx::RowIdx;
use crate::plan::ExpressionPlan;
use crate::plan::Plan;
use crate::plan::PlanRef;

/// A physical plan that adds row-index expression support to its child.
pub struct RowIdxPlan {
    row_offset: u64,
    child: PlanRef,
}

impl RowIdxPlan {
    /// Creates a shared row-index plan with `row_offset` applied to its child domain.
    pub fn new_ref(row_offset: u64, child: PlanRef) -> PlanRef {
        Arc::new(Self { row_offset, child })
    }
}

impl Plan for RowIdxPlan {
    fn name(&self) -> &'static str {
        "RowIdxPlan"
    }

    fn optimize(&self) -> VortexResult<PlanRef> {
        Ok(Self::new_ref(self.row_offset, self.child.optimize()?))
    }

    fn optimize_expression(&self, expression: &BoundExpression) -> VortexResult<Option<PlanRef>> {
        let partitioned = partition_bound(expression.clone(), |node| {
            if node
                .as_scalar()
                .is_some_and(|scalar_fn| scalar_fn.is::<RowIdx>())
            {
                vec![RowIdxExpressionPartition::RowIdx]
            } else if node.is_root() {
                vec![RowIdxExpressionPartition::Child]
            } else {
                vec![]
            }
        })?;

        if partitioned.partition_annotations.len() == 1 {
            return match partitioned.partition_annotations[0] {
                RowIdxExpressionPartition::RowIdx => {
                    let expression = replace_row_idx(expression.clone())?;
                    let values = RowIdxValuesPlan::new_ref(self.row_offset, self.row_count());
                    Ok(Some(ExpressionPlan::new(expression, values).optimize()?))
                }
                RowIdxExpressionPartition::Child => Ok(Some(
                    ExpressionPlan::new(expression.clone(), Arc::clone(&self.child)).optimize()?,
                )),
            };
        }

        if partitioned.partition_annotations.len() != 2 {
            return Ok(None);
        }
        let Some(row_idx_index) = partitioned
            .partition_annotations
            .iter()
            .position(|partition| *partition == RowIdxExpressionPartition::RowIdx)
        else {
            return Ok(None);
        };
        let Some(child_index) = partitioned
            .partition_annotations
            .iter()
            .position(|partition| *partition == RowIdxExpressionPartition::Child)
        else {
            return Ok(None);
        };

        let row_idx_partition = &partitioned.partitions[row_idx_index];
        let child_partition = &partitioned.partitions[child_index];
        let (Some(row_idx_pack), Some(child_pack)) = (
            row_idx_partition
                .as_scalar()
                .and_then(|scalar_fn| scalar_fn.as_opt::<Pack>()),
            child_partition
                .as_scalar()
                .and_then(|scalar_fn| scalar_fn.as_opt::<Pack>()),
        ) else {
            return Ok(None);
        };
        if row_idx_partition.children().len() != 1 || child_partition.children().len() != 1 {
            return Ok(None);
        }
        let (Some(row_idx_value_name), Some(child_value_name)) =
            (row_idx_pack.names.get(0), child_pack.names.get(0))
        else {
            return Ok(None);
        };
        let row_idx_value_name = row_idx_value_name.clone();
        let child_value_name = child_value_name.clone();
        let row_idx_partition_name = partitioned.partition_names[row_idx_index].clone();
        let child_partition_name = partitioned.partition_names[child_index].clone();

        let row_idx_expression = replace_row_idx(row_idx_partition.children()[0].clone())?;
        let child_expression = child_partition.children()[0].clone();
        let row_idx_plan = ExpressionPlan::new(
            row_idx_expression,
            RowIdxValuesPlan::new_ref(self.row_offset, self.row_count()),
        )
        .optimize()?;
        let child_plan =
            ExpressionPlan::new(child_expression, Arc::clone(&self.child)).optimize()?;
        let partitions = RowIdxPartitionPlan::try_new(row_idx_plan, child_plan)?;
        let residual = collapse_partition_root(
            partitioned.root,
            partitions.dtype().clone(),
            [
                (row_idx_partition_name, row_idx_value_name),
                (child_partition_name, child_value_name),
            ],
        )?;

        Ok(Some(ExpressionPlan::new(residual, partitions).optimize()?))
    }

    fn dtype(&self) -> &DType {
        self.child.dtype()
    }

    fn row_count(&self) -> u64 {
        self.child.row_count()
    }

    fn child_count(&self) -> usize {
        1
    }

    fn child(&self, index: usize) -> VortexResult<Option<PlanRef>> {
        if index != 0 {
            vortex_bail!("Row-index plan has no child {index}")
        }
        Ok(Some(Arc::clone(&self.child)))
    }

    fn child_name(&self, index: usize) -> Cow<'_, str> {
        if index == 0 {
            Cow::Borrowed("child")
        } else {
            Cow::Owned(format!("child[{index}]"))
        }
    }
}

fn replace_row_idx(expression: BoundExpression) -> VortexResult<BoundExpression> {
    Ok(expression
        .transform_down(|node| {
            if node
                .as_scalar()
                .is_some_and(|scalar_fn| scalar_fn.is::<RowIdx>())
            {
                Ok(Transformed {
                    value: BoundExpression::new_root(row_idx_dtype()),
                    changed: true,
                    order: TraversalOrder::Skip,
                })
            } else {
                Ok(Transformed::no(node))
            }
        })?
        .into_inner())
}

fn row_idx_dtype() -> DType {
    DType::Primitive(PType::U64, Nullability::NonNullable)
}

fn collapse_partition_root(
    expression: BoundExpression,
    root_dtype: DType,
    mappings: [(FieldName, FieldName); 2],
) -> VortexResult<BoundExpression> {
    Ok(expression
        .transform_down(|node| {
            if let Some(value_name) = node
                .as_scalar()
                .and_then(|scalar_fn| scalar_fn.as_opt::<GetItem>())
            {
                let partition_access = &node.children()[0];
                if let Some(partition_name) = partition_access
                    .as_scalar()
                    .and_then(|scalar_fn| scalar_fn.as_opt::<GetItem>())
                    && partition_access.children()[0].is_root()
                    && mappings.iter().any(|(partition, value)| {
                        partition == partition_name && value == value_name
                    })
                {
                    return Ok(Transformed {
                        value: BoundExpression::try_new(
                            GetItem.bind(partition_name.clone()),
                            [BoundExpression::new_root(root_dtype.clone())],
                        )?,
                        changed: true,
                        order: TraversalOrder::Skip,
                    });
                }
            }

            if node.is_root() {
                Ok(Transformed {
                    value: BoundExpression::new_root(root_dtype.clone()),
                    changed: true,
                    order: TraversalOrder::Skip,
                })
            } else {
                Ok(Transformed::no(node))
            }
        })?
        .into_inner())
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum RowIdxExpressionPartition {
    RowIdx,
    Child,
}

impl RowIdxExpressionPartition {
    fn name(self) -> &'static str {
        match self {
            Self::RowIdx => "row_idx",
            Self::Child => "child",
        }
    }
}

impl Display for RowIdxExpressionPartition {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl From<RowIdxExpressionPartition> for FieldName {
    fn from(partition: RowIdxExpressionPartition) -> Self {
        partition.name().into()
    }
}

/// A plan that generates the global row-index values for a row domain.
pub struct RowIdxValuesPlan {
    row_offset: u64,
    row_count: u64,
    dtype: DType,
}

impl RowIdxValuesPlan {
    /// Creates a shared row-index values plan starting at `row_offset`.
    pub fn new_ref(row_offset: u64, row_count: u64) -> PlanRef {
        Arc::new(Self {
            row_offset,
            row_count,
            dtype: DType::Primitive(PType::U64, Nullability::NonNullable),
        })
    }

    /// Returns the global row index assigned to the first row.
    pub fn row_offset(&self) -> u64 {
        self.row_offset
    }
}

impl Plan for RowIdxValuesPlan {
    fn name(&self) -> &'static str {
        "RowIdxValuesPlan"
    }

    fn optimize(&self) -> VortexResult<PlanRef> {
        Ok(Self::new_ref(self.row_offset, self.row_count))
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count(&self) -> u64 {
        self.row_count
    }
}

/// A plan that combines independently evaluated row-index and data expression partitions.
pub struct RowIdxPartitionPlan {
    row_idx: PlanRef,
    child: PlanRef,
    dtype: DType,
}

impl RowIdxPartitionPlan {
    /// Creates a shared partition plan whose branches have the same row domain.
    pub fn try_new(row_idx: PlanRef, child: PlanRef) -> VortexResult<PlanRef> {
        if row_idx.row_count() != child.row_count() {
            vortex_bail!(
                "Row-index partition row count {} does not match child row count {}",
                row_idx.row_count(),
                child.row_count()
            )
        }
        let dtype = DType::Struct(
            StructFields::from_iter([
                (
                    RowIdxExpressionPartition::RowIdx.name(),
                    row_idx.dtype().clone(),
                ),
                (
                    RowIdxExpressionPartition::Child.name(),
                    child.dtype().clone(),
                ),
            ]),
            Nullability::NonNullable,
        );
        Ok(Arc::new(Self {
            row_idx,
            child,
            dtype,
        }))
    }

    /// Returns the plan that evaluates the row-index expression partition.
    pub fn row_idx_plan(&self) -> &PlanRef {
        &self.row_idx
    }

    /// Returns the plan that evaluates the data-child expression partition.
    pub fn child_plan(&self) -> &PlanRef {
        &self.child
    }
}

impl Plan for RowIdxPartitionPlan {
    fn name(&self) -> &'static str {
        "RowIdxPartitionPlan"
    }

    fn optimize(&self) -> VortexResult<PlanRef> {
        Self::try_new(self.row_idx.optimize()?, self.child.optimize()?)
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count(&self) -> u64 {
        self.child.row_count()
    }

    fn child_count(&self) -> usize {
        2
    }

    fn child(&self, index: usize) -> VortexResult<Option<PlanRef>> {
        match index {
            0 => Ok(Some(Arc::clone(&self.row_idx))),
            1 => Ok(Some(Arc::clone(&self.child))),
            _ => vortex_bail!("Row-index partition plan has no child {index}"),
        }
    }

    fn child_name(&self, index: usize) -> Cow<'_, str> {
        match index {
            0 => Cow::Borrowed(RowIdxExpressionPartition::RowIdx.name()),
            1 => Cow::Borrowed(RowIdxExpressionPartition::Child.name()),
            _ => Cow::Owned(format!("child[{index}]")),
        }
    }
}
