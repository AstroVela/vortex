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
use vortex_array::expr::Expression;
use vortex_array::expr::get_item;
use vortex_array::expr::root;
use vortex_array::expr::transform::partition;
use vortex_array::expr::transform::replace;
use vortex_array::scalar_fn::fns::pack::Pack;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::layouts::row_idx::RowIdx;
use crate::layouts::row_idx::row_idx;
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

    fn optimize_expression(&self, expression: &Expression) -> VortexResult<Option<PlanRef>> {
        let partitioned = partition(expression.clone(), self.dtype(), |node| {
            if node.is::<RowIdx>() {
                vec![RowIdxExpressionPartition::RowIdx]
            } else if vortex_array::expr::is_root(node) {
                vec![RowIdxExpressionPartition::Child]
            } else {
                vec![]
            }
        })?;

        if partitioned.partition_annotations.len() == 1 {
            return match partitioned.partition_annotations[0] {
                RowIdxExpressionPartition::RowIdx => {
                    let expression = replace(expression.clone(), &row_idx(), root());
                    let values = RowIdxValuesPlan::new_ref(self.row_offset, self.row_count());
                    Ok(Some(
                        ExpressionPlan::try_new(expression, values)?.optimize()?,
                    ))
                }
                RowIdxExpressionPartition::Child => Ok(Some(
                    ExpressionPlan::try_new(expression.clone(), Arc::clone(&self.child))?
                        .optimize()?,
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
            row_idx_partition.as_opt::<Pack>(),
            child_partition.as_opt::<Pack>(),
        ) else {
            return Ok(None);
        };
        // A general pack plan is needed to expose more than one result from either branch.
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
        let row_idx_expression = row_idx_partition.child(0).clone();
        let child_expression = child_partition.child(0).clone();

        let residual = replace(
            partitioned.root,
            &get_item(row_idx_value_name, get_item(row_idx_partition_name, root())),
            get_item(RowIdxExpressionPartition::RowIdx.name(), root()),
        );
        let residual = replace(
            residual,
            &get_item(child_value_name, get_item(child_partition_name, root())),
            get_item(RowIdxExpressionPartition::Child.name(), root()),
        );

        let row_idx_expression = replace(row_idx_expression, &row_idx(), root());
        let row_idx_plan = ExpressionPlan::try_new(
            row_idx_expression,
            RowIdxValuesPlan::new_ref(self.row_offset, self.row_count()),
        )?
        .optimize()?;
        let child_plan =
            ExpressionPlan::try_new(child_expression, Arc::clone(&self.child))?.optimize()?;
        let partitions = RowIdxPartitionPlan::try_new(row_idx_plan, child_plan)?;

        Ok(Some(
            ExpressionPlan::try_new(residual, partitions)?.optimize()?,
        ))
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
