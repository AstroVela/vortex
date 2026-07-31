// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;
use std::sync::OnceLock;

use vortex_array::MaskFuture;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::dtype::Nullability;
use vortex_array::expr::ExactExpr;
use vortex_array::expr::Expression;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_utils::aliases::dash_map::DashMap;

use crate::ArrayFuture;
use crate::LayoutChildType;
use crate::LayoutReader;
use crate::LayoutReaderRef;
use crate::LazyReaderChildren;
use crate::RowSplits;
use crate::SplitRange;
use crate::layouts::partitioned::PartitionedExprEval;
use crate::layouts::struct_::StructLayout;
use crate::layouts::struct_::partition::StructPartitioned;
use crate::layouts::struct_::partition::StructPartitioner;
use crate::layouts::struct_::partition::StructSlot;
use crate::segments::SegmentSource;

pub struct StructReader {
    layout: StructLayout,
    name: Arc<str>,
    lazy_children: LazyReaderChildren,
    session: VortexSession,

    /// Partitions expressions over the children of this struct layout, including its validity.
    partitioner: StructPartitioner,

    partitioned_expr_cache: DashMap<ExactExpr, Arc<OnceLock<StructPartitioned>>>,
}

impl StructReader {
    pub(super) fn try_new(
        layout: StructLayout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: VortexSession,
        ctx: crate::LayoutReaderContext,
    ) -> VortexResult<Self> {
        let struct_dt = layout.struct_fields();

        let nullable = layout.dtype().is_nullable();
        let extra = nullable as usize;

        let mut dtypes: Vec<DType> = Vec::with_capacity(struct_dt.nfields() + extra);
        let mut names: Vec<Arc<str>> = Vec::with_capacity(struct_dt.nfields() + extra);
        if nullable {
            dtypes.push(DType::Bool(Nullability::NonNullable));
            names.push(Arc::from("validity"));
        }
        dtypes.extend(struct_dt.fields());
        names.extend(struct_dt.names().iter().map(|x| Arc::clone(x.inner())));

        let lazy_children = LazyReaderChildren::new(
            Arc::clone(layout.children()),
            dtypes,
            names,
            Arc::clone(&segment_source),
            session.clone(),
            ctx,
        );

        let partitioner = StructPartitioner::new(layout.dtype())?;

        // This is where we need to do some complex things with the scan in order to split it into
        // different scans for different fields.
        Ok(Self {
            layout,
            name,
            session,
            partitioner,
            lazy_children,
            partitioned_expr_cache: Default::default(),
        })
    }

    /// Return the child reader for the field, by index.
    fn field_reader_by_index(&self, idx: usize) -> VortexResult<&LayoutReaderRef> {
        self.slot_reader(StructSlot::field(idx))
    }

    /// Return the reader for the struct validity, if present
    fn validity(&self) -> VortexResult<Option<&LayoutReaderRef>> {
        self.layout
            .slot_to_child(StructSlot::VALIDITY.index())
            .map(|child_index| self.lazy_children.get(child_index))
            .transpose()
    }

    /// Return the child reader that evaluates the given partition slot.
    fn slot_reader(&self, slot: StructSlot) -> VortexResult<&LayoutReaderRef> {
        // Partition slots are the layout's own slot indices, so this is a direct lookup: the
        // layout maps the slot to a dense child index, accounting for the validity slot.
        let child_index = self.layout.slot_to_child(slot.index()).ok_or_else(|| {
            vortex_err!("Struct layout {} has no child for slot {slot}", self.name())
        })?;
        self.lazy_children.get(child_index)
    }

    /// A human-readable name for a slot, used only in error messages.
    fn slot_label(&self, slot: StructSlot) -> String {
        match self.layout.slot_type(slot.index()) {
            Some(LayoutChildType::Field(name)) => name.to_string(),
            Some(LayoutChildType::Auxiliary(name)) => name.to_string(),
            _ => slot.to_string(),
        }
    }

    /// Utility for partitioning an expression over the children of a struct.
    fn partition_expr(&self, expr: Expression) -> VortexResult<StructPartitioned> {
        let key = ExactExpr(expr.clone());

        // Look up the cell under a shared shard lock; only a miss takes the write lock, and
        // only for as long as it takes to insert an empty cell.
        let cell = match self.partitioned_expr_cache.get(&key) {
            Some(entry) => Arc::clone(entry.value()),
            None => Arc::clone(
                self.partitioned_expr_cache
                    .entry(key)
                    .or_insert_with(|| Arc::new(OnceLock::new()))
                    .value(),
            ),
        };
        // All map guards are dropped here, so partitioning runs outside any shard lock.
        // Concurrent misses may compute redundantly; `get_or_init` keeps a single winner.

        if let Some(value) = cell.get() {
            return Ok(value.clone());
        }
        let result = self.compute_partitioned_expr(expr)?;
        Ok(cell.get_or_init(|| result).clone())
    }

    fn compute_partitioned_expr(&self, expr: Expression) -> VortexResult<StructPartitioned> {
        self.partitioner
            .partition(expr.optimize_recursive(self.dtype())?)
    }
}

impl LayoutReader for StructReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn dtype(&self) -> &DType {
        self.layout.dtype()
    }

    fn row_count(&self) -> u64 {
        self.layout.row_count()
    }

    fn register_splits(
        &self,
        field_mask: &[FieldMask],
        split_range: &SplitRange,
        splits: &mut RowSplits,
    ) -> VortexResult<()> {
        // In the case of an empty struct, we need to register the end split.
        splits.push(split_range.root_row_range().end);

        // Register splits for the validity child, if there is one
        if let Some(validity_ref) = self.validity()? {
            validity_ref.register_splits(field_mask, split_range, splits)?;
        }

        self.layout.matching_fields(field_mask, |mask, idx| {
            self.field_reader_by_index(idx)?
                .register_splits(&[mask], split_range, splits)
        })
    }

    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        // Partition the expression into expressions that can be evaluated over individual children
        match &self.partition_expr(expr.clone())? {
            StructPartitioned::Single(slot, partition) => self
                .slot_reader(*slot)?
                .pruning_evaluation(row_range, partition, mask)
                .map_err(|err| {
                    err.with_context(format!(
                        "While evaluating pruning filter partition {}",
                        self.slot_label(*slot)
                    ))
                }),
            StructPartitioned::Multi(_) => {
                // TODO(ngates): if all partitions are boolean, we can use a pruning evaluation. Otherwise
                //  there's not much we can do? Maybe... it's complicated...
                Ok(MaskFuture::ready(mask))
            }
        }
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        // Partition the expression into expressions that can be evaluated over individual children
        match &self.partition_expr(expr.clone())? {
            StructPartitioned::Single(slot, partition) => self
                .slot_reader(*slot)?
                .filter_evaluation(row_range, partition, mask)
                .map_err(|err| {
                    err.with_context(format!(
                        "While evaluating filter partition {}",
                        self.slot_label(*slot)
                    ))
                }),
            StructPartitioned::Multi(partitioned) => Arc::clone(partitioned).into_mask_future(
                mask,
                |slot, expr, mask| {
                    self.slot_reader(*slot)?
                        .filter_evaluation(row_range, expr, mask)
                        .map_err(|err| {
                            err.with_context(format!(
                                "While evaluating filter partition {}",
                                self.slot_label(*slot)
                            ))
                        })
                },
                |slot, expr, mask| {
                    self.slot_reader(*slot)?
                        .projection_evaluation(row_range, expr, mask)
                        .map_err(|err| {
                            err.with_context(format!(
                                "While evaluating projection partition {}",
                                self.slot_label(*slot)
                            ))
                        })
                },
                self.session.clone(),
            ),
        }
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask_fut: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        // Partition the expression into expressions that can be evaluated over individual
        // children. The struct's own validity is one of those children, so — unlike previously —
        // there is no need to apply it to the result afterwards: the partitioned root expression
        // already places it exactly where the semantics of the expression require it.
        match &self.partition_expr(expr.clone())? {
            StructPartitioned::Single(slot, partition) => self
                .slot_reader(*slot)?
                .projection_evaluation(row_range, partition, mask_fut)
                .map_err(|err| {
                    err.with_context(format!(
                        "While evaluating projection partition {}",
                        self.slot_label(*slot)
                    ))
                }),

            StructPartitioned::Multi(partitioned) => {
                Arc::clone(partitioned).into_array_future(mask_fut, |slot, expr, mask| {
                    self.slot_reader(*slot)?
                        .projection_evaluation(row_range, expr, mask)
                        .map_err(|err| {
                            err.with_context(format!(
                                "While evaluating projection partition {slot}"
                            ))
                        })
                })
            }
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests;
