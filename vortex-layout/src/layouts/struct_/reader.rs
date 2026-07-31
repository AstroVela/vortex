// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;
use std::sync::OnceLock;

use vortex_array::MaskFuture;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::Nullability;
use vortex_array::expr::ExactExpr;
use vortex_array::expr::Expression;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_utils::aliases::dash_map::DashMap;
use vortex_utils::aliases::hash_map::HashMap;

use crate::ArrayFuture;
use crate::LayoutReader;
use crate::LayoutReaderRef;
use crate::LazyReaderChildren;
use crate::RowSplits;
use crate::SplitRange;
use crate::layouts::partitioned::PartitionedExprEval;
use crate::layouts::struct_::StructLayout;
use crate::layouts::struct_::partition::StructPartitioned;
use crate::layouts::struct_::partition::StructSlot;
use crate::layouts::struct_::partition::partition_struct_expr;
use crate::layouts::struct_::partition::pruning_partition;
use crate::segments::SegmentSource;

pub struct StructReader {
    layout: StructLayout,
    name: Arc<str>,
    lazy_children: LazyReaderChildren,
    session: VortexSession,

    field_lookup: Option<HashMap<FieldName, usize>>,
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

        // NOTE: This number is arbitrary and likely depends on the longest common prefix of field names
        let field_lookup = (struct_dt.nfields() > 80).then(|| {
            struct_dt
                .names()
                .iter()
                .enumerate()
                .map(|(i, n)| (n.clone(), i))
                .collect()
        });

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

        // This is where we need to do some complex things with the scan in order to split it into
        // different scans for different fields.
        Ok(Self {
            layout,
            name,
            session,
            lazy_children,
            field_lookup,
            partitioned_expr_cache: Default::default(),
        })
    }

    /// Return the child reader for the field, by index.
    fn field_reader_by_index(&self, idx: usize) -> VortexResult<&LayoutReaderRef> {
        // Field `idx` always occupies slot `idx + 1`; the layout maps that to a dense child index,
        // accounting for the validity slot when the struct is nullable.
        let child_index = self
            .layout
            .slot_to_child(idx + 1)
            .vortex_expect("struct field slot is always present");
        self.lazy_children.get(child_index)
    }

    /// Return the reader for the struct validity, if present
    fn validity(&self) -> VortexResult<Option<&LayoutReaderRef>> {
        self.layout
            .slot_to_child(0)
            .map(|child_index| self.lazy_children.get(child_index))
            .transpose()
    }

    /// Return the child reader backing a partition slot.
    fn slot_reader(&self, slot: &StructSlot) -> VortexResult<&LayoutReaderRef> {
        match slot {
            StructSlot::Validity => self
                .validity()?
                .ok_or_else(|| vortex_err!("Non-nullable struct layout has no validity child")),
            StructSlot::Field(idx) => self.field_reader_by_index(*idx),
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
        let result = partition_struct_expr(&expr, self.dtype(), self.field_lookup.as_ref())?;
        Ok(cell.get_or_init(|| result).clone())
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
                .slot_reader(slot)?
                .pruning_evaluation(row_range, partition, mask)
                .map_err(|err| {
                    err.with_context(format!("While evaluating pruning filter partition {slot}"))
                }),
            StructPartitioned::Multi(partitioned) => {
                // TODO(ngates): if all partitions are boolean, we can use a pruning evaluation.
                //  Otherwise there's not much we can do? Maybe... it's complicated...
                let Some(idx) = pruning_partition(partitioned) else {
                    return Ok(MaskFuture::ready(mask));
                };
                let slot = &partitioned.partition_annotations[idx];
                self.slot_reader(slot)?
                    .pruning_evaluation(row_range, &partitioned.partitions[idx], mask)
                    .map_err(|err| {
                        err.with_context(format!(
                            "While evaluating pruning filter partition {slot}"
                        ))
                    })
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
                .slot_reader(slot)?
                .filter_evaluation(row_range, partition, mask)
                .map_err(|err| {
                    err.with_context(format!("While evaluating filter partition {slot}"))
                }),
            StructPartitioned::Multi(partitioned) => Arc::clone(partitioned).into_mask_future(
                mask,
                |slot, expr, mask| {
                    self.slot_reader(slot)?
                        .filter_evaluation(row_range, expr, mask)
                        .map_err(|err| {
                            err.with_context(format!("While evaluating filter partition {slot}"))
                        })
                },
                |slot, expr, mask| {
                    self.slot_reader(slot)?
                        .projection_evaluation(row_range, expr, mask)
                        .map_err(|err| {
                            err.with_context(format!(
                                "While evaluating projection partition {slot}"
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
        // The struct's validity is just another child, so partitioning already places it wherever
        // the expression needs it — there is nothing to apply after the fact.
        match &self.partition_expr(expr.clone())? {
            StructPartitioned::Single(slot, partition) => self
                .slot_reader(slot)?
                .projection_evaluation(row_range, partition, mask_fut)
                .map_err(|err| {
                    err.with_context(format!("While evaluating projection partition {slot}"))
                }),

            StructPartitioned::Multi(partitioned) => {
                Arc::clone(partitioned).into_array_future(mask_fut, |slot, expr, mask| {
                    self.slot_reader(slot)?
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
mod tests {
    use std::sync::Arc;

    use rstest::fixture;
    use rstest::rstest;
    use vortex_array::ArrayContext;
    use vortex_array::IntoArray;
    use vortex_array::MaskFuture;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::struct_::StructArrayExt;
    use vortex_array::assert_arrays_eq;
    use vortex_array::assert_nth_scalar;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::FieldName;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_array::expr::Expression;
    use vortex_array::expr::col;
    use vortex_array::expr::eq;
    use vortex_array::expr::get_item;
    use vortex_array::expr::gt;
    use vortex_array::expr::is_not_null;
    use vortex_array::expr::is_null;
    use vortex_array::expr::lit;
    use vortex_array::expr::or;
    use vortex_array::expr::pack;
    use vortex_array::expr::root;
    use vortex_array::expr::select;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_io::runtime::single::block_on;
    use vortex_io::session::RuntimeSessionExt;
    use vortex_mask::Mask;

    use crate::LayoutRef;
    use crate::LayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::table::TableStrategy;
    use crate::segments::SegmentSource;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::test::SESSION;
    use crate::test::new_session;

    /// A nullable struct with no fields, so the validity child is its only child.
    #[fixture]
    fn empty_nullable_struct() -> (Arc<dyn SegmentSource>, LayoutRef) {
        let ctx = ArrayContext::empty();

        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();
        let strategy = TableStrategy::new(
            Arc::new(FlatLayoutStrategy::default()),
            Arc::new(FlatLayoutStrategy::default()),
        );
        let segments2 = Arc::<TestSegments>::clone(&segments);
        let layout = block_on(|handle| async move {
            let session = new_session().with_handle(handle);
            strategy
                .write_stream(
                    ctx.into(),
                    segments2,
                    StructArray::try_new(
                        Vec::<FieldName>::new().into(),
                        vec![],
                        3,
                        Validity::Array(BoolArray::from_iter([false, true, true]).into_array()),
                    )
                    .unwrap()
                    .into_array()
                    .to_array_stream()
                    .sequenced(ptr),
                    eof,
                    &session,
                )
                .await
        })
        .unwrap();

        (segments, layout)
    }

    #[fixture]
    fn empty_struct() -> (Arc<dyn SegmentSource>, LayoutRef) {
        let ctx = ArrayContext::empty();

        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();
        let strategy = TableStrategy::new(
            Arc::new(FlatLayoutStrategy::default()),
            Arc::new(FlatLayoutStrategy::default()),
        );
        let segments2 = Arc::<TestSegments>::clone(&segments);
        let layout = block_on(|handle| async move {
            let session = new_session().with_handle(handle);
            strategy
                .write_stream(
                    ctx.into(),
                    segments2,
                    StructArray::try_new(
                        Vec::<FieldName>::new().into(),
                        vec![],
                        5,
                        Validity::NonNullable,
                    )
                    .unwrap()
                    .into_array()
                    .to_array_stream()
                    .sequenced(ptr),
                    eof,
                    &session,
                )
                .await
        })
        .unwrap();

        (segments, layout)
    }

    #[fixture]
    /// Create a chunked layout with three chunks of primitive arrays.
    fn struct_layout() -> (Arc<dyn SegmentSource>, LayoutRef) {
        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();
        let strategy = TableStrategy::new(
            Arc::new(FlatLayoutStrategy::default()),
            Arc::new(FlatLayoutStrategy::default()),
        );
        let segments2 = Arc::<TestSegments>::clone(&segments);
        let layout = block_on(|handle| async move {
            let session = new_session().with_handle(handle);
            strategy
                .write_stream(
                    ctx.into(),
                    segments2,
                    StructArray::from_fields(
                        [
                            ("a", buffer![7, 2, 3].into_array()),
                            ("b", buffer![4, 5, 6].into_array()),
                            ("c", buffer![4, 5, 6].into_array()),
                        ]
                        .as_slice(),
                    )
                    .unwrap()
                    .into_array()
                    .to_array_stream()
                    .sequenced(ptr),
                    eof,
                    &session,
                )
                .await
        })
        .unwrap();

        (segments, layout)
    }

    #[fixture]
    /// Create a chunked layout with three chunks of primitive arrays.
    fn null_struct_layout() -> (Arc<dyn SegmentSource>, LayoutRef) {
        let ctx = ArrayContext::empty();

        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();
        let strategy = TableStrategy::new(
            Arc::new(FlatLayoutStrategy::default()),
            Arc::new(FlatLayoutStrategy::default()),
        );
        let segments2 = Arc::<TestSegments>::clone(&segments);
        let layout = block_on(|handle| async move {
            let session = new_session().with_handle(handle);
            strategy
                .write_stream(
                    ctx.into(),
                    segments2,
                    StructArray::try_from_iter_with_validity(
                        [
                            ("a", buffer![7, 2, 3].into_array()),
                            ("b", buffer![4, 5, 6].into_array()),
                            ("c", buffer![4, 5, 6].into_array()),
                        ],
                        Validity::Array(BoolArray::from_iter([false, true, true]).into_array()),
                    )
                    .unwrap()
                    .into_array()
                    .to_array_stream()
                    .sequenced(ptr),
                    eof,
                    &session,
                )
                .await
        })
        .unwrap();

        (segments, layout)
    }

    /// Writes a nested struct layout with the following values:
    ///
    /// |        a         |
    /// |------------------|
    /// |`{"b": {"c": 4 }}`|
    /// |     `NULL`       |
    /// |`{"b": {"c": 6 }}`|
    #[fixture]
    fn nested_struct_layout() -> (Arc<dyn SegmentSource>, LayoutRef) {
        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();
        let strategy = TableStrategy::new(
            Arc::new(FlatLayoutStrategy::default()),
            Arc::new(FlatLayoutStrategy::default()),
        );
        let segments2 = Arc::<TestSegments>::clone(&segments);
        let layout = block_on(|handle| async move {
            let session = new_session().with_handle(handle);
            strategy
                .write_stream(
                    ctx.into(),
                    segments2,
                    StructArray::try_from_iter_with_validity(
                        [(
                            "a",
                            StructArray::try_from_iter_with_validity(
                                [(
                                    "b",
                                    StructArray::try_from_iter_with_validity(
                                        [("c", buffer![4, 5, 6].into_array())],
                                        Validity::NonNullable,
                                    )
                                    .unwrap()
                                    .into_array(),
                                )],
                                Validity::Array(
                                    BoolArray::from_iter([true, false, true]).into_array(),
                                ),
                            )
                            .unwrap()
                            .into_array(),
                        )],
                        Validity::NonNullable,
                    )
                    .unwrap()
                    .into_array()
                    .to_array_stream()
                    .sequenced(ptr),
                    eof,
                    &session,
                )
                .await
        })
        .unwrap();

        (segments, layout)
    }

    #[rstest]
    fn test_struct_layout_or(
        #[from(struct_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) {
        let reader = layout
            .new_reader("".into(), segments, &SESSION, &Default::default())
            .unwrap();
        let filt = or(
            eq(col("a"), lit(7)),
            or(eq(col("b"), lit(5)), eq(col("a"), lit(3))),
        );
        let result = block_on(|_| {
            reader
                .filter_evaluation(&(0..3), &filt, MaskFuture::new_true(3))
                .unwrap()
        })
        .unwrap();
        assert_eq!(result, Mask::from_iter([true, true, true]));
    }

    #[rstest]
    fn test_struct_layout(
        #[from(struct_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) {
        let mut ctx = SESSION.create_execution_ctx();
        let reader = layout
            .new_reader("".into(), segments, &SESSION, &Default::default())
            .unwrap();
        let expr = gt(get_item("a", root()), get_item("b", root()));
        let result = block_on(|_| {
            reader
                .projection_evaluation(&(0..3), &expr, MaskFuture::new_true(3))
                .unwrap()
        })
        .unwrap();
        let expected = BoolArray::from_iter([true, false, false]);
        assert_arrays_eq!(result, expected, &mut ctx);
    }

    #[rstest]
    fn test_struct_layout_row_mask(
        #[from(struct_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) {
        let mut ctx = SESSION.create_execution_ctx();
        let reader = layout
            .new_reader("".into(), segments, &SESSION, &Default::default())
            .unwrap();
        let expr = gt(get_item("a", root()), get_item("b", root()));
        let result = block_on(|_| {
            reader
                .projection_evaluation(
                    &(0..3),
                    &expr,
                    MaskFuture::ready(Mask::from_iter([true, true, false])),
                )
                .unwrap()
        })
        .unwrap();

        let expected = BoolArray::from_iter([true, false]);
        assert_arrays_eq!(result, expected, &mut ctx);
    }

    #[rstest]
    fn test_struct_layout_select(
        #[from(struct_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) {
        let mut ctx = array_session().create_execution_ctx();
        let reader = layout
            .new_reader("".into(), segments, &SESSION, &Default::default())
            .unwrap();
        let expr = pack(
            [("a", get_item("a", root())), ("b", get_item("b", root()))],
            Nullability::NonNullable,
        );
        let result = block_on(|_| {
            reader
                .projection_evaluation(
                    &(0..3),
                    &expr,
                    // Take rows 0 and 1, skip row 2, and anything after that
                    MaskFuture::ready(Mask::from_iter([true, true, false])),
                )
                .unwrap()
        })
        .unwrap();

        assert_eq!(result.len(), 2);

        let expected_a = PrimitiveArray::from_iter([7i32, 2]);
        let result_struct_a = result.clone().execute::<StructArray>(&mut ctx).unwrap();
        assert_arrays_eq!(
            result_struct_a.unmasked_field_by_name("a").unwrap(),
            expected_a,
            &mut ctx
        );

        let expected_b = PrimitiveArray::from_iter([4i32, 5]);
        let result_struct_b = result.execute::<StructArray>(&mut ctx).unwrap();
        assert_arrays_eq!(
            result_struct_b.unmasked_field_by_name("b").unwrap(),
            expected_b,
            &mut ctx
        );
    }

    #[rstest]
    fn test_struct_layout_nulls(
        #[from(null_struct_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) {
        let mut ctx = SESSION.create_execution_ctx();
        // Read the layout source from the top.
        let reader = layout
            .new_reader("".into(), segments, &SESSION, &Default::default())
            .unwrap();
        let expr = get_item("a", root());
        let project = reader
            .projection_evaluation(&(0..3), &expr, MaskFuture::new_true(3))
            .unwrap();

        let result = block_on(move |_| project).unwrap();
        // Result should be the primitive array with a single field.
        assert_eq!(
            result.dtype(),
            &DType::Primitive(PType::I32, Nullability::Nullable)
        );

        // ...and the result is masked with the validity of the parent StructArray
        assert_eq!(
            result
                .execute_scalar(0, &mut array_session().create_execution_ctx())
                .unwrap(),
            Scalar::null(result.dtype().clone()),
        );
        assert_nth_scalar!(result, 1, 2, &mut ctx);
        assert_nth_scalar!(result, 2, 3, &mut ctx);
    }

    #[rstest]
    fn test_struct_layout_nested(
        #[from(nested_struct_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) {
        // Project out the nested struct field.
        // The projection should preserve the nulls of the `b` struct when we select out the
        // child column `c`.
        let expr = select(
            vec![FieldName::from("c")],
            get_item("b", get_item("a", root())),
        );
        let result = block_on(move |handle| {
            let session = new_session().with_handle(handle);
            async move {
                layout
                    .new_reader("".into(), segments, &session, &Default::default())?
                    .projection_evaluation(&(0..3), &expr, MaskFuture::new_true(3))?
                    .await
            }
        })
        .unwrap();

        // The result is a nullable struct (because root.a.b is nullable) with a non-nullable
        // field "c" (because the original field was non-nullable).
        assert_eq!(
            result.dtype(),
            &DType::Struct(
                StructFields::from_iter([(
                    "c",
                    DType::Primitive(PType::I32, Nullability::NonNullable)
                )]),
                Nullability::Nullable,
            )
        );

        // Row 0: struct is valid, field "c" is 4.
        assert_eq!(
            result
                .execute_scalar(0, &mut array_session().create_execution_ctx())
                .unwrap()
                .as_struct()
                .field_by_idx(0)
                .unwrap(),
            Scalar::primitive(4, Nullability::NonNullable)
        );

        // Row 1: struct is null (because root.a.b was null at this row).
        assert!(
            result
                .execute_scalar(1, &mut array_session().create_execution_ctx())
                .unwrap()
                .as_struct()
                .is_null()
        );

        // Row 2: struct is valid, field "c" is 6.
        assert_eq!(
            result
                .execute_scalar(2, &mut array_session().create_execution_ctx())
                .unwrap()
                .as_struct()
                .field_by_idx(0)
                .unwrap(),
            Scalar::primitive(6, Nullability::NonNullable)
        );
    }

    #[rstest]
    fn test_empty_struct(
        #[from(empty_struct)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) {
        let reader = layout
            .new_reader("".into(), segments, &SESSION, &Default::default())
            .unwrap();
        let expr = pack(Vec::<(String, Expression)>::new(), Nullability::Nullable);

        let project = reader
            .projection_evaluation(&(0..5), &expr, MaskFuture::new_true(5))
            .unwrap();

        let result = block_on(move |_| project).unwrap();
        assert!(result.dtype().is_struct());

        assert_eq!(result.len(), 5);
    }

    /// A struct with no fields still has a validity child, which `root()` has to read.
    #[rstest]
    fn test_empty_nullable_struct(
        #[from(empty_nullable_struct)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) {
        let mut ctx = SESSION.create_execution_ctx();
        let reader = layout
            .new_reader("".into(), segments, &SESSION, &Default::default())
            .unwrap();

        let projected = block_on(|_| {
            reader
                .projection_evaluation(&(0..3), &root(), MaskFuture::new_true(3))
                .unwrap()
        })
        .unwrap();
        assert_eq!(projected.len(), 3);
        assert!(projected.dtype().is_struct());
        assert!(projected.dtype().is_nullable());

        let is_null = block_on(|_| {
            reader
                .projection_evaluation(&(0..3), &is_null(root()), MaskFuture::new_true(3))
                .unwrap()
        })
        .unwrap();
        assert_arrays_eq!(
            is_null,
            BoolArray::from_iter([true, false, false]),
            &mut ctx
        );
    }

    /// Regression test for https://github.com/vortex-data/vortex/issues/1907
    ///
    /// A filter over a field of a *nullable* struct must intersect with the struct's own validity.
    /// Row 0 holds `a = 7` but the struct row itself is null, so the predicate must not match it.
    #[rstest]
    fn test_struct_layout_filter_respects_struct_validity(
        #[from(null_struct_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) {
        let reader = layout
            .new_reader("".into(), segments, &SESSION, &Default::default())
            .unwrap();
        let result = block_on(|_| {
            reader
                .filter_evaluation(&(0..3), &eq(col("a"), lit(7)), MaskFuture::new_true(3))
                .unwrap()
        })
        .unwrap();
        assert_eq!(result, Mask::from_iter([false, false, false]));
    }

    /// `is_null` of the struct itself reads only the validity child.
    #[rstest]
    fn test_struct_layout_is_null(
        #[from(null_struct_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) {
        let mut ctx = SESSION.create_execution_ctx();
        let reader = layout
            .new_reader("".into(), segments, &SESSION, &Default::default())
            .unwrap();
        let result = block_on(|_| {
            reader
                .projection_evaluation(&(0..3), &is_null(root()), MaskFuture::new_true(3))
                .unwrap()
        })
        .unwrap();
        assert_arrays_eq!(result, BoolArray::from_iter([true, false, false]), &mut ctx);
    }

    /// `is_not_null` of the struct itself reads only the validity child.
    #[rstest]
    fn test_struct_layout_is_not_null(
        #[from(null_struct_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) {
        let mut ctx = SESSION.create_execution_ctx();
        let reader = layout
            .new_reader("".into(), segments, &SESSION, &Default::default())
            .unwrap();
        let result = block_on(|_| {
            reader
                .projection_evaluation(&(0..3), &is_not_null(root()), MaskFuture::new_true(3))
                .unwrap()
        })
        .unwrap();
        assert_arrays_eq!(result, BoolArray::from_iter([false, true, true]), &mut ctx);
    }

    /// `is_null` of a non-nullable *field* of a nullable struct is driven by the struct validity.
    #[rstest]
    fn test_struct_layout_is_null_of_field(
        #[from(null_struct_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) {
        let mut ctx = SESSION.create_execution_ctx();
        let reader = layout
            .new_reader("".into(), segments, &SESSION, &Default::default())
            .unwrap();
        let result = block_on(|_| {
            reader
                .projection_evaluation(&(0..3), &is_null(col("a")), MaskFuture::new_true(3))
                .unwrap()
        })
        .unwrap();
        assert_arrays_eq!(result, BoolArray::from_iter([true, false, false]), &mut ctx);
    }

    /// A `select` keeps the struct's validity on the struct rather than pushing it into the
    /// selected fields.
    #[rstest]
    fn test_struct_layout_select_keeps_struct_validity(
        #[from(null_struct_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) {
        let reader = layout
            .new_reader("".into(), segments, &SESSION, &Default::default())
            .unwrap();
        let expr = select(vec![FieldName::from("a")], root());
        let result = block_on(|_| {
            reader
                .projection_evaluation(&(0..3), &expr, MaskFuture::new_true(3))
                .unwrap()
        })
        .unwrap();

        // The struct is nullable, but the selected field keeps its own non-nullable dtype.
        assert_eq!(
            result.dtype(),
            &DType::Struct(
                StructFields::from_iter([(
                    "a",
                    DType::Primitive(PType::I32, Nullability::NonNullable)
                )]),
                Nullability::Nullable,
            )
        );

        let mut ctx = array_session().create_execution_ctx();
        assert!(
            result
                .execute_scalar(0, &mut ctx)
                .unwrap()
                .as_struct()
                .is_null()
        );
        assert_eq!(
            result
                .execute_scalar(1, &mut ctx)
                .unwrap()
                .as_struct()
                .field_by_idx(0)
                .unwrap(),
            Scalar::primitive(2, Nullability::NonNullable)
        );
    }

    /// A `pack` of `get_item`s pushes the struct validity into each packed field.
    #[rstest]
    fn test_struct_layout_pack_masks_fields(
        #[from(null_struct_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) {
        let mut ctx = SESSION.create_execution_ctx();
        let reader = layout
            .new_reader("".into(), segments, &SESSION, &Default::default())
            .unwrap();
        let expr = pack([("x", col("a")), ("y", col("b"))], Nullability::NonNullable);
        let result = block_on(|_| {
            reader
                .projection_evaluation(&(0..3), &expr, MaskFuture::new_true(3))
                .unwrap()
        })
        .unwrap();

        let result = result.execute::<StructArray>(&mut ctx).unwrap();
        assert!(!result.dtype().is_nullable());
        assert_arrays_eq!(
            result.unmasked_field_by_name("x").unwrap(),
            PrimitiveArray::from_option_iter([None, Some(2i32), Some(3)]),
            &mut ctx
        );
        assert_arrays_eq!(
            result.unmasked_field_by_name("y").unwrap(),
            PrimitiveArray::from_option_iter([None, Some(5i32), Some(6)]),
            &mut ctx
        );
    }

    /// Regression test for https://github.com/vortex-data/vortex/issues/7808
    ///
    /// A filter expression whose DType is incompatible with the scanned schema
    /// (e.g. comparing a u8 column to an i32 literal) must return an error, not panic.
    #[test]
    fn test_struct_filter_dtype_mismatch_returns_error() {
        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();
        let strategy = TableStrategy::new(
            Arc::new(FlatLayoutStrategy::default()),
            Arc::new(FlatLayoutStrategy::default()),
        );
        let segments2 = Arc::<TestSegments>::clone(&segments);
        let layout = block_on(|handle| async move {
            let session = new_session().with_handle(handle);
            strategy
                .write_stream(
                    ctx.into(),
                    segments2,
                    StructArray::from_fields(
                        [
                            ("age", buffer![7u8, 2, 3].into_array()),
                            ("score", buffer![4u8, 5, 6].into_array()),
                        ]
                        .as_slice(),
                    )
                    .unwrap()
                    .into_array()
                    .to_array_stream()
                    .sequenced(ptr),
                    eof,
                    &session,
                )
                .await
        })
        .unwrap();

        let reader = layout
            .new_reader("".into(), segments, &SESSION, &Default::default())
            .unwrap();

        // DType mismatch: "age" is u8 but literal is i32
        let filt = eq(col("age"), lit(67i32));

        let result = reader.filter_evaluation(&(0..3), &filt, MaskFuture::new_true(3));
        assert!(result.is_err());
        let err = result.err().unwrap().to_string();
        assert!(err.contains("Cannot compare different DTypes"), "{err}");
    }
}
