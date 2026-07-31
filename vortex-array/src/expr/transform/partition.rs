// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::marker::PhantomData;

use itertools::Itertools;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_utils::aliases::hash_map::HashMap;

use crate::dtype::DType;
use crate::dtype::FieldName;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;
use crate::dtype::StructFields;
use crate::expr::Expression;
use crate::expr::analysis::Annotation;
use crate::expr::analysis::AnnotationFn;
use crate::expr::analysis::Annotations;
use crate::expr::analysis::descendent_annotations;
use crate::expr::col;
use crate::expr::get_item;
use crate::expr::is_root;
use crate::expr::pack;
use crate::expr::traversal::NodeExt;
use crate::expr::traversal::NodeRewriter;
use crate::expr::traversal::Transformed;
use crate::expr::traversal::TraversalOrder;
use crate::scalar_fn::fns::get_item::GetItem;

/// Describes how to split expressions over a scope into per-slot partitions.
///
/// A *slot* is whatever a sub-expression can be dispatched to — a field of a struct layout, the
/// row index, the values of a dictionary. Partitioning itself is the same for all of them:
/// annotate each node with the slots it touches, cut out the maximal subtrees that touch exactly
/// one slot, and leave behind a root expression that re-assembles their results.
///
/// Implementations supply the parts that do differ, all of which have identity defaults:
///
/// 1. [`flatten`](Self::flatten) rewrites the expression so that every access to the scope is an
///    access of a slot. A partitioner whose slots are already addressable in the scope needs no
///    such stage.
/// 2. [`slot_name`](Self::slot_name) names a slot within the root expression's scope.
/// 3. [`step_into`](Self::step_into) rewrites a finished partition into the scope of whatever will
///    evaluate it.
pub trait Partitioner {
    /// Identifies one destination that a sub-expression can be dispatched to.
    type Slot: Annotation + Display;

    /// The scope that the flattened expression and its partitions are typed against.
    fn scope(&self) -> &DType;

    /// The slot's field name within the root expression's scope. Must be injective over slots.
    fn slot_name(&self, slot: &Self::Slot) -> FieldName;

    /// Stage 1: rewrite every access of the scope into an access of a slot. Identity by default.
    fn flatten(&self, expr: Expression) -> VortexResult<Expression> {
        Ok(expr)
    }

    /// Stage 3: rewrite a partition into the scope of whatever evaluates it. Identity by default.
    fn step_into(&self, expr: Expression, slot: &Self::Slot) -> VortexResult<Expression> {
        let _ = slot;
        Ok(expr)
    }

    /// Whether a slot holding exactly one sub-expression should skip the enclosing `pack`, so the
    /// root expression references the partition's result directly.
    ///
    /// Consumers that rely on results always being packed must leave this `false`.
    fn unwrap_single_sub_expression(&self) -> bool {
        false
    }
}

/// Partition `expr` over `partitioner`'s slots, annotating each node with the slots that any of
/// its descendents touch.
///
/// Callers needing a different annotation strategy — for example annotating nodes individually
/// rather than propagating up — should compute the annotations themselves and use
/// [`partition_with`].
pub fn partition_annotated<P, F>(
    partitioner: &P,
    expr: Expression,
    annotate: F,
) -> VortexResult<PartitionedExpr<P::Slot>>
where
    P: Partitioner,
    F: Fn(&Expression) -> Vec<P::Slot>,
{
    // Note that the expression is *not* optimized here: where an expression is simplified
    // changes how finely it splits, so that choice belongs to the partitioner's `flatten`.
    let expr = partitioner.flatten(expr)?;
    let annotations = descendent_annotations(&expr, annotate);
    partition_with(partitioner, &expr, &annotations)
}

/// Partition `expr` over `partitioner`'s slots using pre-computed `annotations`.
///
/// `expr` must already be flattened — that is, it must be the output of
/// [`Partitioner::flatten`] — since the annotations borrow it.
pub fn partition_with<P: Partitioner>(
    partitioner: &P,
    expr: &Expression,
    annotations: &Annotations<'_, P::Slot>,
) -> VortexResult<PartitionedExpr<P::Slot>> {
    let mut splitter = ExpressionSplitter {
        partitioner,
        annotations,
        slots: Vec::new(),
        sub_expressions: Vec::new(),
    };
    let root = expr.clone().rewrite(&mut splitter)?.value;

    let mut partitions = Vec::with_capacity(splitter.slots.len());
    let mut partition_names = Vec::with_capacity(splitter.slots.len());
    let mut partition_dtypes = Vec::with_capacity(splitter.slots.len());
    let mut unwrapped = HashMap::new();

    for (slot, exprs) in splitter.slots.iter().zip(splitter.sub_expressions.iter()) {
        let slot_name = partitioner.slot_name(slot);

        // All of a slot's sub-expressions are packed into one expression, so that the slot is
        // read exactly once. A single sub-expression may skip the pack if the partitioner allows.
        let partition = match exprs.as_slice() {
            [only] if partitioner.unwrap_single_sub_expression() => {
                unwrapped.insert(slot_name.clone(), sub_expression_name(&slot_name, 0));
                only.clone()
            }
            exprs => pack(
                exprs
                    .iter()
                    .enumerate()
                    .map(|(idx, expr)| (sub_expression_name(&slot_name, idx), expr.clone())),
                Nullability::NonNullable,
            ),
        };

        let partition = partition.optimize_recursive(partitioner.scope())?;
        partition_dtypes.push(partition.return_dtype(partitioner.scope())?);
        partition_names.push(slot_name);
        partitions.push(partitioner.step_into(partition, slot)?);
    }

    let partition_names = FieldNames::from(partition_names);
    let root_scope = DType::Struct(
        StructFields::new(partition_names.clone(), partition_dtypes.clone()),
        Nullability::NonNullable,
    );
    let root = unwrap_single(root, &unwrapped);

    Ok(PartitionedExpr {
        root: root.optimize_recursive(&root_scope)?,
        partitions: partitions.into_boxed_slice(),
        partition_names,
        partition_dtypes: partition_dtypes.into_boxed_slice(),
        partition_annotations: splitter.slots.into_boxed_slice(),
    })
}

/// Each slot may hold several sub-expressions, so each needs a unique name within its pack.
fn sub_expression_name(slot_name: &FieldName, idx: usize) -> FieldName {
    FieldName::from(format!("{slot_name}_{idx}"))
}

/// Rewrite the root expression's references to slots whose partition skipped its pack, replacing
/// `$.<slot>.<slot>_0` with `$.<slot>`.
fn unwrap_single(root: Expression, unwrapped: &HashMap<FieldName, FieldName>) -> Expression {
    if unwrapped.is_empty() {
        return root;
    }
    root.transform_down(|node| {
        if let Some(field_name) = node.as_opt::<GetItem>()
            && let Some(slot_field) = node.child(0).as_opt::<GetItem>()
            && is_root(node.child(0).child(0))
            && let Some(expected) = unwrapped.get(slot_field)
            && expected == field_name
        {
            return Ok(Transformed {
                value: col(slot_field.clone()),
                order: TraversalOrder::Skip,
                changed: true,
            });
        }
        Ok(Transformed::no(node))
    })
    .vortex_expect("unwrap_single is infallible")
    .into_inner()
}

/// The partitioner behind [`partition`] and [`partition_annotations`]: slots are named by
/// converting the annotation into a [`FieldName`], and neither stage 1 nor stage 3 does anything.
struct FieldNamePartitioner<'a, A> {
    scope: &'a DType,
    slot: PhantomData<A>,
}

impl<A> Partitioner for FieldNamePartitioner<'_, A>
where
    A: Annotation + Display,
    FieldName: From<A>,
{
    type Slot = A;

    fn scope(&self) -> &DType {
        self.scope
    }

    fn slot_name(&self, slot: &Self::Slot) -> FieldName {
        FieldName::from(slot.clone())
    }
}

/// Partition an expression into sub-expressions that are uniquely associated with an annotation.
/// A root expression is also returned that can be used to recombine the results of the partitions
/// into the result of the original expression.
///
/// ## Note
///
/// This function currently respects the validity of each field in the scope, but the not validity
/// of the scope itself. The fix would be for the returned `PartitionedExpr` to include a partition
/// expression for computing the validity, or to include that expression as part of the root.
///
/// The struct layout works around this with a [`Partitioner`] of its own, which treats the
/// struct's validity as a slot.
///
/// See <https://github.com/vortex-data/vortex/issues/1907>.
pub fn partition<A: AnnotationFn>(
    expr: Expression,
    scope: &DType,
    annotate_fn: A,
) -> VortexResult<PartitionedExpr<A::Annotation>>
where
    A::Annotation: Display,
    FieldName: From<A::Annotation>,
{
    partition_annotated(
        &FieldNamePartitioner {
            scope,
            slot: PhantomData,
        },
        expr,
        annotate_fn,
    )
}

/// As [`partition`], but over annotations the caller has already computed.
pub fn partition_annotations<A>(
    expr: Expression,
    scope: &DType,
    annotations: Annotations<A>,
) -> VortexResult<PartitionedExpr<A>>
where
    A: Display + Clone + Eq + Hash,
    FieldName: From<A>,
{
    partition_with(
        &FieldNamePartitioner {
            scope,
            slot: PhantomData,
        },
        &expr,
        &annotations,
    )
}

/// The result of partitioning an expression.
#[derive(Debug)]
pub struct PartitionedExpr<A> {
    /// The root expression used to re-assemble the results.
    pub root: Expression,
    /// The partition expressions themselves.
    pub partitions: Box<[Expression]>,
    /// The field name of each partition as referenced in the root expression.
    pub partition_names: FieldNames,
    /// The return dtype of each partition expression.
    pub partition_dtypes: Box<[DType]>,
    /// The annotation associated with each partition.
    pub partition_annotations: Box<[A]>,
}

impl<A: Display> Display for PartitionedExpr<A> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "root: {} {{{}}}",
            self.root,
            self.partition_names
                .iter()
                .zip(self.partitions.iter())
                .map(|(name, partition)| format!("{name}: {partition}"))
                .join(", ")
        )
    }
}

impl<A: Annotation> PartitionedExpr<A>
where
    FieldName: From<A>,
{
    /// Return the partition for a given field, if it exists.
    // FIXME(ngates): this should return an iterator since an annotation may have multiple partitions.
    pub fn find_partition(&self, id: &A) -> Option<&Expression> {
        let id = FieldName::from(id.clone());
        self.partition_names
            .iter()
            .position(|field| field == id)
            .map(|idx| &self.partitions[idx])
    }
}

/// Cuts an expression into the maximal subtrees that each touch exactly one slot.
struct ExpressionSplitter<'a, P: Partitioner> {
    partitioner: &'a P,
    annotations: &'a Annotations<'a, P::Slot>,
    /// The slots encountered, in the order they were first encountered. Partitions are emitted in
    /// this order, so that partitioning the same expression twice gives the same plan.
    slots: Vec<P::Slot>,
    /// The sub-expressions of each slot in [`Self::slots`], parallel to it.
    sub_expressions: Vec<Vec<Expression>>,
}

impl<P: Partitioner> ExpressionSplitter<'_, P> {
    /// Record `expr` as a sub-expression of `slot`, returning the expression that reads its
    /// result back out of the root scope.
    fn push(&mut self, slot: &P::Slot, expr: Expression) -> Expression {
        let slot_idx = self
            .slots
            .iter()
            .position(|s| s == slot)
            .unwrap_or_else(|| {
                self.slots.push(slot.clone());
                self.sub_expressions.push(Vec::new());
                self.slots.len() - 1
            });

        let sub_exprs = &mut self.sub_expressions[slot_idx];
        let idx = sub_exprs.len();
        sub_exprs.push(expr);

        let slot_name = self.partitioner.slot_name(slot);
        get_item(sub_expression_name(&slot_name, idx), col(slot_name))
    }
}

impl<P: Partitioner> NodeRewriter for ExpressionSplitter<'_, P> {
    type NodeTy = Expression;

    fn visit_down(&mut self, node: Self::NodeTy) -> VortexResult<Transformed<Self::NodeTy>> {
        match self.annotations.get(&node) {
            // If this expression only touches a single slot, it becomes a partition and we can
            // skip its children.
            Some(slots) if slots.len() == 1 => {
                let slot = slots
                    .iter()
                    .next()
                    .vortex_expect("expected one slot")
                    .clone();
                let value = self.push(&slot, node.clone());
                Ok(Transformed {
                    value,
                    changed: true,
                    order: TraversalOrder::Skip,
                })
            }

            // Otherwise, continue traversing.
            _ => Ok(Transformed::no(node)),
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::fixture;
    use rstest::rstest;

    use super::*;
    use crate::dtype::DType;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::Nullability::Nullable;
    use crate::dtype::PType::I32;
    use crate::dtype::StructFields;
    use crate::expr::analysis::make_free_field_annotator;
    use crate::expr::and;
    use crate::expr::col;
    use crate::expr::get_item;
    use crate::expr::lit;
    use crate::expr::merge;
    use crate::expr::pack;
    use crate::expr::root;
    use crate::expr::transform::replace::replace_root_fields;
    use crate::scalar_fn::fns::pack::Pack;

    #[fixture]
    fn dtype() -> DType {
        DType::Struct(
            StructFields::from_iter([
                (
                    "a",
                    DType::Struct(
                        StructFields::from_iter([("x", I32.into()), ("y", DType::from(I32))]),
                        NonNullable,
                    ),
                ),
                ("b", I32.into()),
                ("c", I32.into()),
            ]),
            NonNullable,
        )
    }

    /// The same shape as [`dtype`], but with a nullable root struct.
    #[fixture]
    fn nullable_dtype(dtype: DType) -> DType {
        let DType::Struct(fields, _) = dtype else {
            unreachable!("fixture is a struct")
        };
        DType::Struct(fields, Nullable)
    }

    /// Re-assemble a partitioning and return the dtype it produces, after checking that every
    /// partition can be evaluated in the scope it will be dispatched to.
    ///
    /// This partitioner does not step partitions down into a child scope — that is left to the
    /// caller — so every partition is evaluated in the original `scope`.
    fn assembled_dtype(partitioned: &PartitionedExpr<FieldName>, scope: &DType) -> DType {
        for partition in partitioned.partitions.iter() {
            partition
                .return_dtype(scope)
                .vortex_expect("partition must type-check in the scope it is evaluated in");
        }
        let root_scope = DType::Struct(
            StructFields::new(
                partitioned.partition_names.clone(),
                partitioned.partition_dtypes.to_vec(),
            ),
            NonNullable,
        );
        partitioned
            .root
            .return_dtype(&root_scope)
            .vortex_expect("root must type-check over the partition results")
    }

    fn partition_expanded(expr: Expression, scope: &DType) -> PartitionedExpr<FieldName> {
        let fields = scope
            .as_struct_fields_opt()
            .vortex_expect("scope is a struct");
        let expanded = replace_root_fields(expr, fields)
            .optimize_recursive(scope)
            .vortex_expect("optimize");
        partition(expanded, scope, make_free_field_annotator(fields)).vortex_expect("partition")
    }

    /// Over a non-nullable scope, partitioning preserves the expression's dtype.
    #[rstest]
    #[case(root())]
    #[case(col("b"))]
    #[case(get_item("x", col("a")))]
    #[case(and(get_item("x", col("a")), get_item("y", col("a"))))]
    #[case(pack([("b", col("b")), ("c", col("c"))], NonNullable))]
    fn round_trips_dtype_over_non_nullable_scope(dtype: DType, #[case] expr: Expression) {
        let expected = expr.return_dtype(&dtype).unwrap();
        let partitioned = partition_expanded(expr, &dtype);
        assert_eq!(assembled_dtype(&partitioned, &dtype), expected);
    }

    /// Over a *nullable* scope this partitioner drops the struct's own validity: it has no
    /// partition for it, so the re-assembled expression reports a non-nullable struct where the
    /// original reports a nullable one.
    ///
    /// This is the known unsoundness described on [`partition`]; the struct layout works around it
    /// with its own partitioner. See <https://github.com/vortex-data/vortex/issues/1907>.
    #[rstest]
    fn drops_validity_over_nullable_scope(nullable_dtype: DType) {
        let expr = root();
        let expected = expr.return_dtype(&nullable_dtype).unwrap();
        assert!(expected.is_nullable());

        let partitioned = partition_expanded(expr, &nullable_dtype);
        let assembled = assembled_dtype(&partitioned, &nullable_dtype);

        assert!(
            !assembled.is_nullable(),
            "expected the known-unsound behaviour: validity is dropped, got {assembled}"
        );
        // ...and the nullability is pushed down into the fields instead.
        assert!(
            assembled
                .as_struct_fields()
                .field("b")
                .unwrap()
                .is_nullable()
        );
    }

    /// Each partition is a `pack` of that annotation's sub-expressions, and the root reads results
    /// back out by index. Several consumers depend on this shape.
    #[rstest]
    fn partitions_are_packed_by_annotation(dtype: DType) {
        let expr = and(get_item("x", col("a")), col("b"));
        let partitioned = partition_expanded(expr, &dtype);

        assert_eq!(partitioned.partitions.len(), 2);
        for (name, partition) in partitioned
            .partition_names
            .iter()
            .zip(partitioned.partitions.iter())
        {
            assert!(
                partition.is::<Pack>(),
                "partition {name} should be a pack, got {partition}"
            );
        }
        assert!(
            partitioned
                .partition_dtypes
                .iter()
                .all(|dtype| dtype.is_struct()),
            "packed partitions have struct dtypes: {:?}",
            partitioned.partition_dtypes
        );
    }

    #[rstest]
    fn test_expr_top_level_ref(dtype: DType) {
        let fields = dtype.as_struct_fields_opt().unwrap();

        let expr = root();
        let partitioned =
            partition(expr.clone(), &dtype, make_free_field_annotator(fields)).unwrap();

        // An un-expanded root expression is annotated by all fields, but since it is a single node
        assert_eq!(partitioned.partitions.len(), 0);
        assert_eq!(&partitioned.root, &root());

        // Instead, callers must expand the root expression themselves.
        let expr = replace_root_fields(expr, fields);
        let partitioned = partition(expr, &dtype, make_free_field_annotator(fields)).unwrap();

        assert_eq!(partitioned.partitions.len(), fields.names().len());
    }

    #[rstest]
    fn test_expr_top_level_ref_get_item_and_split(dtype: DType) {
        let fields = dtype.as_struct_fields_opt().unwrap();

        let expr = get_item("y", get_item("a", root()));

        let partitioned = partition(expr, &dtype, make_free_field_annotator(fields)).unwrap();
        assert_eq!(&partitioned.root, &get_item("a_0", get_item("a", root())));
    }

    #[rstest]
    fn test_expr_top_level_ref_get_item_and_split_pack(dtype: DType) {
        let fields = dtype.as_struct_fields_opt().unwrap();

        let expr = pack(
            [
                ("x", get_item("x", get_item("a", root()))),
                ("y", get_item("y", get_item("a", root()))),
                ("c", get_item("c", root())),
            ],
            NonNullable,
        );
        let partitioned = partition(expr, &dtype, make_free_field_annotator(fields)).unwrap();

        let split_a = partitioned.find_partition(&"a".into()).unwrap();
        assert_eq!(
            &split_a.optimize_recursive(&dtype).unwrap(),
            &pack(
                [
                    ("a_0", get_item("x", get_item("a", root()))),
                    ("a_1", get_item("y", get_item("a", root())))
                ],
                NonNullable
            )
        );
    }

    #[rstest]
    fn test_expr_top_level_ref_get_item_add(dtype: DType) {
        let fields = dtype.as_struct_fields_opt().unwrap();

        let expr = and(get_item("y", get_item("a", root())), lit(1));
        let partitioned = partition(expr, &dtype, make_free_field_annotator(fields)).unwrap();

        // Whole expr is a single split
        assert_eq!(partitioned.partitions.len(), 1);
    }

    #[rstest]
    fn test_expr_top_level_ref_get_item_add_cannot_split(dtype: DType) {
        let fields = dtype.as_struct_fields_opt().unwrap();

        let expr = and(get_item("y", get_item("a", root())), get_item("b", root()));
        let partitioned = partition(expr, &dtype, make_free_field_annotator(fields)).unwrap();

        // One for id.a and id.b
        assert_eq!(partitioned.partitions.len(), 2);
    }

    #[rstest]
    fn test_expr_merge(dtype: DType) {
        let fields = dtype.as_struct_fields_opt().unwrap();

        let expr = merge([col("a"), pack([("b", col("b"))], NonNullable)]);

        let partitioned = partition(expr, &dtype, make_free_field_annotator(fields)).unwrap();
        let expected = pack(
            [
                ("x", get_item("x", get_item("a_0", col("a")))),
                ("y", get_item("y", get_item("a_0", col("a")))),
                ("b", get_item("b", get_item("b_0", col("b")))),
            ],
            NonNullable,
        );
        assert_eq!(
            &partitioned.root, &expected,
            "{} {}",
            partitioned.root, expected
        );

        assert_eq!(partitioned.partitions.len(), 2);

        let part_a = partitioned.find_partition(&"a".into()).unwrap();
        let expected_a = pack([("a_0", col("a"))], NonNullable);
        assert_eq!(part_a, &expected_a, "{part_a} {expected_a}");

        let part_b = partitioned.find_partition(&"b".into()).unwrap();
        let expected_b = pack([("b_0", pack([("b", col("b"))], NonNullable))], NonNullable);
        assert_eq!(part_b, &expected_b, "{part_b} {expected_b}");
    }
}
