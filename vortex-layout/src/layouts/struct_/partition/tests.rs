// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::dtype::Nullability::Nullable;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::expr::Expression;
use vortex_array::expr::col;
use vortex_array::expr::eq;
use vortex_array::expr::get_item;
use vortex_array::expr::is_not_null;
use vortex_array::expr::is_null;
use vortex_array::expr::lit;
use vortex_array::expr::or;
use vortex_array::expr::pack;
use vortex_array::expr::root;
use vortex_array::expr::select;
use vortex_error::VortexResult;

use super::StructPartitioned;
use super::StructPartitioner;
use super::StructSlot;

fn i32_() -> DType {
    DType::Primitive(PType::I32, NonNullable)
}

fn scope(nullability: vortex_array::dtype::Nullability) -> DType {
    DType::Struct(
        StructFields::from_iter([("a", i32_()), ("b", i32_())]),
        nullability,
    )
}

/// The dtype of the child layout that evaluates a slot. Slot 0 is the validity, slot `i + 1` is
/// field `i`.
fn child_dtype(scope: &DType, slot: StructSlot) -> DType {
    if slot.is_validity() {
        return DType::Bool(NonNullable);
    }
    scope
        .as_struct_fields_opt()
        .unwrap()
        .field_by_index(slot.index() - 1)
        .unwrap()
}

fn slots(partitioned: &StructPartitioned) -> Vec<StructSlot> {
    match partitioned {
        StructPartitioned::Single(slot, _) => vec![*slot],
        StructPartitioned::Multi(partitioned) => partitioned.partition_annotations.to_vec(),
    }
}

/// Type-check the partitioning: every partition must be evaluable in the scope of the child it is
/// dispatched to, and re-assembling them must reproduce the original expression's dtype.
fn assert_round_trips(scope: &DType, expr: &Expression) -> VortexResult<StructPartitioned> {
    let partitioned = StructPartitioner::new(scope)?.partition(expr.clone())?;

    let assembled = match &partitioned {
        StructPartitioned::Single(slot, partition) => {
            partition.return_dtype(&child_dtype(scope, *slot))?
        }
        StructPartitioned::Multi(multi) => {
            for (slot, partition) in multi
                .partition_annotations
                .iter()
                .zip(multi.partitions.iter())
            {
                partition.return_dtype(&child_dtype(scope, *slot))?;
            }
            let root_scope = DType::Struct(
                StructFields::new(
                    multi.partition_names.clone(),
                    multi.partition_dtypes.to_vec(),
                ),
                NonNullable,
            );
            multi.root.return_dtype(&root_scope)?
        }
    };

    assert_eq!(
        assembled,
        expr.return_dtype(scope)?,
        "partitioning {expr} over {scope} changed its dtype"
    );
    Ok(partitioned)
}

/// Partitioning must never change the dtype of an expression, whatever the struct's nullability.
#[rstest]
#[case(root())]
#[case(col("a"))]
#[case(select(vec![FieldName::from("a")], root()))]
#[case(is_null(root()))]
#[case(is_not_null(root()))]
#[case(eq(col("a"), lit(1)))]
#[case(or(eq(col("a"), lit(1)), eq(col("b"), lit(2))))]
#[case(pack([("x", col("a")), ("y", col("b"))], NonNullable))]
#[case(is_null(col("a")))]
fn round_trips_dtype(#[case] expr: Expression) -> VortexResult<()> {
    assert_round_trips(&scope(NonNullable), &expr)?;
    assert_round_trips(&scope(Nullable), &expr)?;
    Ok(())
}

/// Over a non-nullable struct, an expression touching one field is delegated straight to that
/// field's child, stepped down into its scope.
#[rstest]
fn single_field_is_delegated() -> VortexResult<()> {
    let partitioned = assert_round_trips(&scope(NonNullable), &eq(col("a"), lit(1)))?;

    let StructPartitioned::Single(slot, partition) = partitioned else {
        panic!("expected a single partition");
    };
    assert_eq!(slot, StructSlot::field(0));
    assert_eq!(partition, eq(root(), lit(1)));
    Ok(())
}

/// The same expression over a nullable struct additionally reads the validity child, because the
/// struct being null makes the comparison null.
#[rstest]
fn single_field_of_nullable_struct_reads_validity() -> VortexResult<()> {
    let partitioned = assert_round_trips(&scope(Nullable), &eq(col("a"), lit(1)))?;

    assert_eq!(
        slots(&partitioned),
        vec![StructSlot::field(0), StructSlot::VALIDITY]
    );
    Ok(())
}

/// `is_null($)` only needs the validity child; none of the fields are read.
#[rstest]
#[case(is_null(root()))]
#[case(is_not_null(root()))]
fn null_check_only_reads_validity(#[case] expr: Expression) -> VortexResult<()> {
    let partitioned = assert_round_trips(&scope(Nullable), &expr)?;

    assert_eq!(slots(&partitioned), vec![StructSlot::VALIDITY]);
    Ok(())
}

/// A non-nullable struct is never null, so the check folds away without reading anything.
#[rstest]
fn null_check_of_non_nullable_struct_reads_nothing() -> VortexResult<()> {
    let partitioned = assert_round_trips(&scope(NonNullable), &is_null(root()))?;

    assert!(slots(&partitioned).is_empty());
    Ok(())
}

/// Projecting the root reads every child: the validity plus every field.
#[rstest]
fn root_reads_every_child() -> VortexResult<()> {
    let partitioned = assert_round_trips(&scope(Nullable), &root())?;

    assert_eq!(
        slots(&partitioned),
        vec![
            StructSlot::field(0),
            StructSlot::field(1),
            StructSlot::VALIDITY,
        ]
    );

    // Each field partition is just the child's root: no `get_item` is pushed into the child.
    let StructPartitioned::Multi(multi) = &partitioned else {
        panic!("expected multiple partitions");
    };
    assert!(multi.partitions.iter().all(|p| p == &root()));
    Ok(())
}

/// `select` only reads the selected fields, plus the validity.
#[rstest]
fn select_reads_selected_fields_and_validity() -> VortexResult<()> {
    let expr = select(vec![FieldName::from("a")], root());
    let partitioned = assert_round_trips(&scope(Nullable), &expr)?;

    assert_eq!(
        slots(&partitioned),
        vec![StructSlot::field(0), StructSlot::VALIDITY]
    );
    Ok(())
}

/// Slots are addressed by index in the flat scope, so no user field name can be mistaken for
/// another slot — not even a field literally named after a slot index.
#[rstest]
fn field_names_cannot_clash_with_slots() -> VortexResult<()> {
    let scope = DType::Struct(
        StructFields::from_iter([("0", i32_()), ("1", DType::Bool(NonNullable))]),
        Nullable,
    );

    let partitioned = assert_round_trips(&scope, &root())?;
    let StructPartitioned::Multi(multi) = &partitioned else {
        panic!("expected multiple partitions");
    };

    // The field named "0" is slot 1 and the field named "1" is slot 2; the validity is slot 0.
    assert_eq!(
        multi.partition_annotations.to_vec(),
        vec![
            StructSlot::field(0),
            StructSlot::field(1),
            StructSlot::VALIDITY,
        ]
    );

    // And `$."0"` addresses the field named "0" — which is slot 1 — not the validity in slot 0.
    assert_eq!(
        slots(&assert_round_trips(&scope, &col("0"))?),
        vec![StructSlot::field(0), StructSlot::VALIDITY]
    );
    Ok(())
}

/// Two references to the same field are evaluated once, and the sub-expressions of a field are
/// packed into a single partition.
#[rstest]
fn repeated_references_are_deduplicated() -> VortexResult<()> {
    let expr = or(eq(col("a"), lit(1)), eq(col("a"), lit(2)));
    let partitioned = assert_round_trips(&scope(NonNullable), &expr)?;

    assert_eq!(slots(&partitioned), vec![StructSlot::field(0)]);

    // `$.a` is referenced twice but the partition is a single expression evaluated once: the
    // whole `or` is pushed into the field's child.
    let StructPartitioned::Single(_, partition) = &partitioned else {
        panic!("expected a single partition");
    };
    assert_eq!(partition, &or(eq(root(), lit(1)), eq(root(), lit(2))));
    Ok(())
}

/// A field referenced by two expressions that cannot be pushed down together is packed into one
/// partition, so its child is still read exactly once.
#[rstest]
fn multiple_sub_expressions_are_packed() -> VortexResult<()> {
    // `$.a` is compared against `$.b`, so neither field's expression can subsume the other.
    let expr = pack(
        [("x", eq(col("a"), col("b"))), ("y", get_item("a", root()))],
        NonNullable,
    );
    let partitioned = assert_round_trips(&scope(NonNullable), &expr)?;

    assert_eq!(
        slots(&partitioned),
        vec![StructSlot::field(0), StructSlot::field(1)]
    );

    let StructPartitioned::Multi(multi) = &partitioned else {
        panic!("expected multiple partitions");
    };
    // Both uses of `$.a` are deduplicated into a single sub-expression, so the partition is the
    // child's root rather than a pack of two identical expressions.
    assert!(multi.partitions.iter().all(|p| p == &root()));
    Ok(())
}
