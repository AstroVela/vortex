// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::fixture;
use rstest::rstest;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::dtype::Nullability::Nullable;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::expr::Expression;
use vortex_array::expr::and;
use vortex_array::expr::cast;
use vortex_array::expr::col;
use vortex_array::expr::eq;
use vortex_array::expr::get_item;
use vortex_array::expr::is_not_null;
use vortex_array::expr::is_null;
use vortex_array::expr::lit;
use vortex_array::expr::mask;
use vortex_array::expr::not;
use vortex_array::expr::pack;
use vortex_array::expr::root;
use vortex_array::expr::select;
use vortex_error::VortexResult;

use super::StructPartitioned;
use super::StructSlot;
use super::partition_struct_expr;
use super::pruning_partition;

fn i32_field(nullability: vortex_array::dtype::Nullability) -> DType {
    DType::Primitive(PType::I32, nullability)
}

/// `{a: i32, b: i32, c: i32}`
#[fixture]
fn non_nullable() -> DType {
    DType::Struct(
        StructFields::from_iter([
            ("a", i32_field(NonNullable)),
            ("b", i32_field(NonNullable)),
            ("c", i32_field(NonNullable)),
        ]),
        NonNullable,
    )
}

/// `{a: i32, b: i32, c: i32}?`
#[fixture]
fn nullable() -> DType {
    DType::Struct(
        StructFields::from_iter([
            ("a", i32_field(NonNullable)),
            ("b", i32_field(NonNullable)),
            ("c", i32_field(NonNullable)),
        ]),
        Nullable,
    )
}

fn partition(expr: Expression, dtype: &DType) -> VortexResult<StructPartitioned> {
    partition_struct_expr(&expr, dtype, None)
}

/// The `(slot, expression)` pairs of a multi-partition result, in partition order.
fn parts(partitioned: &StructPartitioned) -> Vec<(StructSlot, Expression)> {
    match partitioned {
        StructPartitioned::Single(slot, expr) => vec![(*slot, expr.clone())],
        StructPartitioned::Multi(partitioned) => partitioned
            .partition_annotations
            .iter()
            .zip(partitioned.partitions.iter())
            .map(|(slot, expr)| (*slot, expr.clone()))
            .collect(),
    }
}

fn root_expr(partitioned: &StructPartitioned) -> Option<Expression> {
    match partitioned {
        StructPartitioned::Single(..) => None,
        StructPartitioned::Multi(partitioned) => Some(partitioned.root.clone()),
    }
}

#[rstest]
fn get_item_non_nullable(non_nullable: DType) -> VortexResult<()> {
    // A field of a non-nullable struct is answered entirely by that field's child.
    let partitioned = partition(col("b"), &non_nullable)?;
    assert_eq!(
        parts(&partitioned),
        vec![(StructSlot::Field(1), root())],
        "{partitioned:?}"
    );
    assert!(matches!(partitioned, StructPartitioned::Single(..)));
    Ok(())
}

#[rstest]
fn get_item_nullable(nullable: DType) -> VortexResult<()> {
    // `get_item` intersects the field with the struct validity, so it reads both children.
    let partitioned = partition(col("b"), &nullable)?;
    assert_eq!(
        parts(&partitioned),
        vec![
            (StructSlot::Validity, root()),
            (StructSlot::Field(1), root()),
        ],
        "{partitioned:?}"
    );
    assert_eq!(
        root_expr(&partitioned),
        Some(mask(col("$1"), col("$validity")))
    );
    Ok(())
}

#[rstest]
fn is_null_reads_only_validity(nullable: DType) -> VortexResult<()> {
    // The whole point of the fix: `is_null` of the struct itself touches no field at all.
    let partitioned = partition(is_null(root()), &nullable)?;
    assert_eq!(
        parts(&partitioned),
        vec![(StructSlot::Validity, not(root()))],
        "{partitioned:?}"
    );
    assert!(matches!(partitioned, StructPartitioned::Single(..)));
    Ok(())
}

#[rstest]
fn is_not_null_reads_only_validity(nullable: DType) -> VortexResult<()> {
    let partitioned = partition(is_not_null(root()), &nullable)?;
    assert_eq!(
        parts(&partitioned),
        vec![(StructSlot::Validity, root())],
        "{partitioned:?}"
    );
    Ok(())
}

#[rstest]
fn is_null_of_non_nullable_struct_reads_nothing(non_nullable: DType) -> VortexResult<()> {
    // There is no validity child to read, and the answer is a constant.
    let partitioned = partition(is_null(root()), &non_nullable)?;
    assert_eq!(parts(&partitioned), vec![], "{partitioned:?}");
    assert_eq!(root_expr(&partitioned), Some(lit(false)));
    Ok(())
}

#[rstest]
fn is_null_of_field_reads_validity(nullable: DType) -> VortexResult<()> {
    // `is_null` is not strict, so the validity mask cannot be hoisted over it: the field and the
    // validity are read separately and `is_null` is evaluated on the recombined value.
    let partitioned = partition(is_null(col("a")), &nullable)?;
    assert_eq!(
        parts(&partitioned),
        vec![
            (StructSlot::Validity, root()),
            (StructSlot::Field(0), root()),
        ],
        "{partitioned:?}"
    );
    assert_eq!(
        root_expr(&partitioned),
        Some(is_null(mask(col("$0"), col("$validity"))))
    );
    Ok(())
}

#[rstest]
fn select_keeps_struct_validity(nullable: DType) -> VortexResult<()> {
    // Unlike `get_item`, a selection does not push validity into the field values: the result is
    // a nullable struct whose fields are unmasked.
    let partitioned = partition(select(["c", "a"], root()), &nullable)?;
    assert_eq!(
        parts(&partitioned),
        vec![
            (StructSlot::Validity, root()),
            (StructSlot::Field(0), root()),
            (StructSlot::Field(2), root()),
        ],
        "{partitioned:?}"
    );
    assert_eq!(
        root_expr(&partitioned),
        Some(mask(
            pack([("c", col("$2")), ("a", col("$0"))], NonNullable),
            col("$validity")
        ))
    );
    Ok(())
}

#[rstest]
fn select_non_nullable(non_nullable: DType) -> VortexResult<()> {
    let partitioned = partition(select(["a", "b"], root()), &non_nullable)?;
    assert_eq!(
        parts(&partitioned),
        vec![
            (StructSlot::Field(0), root()),
            (StructSlot::Field(1), root()),
        ],
        "{partitioned:?}"
    );
    assert_eq!(
        root_expr(&partitioned),
        Some(pack([("a", col("$0")), ("b", col("$1"))], NonNullable))
    );
    Ok(())
}

#[rstest]
fn pack_masks_each_field(nullable: DType) -> VortexResult<()> {
    // `pack` of `get_item`s builds a non-nullable struct whose fields carry the struct validity.
    let expr = pack([("x", col("a")), ("y", col("b"))], NonNullable);
    let partitioned = partition(expr, &nullable)?;
    assert_eq!(
        parts(&partitioned),
        vec![
            (StructSlot::Validity, root()),
            (StructSlot::Field(0), root()),
            (StructSlot::Field(1), root()),
        ],
        "{partitioned:?}"
    );
    assert_eq!(
        root_expr(&partitioned),
        Some(pack(
            [
                ("x", mask(col("$0"), col("$validity"))),
                ("y", mask(col("$1"), col("$validity"))),
            ],
            NonNullable
        ))
    );
    Ok(())
}

#[rstest]
fn strict_predicate_pushes_into_field(nullable: DType) -> VortexResult<()> {
    // `eq` is strict, so `eq(mask(a, v), 5) == mask(eq(a, 5), v)`: the comparison itself is
    // pushed into the field's child and only the mask is left at the top.
    let partitioned = partition(eq(col("a"), lit(5)), &nullable)?;
    assert_eq!(
        parts(&partitioned),
        vec![
            (StructSlot::Validity, root()),
            (StructSlot::Field(0), eq(root(), lit(5))),
        ],
        "{partitioned:?}"
    );
    assert_eq!(
        root_expr(&partitioned),
        Some(mask(col("$0"), col("$validity")))
    );
    Ok(())
}

#[rstest]
fn strict_predicate_pushes_into_field_non_nullable(non_nullable: DType) -> VortexResult<()> {
    let partitioned = partition(eq(col("a"), lit(5)), &non_nullable)?;
    assert_eq!(
        parts(&partitioned),
        vec![(StructSlot::Field(0), eq(root(), lit(5)))],
        "{partitioned:?}"
    );
    assert!(matches!(partitioned, StructPartitioned::Single(..)));
    Ok(())
}

#[rstest]
fn predicate_over_two_fields(non_nullable: DType) -> VortexResult<()> {
    let expr = and(eq(col("a"), lit(5)), eq(col("c"), lit(6)));
    let partitioned = partition(expr, &non_nullable)?;
    assert_eq!(
        parts(&partitioned),
        vec![
            (StructSlot::Field(0), eq(root(), lit(5))),
            (StructSlot::Field(2), eq(root(), lit(6))),
        ],
        "{partitioned:?}"
    );
    assert_eq!(root_expr(&partitioned), Some(and(col("$0"), col("$2"))));
    Ok(())
}

#[rstest]
fn non_strict_predicate_is_not_hoisted(nullable: DType) -> VortexResult<()> {
    // `is_null` blocks the hoist, so the `and` stays at the top over both reads of the validity.
    let expr = and(is_null(root()), eq(col("a"), lit(5)));
    let partitioned = partition(expr, &nullable)?;
    assert_eq!(
        parts(&partitioned),
        vec![
            (
                StructSlot::Validity,
                pack([("$sub0", not(root())), ("$sub1", root())], NonNullable)
            ),
            (StructSlot::Field(0), eq(root(), lit(5))),
        ],
        "{partitioned:?}"
    );
    assert_eq!(
        root_expr(&partitioned),
        Some(and(
            get_item("$sub0", col("$validity")),
            mask(col("$0"), get_item("$sub1", col("$validity")))
        ))
    );
    Ok(())
}

#[rstest]
fn subtree_over_one_field_is_pushed_whole(non_nullable: DType) -> VortexResult<()> {
    // Everything below the pack reads field `a` only, so the whole subtree — pack included — is
    // handed to that one child.
    let expr = pack([("x", col("a")), ("y", eq(col("a"), lit(5)))], NonNullable);
    let partitioned = partition(expr, &non_nullable)?;
    assert_eq!(
        parts(&partitioned),
        vec![(
            StructSlot::Field(0),
            pack([("x", root()), ("y", eq(root(), lit(5)))], NonNullable)
        )],
        "{partitioned:?}"
    );
    assert!(matches!(partitioned, StructPartitioned::Single(..)));
    Ok(())
}

#[rstest]
fn root_reconstructs_the_struct(nullable: DType) -> VortexResult<()> {
    let partitioned = partition(root(), &nullable)?;
    assert_eq!(
        parts(&partitioned),
        vec![
            (StructSlot::Validity, root()),
            (StructSlot::Field(0), root()),
            (StructSlot::Field(1), root()),
            (StructSlot::Field(2), root()),
        ],
        "{partitioned:?}"
    );
    assert_eq!(
        root_expr(&partitioned),
        Some(mask(
            pack(
                [("a", col("$0")), ("b", col("$1")), ("c", col("$2"))],
                NonNullable
            ),
            col("$validity")
        ))
    );
    Ok(())
}

#[rstest]
fn opaque_function_over_root_reads_everything(nullable: DType) -> VortexResult<()> {
    // `cast` consumes the struct opaquely, so the struct is rebuilt from all of its children —
    // validity included — and the function is evaluated on top of the reconstruction.
    let target = DType::Struct(
        StructFields::from_iter([
            ("a", i32_field(Nullable)),
            ("b", i32_field(Nullable)),
            ("c", i32_field(Nullable)),
        ]),
        Nullable,
    );
    let partitioned = partition(cast(root(), target), &nullable)?;
    assert_eq!(
        parts(&partitioned)
            .into_iter()
            .map(|(slot, _)| slot)
            .collect::<Vec<_>>(),
        vec![
            StructSlot::Validity,
            StructSlot::Field(0),
            StructSlot::Field(1),
            StructSlot::Field(2),
        ],
        "{partitioned:?}"
    );
    Ok(())
}

#[rstest]
fn nested_field_access_steps_into_the_child() -> VortexResult<()> {
    let dtype = DType::Struct(
        StructFields::from_iter([(
            "a",
            DType::Struct(
                StructFields::from_iter([("x", i32_field(NonNullable))]),
                Nullable,
            ),
        )]),
        NonNullable,
    );

    // The outer struct is non-nullable, so everything below `root().a` is handed to the child.
    let partitioned = partition(get_item("x", col("a")), &dtype)?;
    assert_eq!(
        parts(&partitioned),
        vec![(StructSlot::Field(0), get_item("x", root()))],
        "{partitioned:?}"
    );
    Ok(())
}

#[rstest]
fn scope_independent_expression_has_no_partitions(nullable: DType) -> VortexResult<()> {
    let partitioned = partition(lit(5), &nullable)?;
    assert_eq!(parts(&partitioned), vec![], "{partitioned:?}");
    assert_eq!(root_expr(&partitioned), Some(lit(5)));
    Ok(())
}

#[rstest]
fn pruning_partition_sees_through_validity_mask(nullable: DType) -> VortexResult<()> {
    let partitioned = partition(eq(col("a"), lit(5)), &nullable)?;
    let StructPartitioned::Multi(partitioned) = &partitioned else {
        panic!("expected a multi-partition split, got {partitioned:?}");
    };

    // `mask` can only remove rows, so the field predicate alone is a sound pruning filter.
    let idx = pruning_partition(partitioned).expect("field predicate is prunable");
    assert_eq!(partitioned.partition_annotations[idx], StructSlot::Field(0));
    Ok(())
}

#[rstest]
fn pruning_partition_declines_non_boolean(nullable: DType) -> VortexResult<()> {
    let partitioned = partition(col("a"), &nullable)?;
    let StructPartitioned::Multi(partitioned) = &partitioned else {
        panic!("expected a multi-partition split, got {partitioned:?}");
    };
    assert_eq!(pruning_partition(partitioned), None);
    Ok(())
}

#[rstest]
fn dtype_mismatch_is_rejected(non_nullable: DType) {
    // Partitioning type-checks the expression before splitting it up.
    let err = partition(eq(col("a"), lit(5u8)), &non_nullable)
        .expect_err("comparing i32 to u8 must fail");
    assert!(
        err.to_string().contains("Cannot compare different DTypes"),
        "{err}"
    );
}
