// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Expressions over nullable structs, evaluated by an in-memory scan.
//!
//! No layouts and no partitioning are involved: the expression is applied straight to a struct
//! array. This is the reference semantics that the layout group must reproduce.

use rstest::fixture;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::BoolArray;
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
use vortex_array::expr::is_not_null;
use vortex_array::expr::is_null;
use vortex_array::expr::lit;
use vortex_array::expr::pack;
use vortex_array::expr::root;
use vortex_array::expr::select;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;

// ---- Common components for the in-memory group ----

fn ctx() -> ExecutionCtx {
    array_session().create_execution_ctx()
}

/// `{a: i32, b: i32, c: i32}?` where row 0 is a null struct.
#[fixture]
fn null_struct() -> ArrayRef {
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
}

/// `{a: {b: {c: i32}?}}` where `a.b` is null at row 1.
#[fixture]
fn nested_struct() -> ArrayRef {
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
                Validity::Array(BoolArray::from_iter([true, false, true]).into_array()),
            )
            .unwrap()
            .into_array(),
        )],
        Validity::NonNullable,
    )
    .unwrap()
    .into_array()
}

/// Evaluate a projection expression over the array.
fn project(array: ArrayRef, expr: &Expression) -> VortexResult<ArrayRef> {
    array.apply(expr)?.execute::<ArrayRef>(&mut ctx())
}

/// Evaluate a filter expression over the array, coercing nulls to `false`.
fn filter(array: ArrayRef, expr: &Expression) -> VortexResult<Mask> {
    array.apply(expr)?.null_as_false().execute(&mut ctx())
}

/// The dtype of `{a: i32, b: i32}` with the given struct nullability and non-nullable fields.
fn ab_struct(nullability: Nullability) -> DType {
    DType::Struct(
        StructFields::from_iter([
            ("a", DType::Primitive(PType::I32, Nullability::NonNullable)),
            ("b", DType::Primitive(PType::I32, Nullability::NonNullable)),
        ]),
        nullability,
    )
}

// ---- Tests ----

/// Projecting the whole struct returns it unchanged, nullability and all.
#[rstest]
fn project_root(null_struct: ArrayRef) -> VortexResult<()> {
    let expected = null_struct.dtype().clone();
    let result = project(null_struct, &root())?;

    assert_eq!(result.dtype(), &expected);
    assert!(result.execute_scalar(0, &mut ctx())?.is_null());
    assert!(!result.execute_scalar(1, &mut ctx())?.is_null());
    Ok(())
}

/// `select` keeps the struct's own nullability rather than pushing it into the fields.
#[rstest]
fn select_fields(null_struct: ArrayRef) -> VortexResult<()> {
    let expr = select(vec![FieldName::from("a"), FieldName::from("b")], root());
    let result = project(null_struct, &expr)?;

    assert_eq!(result.dtype(), &ab_struct(Nullability::Nullable));
    assert!(result.execute_scalar(0, &mut ctx())?.is_null());
    assert!(!result.execute_scalar(1, &mut ctx())?.is_null());
    Ok(())
}

/// `pack` of `get_item`s pushes the struct's nullability down into the packed fields, leaving the
/// new struct itself non-nullable.
#[rstest]
fn pack_fields(null_struct: ArrayRef) -> VortexResult<()> {
    let expr = pack(
        [("a", get_item("a", root())), ("b", get_item("b", root()))],
        Nullability::NonNullable,
    );
    let result = project(null_struct, &expr)?;

    assert_eq!(
        result.dtype(),
        &DType::Struct(
            StructFields::from_iter([
                ("a", DType::Primitive(PType::I32, Nullability::Nullable)),
                ("b", DType::Primitive(PType::I32, Nullability::Nullable)),
            ]),
            Nullability::NonNullable,
        )
    );

    let mut ctx = ctx();
    let a = result
        .execute::<StructArray>(&mut ctx)?
        .unmasked_field_by_name("a")?
        .clone();
    assert!(a.execute_scalar(0, &mut ctx)?.is_null());
    assert_nth_scalar!(a, 1, 2, &mut ctx);
    Ok(())
}

/// `get_item` intersects the struct's validity with the field's.
#[rstest]
fn get_item_field(null_struct: ArrayRef) -> VortexResult<()> {
    let result = project(null_struct, &col("a"))?;

    assert_eq!(
        result.dtype(),
        &DType::Primitive(PType::I32, Nullability::Nullable)
    );

    let mut ctx = ctx();
    assert!(result.execute_scalar(0, &mut ctx)?.is_null());
    assert_nth_scalar!(result, 1, 2, &mut ctx);
    Ok(())
}

/// `is_null` / `is_not_null` of the root scope report the struct's own validity.
#[rstest]
#[case(false, [true, false, false])]
#[case(true, [false, true, true])]
fn null_checks(
    null_struct: ArrayRef,
    #[case] negate: bool,
    #[case] expected: [bool; 3],
) -> VortexResult<()> {
    let expr = if negate {
        is_not_null(root())
    } else {
        is_null(root())
    };
    let result = project(null_struct, &expr)?;

    assert_arrays_eq!(result, BoolArray::from_iter(expected), &mut ctx());
    Ok(())
}

/// Filtering on the struct's own validity.
#[rstest]
fn filter_is_null(null_struct: ArrayRef) -> VortexResult<()> {
    let result = filter(null_struct, &is_null(root()))?;
    assert_eq!(result, Mask::from_iter([true, false, false]));
    Ok(())
}

/// Row 0 holds `a == 7`, but the struct is null there, so the comparison is null and the row must
/// not match.
#[rstest]
fn filter_field_of_null_struct(null_struct: ArrayRef) -> VortexResult<()> {
    let result = filter(null_struct, &eq(col("a"), lit(7)))?;
    assert_eq!(result, Mask::from_iter([false, false, false]));
    Ok(())
}

/// Projecting a nested nullable struct preserves that struct's own nullability.
#[rstest]
fn project_nested_child(nested_struct: ArrayRef) -> VortexResult<()> {
    let expected = nested_struct
        .dtype()
        .as_struct_fields()
        .field_by_index(0)
        .vortex_expect("field 0 exists");
    let result = project(nested_struct, &col("a"))?;

    assert_eq!(result.dtype(), &expected);
    assert!(!result.execute_scalar(0, &mut ctx())?.is_null());
    assert!(result.execute_scalar(1, &mut ctx())?.is_null());
    Ok(())
}

/// Selecting a field out of a nested nullable struct keeps the nulls of that struct.
#[rstest]
fn select_from_nested_child(nested_struct: ArrayRef) -> VortexResult<()> {
    let expr = select(
        vec![FieldName::from("c")],
        get_item("b", get_item("a", root())),
    );
    let result = project(nested_struct, &expr)?;

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
    assert!(!result.execute_scalar(0, &mut ctx())?.is_null());
    assert!(result.execute_scalar(1, &mut ctx())?.is_null());
    Ok(())
}
