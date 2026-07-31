// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Expressions over struct layouts, evaluated by [`StructReader`].
//!
//! Every expression here is partitioned over the layout's children — one per field, plus the
//! struct's own validity — and re-assembled by the partitioned root expression.
//!
//! The same scenarios are covered end to end, over a serialized Vortex file, by
//! `vortex-file/tests/struct_nullability.rs`. The two suites are deliberately independent — they
//! share no fixtures or expectations — so that a change to one side cannot silently move both.
//!
//! [`StructReader`]: super::StructReader

use std::sync::Arc;

use rstest::fixture;
use rstest::rstest;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
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
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
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

// ---- Common components for the layout group ----

/// A written struct layout, together with the segments holding its data.
struct Written {
    segments: Arc<dyn SegmentSource>,
    layout: LayoutRef,
}

/// Write `array` through a [`TableStrategy`], producing a struct layout with one child per field
/// plus, when the struct is nullable, a validity child.
fn write(array: ArrayRef) -> Written {
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
                ArrayContext::empty().into(),
                segments2,
                array.to_array_stream().sequenced(ptr),
                eof,
                &session,
            )
            .await
    })
    .unwrap();

    Written { segments, layout }
}

impl Written {
    /// Evaluate a projection expression over every row of the layout.
    fn project(&self, expr: &Expression) -> VortexResult<ArrayRef> {
        self.project_masked(expr, Mask::new_true(self.row_count()))
    }

    /// Evaluate a projection expression over the rows selected by `mask`.
    fn project_masked(&self, expr: &Expression, mask: Mask) -> VortexResult<ArrayRef> {
        let (segments, layout, expr) = self.parts(expr);
        let row_range = 0..layout.row_count();
        block_on(move |handle| {
            let session = new_session().with_handle(handle);
            async move {
                layout
                    .new_reader("".into(), segments, &session, &Default::default())?
                    .projection_evaluation(&row_range, &expr, MaskFuture::ready(mask))?
                    .await
            }
        })
    }

    /// Evaluate a filter expression over every row of the layout.
    fn filter(&self, expr: &Expression) -> VortexResult<Mask> {
        let (segments, layout, expr) = self.parts(expr);
        let row_range = 0..layout.row_count();
        let mask = MaskFuture::new_true(row_count(&layout));
        block_on(move |handle| {
            let session = new_session().with_handle(handle);
            async move {
                layout
                    .new_reader("".into(), segments, &session, &Default::default())?
                    .filter_evaluation(&row_range, &expr, mask)?
                    .await
            }
        })
    }

    fn row_count(&self) -> usize {
        row_count(&self.layout)
    }

    fn dtype(&self) -> &DType {
        self.layout.dtype()
    }

    fn parts(&self, expr: &Expression) -> (Arc<dyn SegmentSource>, LayoutRef, Expression) {
        (
            Arc::clone(&self.segments),
            Arc::clone(&self.layout),
            expr.clone(),
        )
    }
}

fn row_count(layout: &LayoutRef) -> usize {
    usize::try_from(layout.row_count()).vortex_expect("row count fits in a usize")
}

/// `{}` with five rows and no fields.
#[fixture]
fn empty_struct() -> Written {
    write(
        StructArray::try_new(
            Vec::<FieldName>::new().into(),
            vec![],
            5,
            Validity::NonNullable,
        )
        .unwrap()
        .into_array(),
    )
}

/// `{a: i32, b: i32, c: i32}`.
#[fixture]
fn struct_layout() -> Written {
    write(
        StructArray::from_fields(
            [
                ("a", buffer![7, 2, 3].into_array()),
                ("b", buffer![4, 5, 6].into_array()),
                ("c", buffer![4, 5, 6].into_array()),
            ]
            .as_slice(),
        )
        .unwrap()
        .into_array(),
    )
}

/// `{a: i32, b: i32, c: i32}?` where row 0 is a null struct.
#[fixture]
fn null_struct_layout() -> Written {
    write(
        StructArray::try_from_iter_with_validity(
            [
                ("a", buffer![7, 2, 3].into_array()),
                ("b", buffer![4, 5, 6].into_array()),
                ("c", buffer![4, 5, 6].into_array()),
            ],
            Validity::Array(BoolArray::from_iter([false, true, true]).into_array()),
        )
        .unwrap()
        .into_array(),
    )
}

/// `{a: {b: {c: i32}?}}` where `a.b` is null at row 1.
#[fixture]
fn nested_struct_layout() -> Written {
    write(
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
        .into_array(),
    )
}

// ---- Tests over a non-nullable struct ----

#[rstest]
fn filter_or(struct_layout: Written) -> VortexResult<()> {
    let filt = or(
        eq(col("a"), lit(7)),
        or(eq(col("b"), lit(5)), eq(col("a"), lit(3))),
    );
    assert_eq!(
        struct_layout.filter(&filt)?,
        Mask::from_iter([true, true, true])
    );
    Ok(())
}

#[rstest]
fn project_comparison(struct_layout: Written) -> VortexResult<()> {
    let expr = gt(get_item("a", root()), get_item("b", root()));
    let result = struct_layout.project(&expr)?;

    let mut ctx = SESSION.create_execution_ctx();
    assert_arrays_eq!(result, BoolArray::from_iter([true, false, false]), &mut ctx);
    Ok(())
}

#[rstest]
fn project_comparison_with_row_mask(struct_layout: Written) -> VortexResult<()> {
    let expr = gt(get_item("a", root()), get_item("b", root()));
    let result = struct_layout.project_masked(&expr, Mask::from_iter([true, true, false]))?;

    let mut ctx = SESSION.create_execution_ctx();
    assert_arrays_eq!(result, BoolArray::from_iter([true, false]), &mut ctx);
    Ok(())
}

#[rstest]
fn project_pack_with_row_mask(struct_layout: Written) -> VortexResult<()> {
    let expr = pack(
        [("a", get_item("a", root())), ("b", get_item("b", root()))],
        Nullability::NonNullable,
    );
    // Take rows 0 and 1, skip row 2 and anything after that.
    let result = struct_layout.project_masked(&expr, Mask::from_iter([true, true, false]))?;

    assert_eq!(result.len(), 2);

    let mut ctx = SESSION.create_execution_ctx();
    let result = result.execute::<StructArray>(&mut ctx)?;
    assert_arrays_eq!(
        result.unmasked_field_by_name("a")?,
        PrimitiveArray::from_iter([7i32, 2]),
        &mut ctx
    );
    assert_arrays_eq!(
        result.unmasked_field_by_name("b")?,
        PrimitiveArray::from_iter([4i32, 5]),
        &mut ctx
    );
    Ok(())
}

#[rstest]
fn project_empty_struct(empty_struct: Written) -> VortexResult<()> {
    let expr = pack(Vec::<(String, Expression)>::new(), Nullability::Nullable);
    let result = empty_struct.project(&expr)?;

    assert!(result.dtype().is_struct());
    assert_eq!(result.len(), 5);
    Ok(())
}

/// Regression test for <https://github.com/vortex-data/vortex/issues/7808>.
///
/// A filter expression whose DType is incompatible with the scanned schema (e.g. comparing a u8
/// column to an i32 literal) must return an error, not panic.
#[test]
fn filter_dtype_mismatch_returns_error() {
    let written = write(
        StructArray::from_fields(
            [
                ("age", buffer![7u8, 2, 3].into_array()),
                ("score", buffer![4u8, 5, 6].into_array()),
            ]
            .as_slice(),
        )
        .unwrap()
        .into_array(),
    );

    let err = written
        .filter(&eq(col("age"), lit(67i32)))
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains("Cannot compare different DTypes"), "{err}");
}

// ---- Tests over a nullable struct ----

/// Projecting the whole struct must round-trip its DType: the struct itself stays nullable and
/// its fields stay non-nullable.
#[rstest]
fn project_root_of_null_struct(null_struct_layout: Written) -> VortexResult<()> {
    let result = null_struct_layout.project(&root())?;

    assert_eq!(result.dtype(), null_struct_layout.dtype());

    let mut ctx = SESSION.create_execution_ctx();
    assert!(result.execute_scalar(0, &mut ctx)?.is_null());
    assert!(!result.execute_scalar(1, &mut ctx)?.is_null());
    Ok(())
}

/// `select` keeps the struct's own nullability rather than pushing it into the fields.
#[rstest]
fn select_from_null_struct(null_struct_layout: Written) -> VortexResult<()> {
    let expr = select(vec![FieldName::from("a"), FieldName::from("b")], root());
    let result = null_struct_layout.project(&expr)?;

    assert_eq!(
        result.dtype(),
        &DType::Struct(
            StructFields::from_iter([
                ("a", DType::Primitive(PType::I32, Nullability::NonNullable)),
                ("b", DType::Primitive(PType::I32, Nullability::NonNullable)),
            ]),
            Nullability::Nullable,
        )
    );

    let mut ctx = SESSION.create_execution_ctx();
    assert!(result.execute_scalar(0, &mut ctx)?.is_null());
    assert!(!result.execute_scalar(1, &mut ctx)?.is_null());
    Ok(())
}

/// `pack` of `get_item`s pushes the struct's nullability down into the packed fields, leaving the
/// new struct itself non-nullable.
#[rstest]
fn pack_from_null_struct(null_struct_layout: Written) -> VortexResult<()> {
    let expr = pack(
        [("a", get_item("a", root())), ("b", get_item("b", root()))],
        Nullability::NonNullable,
    );
    let result = null_struct_layout.project(&expr)?;

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

    let mut ctx = SESSION.create_execution_ctx();
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
fn get_item_from_null_struct(null_struct_layout: Written) -> VortexResult<()> {
    let result = null_struct_layout.project(&col("a"))?;

    assert_eq!(
        result.dtype(),
        &DType::Primitive(PType::I32, Nullability::Nullable)
    );

    let mut ctx = SESSION.create_execution_ctx();
    assert!(result.execute_scalar(0, &mut ctx)?.is_null());
    assert_nth_scalar!(result, 1, 2, &mut ctx);
    assert_nth_scalar!(result, 2, 3, &mut ctx);
    Ok(())
}

/// `is_null` / `is_not_null` of the root scope report the struct's own validity, which lives in
/// its own partition.
#[rstest]
#[case(false, [true, false, false])]
#[case(true, [false, true, true])]
fn null_checks_on_null_struct(
    null_struct_layout: Written,
    #[case] negate: bool,
    #[case] expected: [bool; 3],
) -> VortexResult<()> {
    let expr = if negate {
        is_not_null(root())
    } else {
        is_null(root())
    };
    let result = null_struct_layout.project(&expr)?;

    let mut ctx = SESSION.create_execution_ctx();
    assert_arrays_eq!(result, BoolArray::from_iter(expected), &mut ctx);
    Ok(())
}

/// Filtering on the struct's own validity.
#[rstest]
fn filter_is_null_of_null_struct(null_struct_layout: Written) -> VortexResult<()> {
    assert_eq!(
        null_struct_layout.filter(&is_null(root()))?,
        Mask::from_iter([true, false, false])
    );
    Ok(())
}

/// Row 0 holds `a == 7`, but the struct is null there, so the comparison is null and the row must
/// not match.
#[rstest]
fn filter_field_of_null_struct(null_struct_layout: Written) -> VortexResult<()> {
    assert_eq!(
        null_struct_layout.filter(&eq(col("a"), lit(7)))?,
        Mask::from_iter([false, false, false])
    );
    Ok(())
}

// ---- Tests over a nested nullable struct ----

/// Projecting a nested nullable struct preserves that struct's own nullability.
#[rstest]
fn project_nested_child(nested_struct_layout: Written) -> VortexResult<()> {
    let expected = nested_struct_layout
        .dtype()
        .as_struct_fields()
        .field_by_index(0)
        .vortex_expect("field 0 exists");
    let result = nested_struct_layout.project(&col("a"))?;

    assert_eq!(result.dtype(), &expected);

    let mut ctx = SESSION.create_execution_ctx();
    assert!(!result.execute_scalar(0, &mut ctx)?.is_null());
    assert!(result.execute_scalar(1, &mut ctx)?.is_null());
    Ok(())
}

/// Selecting a field out of a nested nullable struct keeps the nulls of that struct.
#[rstest]
fn select_from_nested_child(nested_struct_layout: Written) -> VortexResult<()> {
    let expr = select(
        vec![FieldName::from("c")],
        get_item("b", get_item("a", root())),
    );
    let result = nested_struct_layout.project(&expr)?;

    // The result is a nullable struct (because `$.a.b` is nullable) with a non-nullable field "c"
    // (because the original field was non-nullable).
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

    let mut ctx = SESSION.create_execution_ctx();
    // Row 0: struct is valid, field "c" is 4.
    assert_eq!(
        result
            .execute_scalar(0, &mut ctx)?
            .as_struct()
            .field_by_idx(0)
            .vortex_expect("field 0 exists"),
        vortex_array::scalar::Scalar::primitive(4, Nullability::NonNullable)
    );
    // Row 1: struct is null, because `$.a.b` was null at this row.
    assert!(result.execute_scalar(1, &mut ctx)?.as_struct().is_null());
    // Row 2: struct is valid, field "c" is 6.
    assert_eq!(
        result
            .execute_scalar(2, &mut ctx)?
            .as_struct()
            .field_by_idx(0)
            .vortex_expect("field 0 exists"),
        vortex_array::scalar::Scalar::primitive(6, Nullability::NonNullable)
    );
    Ok(())
}
