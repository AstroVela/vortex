// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end tests for expressions over nullable structs.
//!
//! Each test writes a real Vortex file to an in-memory buffer — no disk — and scans it back
//! through the full stack: writer, footer, layouts, and the layout readers that partition the
//! scan's expressions over a struct layout's children.
//!
//! A struct layout stores its own validity as a child alongside its fields, so a nullable struct
//! is where a scan can most easily lose track of nullability: pushing it down into the fields,
//! dropping it, or failing to consult it at all. These tests pin down what a scan must return.
//!
//! The same scenarios are covered against the layout reader in isolation by
//! `vortex-layout/src/layouts/struct_/reader/tests.rs`. The two suites are deliberately
//! independent — they share no fixtures or expectations — so that a change to one side cannot
//! silently move both.

#![expect(clippy::tests_outside_test_module)]

use std::sync::LazyLock;

use rstest::fixture;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
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
use vortex_array::stream::ArrayStreamExt;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::session::RuntimeSession;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

mod common;

use common::enable_all_registered_array_encodings;

// ---- Common components ----

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session()
        .with::<LayoutSession>()
        .with::<RuntimeSession>();
    vortex_file::register_default_encodings(&session);
    enable_all_registered_array_encodings(&session);
    session
});

fn ctx() -> ExecutionCtx {
    SESSION.create_execution_ctx()
}

/// A Vortex file held entirely in memory.
struct File {
    buffer: ByteBuffer,
    dtype: DType,
}

/// Serialize `array` into an in-memory Vortex file.
async fn write(array: ArrayRef) -> VortexResult<File> {
    let dtype = array.dtype().clone();
    let mut buffer = ByteBufferMut::empty();
    SESSION
        .write_options()
        .write(&mut buffer, array.to_array_stream())
        .await?;
    Ok(File {
        buffer: buffer.freeze(),
        dtype,
    })
}

impl File {
    /// Scan the file, applying an optional filter and projection.
    async fn scan(
        &self,
        filter: Option<Expression>,
        projection: Option<Expression>,
    ) -> VortexResult<ArrayRef> {
        let mut scan = SESSION
            .open_options()
            .open_buffer(self.buffer.clone())?
            .scan()?;
        if let Some(filter) = filter {
            scan = scan.with_filter(filter);
        }
        if let Some(projection) = projection {
            scan = scan.with_projection(projection);
        }
        scan.into_array_stream()?.read_all().await
    }

    /// Scan the whole file, unprojected and unfiltered.
    async fn read_all(&self) -> VortexResult<ArrayRef> {
        self.scan(None, None).await
    }

    /// Scan the file with a projection.
    async fn project(&self, projection: Expression) -> VortexResult<ArrayRef> {
        self.scan(None, Some(projection)).await
    }

    /// Scan the file with a filter, projecting the surviving rows.
    async fn filter(&self, filter: Expression, projection: Expression) -> VortexResult<ArrayRef> {
        self.scan(Some(filter), Some(projection)).await
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }
}

fn i32_(nullability: Nullability) -> DType {
    DType::Primitive(PType::I32, nullability)
}

/// The dtype of the nullable struct column: `{a: i32, b: i32}?`.
fn abc(nullability: Nullability) -> DType {
    DType::Struct(
        StructFields::from_iter([
            ("a", i32_(Nullability::NonNullable)),
            ("b", i32_(Nullability::NonNullable)),
        ]),
        nullability,
    )
}

/// One chunk of `{s: {a: i32, b: i32}?}`, where `s` is null wherever `valid` is false.
fn chunk(field_a: [i32; 3], field_b: [i32; 3], valid: [bool; 3]) -> ArrayRef {
    let nullable_struct = StructArray::try_from_iter_with_validity(
        [
            ("a", Buffer::copy_from(field_a).into_array()),
            ("b", Buffer::copy_from(field_b).into_array()),
        ],
        Validity::Array(BoolArray::from_iter(valid).into_array()),
    )
    .vortex_expect("chunk fields are the same length")
    .into_array();

    StructArray::from_fields([("s", nullable_struct)].as_slice())
        .vortex_expect("a single-field struct is well formed")
        .into_array()
}

/// A two-chunk file whose single column `s` is a nullable struct.
///
/// | row | s              |
/// |-----|----------------|
/// | 0   | `NULL`         |
/// | 1   | `{a: 2, b: 5}` |
/// | 2   | `{a: 3, b: 6}` |
/// | 3   | `{a: 8, b: 9}` |
/// | 4   | `NULL`         |
/// | 5   | `{a: 4, b: 1}` |
#[fixture]
fn nullable_column() -> ArrayRef {
    ChunkedArray::from_iter([
        chunk([7, 2, 3], [4, 5, 6], [false, true, true]),
        chunk([8, 5, 4], [9, 3, 1], [true, false, true]),
    ])
    .into_array()
}

/// A file whose column `a` holds a nested nullable struct: `{a: {b: {c: i32}?}}`, where `a.b` is
/// null at row 1.
#[fixture]
fn nested_column() -> ArrayRef {
    let inner = StructArray::try_from_iter_with_validity(
        [("c", buffer![4, 5, 6].into_array())],
        Validity::NonNullable,
    )
    .vortex_expect("a single-field struct is well formed")
    .into_array();

    let nullable_struct = StructArray::try_from_iter_with_validity(
        [("b", inner)],
        Validity::Array(BoolArray::from_iter([true, false, true]).into_array()),
    )
    .vortex_expect("validity matches the field length")
    .into_array();

    StructArray::from_fields([("a", nullable_struct)].as_slice())
        .vortex_expect("a single-field struct is well formed")
        .into_array()
}

/// The `s` column of a scan result, as a struct array.
fn column(array: &ArrayRef, name: &str, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    Ok(array
        .clone()
        .execute::<StructArray>(ctx)?
        .unmasked_field_by_name(name)?
        .clone())
}

// ---- Round trip ----

/// Reading a file back must reproduce the DType it was written with, nullable struct column and
/// all — the nullability must not migrate into the struct's fields.
#[rstest]
#[tokio::test]
async fn round_trips_dtype(nullable_column: ArrayRef) -> VortexResult<()> {
    let file = write(nullable_column).await?;
    let result = file.read_all().await?;

    assert_eq!(result.dtype(), file.dtype());
    assert_eq!(result.len(), 6);

    let mut ctx = ctx();
    let s = column(&result, "s", &mut ctx)?;
    assert_eq!(s.dtype(), &abc(Nullability::Nullable));
    for (row, expected_null) in [true, false, false, false, true, false]
        .into_iter()
        .enumerate()
    {
        assert_eq!(
            s.execute_scalar(row, &mut ctx)?.is_null(),
            expected_null,
            "row {row}"
        );
    }
    Ok(())
}

// ---- Projections ----

/// Projecting the nullable struct column keeps the struct itself nullable.
#[rstest]
#[tokio::test]
async fn project_nullable_struct(nullable_column: ArrayRef) -> VortexResult<()> {
    let file = write(nullable_column).await?;
    let result = file.project(select(["s"], root())).await?;

    let mut ctx = ctx();
    let s = column(&result, "s", &mut ctx)?;
    assert_eq!(s.dtype(), &abc(Nullability::Nullable));
    assert!(s.execute_scalar(0, &mut ctx)?.is_null());
    assert!(!s.execute_scalar(1, &mut ctx)?.is_null());
    Ok(())
}

/// `select` over the nullable struct keeps its nullability at the struct level rather than
/// pushing it into the selected fields.
#[rstest]
#[tokio::test]
async fn select_from_nullable_struct(nullable_column: ArrayRef) -> VortexResult<()> {
    let file = write(nullable_column).await?;
    let expr = pack([("s", select(["a"], col("s")))], Nullability::NonNullable);
    let result = file.project(expr).await?;

    let mut ctx = ctx();
    let s = column(&result, "s", &mut ctx)?;
    assert_eq!(
        s.dtype(),
        &DType::Struct(
            StructFields::from_iter([("a", i32_(Nullability::NonNullable))]),
            Nullability::Nullable,
        )
    );
    assert!(s.execute_scalar(0, &mut ctx)?.is_null());
    assert!(!s.execute_scalar(1, &mut ctx)?.is_null());
    Ok(())
}

/// `pack` of `get_item`s does the opposite: the struct's nullability lands in the packed fields
/// and the new struct is non-nullable.
#[rstest]
#[tokio::test]
async fn pack_from_nullable_struct(nullable_column: ArrayRef) -> VortexResult<()> {
    let file = write(nullable_column).await?;
    let expr = pack(
        [(
            "s",
            pack(
                [
                    ("a", get_item("a", col("s"))),
                    ("b", get_item("b", col("s"))),
                ],
                Nullability::NonNullable,
            ),
        )],
        Nullability::NonNullable,
    );
    let result = file.project(expr).await?;

    let mut ctx = ctx();
    let s = column(&result, "s", &mut ctx)?;
    assert_eq!(
        s.dtype(),
        &DType::Struct(
            StructFields::from_iter([
                ("a", i32_(Nullability::Nullable)),
                ("b", i32_(Nullability::Nullable)),
            ]),
            Nullability::NonNullable,
        )
    );
    assert!(!s.execute_scalar(0, &mut ctx)?.is_null());
    assert!(
        column(&s, "a", &mut ctx)?
            .execute_scalar(0, &mut ctx)?
            .is_null()
    );
    Ok(())
}

/// `get_item` through the nullable struct intersects the struct's validity with the field's, so
/// rows where the struct is null read as null.
#[rstest]
#[tokio::test]
async fn get_item_through_nullable_struct(nullable_column: ArrayRef) -> VortexResult<()> {
    let file = write(nullable_column).await?;
    let expr = pack([("a", get_item("a", col("s")))], Nullability::NonNullable);
    let result = file.project(expr).await?;

    let mut ctx = ctx();
    let a = column(&result, "a", &mut ctx)?;
    assert_eq!(a.dtype(), &i32_(Nullability::Nullable));
    assert!(a.execute_scalar(0, &mut ctx)?.is_null());
    assert!(a.execute_scalar(4, &mut ctx)?.is_null());
    assert!(!a.execute_scalar(1, &mut ctx)?.is_null());
    Ok(())
}

/// `is_null` / `is_not_null` of the struct column must consult the struct's own validity, which
/// the layout stores as a child of its own.
#[rstest]
#[case(false, [true, false, false, false, true, false])]
#[case(true, [false, true, true, true, false, true])]
#[tokio::test]
async fn null_checks(
    nullable_column: ArrayRef,
    #[case] negate: bool,
    #[case] expected: [bool; 6],
) -> VortexResult<()> {
    let file = write(nullable_column).await?;
    let check = if negate {
        is_not_null(col("s"))
    } else {
        is_null(col("s"))
    };
    let result = file
        .project(pack([("n", check)], Nullability::NonNullable))
        .await?;

    let mut ctx = ctx();
    let n = column(&result, "n", &mut ctx)?;
    assert_arrays_eq!(n, BoolArray::from_iter(expected), &mut ctx);
    Ok(())
}

/// Projecting a nested nullable struct preserves that struct's own nullability.
#[rstest]
#[tokio::test]
async fn project_nested_nullable_struct(nested_column: ArrayRef) -> VortexResult<()> {
    let file = write(nested_column).await?;
    let result = file.read_all().await?;
    assert_eq!(result.dtype(), file.dtype());

    let expr = pack(
        [("c", select(["c"], get_item("b", col("a"))))],
        Nullability::NonNullable,
    );
    let result = file.project(expr).await?;

    let mut ctx = ctx();
    let c = column(&result, "c", &mut ctx)?;
    assert_eq!(
        c.dtype(),
        &DType::Struct(
            StructFields::from_iter([("c", i32_(Nullability::NonNullable))]),
            Nullability::Nullable,
        )
    );
    assert!(!c.execute_scalar(0, &mut ctx)?.is_null());
    assert!(c.execute_scalar(1, &mut ctx)?.is_null());
    assert!(!c.execute_scalar(2, &mut ctx)?.is_null());
    Ok(())
}

// ---- Filters ----

/// Filtering on the struct column's own validity.
#[rstest]
#[case(false, 2)]
#[case(true, 4)]
#[tokio::test]
async fn filter_on_null_check(
    nullable_column: ArrayRef,
    #[case] negate: bool,
    #[case] expected_rows: usize,
) -> VortexResult<()> {
    let file = write(nullable_column).await?;
    let filter = if negate {
        is_not_null(col("s"))
    } else {
        is_null(col("s"))
    };
    let result = file.filter(filter, select(["s"], root())).await?;

    assert_eq!(result.len(), expected_rows);

    // Every surviving row is null exactly when we filtered for nulls.
    let mut ctx = ctx();
    let s = column(&result, "s", &mut ctx)?;
    for row in 0..s.len() {
        assert_eq!(
            s.execute_scalar(row, &mut ctx)?.is_null(),
            !negate,
            "row {row}"
        );
    }
    Ok(())
}

/// Rows 0 and 3 both hold `a == 7 || a == 8`, but row 0's struct is null, so the comparison there
/// is null and the row must not survive the filter.
#[rstest]
#[tokio::test]
async fn filter_field_of_null_struct(nullable_column: ArrayRef) -> VortexResult<()> {
    let file = write(nullable_column).await?;
    let result = file
        .filter(eq(get_item("a", col("s")), lit(7)), select(["s"], root()))
        .await?;
    assert_eq!(result.len(), 0, "row 0 has a == 7 but its struct is null");

    let result = file
        .filter(eq(get_item("a", col("s")), lit(8)), select(["s"], root()))
        .await?;
    assert_eq!(result.len(), 1);

    let mut ctx = ctx();
    let a = column(&column(&result, "s", &mut ctx)?, "a", &mut ctx)?;
    assert_eq!(
        a.execute_scalar(0, &mut ctx)?,
        vortex_array::scalar::Scalar::primitive(8, Nullability::NonNullable)
    );
    Ok(())
}
