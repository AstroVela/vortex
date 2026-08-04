// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Microbenchmark for the incremental cost of native geometry validation during Arrow import.
//!
//! Run with `cargo bench -p vortex-geo --bench coordinate_validation`.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use arrow_array::Array as ArrowArray;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::Field;
use divan::Bencher;
use divan::counter::BytesCount;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ExtensionArray;
use vortex_array::dtype::DType;
use vortex_arrow::ArrowSessionExt;
use vortex_arrow::FromArrowArray;
use vortex_error::VortexResult;
use vortex_geo::test_harness::MultiPolygonRings;
use vortex_geo::test_harness::geo_session;
use vortex_geo::test_harness::multipoint_column;
use vortex_geo::test_harness::multipolygon_column;
use vortex_geo::test_harness::nullable_multipolygon_column;
use vortex_geo::test_harness::nullable_point_column;
use vortex_geo::test_harness::point_column;
use vortex_geo::test_harness::rect_column;
use vortex_geo::test_harness::validate_list_geometry;
use vortex_geo::test_harness::validate_point;
use vortex_geo::test_harness::validate_rect;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(geo_session);

const COORDINATES: usize = 1 << 20;
const COORDINATES_PER_ROW: usize = 32;
const NESTED_ROWS: usize = COORDINATES / COORDINATES_PER_ROW;
const VERTICES_PER_RING: usize = 8;

fn ordinate(index: usize) -> f64 {
    (index.wrapping_mul(2654435761) % 1000) as f64
}

struct ImportCase {
    array: ArrowArrayRef,
    field: Field,
    dtype: DType,
}

#[derive(Debug, Clone, Copy)]
enum Validation {
    Without,
    With,
}

type Validator = fn(&dyn ArrowArray) -> VortexResult<()>;

fn to_arrow(array: ArrayRef) -> ImportCase {
    let mut ctx = SESSION.create_execution_ctx();
    let field = SESSION
        .arrow()
        .to_arrow_field("geometry", array.dtype())
        .unwrap();
    let arrow = SESSION
        .arrow()
        .execute_arrow(array, Some(&field), &mut ctx)
        .unwrap();
    let dtype = SESSION.arrow().from_arrow_field(&field).unwrap();
    ImportCase {
        array: arrow,
        field,
        dtype,
    }
}

fn import_native(case: &ImportCase, validation: Validation, validate: Validator) -> ArrayRef {
    if matches!(validation, Validation::With) {
        validate(case.array.as_ref()).unwrap();
    }

    // Both modes perform the same zero-copy import and extension wrapping. `Without` deliberately
    // skips reading the coordinate buffers, so its reported throughput is not memory bandwidth.
    let storage = ArrayRef::from_arrow(case.array.as_ref(), case.field.is_nullable()).unwrap();
    ExtensionArray::try_new(case.dtype.as_extension().clone(), storage)
        .unwrap()
        .into_array()
}

fn xy_bytes(coordinates: usize) -> BytesCount {
    BytesCount::of_many::<f64>(coordinates * 2)
}

fn multipolygon_row(row: usize) -> MultiPolygonRings {
    let ring = |part: usize| {
        (0..VERTICES_PER_RING)
            .map(|vertex| {
                (
                    ordinate(row + part + vertex),
                    ordinate(row + part + vertex + 1),
                )
            })
            .collect()
    };
    vec![vec![ring(0), ring(1)], vec![ring(2), ring(3)]]
}

#[divan::bench(args = [Validation::Without, Validation::With])]
fn point(bencher: Bencher, validation: Validation) {
    let xs = (0..COORDINATES).map(ordinate).collect();
    let ys = (0..COORDINATES).map(|index| ordinate(index + 1)).collect();
    let case = to_arrow(point_column(xs, ys).unwrap());

    bencher
        .counter(xy_bytes(COORDINATES))
        .bench(|| import_native(&case, validation, validate_point));
}

#[divan::bench(args = [Validation::Without, Validation::With])]
fn point_sparse_nulls(bencher: Bencher, validation: Validation) {
    let points: Vec<_> = (0..COORDINATES)
        .map(|index| (!index.is_multiple_of(10)).then(|| (ordinate(index), ordinate(index + 1))))
        .collect();
    let valid_points = points.iter().filter(|point| point.is_some()).count();
    let case = to_arrow(nullable_point_column(points).unwrap());

    bencher
        .counter(xy_bytes(valid_points))
        .bench(|| import_native(&case, validation, validate_point));
}

#[divan::bench(args = [Validation::Without, Validation::With])]
fn rect(bencher: Bencher, validation: Validation) {
    let boxes = (0..COORDINATES)
        .map(|index| {
            let xmin = ordinate(index);
            let ymin = ordinate(index + 1);
            (xmin, ymin, xmin + 1.0, ymin + 1.0)
        })
        .collect();
    let case = to_arrow(rect_column(boxes).unwrap());

    bencher
        .counter(BytesCount::of_many::<f64>(COORDINATES * 4))
        .bench(|| import_native(&case, validation, validate_rect));
}

#[divan::bench(args = [Validation::Without, Validation::With])]
fn multipoint(bencher: Bencher, validation: Validation) {
    let rows: Vec<_> = (0..NESTED_ROWS)
        .map(|row| {
            (0..COORDINATES_PER_ROW)
                .map(|point| (ordinate(row + point), ordinate(row + point + 1)))
                .collect()
        })
        .collect();
    let case = to_arrow(multipoint_column(rows).unwrap());

    bencher
        .counter(xy_bytes(COORDINATES))
        .bench(|| import_native(&case, validation, validate_list_geometry));
}

#[divan::bench(args = [Validation::Without, Validation::With])]
fn multipolygon(bencher: Bencher, validation: Validation) {
    let rows = (0..NESTED_ROWS).map(multipolygon_row).collect();
    let case = to_arrow(multipolygon_column(rows).unwrap());

    bencher
        .counter(xy_bytes(COORDINATES))
        .bench(|| import_native(&case, validation, validate_list_geometry));
}

#[divan::bench(args = [Validation::Without, Validation::With])]
fn multipolygon_sparse_nulls(bencher: Bencher, validation: Validation) {
    let rows: Vec<_> = (0..NESTED_ROWS)
        .map(|row| (!row.is_multiple_of(10)).then(|| multipolygon_row(row)))
        .collect();
    let valid_coordinates = rows.iter().filter(|row| row.is_some()).count() * COORDINATES_PER_ROW;
    let case = to_arrow(nullable_multipolygon_column(rows).unwrap());

    bencher
        .counter(xy_bytes(valid_coordinates))
        .bench(|| import_native(&case, validation, validate_list_geometry));
}
