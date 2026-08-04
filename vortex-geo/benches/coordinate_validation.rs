// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Microbenchmark for validating native geometry coordinates during Arrow import.
//!
//! Run with `cargo bench -p vortex-geo --bench coordinate_validation`.

#![expect(clippy::unwrap_used)]

use std::sync::Arc;
use std::sync::LazyLock;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::Field;
use divan::Bencher;
use divan::counter::BytesCount;
use vortex_array::ArrayRef;
use vortex_array::VortexSessionExecute;
use vortex_arrow::ArrowSessionExt;
use vortex_geo::test_harness::MultiPolygonRings;
use vortex_geo::test_harness::geo_session;
use vortex_geo::test_harness::multipoint_column;
use vortex_geo::test_harness::multipolygon_column;
use vortex_geo::test_harness::nullable_multipolygon_column;
use vortex_geo::test_harness::nullable_point_column;
use vortex_geo::test_harness::point_column;
use vortex_geo::test_harness::rect_column;
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

fn to_arrow(array: ArrayRef) -> (ArrowArrayRef, Field) {
    let mut ctx = SESSION.create_execution_ctx();
    let field = SESSION
        .arrow()
        .to_arrow_field("geometry", array.dtype())
        .unwrap();
    let arrow = SESSION
        .arrow()
        .execute_arrow(array, Some(&field), &mut ctx)
        .unwrap();
    (arrow, field)
}

fn import(array: &ArrowArrayRef, field: &Field) -> ArrayRef {
    SESSION
        .arrow()
        .from_arrow_array(Arc::clone(array), field)
        .unwrap()
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

#[divan::bench]
fn point(bencher: Bencher) {
    let xs = (0..COORDINATES).map(ordinate).collect();
    let ys = (0..COORDINATES).map(|index| ordinate(index + 1)).collect();
    let (array, field) = to_arrow(point_column(xs, ys).unwrap());

    bencher
        .counter(xy_bytes(COORDINATES))
        .bench(|| import(&array, &field));
}

#[divan::bench]
fn point_sparse_nulls(bencher: Bencher) {
    let points: Vec<_> = (0..COORDINATES)
        .map(|index| (!index.is_multiple_of(10)).then(|| (ordinate(index), ordinate(index + 1))))
        .collect();
    let valid_points = points.iter().filter(|point| point.is_some()).count();
    let (array, field) = to_arrow(nullable_point_column(points).unwrap());

    bencher
        .counter(xy_bytes(valid_points))
        .bench(|| import(&array, &field));
}

#[divan::bench]
fn rect(bencher: Bencher) {
    let boxes = (0..COORDINATES)
        .map(|index| {
            let xmin = ordinate(index);
            let ymin = ordinate(index + 1);
            (xmin, ymin, xmin + 1.0, ymin + 1.0)
        })
        .collect();
    let (array, field) = to_arrow(rect_column(boxes).unwrap());

    bencher
        .counter(BytesCount::of_many::<f64>(COORDINATES * 4))
        .bench(|| import(&array, &field));
}

#[divan::bench]
fn multipoint(bencher: Bencher) {
    let rows: Vec<_> = (0..NESTED_ROWS)
        .map(|row| {
            (0..COORDINATES_PER_ROW)
                .map(|point| (ordinate(row + point), ordinate(row + point + 1)))
                .collect()
        })
        .collect();
    let (array, field) = to_arrow(multipoint_column(rows).unwrap());

    bencher
        .counter(xy_bytes(COORDINATES))
        .bench(|| import(&array, &field));
}

#[divan::bench]
fn multipolygon(bencher: Bencher) {
    let rows = (0..NESTED_ROWS).map(multipolygon_row).collect();
    let (array, field) = to_arrow(multipolygon_column(rows).unwrap());

    bencher
        .counter(xy_bytes(COORDINATES))
        .bench(|| import(&array, &field));
}

#[divan::bench]
fn multipolygon_sparse_nulls(bencher: Bencher) {
    let rows: Vec<_> = (0..NESTED_ROWS)
        .map(|row| (!row.is_multiple_of(10)).then(|| multipolygon_row(row)))
        .collect();
    let valid_coordinates = rows.iter().filter(|row| row.is_some()).count() * COORDINATES_PER_ROW;
    let (array, field) = to_arrow(nullable_multipolygon_column(rows).unwrap());

    bencher
        .counter(xy_bytes(valid_coordinates))
        .bench(|| import(&array, &field));
}
