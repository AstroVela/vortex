// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Microbenchmarks for the binary geometry predicates `ST_Contains` and `ST_Intersects`, focused
//! on the cost of a batch-constant operand.
//!
//! The constant is a 128-vertex query polygon, the shape a spatial filter broadcasts against a
//! column. Arms pair it with a point column (geo answers those pairings with direct
//! point-in-polygon algorithms) and with a small-polygon column (geo routes those pairings through
//! bounding-box prechecks and `relate`), split into a mostly-disjoint and a mostly-overlapping
//! dataset so the bbox early-out's win and its overhead are both visible. The column-x-column arms
//! are the control: no operand is constant, so a prepared path has nothing to hoist and must not
//! regress them.
//!
//! Run with `cargo bench -p vortex-geo --bench binary_predicates`.

#![expect(clippy::unwrap_used)]

use std::f64::consts::TAU;
use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_error::VortexResult;
use vortex_geo::scalar_fn::contains::GeoContains;
use vortex_geo::scalar_fn::intersects::GeoIntersects;
use vortex_geo::test_harness::geo_session;
use vortex_geo::test_harness::point_column;
use vortex_geo::test_harness::polygon_column;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(geo_session);

/// Every arm has the same row count so results are comparable across shapes.
const ROWS: usize = 1 << 14;

/// Deterministic pseudo-random value in `[0, 1)`.
fn unit(i: usize) -> f64 {
    ((i.wrapping_mul(2654435761) >> 8) % 10_000) as f64 / 10_000.0
}

/// The exterior ring of a convex 128-gon of radius 100 centered at `(cx, cy)`: enough vertices
/// that per-row work proportional to the constant's size shows up clearly.
fn query_ring(cx: f64, cy: f64) -> Vec<(f64, f64)> {
    let n = 128;
    (0..=n)
        .map(|i| {
            let theta = (i % n) as f64 / n as f64 * TAU;
            (cx + 100.0 * theta.cos(), cy + 100.0 * theta.sin())
        })
        .collect()
}

/// The query polygon as a batch-constant operand: a top-level `ConstantArray` over the geometry
/// extension scalar, the shape that reaches the row loop's stride-0 path.
fn query_constant(ctx: &mut ExecutionCtx) -> ArrayRef {
    let scalar = polygon_column(vec![vec![query_ring(0.0, 0.0)]])
        .unwrap()
        .execute_scalar(0, ctx)
        .unwrap();
    ConstantArray::new(scalar, ROWS).into_array()
}

/// A small square (side 2) centered at `(cx, cy)`.
fn square(cx: f64, cy: f64) -> Vec<Vec<(f64, f64)>> {
    vec![vec![
        (cx - 1.0, cy - 1.0),
        (cx + 1.0, cy - 1.0),
        (cx + 1.0, cy + 1.0),
        (cx - 1.0, cy + 1.0),
        (cx - 1.0, cy - 1.0),
    ]]
}

/// [`ROWS`] small squares whose centers avoid the query polygon almost always: the shape of a
/// selective spatial filter, where a bbox check rejects nearly every row.
fn squares_mostly_disjoint() -> ArrayRef {
    let rows = (0..ROWS)
        .map(|i| square(150.0 + 700.0 * unit(i), 150.0 + 700.0 * unit(i + 1)))
        .collect();
    polygon_column(rows).unwrap()
}

/// [`ROWS`] small squares whose centers all fall well inside the query polygon, so a bbox check
/// never rejects and the full pairwise predicate always runs.
fn squares_mostly_overlapping() -> ArrayRef {
    let rows = (0..ROWS)
        .map(|i| square(120.0 * unit(i) - 60.0, 120.0 * unit(i + 1) - 60.0))
        .collect();
    polygon_column(rows).unwrap()
}

/// [`ROWS`] points spread over `[-150, 150)^2`, mixing rows inside and outside the query polygon.
fn points() -> ArrayRef {
    let xs = (0..ROWS).map(|i| 300.0 * unit(i) - 150.0).collect();
    let ys = (0..ROWS).map(|i| 300.0 * unit(i + 1) - 150.0).collect();
    point_column(xs, ys).unwrap()
}

/// Execute `array` to completion.
fn execute(array: VortexResult<ScalarFnArray>, ctx: &mut ExecutionCtx) -> ArrayRef {
    array
        .unwrap()
        .into_array()
        .execute::<Canonical>(ctx)
        .unwrap()
        .into_array()
}

mod contains {
    use super::*;

    /// Control: no constant operand, direct point-in-polygon per row.
    #[divan::bench]
    fn column_x_column_points(bencher: Bencher) {
        let mut ctx = SESSION.create_execution_ctx();
        let polygons = squares_mostly_overlapping();
        let points = points();
        bencher.counter(ItemsCount::new(ROWS)).bench_local(|| {
            execute(
                GeoContains::try_new_array(polygons.clone(), points.clone()),
                &mut ctx,
            )
        });
    }

    /// Control: no constant operand, relate-routed polygon pairs per row.
    #[divan::bench]
    fn column_x_column_polygons(bencher: Bencher) {
        let mut ctx = SESSION.create_execution_ctx();
        let a = squares_mostly_overlapping();
        let b = squares_mostly_disjoint();
        bencher
            .counter(ItemsCount::new(ROWS))
            .bench_local(|| execute(GeoContains::try_new_array(a.clone(), b.clone()), &mut ctx));
    }

    /// Constant container against a point column: geo's direct point-in-polygon pairing.
    #[divan::bench]
    fn constant_x_points(bencher: Bencher) {
        let mut ctx = SESSION.create_execution_ctx();
        let query = query_constant(&mut ctx);
        let points = points();
        bencher.counter(ItemsCount::new(ROWS)).bench_local(|| {
            execute(
                GeoContains::try_new_array(query.clone(), points.clone()),
                &mut ctx,
            )
        });
    }

    /// Constant container against mostly-disjoint polygons: relate-routed, and almost every row
    /// short-circuits on bounding boxes inside relate.
    #[divan::bench]
    fn constant_x_polygons_disjoint(bencher: Bencher) {
        let mut ctx = SESSION.create_execution_ctx();
        let query = query_constant(&mut ctx);
        let polygons = squares_mostly_disjoint();
        bencher.counter(ItemsCount::new(ROWS)).bench_local(|| {
            execute(
                GeoContains::try_new_array(query.clone(), polygons.clone()),
                &mut ctx,
            )
        });
    }

    /// Constant container against mostly-overlapping polygons: relate-routed, and every row pays
    /// for topology graphs.
    #[divan::bench]
    fn constant_x_polygons_overlapping(bencher: Bencher) {
        let mut ctx = SESSION.create_execution_ctx();
        let query = query_constant(&mut ctx);
        let polygons = squares_mostly_overlapping();
        bencher.counter(ItemsCount::new(ROWS)).bench_local(|| {
            execute(
                GeoContains::try_new_array(query.clone(), polygons.clone()),
                &mut ctx,
            )
        });
    }
}

mod intersects {
    use super::*;

    /// Control: no constant operand, polygon pairs per row.
    #[divan::bench]
    fn column_x_column_polygons(bencher: Bencher) {
        let mut ctx = SESSION.create_execution_ctx();
        let a = squares_mostly_overlapping();
        let b = squares_mostly_disjoint();
        bencher
            .counter(ItemsCount::new(ROWS))
            .bench_local(|| execute(GeoIntersects::try_new_array(a.clone(), b.clone()), &mut ctx));
    }

    /// Point column against the constant query: geo answers point-x-polygon directly, with no
    /// bbox precheck to hoist.
    #[divan::bench]
    fn points_x_constant(bencher: Bencher) {
        let mut ctx = SESSION.create_execution_ctx();
        let points = points();
        let query = query_constant(&mut ctx);
        bencher.counter(ItemsCount::new(ROWS)).bench_local(|| {
            execute(
                GeoIntersects::try_new_array(points.clone(), query.clone()),
                &mut ctx,
            )
        });
    }

    /// Mostly-disjoint polygons against the constant query: the bbox precheck rejects nearly
    /// every row, so the constant's per-row bounding-box fold dominates the baseline.
    #[divan::bench]
    fn polygons_disjoint_x_constant(bencher: Bencher) {
        let mut ctx = SESSION.create_execution_ctx();
        let polygons = squares_mostly_disjoint();
        let query = query_constant(&mut ctx);
        bencher.counter(ItemsCount::new(ROWS)).bench_local(|| {
            execute(
                GeoIntersects::try_new_array(polygons.clone(), query.clone()),
                &mut ctx,
            )
        });
    }

    /// Mostly-overlapping polygons against the constant query: the bbox precheck never rejects,
    /// so every row still pays for the full pairwise predicate.
    #[divan::bench]
    fn polygons_overlapping_x_constant(bencher: Bencher) {
        let mut ctx = SESSION.create_execution_ctx();
        let polygons = squares_mostly_overlapping();
        let query = query_constant(&mut ctx);
        bencher.counter(ItemsCount::new(ROWS)).bench_local(|| {
            execute(
                GeoIntersects::try_new_array(polygons.clone(), query.clone()),
                &mut ctx,
            )
        });
    }
}
