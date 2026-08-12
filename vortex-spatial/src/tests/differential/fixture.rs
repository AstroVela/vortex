// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The data model of a generated case.
//!
//! A case is described by plain owned values that belong to neither Vortex nor `geo`, so one
//! generated value can be materialized on both sides of the comparison:
//! [`GeometryColumn::to_array`] builds the native column the kernel executes over, and
//! [`GeometryColumn::oracle_rows`] builds the per-row `geo` values the result must equal.

use geo_types::Geometry;
use vortex_array::ArrayRef;
use vortex_error::VortexResult;

use crate::test_harness::linestring_column;
use crate::test_harness::multilinestring_column;
use crate::test_harness::multipoint_column;
use crate::test_harness::multipolygon_column;
use crate::test_harness::point_column;
use crate::test_harness::polygon_column;
use crate::test_harness::rect_column;

/// A row of `(x, y)` vertices: an open path or a polygon ring.
pub(super) type Vertices = Vec<(f64, f64)>;

/// The native geometry families a generated column can hold.
#[derive(Debug, Clone, Copy)]
pub(super) enum Family {
    Point,
    LineString,
    MultiPoint,
    Polygon,
    MultiLineString,
    MultiPolygon,
    Rect,
}

/// Every family, for strategies that pick one at random.
pub(super) const FAMILIES: [Family; 7] = [
    Family::Point,
    Family::LineString,
    Family::MultiPoint,
    Family::Polygon,
    Family::MultiLineString,
    Family::MultiPolygon,
    Family::Rect,
];

/// One owned geometry.
#[derive(Debug, Clone)]
pub(super) enum Fixture {
    Point(f64, f64),
    LineString(Vertices),
    MultiPoint(Vertices),
    /// Exterior ring first, then holes.
    Polygon(Vec<Vertices>),
    MultiLineString(Vec<Vertices>),
    MultiPolygon(Vec<Vec<Vertices>>),
    /// Two corner points, `(x1, y1, x2, y2)`.
    Rect(f64, f64, f64, f64),
}

impl Fixture {
    /// The `geo` oracle value, built through the same `geo_types` constructors the kernels'
    /// decode path uses (notably `Polygon::new`, which closes rings), so the oracle sees
    /// exactly the geometry the kernel sees.
    pub(super) fn oracle(&self) -> Geometry<f64> {
        match self {
            Fixture::Point(x, y) => geo_types::Point::new(*x, *y).into(),
            Fixture::LineString(line) => oracle_line(line).into(),
            Fixture::MultiPoint(points) => geo_types::MultiPoint::from(points.clone()).into(),
            Fixture::Polygon(rings) => oracle_polygon(rings).into(),
            Fixture::MultiLineString(lines) => {
                geo_types::MultiLineString::new(lines.iter().map(oracle_line).collect()).into()
            }
            Fixture::MultiPolygon(polygons) => {
                geo_types::MultiPolygon::new(polygons.iter().map(|p| oracle_polygon(p)).collect())
                    .into()
            }
            Fixture::Rect(x1, y1, x2, y2) => geo_types::Rect::new((*x1, *y1), (*x2, *y2)).into(),
        }
    }
}

fn oracle_line(vertices: &Vertices) -> geo_types::LineString<f64> {
    geo_types::LineString::from(vertices.clone())
}

fn oracle_polygon(rings: &[Vertices]) -> geo_types::Polygon<f64> {
    let exterior = rings
        .first()
        .map(oracle_line)
        .unwrap_or_else(|| geo_types::LineString::new(Vec::new()));
    geo_types::Polygon::new(exterior, rings.iter().skip(1).map(oracle_line).collect())
}

/// A generated geometry column: one family, one geometry per row.
#[derive(Debug)]
pub(super) struct GeometryColumn {
    pub(super) family: Family,
    pub(super) rows: Vec<Fixture>,
}

impl GeometryColumn {
    /// Materialize as a native column.
    pub(super) fn to_array(&self) -> VortexResult<ArrayRef> {
        match self.family {
            Family::Point => {
                let (xs, ys) = self
                    .rows
                    .iter()
                    .map(|row| match row {
                        Fixture::Point(x, y) => (*x, *y),
                        _ => unreachable!("column rows share one family"),
                    })
                    .unzip();
                point_column(xs, ys)
            }
            Family::LineString => linestring_column(
                self.rows
                    .iter()
                    .map(|row| match row {
                        Fixture::LineString(line) => line.clone(),
                        _ => unreachable!("column rows share one family"),
                    })
                    .collect(),
            ),
            Family::MultiPoint => multipoint_column(
                self.rows
                    .iter()
                    .map(|row| match row {
                        Fixture::MultiPoint(points) => points.clone(),
                        _ => unreachable!("column rows share one family"),
                    })
                    .collect(),
            ),
            Family::Polygon => polygon_column(
                self.rows
                    .iter()
                    .map(|row| match row {
                        Fixture::Polygon(rings) => rings.clone(),
                        _ => unreachable!("column rows share one family"),
                    })
                    .collect(),
            ),
            Family::MultiLineString => multilinestring_column(
                self.rows
                    .iter()
                    .map(|row| match row {
                        Fixture::MultiLineString(lines) => lines.clone(),
                        _ => unreachable!("column rows share one family"),
                    })
                    .collect(),
            ),
            Family::MultiPolygon => multipolygon_column(
                self.rows
                    .iter()
                    .map(|row| match row {
                        Fixture::MultiPolygon(polygons) => polygons.clone(),
                        _ => unreachable!("column rows share one family"),
                    })
                    .collect(),
            ),
            Family::Rect => rect_column(
                self.rows
                    .iter()
                    .map(|row| match row {
                        Fixture::Rect(x1, y1, x2, y2) => (*x1, *y1, *x2, *y2),
                        _ => unreachable!("column rows share one family"),
                    })
                    .collect(),
            ),
        }
    }

    /// The oracle-side rows.
    pub(super) fn oracle_rows(&self) -> Vec<Geometry<f64>> {
        self.rows.iter().map(Fixture::oracle).collect()
    }
}

/// Which binary operand, if any, is collapsed to a constant taken from one of its rows.
#[derive(Debug, Clone, Copy)]
pub(super) enum ConstSide {
    Neither,
    Left(usize),
    Right(usize),
}

/// A generated binary invocation: two equal-length columns, one side optionally constant.
#[derive(Debug)]
pub(super) struct BinaryInput {
    pub(super) a: GeometryColumn,
    pub(super) b: GeometryColumn,
    pub(super) constant: ConstSide,
}
