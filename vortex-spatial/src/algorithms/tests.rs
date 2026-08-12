// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Differential tests of the computation core against its `geo` oracle, over fixed cases and
//! property-based sweeps: failures shrink to a minimal counterexample and persist to
//! `proptest-regressions/`. Inputs are finite by construction, matching the core's coordinate
//! policy; the non-finite edge cases live with their kernels.

use geo::Area;
use geo::BoundingRect;
use proptest::prelude::*;
use rstest::rstest;

use super::Aabb;
use super::Coord;
use super::Coords;
use super::GeometryRef;
use super::MultiLineStringRef;
use super::MultiPolygonRef;
use super::PolygonRef;
use super::bounding_rect;
use super::coords_aabb;
use super::unsigned_area;

type Ring = Vec<(f64, f64)>;

/// An owned geometry that presents itself both as a core view (via [`Fixture::flat`]) and as its
/// `geo_types` oracle form (via [`Fixture::geo`]).
#[derive(Debug, Clone)]
enum Fixture {
    Point(f64, f64),
    LineString(Ring),
    MultiPoint(Ring),
    Polygon(Vec<Ring>),
    MultiLineString(Vec<Ring>),
    MultiPolygon(Vec<Vec<Ring>>),
    Rect(f64, f64, f64, f64),
}

/// Owned flattened storage mirroring `GeometryBatch`'s single-row layout, backing the borrowed
/// view under test.
enum Flat {
    Point(f64, f64),
    LineString(Vec<f64>, Vec<f64>),
    MultiPoint(Vec<f64>, Vec<f64>),
    Polygon(Vec<f64>, Vec<f64>, Vec<usize>),
    MultiLineString(Vec<f64>, Vec<f64>, Vec<usize>),
    MultiPolygon(Vec<f64>, Vec<f64>, Vec<usize>, Vec<usize>),
    Rect(f64, f64, f64, f64),
}

/// Flatten coordinate sequences into parallel ordinate vectors plus their vertex offsets.
fn flatten_seqs(seqs: &[Ring]) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    let mut offsets = vec![0];
    for seq in seqs {
        for &(x, y) in seq {
            xs.push(x);
            ys.push(y);
        }
        offsets.push(xs.len());
    }
    (xs, ys, offsets)
}

impl Fixture {
    fn flat(&self) -> Flat {
        match self {
            Fixture::Point(x, y) => Flat::Point(*x, *y),
            Fixture::LineString(line) => {
                let (xs, ys, _) = flatten_seqs(std::slice::from_ref(line));
                Flat::LineString(xs, ys)
            }
            Fixture::MultiPoint(points) => {
                let (xs, ys, _) = flatten_seqs(std::slice::from_ref(points));
                Flat::MultiPoint(xs, ys)
            }
            Fixture::Polygon(rings) => {
                let (xs, ys, offsets) = flatten_seqs(rings);
                Flat::Polygon(xs, ys, offsets)
            }
            Fixture::MultiLineString(lines) => {
                let (xs, ys, offsets) = flatten_seqs(lines);
                Flat::MultiLineString(xs, ys, offsets)
            }
            Fixture::MultiPolygon(polygons) => {
                let all_rings = polygons.iter().flatten().cloned().collect::<Vec<_>>();
                let (xs, ys, rings) = flatten_seqs(&all_rings);
                let mut offsets = vec![0];
                for polygon in polygons {
                    offsets.push(offsets.last().copied().unwrap_or(0) + polygon.len());
                }
                Flat::MultiPolygon(xs, ys, offsets, rings)
            }
            Fixture::Rect(x1, y1, x2, y2) => Flat::Rect(*x1, *y1, *x2, *y2),
        }
    }

    fn geo(&self) -> geo_types::Geometry<f64> {
        match self {
            Fixture::Point(x, y) => geo_types::Point::new(*x, *y).into(),
            Fixture::LineString(line) => geo_line(line).into(),
            Fixture::MultiPoint(points) => geo_types::MultiPoint::from(points.clone()).into(),
            Fixture::Polygon(rings) => geo_polygon(rings).into(),
            Fixture::MultiLineString(lines) => {
                geo_types::MultiLineString::new(lines.iter().map(geo_line).collect()).into()
            }
            Fixture::MultiPolygon(polygons) => {
                geo_types::MultiPolygon::new(polygons.iter().map(|p| geo_polygon(p)).collect())
                    .into()
            }
            Fixture::Rect(x1, y1, x2, y2) => geo_types::Rect::new((*x1, *y1), (*x2, *y2)).into(),
        }
    }
}

impl Flat {
    fn view(&self) -> GeometryRef<'_> {
        match self {
            Flat::Point(x, y) => GeometryRef::Point(Coord { x: *x, y: *y }),
            Flat::LineString(xs, ys) => GeometryRef::LineString(Coords::new(xs, ys)),
            Flat::MultiPoint(xs, ys) => GeometryRef::MultiPoint(Coords::new(xs, ys)),
            Flat::Polygon(xs, ys, rings) => GeometryRef::Polygon(PolygonRef::new(xs, ys, rings)),
            Flat::MultiLineString(xs, ys, lines) => {
                GeometryRef::MultiLineString(MultiLineStringRef::new(xs, ys, lines))
            }
            Flat::MultiPolygon(xs, ys, polygons, rings) => {
                GeometryRef::MultiPolygon(MultiPolygonRef::new(xs, ys, polygons, rings))
            }
            Flat::Rect(x1, y1, x2, y2) => GeometryRef::Rect(Aabb::new(*x1, *y1, *x2, *y2)),
        }
    }
}

fn geo_line(ring: &Ring) -> geo_types::LineString<f64> {
    geo_types::LineString::from(ring.clone())
}

fn geo_polygon(rings: &[Ring]) -> geo_types::Polygon<f64> {
    let exterior = rings
        .first()
        .map(geo_line)
        .unwrap_or_else(|| geo_types::LineString::new(Vec::new()));
    geo_types::Polygon::new(exterior, rings.iter().skip(1).map(geo_line).collect())
}

/// Both algorithms agree with their `geo` oracle on `fixture`. Equality is exact: on finite
/// inputs each implementation performs the same floating-point operations.
fn assert_matches_geo(fixture: &Fixture) {
    let flat = fixture.flat();
    let oracle = fixture.geo();
    assert_eq!(
        unsigned_area(flat.view()),
        oracle.unsigned_area(),
        "area mismatch for {fixture:?}"
    );
    let expected = oracle
        .bounding_rect()
        .map(|rect| Aabb::new(rect.min().x, rect.min().y, rect.max().x, rect.max().y));
    assert_eq!(
        bounding_rect(flat.view()),
        expected,
        "bounding rect mismatch for {fixture:?}"
    );
}

#[rstest]
#[case::point(Fixture::Point(1.5, -2.0))]
#[case::empty_linestring(Fixture::LineString(vec![]))]
#[case::linestring(Fixture::LineString(vec![(0.0, 0.0), (3.0, 4.0), (-1.0, 2.0)]))]
#[case::multipoint(Fixture::MultiPoint(vec![(2.0, 1.0), (-5.0, 3.0)]))]
#[case::square_with_hole(Fixture::Polygon(vec![
    vec![(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0), (0.0, 0.0)],
    vec![(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0), (1.0, 1.0)],
]))]
#[case::clockwise_exterior(Fixture::Polygon(vec![
    vec![(0.0, 0.0), (0.0, 3.0), (4.0, 3.0), (4.0, 0.0), (0.0, 0.0)],
]))]
#[case::unclosed_ring_is_implicitly_closed(Fixture::Polygon(vec![
    vec![(0.0, 0.0), (4.0, 0.0), (4.0, 3.0)],
]))]
#[case::hole_larger_than_exterior(Fixture::Polygon(vec![
    vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)],
    vec![(-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0), (-5.0, -5.0)],
]))]
#[case::empty_polygon(Fixture::Polygon(vec![]))]
#[case::multilinestring(Fixture::MultiLineString(vec![
    vec![(0.0, 0.0), (1.0, 1.0)],
    vec![(2.0, 2.0), (3.0, 3.0)],
]))]
#[case::multipolygon_mixed_winding(Fixture::MultiPolygon(vec![
    vec![vec![(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (0.0, 0.0)]],
    vec![vec![(3.0, 0.0), (3.0, 3.0), (6.0, 3.0), (6.0, 0.0), (3.0, 0.0)]],
]))]
#[case::empty_multipolygon(Fixture::MultiPolygon(vec![]))]
#[case::rect(Fixture::Rect(0.0, 0.0, 5.0, 3.0))]
#[case::inverted_rect_corners(Fixture::Rect(5.0, 3.0, 0.0, 0.0))]
fn matches_geo_on_fixed_cases(#[case] fixture: Fixture) {
    assert_matches_geo(&fixture);
}

/// A finite ordinate; the bounded range never produces NaN, matching the core's coordinate
/// policy.
fn ordinate() -> impl Strategy<Value = f64> {
    -100.0..100.0f64
}

/// An open path of up to eight vertices, possibly empty.
fn path() -> impl Strategy<Value = Ring> {
    prop::collection::vec((ordinate(), ordinate()), 0..=8)
}

/// A ring of at least three vertices, mostly closed but occasionally left unclosed to exercise
/// implicit ring closure (the oracle's `geo_types::Polygon` closes rings on construction).
fn ring() -> impl Strategy<Value = Ring> {
    (
        prop::collection::vec((ordinate(), ordinate()), 3..=8),
        0u8..8,
    )
        .prop_map(|(mut ring, unclosed)| {
            if unclosed != 0 {
                ring.push(ring[0]);
            }
            ring
        })
}

/// One to three rings: an exterior plus up to two holes.
fn rings() -> impl Strategy<Value = Vec<Ring>> {
    prop::collection::vec(ring(), 1..=3)
}

proptest! {
    #[test]
    fn matches_geo_on_points(x in ordinate(), y in ordinate()) {
        assert_matches_geo(&Fixture::Point(x, y));
    }

    #[test]
    fn matches_geo_on_linestrings(line in path()) {
        assert_matches_geo(&Fixture::LineString(line));
    }

    #[test]
    fn matches_geo_on_multipoints(points in path()) {
        assert_matches_geo(&Fixture::MultiPoint(points));
    }

    #[test]
    fn matches_geo_on_polygons(polygon in rings()) {
        assert_matches_geo(&Fixture::Polygon(polygon));
    }

    #[test]
    fn matches_geo_on_multilinestrings(lines in prop::collection::vec(path(), 0..=3)) {
        assert_matches_geo(&Fixture::MultiLineString(lines));
    }

    #[test]
    fn matches_geo_on_multipolygons(polygons in prop::collection::vec(rings(), 0..=3)) {
        assert_matches_geo(&Fixture::MultiPolygon(polygons));
    }

    #[test]
    fn matches_geo_on_rects(
        x1 in ordinate(),
        y1 in ordinate(),
        x2 in ordinate(),
        y2 in ordinate(),
    ) {
        assert_matches_geo(&Fixture::Rect(x1, y1, x2, y2));
    }
}

/// `Aabb::new` accepts corners in any order.
#[test]
fn aabb_normalizes_inverted_corners() {
    assert_eq!(Aabb::new(5.0, 3.0, 0.0, 1.0), Aabb::new(0.0, 1.0, 5.0, 3.0));
}

/// The inverted infinite corners of an all-NaN fold normalize to the whole plane, so such rows
/// are kept by every box proof (they can never be proven absent).
#[test]
fn all_nan_coords_aabb_is_whole_plane() {
    let nans = [f64::NAN, f64::NAN];
    let aabb = coords_aabb(&nans, &nans).expect("non-empty slices have a box");
    assert_eq!(
        aabb,
        Aabb::new(
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::INFINITY
        )
    );
}

/// `union` covers both boxes.
#[test]
fn aabb_union_covers_both() {
    let union = Aabb::new(0.0, 0.0, 1.0, 1.0).union(Aabb::new(5.0, -2.0, 7.0, 3.0));
    assert_eq!(union, Aabb::new(0.0, -2.0, 7.0, 3.0));
}
