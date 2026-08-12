// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Unsigned planar area.

use super::Coords;
use super::GeometryRef;
use super::PolygonRef;

/// Twice the signed shoelace area of `ring`, with every vertex shifted by the first to condition
/// the determinant sum (as `geo` does). The ring is implicitly closed: native storage does not
/// guarantee a closing vertex, and `geo_types::Polygon` - which the kernels previously decoded
/// through - closes rings on construction. A stored closing vertex contributes a zero term, so
/// closed rings compute the identical sum.
fn twice_signed_ring_area(ring: Coords<'_>) -> f64 {
    if ring.len() < 3 {
        return 0.0;
    }
    let first = ring.coord(0);
    let mut sum = 0.0;
    for index in 0..ring.len() {
        let a = ring.coord(index);
        let b = ring.coord(if index + 1 == ring.len() {
            0
        } else {
            index + 1
        });
        sum += (a.x - first.x) * (b.y - first.y) - (a.y - first.y) * (b.x - first.x);
    }
    sum
}

/// The signed area of a polygon: the exterior ring's magnitude minus each hole's magnitude,
/// carrying the exterior ring's winding sign. This is the exact shape of `geo`'s implementation,
/// including its treatment of mis-oriented holes.
fn polygon_signed_area(polygon: PolygonRef<'_>) -> f64 {
    if polygon.ring_count() == 0 {
        return 0.0;
    }
    let exterior = twice_signed_ring_area(polygon.ring(0)) / 2.0;
    let mut magnitude = exterior.abs();
    for index in 1..polygon.ring_count() {
        magnitude -= (twice_signed_ring_area(polygon.ring(index)) / 2.0).abs();
    }
    if exterior < 0.0 {
        -magnitude
    } else {
        magnitude
    }
}

/// The unsigned planar area of `geometry`, matching `geo::Area::unsigned_area` for every native
/// type: zero- and one-dimensional geometries measure zero, a multi-polygon sums its members'
/// unsigned areas, and a rectangle measures width times height.
pub(crate) fn unsigned_area(geometry: GeometryRef<'_>) -> f64 {
    match geometry {
        GeometryRef::Point(_)
        | GeometryRef::LineString(_)
        | GeometryRef::MultiPoint(_)
        | GeometryRef::MultiLineString(_) => 0.0,
        GeometryRef::Polygon(polygon) => polygon_signed_area(polygon).abs(),
        GeometryRef::MultiPolygon(multipolygon) => (0..multipolygon.polygon_count())
            .map(|index| polygon_signed_area(multipolygon.polygon(index)).abs())
            .sum(),
        GeometryRef::Rect(aabb) => aabb.width() * aabb.height(),
    }
}
