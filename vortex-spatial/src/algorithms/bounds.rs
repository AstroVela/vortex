// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Axis-aligned bounding boxes.

use super::Aabb;
use super::GeometryRef;
use super::PolygonRef;

/// The corners of the box containing the `(xs, ys)` coordinates, in
/// `[xmin, ymin, xmax, ymax]` order. Inverted (infinite) corners when the slices are empty,
/// since `f64::min`/`max` also skip any NaN ordinates.
pub(crate) fn box_corners(xs: &[f64], ys: &[f64]) -> [f64; 4] {
    let (mut xmin, mut ymin) = (f64::INFINITY, f64::INFINITY);
    let (mut xmax, mut ymax) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for (&x, &y) in xs.iter().zip(ys) {
        xmin = xmin.min(x);
        ymin = ymin.min(y);
        xmax = xmax.max(x);
        ymax = ymax.max(y);
    }
    [xmin, ymin, xmax, ymax]
}

/// The [`Aabb`] of the raw ordinate slices, or `None` when they are empty. All-NaN coordinates
/// normalize to the whole plane via [`Aabb::new`], so such rows are never pruned.
pub(crate) fn coords_aabb(xs: &[f64], ys: &[f64]) -> Option<Aabb> {
    (!xs.is_empty()).then(|| {
        let [xmin, ymin, xmax, ymax] = box_corners(xs, ys);
        Aabb::new(xmin, ymin, xmax, ymax)
    })
}

/// The 2-D bounding box of `geometry`, or `None` for one without an extent (an empty geometry).
/// Matches `geo::BoundingRect` for every native type; in particular a polygon's box is its
/// exterior ring's box alone, since the holes of a valid polygon lie inside it.
pub(crate) fn bounding_rect(geometry: GeometryRef<'_>) -> Option<Aabb> {
    match geometry {
        GeometryRef::Point(coord) => Some(Aabb::new(coord.x, coord.y, coord.x, coord.y)),
        GeometryRef::Rect(aabb) => Some(aabb),
        GeometryRef::LineString(coords) | GeometryRef::MultiPoint(coords) => {
            coords_aabb(coords.xs(), coords.ys())
        }
        GeometryRef::Polygon(polygon) => polygon_bounding_rect(polygon),
        GeometryRef::MultiLineString(multiline) => {
            let coords = multiline.coords();
            coords_aabb(coords.xs(), coords.ys())
        }
        GeometryRef::MultiPolygon(multipolygon) => (0..multipolygon.polygon_count())
            .filter_map(|index| polygon_bounding_rect(multipolygon.polygon(index)))
            .reduce(Aabb::union),
    }
}

/// A polygon's bounding box: its exterior ring's box, `None` when the polygon or its exterior is
/// empty.
fn polygon_bounding_rect(polygon: PolygonRef<'_>) -> Option<Aabb> {
    if polygon.ring_count() == 0 {
        return None;
    }
    let exterior = polygon.ring(0);
    coords_aabb(exterior.xs(), exterior.ys())
}
