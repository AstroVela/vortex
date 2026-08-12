// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The planar geometry computation core behind the spatial functions.
//!
//! Algorithms here are pure functions over [`GeometryRef`] views: borrowed slices of the
//! canonical separated ordinate buffers plus list offsets, exactly as
//! [`GeometryBatch`](crate::extension::GeometryBatch) lays them out. Keeping the math over plain
//! slices keeps it free of per-row allocation, testable without an execution context, and
//! directly comparable against the `geo` crate, which the differential tests use as an oracle.
//!
//! All computation is planar and two-dimensional: `z`/`m` ordinates never participate.
//! Coordinates are assumed finite; non-finite values yield unspecified (but never panicking)
//! results, matching the ingest-time validation policy of the native geometry types.

mod area;
mod bounds;
#[cfg(test)]
mod tests;

pub(crate) use area::unsigned_area;
pub(crate) use bounds::bounding_rect;
pub(crate) use bounds::box_corners;
pub(crate) use bounds::coords_aabb;

/// A 2-D coordinate, the vertex unit of every computation. Unlike
/// [`Coordinate`](crate::extension::coordinate::Coordinate), which decodes full `z`/`m`-aware
/// storage values, this is the planar projection the algorithms operate on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Coord {
    /// The x (longitude/easting) ordinate.
    pub(crate) x: f64,
    /// The y (latitude/northing) ordinate.
    pub(crate) y: f64,
}

/// A 2-D axis-aligned bounding box with `min <= max` on both axes by construction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Aabb {
    /// The minimum x ordinate.
    pub(crate) min_x: f64,
    /// The minimum y ordinate.
    pub(crate) min_y: f64,
    /// The maximum x ordinate.
    pub(crate) max_x: f64,
    /// The maximum y ordinate.
    pub(crate) max_y: f64,
}

impl Aabb {
    /// A box over two corners given in any order, normalized so `min <= max` per axis. The
    /// comparison matches `geo::Rect::new`, which also sends the inverted infinities of an
    /// all-NaN corner fold to the whole plane.
    pub(crate) fn new(x1: f64, y1: f64, x2: f64, y2: f64) -> Self {
        let (min_x, max_x) = if x1 < x2 { (x1, x2) } else { (x2, x1) };
        let (min_y, max_y) = if y1 < y2 { (y1, y2) } else { (y2, y1) };
        Aabb {
            min_x,
            min_y,
            max_x,
            max_y,
        }
    }

    /// The smallest box covering both `self` and `other`.
    pub(crate) fn union(self, other: Self) -> Self {
        Aabb {
            min_x: self.min_x.min(other.min_x),
            min_y: self.min_y.min(other.min_y),
            max_x: self.max_x.max(other.max_x),
            max_y: self.max_y.max(other.max_y),
        }
    }

    /// The box's extent along the x axis.
    pub(crate) fn width(&self) -> f64 {
        self.max_x - self.min_x
    }

    /// The box's extent along the y axis.
    pub(crate) fn height(&self) -> f64 {
        self.max_y - self.min_y
    }
}

/// A borrowed coordinate sequence: parallel x/y ordinate slices of equal length. The semantics
/// depend on context: a line string's path, a multi-point's members, or one polygon ring.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Coords<'a> {
    xs: &'a [f64],
    ys: &'a [f64],
}

impl<'a> Coords<'a> {
    /// A sequence over parallel ordinate slices, which must have equal lengths.
    pub(crate) fn new(xs: &'a [f64], ys: &'a [f64]) -> Self {
        debug_assert_eq!(xs.len(), ys.len());
        Coords { xs, ys }
    }

    /// The number of coordinates.
    pub(crate) fn len(&self) -> usize {
        self.xs.len()
    }

    /// The coordinate at `index`.
    pub(crate) fn coord(&self, index: usize) -> Coord {
        Coord {
            x: self.xs[index],
            y: self.ys[index],
        }
    }

    /// The raw x ordinate slice, for bulk folds.
    pub(crate) fn xs(&self) -> &'a [f64] {
        self.xs
    }

    /// The raw y ordinate slice, for bulk folds.
    pub(crate) fn ys(&self) -> &'a [f64] {
        self.ys
    }
}

/// A borrowed polygon: full ordinate buffers plus `ring_count + 1` absolute vertex offsets.
/// Ring 0 is the exterior; any further rings are holes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PolygonRef<'a> {
    xs: &'a [f64],
    ys: &'a [f64],
    rings: &'a [usize],
}

impl<'a> PolygonRef<'a> {
    /// A polygon over `ring_count + 1` monotonic vertex offsets into the ordinate buffers.
    pub(crate) fn new(xs: &'a [f64], ys: &'a [f64], rings: &'a [usize]) -> Self {
        debug_assert!(!rings.is_empty());
        PolygonRef { xs, ys, rings }
    }

    /// The number of rings, counting the exterior.
    pub(crate) fn ring_count(&self) -> usize {
        self.rings.len() - 1
    }

    /// The ring at `index`; ring 0 is the exterior.
    pub(crate) fn ring(&self, index: usize) -> Coords<'a> {
        let (start, end) = (self.rings[index], self.rings[index + 1]);
        Coords::new(&self.xs[start..end], &self.ys[start..end])
    }
}

/// A borrowed multi-line-string: full ordinate buffers plus `line_count + 1` absolute vertex
/// offsets.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MultiLineStringRef<'a> {
    xs: &'a [f64],
    ys: &'a [f64],
    lines: &'a [usize],
}

impl<'a> MultiLineStringRef<'a> {
    /// A multi-line-string over `line_count + 1` monotonic vertex offsets into the ordinate
    /// buffers.
    pub(crate) fn new(xs: &'a [f64], ys: &'a [f64], lines: &'a [usize]) -> Self {
        debug_assert!(!lines.is_empty());
        MultiLineStringRef { xs, ys, lines }
    }

    /// Every vertex of every member as one contiguous sequence (members partition the span).
    pub(crate) fn coords(&self) -> Coords<'a> {
        let (start, end) = (self.lines[0], self.lines[self.lines.len() - 1]);
        Coords::new(&self.xs[start..end], &self.ys[start..end])
    }
}

/// A borrowed multi-polygon: full ordinate buffers, `polygon_count + 1` absolute ring offsets,
/// and the full ring-to-vertex offset level shared by its polygons.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MultiPolygonRef<'a> {
    xs: &'a [f64],
    ys: &'a [f64],
    polygons: &'a [usize],
    rings: &'a [usize],
}

impl<'a> MultiPolygonRef<'a> {
    /// A multi-polygon over `polygon_count + 1` monotonic offsets into `rings`, itself the full
    /// monotonic ring-to-vertex offset level.
    pub(crate) fn new(
        xs: &'a [f64],
        ys: &'a [f64],
        polygons: &'a [usize],
        rings: &'a [usize],
    ) -> Self {
        debug_assert!(!polygons.is_empty());
        MultiPolygonRef {
            xs,
            ys,
            polygons,
            rings,
        }
    }

    /// The number of member polygons.
    pub(crate) fn polygon_count(&self) -> usize {
        self.polygons.len() - 1
    }

    /// The member polygon at `index`.
    pub(crate) fn polygon(&self, index: usize) -> PolygonRef<'a> {
        let (start, end) = (self.polygons[index], self.polygons[index + 1]);
        PolygonRef::new(self.xs, self.ys, &self.rings[start..=end])
    }
}

/// One row of any native geometry type, borrowed from its canonicalized storage. Row validity is
/// not part of the view; callers mask computed results separately.
#[derive(Debug, Clone, Copy)]
pub(crate) enum GeometryRef<'a> {
    /// A single coordinate.
    Point(Coord),
    /// An open path of coordinates.
    LineString(Coords<'a>),
    /// An unordered set of coordinates.
    MultiPoint(Coords<'a>),
    /// An exterior ring plus optional holes.
    Polygon(PolygonRef<'a>),
    /// A set of open paths.
    MultiLineString(MultiLineStringRef<'a>),
    /// A set of polygons.
    MultiPolygon(MultiPolygonRef<'a>),
    /// An axis-aligned box.
    Rect(Aabb),
}
