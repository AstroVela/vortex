// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::num::NonZeroU32;
use std::ops::Range;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

/// The number of logical rows and dimensions in a physical tile.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TileGeometry {
    rows: NonZeroU32,
    dimensions: NonZeroU32,
}

impl TileGeometry {
    /// Creates a tile geometry with nonzero row and dimension capacities.
    pub const fn new(rows: NonZeroU32, dimensions: NonZeroU32) -> Self {
        Self { rows, dimensions }
    }

    /// Returns the number of rows in each tile.
    pub const fn rows(self) -> NonZeroU32 {
        self.rows
    }

    /// Returns the number of dimensions in each tile.
    pub const fn dimensions(self) -> NonZeroU32 {
        self.dimensions
    }
}

/// The logical and physical extents of one tile.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TileBounds {
    /// The logical rows contained by this tile.
    pub row_range: Range<usize>,
    /// The logical dimensions contained by this tile.
    pub dimension_range: Range<usize>,
    /// The contiguous, unpadded physical values occupied by this tile.
    pub physical_range: Range<usize>,
    /// The visible rows, relative to the beginning of the retained physical tile.
    pub rows_within_tile: Range<usize>,
    full_tile: bool,
}

impl TileBounds {
    pub(crate) fn new(
        row_range: Range<usize>,
        dimension_range: Range<usize>,
        physical_range: Range<usize>,
        rows_within_tile: Range<usize>,
        full_tile: bool,
    ) -> Self {
        Self {
            row_range,
            dimension_range,
            physical_range,
            rows_within_tile,
            full_tile,
        }
    }

    /// Returns whether the logical view selects every row in a full-height physical tile.
    pub fn is_full_tile(&self) -> bool {
        self.full_tile
    }
}

/// Iterates tiled fixed-size-list bounds in physical storage order.
///
/// The iterator stores only scalar geometry and tile counters. Constructing or advancing it does
/// not access the array's physical child.
#[derive(Clone, Debug)]
pub struct TileBoundsIter {
    len: usize,
    list_size: usize,
    rows: usize,
    dimensions: usize,
    row_offset: usize,
    backing_rows: usize,
    row_tile_count: usize,
    dimension_tile_count: usize,
    next_row_tile: usize,
    next_dimension_tile: usize,
}

#[derive(Clone, Copy, Debug)]
struct TileLayout {
    len: usize,
    list_size: usize,
    rows: usize,
    dimensions: usize,
    row_offset: usize,
    backing_rows: usize,
}

impl TileBoundsIter {
    pub(crate) fn new(
        len: usize,
        list_size: usize,
        geometry: TileGeometry,
        row_tile_count: usize,
        dimension_tile_count: usize,
    ) -> Self {
        Self::new_view(
            len,
            list_size,
            geometry,
            0,
            len,
            row_tile_count,
            dimension_tile_count,
        )
    }

    pub(crate) fn new_view(
        len: usize,
        list_size: usize,
        geometry: TileGeometry,
        row_offset: usize,
        backing_rows: usize,
        row_tile_count: usize,
        dimension_tile_count: usize,
    ) -> Self {
        #[allow(
            clippy::expect_used,
            reason = "validated tiled fixed-size-list geometry must fit usize"
        )]
        let (rows, dimensions) = geometry_usizes(geometry)
            .expect("validated tiled fixed-size-list geometry must fit usize");
        Self {
            len,
            list_size,
            rows,
            dimensions,
            row_offset,
            backing_rows,
            row_tile_count,
            dimension_tile_count,
            next_row_tile: 0,
            next_dimension_tile: 0,
        }
    }
}

impl Iterator for TileBoundsIter {
    type Item = TileBounds;

    fn next(&mut self) -> Option<Self::Item> {
        if self.row_tile_count == 0 || self.next_dimension_tile == self.dimension_tile_count {
            return None;
        }

        let bounds = tile_bounds_for_validated_array(
            TileLayout {
                len: self.len,
                list_size: self.list_size,
                rows: self.rows,
                dimensions: self.dimensions,
                row_offset: self.row_offset,
                backing_rows: self.backing_rows,
            },
            self.next_row_tile,
            self.next_dimension_tile,
        );
        self.next_row_tile += 1;
        if self.next_row_tile == self.row_tile_count {
            self.next_row_tile = 0;
            self.next_dimension_tile += 1;
        }
        Some(bounds)
    }
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn tile_bounds(
    len: usize,
    list_size: usize,
    geometry: TileGeometry,
    row_tile: usize,
    dimension_tile: usize,
) -> VortexResult<TileBounds> {
    tile_bounds_view(len, list_size, geometry, 0, len, row_tile, dimension_tile)
}

pub(crate) fn tile_bounds_view(
    len: usize,
    list_size: usize,
    geometry: TileGeometry,
    row_offset: usize,
    backing_rows: usize,
    row_tile: usize,
    dimension_tile: usize,
) -> VortexResult<TileBounds> {
    if len == 0 || list_size == 0 {
        vortex_bail!(InvalidArgument: "cannot compute tiles for an empty logical extent");
    }

    let (rows, dimensions) = geometry_usizes(geometry)?;
    let logical_end = row_offset.checked_add(len).ok_or_else(|| {
        vortex_error::vortex_err!(InvalidArgument: "row offset plus length overflows logical extent")
    })?;
    if row_offset >= rows || logical_end > backing_rows {
        vortex_bail!(InvalidArgument: "invalid tiled fixed-size-list row window");
    }
    let row_start = row_tile.checked_mul(rows).ok_or_else(|| {
        vortex_error::vortex_err!(InvalidArgument: "row tile index {row_tile} overflows tile geometry")
    })?;
    let dimension_start = dimension_tile.checked_mul(dimensions).ok_or_else(|| {
        vortex_error::vortex_err!(InvalidArgument: "dimension tile index {dimension_tile} overflows tile geometry")
    })?;

    if row_start >= logical_end
        || row_start
            .checked_add(rows)
            .is_none_or(|end| end <= row_offset)
        || dimension_start >= list_size
    {
        vortex_bail!(
            InvalidArgument:
            "tile ({row_tile}, {dimension_tile}) is outside logical extent ({len}, {list_size})"
        );
    }

    tile_bounds_from_starts(
        TileLayout {
            len,
            list_size,
            rows,
            dimensions,
            row_offset,
            backing_rows,
        },
        row_start,
        dimension_start,
    )
}

fn tile_bounds_for_validated_array(
    layout: TileLayout,
    row_tile: usize,
    dimension_tile: usize,
) -> TileBounds {
    let TileLayout {
        len,
        list_size,
        rows,
        dimensions,
        row_offset,
        ..
    } = layout;
    let row_tile_count = if len == 0 {
        0
    } else {
        (row_offset + len).div_ceil(rows)
    };
    let dimension_tile_count = list_size.div_ceil(dimensions);
    // Callers must provide only counters generated from this validated array's geometry.
    debug_assert!(row_tile < row_tile_count);
    debug_assert!(dimension_tile < dimension_tile_count);
    debug_assert!(len.checked_mul(list_size).is_some());

    let row_start = row_tile * rows;
    let dimension_start = dimension_tile * dimensions;
    match tile_bounds_from_starts(layout, row_start, dimension_start) {
        Ok(bounds) => bounds,
        Err(_) => unreachable!("validated tiled array has in-range tile bounds"),
    }
}

pub(crate) fn geometry_usizes(geometry: TileGeometry) -> VortexResult<(usize, usize)> {
    let rows = usize::try_from(geometry.rows().get()).map_err(|_| {
        vortex_error::vortex_err!(
            InvalidArgument: "tile row geometry {} does not fit usize",
            geometry.rows().get()
        )
    })?;
    let dimensions = usize::try_from(geometry.dimensions().get()).map_err(|_| {
        vortex_error::vortex_err!(
            InvalidArgument: "tile dimension geometry {} does not fit usize",
            geometry.dimensions().get()
        )
    })?;
    Ok((rows, dimensions))
}

fn tile_bounds_from_starts(
    layout: TileLayout,
    row_start: usize,
    dimension_start: usize,
) -> VortexResult<TileBounds> {
    let TileLayout {
        len,
        list_size,
        rows,
        dimensions,
        row_offset,
        backing_rows,
    } = layout;
    let retained_row_end = row_start
        .checked_add(rows)
        .ok_or_else(|| vortex_error::vortex_err!(InvalidArgument: "row tile range overflows logical extent"))?
        .min(backing_rows);
    let logical_end = row_offset.checked_add(len).ok_or_else(
        || vortex_error::vortex_err!(InvalidArgument: "row window overflows logical extent"),
    )?;
    let visible_row_start = row_start.max(row_offset);
    let visible_row_end = retained_row_end.min(logical_end);
    let dimension_end = dimension_start
        .checked_add(dimensions)
        .ok_or_else(|| vortex_error::vortex_err!(InvalidArgument: "dimension tile range overflows logical extent"))?
        .min(list_size);
    let retained_row_height = retained_row_end - row_start;
    let dimension_width = dimension_end - dimension_start;

    let physical_start = dimension_start
        .checked_mul(backing_rows)
        .and_then(|offset| {
            row_start
                .checked_mul(dimension_width)
                .and_then(|rows| offset.checked_add(rows))
        })
        .ok_or_else(
            || vortex_error::vortex_err!(InvalidArgument: "tile physical offset overflows usize"),
        )?;
    let physical_len = retained_row_height
        .checked_mul(dimension_width)
        .ok_or_else(
            || vortex_error::vortex_err!(InvalidArgument: "tile physical length overflows usize"),
        )?;
    let physical_end = physical_start.checked_add(physical_len).ok_or_else(
        || vortex_error::vortex_err!(InvalidArgument: "tile physical range overflows usize"),
    )?;

    let rows_within_tile = (visible_row_start - row_start)..(visible_row_end - row_start);
    let full_tile = rows_within_tile.start == 0 && rows_within_tile.end == rows;
    Ok(TileBounds::new(
        (visible_row_start - row_offset)..(visible_row_end - row_offset),
        dimension_start..dimension_end,
        physical_start..physical_end,
        rows_within_tile,
        full_tile,
    ))
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn physical_offset(
    len: usize,
    list_size: usize,
    geometry: TileGeometry,
    row: usize,
    dimension: usize,
) -> VortexResult<usize> {
    physical_offset_view(len, list_size, geometry, 0, len, row, dimension)
}

pub(crate) fn physical_offset_view(
    len: usize,
    list_size: usize,
    geometry: TileGeometry,
    row_offset: usize,
    backing_rows: usize,
    row: usize,
    dimension: usize,
) -> VortexResult<usize> {
    if row >= len || dimension >= list_size {
        vortex_bail!(
            InvalidArgument:
            "logical position ({row}, {dimension}) is outside extent ({len}, {list_size})"
        );
    }

    let (rows, dimensions) = geometry_usizes(geometry)?;
    let physical_row = row_offset.checked_add(row).ok_or_else(
        || vortex_error::vortex_err!(InvalidArgument: "physical row overflows usize"),
    )?;
    let row_tile = physical_row / rows;
    let dimension_tile = dimension / dimensions;
    let bounds = tile_bounds_view(
        len,
        list_size,
        geometry,
        row_offset,
        backing_rows,
        row_tile,
        dimension_tile,
    )?;
    let row_within_tile = physical_row - row_tile * rows;
    let dimension_within_tile = dimension - bounds.dimension_range.start;
    let dimension_width = bounds.dimension_range.len();
    let row_height = bounds.physical_range.len() / dimension_width;
    bounds
        .physical_range
        .start
        .checked_add(dimension_within_tile.checked_mul(row_height).ok_or_else(
            || vortex_error::vortex_err!(InvalidArgument: "dimension offset overflows usize"),
        )?)
        .and_then(|offset| offset.checked_add(row_within_tile))
        .ok_or_else(
            || vortex_error::vortex_err!(InvalidArgument: "physical offset overflows usize"),
        )
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use vortex_error::VortexResult;

    use super::TileGeometry;
    use super::physical_offset;
    use super::tile_bounds;

    fn geometry() -> TileGeometry {
        TileGeometry::new(NonZeroU32::new(2).unwrap(), NonZeroU32::new(3).unwrap())
    }

    #[test]
    fn physical_offsets_match_golden_layout() -> VortexResult<()> {
        let expected = [[0, 2, 4, 9, 11], [1, 3, 5, 10, 12], [6, 7, 8, 13, 14]];
        for (row, offsets) in expected.into_iter().enumerate() {
            for (dimension, expected_offset) in offsets.into_iter().enumerate() {
                assert_eq!(
                    physical_offset(3, 5, geometry(), row, dimension)?,
                    expected_offset
                );
            }
        }
        Ok(())
    }

    #[test]
    fn tile_bounds_cover_unpadded_tails() -> VortexResult<()> {
        let bounds = [
            tile_bounds(3, 5, geometry(), 0, 0)?,
            tile_bounds(3, 5, geometry(), 1, 0)?,
            tile_bounds(3, 5, geometry(), 0, 1)?,
            tile_bounds(3, 5, geometry(), 1, 1)?,
        ];
        assert_eq!(bounds[0].physical_range, 0..6);
        assert_eq!(bounds[1].physical_range, 6..9);
        assert_eq!(bounds[2].physical_range, 9..13);
        assert_eq!(bounds[3].physical_range, 13..15);
        assert_eq!(bounds[3].row_range, 2..3);
        assert_eq!(bounds[3].dimension_range, 3..5);
        Ok(())
    }
}
