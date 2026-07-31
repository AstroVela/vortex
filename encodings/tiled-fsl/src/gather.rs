// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::TileBoundsIter;
use crate::TileGeometry;
use crate::TiledFixedSizeList;
use crate::TiledFixedSizeListArray;
use crate::TiledFixedSizeListArrayExt;
use crate::TiledFixedSizeListArraySlotsExt;
use crate::geometry::geometry_usizes;
use crate::geometry::physical_offset;

/// Builds physical element indices for logical rows in the tiled output layout.
pub(crate) fn physical_indices_for_rows(
    source_len: usize,
    list_size: usize,
    geometry: TileGeometry,
    rows: &[Option<usize>],
) -> VortexResult<Buffer<u64>> {
    assert!(
        source_len != 0 || rows.is_empty(),
        "nonempty tiled row gather requires a nonempty source"
    );
    rows.len().checked_mul(list_size).ok_or_else(|| {
        vortex_err!(
            InvalidArgument:
            "output row count {} times list size {list_size} overflows usize",
            rows.len()
        )
    })?;

    let (tile_rows, tile_dimensions) = geometry_usizes(geometry)?;
    let row_tile_count = rows.len().div_ceil(tile_rows);
    let dimension_tile_count = list_size.div_ceil(tile_dimensions);
    TileBoundsIter::new(
        rows.len(),
        list_size,
        geometry,
        row_tile_count,
        dimension_tile_count,
    )
    .flat_map(|bounds| {
        bounds.dimension_range.flat_map(move |dimension| {
            bounds.row_range.clone().map(move |output_row| {
                let source_row = rows[output_row].unwrap_or(0);
                vortex_ensure!(
                    source_row < source_len,
                    OutOfBounds: source_row,
                    0,
                    source_len
                );
                Ok(u64::try_from(physical_offset(
                    source_len, list_size, geometry, source_row, dimension,
                )?)?)
            })
        })
    })
    .collect()
}

/// Gathers logical rows while retaining the source tile geometry.
pub(crate) fn gather_tiled_rows(
    array: ArrayView<'_, TiledFixedSizeList>,
    rows: &[Option<usize>],
    validity: Validity,
) -> VortexResult<TiledFixedSizeListArray> {
    let indices = physical_indices_for_rows(
        array.len(),
        array.list_size() as usize,
        array.geometry(),
        rows,
    )?;
    let elements = array
        .elements()
        .take(PrimitiveArray::from_iter(indices).into_array())?;
    TiledFixedSizeList::try_new(
        elements,
        array.list_size(),
        validity,
        rows.len(),
        array.geometry(),
    )
}
