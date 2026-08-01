// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PiecewiseSequenceArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::TileBoundsIter;
use crate::TiledFixedSizeList;
use crate::TiledFixedSizeListArray;
use crate::TiledFixedSizeListArrayExt;
use crate::TiledFixedSizeListArraySlotsExt;
use crate::geometry::geometry_usizes;
use crate::geometry::physical_offset;

/// Gathers a logical row range with compact runs of consecutive physical indices.
pub(crate) fn gather_tiled_slice(
    array: ArrayView<'_, TiledFixedSizeList>,
    range: Range<usize>,
    validity: Validity,
) -> VortexResult<TiledFixedSizeListArray> {
    let list_size = array.list_size() as usize;
    let output_len = range.len();
    let scalar_count = output_len.checked_mul(list_size).ok_or_else(|| {
        vortex_err!(
            InvalidArgument:
            "output row count {output_len} times list size {list_size} overflows usize"
        )
    })?;
    let geometry = array.geometry();
    let (tile_rows, tile_dimensions) = geometry_usizes(geometry)?;
    let row_tile_count = output_len.div_ceil(tile_rows);
    let dimension_tile_count = list_size.div_ceil(tile_dimensions);

    let mut starts = Vec::<u64>::new();
    let mut lengths = Vec::<u64>::new();
    for bounds in TileBoundsIter::new(
        output_len,
        list_size,
        geometry,
        row_tile_count,
        dimension_tile_count,
    ) {
        for dimension in bounds.dimension_range {
            let mut source_row = range
                .start
                .checked_add(bounds.row_range.start)
                .ok_or_else(|| vortex_err!(InvalidArgument: "slice source row overflows usize"))?;
            let source_end = range
                .start
                .checked_add(bounds.row_range.end)
                .ok_or_else(|| vortex_err!(InvalidArgument: "slice source row overflows usize"))?;
            while source_row < source_end {
                let source_tile_end = source_row
                    .checked_div(tile_rows)
                    .and_then(|tile| tile.checked_add(1))
                    .and_then(|tile| tile.checked_mul(tile_rows))
                    .map(|end| end.min(array.len()))
                    .ok_or_else(
                        || vortex_err!(InvalidArgument: "source tile end overflows usize"),
                    )?;
                let run_end = source_end.min(source_tile_end);
                let start = u64::try_from(physical_offset(
                    array.len(),
                    list_size,
                    geometry,
                    source_row,
                    dimension,
                )?)?;
                let length = u64::try_from(run_end - source_row)?;
                let previous_end = starts
                    .last()
                    .zip(lengths.last())
                    .and_then(|(&start, &length)| start.checked_add(length));
                if previous_end == Some(start) {
                    let previous_length = lengths.last_mut().ok_or_else(
                        || vortex_err!(InvalidArgument: "slice run metadata is inconsistent"),
                    )?;
                    *previous_length = previous_length.checked_add(length).ok_or_else(
                        || vortex_err!(InvalidArgument: "slice run length overflows u64"),
                    )?;
                } else {
                    starts.push(start);
                    lengths.push(length);
                }
                source_row = run_end;
            }
        }
    }

    let run_count = starts.len();
    let indices = PiecewiseSequenceArray::try_new(
        PrimitiveArray::new(Buffer::from(starts), Validity::NonNullable).into_array(),
        PrimitiveArray::new(Buffer::from(lengths), Validity::NonNullable).into_array(),
        ConstantArray::new(1u64, run_count).into_array(),
        scalar_count,
    )?
    .into_array();
    let elements = array.elements().take(indices)?;
    TiledFixedSizeList::try_new(elements, array.list_size(), validity, output_len, geometry)
}
