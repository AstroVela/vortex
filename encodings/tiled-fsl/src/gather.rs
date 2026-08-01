// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PiecewiseSequenceArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::TiledFixedSizeList;
use crate::TiledFixedSizeListArray;
use crate::TiledFixedSizeListArrayExt;
use crate::TiledFixedSizeListArraySlotsExt;
use crate::geometry::geometry_usizes;
use crate::geometry::physical_offset;

/// Builds one contiguous physical-index run for each dimension slab in a row-tile-aligned span.
pub(crate) fn gather_physical_row_tile_span(
    array: ArrayView<'_, TiledFixedSizeList>,
    range: Range<usize>,
) -> VortexResult<ArrayRef> {
    let list_size = array.list_size() as usize;
    let scalar_count = range.len().checked_mul(list_size).ok_or_else(|| {
        vortex_err!(
            InvalidArgument:
            "row span {} times list size {list_size} overflows usize",
            range.len()
        )
    })?;
    let geometry = array.geometry();
    let (_, tile_dimensions) = geometry_usizes(geometry)?;
    let dimension_slab_count = list_size.div_ceil(tile_dimensions);
    let mut starts = Vec::<u64>::with_capacity(dimension_slab_count);
    let mut lengths = Vec::<u64>::with_capacity(dimension_slab_count);

    for dimension_start in (0..list_size).step_by(tile_dimensions) {
        let dimension_width = list_size.min(dimension_start + tile_dimensions) - dimension_start;
        starts.push(u64::try_from(physical_offset(
            array.len(),
            list_size,
            geometry,
            range.start,
            dimension_start,
        )?)?);
        lengths.push(u64::try_from(
            range.len().checked_mul(dimension_width).ok_or_else(|| {
                vortex_err!(
                    InvalidArgument:
                    "row span {} times dimension width {dimension_width} overflows usize",
                    range.len()
                )
            })?,
        )?);
    }

    Ok(PiecewiseSequenceArray::try_new(
        PrimitiveArray::new(Buffer::from(starts), Validity::NonNullable).into_array(),
        PrimitiveArray::new(Buffer::from(lengths), Validity::NonNullable).into_array(),
        ConstantArray::new(1u64, dimension_slab_count).into_array(),
        scalar_count,
    )?
    .into_array())
}

/// Gathers a row-tile-aligned logical row range while preserving the tiled encoding.
pub(crate) fn gather_tiled_slice(
    array: ArrayView<'_, TiledFixedSizeList>,
    range: Range<usize>,
    validity: Validity,
) -> VortexResult<TiledFixedSizeListArray> {
    let output_len = range.len();
    let geometry = array.geometry();
    let indices = gather_physical_row_tile_span(array, range)?;
    let elements = array.elements().take(indices)?;
    TiledFixedSizeList::try_new(elements, array.list_size(), validity, output_len, geometry)
}
