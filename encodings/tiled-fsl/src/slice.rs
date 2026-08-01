// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::slice::SliceReduce;
use vortex_error::VortexResult;

use crate::TiledFixedSizeList;
use crate::TiledFixedSizeListArrayExt;
use crate::TiledFixedSizeListArraySlotsExt;
use crate::gather::gather_tiled_slice;
use crate::geometry::geometry_usizes;

impl SliceReduce for TiledFixedSizeList {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        if array.list_size() == 0 {
            let validity = array.array_validity().slice(range.clone())?;
            return Ok(Some(
                TiledFixedSizeList::try_new_view(
                    array.elements().clone(),
                    array.list_size(),
                    validity,
                    range.len(),
                    array.geometry(),
                    0,
                    range.len(),
                )?
                .into_array(),
            ));
        }

        if array.is_full_width() {
            let validity = array.array_validity().slice(range.clone())?;
            let (tile_rows, _) = geometry_usizes(array.geometry())?;
            let list_size = array.list_size() as usize;
            let absolute_start = array.row_offset() + range.start;
            let absolute_end = array.row_offset() + range.end;
            let retained_start = (absolute_start / tile_rows) * tile_rows;
            let retained_end =
                (absolute_end.div_ceil(tile_rows) * tile_rows).min(array.backing_rows());
            let physical = retained_start * list_size..retained_end * list_size;
            let elements = array.elements().slice(physical)?;
            return Ok(Some(
                TiledFixedSizeList::try_new_view(
                    elements,
                    array.list_size(),
                    validity,
                    range.len(),
                    array.geometry(),
                    absolute_start - retained_start,
                    retained_end - retained_start,
                )?
                .into_array(),
            ));
        }

        let (tile_rows, _) = geometry_usizes(array.geometry())?;
        if range.start % tile_rows != 0 || (range.end % tile_rows != 0 && range.end != array.len())
        {
            return Ok(None);
        }

        let validity = array.array_validity().slice(range.clone())?;
        Ok(Some(
            gather_tiled_slice(array, range, validity)?.into_array(),
        ))
    }
}
