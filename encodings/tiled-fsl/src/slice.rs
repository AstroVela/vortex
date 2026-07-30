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
use crate::gather::gather_tiled_rows;

impl SliceReduce for TiledFixedSizeList {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let rows = range.clone().map(Some).collect::<Vec<_>>();
        let validity = array.array_validity().slice(range)?;
        Ok(Some(
            gather_tiled_rows(array, &rows, validity)?.into_array(),
        ))
    }
}
