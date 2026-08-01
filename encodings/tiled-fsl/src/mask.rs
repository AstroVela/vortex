// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::scalar_fn::fns::mask::MaskReduce;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;

use crate::TiledFixedSizeList;
use crate::TiledFixedSizeListArrayExt;
use crate::TiledFixedSizeListArraySlotsExt;

impl MaskReduce for TiledFixedSizeList {
    fn mask(array: ArrayView<'_, Self>, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            TiledFixedSizeList::try_new_view(
                array.elements().clone(),
                array.list_size(),
                array.array_validity().and(Validity::Array(mask.clone()))?,
                array.len(),
                array.geometry(),
                array.row_offset(),
                array.backing_rows(),
            )?
            .into_array(),
        ))
    }
}
