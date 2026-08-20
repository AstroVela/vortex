// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::slice::SliceReduce;
use vortex_error::VortexResult;

use crate::IntMult;
use crate::IntMultArrayExt;
use crate::IntMultArraySlotsExt;

impl SliceReduce for IntMult {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            IntMult::try_new(
                array.primary().slice(range.clone())?,
                array.secondary().slice(range)?,
                array.base(),
            )?
            .into_array(),
        ))
    }
}
