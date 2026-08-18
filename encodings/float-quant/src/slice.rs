// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::slice::SliceReduce;
use vortex_array::dtype::PType;
use vortex_error::VortexResult;

use crate::FloatMult;
use crate::FloatMultArrayExt;
use crate::FloatMultArraySlotsExt;
use crate::FloatQuant;
use crate::FloatQuantArraySlotsExt;

impl SliceReduce for FloatQuant {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            FloatQuant::try_new(
                array.primary().slice(range.clone())?,
                array.secondary().slice(range)?,
                PType::try_from(array.dtype())?,
                array.data().k,
            )?
            .into_array(),
        ))
    }
}

impl SliceReduce for FloatMult {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            FloatMult::try_new(
                array.primary().slice(range.clone())?,
                array.secondary().slice(range)?,
                PType::try_from(array.dtype())?,
                array.base(),
            )?
            .into_array(),
        ))
    }
}
