// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::slice::SliceReduce;
use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_error::VortexResult;

use crate::ZstdV2;
use crate::array::unsliced_validity;

pub(crate) static RULES: ParentRuleSet<ZstdV2> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&SliceReduceAdaptor(ZstdV2))]);

impl SliceReduce for ZstdV2 {
    /// Slicing is metadata-only: the slice bounds move, and nothing is decompressed until the
    /// array is decoded, at which point only the frames covering the slice are read.
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            ZstdV2::try_new(
                array.dtype().clone(),
                array.data().with_slice(range.start, range.end),
                unsliced_validity(array),
            )?
            .into_array(),
        ))
    }
}
