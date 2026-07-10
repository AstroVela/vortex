// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Masked;
use crate::arrays::MaskedArray;
use crate::arrays::TakeSlicesArray;
use crate::arrays::masked::MaskedArraySlotsExt;
use crate::arrays::take_slices::TakeSlicesReduce;

impl TakeSlicesReduce for Masked {
    fn take_slices(
        array: ArrayView<'_, Masked>,
        starts: &ArrayRef,
        lengths: &ArrayRef,
        output_len: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let child = TakeSlicesArray::try_new(
            array.child().clone(),
            starts.clone(),
            lengths.clone(),
            output_len,
        )?
        .into_array();
        let validity = array.validity()?.take_slices(starts, lengths, output_len)?;

        // SAFETY: `MaskedArray` guarantees its child has no logical nulls. Taking slices from that
        // child preserves all-valid child values; nulls remain represented solely by `validity`.
        unsafe { MaskedArray::new_unchecked_child_all_valid(child, validity) }
            .map(IntoArray::into_array)
            .map(Some)
    }
}
