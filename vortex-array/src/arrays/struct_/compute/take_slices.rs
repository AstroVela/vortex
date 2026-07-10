// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Struct;
use crate::arrays::StructArray;
use crate::arrays::TakeSlicesArray;
use crate::arrays::struct_::StructArrayExt;
use crate::arrays::take_slices::TakeSlicesReduce;
use crate::arrays::take_slices::check_index_arrays;

impl TakeSlicesReduce for Struct {
    fn take_slices(
        array: ArrayView<'_, Self>,
        starts: &ArrayRef,
        lengths: &ArrayRef,
        output_len: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        check_index_arrays(starts, lengths)?;

        let fields = array
            .iter_unmasked_fields()
            .map(|field| {
                TakeSlicesArray::try_new(field.clone(), starts.clone(), lengths.clone(), output_len)
                    .map(IntoArray::into_array)
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let validity = array.validity()?.take_slices(starts, lengths, output_len)?;

        StructArray::try_new_with_dtype(fields, array.struct_fields().clone(), output_len, validity)
            .map(StructArray::into_array)
            .map(Some)
    }
}
