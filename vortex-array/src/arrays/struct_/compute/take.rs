// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Struct;
use crate::arrays::StructArray;
use crate::arrays::dict::TakeReduce;
use crate::arrays::struct_::StructArrayExt;
use crate::arrays::take_slices::TakeSlicesReduce;
use crate::arrays::take_slices::selector_output_len;
use crate::builtins::ArrayBuiltins;
use crate::scalar::Scalar;
use crate::validity::Validity;

impl TakeReduce for Struct {
    fn take(array: ArrayView<'_, Struct>, indices: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        // If the struct array is empty then the indices must be all null, otherwise it will access
        // an out of bounds element.
        if array.is_empty() {
            return StructArray::try_new_with_dtype(
                array.iter_unmasked_fields().cloned().collect::<Vec<_>>(),
                array.struct_fields().clone(),
                indices.len(),
                Validity::AllInvalid,
            )
            .map(StructArray::into_array)
            .map(Some);
        }

        // TODO(connor): This could be bad for cache locality...

        // Fill null indices with zero so they point at a valid row.
        // Note that we strip nullability so that `Take::return_dtype` doesn't union nullable into
        // each field's dtype (the struct-level validity already captures which rows are null).
        let fill_scalar = Scalar::zero_value(&indices.dtype().as_nonnullable());
        let inner_indices = indices.clone().fill_null(fill_scalar)?;

        StructArray::try_new_with_dtype(
            array
                .iter_unmasked_fields()
                .map(|field| field.take(inner_indices.clone()))
                .collect::<Result<Vec<_>, _>>()?,
            array.struct_fields().clone(),
            indices.len(),
            array.validity()?.take(indices)?,
        )
        .map(|a| a.into_array())
        .map(Some)
    }
}

impl TakeSlicesReduce for Struct {
    fn take_slices(
        array: ArrayView<'_, Struct>,
        starts: &ArrayRef,
        lengths: &ArrayRef,
    ) -> VortexResult<Option<ArrayRef>> {
        let len = selector_output_len(array.len(), starts, lengths)?;
        let fields = array
            .iter_unmasked_fields()
            .map(|field| field.take_slices(starts.clone(), lengths.clone()))
            .collect::<VortexResult<Vec<_>>>()?;

        StructArray::try_new_with_dtype(
            fields,
            array.struct_fields().clone(),
            len,
            array.validity()?.take_slices(starts, lengths)?,
        )
        .map(StructArray::into_array)
        .map(Some)
    }
}
