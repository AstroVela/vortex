// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::PrimitiveArray;
use crate::arrays::Struct;
use crate::arrays::StructArray;
use crate::arrays::TakeSlicesArray;
use crate::arrays::struct_::StructArrayExt;
use crate::arrays::take_slices::TakeSlicesKernel;
use crate::arrays::take_slices::check_index_arrays;
use crate::arrays::take_slices::validate_index_ranges;
use crate::dtype::IntegerPType;
use crate::executor::ExecutionCtx;
use crate::match_each_unsigned_integer_ptype;

impl TakeSlicesKernel for Struct {
    fn take_slices(
        array: ArrayView<'_, Self>,
        starts: &ArrayRef,
        lengths: &ArrayRef,
        output_len: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        check_index_arrays(starts, lengths)?;

        match_each_unsigned_integer_ptype!(starts.dtype().as_ptype(), |S| {
            match_each_unsigned_integer_ptype!(lengths.dtype().as_ptype(), |L| {
                take_slices_typed::<S, L>(array, starts, lengths, output_len, ctx)
            })
        })
        .map(Some)
    }
}

fn take_slices_typed<S, L>(
    array: ArrayView<'_, Struct>,
    starts: &ArrayRef,
    lengths: &ArrayRef,
    output_len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    S: IntegerPType,
    L: IntegerPType,
{
    let starts = starts.clone().execute::<PrimitiveArray>(ctx)?;
    let lengths = lengths.clone().execute::<PrimitiveArray>(ctx)?;
    validate_index_ranges(
        array.len(),
        starts.as_slice::<S>(),
        lengths.as_slice::<L>(),
        output_len,
    )?;

    let starts = starts.into_array();
    let lengths = lengths.into_array();
    let fields = array
        .iter_unmasked_fields()
        .map(|field| {
            // SAFETY: the index arrays and declared output length were validated above, and the
            // child dtype becomes the outer dtype of this per-field TakeSlices array.
            unsafe {
                TakeSlicesArray::new_unchecked(
                    field.clone(),
                    starts.clone(),
                    lengths.clone(),
                    output_len,
                )
                .into_array()
            }
        })
        .collect::<Vec<_>>();
    let validity = array
        .validity()?
        .take_slices(&starts, &lengths, output_len)?;

    StructArray::try_new_with_dtype(fields, array.struct_fields().clone(), output_len, validity)
        .map(StructArray::into_array)
}
