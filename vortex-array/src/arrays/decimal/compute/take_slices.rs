// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools as _;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::take_slices::TakeSlicesKernel;
use crate::arrays::take_slices::check_index_arrays;
use crate::arrays::take_slices::index_value_to_usize;
use crate::arrays::take_slices::validate_index_ranges;
use crate::dtype::IntegerPType;
use crate::dtype::NativeDecimalType;
use crate::executor::ExecutionCtx;
use crate::match_each_decimal_value_type;
use crate::match_each_unsigned_integer_ptype;

impl TakeSlicesKernel for Decimal {
    fn take_slices(
        array: ArrayView<'_, Self>,
        starts: &ArrayRef,
        lengths: &ArrayRef,
        output_len: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        check_index_arrays(starts, lengths)?;

        match_each_decimal_value_type!(array.values_type(), |D| {
            take_slices_for_value_type::<D>(array, starts, lengths, output_len, ctx)
        })
    }
}

fn take_slices_for_value_type<T>(
    array: ArrayView<'_, Decimal>,
    starts: &ArrayRef,
    lengths: &ArrayRef,
    output_len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>>
where
    T: NativeDecimalType,
{
    match_each_unsigned_integer_ptype!(starts.dtype().as_ptype(), |S| {
        take_slices_for_start_type::<T, S>(array, starts, lengths, output_len, ctx)
    })
}

fn take_slices_for_start_type<T, S>(
    array: ArrayView<'_, Decimal>,
    starts: &ArrayRef,
    lengths: &ArrayRef,
    output_len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>>
where
    T: NativeDecimalType,
    S: IntegerPType,
{
    match_each_unsigned_integer_ptype!(lengths.dtype().as_ptype(), |L| {
        take_slices_typed::<T, S, L>(array, starts, lengths, output_len, ctx).map(Some)
    })
}

fn take_slices_typed<T, S, L>(
    array: ArrayView<'_, Decimal>,
    starts: &ArrayRef,
    lengths: &ArrayRef,
    output_len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    T: NativeDecimalType,
    S: IntegerPType,
    L: IntegerPType,
{
    let starts = starts.clone().execute::<PrimitiveArray>(ctx)?;
    let lengths = lengths.clone().execute::<PrimitiveArray>(ctx)?;
    let values = gather_values(
        array.buffer::<T>().as_slice(),
        starts.as_slice::<S>(),
        lengths.as_slice::<L>(),
        output_len,
    )?;

    let starts = starts.into_array();
    let lengths = lengths.into_array();
    let validity = array
        .validity()?
        .take_slices(&starts, &lengths, output_len)?;

    // SAFETY: contiguous gather preserves the decimal dtype and value representation.
    let array = unsafe { DecimalArray::new_unchecked(values, array.decimal_dtype(), validity) };
    Ok(array.into_array())
}

fn gather_values<T, S, L>(
    source: &[T],
    starts: &[S],
    lengths: &[L],
    output_len: usize,
) -> VortexResult<Buffer<T>>
where
    T: NativeDecimalType,
    S: IntegerPType,
    L: IntegerPType,
{
    validate_index_ranges(source.len(), starts, lengths, output_len)?;

    let mut values = BufferMut::<T>::with_capacity(output_len);
    for (&start, &length) in starts.iter().zip_eq(lengths) {
        let start = index_value_to_usize("start", start)?;
        let length = index_value_to_usize("length", length)?;
        let end = start + length;
        values.extend_from_slice(&source[start..end]);
    }

    Ok(values.freeze())
}
