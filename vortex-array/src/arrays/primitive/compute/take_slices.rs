// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools as _;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::take_slices::TakeSlicesKernel;
use crate::arrays::take_slices::check_index_arrays;
use crate::arrays::take_slices::index_value_to_usize;
use crate::arrays::take_slices::validate_index_ranges;
use crate::dtype::IntegerPType;
use crate::dtype::NativePType;
use crate::executor::ExecutionCtx;
use crate::match_each_native_ptype;
use crate::match_each_unsigned_integer_ptype;

impl TakeSlicesKernel for Primitive {
    fn take_slices(
        array: ArrayView<'_, Self>,
        starts: &ArrayRef,
        lengths: &ArrayRef,
        output_len: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        execute_primitive_selected_ranges(array, starts, lengths, output_len, ctx).map(Some)
    }
}

fn execute_primitive_selected_ranges(
    array: ArrayView<'_, Primitive>,
    starts: &ArrayRef,
    lengths: &ArrayRef,
    output_len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    match_each_native_ptype!(array.ptype(), |T| {
        execute_primitive_selected_ranges_for_type::<T>(array, starts, lengths, output_len, ctx)
    })
}

fn execute_primitive_selected_ranges_for_type<T: NativePType>(
    array: ArrayView<'_, Primitive>,
    starts: &ArrayRef,
    lengths: &ArrayRef,
    output_len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    check_index_arrays(starts, lengths)?;

    match_each_unsigned_integer_ptype!(starts.dtype().as_ptype(), |S| {
        execute_primitive_selected_ranges_for_start_type::<T, S>(
            array, starts, lengths, output_len, ctx,
        )
    })
}

fn execute_primitive_selected_ranges_for_start_type<T, S>(
    array: ArrayView<'_, Primitive>,
    starts: &ArrayRef,
    lengths: &ArrayRef,
    output_len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    T: NativePType,
    S: IntegerPType,
{
    match_each_unsigned_integer_ptype!(lengths.dtype().as_ptype(), |L| {
        execute_primitive_selected_ranges_typed::<T, S, L>(array, starts, lengths, output_len, ctx)
    })
}

fn execute_primitive_selected_ranges_typed<T, S, L>(
    array: ArrayView<'_, Primitive>,
    starts: &ArrayRef,
    lengths: &ArrayRef,
    output_len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    T: NativePType,
    S: IntegerPType,
    L: IntegerPType,
{
    let starts = starts.clone().execute::<PrimitiveArray>(ctx)?;
    let lengths = lengths.clone().execute::<PrimitiveArray>(ctx)?;
    let values = primitive_array_ranges::<T, S, L>(
        array,
        starts.as_slice::<S>(),
        lengths.as_slice::<L>(),
        output_len,
    )?;
    let starts = starts.into_array();
    let lengths = lengths.into_array();
    let validity = array
        .validity()?
        .take_slices(&starts, &lengths, output_len)?;
    Ok(PrimitiveArray::new(values, validity).into_array())
}

fn primitive_array_ranges<T, S, L>(
    array: ArrayView<'_, Primitive>,
    starts: &[S],
    lengths: &[L],
    output_len: usize,
) -> VortexResult<Buffer<T>>
where
    T: NativePType,
    S: IntegerPType,
    L: IntegerPType,
{
    let source = array.as_slice::<T>();
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
