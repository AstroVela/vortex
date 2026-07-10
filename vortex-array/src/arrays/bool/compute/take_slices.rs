// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools as _;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Bool;
use crate::arrays::BoolArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::bool::BoolArrayExt;
use crate::arrays::take_slices::TakeSlicesKernel;
use crate::arrays::take_slices::check_index_arrays;
use crate::arrays::take_slices::index_value_to_usize;
use crate::arrays::take_slices::validate_index_ranges;
use crate::dtype::IntegerPType;
use crate::executor::ExecutionCtx;
use crate::match_each_unsigned_integer_ptype;

impl TakeSlicesKernel for Bool {
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
    array: ArrayView<'_, Bool>,
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
    let values = gather_bits(
        &array.to_bit_buffer(),
        starts.as_slice::<S>(),
        lengths.as_slice::<L>(),
        output_len,
    )?;

    let starts = starts.into_array();
    let lengths = lengths.into_array();
    let validity = array
        .validity()?
        .take_slices(&starts, &lengths, output_len)?;

    Ok(BoolArray::new(values, validity).into_array())
}

fn gather_bits<S, L>(
    source: &BitBuffer,
    starts: &[S],
    lengths: &[L],
    output_len: usize,
) -> VortexResult<BitBuffer>
where
    S: IntegerPType,
    L: IntegerPType,
{
    validate_index_ranges(source.len(), starts, lengths, output_len)?;

    let mut values = BitBufferMut::with_capacity(output_len);
    for (&start, &length) in starts.iter().zip_eq(lengths) {
        let start = index_value_to_usize("start", start)?;
        let length = index_value_to_usize("length", length)?;
        let end = start + length;
        values.append_buffer(&source.slice(start..end));
    }

    Ok(values.freeze())
}
