// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools as _;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::FixedSizeList;
use crate::arrays::FixedSizeListArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::TakeSlicesArray;
use crate::arrays::fixed_size_list::FixedSizeListArrayExt;
use crate::arrays::take_slices::TakeSlicesKernel;
use crate::arrays::take_slices::check_index_arrays;
use crate::arrays::take_slices::index_value_to_usize;
use crate::arrays::take_slices::validate_index_ranges;
use crate::dtype::IntegerPType;
use crate::executor::ExecutionCtx;
use crate::match_each_unsigned_integer_ptype;

impl TakeSlicesKernel for FixedSizeList {
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
    array: ArrayView<'_, FixedSizeList>,
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

    let list_size = array.list_size() as usize;
    let elements_len = output_len.checked_mul(list_size).ok_or_else(|| {
        vortex_err!(
            "FixedSizeList TakeSlices output length overflow: {output_len} lists of size {list_size}"
        )
    })?;
    let elements = if list_size == 0 {
        array.elements().clone()
    } else {
        gather_elements(
            array.elements(),
            starts.as_slice::<S>(),
            lengths.as_slice::<L>(),
            list_size,
            elements_len,
        )?
    };

    let starts = starts.into_array();
    let lengths = lengths.into_array();
    let validity = array
        .validity()?
        .take_slices(&starts, &lengths, output_len)?;

    // SAFETY: row ranges were validated, element ranges are the corresponding checked
    // fixed-width expansions, and validity has one entry per output row.
    Ok(unsafe {
        FixedSizeListArray::new_unchecked(elements, array.list_size(), validity, output_len)
    }
    .into_array())
}

fn gather_elements<S, L>(
    elements: &ArrayRef,
    starts: &[S],
    lengths: &[L],
    list_size: usize,
    elements_len: usize,
) -> VortexResult<ArrayRef>
where
    S: IntegerPType,
    L: IntegerPType,
{
    let mut element_starts = BufferMut::<u64>::with_capacity(starts.len());
    let mut element_lengths = BufferMut::<u64>::with_capacity(lengths.len());

    for (&start, &length) in starts.iter().zip_eq(lengths) {
        let start = index_value_to_usize("start", start)?;
        let length = index_value_to_usize("length", length)?;
        let element_start = start.checked_mul(list_size).ok_or_else(|| {
            vortex_err!(
                "FixedSizeList TakeSlices element start overflow for start {start} and list size {list_size}"
            )
        })?;
        let element_length = length.checked_mul(list_size).ok_or_else(|| {
            vortex_err!(
                "FixedSizeList TakeSlices element length overflow for length {length} and list size {list_size}"
            )
        })?;
        element_starts.push(element_start as u64);
        element_lengths.push(element_length as u64);
    }

    // SAFETY: row ranges have already been validated against the FSL length; multiplying by
    // `list_size` maps them to valid element ranges, and `elements_len` is `output_len * list_size`.
    Ok(unsafe {
        TakeSlicesArray::new_unchecked(
            elements.clone(),
            element_starts.into_array(),
            element_lengths.into_array(),
            elements_len,
        )
    }
    .into_array())
}
