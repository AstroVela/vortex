// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools as _;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::List;
use crate::arrays::ListArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::TakeSlicesArray;
use crate::arrays::list::ListArrayExt;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::arrays::take_slices::TakeSlicesKernel;
use crate::arrays::take_slices::check_index_arrays;
use crate::arrays::take_slices::index_value_to_usize;
use crate::arrays::take_slices::validate_index_ranges;
use crate::dtype::IntegerPType;
use crate::dtype::PType;
use crate::executor::ExecutionCtx;
use crate::match_each_unsigned_integer_ptype;
use crate::match_smallest_offset_type;
use crate::validity::Validity;

impl TakeSlicesKernel for List {
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
    array: ArrayView<'_, List>,
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

    let offsets = array.offsets().clone().execute::<PrimitiveArray>(ctx)?;
    let offsets = offsets.reinterpret_cast(offsets.ptype().to_unsigned());
    let total_elements = match offsets.ptype() {
        PType::U8 => total_elements::<S, L, u8>(
            array.elements().len(),
            offsets.as_slice::<u8>(),
            starts.as_slice::<S>(),
            lengths.as_slice::<L>(),
        ),
        PType::U16 => total_elements::<S, L, u16>(
            array.elements().len(),
            offsets.as_slice::<u16>(),
            starts.as_slice::<S>(),
            lengths.as_slice::<L>(),
        ),
        PType::U32 => total_elements::<S, L, u32>(
            array.elements().len(),
            offsets.as_slice::<u32>(),
            starts.as_slice::<S>(),
            lengths.as_slice::<L>(),
        ),
        PType::U64 => total_elements::<S, L, u64>(
            array.elements().len(),
            offsets.as_slice::<u64>(),
            starts.as_slice::<S>(),
            lengths.as_slice::<L>(),
        ),
        _ => unreachable!("offsets were reinterpreted to an unsigned integer ptype"),
    }?;

    let gathered = match_smallest_offset_type!(total_elements, |OutputOffset| {
        match offsets.ptype() {
            PType::U8 => gather_list::<S, L, u8, OutputOffset>(
                array.elements(),
                offsets.as_slice::<u8>(),
                starts.as_slice::<S>(),
                lengths.as_slice::<L>(),
                output_len,
                total_elements,
            ),
            PType::U16 => gather_list::<S, L, u16, OutputOffset>(
                array.elements(),
                offsets.as_slice::<u16>(),
                starts.as_slice::<S>(),
                lengths.as_slice::<L>(),
                output_len,
                total_elements,
            ),
            PType::U32 => gather_list::<S, L, u32, OutputOffset>(
                array.elements(),
                offsets.as_slice::<u32>(),
                starts.as_slice::<S>(),
                lengths.as_slice::<L>(),
                output_len,
                total_elements,
            ),
            PType::U64 => gather_list::<S, L, u64, OutputOffset>(
                array.elements(),
                offsets.as_slice::<u64>(),
                starts.as_slice::<S>(),
                lengths.as_slice::<L>(),
                output_len,
                total_elements,
            ),
            _ => unreachable!("offsets were reinterpreted to an unsigned integer ptype"),
        }
    })?;

    let starts = starts.into_array();
    let lengths = lengths.into_array();
    let validity = array
        .validity()?
        .take_slices(&starts, &lengths, output_len)?;

    // SAFETY: output offsets are rebuilt from valid, monotonic source offsets; output elements are
    // exactly the gathered element ranges referenced by those offsets; validity has output_len rows.
    Ok(
        unsafe { ListArray::new_unchecked(gathered.elements, gathered.offsets, validity) }
            .into_array(),
    )
}

struct GatheredList {
    elements: ArrayRef,
    offsets: ArrayRef,
}

fn total_elements<S, L, Offset>(
    elements_len: usize,
    offsets: &[Offset],
    starts: &[S],
    lengths: &[L],
) -> VortexResult<usize>
where
    S: IntegerPType,
    L: IntegerPType,
    Offset: IntegerPType,
{
    let mut total = 0usize;
    for (&start, &length) in starts.iter().zip_eq(lengths) {
        let start = index_value_to_usize("start", start)?;
        let length = index_value_to_usize("length", length)?;
        let end = start + length;
        if length == 0 {
            continue;
        }

        let element_start = index_value_to_usize("offset", offsets[start])?;
        let element_end = index_value_to_usize("offset", offsets[end])?;
        vortex_ensure!(
            element_start <= element_end && element_end <= elements_len,
            "List offsets range {element_start}..{element_end} exceeds elements length {elements_len}",
        );
        total = total
            .checked_add(element_end - element_start)
            .ok_or_else(|| vortex_err!("TakeSlicesArray List output elements length overflow"))?;
    }
    Ok(total)
}

fn gather_list<S, L, Offset, OutputOffset>(
    elements: &ArrayRef,
    offsets: &[Offset],
    starts: &[S],
    lengths: &[L],
    output_len: usize,
    total_elements: usize,
) -> VortexResult<GatheredList>
where
    S: IntegerPType,
    L: IntegerPType,
    Offset: IntegerPType,
    OutputOffset: IntegerPType,
{
    let offsets_capacity = output_len
        .checked_add(1)
        .ok_or_else(|| vortex_err!("TakeSlicesArray List offsets length overflow"))?;
    let mut new_offsets = BufferMut::<OutputOffset>::with_capacity(offsets_capacity);
    let mut element_starts = BufferMut::<u64>::with_capacity(starts.len());
    let mut element_lengths = BufferMut::<u64>::with_capacity(lengths.len());
    let mut output_elements = 0usize;

    new_offsets.push(OutputOffset::zero());
    for (&start, &length) in starts.iter().zip_eq(lengths) {
        let start = index_value_to_usize("start", start)?;
        let length = index_value_to_usize("length", length)?;
        let end = start + length;
        if length == 0 {
            continue;
        }

        let element_start = index_value_to_usize("offset", offsets[start])?;
        let element_end = index_value_to_usize("offset", offsets[end])?;
        for &offset in &offsets[start + 1..=end] {
            let offset = index_value_to_usize("offset", offset)?;
            let relative = offset
                .checked_sub(element_start)
                .ok_or_else(|| vortex_err!("List offsets are not monotonic at offset {offset}"))?;
            let output_offset = output_elements.checked_add(relative).ok_or_else(|| {
                vortex_err!("TakeSlicesArray List output elements length overflow")
            })?;
            new_offsets.push(new_offset_value::<OutputOffset>(output_offset)?);
        }

        let element_length = element_end - element_start;
        element_starts.push(element_start as u64);
        element_lengths.push(element_length as u64);
        output_elements = output_elements
            .checked_add(element_length)
            .ok_or_else(|| vortex_err!("TakeSlicesArray List output elements length overflow"))?;
    }
    debug_assert_eq!(output_elements, total_elements);

    let offsets = PrimitiveArray::new(new_offsets.freeze(), Validity::NonNullable).into_array();
    // SAFETY: element ranges are derived from validated list offsets, and total_elements is the sum
    // of all gathered element ranges.
    let elements = unsafe {
        TakeSlicesArray::new_unchecked(
            elements.clone(),
            element_starts.into_array(),
            element_lengths.into_array(),
            total_elements,
        )
    }
    .into_array();

    Ok(GatheredList { elements, offsets })
}

fn new_offset_value<T: IntegerPType>(value: usize) -> VortexResult<T> {
    T::from(value).ok_or_else(|| {
        vortex_err!(
            "TakeSlicesArray List offset value {value} does not fit in {}",
            T::PTYPE
        )
    })
}
