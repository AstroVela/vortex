// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools as _;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBin;
use crate::arrays::VarBinArray;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::arrays::take_slices::TakeSlicesKernel;
use crate::arrays::take_slices::check_index_arrays;
use crate::arrays::take_slices::index_value_to_usize;
use crate::arrays::take_slices::validate_index_ranges;
use crate::arrays::varbin::VarBinArrayExt;
use crate::arrays::varbin::compute::take::taken_offset_ptype;
use crate::dtype::DType;
use crate::dtype::IntegerPType;
use crate::dtype::PType;
use crate::executor::ExecutionCtx;
use crate::match_each_unsigned_integer_ptype;
use crate::validity::Validity;

impl TakeSlicesKernel for VarBin {
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
    array: ArrayView<'_, VarBin>,
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
    let out_offset_ptype = taken_offset_ptype(offsets.ptype());
    let offsets = offsets.reinterpret_cast(offsets.ptype().to_unsigned());

    let result = match offsets.ptype() {
        PType::U8 => gather_varbin::<S, L, u8, u32>(
            array.dtype().clone(),
            offsets.as_slice::<u8>(),
            array.bytes().as_slice(),
            starts.as_slice::<S>(),
            lengths.as_slice::<L>(),
            output_len,
            out_offset_ptype,
        ),
        PType::U16 => gather_varbin::<S, L, u16, u32>(
            array.dtype().clone(),
            offsets.as_slice::<u16>(),
            array.bytes().as_slice(),
            starts.as_slice::<S>(),
            lengths.as_slice::<L>(),
            output_len,
            out_offset_ptype,
        ),
        PType::U32 => gather_varbin::<S, L, u32, u32>(
            array.dtype().clone(),
            offsets.as_slice::<u32>(),
            array.bytes().as_slice(),
            starts.as_slice::<S>(),
            lengths.as_slice::<L>(),
            output_len,
            out_offset_ptype,
        ),
        PType::U64 => gather_varbin::<S, L, u64, u64>(
            array.dtype().clone(),
            offsets.as_slice::<u64>(),
            array.bytes().as_slice(),
            starts.as_slice::<S>(),
            lengths.as_slice::<L>(),
            output_len,
            out_offset_ptype,
        ),
        _ => unreachable!("offsets were reinterpreted to an unsigned integer ptype"),
    }?;

    let starts = starts.into_array();
    let lengths = lengths.into_array();
    let validity = array
        .validity()?
        .take_slices(&starts, &lengths, output_len)?;

    // SAFETY: output offsets are built from valid input offsets, start at zero, are monotonically
    // non-decreasing, and the copied data buffer has exactly the referenced byte length.
    unsafe {
        Ok(
            VarBinArray::new_unchecked(
                result.offsets,
                result.data.freeze(),
                result.dtype,
                validity,
            )
            .into_array(),
        )
    }
}

struct GatheredVarBin {
    dtype: DType,
    offsets: ArrayRef,
    data: ByteBufferMut,
}

fn gather_varbin<S, L, Offset, NewOffset>(
    dtype: DType,
    offsets: &[Offset],
    data: &[u8],
    starts: &[S],
    lengths: &[L],
    output_len: usize,
    out_offset_ptype: PType,
) -> VortexResult<GatheredVarBin>
where
    S: IntegerPType,
    L: IntegerPType,
    Offset: IntegerPType,
    NewOffset: IntegerPType,
{
    let mut new_offsets = BufferMut::<NewOffset>::with_capacity(output_len + 1);
    new_offsets.push(NewOffset::zero());
    let mut output_bytes = 0usize;

    for (&start, &length) in starts.iter().zip_eq(lengths) {
        let start = index_value_to_usize("start", start)?;
        let length = index_value_to_usize("length", length)?;
        let end = start + length;
        if length == 0 {
            continue;
        }

        let byte_start = index_value_to_usize("offset", offsets[start])?;
        let byte_end = index_value_to_usize("offset", offsets[end])?;
        vortex_ensure!(
            byte_start <= byte_end && byte_end <= data.len(),
            "VarBin offsets range {byte_start}..{byte_end} exceeds data length {}",
            data.len()
        );

        for &offset in &offsets[start + 1..=end] {
            let offset = index_value_to_usize("offset", offset)?;
            let relative = offset.checked_sub(byte_start).ok_or_else(|| {
                vortex_err!("VarBin offsets are not monotonic at offset {offset}")
            })?;
            let output_offset = output_bytes
                .checked_add(relative)
                .ok_or_else(|| vortex_err!("TakeSlicesArray VarBin output byte length overflow"))?;
            new_offsets.push(new_offset_value::<NewOffset>(output_offset)?);
        }

        output_bytes = output_bytes
            .checked_add(byte_end - byte_start)
            .ok_or_else(|| vortex_err!("TakeSlicesArray VarBin output byte length overflow"))?;
    }

    let mut new_data = ByteBufferMut::with_capacity(output_bytes);
    for (&start, &length) in starts.iter().zip_eq(lengths) {
        let start = index_value_to_usize("start", start)?;
        let length = index_value_to_usize("length", length)?;
        let end = start + length;
        if length == 0 {
            continue;
        }

        let byte_start = index_value_to_usize("offset", offsets[start])?;
        let byte_end = index_value_to_usize("offset", offsets[end])?;
        new_data.extend_from_slice(&data[byte_start..byte_end]);
    }

    let offsets = PrimitiveArray::new(new_offsets.freeze(), Validity::NonNullable)
        .reinterpret_cast(out_offset_ptype)
        .into_array();
    Ok(GatheredVarBin {
        dtype,
        offsets,
        data: new_data,
    })
}

fn new_offset_value<T: IntegerPType>(value: usize) -> VortexResult<T> {
    T::from(value).ok_or_else(|| {
        vortex_err!(
            "TakeSlicesArray VarBin offset value {value} does not fit in {}",
            T::PTYPE
        )
    })
}
