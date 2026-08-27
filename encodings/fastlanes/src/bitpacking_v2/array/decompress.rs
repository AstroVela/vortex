// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem;
use std::mem::MaybeUninit;

use fastlanes::BitPacking;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::PrimitiveBuilder;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::patches_v2::PatchesV2;
use vortex_array::scalar::Scalar;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::BitPackedV2;
use crate::BitPackedV2ArrayExt;
use crate::FL_CHUNK_SIZE;
use crate::bitpacking_v2::array::BitPackedV2Data;
use crate::unpack_iter::BitPacked as BitPackedUnpack;

/// Unpacks a chunk-wise bit-packed array into a primitive array.
pub fn unpack_v2_array(
    array: ArrayView<'_, BitPackedV2>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    match_each_integer_ptype!(array.dtype().as_ptype(), |P| {
        unpack_v2_primitive_array::<P>(array, ctx)
    })
}

pub fn unpack_v2_primitive_array<T: BitPackedUnpack>(
    array: ArrayView<'_, BitPackedV2>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let mut builder = PrimitiveBuilder::with_capacity(array.dtype().nullability(), array.len());
    unpack_v2_into_primitive_builder::<T>(array, &mut builder, ctx)?;
    assert_eq!(builder.len(), array.len());
    Ok(builder.finish_into_primitive())
}

/// Unpack a chunk-wise bit-packed array directly into a same-typed `PrimitiveBuilder`.
pub(crate) fn unpack_v2_into_primitive_builder<T: BitPackedUnpack>(
    array: ArrayView<'_, BitPackedV2>,
    builder: &mut PrimitiveBuilder<T>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    if array.is_empty() {
        return Ok(());
    }

    let len = array.len();
    let mut uninit_range = builder.uninit_range(len);

    // SAFETY: We initialize all `len` values below via `decode_into` and the patch loop.
    unsafe {
        uninit_range.append_mask(&array.validity()?.execute_mask(len, ctx)?);
    }

    // SAFETY: `decode_into` writes a value to every slot in this range.
    let uninit_slice = unsafe { uninit_range.slice_uninit_mut(0, len) };
    decode_into::<T>(array.data(), len, uninit_slice);

    if let Some(patches) = array.patches() {
        apply_patches_v2_to_uninit_range(&mut uninit_range, &patches, ctx)?;
    }

    // SAFETY: A correct validity mask of `len` values was set via `append_mask`, and the same
    // number of values was initialized by `decode_into` (and overwritten by patches).
    unsafe {
        uninit_range.finish();
    }
    Ok(())
}

fn apply_patches_v2_to_uninit_range<T: NativePType>(
    dst: &mut vortex_array::builders::UninitRange<T>,
    patches: &PatchesV2,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    debug_assert_eq!(patches.array_len(), dst.len());
    let values = patches.values().clone().execute::<PrimitiveArray>(ctx)?;
    vortex_ensure!(values.all_valid(ctx)?, "Patch values must be all valid");
    let values = values.as_slice::<T>();
    patches.apply_each(ctx, |index, ordinal| dst.set_value(index, values[ordinal]))
}

/// Unpack every chunk at its own bit width into `output`, which must hold exactly `len` slots.
pub(crate) fn decode_into<T: BitPackedUnpack>(
    data: &BitPackedV2Data,
    len: usize,
    output: &mut [MaybeUninit<T>],
) {
    debug_assert_eq!(output.len(), len);
    let offset = data.offset() as usize;
    let padded_len = offset + len;
    let mut scratch = [const { MaybeUninit::<T>::uninit() }; FL_CHUNK_SIZE];
    let mut position = 0;

    for (chunk_idx, &bit_width) in data.bit_widths().iter().enumerate() {
        let chunk_start = chunk_idx * FL_CHUNK_SIZE;
        let start_in_chunk = offset.saturating_sub(chunk_start);
        let end_in_chunk = (padded_len - chunk_start).min(FL_CHUNK_SIZE);
        let packed = data.packed_chunk::<T::Physical>(chunk_idx);

        if start_in_chunk == 0 && end_in_chunk == FL_CHUNK_SIZE {
            // SAFETY: the destination range holds exactly 1024 elements, `packed` holds exactly
            // `128 * bit_width / size_of::<T>()` of them, and `MaybeUninit<T>` and `T::Physical`
            // share a layout.
            unsafe {
                let dst: &mut [T::Physical] =
                    mem::transmute(&mut output[position..position + FL_CHUNK_SIZE]);
                BitPacking::unchecked_unpack(bit_width as usize, packed, dst);
            }
            position += FL_CHUNK_SIZE;
            continue;
        }

        // A chunk clipped by the array's offset or by its length decodes via the scratch buffer.
        // SAFETY: as above, with a scratch buffer of exactly 1024 elements.
        unsafe {
            let dst: &mut [T::Physical] = mem::transmute(&mut scratch[..]);
            BitPacking::unchecked_unpack(bit_width as usize, packed, dst);
        }
        let taken = end_in_chunk - start_in_chunk;
        output[position..position + taken].copy_from_slice(&scratch[start_in_chunk..end_in_chunk]);
        position += taken;
    }

    debug_assert_eq!(position, len);
}

/// Unpack the single value at `index`, ignoring patches.
pub fn unpack_v2_single(array: ArrayView<'_, BitPackedV2>, index: usize) -> Scalar {
    let ptype = array.dtype().as_ptype();
    let index_in_encoded = index + array.offset() as usize;
    let chunk_idx = index_in_encoded / FL_CHUNK_SIZE;
    let index_in_chunk = index_in_encoded % FL_CHUNK_SIZE;
    let bit_width = array.bit_widths()[chunk_idx] as usize;

    let scalar: Scalar = match_each_unsigned_integer_ptype!(ptype.to_unsigned(), |P| {
        unpack_single_primitive::<P>(array.data(), chunk_idx, bit_width, index_in_chunk).into()
    });
    // Cast to fix signedness and nullability
    scalar.cast(array.dtype()).vortex_expect("cast failure")
}

fn unpack_single_primitive<T: NativePType + BitPacking>(
    data: &BitPackedV2Data,
    chunk_idx: usize,
    bit_width: usize,
    index_in_chunk: usize,
) -> T {
    let packed_chunk = data.packed_chunk::<T>(chunk_idx);
    // SAFETY: `packed_chunk` holds exactly `128 * bit_width / size_of::<T>()` elements and
    // `index_in_chunk < 1024`.
    unsafe { BitPacking::unchecked_unpack_single(bit_width, packed_chunk, index_in_chunk) }
}
