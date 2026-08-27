// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fastlanes::BitPacking;
use num_traits::PrimInt;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::patches::Patches;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::BitPackedV2;
use crate::BitPackedV2Array;
use crate::FL_CHUNK_SIZE;
use crate::bitpack_compress::find_best_bit_width;
use crate::bitpacking_v2::array::BYTES_PER_CHUNK_BIT;

/// Bit-pack an integer array, choosing a bit width independently for every FastLanes chunk.
///
/// Each 1024-element chunk is packed at the width that minimises its own packed size plus the
/// cost of the exceptions it leaves behind, so a run of small values costs few bits per value
/// even when another part of the array needs the full type width.
///
/// # Errors
///
/// Returns an error if the array is not an integer array, or if it holds negative values.
#[expect(unused_comparisons, clippy::absurd_extreme_comparisons)]
pub fn bitpack_v2_encode(
    array: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<BitPackedV2Array> {
    let ptype = array.ptype();
    if !ptype.is_int() {
        vortex_bail!(InvalidArgument: "cannot bitpack_v2_encode non-integer array of type {ptype}");
    }

    if ptype.is_signed_int() {
        let has_negative_values = match_each_integer_ptype!(ptype, |P| {
            array.statistics().compute_min::<P>(ctx).unwrap_or_default() < 0
        });
        if has_negative_values {
            vortex_bail!(InvalidArgument: "cannot bitpack_v2_encode array containing negative integers")
        }
    }

    let validity_mask = array.as_ref().validity()?.execute_mask(array.len(), ctx)?;
    let patch_validity = match array.validity()? {
        Validity::NonNullable => Validity::NonNullable,
        _ => Validity::AllValid,
    };

    let unsigned = array.reinterpret_cast(ptype.to_unsigned());
    let (packed, bit_widths, patches) = match_each_unsigned_integer_ptype!(unsigned.ptype(), |P| {
        encode_typed::<P>(
            unsigned.as_slice::<P>(),
            &validity_mask,
            ptype,
            patch_validity,
        )?
    });

    let encoded = BitPackedV2::try_new(
        BufferHandle::new_host(packed),
        BufferHandle::new_host(bit_widths.into_byte_buffer()),
        ptype,
        array.validity()?,
        patches,
        array.len(),
        0,
    )?;
    encoded.statistics().inherit_from(array.statistics());
    Ok(encoded)
}

fn encode_typed<T: NativePType + BitPacking + PrimInt>(
    values: &[T],
    validity_mask: &Mask,
    ptype: PType,
    patch_validity: Validity,
) -> VortexResult<(ByteBuffer, Buffer<u8>, Option<Patches>)> {
    let bit_widths = chunk_bit_widths::<T>(values, validity_mask, ptype)?;
    let packed = pack_chunks(values, &bit_widths);
    let patches = gather_patches::<T>(values, &bit_widths, validity_mask, ptype, patch_validity)?;
    Ok((packed, bit_widths, patches))
}

/// The number of bits needed to represent `value`.
#[inline]
fn value_bit_width<T: NativePType + PrimInt>(value: T) -> usize {
    T::PTYPE.bit_width() - PrimInt::leading_zeros(value) as usize
}

/// Choose the bit width of every FastLanes chunk independently.
///
/// Null slots hold undefined values, so they are counted as zero-width and are never patched:
/// packing truncates whatever they happen to contain.
fn chunk_bit_widths<T: NativePType + PrimInt>(
    values: &[T],
    validity_mask: &Mask,
    ptype: PType,
) -> VortexResult<Buffer<u8>> {
    let num_chunks = values.len().div_ceil(FL_CHUNK_SIZE);
    let mut bit_widths = BufferMut::<u8>::with_capacity(num_chunks);
    let mut histogram = vec![0usize; T::PTYPE.bit_width() + 1];

    match validity_mask.bit_buffer() {
        // Every value is null, so nothing needs to be stored at all.
        AllOr::None => return Ok(Buffer::zeroed(num_chunks)),
        AllOr::All => {
            for chunk in values.chunks(FL_CHUNK_SIZE) {
                histogram.fill(0);
                for &value in chunk {
                    histogram[value_bit_width(value)] += 1;
                }
                histogram[0] += FL_CHUNK_SIZE - chunk.len();
                bit_widths.push(find_best_bit_width(ptype, &histogram)?);
            }
        }
        AllOr::Some(buffer) => {
            let mut validity = buffer.iter();
            for chunk in values.chunks(FL_CHUNK_SIZE) {
                histogram.fill(0);
                for &value in chunk {
                    if validity.next().unwrap_or(false) {
                        histogram[value_bit_width(value)] += 1;
                    } else {
                        histogram[0] += 1;
                    }
                }
                histogram[0] += FL_CHUNK_SIZE - chunk.len();
                bit_widths.push(find_best_bit_width(ptype, &histogram)?);
            }
        }
    }

    Ok(bit_widths.freeze())
}

/// Pack every chunk at its own bit width into a single contiguous buffer.
///
/// The trailing chunk is zero-padded out to a full 1024 elements.
fn pack_chunks<T: NativePType + BitPacking>(values: &[T], bit_widths: &[u8]) -> ByteBuffer {
    let packed_elems = |width: u8| BYTES_PER_CHUNK_BIT * width as usize / size_of::<T>();
    let total_elems: usize = bit_widths.iter().copied().map(packed_elems).sum();

    let mut output = BufferMut::<T>::with_capacity(total_elems);
    let mut last_chunk = [T::zero(); FL_CHUNK_SIZE];

    for (chunk_idx, &width) in bit_widths.iter().enumerate() {
        let elems = packed_elems(width);
        let output_len = output.len();
        // SAFETY: the buffer was reserved for `total_elems` and `unchecked_pack` writes every
        // element of the range below.
        unsafe { output.set_len(output_len + elems) };
        if width == 0 {
            continue;
        }

        let chunk = &values[chunk_idx * FL_CHUNK_SIZE..];
        let input = if chunk.len() >= FL_CHUNK_SIZE {
            &chunk[..FL_CHUNK_SIZE]
        } else {
            last_chunk[..chunk.len()].copy_from_slice(chunk);
            last_chunk[chunk.len()..].fill(T::zero());
            &last_chunk[..]
        };

        // SAFETY: `input` holds exactly 1024 values, the output range holds exactly
        // `128 * width / size_of::<T>()` elements, and `width <= T::T`.
        unsafe {
            BitPacking::unchecked_pack(width as usize, input, &mut output[output_len..][..elems]);
        }
    }

    output.freeze().into_byte_buffer()
}

/// Collect the valid values that do not fit within their own chunk's bit width.
fn gather_patches<T: NativePType + PrimInt>(
    values: &[T],
    bit_widths: &[u8],
    validity_mask: &Mask,
    ptype: PType,
    patch_validity: Validity,
) -> VortexResult<Option<Patches>> {
    if values.len() < u8::MAX as usize {
        gather_patches_impl::<T, u8>(values, bit_widths, validity_mask, ptype, patch_validity)
    } else if values.len() < u16::MAX as usize {
        gather_patches_impl::<T, u16>(values, bit_widths, validity_mask, ptype, patch_validity)
    } else if values.len() < u32::MAX as usize {
        gather_patches_impl::<T, u32>(values, bit_widths, validity_mask, ptype, patch_validity)
    } else {
        gather_patches_impl::<T, u64>(values, bit_widths, validity_mask, ptype, patch_validity)
    }
}

fn gather_patches_impl<T: NativePType + PrimInt, P: IntegerPType>(
    values: &[T],
    bit_widths: &[u8],
    validity_mask: &Mask,
    ptype: PType,
    patch_validity: Validity,
) -> VortexResult<Option<Patches>> {
    match validity_mask.bit_buffer() {
        // All-null arrays never patch: every value is undefined.
        AllOr::None => Ok(None),
        AllOr::All => gather_patches_with_validity::<T, P, _>(
            values,
            bit_widths,
            std::iter::repeat(true),
            ptype,
            patch_validity,
        ),
        AllOr::Some(buffer) => gather_patches_with_validity::<T, P, _>(
            values,
            bit_widths,
            buffer.iter(),
            ptype,
            patch_validity,
        ),
    }
}

fn gather_patches_with_validity<
    T: NativePType + PrimInt,
    P: IntegerPType,
    I: Iterator<Item = bool>,
>(
    values: &[T],
    bit_widths: &[u8],
    mut validity: I,
    ptype: PType,
    patch_validity: Validity,
) -> VortexResult<Option<Patches>> {
    let mut indices = BufferMut::<P>::empty();
    let mut patch_values = BufferMut::<T>::empty();
    let mut chunk_offsets = BufferMut::<u64>::with_capacity(bit_widths.len());

    for (chunk_idx, chunk) in values.chunks(FL_CHUNK_SIZE).enumerate() {
        chunk_offsets.push(patch_values.len() as u64);
        let width = bit_widths[chunk_idx] as usize;
        let chunk_start = chunk_idx * FL_CHUNK_SIZE;

        for (idx, &value) in chunk.iter().enumerate() {
            let is_valid = validity.next().unwrap_or(false);
            if is_valid && value_bit_width(value) > width {
                indices.push(P::from(chunk_start + idx).vortex_expect("cast index from usize"));
                patch_values.push(value);
            }
        }
    }

    if indices.is_empty() {
        return Ok(None);
    }

    Ok(Some(Patches::new(
        values.len(),
        0,
        indices.into_array(),
        PrimitiveArray::new(patch_values.freeze(), patch_validity)
            .reinterpret_cast(ptype)
            .into_array(),
        Some(chunk_offsets.into_array()),
    )?))
}
