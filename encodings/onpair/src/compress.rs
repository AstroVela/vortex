// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Train + compress entry points for the OnPair encoding.

use onpair::Config;
use onpair::Offset;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::NativePType;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::OnPair;
use crate::OnPairData;

/// Compress any [`ArrayRef`] whose canonical form is a string array.
///
/// All-null inputs are returned as a [`ConstantArray`].
pub fn onpair_compress(
    array: &ArrayRef,
    config: Config,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let array = array.clone().execute::<VarBinViewArray>(ctx)?;
    let len = array.len();
    let validity = array.validity()?;
    let mask = validity.execute_mask(len, ctx)?;
    if matches!(mask.bit_buffer(), AllOr::None) {
        // CascadingCompressor handles this earlier, but direct callers can reach it.
        return Ok(ConstantArray::new(Scalar::null(array.dtype().clone()), len).into_array());
    }

    let flat_bytes: usize = array.views().iter().map(|v| v.len() as usize).sum();

    // `onpair` uses one offset width for both the byte offsets it is handed and the
    // per-row code offsets it hands back. A row emits at most one code per input byte,
    // so a width that holds the byte offsets also holds the code offsets: narrowing the
    // request to `u32` halves both arrays and lets `codes_offsets` adopt the returned
    // vector instead of re-narrowing it. Mirrors the offset-width choice FSST makes in
    // `vortex_fsst::fsst_compress`.
    if u32::try_from(flat_bytes).is_ok() {
        compress_flattened::<u32>(&array, &mask, flat_bytes, validity, config)
    } else {
        compress_flattened::<u64>(&array, &mask, flat_bytes, validity, config)
    }
}

/// Offset widths accepted by [`onpair::compress`], plus how each one lowers the
/// returned per-row code offsets into the `codes_offsets` child.
trait FlatOffset: Offset + NativePType {
    /// Narrow `n` into this offset width. `onpair_compress` picks the width from the
    /// flattened byte count, so every offset it builds fits.
    fn from_len(n: usize) -> Self;

    /// Build the `codes_offsets` child from the row offsets `onpair` returned.
    fn codes_offsets_array(row_offsets: Vec<Self>) -> ArrayRef
    where
        Self: Sized;
}

impl FlatOffset for u32 {
    fn from_len(n: usize) -> Self {
        u32::try_from(n).vortex_expect("byte offset fits the width chosen for the input")
    }

    /// Already the narrowest width Vortex stores, so adopt the vector as-is. The
    /// cascading compressor narrows it further to `u16`/`u8`.
    fn codes_offsets_array(row_offsets: Vec<Self>) -> ArrayRef {
        Buffer::from(row_offsets).into_array()
    }
}

impl FlatOffset for u64 {
    fn from_len(n: usize) -> Self {
        u64::try_from(n).vortex_expect("byte offset fits the width chosen for the input")
    }

    /// Reached only when a chunk carries more than `u32::MAX` bytes. Tokens are still
    /// often far fewer, so keep the narrowing pass rather than storing `u64` offsets
    /// that do not need the range.
    fn codes_offsets_array(row_offsets: Vec<Self>) -> ArrayRef {
        let total_tokens = row_offsets.last().copied().unwrap_or(0);
        if u32::try_from(total_tokens).is_ok() {
            Buffer::from(
                row_offsets
                    .iter()
                    .map(|&o| u32::try_from(o).vortex_expect("code boundary fits u32"))
                    .collect::<Vec<u32>>(),
            )
            .into_array()
        } else {
            Buffer::from(row_offsets).into_array()
        }
    }
}

fn compress_flattened<O: FlatOffset>(
    array: &VarBinViewArray,
    mask: &Mask,
    flat_bytes: usize,
    validity: Validity,
    config: Config,
) -> VortexResult<ArrayRef> {
    let views = array.views();
    let len = array.len();

    // TODO(francesco): we flatten because onpair training needs a contiguous `(bytes, offsets)`
    // pair. Allowing onpair to train on a slice-of-slices would let us skip this copy.
    let mut flat: Vec<u8> = Vec::with_capacity(flat_bytes);
    let mut offsets: Vec<O> = Vec::with_capacity(len + 1);
    let mut uncompressed_lengths: BufferMut<i32> = BufferMut::with_capacity(len);
    offsets.push(O::from_len(0));
    let buffers = array
        .data_buffers()
        .as_ref()
        .iter()
        .map(|b| b.as_host())
        .collect::<Vec<_>>();

    match mask.bit_buffer() {
        AllOr::All => {
            for view in views {
                let bytes = view_bytes(view, &buffers);
                flat.extend_from_slice(bytes);
                offsets.push(O::from_len(flat.len()));
                uncompressed_lengths
                    .push(i32::try_from(view.len()).vortex_expect("must fit in i32"));
            }
        }
        AllOr::None => unreachable!("all-null input handled above"),
        AllOr::Some(validity) => {
            for (view, valid) in views.iter().zip(validity.iter()) {
                if valid {
                    let bytes = view_bytes(view, &buffers);
                    flat.extend_from_slice(bytes);
                    offsets.push(O::from_len(flat.len()));
                    uncompressed_lengths
                        .push(i32::try_from(view.len()).vortex_expect("must fit in i32"));
                } else {
                    offsets.push(O::from_len(flat.len()));
                    uncompressed_lengths.push(0);
                }
            }
        }
    }

    let column = onpair::compress(&flat, &offsets, config)
        .map_err(|e| vortex_err!("OnPair compress failed: {e}"))?;
    let (dict, codes, row_offsets) = column.into_raw();
    let (dict_bytes, dict_offsets) = dict.into_raw();
    let codes_offsets = O::codes_offsets_array(row_offsets);
    let codes = Buffer::from(shrink_codes(codes)).into_array();
    // The `dict_offsets` child and the memoized widened-offsets cell share
    // this buffer, so seeding below costs no copy.
    let dict_offsets = Buffer::from(dict_offsets);

    let uncompressed_lengths = uncompressed_lengths.into_array();

    let data = OnPairData::try_new_with_dictionary(
        dict_bytes_to_buffer(dict_bytes),
        dict_offsets.clone(),
    )?;
    let encoded = OnPair::try_new_with_data(
        array.dtype().clone(),
        data,
        dict_offsets.into_array(),
        codes,
        codes_offsets,
        uncompressed_lengths,
        validity,
    )?;
    Ok(encoded.into_array())
}

fn view_bytes<'a>(view: &'a BinaryView, buffers: &'a [&ByteBuffer]) -> &'a [u8] {
    if view.is_inlined() {
        view.as_inlined().value()
    } else {
        let view_ref = view.as_view();
        &buffers[view_ref.buffer_index as usize][view_ref.as_range()]
    }
}

/// Release the slack `onpair` leaves in the code stream.
///
/// `onpair` sizes the code vector at one token per input byte and never shrinks it,
/// while real corpora tokenise several bytes at a time. `Buffer::from` adopts the
/// allocation rather than copying it, so without this the compressed array pins the
/// full worst-case reservation for as long as it lives — on TPC-H text that is
/// roughly nine times the bytes actually held. The shrink is a realloc the allocator
/// can usually satisfy in place.
fn shrink_codes(mut codes: Vec<u16>) -> Vec<u16> {
    codes.shrink_to_fit();
    codes
}

fn dict_bytes_to_buffer(dict_bytes: Vec<u8>) -> BufferHandle {
    // Align dict_bytes to 8 bytes so the segment that ultimately holds the
    // OnPair tree starts at an 8-aligned in-memory address. Without this anchor,
    // downstream primitive children may deserialize from a misaligned segment.
    let mut aligned = ByteBufferMut::with_capacity_aligned(dict_bytes.len(), Alignment::new(8));
    aligned.extend_from_slice(&dict_bytes);
    BufferHandle::new_host(aligned.freeze())
}

#[cfg(test)]
mod tests {
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::dtype::PType;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::*;
    use crate::array::OnPairArraySlotsExt;

    fn session() -> VortexSession {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    }

    /// A corpus far under `u32::MAX` bytes takes the `u32` offset path, so the
    /// `codes_offsets` child is stored at `u32` without a narrowing pass.
    #[test]
    fn codes_offsets_are_u32_for_small_inputs() -> VortexResult<()> {
        let session = session();
        let mut ctx = session.create_execution_ctx();
        let array = VarBinViewArray::from_iter_str([
            "the quick brown fox",
            "jumps over the lazy dog",
            "the quick brown fox jumps",
        ])
        .into_array();

        let encoded = onpair_compress(&array, Config::default(), &mut ctx)?;
        let onpair = encoded
            .as_opt::<OnPair>()
            .vortex_expect("input compresses to OnPair");
        assert_eq!(onpair.codes_offsets().dtype().as_ptype(), PType::U32);
        Ok(())
    }

    /// `u64` row offsets still narrow to `u32` when the token count allows it, so
    /// the wide-input path stores the same width it did before.
    #[test]
    fn u64_row_offsets_narrow_to_u32() {
        let offsets: Vec<u64> = vec![0, 3, 7, 11];
        let array = <u64 as FlatOffset>::codes_offsets_array(offsets);
        assert_eq!(array.dtype().as_ptype(), PType::U32);
    }

    /// The code stream is handed on without the worst-case slack `onpair` reserves,
    /// which `Buffer::from` would otherwise adopt and pin.
    #[test]
    fn shrink_codes_releases_reserved_slack() {
        let mut codes: Vec<u16> = Vec::with_capacity(1 << 16);
        codes.extend_from_slice(&[1, 2, 3, 4]);
        let shrunk = shrink_codes(codes);
        assert_eq!(shrunk.as_slice(), &[1, 2, 3, 4]);
        // `shrink_to_fit` may keep some excess, but never the original reservation.
        assert!(shrunk.capacity() < 1 << 16);
    }
}
