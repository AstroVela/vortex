// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Train + compress entry points for the OnPair encoding.

use onpair::Config;
use onpair::Offset;
use onpair::Token;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::VarBinView;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_array::buffer::BufferHandle;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::OnPair;
use crate::OnPairData;

/// Alignment the dictionary blob is published at, so the segment that ultimately
/// holds the OnPair tree starts at an 8-aligned in-memory address. Without this
/// anchor, downstream primitive children may deserialize from a misaligned
/// segment.
const DICT_ALIGNMENT: Alignment = Alignment::new(8);

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

    // Byte offsets into the gathered corpus are handed to `onpair::compress`,
    // which echoes them back as the row layer over the code stream. Picking the
    // narrowest width that spans the corpus therefore does double duty: it
    // halves the offsets written during the gather, and it lets the
    // `codes_offsets` child adopt the returned row layer instead of re-collecting
    // it at another width.
    let corpus_bytes = valid_bytes(array.as_view(), &mask);
    if u32::try_from(corpus_bytes).is_ok() {
        compress_at_offset_width::<u32>(array.as_view(), &mask, corpus_bytes, validity, config)
    } else {
        compress_at_offset_width::<u64>(array.as_view(), &mask, corpus_bytes, validity, config)
    }
}

/// Total byte length of the array's valid rows — the exact size of the corpus
/// the gather below produces.
fn valid_bytes(array: ArrayView<'_, VarBinView>, mask: &Mask) -> usize {
    let views = array.views();
    match mask.bit_buffer() {
        AllOr::All => views.iter().map(|v| v.len() as usize).sum(),
        AllOr::None => 0,
        AllOr::Some(validity) => views
            .iter()
            .zip(validity.iter())
            .filter(|&(_, valid)| valid)
            .map(|(v, _)| v.len() as usize)
            .sum(),
    }
}

fn compress_at_offset_width<O: CodeOffset>(
    array: ArrayView<'_, VarBinView>,
    mask: &Mask,
    corpus_bytes: usize,
    validity: Validity,
    config: Config,
) -> VortexResult<ArrayRef> {
    let len = array.len();

    // TODO(francesco): we flatten because onpair training needs a contiguous
    // `(bytes, offsets)` pair. Allowing onpair to train and encode over a row
    // source — an index-addressable slice-of-slices — would let us skip this
    // copy, which is the largest single allocation on the compress path.
    let mut flat: Vec<u8> = Vec::with_capacity(corpus_bytes);
    let mut offsets: Vec<O> = Vec::with_capacity(len + 1);
    let mut uncompressed_lengths: BufferMut<i32> = BufferMut::with_capacity(len);
    offsets.push(O::from_usize(0));
    let views = array.views();
    let buffers = array
        .data_buffers()
        .as_ref()
        .iter()
        .map(|b| b.as_host())
        .collect::<Vec<_>>();

    match mask.bit_buffer() {
        AllOr::All => {
            for view in views {
                flat.extend_from_slice(view_bytes(view, &buffers));
                offsets.push(O::from_usize(flat.len()));
                uncompressed_lengths
                    .push(i32::try_from(view.len()).vortex_expect("must fit in i32"));
            }
        }
        AllOr::None => unreachable!("all-null input handled above"),
        AllOr::Some(validity) => {
            for (view, valid) in views.iter().zip(validity.iter()) {
                if valid {
                    flat.extend_from_slice(view_bytes(view, &buffers));
                    uncompressed_lengths
                        .push(i32::try_from(view.len()).vortex_expect("must fit in i32"));
                } else {
                    uncompressed_lengths.push(0);
                }
                offsets.push(O::from_usize(flat.len()));
            }
        }
    }

    let column = onpair::compress(&flat, &offsets, config)
        .map_err(|e| vortex_err!("OnPair compress failed: {e}"))?;
    // The gathered corpus is dead once the column exists; release it before the
    // slot children below allocate.
    drop(flat);
    drop(offsets);

    let (dict, codes, row_offsets) = column.into_raw();
    let (dict_bytes, dict_offsets) = dict.into_raw();
    let codes_offsets = O::into_codes_offsets(row_offsets);
    let codes = codes_buffer(codes).into_array();
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

/// Adopt the dictionary blob as an 8-aligned buffer; see [`DICT_ALIGNMENT`].
///
/// The allocator hands back 8-aligned blocks for a dictionary-sized request, so
/// claiming the alignment is normally free — [`Buffer::aligned`] falls back to a
/// copy only when the allocation happens not to satisfy it.
fn dict_bytes_to_buffer(dict_bytes: Vec<u8>) -> BufferHandle {
    BufferHandle::new_host(Buffer::from(dict_bytes).aligned(DICT_ALIGNMENT))
}

/// Ratio of unused capacity to length past which the code stream is worth
/// reallocating. Upstream sizes its code vector for the worst case of one token
/// per corpus byte, so on any input OnPair actually compresses the vector comes
/// back several times larger than it needs to be — around 7x on a URL column.
/// The slack is untouched, so it costs no resident memory, but the allocation
/// is held for as long as the encoded array is, and shrinking it is a bounded
/// cost the gate keeps proportional to the codes actually kept.
const CODES_SHRINK_SLACK_RATIO: usize = 2;

/// Adopt the code stream, releasing upstream's worst-case over-allocation first.
fn codes_buffer(mut codes: Vec<Token>) -> Buffer<Token> {
    if codes.capacity() > codes.len().saturating_mul(CODES_SHRINK_SLACK_RATIO) {
        codes.shrink_to_fit();
    }
    Buffer::from(codes)
}

/// The two offset widths [`onpair::compress`] accepts, plus the conversion from
/// its row layer to the `codes_offsets` child.
///
/// The row layer comes back at the same width as the byte offsets handed in, so
/// the width chosen in [`onpair_compress`] decides whether that child is adopted
/// or rebuilt.
trait CodeOffset: Offset + Send {
    /// Build the `codes_offsets` child from the library's per-row code
    /// boundaries.
    fn into_codes_offsets(row_offsets: Vec<Self>) -> ArrayRef;
}

impl CodeOffset for u32 {
    /// Adopted without copying — `u32` is already the width the cascading
    /// compressor narrows from.
    fn into_codes_offsets(row_offsets: Vec<Self>) -> ArrayRef {
        Buffer::from(row_offsets).into_array()
    }
}

impl CodeOffset for u64 {
    /// Reached only for a corpus larger than `u32::MAX` bytes. Store the
    /// narrowest of `u32`/`u64` that holds the largest boundary: `row_offsets`
    /// is non-decreasing, so its last entry is that maximum and one bound check
    /// picks the width. `u64` engages only when a single chunk carries more than
    /// `u32::MAX` tokens.
    fn into_codes_offsets(row_offsets: Vec<Self>) -> ArrayRef {
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

#[cfg(test)]
mod tests {
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::PType;
    use vortex_error::VortexResult;

    use super::*;

    /// `u32` row offsets are adopted as-is: the child keeps the width the
    /// library returned, so nothing is re-collected.
    #[test]
    fn u32_row_offsets_are_adopted() -> VortexResult<()> {
        let offsets = u32::into_codes_offsets(vec![0u32, 3, 3, 9]);
        assert_eq!(offsets.dtype().as_ptype(), PType::U32);
        let mut ctx = array_session().create_execution_ctx();
        let values = offsets.execute::<PrimitiveArray>(&mut ctx)?;
        assert_eq!(values.as_slice::<u32>(), &[0, 3, 3, 9]);
        Ok(())
    }

    /// The `u64` width is only reached for a corpus larger than `u32::MAX`
    /// bytes, which is impractical to build in a test. Drive the conversion
    /// directly instead: boundaries that fit `u32` narrow to it.
    #[test]
    fn u64_row_offsets_narrow_when_they_fit() -> VortexResult<()> {
        let offsets = u64::into_codes_offsets(vec![0u64, 3, 3, 9]);
        assert_eq!(offsets.dtype().as_ptype(), PType::U32);
        let mut ctx = array_session().create_execution_ctx();
        let values = offsets.execute::<PrimitiveArray>(&mut ctx)?;
        assert_eq!(values.as_slice::<u32>(), &[0, 3, 3, 9]);
        Ok(())
    }

    /// More than `u32::MAX` tokens in one chunk keeps the boundaries at `u64`.
    #[test]
    fn u64_row_offsets_stay_wide_when_they_must() -> VortexResult<()> {
        let last = u32::MAX as u64 + 1;
        let offsets = u64::into_codes_offsets(vec![0u64, last]);
        assert_eq!(offsets.dtype().as_ptype(), PType::U64);
        let mut ctx = array_session().create_execution_ctx();
        let values = offsets.execute::<PrimitiveArray>(&mut ctx)?;
        assert_eq!(values.as_slice::<u64>(), &[0, last]);
        Ok(())
    }

    /// An empty row layer cannot happen (the layer always carries the leading
    /// zero), but the conversion must not index out of bounds if it does.
    #[test]
    fn empty_u64_row_offsets_do_not_panic() {
        assert_eq!(u64::into_codes_offsets(Vec::new()).len(), 0);
    }
}
