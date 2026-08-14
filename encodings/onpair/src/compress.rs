// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Train + compress entry points for the OnPair encoding.

use onpair::Config;
use onpair::Parser;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_array::buffer::BufferHandle;
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

use crate::OnPair;
use crate::OnPairData;

/// Compress any [`ArrayRef`] whose canonical form is a string array.
///
/// Trains a dictionary on `array` and immediately encodes `array` with it, so the
/// result carries its own dictionary. To reuse one dictionary across several
/// arrays, train with [`onpair_train`] and encode with [`onpair_encode`].
///
/// All-null inputs are returned as a [`ConstantArray`].
pub fn onpair_compress(
    array: &ArrayRef,
    config: Config,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let array = array.clone().execute::<VarBinViewArray>(ctx)?;
    let flat = FlatStrings::new(&array, ctx)?;
    if flat.all_null {
        // CascadingCompressor handles this earlier, but direct callers can reach it.
        return Ok(
            ConstantArray::new(Scalar::null(array.dtype().clone()), array.len()).into_array(),
        );
    }

    let column = onpair::compress(&flat.bytes, &flat.offsets, config)
        .map_err(|e| vortex_err!("OnPair compress failed: {e}"))?;
    let (dict, codes, row_offsets) = column.into_raw();
    let (dict_bytes, dict_offsets) = dict.into_raw();
    // The `dict_offsets` child and the memoized widened-offsets cell share
    // this buffer, so seeding below costs no copy.
    let dict_offsets = Buffer::from(dict_offsets);

    let data = OnPairData::try_new_with_dictionary(
        dict_bytes_to_buffer(&dict_bytes),
        dict_offsets.clone(),
    )?;
    let encoded = OnPair::try_new_with_data(
        array.dtype().clone(),
        data,
        dict_offsets.into_array(),
        Buffer::from(codes).into_array(),
        codes_offsets_array(&row_offsets),
        flat.uncompressed_lengths.into_array(),
        flat.validity,
    )?;
    Ok(encoded.into_array())
}

/// Train an OnPair dictionary on the string bytes of `array`.
///
/// [`onpair_compress`] fuses training and encoding, which suits the per-chunk
/// array encoding: every chunk carries the dictionary it was trained on. Callers
/// that want one dictionary to span many arrays — the `vortex.onpair` layout,
/// across the chunks of a column — train once here and then call
/// [`onpair_encode`] per array.
///
/// Reusing a dictionary is always correct: an OnPair dictionary always contains
/// all 256 single-byte tokens, so every string encodes under any dictionary. Only
/// the compression ratio varies.
///
/// Read the trained dictionary's serialized form back through
/// [`Parser::dict`](onpair::Parser::dict) — `bytes()` is already read-padded and
/// `offsets()` is the matching index, exactly what the OnPair array holds in
/// buffer 0 and its `dict_offsets` child.
pub fn onpair_train(
    array: &ArrayRef,
    config: Config,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Parser> {
    let array = array.clone().execute::<VarBinViewArray>(ctx)?;
    let flat = FlatStrings::new(&array, ctx)?;
    Parser::train(&flat.bytes, &flat.offsets, config)
        .map_err(|e| vortex_err!("OnPair train failed: {e}"))
}

/// Encode `array` against an already-trained dictionary.
///
/// The returned `codes_offsets` are local to `array`, so a caller concatenating
/// the codes of several arrays rebases them onto its running token total itself.
pub fn onpair_encode(
    parser: &Parser,
    array: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<OnPairEncodedChunk> {
    let array = array.clone().execute::<VarBinViewArray>(ctx)?;
    let flat = FlatStrings::new(&array, ctx)?;
    // `Parser::parse` clones the whole dictionary into the returned `Column`, and
    // we drop it. That is ~80 KB per call (12 dictionary bits, tokens of at most
    // 16 bytes) against a parse that walks the array's entire byte payload, so it
    // stays. An upstream codes-only entry point would make this a one-line change.
    let column = parser
        .parse(&flat.bytes, &flat.offsets)
        .map_err(|e| vortex_err!("OnPair parse failed: {e}"))?;
    let (_dict, codes, codes_offsets) = column.into_raw();
    Ok(OnPairEncodedChunk {
        codes: Buffer::from(codes),
        codes_offsets: Buffer::from(codes_offsets),
        uncompressed_lengths: flat.uncompressed_lengths.freeze(),
        validity: flat.validity,
    })
}

/// One array encoded against a shared dictionary: the OnPair array's per-row
/// children, minus the dictionary the caller's [`Parser`] owns.
pub struct OnPairEncodedChunk {
    /// The flat token stream.
    pub codes: Buffer<u16>,
    /// Token boundaries local to this chunk, of length `array.len() + 1`.
    pub codes_offsets: Buffer<u64>,
    /// Decoded byte length per row, zero for null rows.
    pub uncompressed_lengths: Buffer<i32>,
    /// Validity of the encoded array.
    pub validity: Validity,
}

/// A canonical string array flattened into the contiguous `(bytes, offsets)` pair
/// the OnPair trainer and parser both consume, plus the per-row bookkeeping an
/// encoded array needs alongside its codes.
///
/// Null rows contribute no bytes, a repeated offset, and a zero length.
struct FlatStrings {
    bytes: Vec<u8>,
    offsets: Vec<u64>,
    uncompressed_lengths: BufferMut<i32>,
    validity: Validity,
    /// Set when every row is null, so there is nothing to train on or encode.
    all_null: bool,
}

impl FlatStrings {
    // TODO(francesco): we flatten because onpair training needs a contiguous `(bytes, offsets)`
    // pair. Allowing onpair to train on a slice-of-slices would let us skip this copy.
    fn new(array: &VarBinViewArray, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        let len = array.len();
        let validity = array.validity()?;
        let mask = validity.execute_mask(len, ctx)?;

        let views = array.views();
        let flat_bytes: usize = views.iter().map(|v| v.len() as usize).sum();

        let mut bytes: Vec<u8> = Vec::with_capacity(flat_bytes);
        let mut offsets: Vec<u64> = Vec::with_capacity(len + 1);
        let mut uncompressed_lengths: BufferMut<i32> = BufferMut::with_capacity(len);
        offsets.push(0);
        let buffers = array
            .data_buffers()
            .as_ref()
            .iter()
            .map(|b| b.as_host())
            .collect::<Vec<_>>();

        let mut all_null = false;
        match mask.bit_buffer() {
            AllOr::All => {
                for view in views {
                    bytes.extend_from_slice(view_bytes(view, &buffers));
                    offsets
                        .push(u64::try_from(bytes.len()).vortex_expect("offset must fit in u64"));
                    uncompressed_lengths
                        .push(i32::try_from(view.len()).vortex_expect("must fit in i32"));
                }
            }
            AllOr::None => {
                all_null = true;
                offsets.resize(len + 1, 0);
                uncompressed_lengths.push_n(0, len);
            }
            AllOr::Some(validity) => {
                for (view, valid) in views.iter().zip(validity.iter()) {
                    if valid {
                        bytes.extend_from_slice(view_bytes(view, &buffers));
                        uncompressed_lengths
                            .push(i32::try_from(view.len()).vortex_expect("must fit in i32"));
                    } else {
                        uncompressed_lengths.push(0);
                    }
                    offsets
                        .push(u64::try_from(bytes.len()).vortex_expect("offset must fit in u64"));
                }
            }
        }

        Ok(Self {
            bytes,
            offsets,
            uncompressed_lengths,
            validity,
            all_null,
        })
    }
}

fn view_bytes<'a>(view: &'a BinaryView, buffers: &'a [&ByteBuffer]) -> &'a [u8] {
    if view.is_inlined() {
        view.as_inlined().value()
    } else {
        let view_ref = view.as_view();
        &buffers[view_ref.buffer_index as usize][view_ref.as_range()]
    }
}

/// Copy the dictionary blob into an 8-byte-aligned buffer handle.
///
/// Alignment anchors the segment that ultimately holds the OnPair tree at an
/// 8-aligned in-memory address. Without this anchor, downstream primitive
/// children may deserialize from a misaligned segment.
fn dict_bytes_to_buffer(dict_bytes: &[u8]) -> BufferHandle {
    let mut aligned = ByteBufferMut::with_capacity_aligned(dict_bytes.len(), Alignment::new(8));
    aligned.extend_from_slice(dict_bytes);
    BufferHandle::new_host(aligned.freeze())
}

/// Build the `codes_offsets` child from the library's per-row code boundaries,
/// storing the narrowest of `u32`/`u64` that holds the largest boundary.
/// `row_offsets` is non-decreasing, so its last entry is that maximum and one
/// bound check picks the width. `u32` covers the common case (the cascading
/// compressor narrows it further to `u16`/`u8`); `u64` engages only when a
/// single chunk carries more than `u32::MAX` tokens, matching the `u64` byte
/// offsets accepted at compression.
fn codes_offsets_array(row_offsets: &[u64]) -> ArrayRef {
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
        Buffer::from(row_offsets.to_vec()).into_array()
    }
}
