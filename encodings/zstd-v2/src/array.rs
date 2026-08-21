// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::mem::MaybeUninit;
use std::ops::Range;
use std::sync::Arc;

use prost::Message as _;
use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::EqMode;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::array_slots;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::varbinview::build_views::BinaryView;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::scalar::Scalar;
use vortex_array::serde::ArrayChildren;
use vortex_array::smallvec::smallvec;
use vortex_array::validity::Validity;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTable;
use vortex_array::vtable::child_to_validity;
use vortex_array::vtable::validity_to_child;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;
use zstd::zstd_safe::WriteBuf;

use crate::ZstdV2FrameMetadata;
use crate::ZstdV2Metadata;

/// Width of one stored length.
type ValueLen = u32;

/// A view's offset is a `u32` into its buffer, so no frame may exceed what one can address.
const MAX_FRAME_BYTES: usize = u32::MAX as usize;

/// Encoding marker for zstd compression with a separate lengths stream.
#[derive(Clone, Debug)]
pub struct ZstdV2;

/// A [`ZstdV2`]-encoded Vortex array.
pub type ZstdV2Array = Array<ZstdV2>;

#[array_slots(ZstdV2)]
pub struct ZstdV2Slots {
    /// The validity bitmap indicating which elements are non-null.
    #[slot(0)]
    pub validity: Option<ArrayRef>,
}

/// Encoding-specific data for a [`ZstdV2Array`].
#[derive(Clone, Debug)]
pub struct ZstdV2Data {
    /// One zstd frame holding a `u32` length per stored value.
    pub(crate) lengths: ByteBuffer,
    /// The value bytes, in order, cut into independently compressed frames.
    pub(crate) frames: Vec<ByteBuffer>,
    pub(crate) metadata: ZstdV2Metadata,
    unsliced_n_rows: usize,
    slice_start: usize,
    slice_stop: usize,
}

impl Display for ZstdV2Data {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "nrows: {}, slice: {}..{}",
            self.unsliced_n_rows, self.slice_start, self.slice_stop
        )
    }
}

impl ArrayHash for ZstdV2Data {
    fn array_hash<H: Hasher>(&self, state: &mut H, accuracy: EqMode) {
        self.lengths.array_hash(state, accuracy);
        for frame in &self.frames {
            frame.array_hash(state, accuracy);
        }
        self.unsliced_n_rows.hash(state);
        self.slice_start.hash(state);
        self.slice_stop.hash(state);
    }
}

impl ArrayEq for ZstdV2Data {
    fn array_eq(&self, other: &Self, accuracy: EqMode) -> bool {
        self.lengths.array_eq(&other.lengths, accuracy)
            && self.frames.len() == other.frames.len()
            && self
                .frames
                .iter()
                .zip(&other.frames)
                .all(|(a, b)| a.array_eq(b, accuracy))
            && self.unsliced_n_rows == other.unsliced_n_rows
            && self.slice_start == other.slice_start
            && self.slice_stop == other.slice_stop
    }
}

/// The stored values one frame holds, and the bytes it decompresses to.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FrameSpan {
    pub(crate) value_start: usize,
    pub(crate) n_values: usize,
    pub(crate) uncompressed_size: usize,
}

impl FrameSpan {
    pub(crate) fn value_stop(&self) -> usize {
        self.value_start + self.n_values
    }
}

/// A zstd output buffer over uninitialized spare capacity.
///
/// `decompress_to_buffer` writes through a raw pointer and reports how many bytes it produced, so
/// it never reads its destination — but handing it a `&mut [u8]` covering uninitialized memory
/// would be undefined behaviour regardless. [`WriteBuf`] is the interface zstd provides for this
/// case, and it keeps zeroing the buffer off the decode path.
struct UninitDestination<'a> {
    spare: &'a mut [MaybeUninit<u8>],
    filled: usize,
}

// SAFETY: `as_mut_ptr` and `capacity` describe the whole spare region, so zstd only ever writes
// within it, and `filled_until` merely records the count it reports. `as_slice` is bounded by that
// count, so it never exposes a byte zstd did not write.
unsafe impl WriteBuf for UninitDestination<'_> {
    fn as_slice(&self) -> &[u8] {
        // SAFETY: zstd reported writing `filled` bytes from the start of `spare`.
        unsafe { std::slice::from_raw_parts(self.spare.as_ptr().cast::<u8>(), self.filled) }
    }

    fn capacity(&self) -> usize {
        self.spare.len()
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.spare.as_mut_ptr().cast::<u8>()
    }

    unsafe fn filled_until(&mut self, n: usize) {
        self.filled = n;
    }
}

impl ZstdV2 {
    /// Construct a [`ZstdV2Array`] from validated compressed data and validity.
    pub fn try_new(
        dtype: DType,
        data: ZstdV2Data,
        validity: Validity,
    ) -> VortexResult<ZstdV2Array> {
        let len = data.len();
        data.validate(&dtype, len, &validity)?;
        let slots = smallvec![validity_to_child(&validity, data.unsliced_n_rows)];
        Array::try_from_parts(ArrayParts::new(ZstdV2, dtype, len, data).with_slots(slots))
    }

    /// Compress a [`VarBinViewArray`], storing its lengths apart from its value bytes.
    ///
    /// `values_per_frame` of `0` puts every value in a single frame. Frames are also cut whenever
    /// one would grow past what a view's `u32` offset can address.
    pub fn from_var_bin_view(
        vbv: &VarBinViewArray,
        level: i32,
        values_per_frame: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ZstdV2Array> {
        let validity = vbv.validity()?;
        let data = ZstdV2Data::from_var_bin_view(vbv, level, values_per_frame, ctx)?;
        Self::try_new(vbv.dtype().clone(), data, validity)
    }
}

impl VTable for ZstdV2 {
    type TypedArrayData = ZstdV2Data;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.zstd.v2");
        *ID
    }

    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let validity =
            child_to_validity(slots[ZstdV2Slots::VALIDITY].as_ref(), dtype.nullability());
        data.validate(dtype, len, &validity)
    }

    fn nbuffers(array: ArrayView<'_, Self>) -> usize {
        1 + array.frames.len()
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        if idx == 0 {
            BufferHandle::new_host(array.lengths.clone())
        } else {
            BufferHandle::new_host(array.frames[idx - 1].clone())
        }
    }

    fn buffer_name(array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        let _ = array;
        Some(if idx == 0 {
            "lengths".to_string()
        } else {
            format!("frame_{}", idx - 1)
        })
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        let Some((lengths, frames)) = buffers.split_first() else {
            vortex_bail!("Expected a lengths buffer");
        };
        let mut data = array.data().clone();
        data.lengths = lengths.clone().try_to_host_sync()?;
        data.frames = frames
            .iter()
            .map(|buffer| buffer.clone().try_to_host_sync())
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(
            ArrayParts::new(self.clone(), array.dtype().clone(), array.len(), data)
                .with_slots(array.slots().iter().cloned().collect()),
        )
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(array.metadata.clone().encode_to_vec()))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        let metadata = ZstdV2Metadata::decode(metadata)?;
        let validity = if children.is_empty() {
            Validity::from(dtype.nullability())
        } else if children.len() == 1 {
            Validity::Array(children.get(0, &Validity::DTYPE, len)?)
        } else {
            vortex_bail!("ZstdV2Array expected 0 or 1 child, got {}", children.len());
        };

        let Some((lengths, frames)) = buffers.split_first() else {
            vortex_bail!("ZstdV2Array expected a lengths buffer");
        };
        let data = ZstdV2Data {
            lengths: lengths.clone().try_to_host_sync()?,
            frames: frames
                .iter()
                .map(|buffer| buffer.clone().try_to_host_sync())
                .collect::<VortexResult<Vec<_>>>()?,
            metadata,
            unsliced_n_rows: len,
            slice_start: 0,
            slice_stop: len,
        };
        let slots = smallvec![validity_to_child(&validity, len)];
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        ZstdV2Slots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let unsliced_validity = unsliced_validity(array.as_view());
        array
            .data()
            .decompress(array.dtype(), &unsliced_validity, ctx)?
            .execute::<ArrayRef>(ctx)
            .map(ExecutionResult::done)
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        crate::rules::RULES.evaluate(array, parent, child_idx)
    }
}

impl ValidityVTable<ZstdV2> for ZstdV2 {
    fn validity(array: ArrayView<'_, ZstdV2>) -> VortexResult<Validity> {
        unsliced_validity(array).slice(array.slice_start..array.slice_stop)
    }
}

impl OperationsVTable<ZstdV2> for ZstdV2 {
    fn scalar_at(
        array: ArrayView<'_, ZstdV2>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let unsliced_validity = unsliced_validity(array);
        array
            .data()
            .with_slice(index, index + 1)
            .decompress(array.dtype(), &unsliced_validity, ctx)?
            .execute_scalar(0, ctx)
    }
}

pub(crate) fn unsliced_validity(array: ArrayView<'_, ZstdV2>) -> Validity {
    child_to_validity(
        array.slots()[ZstdV2Slots::VALIDITY].as_ref(),
        array.dtype().nullability(),
    )
}

impl ZstdV2Data {
    /// The number of rows this array covers.
    pub fn len(&self) -> usize {
        self.slice_stop - self.slice_start
    }

    /// Returns whether the array covers no rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(crate) fn with_slice(&self, start: usize, stop: usize) -> Self {
        let new_start = self.slice_start + start;
        let new_stop = self.slice_start + stop;
        assert!(new_start <= self.slice_stop, "slice start out of bounds");
        assert!(new_stop <= self.slice_stop, "slice stop out of bounds");
        Self {
            slice_start: new_start,
            slice_stop: new_stop,
            ..self.clone()
        }
    }

    /// Validates the invariants that decoding relies on.
    pub fn validate(&self, dtype: &DType, len: usize, validity: &Validity) -> VortexResult<()> {
        vortex_ensure!(
            matches!(dtype, DType::Binary(_) | DType::Utf8(_)),
            "Unsupported dtype for ZstdV2 array: {dtype}"
        );
        vortex_ensure!(
            self.slice_start <= self.slice_stop,
            "Invalid slice range {}..{}",
            self.slice_start,
            self.slice_stop
        );
        vortex_ensure!(
            self.slice_stop <= self.unsliced_n_rows,
            "Slice stop {} exceeds unsliced row count {}",
            self.slice_stop,
            self.unsliced_n_rows
        );
        vortex_ensure!(
            self.slice_stop - self.slice_start == len,
            "Slice length {} does not match array length {len}",
            self.slice_stop - self.slice_start
        );
        if let Some(validity_len) = validity.maybe_len() {
            vortex_ensure!(
                validity_len == self.unsliced_n_rows,
                "Validity length {validity_len} does not match unsliced row count {}",
                self.unsliced_n_rows
            );
        }
        vortex_ensure!(
            self.frames.len() == self.metadata.frames.len(),
            "Frame count {} does not match metadata frame count {}",
            self.frames.len(),
            self.metadata.frames.len()
        );
        let n_values = usize::try_from(self.metadata.n_values())?;
        vortex_ensure!(
            usize::try_from(self.metadata.lengths_uncompressed_size)?
                == n_values * size_of::<ValueLen>(),
            "Lengths frame holds {} bytes, which is not {n_values} lengths",
            self.metadata.lengths_uncompressed_size
        );
        validate_frame_content_size(
            self.lengths.as_slice(),
            self.metadata.lengths_uncompressed_size,
            "lengths",
        )?;
        for (index, (frame, metadata)) in self.frames.iter().zip(&self.metadata.frames).enumerate()
        {
            validate_frame_content_size(
                frame.as_slice(),
                metadata.uncompressed_size,
                &format!("frame {index}"),
            )?;
        }
        Ok(())
    }

    /// The value range each frame covers, in the order the frames are stored.
    pub(crate) fn frame_spans(&self) -> VortexResult<Vec<FrameSpan>> {
        let mut spans = Vec::with_capacity(self.metadata.frames.len());
        let mut value_start = 0usize;
        for frame in &self.metadata.frames {
            let n_values = usize::try_from(frame.n_values)?;
            let uncompressed_size = usize::try_from(frame.uncompressed_size)?;
            spans.push(FrameSpan {
                value_start,
                n_values,
                uncompressed_size,
            });
            value_start = value_start.checked_add(n_values).ok_or_else(|| {
                vortex_err!("Corrupt zstd.v2 metadata: frame value counts overflow a usize")
            })?;
        }
        Ok(spans)
    }

    /// Decompresses the lengths of every stored value.
    ///
    /// This is the whole point of the encoding: the lengths are a small, separately compressed
    /// stream, so a reader can learn where every value lives without touching the value bytes.
    pub(crate) fn decompress_lengths(&self) -> VortexResult<Buffer<ValueLen>> {
        let n_bytes = usize::try_from(self.metadata.lengths_uncompressed_size)?;
        let bytes = decompress_frame(&mut new_decompressor()?, &self.lengths, n_bytes)?;
        let n_values = n_bytes / size_of::<ValueLen>();
        let mut lengths = BufferMut::<ValueLen>::with_capacity(n_values);
        let (chunks, _) = bytes.as_slice().as_chunks::<{ size_of::<ValueLen>() }>();
        for chunk in chunks {
            lengths.push(ValueLen::from_le_bytes(*chunk));
        }
        Ok(lengths.freeze())
    }

    /// Decompresses `frame`, checking it produces the bytes its metadata promised.
    pub(crate) fn decompress_frame_at(
        &self,
        decompressor: &mut zstd::bulk::Decompressor<'_>,
        index: usize,
        span: &FrameSpan,
    ) -> VortexResult<ByteBuffer> {
        decompress_frame(decompressor, &self.frames[index], span.uncompressed_size)
    }
}

fn new_decompressor() -> VortexResult<zstd::bulk::Decompressor<'static>> {
    Ok(zstd::bulk::Decompressor::new()?)
}

fn decompress_frame(
    decompressor: &mut zstd::bulk::Decompressor<'_>,
    frame: &ByteBuffer,
    uncompressed_size: usize,
) -> VortexResult<ByteBuffer> {
    let mut buffer = ByteBufferMut::with_capacity(uncompressed_size);
    let mut destination =
        UninitDestination::new(&mut buffer.spare_capacity_mut()[..uncompressed_size]);
    let n = decompressor.decompress_to_buffer(frame.as_slice(), &mut destination)?;
    vortex_ensure!(
        n == uncompressed_size,
        "Corrupt zstd.v2 frame: expected {uncompressed_size} bytes but decompressed {n}"
    );
    // SAFETY: zstd reported writing exactly `n` bytes into the front of the spare capacity.
    unsafe { buffer.set_len(n) };
    Ok(buffer.freeze())
}

impl<'a> UninitDestination<'a> {
    fn new(spare: &'a mut [MaybeUninit<u8>]) -> Self {
        Self { spare, filled: 0 }
    }
}

/// Ensures a frame's header agrees with the size its metadata declares.
fn validate_frame_content_size(frame: &[u8], metadata_size: u64, what: &str) -> VortexResult<()> {
    let declared = zstd::zstd_safe::get_frame_content_size(frame)
        .map_err(|error| vortex_err!("Invalid zstd.v2 {what}: {error}"))?
        .ok_or_else(|| vortex_err!("Zstd.v2 {what} does not declare a content size"))?;
    vortex_ensure!(
        metadata_size == declared,
        "Zstd.v2 {what} metadata declares {metadata_size} uncompressed bytes, but its header \
         declares {declared}"
    );
    Ok(())
}

/// The frames holding `values`, and where each frame's values start.
pub(crate) fn frames_covering(spans: &[FrameSpan], values: Range<usize>) -> Range<usize> {
    let first = spans
        .iter()
        .position(|span| span.value_stop() > values.start)
        .unwrap_or(spans.len());
    let last = spans
        .iter()
        .position(|span| span.value_stop() >= values.end)
        .map_or(spans.len(), |index| index + 1);
    first..last.max(first)
}

impl ZstdV2Data {
    /// Compresses the valid values of `vbv`, keeping their lengths in a stream of their own.
    pub fn from_var_bin_view(
        vbv: &VarBinViewArray,
        level: i32,
        values_per_frame: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        let n_rows = vbv.as_ref().len();
        let mask = vbv.as_ref().validity()?.execute_mask(n_rows, ctx)?;

        let mut lengths = BufferMut::<ValueLen>::with_capacity(mask.true_count());
        // Frames are cut on value boundaries: at `values_per_frame` values, or before a value
        // whose bytes would push the frame past what a view's offset can address.
        let mut frame_bytes = ByteBufferMut::empty();
        let mut frame_values = 0usize;
        let mut frames = Vec::new();
        let mut frame_metas = Vec::new();
        let mut compressor = zstd::bulk::Compressor::new(level)?;

        let views = vbv.views();
        let buffers = vbv
            .data_buffers()
            .iter()
            .map(|buffer| buffer.as_host())
            .collect::<Vec<_>>();
        let valid = mask.bit_buffer();
        for (index, view) in views.iter().enumerate() {
            if match valid {
                AllOr::All => false,
                AllOr::None => true,
                AllOr::Some(bits) => !bits.value(index),
            } {
                continue;
            }
            let value = if view.is_inlined() {
                view.as_inlined().value()
            } else {
                let view_ref = view.as_view();
                &buffers[view_ref.buffer_index as usize][view_ref.as_range()]
            };

            let full = (values_per_frame > 0 && frame_values == values_per_frame)
                || frame_bytes.len() + value.len() > MAX_FRAME_BYTES;
            if full && frame_values > 0 {
                flush_frame(
                    &mut compressor,
                    &mut frame_bytes,
                    &mut frame_values,
                    &mut frames,
                    &mut frame_metas,
                )?;
            }

            lengths.push(ValueLen::try_from(value.len())?);
            frame_bytes.extend_from_slice(value);
            frame_values += 1;
        }
        if frame_values > 0 || frames.is_empty() {
            flush_frame(
                &mut compressor,
                &mut frame_bytes,
                &mut frame_values,
                &mut frames,
                &mut frame_metas,
            )?;
        }

        let lengths = lengths.freeze();
        let lengths_bytes = lengths
            .iter()
            .flat_map(|len| len.to_le_bytes())
            .collect::<ByteBufferMut>()
            .freeze();
        let compressed_lengths = compress_frame(&mut compressor, lengths_bytes.as_slice())?;

        Ok(Self {
            lengths: compressed_lengths,
            frames,
            metadata: ZstdV2Metadata {
                lengths_uncompressed_size: lengths_bytes.len() as u64,
                frames: frame_metas,
            },
            unsliced_n_rows: n_rows,
            slice_start: 0,
            slice_stop: n_rows,
        })
    }

    /// Decompresses this array's slice into its canonical form.
    ///
    /// Only the frames covering the slice are decompressed, and the views point straight into
    /// them: no value bytes are copied.
    pub(crate) fn decompress(
        &self,
        dtype: &DType,
        unsliced_validity: &Validity,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let unsliced_mask = unsliced_validity.execute_mask(self.unsliced_n_rows, ctx)?;
        let value_bounds =
            unsliced_mask.valid_counts_for_indices(&[self.slice_start, self.slice_stop]);
        let values = value_bounds[0]..value_bounds[1];

        let spans = self.frame_spans()?;
        let frames = frames_covering(&spans, values.clone());
        let lengths = self.decompress_lengths()?;

        let mut decompressor = new_decompressor()?;
        let mut buffers = Vec::with_capacity(frames.len());
        for index in frames.clone() {
            buffers.push(self.decompress_frame_at(&mut decompressor, index, &spans[index])?);
        }

        let offsets = value_offsets(&lengths, &spans[frames.clone()])?;

        // One slot per row of the slice, carrying the value index of the rows that hold one.
        // Walking the rows in order keeps the value index a running count instead of a rank per
        // row.
        let mut slots = Vec::with_capacity(self.slice_stop - self.slice_start);
        let mut value = values.start;
        match unsliced_mask.bit_buffer() {
            AllOr::All => {
                for _ in self.slice_start..self.slice_stop {
                    slots.push(Some(value));
                    value += 1;
                }
            }
            AllOr::None => slots.resize(self.slice_stop - self.slice_start, None),
            AllOr::Some(bits) => {
                for row in self.slice_start..self.slice_stop {
                    if bits.value(row) {
                        slots.push(Some(value));
                        value += 1;
                    } else {
                        slots.push(None);
                    }
                }
            }
        }

        let views = build_views(&lengths, &offsets, &spans[frames], &buffers, &slots)?;

        let validity = align_validity_to_dtype(
            unsliced_validity.slice(self.slice_start..self.slice_stop)?,
            dtype,
        )?;
        // SAFETY: `build_views` only emits views inside the buffers it was handed.
        Ok(unsafe {
            VarBinViewArray::new_unchecked(views, Arc::from(buffers), dtype.clone(), validity)
        }
        .into_array())
    }
}

/// Compresses `frame_bytes` and records the frame, resetting the accumulators.
fn flush_frame(
    compressor: &mut zstd::bulk::Compressor<'_>,
    frame_bytes: &mut ByteBufferMut,
    frame_values: &mut usize,
    frames: &mut Vec<ByteBuffer>,
    frame_metas: &mut Vec<ZstdV2FrameMetadata>,
) -> VortexResult<()> {
    frames.push(compress_frame(compressor, frame_bytes.as_slice())?);
    frame_metas.push(ZstdV2FrameMetadata {
        uncompressed_size: frame_bytes.len() as u64,
        n_values: *frame_values as u64,
    });
    frame_bytes.clear();
    *frame_values = 0;
    Ok(())
}

/// Compresses `bytes` into an exactly sized buffer.
fn compress_frame(
    compressor: &mut zstd::bulk::Compressor<'_>,
    bytes: &[u8],
) -> VortexResult<ByteBuffer> {
    let mut compressed = Vec::with_capacity(zstd::zstd_safe::compress_bound(bytes.len()));
    compressor
        .compress_to_buffer(bytes, &mut compressed)
        .map_err(|err| VortexError::from(err).with_context("while compressing"))?;
    Ok(ByteBuffer::from(compressed))
}

/// Reconciles a validity taken from the array with the one its dtype implies.
///
/// Null values occupy no bytes, so an array keeps its full validity bitmap even when sliced down
/// to only valid rows. Decoded output must still carry the validity its dtype implies.
fn align_validity_to_dtype(validity: Validity, dtype: &DType) -> VortexResult<Validity> {
    if !dtype.is_nullable() && !matches!(validity, Validity::NonNullable) {
        vortex_ensure!(
            matches!(validity, Validity::AllValid),
            "ZstdV2 array expects to be non-nullable but there are nulls after decompression"
        );
        return Ok(Validity::NonNullable);
    }
    if dtype.is_nullable() && matches!(validity, Validity::NonNullable) {
        return Ok(Validity::AllValid);
    }
    Ok(validity)
}

/// The offset of every value within its own frame, for the frames in `spans`.
///
/// This is the prefix sum the separate lengths stream buys: no walk through the value bytes.
fn value_offsets(lengths: &Buffer<ValueLen>, spans: &[FrameSpan]) -> VortexResult<Buffer<u32>> {
    let n_values = spans.iter().map(|span| span.n_values).sum::<usize>();
    let mut offsets = BufferMut::<u32>::with_capacity(n_values);
    for span in spans {
        let mut offset = 0u32;
        for value in span.value_start..span.value_stop() {
            let length = *lengths.get(value).ok_or_else(|| {
                vortex_err!(
                    "Corrupt zstd.v2 metadata: value {value} is past the {} stored lengths",
                    lengths.len()
                )
            })?;
            offsets.push(offset);
            offset = offset.checked_add(length).ok_or_else(|| {
                vortex_err!("Corrupt zstd.v2 lengths: frame offsets overflow a u32")
            })?;
        }
        vortex_ensure!(
            offset as usize == span.uncompressed_size,
            "Corrupt zstd.v2 metadata: {} values measure {offset} bytes, but their frame holds {}",
            span.n_values,
            span.uncompressed_size
        );
    }
    Ok(offsets.freeze())
}

/// Builds one view per output row, pointing into the decompressed frames.
///
/// A row with no value keeps its zeroed view, which is how a null is spelled. Nothing here copies
/// value bytes: a view carries a buffer index and an offset, and the four-byte prefix it inlines.
fn build_views(
    lengths: &Buffer<ValueLen>,
    offsets: &Buffer<u32>,
    spans: &[FrameSpan],
    buffers: &[ByteBuffer],
    slots: &[Option<usize>],
) -> VortexResult<Buffer<BinaryView>> {
    let mut views = BufferMut::<BinaryView>::zeroed(slots.len());
    let Some(first_value) = spans.first().map(|span| span.value_start) else {
        vortex_ensure!(
            slots.iter().all(Option::is_none),
            "Corrupt zstd.v2 metadata: rows hold values but no frame covers them"
        );
        return Ok(views.freeze());
    };

    // Slots ascend, so the frame a value lives in only ever moves forwards.
    let mut frame = 0usize;
    for (slot, view) in slots.iter().zip(views.iter_mut()) {
        let Some(value) = *slot else { continue };
        while frame < spans.len() && value >= spans[frame].value_stop() {
            frame += 1;
        }
        vortex_ensure!(
            frame < spans.len() && value >= spans[frame].value_start,
            "Corrupt zstd.v2 metadata: value {value} is not covered by the decompressed frames"
        );

        let offset = *offsets
            .get(value - first_value)
            .ok_or_else(|| vortex_err!("Corrupt zstd.v2 metadata: value {value} has no offset"))?;
        let length = *lengths
            .get(value)
            .ok_or_else(|| vortex_err!("Corrupt zstd.v2 metadata: value {value} has no length"))?;
        let bytes = buffers[frame]
            .as_slice()
            .get(offset as usize..offset as usize + length as usize)
            .ok_or_else(|| {
                vortex_err!(
                    "Corrupt zstd.v2 metadata: value {value} of {length} bytes at offset {offset} \
                     runs past the end of its {} byte frame",
                    buffers[frame].len()
                )
            })?;
        *view = BinaryView::make_view(bytes, u32::try_from(frame)?, offset);
    }
    Ok(views.freeze())
}

impl ZstdV2Data {
    /// Filters the array, decompressing only the frames holding a selected value.
    ///
    /// The lengths stream says where every value lives, so the frames that hold none of the
    /// selected values are never read, and the ones that are keep their bytes: the output views
    /// point into them.
    pub(crate) fn filter(
        &self,
        dtype: &DType,
        unsliced_validity: &Validity,
        mask: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        vortex_ensure!(
            mask.len() == self.len(),
            "Filter mask of length {} does not match the {} rows of the array",
            mask.len(),
            self.len()
        );
        let unsliced_mask = unsliced_validity.execute_mask(self.unsliced_n_rows, ctx)?;

        // Rows the mask selects, in this array's own row space.
        let selected_rows: Vec<usize> = match mask.indices() {
            AllOr::All => (self.slice_start..self.slice_stop).collect(),
            AllOr::None => Vec::new(),
            AllOr::Some(indices) => indices.iter().map(|i| self.slice_start + i).collect(),
        };
        // Valid rows before each selected row, which is a selected row's own value index.
        let ranks = unsliced_mask.valid_counts_for_indices(&selected_rows);
        let slots: Vec<Option<usize>> = match unsliced_mask.bit_buffer() {
            AllOr::All => ranks.iter().map(|rank| Some(*rank)).collect(),
            AllOr::None => vec![None; selected_rows.len()],
            AllOr::Some(bits) => selected_rows
                .iter()
                .zip(&ranks)
                .map(|(row, rank)| bits.value(*row).then_some(*rank))
                .collect(),
        };

        let spans = self.frame_spans()?;
        // Only the frames a selected value falls in are decompressed. They stay contiguous in the
        // output buffer list, so a view's buffer index is its frame's position among them.
        let mut needed = Vec::new();
        let mut next = 0usize;
        for (index, span) in spans.iter().enumerate() {
            while next < slots.len() && slots[next].is_none_or(|value| value < span.value_start) {
                next += 1;
            }
            if next < slots.len() && slots[next].is_some_and(|value| value < span.value_stop()) {
                needed.push(index);
            }
        }
        if needed.len() == spans.len() && needed.len() > 1 {
            // Nothing to skip. Decoding the whole array and filtering that is no more work, and
            // it keeps one code path warm rather than two.
            return Ok(None);
        }

        let lengths = self.decompress_lengths()?;
        let mut decompressor = new_decompressor()?;
        let mut buffers = Vec::with_capacity(needed.len());
        let mut kept_spans = Vec::with_capacity(needed.len());
        for index in &needed {
            buffers.push(self.decompress_frame_at(&mut decompressor, *index, &spans[*index])?);
            kept_spans.push(spans[*index]);
        }

        // Offsets are per frame, so a gap between kept frames does not disturb them.
        let mut offsets = BufferMut::<u32>::empty();
        for span in &kept_spans {
            let frame_offsets = value_offsets(&lengths, std::slice::from_ref(span))?;
            offsets.extend_from_slice(frame_offsets.as_slice());
        }

        let offsets = offsets.freeze();
        let views = build_views_gathered(&lengths, &offsets, &kept_spans, &buffers, &slots)?;
        let validity = align_validity_to_dtype(
            unsliced_validity
                .slice(self.slice_start..self.slice_stop)?
                .filter(mask)?,
            dtype,
        )?;
        // SAFETY: `build_views_gathered` only emits views inside the buffers it was handed.
        Ok(Some(
            unsafe {
                VarBinViewArray::new_unchecked(views, Arc::from(buffers), dtype.clone(), validity)
            }
            .into_array(),
        ))
    }
}

/// Builds views over a set of frames that need not be adjacent.
///
/// [`build_views`] can index its offsets by a single value range because the frames it is given
/// are contiguous. A filter keeps only the frames it needs, so offsets are numbered per kept
/// frame instead.
fn build_views_gathered(
    lengths: &Buffer<ValueLen>,
    offsets: &Buffer<u32>,
    spans: &[FrameSpan],
    buffers: &[ByteBuffer],
    slots: &[Option<usize>],
) -> VortexResult<Buffer<BinaryView>> {
    let mut views = BufferMut::<BinaryView>::zeroed(slots.len());
    let mut frame = 0usize;
    // Where each kept frame's offsets begin.
    let mut frame_offset_starts = Vec::with_capacity(spans.len());
    let mut running = 0usize;
    for span in spans {
        frame_offset_starts.push(running);
        running += span.n_values;
    }

    for (slot, view) in slots.iter().zip(views.iter_mut()) {
        let Some(value) = *slot else { continue };
        while frame < spans.len() && value >= spans[frame].value_stop() {
            frame += 1;
        }
        vortex_ensure!(
            frame < spans.len() && value >= spans[frame].value_start,
            "Corrupt zstd.v2 metadata: value {value} is not covered by the decompressed frames"
        );

        let offset_idx = frame_offset_starts[frame] + (value - spans[frame].value_start);
        let offset = *offsets
            .get(offset_idx)
            .ok_or_else(|| vortex_err!("Corrupt zstd.v2 metadata: value {value} has no offset"))?;
        let length = *lengths
            .get(value)
            .ok_or_else(|| vortex_err!("Corrupt zstd.v2 metadata: value {value} has no length"))?;
        let bytes = buffers[frame]
            .as_slice()
            .get(offset as usize..offset as usize + length as usize)
            .ok_or_else(|| {
                vortex_err!(
                    "Corrupt zstd.v2 metadata: value {value} of {length} bytes at offset {offset} \
                     runs past the end of its {} byte frame",
                    buffers[frame].len()
                )
            })?;
        *view = BinaryView::make_view(bytes, u32::try_from(frame)?, offset);
    }
    Ok(views.freeze())
}
