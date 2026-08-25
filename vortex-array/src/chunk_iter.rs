// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Streaming chunked decompression.
//!
//! [`ArrayRef::decompress_chunks`] walks an array's decompressed values in cache-resident chunks
//! (~1024 elements) without materializing the full array. Each encoding can override
//! [`VTable::decompress_chunks`](crate::vtable::VTable::decompress_chunks) to stream chunks
//! straight out of its decompression kernel; wrapper encodings (e.g. FoR) compose by interposing
//! a stack-allocated [`ChunkSink`] adapter around the downstream sink and recursing into their
//! child through the erased [`ArrayRef`] entry point.
//!
//! # Cost model
//!
//! The composition cost is one virtual call per chunk per encoding level (amortized over ~1024
//! elements, i.e. fractions of a nanosecond per element) plus, for each wrapper level, one
//! in-place pass over an L1-resident chunk. There is no per-element dynamic dispatch and no
//! heap-allocated state introduced on the way down: the sink chain lives in the caller's stack
//! frames, one frame per encoding level. Leaf producers reuse whatever scratch buffer their
//! decompressor already maintains (e.g. the FastLanes 1024-element unpack buffer).
//!
//! The default implementation executes the array to canonical and then streams the result, which
//! is exactly the two-pass behavior this API exists to avoid — so `decompress_chunks` is always
//! possible on any encoding, and monotonically improves as encodings add overrides.
//!
//! # Contract
//!
//! - Chunks arrive in order, are contiguous, and cover `0..array.len()` exactly.
//! - Chunks are producer-owned scratch: the sink may mutate them freely (wrappers rely on this to
//!   transform values in place), and their contents are invalid once `accept` returns.
//! - Validity is *not* streamed: positions that are logically null contain unspecified (but
//!   initialized) values, matching `execute`'s decompression behavior. Callers that need null
//!   information should fetch `array.validity()` separately.
//! - Currently only primitive-typed arrays are supported.

use std::marker::PhantomData;
use std::ops::Range;

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::PrimitiveArray;
use crate::dtype::NativePType;
use crate::dtype::PType;
use crate::match_each_native_ptype;

/// The target number of elements per streamed chunk.
///
/// This matches the FastLanes block size so leaf decompressors can hand their unpack scratch
/// buffer to the sink without copying. Producers may emit shorter chunks (e.g. a sliced first or
/// last block).
pub const DECOMPRESS_CHUNK_LEN: usize = 1024;

/// A type-erased, mutable view over one chunk of decompressed primitive values.
///
/// This is a `(PType, *mut, len)` triple rather than a generic slice so it can cross the
/// `dyn ChunkSink` boundary; sinks recover the typed slice with [`Self::as_slice_mut`]. The
/// erasure cost is paid once per chunk, not per element.
pub struct ChunkMut<'a> {
    ptype: PType,
    data: *mut u8,
    len: usize,
    _marker: PhantomData<&'a mut u8>,
}

impl<'a> ChunkMut<'a> {
    /// Wrap a typed slice of decompressed values.
    pub fn new<T: NativePType>(values: &'a mut [T]) -> Self {
        Self {
            ptype: T::PTYPE,
            data: values.as_mut_ptr().cast(),
            len: values.len(),
            _marker: PhantomData,
        }
    }

    /// The primitive type of the values in this chunk.
    #[inline]
    pub fn ptype(&self) -> PType {
        self.ptype
    }

    /// The number of values in this chunk.
    #[inline]
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// View the chunk as a typed slice.
    ///
    /// # Panics
    /// Panics if `T::PTYPE` does not match the chunk's ptype.
    #[inline]
    pub fn as_slice<T: NativePType>(&self) -> &[T] {
        assert_eq!(T::PTYPE, self.ptype, "ChunkMut ptype mismatch");
        // SAFETY: constructed from a valid `&mut [T]` with matching ptype; the lifetime is tied
        // to the original borrow via `_marker`.
        unsafe { std::slice::from_raw_parts(self.data.cast(), self.len) }
    }

    /// View the chunk as a mutable typed slice.
    ///
    /// # Panics
    /// Panics if `T::PTYPE` does not match the chunk's ptype.
    #[inline]
    pub fn as_slice_mut<T: NativePType>(&mut self) -> &mut [T] {
        assert_eq!(T::PTYPE, self.ptype, "ChunkMut ptype mismatch");
        // SAFETY: constructed from a valid, exclusively borrowed `&mut [T]` with matching ptype.
        unsafe { std::slice::from_raw_parts_mut(self.data.cast(), self.len) }
    }

    /// Reborrow this chunk with a shorter lifetime, e.g. to forward it to a downstream sink.
    #[inline]
    pub fn reborrow(&mut self) -> ChunkMut<'_> {
        ChunkMut {
            ptype: self.ptype,
            data: self.data,
            len: self.len,
            _marker: PhantomData,
        }
    }
}

/// Consumer side of [`ArrayRef::decompress_chunks`].
///
/// Implementations receive each decompressed chunk exactly once, in array order. `row_range` is
/// the range of logical rows (relative to the array being iterated) that `chunk` covers; its
/// length always equals `chunk.len()`.
pub trait ChunkSink {
    /// Accept the next chunk of decompressed values.
    fn accept(&mut self, chunk: ChunkMut<'_>, row_range: Range<usize>) -> VortexResult<()>;
}

impl<F> ChunkSink for F
where
    F: FnMut(ChunkMut<'_>, Range<usize>) -> VortexResult<()>,
{
    #[inline]
    fn accept(&mut self, chunk: ChunkMut<'_>, row_range: Range<usize>) -> VortexResult<()> {
        self(chunk, row_range)
    }
}

impl ArrayRef {
    /// Stream the array's decompressed values through `sink` in cache-resident chunks.
    ///
    /// See the [module docs](self) for the contract and cost model. Encodings without a
    /// specialized implementation fall back to full decompression followed by chunked iteration
    /// of the result.
    pub fn decompress_chunks(
        &self,
        ctx: &mut ExecutionCtx,
        sink: &mut dyn ChunkSink,
    ) -> VortexResult<()> {
        vortex_ensure!(
            self.dtype().is_primitive(),
            "decompress_chunks requires a primitive-typed array, got {}",
            self.dtype()
        );

        #[cfg(debug_assertions)]
        {
            let mut checked = CoverageCheckSink {
                inner: sink,
                next_row: 0,
                ptype: self.dtype().as_ptype(),
            };
            self.dyn_array()
                .decompress_chunks(self, ctx, &mut checked)?;
            debug_assert_eq!(
                checked.next_row,
                self.len(),
                "decompress_chunks did not cover the full array"
            );
            Ok(())
        }
        #[cfg(not(debug_assertions))]
        self.dyn_array().decompress_chunks(self, ctx, sink)
    }
}

#[cfg(debug_assertions)]
struct CoverageCheckSink<'a> {
    inner: &'a mut dyn ChunkSink,
    next_row: usize,
    ptype: PType,
}

#[cfg(debug_assertions)]
impl ChunkSink for CoverageCheckSink<'_> {
    fn accept(&mut self, chunk: ChunkMut<'_>, row_range: Range<usize>) -> VortexResult<()> {
        debug_assert_eq!(row_range.start, self.next_row, "non-contiguous chunk");
        debug_assert_eq!(
            row_range.len(),
            chunk.len(),
            "chunk/row_range length mismatch"
        );
        debug_assert_eq!(chunk.ptype(), self.ptype, "chunk ptype mismatch");
        self.next_row = row_range.end;
        self.inner.accept(chunk, row_range)
    }
}

/// Fallback chunked decompression: execute the array to a canonical [`PrimitiveArray`], then
/// stream copies of its values in [`DECOMPRESS_CHUNK_LEN`]-sized chunks.
///
/// This is the two-pass baseline. It copies each chunk into a single reusable scratch buffer
/// (allocated once) because sinks receive exclusive, mutable chunks.
pub fn decompress_chunks_via_canonical(
    array: &ArrayRef,
    ctx: &mut ExecutionCtx,
    sink: &mut dyn ChunkSink,
) -> VortexResult<()> {
    let primitive = array.clone().execute::<PrimitiveArray>(ctx)?;
    match_each_native_ptype!(primitive.ptype(), |T| {
        stream_slice_chunks::<T>(primitive.as_slice::<T>(), sink)
    })
}

fn stream_slice_chunks<T: NativePType>(values: &[T], sink: &mut dyn ChunkSink) -> VortexResult<()> {
    let mut scratch = vec![T::default(); values.len().min(DECOMPRESS_CHUNK_LEN)];
    for (chunk_idx, chunk) in values.chunks(DECOMPRESS_CHUNK_LEN).enumerate() {
        let start = chunk_idx * DECOMPRESS_CHUNK_LEN;
        let scratch = &mut scratch[..chunk.len()];
        scratch.copy_from_slice(chunk);
        sink.accept(ChunkMut::new(scratch), start..start + chunk.len())?;
    }
    Ok(())
}
