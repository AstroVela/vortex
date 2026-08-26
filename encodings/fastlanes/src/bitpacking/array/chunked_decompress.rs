// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Streaming chunked decompression for bit-packed arrays.
//!
//! This backs [`VTable::decompress_chunks`](vortex_array::vtable::VTable::decompress_chunks) for
//! [`BitPacked`]: each FastLanes block is unpacked into the decompressor's cache-resident scratch
//! buffer, patches falling in that block are applied in place, and the block is handed to the
//! sink — the array is never materialized in full.

use num_traits::AsPrimitive;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::chunk_iter::ChunkMut;
use vortex_array::chunk_iter::ChunkSink;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PhysicalPType;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::patches::Patches;
use vortex_error::VortexResult;

use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::unpack_iter::BitPacked as BitPackedUnpack;
use crate::unpack_iter::UnpackStrategy;
use crate::unpack_iter::UnpackedChunks;

pub(crate) fn decompress_chunks(
    array: ArrayView<'_, BitPacked>,
    ctx: &mut ExecutionCtx,
    sink: &mut dyn ChunkSink,
) -> VortexResult<()> {
    match_each_integer_ptype!(array.as_ref().dtype().as_ptype(), |T| {
        decompress_chunks_typed::<T>(array, ctx, sink)
    })
}

fn decompress_chunks_typed<T: BitPackedUnpack>(
    array: ArrayView<'_, BitPacked>,
    ctx: &mut ExecutionCtx,
    sink: &mut dyn ChunkSink,
) -> VortexResult<()> {
    if array.as_ref().is_empty() {
        return Ok(());
    }

    let patch_list = match array.patches() {
        None => Vec::new(),
        Some(patches) => build_patch_list(&patches, ctx, |v: T| v)?,
    };

    let mut chunks = array.unpacked_chunks::<T>()?;
    stream_unpacked_chunks(&mut chunks, &patch_list, sink)
}

/// Materialize sparse patches once as sorted (local row, value) pairs so the per-chunk loop only
/// advances a cursor, applying `map` to each patch value. This is the only heap state the
/// streaming path allocates, and it is proportional to the patch count, not the array length.
pub(crate) fn build_patch_list<T: NativePType>(
    patches: &Patches,
    ctx: &mut ExecutionCtx,
    map: impl Fn(T) -> T,
) -> VortexResult<Vec<(usize, T)>> {
    let indices = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
    let values = patches.values().clone().execute::<PrimitiveArray>(ctx)?;
    let values = values.as_slice::<T>();
    let offset = patches.offset();
    Ok(match_each_unsigned_integer_ptype!(indices.ptype(), |P| {
        indices
            .as_slice::<P>()
            .iter()
            .zip(values)
            .map(|(&idx, &v)| (<P as AsPrimitive<usize>>::as_(idx) - offset, map(v)))
            .collect()
    }))
}

/// Walk every unpacked FastLanes block in order, patch it in place from the pre-built cursor
/// list, and hand it to the sink. Generic over the [`UnpackStrategy`] so fused strategies (e.g.
/// FoR's reference-add unpack) stream through the same loop with zero extra passes.
pub(crate) fn stream_unpacked_chunks<T: PhysicalPType, S: UnpackStrategy<T>>(
    chunks: &mut UnpackedChunks<T, S>,
    patch_list: &[(usize, T)],
    sink: &mut dyn ChunkSink,
) -> VortexResult<()> {
    let mut patch_cursor = 0usize;
    let mut result = Ok(());
    chunks.for_each_unpacked_chunk(|chunk, range| {
        if result.is_err() {
            return;
        }
        while let Some(&(row, value)) = patch_list.get(patch_cursor)
            && row < range.end
        {
            chunk[row - range.start] = value;
            patch_cursor += 1;
        }
        result = sink.accept(ChunkMut::new(chunk), range);
    });
    result
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::chunk_iter::ChunkMut;
    use vortex_array::dtype::NativePType;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::BitPackedArrayExt;
    use crate::FoRArraySlotsExt;
    use crate::FoRData;
    use crate::bitpack_compress::bitpack_encode;

    pub(super) static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    fn collect_chunks<T: NativePType>(array: &vortex_array::ArrayRef) -> VortexResult<Vec<T>> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut out = Vec::with_capacity(array.len());
        array.decompress_chunks_or_materialize(&mut ctx, &mut |chunk: ChunkMut<'_>,
                                                                _range: std::ops::Range<
            usize,
        >|
         -> VortexResult<()> {
            out.extend_from_slice(chunk.as_slice::<T>());
            Ok(())
        })?;
        Ok(out)
    }

    fn assert_chunks_match_execute<T: NativePType>(
        array: vortex_array::ArrayRef,
    ) -> VortexResult<()> {
        let chunked = collect_chunks::<T>(&array)?;
        let mut ctx = SESSION.create_execution_ctx();
        let expected = array.execute::<PrimitiveArray>(&mut ctx)?;
        assert_eq!(chunked.as_slice(), expected.as_slice::<T>());
        Ok(())
    }

    #[test]
    fn bitpacked_chunks_match_execute() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter((0..5000u32).map(|i| i % 900));
        let bp = bitpack_encode(&values, 10, None, &mut ctx)?;
        let array = bp.into_array();
        assert!(array.supports_decompress_chunks());
        assert_chunks_match_execute::<u32>(array)
    }

    #[test]
    fn bitpacked_chunks_with_patches() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter(
            (0..5000u32).map(|i| if i % 700 == 0 { 100_000 + i } else { i % 900 }),
        );
        let bp = bitpack_encode(&values, 10, None, &mut ctx)?;
        assert!(bp.patches().is_some());
        assert_chunks_match_execute::<u32>(bp.into_array())
    }

    #[test]
    fn bitpacked_chunks_sliced() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter(
            (0..5000u32).map(|i| if i % 700 == 0 { 100_000 + i } else { i % 900 }),
        );
        let bp = bitpack_encode(&values, 10, None, &mut ctx)?.into_array();
        // Slice crossing chunk boundaries with a non-zero offset.
        let sliced = bp.slice(517..4013)?;
        // The slice may no longer be a BitPacked array head, but chunked iteration must still
        // stream correct values via whatever encodings the slice resolves to.
        assert_chunks_match_execute::<u32>(sliced)
    }

    #[test]
    fn for_over_bitpacked_chunks() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter((0..5000i64).map(|i| 1_000_000 + (i * 7) % 800));
        let for_array = FoRData::encode(values, &mut ctx)?;
        assert!(
            for_array.encoded().as_opt::<crate::BitPacked>().is_some()
                || for_array
                    .encoded()
                    .as_opt::<vortex_array::arrays::Primitive>()
                    .is_some()
        );
        assert_chunks_match_execute::<i64>(for_array.into_array())
    }

    /// An unsigned reference over a BitPacked child takes the fused `FoRStrategy` streaming
    /// path (reference folded into the unpack kernel), including patch handling.
    #[test]
    fn for_over_bitpacked_fused_chunks_with_patches() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let deltas = PrimitiveArray::from_iter(
            (0..5000u32).map(|i| if i % 700 == 0 { 100_000 + i } else { i % 900 }),
        );
        let bp = bitpack_encode(&deltas, 10, None, &mut ctx)?;
        assert!(bp.patches().is_some());
        let for_array = crate::FoR::try_new(
            bp.into_array(),
            vortex_array::scalar::Scalar::from(1_000_000u32),
        )?;
        assert_chunks_match_execute::<u32>(for_array.into_array())
    }

    #[test]
    fn fallback_primitive_chunks() -> VortexResult<()> {
        let values = PrimitiveArray::from_iter(0..3000i32).into_array();
        assert_chunks_match_execute::<i32>(values)
    }

    #[test]
    fn empty_array_chunks() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter(Vec::<u32>::new());
        let bp = bitpack_encode(&values, 0, None, &mut ctx)?;
        let chunked = collect_chunks::<u32>(&bp.into_array())?;
        assert!(chunked.is_empty());
        Ok(())
    }
}

#[cfg(test)]
mod executor_tests {
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::chunk_iter::set_chunked_execute_enabled;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;

    use super::tests::SESSION;
    use crate::FoR;
    use crate::bitpack_compress::bitpack_encode;

    /// The executor's stream-to-canonical shortcut must produce the same canonical array as
    /// level-wise execution, including validity, on a nullable multi-level tree.
    #[test]
    fn execute_via_chunks_matches_levelwise() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = Buffer::from_iter((0..5000i32).map(|i| i % 900));
        let validity = Validity::from_iter((0..5000).map(|i| i % 7 != 0));
        let deltas = PrimitiveArray::new(values, validity);
        let bp = bitpack_encode(&deltas, 10, None, &mut ctx)?;
        // Signed reference so the generic streaming composition (not the fused path) is used.
        let array = FoR::try_new(bp.into_array(), Scalar::from(-1_000_000i32))?.into_array();

        set_chunked_execute_enabled(false);
        let levelwise = array.clone().execute::<PrimitiveArray>(&mut ctx);
        set_chunked_execute_enabled(true);
        let levelwise = levelwise?;
        let streaming = array.execute::<PrimitiveArray>(&mut ctx)?;

        assert_arrays_eq!(streaming, levelwise, &mut ctx);
        Ok(())
    }
}
