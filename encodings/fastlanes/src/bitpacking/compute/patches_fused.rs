// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fused decompression kernel for a [`PatchesArray`] wrapping a [`BitPackedArray`].
//!
//! Without this kernel, executing `Patches(BitPacked)` first unpacks the entire bit-packed
//! array into a full-length primitive buffer and then walks the patches over that (by then
//! cache-cold) buffer. This kernel instead iterates the underlying array one 1024-element
//! FastLanes chunk at a time: each chunk is unpacked and its patches are combined into the
//! decoded values while they are still cache-resident.
//!
//! The block-relative patch layout is what makes the fusion cheap: the patches of chunk `b` are
//! the contiguous run `indices[skip_indices[b]..skip_indices[b + 1]]`, found in constant time
//! with no searching, and the `u16` in-block positions index directly into the decoded chunk.
//!
//! [`PatchesArray`]: vortex_array::arrays::PatchesArray
//! [`BitPackedArray`]: crate::BitPackedArray

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::Patches;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::patches::PATCH_BLOCK_SIZE;
use vortex_array::arrays::patches::PatchCombine;
use vortex_array::arrays::patches::PatchFn;
use vortex_array::arrays::patches::PatchesArrayExt;
use vortex_array::arrays::patches::PatchesArraySlotsExt;
use vortex_array::arrays::patches::PatchesSlots;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::PrimitiveBuilder;
use vortex_array::kernel::ExecuteParentKernel;
use vortex_array::match_each_integer_ptype;
use vortex_error::VortexResult;

use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::unpack_iter::BitPacked as BitPackedUnpack;

/// Executes a parent [`Patches`] array by unpacking the bit-packed child chunk-by-chunk and
/// applying each block's patches while the chunk is cache-hot.
#[derive(Debug)]
pub(crate) struct PatchesFusedKernel;

impl ExecuteParentKernel<BitPacked> for PatchesFusedKernel {
    type Parent = Patches;

    fn execute_parent(
        &self,
        array: ArrayView<'_, BitPacked>,
        parent: ArrayView<'_, Patches>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if child_idx != PatchesSlots::INNER {
            return Ok(None);
        }

        // The fused path requires the parent's block layout to line up with the FastLanes
        // chunks of the packed child.
        if array.offset() as usize != parent.offset() {
            return Ok(None);
        }

        // Degenerate widths have no packed payload to iterate, and a child with its own interior
        // patches would need a second patch pass; fall back to the standard pathway.
        if array.is_empty() || array.bit_width() == 0 || array.patches().is_some() {
            return Ok(None);
        }

        // The patch children must already be decoded, host-resident primitives.
        let (Some(skip_indices), Some(indices), Some(values)) = (
            parent.skip_indices().as_opt::<Primitive>(),
            parent.indices().as_opt::<Primitive>(),
            parent.values().as_opt::<Primitive>(),
        ) else {
            return Ok(None);
        };
        if !(skip_indices.array().is_host()
            && indices.array().is_host()
            && values.array().is_host()
            && array.array().is_host())
        {
            return Ok(None);
        }

        let patch_fn = parent.patch_fn();
        let result = match_each_integer_ptype!(array.dtype().as_ptype(), |T| {
            fused_unpack_patch::<T>(
                array,
                skip_indices.as_slice::<u32>(),
                indices.as_slice::<u16>(),
                values.as_slice::<T>(),
                patch_fn,
                ctx,
            )?
        });

        Ok(Some(result))
    }
}

/// Unpack `array` chunk-by-chunk, combining each block's contiguous patch run into the decoded
/// chunk before moving on to the next chunk.
fn fused_unpack_patch<T: BitPackedUnpack + PatchCombine>(
    array: ArrayView<'_, BitPacked>,
    skip_indices: &[u32],
    indices: &[u16],
    values: &[T],
    patch_fn: PatchFn,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let len = array.len();
    let offset = array.offset() as usize;

    let mut builder = PrimitiveBuilder::with_capacity(array.dtype().nullability(), len);
    let mut uninit_range = builder.uninit_range(len);

    // SAFETY: We initialize all `len` values below via `decode_for_each_chunk`.
    unsafe {
        uninit_range.append_mask(&array.validity()?.execute_mask(len, ctx)?);
    }

    // SAFETY: `decode_for_each_chunk` writes a value to every slot in this range.
    let uninit_slice = unsafe { uninit_range.slice_uninit_mut(0, len) };

    let mut chunks = array.unpacked_chunks::<T>()?;
    chunks.decode_for_each_chunk(uninit_slice, |chunk, range| {
        // Chunks are yielded in logical coordinates; the chunk's block is found from padded ones.
        let block = (range.start + offset) / PATCH_BLOCK_SIZE;
        let block_start = block * PATCH_BLOCK_SIZE;

        let start = skip_indices[block] as usize;
        let stop = skip_indices[block + 1] as usize;

        for patch_idx in start..stop {
            let padded = block_start + indices[patch_idx] as usize;
            let Some(logical) = padded.checked_sub(offset) else {
                continue;
            };
            if logical < range.start || logical >= range.end {
                continue;
            }
            let out = &mut chunk[logical - range.start];
            *out = T::combine(patch_fn, *out, values[patch_idx]);
        }
    });

    // SAFETY: A correct validity mask of `len` values was set via `append_mask`, and the same
    // number of values was initialized via `decode_for_each_chunk`.
    unsafe {
        uninit_range.finish();
    }

    Ok(builder.finish())
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::Patches;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::patches::PatchFn;
    use vortex_array::assert_arrays_eq;
    use vortex_array::patches::Patches as PatchesStruct;
    use vortex_array::validity::Validity;
    use vortex_buffer::BufferMut;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::BitPackedData;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    #[test]
    fn fused_execute_overwrite() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();

        let mut values = BufferMut::from_iter((0..3000u32).map(|i| i % 512));
        let packed = BitPackedData::encode(&values.clone().into_array(), 9, &mut ctx)?;

        let patches = PatchesStruct::new(
            3000,
            0,
            buffer![5u32, 1030, 2900].into_array(),
            buffer![100_000u32, 200_000, 300_000].into_array(),
            None,
        )?;
        let array = Patches::from_array_and_patches(
            packed.into_array(),
            &patches,
            PatchFn::Overwrite,
            &mut ctx,
        )?
        .into_array();

        values[5] = 100_000;
        values[1030] = 200_000;
        values[2900] = 300_000;
        let expected = PrimitiveArray::new(values.freeze(), Validity::NonNullable);

        let executed = array.execute::<Canonical>(&mut ctx)?.into_primitive();
        assert_arrays_eq!(expected, executed, &mut ctx);
        Ok(())
    }

    #[test]
    fn fused_execute_add_sliced() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();

        let values = BufferMut::from_iter((0..4096u64).map(|i| i % 128));
        let packed = BitPackedData::encode(&values.into_array(), 7, &mut ctx)?;

        let patches = PatchesStruct::new(
            4096,
            0,
            buffer![100u32, 1500, 2500, 3500].into_array(),
            buffer![1000u64, 2000, 3000, 4000].into_array(),
            None,
        )?;
        let array =
            Patches::from_array_and_patches(packed.into_array(), &patches, PatchFn::Add, &mut ctx)?
                .into_array()
                .slice(1200..2600)?;

        let mut expected = BufferMut::from_iter((1200..2600u64).map(|i| i % 128));
        expected[1500 - 1200] += 2000;
        expected[2500 - 1200] += 3000;
        let expected = PrimitiveArray::new(expected.freeze(), Validity::NonNullable);

        let executed = array.execute::<Canonical>(&mut ctx)?.into_primitive();
        assert_arrays_eq!(expected, executed, &mut ctx);
        Ok(())
    }

    #[test]
    fn fused_kernel_handles_directly() -> VortexResult<()> {
        use vortex_array::arrays::patches::PatchesSlots;
        use vortex_array::kernel::ExecuteParentKernel;

        use super::PatchesFusedKernel;
        use crate::BitPackedArray;

        let mut ctx = SESSION.create_execution_ctx();

        let values = BufferMut::from_iter((0..2048u32).map(|i| i % 512));
        let packed = BitPackedData::encode(&values.into_array(), 9, &mut ctx)?;

        let patches = PatchesStruct::new(
            2048,
            0,
            buffer![5u32, 1030].into_array(),
            buffer![100_000u32, 200_000].into_array(),
            None,
        )?;
        let array = Patches::from_array_and_patches(
            packed.into_array(),
            &patches,
            PatchFn::Overwrite,
            &mut ctx,
        )?
        .into_array();

        let inner: BitPackedArray = array.slots()[PatchesSlots::INNER]
            .as_ref()
            .expect("inner slot")
            .clone()
            .downcast();
        let parent_view = array.as_opt::<Patches>().expect("patches array");

        let result = PatchesFusedKernel
            .execute_parent(inner.as_view(), parent_view, PatchesSlots::INNER, &mut ctx)?
            .expect("fused kernel must handle Patches(BitPacked) with primitive patch children");

        let mut expected = BufferMut::from_iter((0..2048u32).map(|i| i % 512));
        expected[5] = 100_000;
        expected[1030] = 200_000;
        let expected = PrimitiveArray::new(expected.freeze(), Validity::NonNullable);
        assert_arrays_eq!(expected, result, &mut ctx);
        Ok(())
    }

    #[test]
    fn patches_layout_is_smaller() -> VortexResult<()> {
        use vortex_array::arrays::Patched;

        use crate::BitPacked;
        use crate::BitPackedArrayExt;

        let mut ctx = SESSION.create_execution_ctx();

        // 1M rows, ~1% patched.
        let len = 1 << 20;
        let mut values = BufferMut::from_iter((0..len).map(|i| (i % 512) as u32));
        for i in 0..len / 100 {
            values[i * 100] = 100_000;
        }
        let bitpacked = BitPackedData::encode(&values.freeze().into_array(), 9, &mut ctx)?;
        let patches = bitpacked.patches().vortex_expect("patches");

        let inner = BitPacked::try_new(
            bitpacked.packed().clone(),
            bitpacked.dtype().as_ptype(),
            bitpacked.validity()?,
            None,
            bitpacked.bit_width(),
            bitpacked.len(),
            bitpacked.offset(),
        )?
        .into_array();

        let interior_nbytes = bitpacked.as_array().nbytes();
        let patched_nbytes = Patched::from_array_and_patches(inner.clone(), &patches, &mut ctx)?
            .into_array()
            .nbytes();
        let patches_nbytes =
            Patches::from_array_and_patches(inner, &patches, PatchFn::Overwrite, &mut ctx)?
                .into_array()
                .nbytes();

        // The block-relative layout must be smaller than the lane-transposed one (its skip
        // index is 1 u32 per block, not n_lanes + 1) and than interior patches (u16 positions
        // instead of absolute u32/u64 indices).
        assert!(
            patches_nbytes < patched_nbytes,
            "PatchesArray ({patches_nbytes}B) must be smaller than PatchedArray ({patched_nbytes}B)"
        );
        assert!(
            patches_nbytes < interior_nbytes,
            "PatchesArray ({patches_nbytes}B) must be smaller than interior patches ({interior_nbytes}B)"
        );
        Ok(())
    }
}
