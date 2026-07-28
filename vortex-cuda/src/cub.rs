// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA wrappers around CUB scan primitives.

use std::ffi::c_void;

use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use vortex::array::buffer::BufferHandle;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex_cub::onpair;
use vortex_cub::scan;
use vortex_cub::scan::cudaStream_t;

use crate::CudaBufferExt;
use crate::CudaExecutionCtx;

/// Regenerate the OnPair decode kernel's per-batch output offsets in one
/// fused sweep (see `cub/kernels/onpair.cu`): the per-batch decoded-size
/// reduction and the exclusive scan over the sizes run in a single kernel via
/// decoupled look-back. The visible token window is read on device from
/// `token_bounds[0..2)`; tokens outside it contribute zero bytes, so the
/// offsets come out window-relative, and a window code outside the dictionary
/// contributes zero bytes — inputs are trusted to be consistent, the guards
/// only bound the sweep's own reads. `code_width` selects the code stream's
/// element size in bytes (1 or 2). Returns `num_batches + 1` offsets; the
/// last is the window's total decoded byte count.
#[allow(clippy::too_many_arguments)]
pub(crate) fn onpair_batch_offsets(
    codes: &BufferHandle,
    code_width: u32,
    lens: &BufferHandle,
    dict_size: u32,
    token_bounds: &CudaSlice<u64>,
    num_tokens: usize,
    num_batches: usize,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<CudaSlice<u64>> {
    let num_batches_i64 = i64::try_from(num_batches)?;
    let temp_bytes = onpair::batch_offsets_temp_size(num_batches_i64)
        .map_err(|err| vortex_err!("CUB onpair_batch_offsets_temp_size failed: {err}"))?;

    let mut temp = ctx.device_alloc::<u8>(temp_bytes.max(1))?;
    let mut chunk_offsets = ctx.device_alloc::<u64>(num_batches + 1)?;
    let codes_ptr = codes.cuda_device_ptr()?;
    let lens_ptr = lens.cuda_device_ptr()?;
    let total_tokens = u64::try_from(num_tokens)?;
    let stream = ctx.stream();
    let stream_ptr = stream.cu_stream() as cudaStream_t;
    let (bounds_ptr, record_bounds) = token_bounds.device_ptr(stream);
    let (offsets_ptr, record_offsets) = chunk_offsets.device_ptr_mut(stream);
    let (temp_ptr, record_temp) = temp.device_ptr_mut(stream);

    ctx.launch_external(num_tokens, || unsafe {
        onpair::batch_offsets(
            temp_ptr as *mut c_void,
            temp_bytes,
            codes_ptr as *const c_void,
            code_width,
            lens_ptr as *const u8,
            dict_size,
            bounds_ptr as *const u64,
            total_tokens,
            offsets_ptr as *mut u64,
            num_batches_i64,
            stream_ptr,
        )
        .map_err(|err| vortex_err!("CUB onpair_batch_offsets failed: {err}"))
    })?;
    drop((record_bounds, record_offsets, record_temp));

    Ok(chunk_offsets)
}

pub(crate) fn exclusive_sum_i32(
    input: &CudaSlice<i32>,
    len: usize,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<CudaSlice<i32>> {
    let len_i64 = i64::try_from(len)?;
    let temp_bytes = scan::exclusive_sum_i32_temp_size(len_i64)
        .map_err(|err| vortex_err!("CUB scan_exclusive_sum_i32_temp_size failed: {err}"))?;

    let mut temp = ctx.device_alloc::<u8>(temp_bytes.max(1))?;
    let mut output = ctx.device_alloc::<i32>(len)?;
    let stream = ctx.stream();
    let stream_ptr = stream.cu_stream() as cudaStream_t;
    let (input_ptr, record_input) = input.device_ptr(stream);
    let (output_ptr, record_output) = output.device_ptr_mut(stream);
    let (temp_ptr, record_temp) = temp.device_ptr_mut(stream);

    ctx.launch_external(len, || unsafe {
        scan::exclusive_sum_i32(
            temp_ptr as *mut c_void,
            temp_bytes,
            input_ptr as *const i32,
            output_ptr as *mut i32,
            len_i64,
            stream_ptr,
        )
        .map_err(|err| vortex_err!("CUB scan_exclusive_sum_i32 failed: {err}"))
    })?;
    drop((record_input, record_output, record_temp));

    Ok(output)
}

#[cfg(test)]
mod tests {
    use vortex::error::VortexExpect;

    use super::*;
    use crate::session::CudaSession;

    /// Upload synthetic codes, lengths, and a token window, regenerate the
    /// window-relative chunk offsets, and read them back. The code width
    /// follows the element type.
    async fn batch_offsets_roundtrip<C>(
        codes: Vec<C>,
        lens: Vec<u8>,
        window: std::ops::Range<u64>,
    ) -> VortexResult<Vec<u64>>
    where
        C: cudarc::driver::DeviceRepr
            + cudarc::driver::ValidAsZeroBits
            + std::fmt::Debug
            + Send
            + Sync
            + 'static,
    {
        let mut ctx = CudaSession::create_execution_ctx(&crate::cuda_session())?;
        let num_tokens = codes.len();
        let num_batches = num_tokens.div_ceil(128);
        let dict_size = u32::try_from(lens.len())?;
        let code_width = u32::try_from(size_of::<C>())?;

        let codes_dev = ctx.copy_to_device(codes)?.await?;
        let lens_dev = ctx.copy_to_device(lens)?.await?;
        let mut bounds = ctx.device_alloc::<u64>(2)?;
        ctx.stream()
            .memcpy_htod(&[window.start, window.end], &mut bounds)
            .map_err(|e| vortex_err!("Failed to upload token bounds: {e}"))?;

        let offsets = onpair_batch_offsets(
            &codes_dev,
            code_width,
            &lens_dev,
            dict_size,
            &bounds,
            num_tokens,
            num_batches,
            &mut ctx,
        )?;

        ctx.stream()
            .clone_dtoh(&offsets)
            .map_err(|e| vortex_err!("Failed to copy offsets to host: {e}"))
    }

    /// The window-relative exclusive prefix at 128-token boundaries, plus the
    /// trailing window total. Tokens outside `window` contribute zero bytes.
    fn host_reference(codes: &[u16], lens: &[u8], window: std::ops::Range<u64>) -> Vec<u64> {
        let mut expected = Vec::with_capacity(codes.len().div_ceil(128) + 1);
        expected.push(0u64);
        let mut acc = 0u64;
        for (i, &code) in codes.iter().enumerate() {
            if window.contains(&u64::try_from(i).vortex_expect("token index fits u64")) {
                acc += u64::from(lens[code as usize]);
            }
            if (i + 1) % 128 == 0 {
                expected.push(acc);
            }
        }
        if !codes.len().is_multiple_of(128) {
            expected.push(acc);
        }
        expected
    }

    /// A single partial batch: one tile, no look-back.
    #[crate::test]
    async fn test_onpair_batch_offsets_single_batch() -> VortexResult<()> {
        let lens: Vec<u8> = (1..=16).collect();
        let codes: Vec<u16> = (0..100u16).map(|i| i % 16).collect();
        let expected = host_reference(&codes, &lens, 0..100);

        let offsets = batch_offsets_roundtrip(codes, lens, 0..100).await?;
        assert_eq!(offsets, expected);
        Ok(())
    }

    /// Many look-back tiles with a ragged tail batch; the offsets must match
    /// a host prefix sum sampled at 128-token boundaries. Regression test for
    /// the look-back prefix being defined only in lane 0.
    #[crate::test]
    async fn test_onpair_batch_offsets_multi_tile() -> VortexResult<()> {
        let lens: Vec<u8> = (1..=16u8).cycle().take(300).collect();
        let num_tokens = 2000u64 * 128 - 57;
        let codes: Vec<u16> = (0..2000u32 * 128 - 57)
            .map(|i| u16::try_from(i * 31 % 300).vortex_expect("bounded by dictionary size"))
            .collect();
        let expected = host_reference(&codes, &lens, 0..num_tokens);

        let offsets = batch_offsets_roundtrip(codes, lens, 0..num_tokens).await?;
        assert_eq!(offsets, expected);
        Ok(())
    }

    /// A mid-stream token window: out-of-window tokens contribute zero bytes
    /// and are not range-checked, so the offsets are window-relative even
    /// when both boundaries land mid-batch.
    #[crate::test]
    async fn test_onpair_batch_offsets_windowed() -> VortexResult<()> {
        let lens: Vec<u8> = (1..=16u8).cycle().take(300).collect();
        let codes: Vec<u16> = (0..10u32 * 128 + 40)
            .map(|i| u16::try_from(i * 31 % 300).vortex_expect("bounded by dictionary size"))
            .collect();
        let window = 173..(codes.len() as u64 - 95);
        let expected = host_reference(&codes, &lens, window.clone());

        let offsets = batch_offsets_roundtrip(codes, lens, window).await?;
        assert_eq!(offsets, expected);
        Ok(())
    }

    /// u8 codes dispatch the narrow sweep instantiation.
    #[crate::test]
    async fn test_onpair_batch_offsets_u8_codes() -> VortexResult<()> {
        let lens: Vec<u8> = (1..=16).collect();
        let codes: Vec<u8> = (0..300u32)
            .map(|i| u8::try_from(i * 7 % 16).vortex_expect("bounded by dictionary size"))
            .collect();
        let widened: Vec<u16> = codes.iter().map(|&c| u16::from(c)).collect();
        let expected = host_reference(&widened, &lens, 0..300);

        let offsets = batch_offsets_roundtrip(codes, lens, 0..300).await?;
        assert_eq!(offsets, expected);
        Ok(())
    }

    /// A code outside the dictionary contributes zero bytes: the guard bounds
    /// the sweep's own reads, it does not report the inconsistency.
    #[crate::test]
    async fn test_onpair_batch_offsets_out_of_range_code_contributes_zero() -> VortexResult<()> {
        let lens = vec![2u8; 4];
        let mut codes = vec![1u16; 200];
        codes[130] = 9;

        let offsets = batch_offsets_roundtrip(codes, lens, 0..200).await?;
        assert_eq!(offsets, vec![0, 256, 256 + 71 * 2]);
        Ok(())
    }
}
