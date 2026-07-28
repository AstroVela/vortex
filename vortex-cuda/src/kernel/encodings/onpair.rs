// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA executor for OnPair decompression.
//!
//! The pipeline is sized and gated from the `uncompressed_lengths` child so
//! that nothing on the host ever waits for the codes sweep: the only readback
//! that depends on the codes chain is the single status word that gates the
//! unchecked decode kernel, and it is checked after the decode inputs are
//! fully staged.
//!
//! 1. The lengths child decodes on device and [`try_i32_offsets_from_lengths`]
//!    turns it into Arrow row offsets plus the window's total decoded byte
//!    count — the one early readback, issued before any codes work exists on
//!    the stream. The total sizes the output allocation and the row offsets
//!    are reused by every output path, replacing the old post-decode offsets
//!    scan and the readback of the sweep's heap total and byte bounds.
//! 2. `onpair_token_bounds` reads this array's token window from the
//!    device-resident `codes_offsets` child — the offsets are nondecreasing,
//!    so the window's min and max are its first and last elements. The window
//!    never reaches the host.
//! 3. `onpair_batch_offsets` (in the CUB shim) regenerates the per-batch
//!    output offsets (`chunk_offsets`) the decode kernel positions its writes
//!    with: one fused sweep reduces every 128-token batch's decoded size and
//!    exclusive-scans the sizes in-kernel via decoupled look-back. Tokens
//!    outside the device-read window contribute zero bytes, so the offsets
//!    come out window-relative and their trailing total is the window size.
//! 4. `onpair_validate` checks — on device — that the window is sane and that
//!    the sweep's window total equals the lengths-derived total the output
//!    was sized with. One readback of the shared status word then gates the
//!    decode kernel.
//! 5. `onpair_shmem_4tpt_split8read` gathers each window token's bytes from
//!    the split dictionary layout and scatters them window-relative into the
//!    window-sized output buffer — a sliced array decodes only its own rows,
//!    and no readback follows the decode launch.
//!
//! Every kernel that reads the codes is instantiated for the two widths
//! OnPair stores (u16 natively, u8 when the compressor narrowed the codes),
//! so the code stream is decompressed on device and never widened.
//!
//! The result is exposed either as a canonical `VarBinView` (views built
//! on-device by `onpair_build_views` from the step-1 row offsets, or on host
//! for windows that exceed a single backing buffer — the only path that
//! materialises the lengths) or as Arrow-compatible i32 offsets plus values
//! via [`decode_onpair_varbin`] — mirroring the FSST varbin path.

use std::fmt::Debug;
use std::sync::Arc;

use async_trait::async_trait;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DeviceRepr;
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use num_traits::AsPrimitive;
use tracing::instrument;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::arrays::primitive::PrimitiveDataParts;
use vortex::array::arrays::varbinview::build_views::MAX_BUFFER_LEN;
use vortex::array::arrays::varbinview::build_views::build_views;
use vortex::array::buffer::BufferHandle;
use vortex::array::buffer::DeviceBuffer;
use vortex::array::match_each_integer_ptype;
use vortex::array::validity::Validity;
use vortex::buffer::Buffer;
use vortex::dtype::DType;
use vortex::dtype::NativePType;
use vortex::dtype::PType;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;
use vortex_array::ArrayView;
use vortex_onpair::DictionaryView;
use vortex_onpair::MAX_TOKEN_SIZE;
use vortex_onpair::OnPair;
use vortex_onpair::OnPairArray;
use vortex_onpair::OnPairArrayExt;
use vortex_onpair::OnPairArraySlotsExt;
use vortex_onpair::dict_view;

use crate::CanonicalCudaExt;
use crate::CudaBufferExt;
use crate::CudaDeviceBuffer;
use crate::arrow::I32Offsets;
use crate::arrow::I32OffsetsOutcome;
use crate::arrow::try_i32_offsets_from_lengths;
use crate::cub::onpair_batch_offsets;
use crate::executor::CudaArrayExt;
use crate::executor::CudaExecute;
use crate::executor::CudaExecutionCtx;
use crate::kernel::encodings::DecodedVarBin;

// The kernels fix the dictionary row stride at 16 bytes (two `uint2` reads).
const _: () = assert!(MAX_TOKEN_SIZE == 16);

/// Tokens per decode batch: one decode-kernel warp emits 128 tokens (4 per
/// thread). Must match `ONPAIR_TOKENS_PER_BATCH` in `cub/kernels/onpair.cu`.
const TOKENS_PER_BATCH: usize = 128;
/// Threads per block for the warp-per-batch kernels (16 warps).
const BLOCK_THREADS: u32 = 512;
const WARPS_PER_BLOCK: usize = (BLOCK_THREADS / 32) as usize;

/// Launch config for the warp-per-batch kernels: one warp per 128-token batch.
fn batch_launch_config(num_batches: usize) -> VortexResult<LaunchConfig> {
    let grid_dim = u32::try_from(num_batches.div_ceil(WARPS_PER_BLOCK))?;
    Ok(LaunchConfig {
        grid_dim: (grid_dim, 1, 1),
        block_dim: (BLOCK_THREADS, 1, 1),
        shared_mem_bytes: 0,
    })
}

/// CUDA decoder for OnPair.
#[derive(Debug)]
pub(crate) struct OnPairExecutor;

#[async_trait]
impl CudaExecute for OnPairExecutor {
    #[instrument(level = "trace", skip_all, fields(executor = ?self))]
    async fn execute(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Canonical> {
        let onpair = array
            .as_typed::<OnPair>()
            .ok_or_else(|| vortex_err!("Expected OnPairArray"))?;
        decode_onpair(onpair, ctx).await
    }
}

/// Checked host sum of the per-row decoded lengths, used only on the cold
/// rollover path whose window exceeds Arrow's i32 offset range. A negative
/// length sign-extends and surfaces as overflow here or as a device-validated
/// mismatch against the GPU-computed window size.
fn sum_lengths(lengths: &PrimitiveArray) -> VortexResult<u64> {
    match_each_integer_ptype!(lengths.ptype(), |P| {
        let mut acc = 0u64;
        for &length in lengths.as_slice::<P>() {
            acc = acc
                .checked_add(AsPrimitive::<u64>::as_(length))
                .ok_or_else(|| vortex_err!("OnPair decoded size overflow"))?;
        }
        Ok(acc)
    })
}

/// All-empty output: `num_rows` inline empty views and no backing buffers.
async fn empty_views(
    num_rows: usize,
    dtype: DType,
    validity: Validity,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Canonical> {
    let views = ctx.copy_to_device(vec![0i128; num_rows])?.await?;
    Ok(Canonical::VarBinView(unsafe {
        VarBinViewArray::new_handle_unchecked(views, Arc::from([]), dtype, validity)
    }))
}

/// The device-staged compressed token stream: the full code stream at its
/// native width, the split dictionary layout, and the regenerated per-batch
/// output offsets.
struct StagedCodes {
    codes: BufferHandle,
    dict_s8: BufferHandle,
    dict_padded: BufferHandle,
    lens: BufferHandle,
    /// Exclusive per-batch window-relative output offsets, `num_batches + 1`
    /// entries; the last is the total decoded byte count of the visible token
    /// window.
    chunk_offsets: CudaSlice<u64>,
    num_batches: usize,
    num_tokens: usize,
    launch_config: LaunchConfig,
}

/// The shared result of the OnPair GPU decode pipeline.
struct OnPairDecoded {
    /// This array's rows' decoded bytes: the window-sized decode output.
    bytes: BufferHandle,
    /// Byte size of the window: the `uncompressed_lengths` sum, validated on
    /// device against the decoded size of the code window.
    total_size: usize,
    /// Device-built Arrow row offsets over the window's decoded bytes, shared
    /// by the varbin path and the canonical fast path. `None` when the window
    /// exceeds Arrow's i32 offset range — the host rollover path.
    row_offsets: Option<I32Offsets>,
    /// Per-row lengths, resident wherever `execute_cuda` produced them. Only
    /// the host rollover path materialises them.
    lengths: PrimitiveArray,
}

/// Stage this array's device-decompressed codes and dictionary and regenerate
/// the decode kernel's window-relative per-batch output offsets from them in
/// one fused sweep (see [`onpair_batch_offsets`]); the token window is read
/// on device from `token_bounds`. The caller has validated that the codes are
/// u8 or u16; the sweep reads them at their native width.
async fn stage_codes(
    onpair: ArrayView<'_, OnPair>,
    codes: PrimitiveArray,
    token_bounds: &CudaSlice<u64>,
    status: &mut CudaSlice<u32>,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<StagedCodes> {
    let num_tokens = codes.len();
    let code_width = u32::try_from(codes.ptype().byte_width())?;
    let PrimitiveDataParts {
        buffer: codes_buffer,
        ..
    } = codes.into_data_parts();

    // Stage the dictionary in the decode kernel's split layout: fixed 16-byte
    // rows (`dict_padded`, the rare `len > 8` read), the first 8 bytes of every
    // row (`dict_s8`, the common-case read), and the per-code lengths.
    let dict = dict_view(onpair, ctx.execution_ctx())?;
    let dict_size = dict.num_tokens();
    let dict_size_u32 = u32::try_from(dict_size)?;
    let mut dict_padded = vec![0u8; dict_size * MAX_TOKEN_SIZE];
    let mut dict_s8 = vec![0u8; dict_size * 8];
    let mut lens = vec![0u8; dict_size];
    for code in 0..dict_size {
        let token =
            dict.token(u16::try_from(code).vortex_expect("dictionary has at most 2^16 tokens"));
        let len = token.len();
        lens[code] = u8::try_from(len).vortex_expect("token length is at most MAX_TOKEN_SIZE");
        dict_padded[code * MAX_TOKEN_SIZE..code * MAX_TOKEN_SIZE + len].copy_from_slice(token);
        let head = len.min(8);
        dict_s8[code * 8..code * 8 + head].copy_from_slice(&token[..head]);
    }

    let (codes_dev, s8_dev, padded_dev, lens_dev) = futures::try_join!(
        ctx.ensure_on_device(codes_buffer),
        ctx.copy_to_device(dict_s8)?,
        ctx.copy_to_device(dict_padded)?,
        ctx.copy_to_device(lens)?,
    )?;

    let num_batches = num_tokens.div_ceil(TOKENS_PER_BATCH);
    let launch_config = batch_launch_config(num_batches)?;
    let chunk_offsets = onpair_batch_offsets(
        &codes_dev,
        code_width,
        &lens_dev,
        dict_size_u32,
        token_bounds,
        num_tokens,
        num_batches,
        status,
        ctx,
    )?;

    Ok(StagedCodes {
        codes: codes_dev,
        dict_s8: s8_dev,
        dict_padded: padded_dev,
        lens: lens_dev,
        chunk_offsets,
        num_batches,
        num_tokens,
        launch_config,
    })
}

/// Run the OnPair decode pipeline: build the row offsets and the window's
/// decoded byte count from the lengths child, stage the codes and dictionary,
/// regenerate the window-relative per-batch output offsets on the device,
/// validate the compressed stream on device, and decode the window's byte
/// stream. A sliced array keeps its whole `codes` child; its token window is
/// resolved on device from `codes_offsets` and only the window's rows are
/// decoded — the codes never round-trip through the host, and the sole
/// readback that depends on them is the status word gating the decode kernel.
/// Returns `Ok(None)` when there is nothing to decode: the array is empty,
/// every row is null, or there are no codes at all.
async fn decode_onpair_bytes(
    onpair: ArrayView<'_, OnPair>,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Option<OnPairDecoded>> {
    let num_rows = onpair.len();

    if num_rows == 0 {
        return Ok(None);
    }

    // Every row null (cheap metadata check): nothing to decode. A sliced
    // all-null window usually carries a validity child instead of the
    // `AllInvalid` marker and decodes as an empty token window instead.
    if onpair.array_validity().definitely_all_null() {
        return Ok(None);
    }

    let lengths = onpair
        .uncompressed_lengths()
        .clone()
        .execute_cuda(ctx)
        .await?
        .into_primitive();

    // The pipeline is sized from the lengths child: the row offsets and the
    // window's decoded byte count are built on device up front, so this — the
    // one readback the decode allocation waits for — depends only on the
    // lengths chain; no codes work has been enqueued yet.
    let (row_offsets, total_size, lengths) =
        match try_i32_offsets_from_lengths(lengths.clone(), ctx).await? {
            I32OffsetsOutcome::Offsets(row_offsets) => {
                let total_size = row_offsets.total;
                (Some(row_offsets), total_size, lengths)
            }
            // Cold path: the window exceeds Arrow's i32 offset range, so the
            // canonical output must roll the bytes over on host — which needs
            // the lengths on host anyway. Carry the host copy forward.
            I32OffsetsOutcome::Overflow => {
                let host_lengths = Canonical::Primitive(lengths)
                    .into_host()
                    .await?
                    .into_primitive();
                let total_size = usize::try_from(sum_lengths(&host_lengths)?)?;
                (None, total_size, host_lengths)
            }
        };

    // No codes at all (e.g. every row empty): the child's length is host
    // metadata and the lengths total is already on host, so this early-out
    // costs no extra device read.
    if onpair.codes().is_empty() {
        vortex_ensure!(
            total_size == 0,
            "OnPair records {total_size} decoded bytes but has no codes"
        );
        return Ok(None);
    }

    // Decompress the per-row code boundaries on device; the token window is
    // resolved from them by a kernel, never by host scalar reads.
    let codes_offsets = onpair
        .codes_offsets()
        .clone()
        .execute_cuda(ctx)
        .await?
        .into_primitive();

    // Decompress the codes child on device. The kernels are instantiated for
    // the two widths OnPair stores — u16 natively, u8 when the compressor
    // narrowed the codes — so no widening pass is needed.
    let codes = onpair
        .codes()
        .clone()
        .execute_cuda(ctx)
        .await?
        .into_primitive();
    let bytes = match codes.ptype() {
        PType::U8 => decode_window::<u8>(onpair, codes, codes_offsets, total_size, ctx).await?,
        PType::U16 => decode_window::<u16>(onpair, codes, codes_offsets, total_size, ctx).await?,
        other => vortex_bail!("OnPair codes must decompress to u8 or u16, got {other}"),
    };

    Ok(Some(OnPairDecoded {
        bytes,
        total_size,
        row_offsets,
        lengths,
    }))
}

/// Stage the codes at their native width `C`, resolve the token window and
/// the window-relative batch offsets on device, validate the stream on
/// device, and decode the window into a window-sized buffer. The only
/// readback is the status word gating the unchecked decode kernel; the output
/// allocation was already sized by the caller from the lengths child, and the
/// device-side validation pins the decoded window to exactly `total_size`
/// bytes before the decode may run.
async fn decode_window<C>(
    onpair: ArrayView<'_, OnPair>,
    codes: PrimitiveArray,
    codes_offsets: PrimitiveArray,
    total_size: usize,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<BufferHandle>
where
    C: NativePType + DeviceRepr + Send + Sync + 'static,
{
    // Corruption flag shared by the whole device-side pipeline: the sweep
    // raises 1 for a window code outside the dictionary and the validate
    // kernel raises 2/3 for bad bounds or a size mismatch. Checked once
    // before the unchecked decode kernel is allowed to run.
    let mut status = ctx.device_alloc::<u32>(1)?;
    ctx.stream()
        .memset_zeros(&mut status)
        .map_err(|e| vortex_err!("Failed to zero OnPair status flag: {e}"))?;

    // Token bounds of this array's rows in the full code stream, read on
    // device from the (possibly slice-narrowed) `codes_offsets` child: the
    // offsets are nondecreasing, so the window's min and max are its first
    // and last elements. The bounds stay on device; the sweep, the validate
    // kernel, and the decode kernel all read them there.
    let mut bounds = ctx.device_alloc::<u64>(2)?;
    let offsets_ptype = codes_offsets.ptype();
    let last_boundary = u64::try_from(codes_offsets.len().saturating_sub(1))?;
    let PrimitiveDataParts {
        buffer: offsets_buffer,
        ..
    } = codes_offsets.into_data_parts();
    let offsets_dev = ctx.ensure_on_device(offsets_buffer).await?;
    let bounds_fn =
        ctx.load_function_with_suffixes("onpair", &["token_bounds", &offsets_ptype.to_string()])?;
    match_each_integer_ptype!(offsets_ptype, |O| {
        let offsets_view = offsets_dev.cuda_view::<O>()?;
        ctx.launch_kernel_config(
            &bounds_fn,
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            },
            1,
            |args| {
                args.arg(&offsets_view).arg(&last_boundary).arg(&mut bounds);
            },
        )?;
    });

    let staged = stage_codes(onpair, codes, &bounds, &mut status, ctx).await?;

    // Device-side validation folds every precondition of the unchecked decode
    // kernel into the shared status word: the token window must be sane and
    // the window's decoded byte count (the sweep's trailing total) must equal
    // the lengths-derived size the output buffer is allocated with.
    let num_batches_u64 = u64::try_from(staged.num_batches)?;
    let num_tokens_u64 = u64::try_from(staged.num_tokens)?;
    let lengths_total = u64::try_from(total_size)?;
    let validate_fn = ctx.load_function_with_suffixes("onpair", &["validate"])?;
    ctx.launch_kernel_config(
        &validate_fn,
        LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        },
        1,
        |args| {
            args.arg(&bounds)
                .arg(&staged.chunk_offsets)
                .arg(&num_batches_u64)
                .arg(&num_tokens_u64)
                .arg(&lengths_total)
                .arg(&mut status);
        },
    )?;

    // The single readback that depends on the codes chain: one status word
    // gates the decode kernel, whose dictionary gathers and output scatters
    // are unchecked. Everything the old pipeline read back here — the heap
    // total and the byte window — is subsumed by the device-side validation
    // against the lengths-derived window size.
    let status = Buffer::<u32>::from_byte_buffer(
        BufferHandle::new_device(Arc::new(CudaDeviceBuffer::new(status)))
            .try_to_host()?
            .await?,
    );
    match status.first().copied().unwrap_or(u32::MAX) {
        0 => {}
        1 => vortex_bail!("OnPair code out of dictionary range"),
        2 => vortex_bail!(
            "OnPair codes_offsets must be nondecreasing and end within the codes child"
        ),
        3 => vortex_bail!(
            "OnPair codes decode to a different byte count than uncompressed_lengths records"
        ),
        status => vortex_bail!("unexpected OnPair decode status {status}"),
    }

    // Decode the window into a window-sized buffer; nothing waits on the
    // decode kernel. The kernel's drain gates 16-byte stores on
    // `out_start % 16` relative to the buffer base, so the base must be
    // 16-aligned.
    let mut bytes = ctx.device_alloc::<u8>(total_size.max(1))?;
    let (bytes_base_ptr, _) = bytes.device_ptr(ctx.stream());
    assert_eq!(
        bytes_base_ptr % 16,
        0,
        "output base not 16-aligned: {bytes_base_ptr:#x}",
    );

    let ptype = C::PTYPE.to_string();
    let codes_view = staged.codes.cuda_view::<C>()?;
    let s8_view = staged.dict_s8.cuda_view::<u8>()?;
    let padded_view = staged.dict_padded.cuda_view::<u8>()?;
    let lens_view = staged.lens.cuda_view::<u8>()?;
    let decode_fn = ctx.load_function_with_suffixes("onpair_shmem_4tpt_split8read", &[&ptype])?;
    ctx.launch_kernel_config(
        &decode_fn,
        staged.launch_config,
        staged.num_tokens,
        |args| {
            args.arg(&codes_view)
                .arg(&staged.chunk_offsets)
                .arg(&s8_view)
                .arg(&padded_view)
                .arg(&lens_view)
                .arg(&mut bytes)
                .arg(&bounds);
        },
    )?;

    let heap = CudaDeviceBuffer::new(bytes);
    Ok(BufferHandle::new_device(heap.slice(0..total_size)))
}

async fn decode_onpair(
    onpair: ArrayView<'_, OnPair>,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Canonical> {
    let dtype = onpair.dtype().clone();
    let validity = onpair.array_validity();
    let num_rows = onpair.len();

    if onpair.is_empty() {
        return Ok(Canonical::empty(&dtype));
    }

    let Some(decoded) = decode_onpair_bytes(onpair, ctx).await? else {
        return empty_views(num_rows, dtype, validity, ctx).await;
    };
    let OnPairDecoded {
        bytes,
        total_size,
        row_offsets,
        lengths,
    } = decoded;

    // An empty window (e.g. a slice covering only null or empty rows) needs
    // no backing buffer at all.
    if total_size == 0 {
        return empty_views(num_rows, dtype, validity, ctx).await;
    }

    // Fast path: the decoded window fits a single BinaryView backing buffer
    // (`MAX_BUFFER_LEN`, i32::MAX), so the row offsets built on device before
    // the decode launched are reused to build the views — nothing touches the
    // host, and nothing here waits for the decode kernel.
    if let Some(I32Offsets {
        buffer: row_offsets,
        ..
    }) = row_offsets
    {
        let row_offsets_view = row_offsets.cuda_view::<i32>()?;
        let bytes_view = bytes.cuda_view::<u8>()?;
        let mut device_views = ctx.device_alloc::<i128>(num_rows)?;
        let num_rows_u64 = u64::try_from(num_rows)?;
        let build_views_fn = ctx.load_function_with_suffixes("onpair", &["build_views"])?;
        ctx.launch_kernel(&build_views_fn, num_rows, |args| {
            args.arg(&row_offsets_view)
                .arg(&bytes_view)
                .arg(&mut device_views)
                .arg(&num_rows_u64);
        })?;

        let views = BufferHandle::new_device(Arc::new(CudaDeviceBuffer::new(device_views)));
        return Ok(Canonical::VarBinView(unsafe {
            VarBinViewArray::new_handle_unchecked(views, Arc::from([bytes]), dtype, validity)
        }));
    }

    // BinaryView offsets are u32. Windows that need multiple backing buffers
    // roll the decoded bytes over on host, mirroring the CPU canonical path;
    // only here do the lengths leave the device. The decoded byte count was
    // validated against the lengths on device before the decode ran.
    let lengths = Canonical::Primitive(lengths)
        .into_host()
        .await?
        .into_primitive();
    let host_bytes = bytes.try_to_host()?.await?;

    let (buffers, views) = match_each_integer_ptype!(lengths.ptype(), |P| {
        build_views(
            0,
            MAX_BUFFER_LEN,
            host_bytes.into_mut(),
            lengths.as_slice::<P>(),
        )
    });

    Ok(Canonical::VarBinView(unsafe {
        VarBinViewArray::new_unchecked(views, Arc::from(buffers), dtype, validity)
    }))
}

/// Decode OnPair directly into Arrow-compatible i32 offsets and contiguous
/// values on device, mirroring the FSST varbin path.
pub(crate) async fn decode_onpair_varbin(
    onpair: OnPairArray,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<DecodedVarBin> {
    let dtype = onpair.dtype().clone();
    let validity = onpair.array_validity();
    let len = onpair.len();

    let Some(decoded) = decode_onpair_bytes(onpair.as_view(), ctx).await? else {
        // Zero decoded bytes: all-zero offsets and an empty values heap.
        let offsets = ctx.copy_to_device(vec![0i32; len + 1])?.await?;
        let allocation = CudaDeviceBuffer::new(ctx.device_alloc::<u8>(1)?);
        let values = BufferHandle::new_device(allocation.slice(0..0));
        return Ok(DecodedVarBin {
            dtype,
            len,
            offsets,
            values,
            validity,
        });
    };

    let OnPairDecoded {
        bytes, row_offsets, ..
    } = decoded;

    // The Arrow i32 offsets were built on device from the lengths before the
    // decode launched; a window beyond Arrow's i32 offset range has no
    // offset-based representation.
    let Some(I32Offsets {
        buffer: offsets, ..
    }) = row_offsets
    else {
        vortex_bail!("length sum exceeds Arrow i32 offset range");
    };

    Ok(DecodedVarBin {
        dtype,
        len,
        offsets,
        values: bytes,
        validity,
    })
}

#[cfg(test)]
mod tests {
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use rstest::rstest;
    use vortex::array::IntoArray;
    use vortex::array::arrays::VarBinArray;
    use vortex::array::assert_arrays_eq;
    use vortex::buffer::Buffer;
    use vortex::dtype::Nullability;
    use vortex::error::VortexExpect;
    use vortex_array::VortexSessionExecute;
    use vortex_onpair::DEFAULT_DICT12_CONFIG;
    use vortex_onpair::onpair_compress;

    use super::*;
    use crate::CanonicalCudaExt;
    use crate::arrow::DeviceArrayExt;
    use crate::arrow::release_device_array;
    use crate::arrow::release_schema;
    use crate::session::CudaSession;
    use crate::session::VarBinExportLayout;

    fn cuda_ctx_with_varbin_layout(layout: VarBinExportLayout) -> VortexResult<CudaExecutionCtx> {
        let session = vortex::array::array_session()
            .with_some(CudaSession::try_default()?.with_varbin_export_layout(layout));
        CudaSession::create_execution_ctx(&session)
    }

    fn assert_device_resident(canonical: &Canonical) {
        let varbinview = canonical.as_varbinview();
        assert!(varbinview.views_handle().is_on_device());
        assert!(
            varbinview
                .data_buffers()
                .iter()
                .all(BufferHandle::is_on_device)
        );
    }

    fn compress_onpair(
        strings: Vec<Option<&'static [u8]>>,
        dtype: DType,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let varbin = VarBinArray::from_iter(strings, dtype).into_array();
        let onpair = onpair_compress(&varbin, DEFAULT_DICT12_CONFIG, ctx.execution_ctx())?;
        vortex_ensure!(
            onpair.as_opt::<OnPair>().is_some(),
            "expected OnPair array, got {}",
            onpair.encoding_id()
        );
        Ok(onpair)
    }

    #[rstest]
    #[case::binary_non_null(
        vec![Some(&b"the quick brown fox"[..]),
             Some(&b"jumps over the lazy dog"[..]),
             Some(&b"hello world"[..]),
             Some(&b"vortex onpair test string"[..])],
        DType::Binary(Nullability::NonNullable),
    )]
    #[case::utf8_non_null(
        vec![Some(&b"the quick brown fox"[..]),
             Some(&b"jumps over the lazy dog"[..]),
             Some(&b"hello world"[..]),
             Some(&b"vortex onpair test string"[..])],
        DType::Utf8(Nullability::NonNullable),
    )]
    #[case::utf8_inline_boundary(
        vec![Some(&b""[..]),
             Some(&b"123456789012"[..]),
             Some(&b"1234567890123"[..]),
             Some(&b"this is another outlined value"[..])],
        DType::Utf8(Nullability::NonNullable),
    )]
    #[case::utf8_partial_nulls(
        vec![Some(&b"alpha"[..]), None, Some(&b"gamma"[..]), None, Some(&b"epsilon"[..])],
        DType::Utf8(Nullability::Nullable),
    )]
    #[case::binary_all_empty(
        vec![Some(&b""[..]), Some(&b""[..]), Some(&b""[..])],
        DType::Binary(Nullability::NonNullable),
    )]
    #[crate::test]
    async fn test_cuda_onpair_decompression_roundtrip(
        #[case] strings: Vec<Option<&'static [u8]>>,
        #[case] dtype: DType,
    ) -> VortexResult<()> {
        let mut ctx = vortex_array::array_session().create_execution_ctx();
        let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())
            .vortex_expect("failed to create execution context");

        let onpair = compress_onpair(strings, dtype.clone(), &mut cuda_ctx)?;

        let gpu_result = OnPairExecutor
            .execute(onpair.clone(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed");
        assert_eq!(gpu_result.dtype(), &dtype);
        assert_device_resident(&gpu_result);

        let host_result = gpu_result.into_host().await?.into_array();
        assert_arrays_eq!(onpair, host_result, &mut ctx);
        Ok(())
    }

    /// A slice keeps the whole `codes` child and narrows only `codes_offsets`,
    /// so this exercises the nonzero `code_start` window.
    #[crate::test]
    async fn test_cuda_onpair_decompression_sliced() -> VortexResult<()> {
        let mut ctx = vortex_array::array_session().create_execution_ctx();
        let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())
            .vortex_expect("failed to create execution context");
        let values = vec![
            Some(&b"before the window"[..]),
            None,
            Some(&b"the quick brown fox"[..]),
            None,
            Some(&b"after the window"[..]),
        ];
        let onpair = compress_onpair(values, DType::Utf8(Nullability::Nullable), &mut cuda_ctx)?;
        let sliced = onpair.slice(1..4)?;

        let gpu_result = OnPairExecutor
            .execute(sliced.clone(), &mut cuda_ctx)
            .await?;
        assert_device_resident(&gpu_result);
        let host_result = gpu_result.into_host().await?.into_array();
        assert_arrays_eq!(sliced, host_result, &mut ctx);
        Ok(())
    }

    /// A slice covering only null rows decodes zero bytes and takes the
    /// empty-views path.
    #[crate::test]
    async fn test_cuda_onpair_decompression_null_slice() -> VortexResult<()> {
        let mut ctx = vortex_array::array_session().create_execution_ctx();
        let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())
            .vortex_expect("failed to create execution context");
        let values = vec![Some(&b"alpha"[..]), None, None, Some(&b"omega"[..])];
        let onpair = compress_onpair(values, DType::Utf8(Nullability::Nullable), &mut cuda_ctx)?;
        let sliced = onpair.slice(1..3)?;

        let gpu_result = OnPairExecutor
            .execute(sliced.clone(), &mut cuda_ctx)
            .await?;
        assert_device_resident(&gpu_result);
        let host_result = gpu_result.into_host().await?.into_array();
        assert_arrays_eq!(sliced, host_result, &mut ctx);
        Ok(())
    }

    /// Exercises many 128-token batches and the multi-block decode grid.
    #[crate::test]
    async fn test_cuda_onpair_decompression_roundtrip_large() -> VortexResult<()> {
        let mut ctx = vortex_array::array_session().create_execution_ctx();
        let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())
            .vortex_expect("failed to create execution context");

        let strings: Vec<String> = (0..100_000)
            .map(|i| format!("https://www.example.com/path/{i}/segment?q={}", i % 97))
            .collect();
        let varbin = VarBinArray::from_iter(
            strings.iter().map(|s| Some(s.as_str())),
            DType::Utf8(Nullability::NonNullable),
        )
        .into_array();
        let onpair = onpair_compress(&varbin, DEFAULT_DICT12_CONFIG, cuda_ctx.execution_ctx())?;

        let gpu_result = OnPairExecutor
            .execute(onpair.clone(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed");
        assert_device_resident(&gpu_result);

        let host_result = gpu_result.into_host().await?.into_array();
        assert_arrays_eq!(onpair, host_result, &mut ctx);
        Ok(())
    }

    /// A slice deep into a large array: both code-window boundaries land
    /// mid-batch in non-zero batches, exercising the on-device window-bounds
    /// resolution (whole-batch prefix plus partial-batch reduction) and the
    /// zero-copy window slice of the full decoded heap.
    #[crate::test]
    async fn test_cuda_onpair_decompression_sliced_large() -> VortexResult<()> {
        let mut ctx = vortex_array::array_session().create_execution_ctx();
        let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())
            .vortex_expect("failed to create execution context");

        let strings: Vec<String> = (0..40_000)
            .map(|i| format!("https://www.example.com/path/{i}/segment?q={}", i % 97))
            .collect();
        let varbin = VarBinArray::from_iter(
            strings.iter().map(|s| Some(s.as_str())),
            DType::Utf8(Nullability::NonNullable),
        )
        .into_array();
        let onpair = onpair_compress(&varbin, DEFAULT_DICT12_CONFIG, cuda_ctx.execution_ctx())?;
        let sliced = onpair.slice(19_997..20_101)?;

        let gpu_result = OnPairExecutor
            .execute(sliced.clone(), &mut cuda_ctx)
            .await?;
        assert_device_resident(&gpu_result);
        let host_result = gpu_result.into_host().await?.into_array();
        assert_arrays_eq!(sliced, host_result, &mut ctx);
        Ok(())
    }

    /// Codes narrowed to u8 dispatch the u8 kernel instantiations end to end.
    /// A trained dictionary always holds the 256 single-byte tokens sorted
    /// among its merges, so real merge codes never fit u8; the u8-addressable
    /// case is the minimal alphabet-only dictionary, where token id `b` is
    /// exactly the byte `b` and every row is coded byte per byte.
    #[crate::test]
    async fn test_cuda_onpair_decompression_u8_codes() -> VortexResult<()> {
        let mut ctx = vortex_array::array_session().create_execution_ctx();
        let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())
            .vortex_expect("failed to create execution context");

        let strings: Vec<&[u8]> = [
            &b"tokenized token stream"[..],
            b"tokenized",
            b"token stream",
            b"stream of tokens",
        ]
        .into_iter()
        .cycle()
        .take(800)
        .collect();

        // The alphabet-only compact dictionary: the 256 single-byte tokens
        // (sorted by construction) plus the trailing read padding.
        let mut dict_bytes: Vec<u8> = (0..=u8::MAX).collect();
        dict_bytes.resize(255 + MAX_TOKEN_SIZE, 0);
        let dict_offsets: Vec<u32> = (0..=256).collect();

        let codes: Vec<u8> = strings.concat();
        let mut codes_offsets = vec![0u32];
        let mut lengths = Vec::with_capacity(strings.len());
        let mut acc = 0u32;
        for s in &strings {
            let len = u32::try_from(s.len())?;
            lengths.push(len);
            acc += len;
            codes_offsets.push(acc);
        }

        let onpair = OnPair::try_new(
            DType::Utf8(Nullability::NonNullable),
            BufferHandle::new_host(Buffer::from(dict_bytes).into_byte_buffer()),
            Buffer::from(dict_offsets).into_array(),
            Buffer::from(codes).into_array(),
            Buffer::from(codes_offsets).into_array(),
            Buffer::from(lengths).into_array(),
            Validity::NonNullable,
        )?;

        let expected = VarBinArray::from_iter(
            strings.iter().map(|s| Some(*s)),
            DType::Utf8(Nullability::NonNullable),
        )
        .into_array();

        let gpu_result = OnPairExecutor
            .execute(onpair.into_array(), &mut cuda_ctx)
            .await?;
        assert_device_resident(&gpu_result);
        let host_result = gpu_result.into_host().await?.into_array();
        assert_arrays_eq!(expected, host_result, &mut ctx);
        Ok(())
    }

    /// The decoded size of the code window is validated on device against the
    /// `uncompressed_lengths` sum before the unchecked decode kernel may run:
    /// a corrupt length must fail the status gate, not scatter out of bounds.
    #[crate::test]
    async fn test_cuda_onpair_rejects_mismatched_lengths() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())
            .vortex_expect("failed to create execution context");

        let strings: Vec<&[u8]> = vec![b"alpha", b"beta", b"gamma"];
        let mut dict_bytes: Vec<u8> = (0..=u8::MAX).collect();
        dict_bytes.resize(255 + MAX_TOKEN_SIZE, 0);
        let dict_offsets: Vec<u32> = (0..=256).collect();

        let codes: Vec<u8> = strings.concat();
        let mut codes_offsets = vec![0u32];
        let mut lengths = Vec::with_capacity(strings.len());
        let mut acc = 0u32;
        for s in &strings {
            let len = u32::try_from(s.len())?;
            lengths.push(len);
            acc += len;
            codes_offsets.push(acc);
        }
        // One row claims an extra byte the codes never decode.
        lengths[1] += 1;

        let onpair = OnPair::try_new(
            DType::Utf8(Nullability::NonNullable),
            BufferHandle::new_host(Buffer::from(dict_bytes).into_byte_buffer()),
            Buffer::from(dict_offsets).into_array(),
            Buffer::from(codes).into_array(),
            Buffer::from(codes_offsets).into_array(),
            Buffer::from(lengths).into_array(),
            Validity::NonNullable,
        )?;

        let result = OnPairExecutor
            .execute(onpair.into_array(), &mut cuda_ctx)
            .await;
        let err = result.err().vortex_expect("mismatched lengths must fail");
        assert!(
            err.to_string().contains("uncompressed_lengths"),
            "unexpected error: {err}"
        );
        Ok(())
    }

    #[crate::test]
    async fn test_cuda_onpair_direct_varbin_output() -> VortexResult<()> {
        let mut cuda_ctx = cuda_ctx_with_varbin_layout(VarBinExportLayout::VarBin)?;
        let values: [&[u8]; 3] = [
            b"",
            b"short",
            b"this value is stored directly in the values buffer",
        ];
        let onpair = compress_onpair(
            values.iter().map(|v| Some(*v)).collect(),
            DType::Utf8(Nullability::NonNullable),
            &mut cuda_ctx,
        )?
        .try_downcast::<OnPair>()
        .map_err(|array| vortex_err!("expected OnPair array, got {}", array.encoding_id()))?;

        let output = decode_onpair_varbin(onpair, &mut cuda_ctx).await?;
        assert_eq!(output.dtype, DType::Utf8(Nullability::NonNullable));
        assert_eq!(output.len, values.len());
        assert!(output.offsets.is_on_device());
        assert!(output.values.is_on_device());

        let offsets = Buffer::<i32>::from_byte_buffer(output.offsets.try_to_host()?.await?);
        assert_eq!(
            offsets.as_slice(),
            &[0, 0, 5, i32::try_from(5 + values[2].len())?,]
        );
        assert_eq!(
            output.values.try_to_host()?.await?.as_ref(),
            values.concat()
        );
        Ok(())
    }

    #[rstest]
    #[case::binary(
        DType::Binary(Nullability::NonNullable),
        VarBinExportLayout::VarBin,
        DataType::Binary,
        3
    )]
    #[case::utf8(
        DType::Utf8(Nullability::NonNullable),
        VarBinExportLayout::VarBin,
        DataType::Utf8,
        3
    )]
    #[case::binary_view(
        DType::Binary(Nullability::NonNullable),
        VarBinExportLayout::VarBinView,
        DataType::BinaryView,
        4
    )]
    #[case::utf8_view(
        DType::Utf8(Nullability::NonNullable),
        VarBinExportLayout::VarBinView,
        DataType::Utf8View,
        4
    )]
    #[crate::test]
    async fn test_cuda_onpair_arrow_export_uses_dtype_layout(
        #[case] dtype: DType,
        #[case] layout: VarBinExportLayout,
        #[case] expected_data_type: DataType,
        #[case] expected_n_buffers: i64,
    ) -> VortexResult<()> {
        let mut cuda_ctx = cuda_ctx_with_varbin_layout(layout)?;
        let values = vec![
            Some(&b"short"[..]),
            Some(&b"this value is stored out of line"[..]),
        ];
        let onpair = compress_onpair(values, dtype, &mut cuda_ctx)?;

        let mut exported = onpair
            .export_device_array_with_schema(&mut cuda_ctx)
            .await?;
        assert_eq!(
            Field::try_from(&exported.schema)?,
            Field::new("", expected_data_type, false)
        );
        assert_eq!(exported.array.array.n_buffers, expected_n_buffers);

        release_device_array(&mut exported.array);
        release_schema(&mut exported.schema);
        Ok(())
    }
}
