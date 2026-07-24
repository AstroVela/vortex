// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::min;
use std::mem::MaybeUninit;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::expr::stats::Precision;
use vortex_array::expr::stats::Stat;
use vortex_array::match_each_native_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_mask::Mask;

/// Run-end encode a `PrimitiveArray`, returning a tuple of `(ends, values)`.
pub fn runend_encode(
    array: ArrayView<Primitive>,
    ctx: &mut ExecutionCtx,
) -> (PrimitiveArray, ArrayRef) {
    let validity = match array
        .validity()
        .vortex_expect("run-end validity should be derivable")
    {
        Validity::NonNullable => None,
        Validity::AllValid => None,
        Validity::AllInvalid => {
            // We can trivially return an all-null REE array
            let ends = PrimitiveArray::new(buffer![array.len() as u64], Validity::NonNullable);
            ends.statistics()
                .set(Stat::IsStrictSorted, Precision::Exact(true.into()));
            return (
                ends,
                ConstantArray::new(Scalar::null(array.dtype().clone()), 1).into_array(),
            );
        }
        Validity::Array(a) => {
            let bool_array = a
                .execute::<BoolArray>(ctx)
                .vortex_expect("validity array must be convertible to bool");
            Some(bool_array.to_bit_buffer())
        }
    };

    let (ends, values) = match validity {
        None => {
            match_each_native_ptype!(array.ptype(), |P| {
                let (ends, values) = runend_encode_primitive(array.as_slice::<P>());
                (
                    PrimitiveArray::new(ends, Validity::NonNullable),
                    PrimitiveArray::new(values, array.dtype().nullability().into()).into_array(),
                )
            })
        }
        Some(validity) => {
            match_each_native_ptype!(array.ptype(), |P| {
                let (ends, values) =
                    runend_encode_nullable_primitive(array.as_slice::<P>(), validity);
                (
                    PrimitiveArray::new(ends, Validity::NonNullable),
                    values.into_array(),
                )
            })
        }
    };

    let ends = ends
        .narrow(ctx)
        .vortex_expect("Ends must succeed downcasting");

    ends.statistics()
        .set(Stat::IsStrictSorted, Precision::Exact(true.into()));

    (ends, values)
}

fn runend_encode_primitive<T: NativePType>(elements: &[T]) -> (Buffer<u64>, Buffer<T>) {
    let mut ends = BufferMut::empty();
    let mut values = BufferMut::empty();

    if elements.is_empty() {
        return (ends.freeze(), values.freeze());
    }

    // Run-end encode the values
    let mut prev = elements[0];
    let mut end = 1;
    for &e in elements.iter().skip(1) {
        if e != prev {
            ends.push(end);
            values.push(prev);
        }
        prev = e;
        end += 1;
    }
    ends.push(end);
    values.push(prev);

    (ends.freeze(), values.freeze())
}

fn runend_encode_nullable_primitive<T: NativePType>(
    elements: &[T],
    element_validity: BitBuffer,
) -> (Buffer<u64>, PrimitiveArray) {
    let mut ends = BufferMut::empty();
    let mut values = BufferMut::empty();
    let mut validity = BitBufferMut::with_capacity(values.capacity());

    if elements.is_empty() {
        return (
            ends.freeze(),
            PrimitiveArray::new(
                values,
                Validity::Array(BoolArray::from(validity.freeze()).into_array()),
            ),
        );
    }

    // Run-end encode the values
    let mut prev = element_validity.value(0).then(|| elements[0]);
    let mut end = 1;
    for e in elements
        .iter()
        .zip(element_validity.iter())
        .map(|(&e, is_valid)| is_valid.then_some(e))
        .skip(1)
    {
        if e != prev {
            ends.push(end);
            match prev {
                None => {
                    validity.append(false);
                    values.push(T::default());
                }
                Some(p) => {
                    validity.append(true);
                    values.push(p);
                }
            }
        }
        prev = e;
        end += 1;
    }
    ends.push(end);

    match prev {
        None => {
            validity.append(false);
            values.push(T::default());
        }
        Some(p) => {
            validity.append(true);
            values.push(p);
        }
    }

    (
        ends.freeze(),
        PrimitiveArray::new(values, Validity::from(validity.freeze())),
    )
}

pub fn runend_decode_primitive(
    ends: PrimitiveArray,
    values: PrimitiveArray,
    offset: usize,
    length: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let validity_mask = values
        .as_ref()
        .validity()?
        .execute_mask(values.as_ref().len(), ctx)?;
    Ok(match_each_native_ptype!(values.ptype(), |P| {
        match_each_unsigned_integer_ptype!(ends.ptype(), |E| {
            runend_decode_typed_primitive(
                ends.as_slice::<E>(),
                offset,
                values.as_slice::<P>(),
                validity_mask,
                values.dtype().nullability(),
                length,
            )
        })
    }))
}

/// Number of bytes written by each unconditional splat store in the decode loop.
///
/// Each run is expanded chunk-at-a-time: short runs complete in a single store into the
/// buffer's padding, so they never pay for a per-element tail loop.
const DECODE_CHUNK_BYTES: usize = 32;

/// Number of elements of `T` written per splat store (at least one).
const fn decode_chunk_len<T>() -> usize {
    let n = DECODE_CHUNK_BYTES / size_of::<T>();
    if n == 0 { 1 } else { n }
}

/// Trim a raw run end down by `offset` and clamp it to `length`, panicking if the end
/// precedes the offset.
#[inline(always)]
fn trim_end<E: IntegerPType>(end: E, offset: E, length: E) -> usize {
    if end < offset {
        vortex_panic!("run end {end} must be >= offset {offset}");
    }
    min(end - offset, length).as_()
}

/// Runs at least this many bytes long are filled by [`fill_run`] instead of the inline
/// chunked stores. The out-of-line fill keeps its own clean codegen (a memset for bytes, an
/// aligned vector loop otherwise) regardless of register pressure in the calling decode
/// loop, and its call cost is amortized over the run.
const LONG_RUN_FILL_BYTES: usize = 256;

/// Fills of at least this many bytes grow by doubling `copy_nonoverlapping` instead of an
/// element loop.
///
/// The element loop is compiled against the baseline target features (16-byte SSE2 stores),
/// whereas `memcpy` is resolved by the libc at runtime to the widest implementation the host
/// supports (AVX2, or `rep movsb` on ERMS parts). Doubling therefore reaches a store width the
/// compiled loop cannot, but each `memcpy` costs a call, so it only pays once the run is long
/// enough to amortize it. Measured crossover is ~2 KiB across u16/u32/u64; below it doubling
/// is up to 2x slower, at and above it is 1.1-1.4x faster. See `run_end_decode_distribution`.
const DOUBLING_FILL_BYTES: usize = 2048;

/// Fill `dst[..len]` with `value`, used for long runs.
///
/// Deliberately not inlined: inlining a fill loop into the decode loop leaves its codegen at
/// the mercy of the surrounding register pressure — inside the nullable decode arm the inline
/// loop otherwise loses its wide vector stores. Byte-sized elements fill via `memset`, which
/// the libc already dispatches to the best available implementation at any length.
///
/// # Safety
///
/// `dst` must be valid for writes of `len` elements.
#[inline(never)]
unsafe fn fill_run<T: Copy>(dst: *mut MaybeUninit<T>, len: usize, value: T) {
    unsafe {
        if size_of::<T>() == 1 {
            let byte: u8 = std::mem::transmute_copy(&value);
            dst.cast::<u8>().write_bytes(byte, len);
            return;
        }
        if len * size_of::<T>() >= DOUBLING_FILL_BYTES {
            // Seed one cache line by hand, then repeatedly copy the filled prefix onto the
            // tail. `seed <= len` holds because `seed * size_of::<T>()` is at most 64 while
            // `len * size_of::<T>()` is at least `DOUBLING_FILL_BYTES`.
            let seed = (64 / size_of::<T>()).max(1);
            for i in 0..seed {
                dst.add(i).write(MaybeUninit::new(value));
            }
            let mut filled = seed;
            while filled < len {
                // `n <= filled` keeps the source `dst[..n]` disjoint from the destination
                // `dst[filled..filled + n]`, and `filled + n <= len` stays in bounds.
                let n = filled.min(len - filled);
                std::ptr::copy_nonoverlapping(dst, dst.add(filled), n);
                filled += n;
            }
            return;
        }
        for i in 0..len {
            dst.add(i).write(MaybeUninit::new(value));
        }
    }
}

/// Splat `value` into `base[pos..end]`.
///
/// Short runs use unconditional chunk-wide stores: they always write at least one full chunk
/// (rounding the run up to a multiple of the chunk length), so the overshoot past `end` is
/// written and later either overwritten by the next run or discarded by the final `set_len`.
/// Long runs dispatch to [`fill_run`] and write exactly `end - pos` elements.
///
/// The unconditional first chunk keeps byte-element codegen fast without a special case: a run
/// no longer than one chunk finishes in the inline, fixed-length vector stores below and never
/// reaches the trailing loop that LLVM's loop-idiom pass would collapse into a per-run
/// `memset` call. (An explicit replicated-word store for bytes was measurably slower here.)
///
/// # Safety
///
/// The allocation behind `base` must have room for at least
/// `max(pos, end) + decode_chunk_len::<T>()` elements.
#[inline(always)]
unsafe fn splat_run<T: Copy>(base: *mut MaybeUninit<T>, pos: usize, end: usize, value: T) {
    let chunk = const { decode_chunk_len::<T>() };
    // SAFETY: the caller guarantees the allocation extends one chunk past max(pos, end),
    // so every store below lands inside it.
    unsafe {
        let mut p = base.add(pos);
        let stop = base.add(end);
        // The first chunk is unconditional: runs up to one chunk finish on one branch.
        for i in 0..chunk {
            p.add(i).write(MaybeUninit::new(value));
        }
        p = p.add(chunk);
        if p >= stop {
            return;
        }
        // Only multi-chunk runs pay for the long-run check; the out-of-line fill keeps wide
        // vector stores regardless of register pressure in the calling loop and uses memset
        // for byte fills.
        let len = end - pos;
        if len * size_of::<T>() >= LONG_RUN_FILL_BYTES {
            // Refilling from `pos` overlaps the chunk written above, which is harmless.
            fill_run(base.add(pos), len, value);
            return;
        }
        loop {
            for i in 0..chunk {
                p.add(i).write(MaybeUninit::new(value));
            }
            p = p.add(chunk);
            if p >= stop {
                break;
            }
        }
    }
}

/// Decode a run-end encoded slice of values into a flat `Buffer<T>` and `Validity`.
///
/// This is the core decode loop shared by primitive and varbinview run-end decoding. Run
/// ends are adjusted by `offset` and clamped to `length` while decoding.
fn runend_decode_slice<E: IntegerPType, T: Copy + Default>(
    run_ends: &[E],
    offset: usize,
    values: &[T],
    values_validity: Mask,
    values_nullability: Nullability,
    length: usize,
) -> (Buffer<T>, Validity) {
    assert_eq!(
        run_ends.len(),
        values.len(),
        "runend ends and values must have equal lengths"
    );
    let offset_e = E::from_usize(offset).unwrap_or_else(|| {
        vortex_panic!(
            "offset {} cannot be converted to {}",
            offset,
            std::any::type_name::<E>()
        )
    });
    let length_e = E::from_usize(length).unwrap_or_else(|| {
        vortex_panic!(
            "length {} cannot be converted to {}",
            length,
            std::any::type_name::<E>()
        )
    });

    match values_validity {
        Mask::AllTrue(_) => {
            let mut decoded = BufferMut::<T>::with_capacity(length + decode_chunk_len::<T>());
            let base = decoded.spare_capacity_mut().as_mut_ptr();
            let mut pos = 0usize;
            for (&end, &value) in run_ends.iter().zip(values) {
                let end = trim_end(end, offset_e, length_e);
                assert!(
                    end >= pos,
                    "Runend ends must be monotonic, got {end} after {pos}"
                );
                // SAFETY: pos <= end <= length and the buffer was allocated with one chunk
                // of padding beyond `length`.
                unsafe { splat_run(base, pos, end, value) };
                pos = end;
            }
            // SAFETY: every element in 0..pos was initialized by the loop above.
            unsafe { decoded.set_len(pos) };
            (decoded.into(), values_nullability.into())
        }
        Mask::AllFalse(_) => (Buffer::<T>::zeroed(length), Validity::AllInvalid),
        Mask::Values(mask) => {
            let run_validity = mask.bit_buffer();
            assert_eq!(
                run_ends.len(),
                run_validity.len(),
                "runend ends and validity must have equal lengths"
            );
            // Prefill the element validity with the majority run validity; only minority
            // runs then need their bit range rewritten.
            let prefill = 2 * run_validity.true_count() > run_validity.len();
            let mut decoded_validity = BitBufferMut::full(prefill, length);
            let mut decoded = BufferMut::<T>::with_capacity(length + decode_chunk_len::<T>());
            let base = decoded.spare_capacity_mut().as_mut_ptr();
            let mut pos = 0usize;
            for ((&end, &value), is_valid) in run_ends.iter().zip(values).zip(run_validity.iter()) {
                let end = trim_end(end, offset_e, length_e);
                assert!(
                    end >= pos,
                    "Runend ends must be monotonic, got {end} after {pos}"
                );
                if is_valid != prefill && end > pos {
                    // SAFETY: pos <= end <= length == decoded_validity.len()
                    unsafe { decoded_validity.fill_range_unchecked(pos, end, is_valid) };
                }
                let value = if is_valid { value } else { T::default() };
                // SAFETY: pos <= end <= length and the buffer was allocated with one chunk
                // of padding beyond `length`.
                unsafe { splat_run(base, pos, end, value) };
                pos = end;
            }
            // SAFETY: every element in 0..pos was initialized by the loop above.
            unsafe { decoded.set_len(pos) };
            decoded_validity.truncate(pos);
            (decoded.into(), Validity::from(decoded_validity.freeze()))
        }
    }
}

/// Decode run-end encoded `values` with the given `run_ends` into a flat [`PrimitiveArray`].
///
/// Run ends are adjusted by `offset` and clamped to `length` while decoding.
pub fn runend_decode_typed_primitive<E: IntegerPType, T: NativePType>(
    run_ends: &[E],
    offset: usize,
    values: &[T],
    values_validity: Mask,
    values_nullability: Nullability,
    length: usize,
) -> PrimitiveArray {
    let (decoded, validity) = runend_decode_slice(
        run_ends,
        offset,
        values,
        values_validity,
        values_nullability,
        length,
    );
    PrimitiveArray::new(decoded, validity)
}

/// Decode a run-end encoded VarBinView array by expanding views directly.
pub fn runend_decode_varbinview(
    ends: PrimitiveArray,
    values: VarBinViewArray,
    offset: usize,
    length: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<VarBinViewArray> {
    let validity_mask = values
        .as_ref()
        .validity()?
        .execute_mask(values.as_ref().len(), ctx)?;
    let views = values.views();

    let (decoded_views, validity) = match_each_unsigned_integer_ptype!(ends.ptype(), |E| {
        runend_decode_slice(
            ends.as_slice::<E>(),
            offset,
            views,
            validity_mask,
            values.dtype().nullability(),
            length,
        )
    });

    let parts = values.into_data_parts();
    let view_handle = BufferHandle::new_host(decoded_views.into_byte_buffer());

    // SAFETY: we are expanding views from a valid VarBinViewArray with the same
    // buffers, so all buffer indices and offsets remain valid.
    Ok(unsafe {
        VarBinViewArray::new_handle_unchecked(view_handle, parts.buffers, parts.dtype, validity)
    })
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::NativePType;
    use vortex_array::validity::Validity;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::compress::runend_decode_primitive;
    use crate::compress::runend_encode;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    #[test]
    fn encode() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let arr = PrimitiveArray::from_iter([1i32, 1, 2, 2, 2, 3, 3, 3, 3, 3]);
        let (ends, values) = runend_encode(arr.as_view(), &mut ctx);
        let values = values.execute::<PrimitiveArray>(&mut ctx)?;

        let expected_ends = PrimitiveArray::from_iter(vec![2u8, 5, 10]);
        assert_arrays_eq!(ends, expected_ends, &mut ctx);
        let expected_values = PrimitiveArray::from_iter(vec![1i32, 2, 3]);
        assert_arrays_eq!(values, expected_values, &mut ctx);
        Ok(())
    }

    #[test]
    fn encode_nullable() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let arr = PrimitiveArray::new(
            buffer![1i32, 1, 2, 2, 2, 3, 3, 3, 3, 3],
            Validity::from(BitBuffer::from(vec![
                true, true, false, false, true, true, true, true, false, false,
            ])),
        );
        let (ends, values) = runend_encode(arr.as_view(), &mut ctx);
        let values = values.execute::<PrimitiveArray>(&mut ctx)?;

        let expected_ends = PrimitiveArray::from_iter(vec![2u8, 4, 5, 8, 10]);
        assert_arrays_eq!(ends, expected_ends, &mut ctx);
        let expected_values =
            PrimitiveArray::from_option_iter(vec![Some(1i32), None, Some(2), Some(3), None]);
        assert_arrays_eq!(values, expected_values, &mut ctx);
        Ok(())
    }

    #[test]
    fn encode_all_null() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let arr = PrimitiveArray::new(
            buffer![0, 0, 0, 0, 0],
            Validity::from(BitBuffer::new_unset(5)),
        );
        let (ends, values) = runend_encode(arr.as_view(), &mut ctx);
        let values = values.execute::<PrimitiveArray>(&mut ctx)?;

        let expected_ends = PrimitiveArray::from_iter(vec![5u64]);
        assert_arrays_eq!(ends, expected_ends, &mut ctx);
        let expected_values = PrimitiveArray::from_option_iter(vec![Option::<i32>::None]);
        assert_arrays_eq!(values, expected_values, &mut ctx);
        Ok(())
    }

    #[test]
    fn decode() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let ends = PrimitiveArray::from_iter([2u32, 5, 10]);
        let values = PrimitiveArray::from_iter([1i32, 2, 3]);
        let decoded = runend_decode_primitive(ends, values, 0, 10, &mut ctx)?;

        let expected = PrimitiveArray::from_iter(vec![1i32, 1, 2, 2, 2, 3, 3, 3, 3, 3]);
        assert_arrays_eq!(decoded, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn decode_with_offset() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let ends = PrimitiveArray::from_iter([2u32, 5, 10]);
        let values = PrimitiveArray::from_iter([1i32, 2, 3]);
        let decoded = runend_decode_primitive(ends, values, 2, 6, &mut ctx)?;

        let expected = PrimitiveArray::from_iter(vec![2i32, 2, 2, 3, 3, 3]);
        assert_arrays_eq!(decoded, expected, &mut ctx);
        Ok(())
    }

    #[rstest::rstest]
    #[case::mostly_valid(vec![Some(1i32), None, Some(3)])]
    #[case::mostly_null(vec![None, Some(2i32), None])]
    fn decode_nullable(#[case] run_values: Vec<Option<i32>>) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let ends = PrimitiveArray::from_iter([2u32, 5, 10]);
        let values = PrimitiveArray::from_option_iter(run_values.clone());
        let decoded = runend_decode_primitive(ends, values, 0, 10, &mut ctx)?;

        let mut expanded = Vec::new();
        let mut prev = 0usize;
        for (&end, value) in [2usize, 5, 10].iter().zip(run_values) {
            expanded.extend(std::iter::repeat_n(value, end - prev));
            prev = end;
        }
        let expected = PrimitiveArray::from_option_iter(expanded);
        assert_arrays_eq!(decoded, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn decode_long_runs() -> VortexResult<()> {
        // Runs longer than the splat chunk exercise the chunked-store loop's iteration path.
        let mut ctx = SESSION.create_execution_ctx();
        let ends = PrimitiveArray::from_iter([100u32, 101, 300]);
        let values = PrimitiveArray::from_iter([1i64, 2, 3]);
        let decoded = runend_decode_primitive(ends, values, 0, 300, &mut ctx)?;

        let mut expanded = vec![1i64; 100];
        expanded.push(2);
        expanded.extend(std::iter::repeat_n(3i64, 199));
        let expected = PrimitiveArray::from_iter(expanded);
        assert_arrays_eq!(decoded, expected, &mut ctx);
        Ok(())
    }

    /// Decode random runs with `runend_decode_primitive` and compare against a naive
    /// element-by-element reference expansion.
    ///
    /// Random run lengths straddle every length-dependent branch in the kernel: the inline
    /// chunk stores, the out-of-line long-run fill, and the doubling `memcpy` fill (2 KiB,
    /// which even the widest tested element type only crosses past ~1024-element runs). Also
    /// covers zero-length runs, offsets that partially trim the first run, and a final run
    /// overshooting the logical length; validity covers non-nullable, mixed, and all-invalid.
    fn check_decode_matches_reference<T: NativePType + From<u8>>(seed: u64) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut rng = StdRng::seed_from_u64(seed);

        for &max_run_len in &[3usize, 40, 700, 3000] {
            for nullable in [false, true] {
                let length = rng.random_range(1..=5000);
                let offset = if rng.random_bool(0.5) {
                    rng.random_range(0..50)
                } else {
                    0
                };
                let total = offset + length;

                let mut ends = Vec::new();
                let mut run_values: Vec<T> = Vec::new();
                let mut run_valid = Vec::new();
                let mut pos = 0usize;
                while pos < total {
                    let run_len = if !ends.is_empty() && rng.random_bool(0.05) {
                        // Zero-length runs exercise the empty-run edge of the splat loop.
                        0
                    } else if ends.is_empty() {
                        // The first run must reach past the offset.
                        rng.random_range(offset + 1..=offset + max_run_len)
                    } else {
                        rng.random_range(1..=max_run_len)
                    };
                    pos += run_len;
                    ends.push(u32::try_from(pos).vortex_expect("test lengths fit in u32"));
                    run_values.push(<T as From<u8>>::from(rng.random::<u8>()));
                    run_valid.push(rng.random_bool(0.9));
                }

                let all_invalid = nullable && rng.random_bool(0.1);
                let validity = if all_invalid {
                    Validity::AllInvalid
                } else if nullable {
                    Validity::from(BitBuffer::from(run_valid.clone()))
                } else {
                    Validity::NonNullable
                };

                let decoded = runend_decode_primitive(
                    PrimitiveArray::from_iter(ends.clone()),
                    PrimitiveArray::new(Buffer::from(run_values.clone()), validity),
                    offset,
                    length,
                    &mut ctx,
                )?;

                let mut expected: Vec<Option<T>> = Vec::with_capacity(length);
                for (&end, (value, valid)) in
                    ends.iter().zip(run_values.iter().zip(run_valid.iter()))
                {
                    let end = (end as usize - offset).min(length);
                    let value = match (nullable, all_invalid, valid) {
                        (false, ..) => Some(*value),
                        (true, true, _) => None,
                        (true, false, valid) => valid.then_some(*value),
                    };
                    expected.resize(end, value);
                }

                if nullable {
                    let expected = PrimitiveArray::from_option_iter(expected);
                    assert_arrays_eq!(decoded, expected, &mut ctx);
                } else {
                    let expected = PrimitiveArray::from_iter(
                        expected.into_iter().map(|v| v.vortex_expect("non-null")),
                    );
                    assert_arrays_eq!(decoded, expected, &mut ctx);
                }
            }
        }
        Ok(())
    }

    #[rstest::rstest]
    #[case(0)]
    #[case(1)]
    #[case(0x5eed)]
    fn decode_matches_reference(#[case] seed: u64) -> VortexResult<()> {
        check_decode_matches_reference::<u8>(seed)?;
        check_decode_matches_reference::<u16>(seed)?;
        check_decode_matches_reference::<u32>(seed)?;
        check_decode_matches_reference::<u64>(seed)?;
        check_decode_matches_reference::<f32>(seed)?;
        Ok(())
    }
}
