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

/// Replicate the bytes of `value` across a `u128` splat word.
///
/// Only used for little-endian targets with element sizes that divide 16; all such decode
/// element types (primitives and binary views) contain no padding, so every copied byte is
/// initialized.
#[inline(always)]
fn splat_word<T: Copy>(value: T) -> u128 {
    let size = size_of::<T>();
    debug_assert!(size <= 16 && 16 % size == 0);
    let mut word = 0u128;
    // SAFETY: at most 16 initialized bytes are copied into the zero-initialized word.
    unsafe {
        std::ptr::copy_nonoverlapping(
            (&raw const value).cast::<u8>(),
            (&raw mut word).cast::<u8>(),
            size,
        );
    }
    if size < 16 {
        // Multiplying by ((2^128 - 1) / (2^(8 * size) - 1)) replicates the low bytes across
        // the whole word. The runtime multiply also keeps LLVM's loop-idiom pass from
        // rewriting the store loop in `splat_run` into a per-run memset call, which is what
        // makes short byte-sized runs slow.
        word = word.wrapping_mul(u128::MAX / ((1u128 << (8 * size)) - 1));
    }
    word
}

/// Splat `value` into `base[pos..end]` using unconditional chunk-wide stores.
///
/// Always writes at least one full chunk (rounding the run up to a multiple of the chunk
/// length), so the overshoot past `end` is written and later either overwritten by the next
/// run or discarded by the final `set_len`.
///
/// # Safety
///
/// The allocation behind `base` must have room for at least
/// `max(pos, end) + decode_chunk_len::<T>()` elements.
#[inline(always)]
unsafe fn splat_run<T: Copy>(base: *mut MaybeUninit<T>, pos: usize, end: usize, value: T) {
    let size = size_of::<T>();
    if cfg!(target_endian = "little") && size.is_power_of_two() && size <= 16 {
        // Fast path: expand the value into a 16-byte word and store it with two unaligned
        // 16-byte writes per chunk, so even 8-byte elements get full-width vector stores.
        let word = splat_word(value);
        // SAFETY: the caller guarantees the allocation extends one chunk
        // (`decode_chunk_len::<T>() * size == DECODE_CHUNK_BYTES` bytes for these sizes)
        // past max(pos, end), so every store below lands inside it.
        unsafe {
            let mut p = base.add(pos).cast::<u8>();
            let stop = base.add(end).cast::<u8>();
            loop {
                p.cast::<u128>().write_unaligned(word);
                p.add(16).cast::<u128>().write_unaligned(word);
                p = p.add(DECODE_CHUNK_BYTES);
                if p >= stop {
                    break;
                }
            }
        }
    } else {
        let chunk = const { decode_chunk_len::<T>() };
        // SAFETY: the caller guarantees the allocation extends one chunk past max(pos, end),
        // so every store below lands inside it.
        unsafe {
            let mut p = base.add(pos);
            let stop = base.add(end);
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
                assert!(end >= pos, "Runend ends must be monotonic, got {end} after {pos}");
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
            for ((&end, &value), is_valid) in
                run_ends.iter().zip(values).zip(run_validity.iter())
            {
                let end = trim_end(end, offset_e, length_e);
                assert!(end >= pos, "Runend ends must be monotonic, got {end} after {pos}");
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

    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::buffer;
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
}
