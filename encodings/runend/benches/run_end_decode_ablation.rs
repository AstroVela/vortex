// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Temporary ablation benchmark: isolates each change to the run-end decode kernel so every
//! change is justified by its own numbers, all measured in one binary run over identical
//! randomized inputs.
//!
//! Variants (non-null):
//! - `v0_original`: iterator chain (`trimmed_ends_iter` + `zip_eq`) + `push_n_unchecked`
//! - `v1_slice_loop`: slice iteration + inline trim, still `push_n_unchecked` per run
//! - `v2_chunk_stores`: v1 + unconditional chunked element splat stores (all types)
//! - `v3_shipped`: the shipped kernel (adds the byte-element word/memset path)
//!
//! Variants (nullable):
//! - `n0_original`: Option-zip + `append_n` validity + `push_n_unchecked`
//! - `n2_chunk_stores`: slice loop + chunk stores, validity still via `append_n`
//! - `n3_shipped`: shipped kernel (majority prefill + `fill_range_unchecked` validity)

#![expect(clippy::cast_possible_truncation)]
#![expect(clippy::unwrap_used)]

use std::cmp::min;
use std::fmt;
use std::mem::MaybeUninit;

use divan::Bencher;
use itertools::Itertools;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_mask::Mask;
use vortex_runend::compress::runend_decode_typed_primitive;
use vortex_runend::trimmed_ends_iter;

fn main() {
    divan::main();
}

#[derive(Clone, Copy)]
struct Args {
    total_length: usize,
    avg_run_length: usize,
}

impl fmt::Display for Args {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}_{}", self.total_length, self.avg_run_length)
    }
}

const ARGS: &[Args] = &[
    Args {
        total_length: 65_536,
        avg_run_length: 2,
    },
    Args {
        total_length: 65_536,
        avg_run_length: 8,
    },
    Args {
        total_length: 65_536,
        avg_run_length: 64,
    },
    Args {
        total_length: 65_536,
        avg_run_length: 1024,
    },
];

fn make_data<T: NativePType + From<u8>>(args: Args) -> (Buffer<u32>, Buffer<T>, BitBuffer) {
    let mut rng = StdRng::seed_from_u64(0x5eed);
    let mut ends = BufferMut::<u32>::empty();
    let mut values = BufferMut::<T>::empty();
    let mut validity = Vec::new();
    let max_run_len = (2 * args.avg_run_length).saturating_sub(1).max(1);
    let mut pos = 0usize;
    while pos < args.total_length {
        let run_len = rng
            .random_range(1..=max_run_len)
            .min(args.total_length - pos);
        pos += run_len;
        ends.push(pos as u32);
        values.push(<T as From<u8>>::from(rng.random::<u8>()));
        validity.push(rng.random_bool(0.9));
    }
    (ends.freeze(), values.freeze(), BitBuffer::from(validity))
}

// ---- v0: original implementation (verbatim from the pre-change kernel) ----

fn decode_v0<T: NativePType>(
    run_ends: impl Iterator<Item = usize>,
    values: &[T],
    values_validity: Mask,
    length: usize,
) -> (Buffer<T>, Validity) {
    match values_validity {
        Mask::AllTrue(_) => {
            let mut decoded: BufferMut<T> = BufferMut::with_capacity(length);
            for (end, value) in run_ends.zip_eq(values) {
                assert!(
                    end >= decoded.len(),
                    "Runend ends must be monotonic, got {end} after {}",
                    decoded.len()
                );
                assert!(end <= length, "Runend end must be less than overall length");
                // SAFETY: we preallocate enough capacity because we know the total length
                unsafe { decoded.push_n_unchecked(*value, end - decoded.len()) };
            }
            (decoded.into(), Nullability::NonNullable.into())
        }
        Mask::AllFalse(_) => (Buffer::<T>::zeroed(length), Validity::AllInvalid),
        Mask::Values(mask) => {
            let mut decoded = BufferMut::with_capacity(length);
            let mut decoded_validity = BitBufferMut::with_capacity(length);
            for (end, value) in run_ends.zip_eq(
                values
                    .iter()
                    .zip(mask.bit_buffer().iter())
                    .map(|(&v, is_valid)| is_valid.then_some(v)),
            ) {
                assert!(
                    end >= decoded.len(),
                    "Runend ends must be monotonic, got {end} after {}",
                    decoded.len()
                );
                assert!(end <= length, "Runend end must be less than overall length");
                match value {
                    None => {
                        decoded_validity.append_n(false, end - decoded.len());
                        // SAFETY: we preallocate enough capacity
                        unsafe { decoded.push_n_unchecked(T::default(), end - decoded.len()) };
                    }
                    Some(value) => {
                        decoded_validity.append_n(true, end - decoded.len());
                        // SAFETY: we preallocate enough capacity
                        unsafe { decoded.push_n_unchecked(value, end - decoded.len()) };
                    }
                }
            }
            (decoded.into(), Validity::from(decoded_validity.freeze()))
        }
    }
}

// ---- shared helpers for v1/v2 (mirrors of the shipped private helpers) ----

const DECODE_CHUNK_BYTES: usize = 32;

const fn decode_chunk_len<T>() -> usize {
    let n = DECODE_CHUNK_BYTES / size_of::<T>();
    if n == 0 { 1 } else { n }
}

#[inline(always)]
fn trim_end<E: IntegerPType>(end: E, offset: E, length: E) -> usize {
    assert!(end >= offset, "run end must be >= offset");
    min(end - offset, length).as_()
}

/// The v2 element-splat store (the shipped wide-element path, applied to all types).
///
/// # Safety
///
/// The allocation behind `base` must have room for at least
/// `max(pos, end) + decode_chunk_len::<T>()` elements.
#[inline(always)]
unsafe fn splat_run_elements<T: Copy>(base: *mut MaybeUninit<T>, pos: usize, end: usize, value: T) {
    let chunk = const { decode_chunk_len::<T>() };
    // SAFETY: caller guarantees one chunk of slack past max(pos, end).
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

// ---- v1: slice loop, still push_n_unchecked ----

fn decode_v1<E: IntegerPType, T: NativePType>(
    run_ends: &[E],
    values: &[T],
    length: usize,
) -> (Buffer<T>, Validity) {
    let offset_e = E::zero();
    let length_e = E::from_usize(length).unwrap();
    let mut decoded: BufferMut<T> = BufferMut::with_capacity(length);
    for (&end, &value) in run_ends.iter().zip(values) {
        let end = trim_end(end, offset_e, length_e);
        assert!(
            end >= decoded.len(),
            "Runend ends must be monotonic, got {end} after {}",
            decoded.len()
        );
        // SAFETY: we preallocate enough capacity because we know the total length
        unsafe { decoded.push_n_unchecked(value, end - decoded.len()) };
    }
    (decoded.into(), Nullability::NonNullable.into())
}

// ---- v2: slice loop + chunked element splat stores ----

fn decode_v2<E: IntegerPType, T: NativePType>(
    run_ends: &[E],
    values: &[T],
    length: usize,
) -> (Buffer<T>, Validity) {
    let offset_e = E::zero();
    let length_e = E::from_usize(length).unwrap();
    let mut decoded = BufferMut::<T>::with_capacity(length + decode_chunk_len::<T>());
    let base = decoded.spare_capacity_mut().as_mut_ptr();
    let mut pos = 0usize;
    for (&end, &value) in run_ends.iter().zip(values) {
        let end = trim_end(end, offset_e, length_e);
        assert!(
            end >= pos,
            "Runend ends must be monotonic, got {end} after {pos}"
        );
        // SAFETY: pos <= end <= length and the buffer has one chunk of padding.
        unsafe { splat_run_elements(base, pos, end, value) };
        pos = end;
    }
    // SAFETY: every element in 0..pos was initialized above.
    unsafe { decoded.set_len(pos) };
    (decoded.into(), Nullability::NonNullable.into())
}

// ---- n2: nullable with chunk stores but validity still via append_n ----

fn decode_n2<E: IntegerPType, T: NativePType>(
    run_ends: &[E],
    values: &[T],
    run_validity: &BitBuffer,
    length: usize,
) -> (Buffer<T>, Validity) {
    let offset_e = E::zero();
    let length_e = E::from_usize(length).unwrap();
    let mut decoded = BufferMut::<T>::with_capacity(length + decode_chunk_len::<T>());
    let mut decoded_validity = BitBufferMut::with_capacity(length);
    let base = decoded.spare_capacity_mut().as_mut_ptr();
    let mut pos = 0usize;
    for ((&end, &value), is_valid) in run_ends.iter().zip(values).zip(run_validity.iter()) {
        let end = trim_end(end, offset_e, length_e);
        assert!(
            end >= pos,
            "Runend ends must be monotonic, got {end} after {pos}"
        );
        decoded_validity.append_n(is_valid, end - pos);
        let value = if is_valid { value } else { T::default() };
        // SAFETY: pos <= end <= length and the buffer has one chunk of padding.
        unsafe { splat_run_elements(base, pos, end, value) };
        pos = end;
    }
    // SAFETY: every element in 0..pos was initialized above.
    unsafe { decoded.set_len(pos) };
    (decoded.into(), Validity::from(decoded_validity.freeze()))
}

// ---- benches ----

#[divan::bench(types = [u8, u32, u64], args = ARGS)]
fn v0_original<T: NativePType + From<u8>>(bencher: Bencher, args: Args) {
    let (ends, values, _) = make_data::<T>(args);
    bencher
        .with_inputs(|| (ends.clone(), values.clone()))
        .bench_refs(|(ends, values)| {
            let (buf, validity) = decode_v0(
                trimmed_ends_iter(ends.as_slice(), 0, args.total_length),
                values.as_slice(),
                Mask::new_true(values.len()),
                args.total_length,
            );
            PrimitiveArray::new(buf, validity)
        });
}

#[divan::bench(types = [u8, u32, u64], args = ARGS)]
fn v1_slice_loop<T: NativePType + From<u8>>(bencher: Bencher, args: Args) {
    let (ends, values, _) = make_data::<T>(args);
    bencher
        .with_inputs(|| (ends.clone(), values.clone()))
        .bench_refs(|(ends, values)| {
            let (buf, validity) = decode_v1(ends.as_slice(), values.as_slice(), args.total_length);
            PrimitiveArray::new(buf, validity)
        });
}

#[divan::bench(types = [u8, u32, u64], args = ARGS)]
fn v2_chunk_stores<T: NativePType + From<u8>>(bencher: Bencher, args: Args) {
    let (ends, values, _) = make_data::<T>(args);
    bencher
        .with_inputs(|| (ends.clone(), values.clone()))
        .bench_refs(|(ends, values)| {
            let (buf, validity) = decode_v2(ends.as_slice(), values.as_slice(), args.total_length);
            PrimitiveArray::new(buf, validity)
        });
}

#[divan::bench(types = [u8, u32, u64], args = ARGS)]
fn v3_shipped<T: NativePType + From<u8>>(bencher: Bencher, args: Args) {
    let (ends, values, _) = make_data::<T>(args);
    bencher
        .with_inputs(|| (ends.clone(), values.clone()))
        .bench_refs(|(ends, values)| {
            runend_decode_typed_primitive(
                ends.as_slice(),
                0,
                values.as_slice(),
                Mask::new_true(values.len()),
                Nullability::NonNullable,
                args.total_length,
            )
        });
}

#[divan::bench(types = [u8, u32, u64], args = ARGS)]
fn n0_original<T: NativePType + From<u8>>(bencher: Bencher, args: Args) {
    let (ends, values, run_validity) = make_data::<T>(args);
    bencher
        .with_inputs(|| (ends.clone(), values.clone(), run_validity.clone()))
        .bench_refs(|(ends, values, run_validity)| {
            let (buf, validity) = decode_v0(
                trimmed_ends_iter(ends.as_slice(), 0, args.total_length),
                values.as_slice(),
                Mask::from_buffer(run_validity.clone()),
                args.total_length,
            );
            PrimitiveArray::new(buf, validity)
        });
}

#[divan::bench(types = [u8, u32, u64], args = ARGS)]
fn n2_chunk_stores<T: NativePType + From<u8>>(bencher: Bencher, args: Args) {
    let (ends, values, run_validity) = make_data::<T>(args);
    bencher
        .with_inputs(|| (ends.clone(), values.clone(), run_validity.clone()))
        .bench_refs(|(ends, values, run_validity)| {
            let (buf, validity) = decode_n2(
                ends.as_slice(),
                values.as_slice(),
                run_validity,
                args.total_length,
            );
            PrimitiveArray::new(buf, validity)
        });
}

#[divan::bench(types = [u8, u32, u64], args = ARGS)]
fn n3_shipped<T: NativePType + From<u8>>(bencher: Bencher, args: Args) {
    let (ends, values, run_validity) = make_data::<T>(args);
    bencher
        .with_inputs(|| (ends.clone(), values.clone(), run_validity.clone()))
        .bench_refs(|(ends, values, run_validity)| {
            runend_decode_typed_primitive(
                ends.as_slice(),
                0,
                values.as_slice(),
                Mask::from_buffer(run_validity.clone()),
                Nullability::Nullable,
                args.total_length,
            )
        });
}
