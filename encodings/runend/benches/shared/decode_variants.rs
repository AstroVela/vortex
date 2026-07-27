// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared run-end decode kernel variants for the ablation benchmarks.
//!
//! Each intermediate stage of the decode optimization lives here exactly once so that
//! `run_end_decode_ablation` and `run_end_decode_distribution` measure byte-identical code.
//! An ablation only proves a change if every benchmark that names a stage runs the same
//! implementation of it.
//!
//! Stages (non-nullable):
//! - [`decode_v0`]: original iterator chain (`trimmed_ends_iter` + `zip_eq`) + `push_n_unchecked`
//! - [`decode_v1`]: slice iteration + inline trim, still `push_n_unchecked` per run
//! - [`decode_v2`]: `decode_v1` + unconditional chunked element splat stores
//! - shipped: `runend_decode_typed_primitive` (adds the byte word/memset + long-run fill paths)
//!
//! Stages (nullable):
//! - `decode_v0` with a `Mask::Values`: Option-zip + `append_n` validity + `push_n_unchecked`
//! - [`decode_n2`]: slice loop + chunk stores, validity still built with per-run `append_n`
//! - shipped: `runend_decode_typed_primitive` (majority prefill + `fill_range_unchecked`)

// Included via `#[path]` into each bench binary, which uses only a subset of these stages.
#![allow(dead_code)]
// The `E::from_usize(length)` conversions cannot fail for benchmark lengths.
#![allow(clippy::unwrap_used)]

use std::cmp::min;
use std::mem::MaybeUninit;

use itertools::Itertools;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_mask::Mask;

/// Build run-end test data with uniformly-random run lengths (average `(max_run_len + 1) / 2`),
/// random values, and random run validity at the requested fraction-valid density.
///
/// Randomized so the branch-heavy decode stages are measured against unpredictable inputs
/// rather than a periodic pattern the predictor can learn.
pub fn make_data<T: NativePType + From<u8>>(
    seed: u64,
    total_length: usize,
    max_run_len: usize,
    validity_density: f64,
) -> (Buffer<u32>, Buffer<T>, BitBuffer) {
    make_data_values(seed, total_length, max_run_len, validity_density, false)
}

/// As [`make_data`], but `zero_values` forces every run value to zero.
///
/// Zero is byte-uniform, which the decode kernel fills with `memset` instead of the general
/// path. Real run-end columns carry both kinds of value, so both are benchmarked.
pub fn make_data_values<T: NativePType + From<u8>>(
    seed: u64,
    total_length: usize,
    max_run_len: usize,
    validity_density: f64,
    zero_values: bool,
) -> (Buffer<u32>, Buffer<T>, BitBuffer) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut ends = BufferMut::<u32>::empty();
    let mut values = BufferMut::<T>::empty();
    let mut validity = Vec::new();
    let max_run_len = max_run_len.max(1);
    let mut pos = 0usize;
    while pos < total_length {
        let run_len = rng.random_range(1..=max_run_len).min(total_length - pos);
        pos += run_len;
        ends.push(pos as u32);
        let byte = if zero_values { 0 } else { rng.random::<u8>() };
        values.push(<T as From<u8>>::from(byte));
        validity.push(rng.random_bool(validity_density));
    }
    (ends.freeze(), values.freeze(), BitBuffer::from(validity))
}

/// Number of runs produced by [`make_data`] for the given parameters, for per-run normalization.
pub fn run_count(ends: &Buffer<u32>) -> usize {
    ends.len()
}

// ---- helpers shared by v1/v2/n2 (mirror the shipped private helpers) ----

pub const DECODE_CHUNK_BYTES: usize = 32;

pub const fn decode_chunk_len<T>() -> usize {
    let n = DECODE_CHUNK_BYTES / size_of::<T>();
    if n == 0 { 1 } else { n }
}

#[inline(always)]
fn trim_end<E: IntegerPType>(end: E, offset: E, length: E) -> usize {
    assert!(end >= offset, "run end must be >= offset");
    min(end - offset, length).as_()
}

/// The v2 element-splat store (the shipped wide-element path, applied to all widths).
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

// ---- REJECTED byte word-splat variant ----
//
// Reconstructs the byte special case that an earlier revision of the shipped kernel carried:
// for `u8`, splat a replicated `0x0101..` word (whose runtime multiply is opaque to LLVM's
// loop-idiom pass) instead of the generic element chunk stores. It existed to stop the byte
// path collapsing into a `memset` call per short run.
//
// It was removed because the unconditional-first-chunk restructuring already prevents that
// collapse for short runs, making the word splat redundant *and* measurably slower for u8
// (the `wrapping_mul` + unaligned `u64` stores lose to the generic 16-byte vector stores).
// `nonnull_v3` (shipped) vs `nonnull_v3_byte_splat` in `run_end_decode_distribution`
// reproduces that: for widths > 1 they are identical; for `u8` the shipped path wins.

const LONG_RUN_FILL_BYTES: usize = 256;

#[inline(never)]
unsafe fn fill_run<T: Copy>(dst: *mut MaybeUninit<T>, len: usize, value: T) {
    unsafe {
        if size_of::<T>() == 1 {
            let byte: u8 = std::mem::transmute_copy(&value);
            dst.cast::<u8>().write_bytes(byte, len);
        } else {
            for i in 0..len {
                dst.add(i).write(MaybeUninit::new(value));
            }
        }
    }
}

/// # Safety
///
/// The allocation behind `base` must have room for at least
/// `max(pos, end) + decode_chunk_len::<T>()` elements.
#[inline(always)]
unsafe fn splat_run_byte_splat<T: Copy>(
    base: *mut MaybeUninit<T>,
    pos: usize,
    end: usize,
    value: T,
) {
    if size_of::<T>() == 1 {
        // SAFETY: size_of::<T>() == 1, and the caller guarantees a chunk of slack.
        unsafe {
            let byte: u8 = std::mem::transmute_copy(&value);
            let word = u64::from(byte).wrapping_mul(0x0101_0101_0101_0101);
            let mut p = base.add(pos).cast::<u8>();
            let stop = base.add(end).cast::<u8>();
            for i in 0..DECODE_CHUNK_BYTES / 8 {
                p.add(i * 8).cast::<u64>().write_unaligned(word);
            }
            p = p.add(DECODE_CHUNK_BYTES);
            if p >= stop {
                return;
            }
            let len = end - pos;
            if len >= LONG_RUN_FILL_BYTES {
                base.add(pos).cast::<u8>().write_bytes(byte, len);
                return;
            }
            loop {
                for i in 0..DECODE_CHUNK_BYTES / 8 {
                    p.add(i * 8).cast::<u64>().write_unaligned(word);
                }
                p = p.add(DECODE_CHUNK_BYTES);
                if p >= stop {
                    break;
                }
            }
        }
    } else {
        let chunk = const { decode_chunk_len::<T>() };
        // SAFETY: caller guarantees one chunk of slack past max(pos, end).
        unsafe {
            let mut p = base.add(pos);
            let stop = base.add(end);
            for i in 0..chunk {
                p.add(i).write(MaybeUninit::new(value));
            }
            p = p.add(chunk);
            if p >= stop {
                return;
            }
            let len = end - pos;
            if len * size_of::<T>() >= LONG_RUN_FILL_BYTES {
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
}

// ---- PREVIOUS fill: doubling memcpy, but no byte-uniform `memset` fast path ----
//
// Reconstructs the kernel as it stood before `repeated_byte` was added: byte-sized elements
// fill with `memset`, every wider element takes the doubling/element path regardless of
// whether its bytes happen to be identical. `zeros_v3` vs `zeros_v3_prev` isolates the
// `memset` fast path on byte-uniform values (zero); `rand_v3_prev` vs `nonnull_v3` confirms
// arbitrary values are unaffected.

/// # Safety
///
/// `dst` must be valid for writes of `len` elements.
#[inline(never)]
unsafe fn fill_run_no_memset<T: Copy>(dst: *mut MaybeUninit<T>, len: usize, value: T) {
    unsafe {
        if size_of::<T>() == 1 {
            let byte: u8 = std::mem::transmute_copy(&value);
            dst.cast::<u8>().write_bytes(byte, len);
            return;
        }
        if len * size_of::<T>() >= DOUBLING_FILL_BYTES {
            let seed = (64 / size_of::<T>()).max(1);
            for i in 0..seed {
                dst.add(i).write(MaybeUninit::new(value));
            }
            let mut filled = seed;
            while filled < len {
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

/// # Safety
///
/// The allocation behind `base` must have room for at least
/// `max(pos, end) + decode_chunk_len::<T>()` elements.
#[inline(always)]
unsafe fn splat_run_no_memset<T: Copy>(
    base: *mut MaybeUninit<T>,
    pos: usize,
    end: usize,
    value: T,
) {
    let chunk = const { decode_chunk_len::<T>() };
    // SAFETY: caller guarantees one chunk of slack past max(pos, end).
    unsafe {
        let mut p = base.add(pos);
        let stop = base.add(end);
        for i in 0..chunk {
            p.add(i).write(MaybeUninit::new(value));
        }
        p = p.add(chunk);
        if p >= stop {
            return;
        }
        let len = end - pos;
        if len * size_of::<T>() >= LONG_RUN_FILL_BYTES {
            fill_run_no_memset(base.add(pos), len, value);
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

/// Non-nullable decode without the byte-uniform `memset` fast path (see above).
pub fn decode_v3_no_memset<E: IntegerPType, T: NativePType>(
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
        unsafe { splat_run_no_memset(base, pos, end, value) };
        pos = end;
    }
    // SAFETY: every element in 0..pos was initialized above.
    unsafe { decoded.set_len(pos) };
    (decoded.into(), Nullability::NonNullable.into())
}

/// The doubling threshold, mirrored from the shipped kernel.
const DOUBLING_FILL_BYTES: usize = 2048;

// ---- PREVIOUS long-run fill: element loop instead of doubling memcpy ----
//
// The shipped `fill_run` grows long fills by doubling `copy_nonoverlapping`, reaching the
// libc's runtime-dispatched (AVX2/ERMS) store width. This variant keeps the structure but
// fills with the plain element loop, which the compiler emits at baseline SSE2 width.
// `nonnull_v3` (shipped) vs `nonnull_v3_elem_fill` isolates that change: identical below the
// 2 KiB doubling threshold, shipped pulls ahead above it.

/// # Safety
///
/// `dst` must be valid for writes of `len` elements.
#[inline(never)]
unsafe fn fill_run_elems<T: Copy>(dst: *mut MaybeUninit<T>, len: usize, value: T) {
    unsafe {
        if size_of::<T>() == 1 {
            let byte: u8 = std::mem::transmute_copy(&value);
            dst.cast::<u8>().write_bytes(byte, len);
        } else {
            for i in 0..len {
                dst.add(i).write(MaybeUninit::new(value));
            }
        }
    }
}

/// # Safety
///
/// The allocation behind `base` must have room for at least
/// `max(pos, end) + decode_chunk_len::<T>()` elements.
#[inline(always)]
unsafe fn splat_run_elem_fill<T: Copy>(
    base: *mut MaybeUninit<T>,
    pos: usize,
    end: usize,
    value: T,
) {
    let chunk = const { decode_chunk_len::<T>() };
    // SAFETY: caller guarantees one chunk of slack past max(pos, end).
    unsafe {
        let mut p = base.add(pos);
        let stop = base.add(end);
        for i in 0..chunk {
            p.add(i).write(MaybeUninit::new(value));
        }
        p = p.add(chunk);
        if p >= stop {
            return;
        }
        let len = end - pos;
        if len * size_of::<T>() >= LONG_RUN_FILL_BYTES {
            fill_run_elems(base.add(pos), len, value);
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

/// Non-nullable decode whose long-run fill uses the element loop (see above).
pub fn decode_v3_elem_fill<E: IntegerPType, T: NativePType>(
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
        unsafe { splat_run_elem_fill(base, pos, end, value) };
        pos = end;
    }
    // SAFETY: every element in 0..pos was initialized above.
    unsafe { decoded.set_len(pos) };
    (decoded.into(), Nullability::NonNullable.into())
}

/// Non-nullable decode using the rejected byte word-splat kernel (see above).
pub fn decode_v3_byte_splat<E: IntegerPType, T: NativePType>(
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
        unsafe { splat_run_byte_splat(base, pos, end, value) };
        pos = end;
    }
    // SAFETY: every element in 0..pos was initialized above.
    unsafe { decoded.set_len(pos) };
    (decoded.into(), Nullability::NonNullable.into())
}

// ---- v0: original implementation (verbatim from the pre-change kernel) ----

pub fn decode_v0<T: NativePType>(
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

// ---- v1: slice loop, still push_n_unchecked ----

pub fn decode_v1<E: IntegerPType, T: NativePType>(
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

pub fn decode_v2<E: IntegerPType, T: NativePType>(
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

// ---- n2: nullable with chunk stores but validity still via per-run append_n ----

pub fn decode_n2<E: IntegerPType, T: NativePType>(
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

// ---- Bench-local copy of the SHIPPED kernel ----
//
// `nonnull_v3` calls the real kernel across a crate boundary, which costs enough to swamp the
// effect being measured (the u8 rows of that comparison differ by up to 1.8x on byte-identical
// algorithms). This copy compiles into the bench binary like every other variant here, so
// `nonnull_v3_local` versus `nonnull_v3_elem_fill` isolates the long-run fill strategy alone.
// Keep it in step with `compress.rs`.

const DOUBLING_FILL_BYTES_LOCAL: usize = 2048;

#[inline(never)]
unsafe fn fill_run_shipped<T: Copy>(dst: *mut MaybeUninit<T>, len: usize, value: T) {
    unsafe {
        if len * size_of::<T>() >= DOUBLING_FILL_BYTES_LOCAL {
            let seed = (64 / size_of::<T>()).max(1);
            for i in 0..seed {
                dst.add(i).write(MaybeUninit::new(value));
            }
            let mut filled = seed;
            while filled < len {
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

/// # Safety
///
/// The allocation behind `base` must have room for at least
/// `max(pos, end) + decode_chunk_len::<T>()` elements.
#[inline(always)]
unsafe fn splat_run_shipped<T: Copy>(base: *mut MaybeUninit<T>, pos: usize, end: usize, value: T) {
    let chunk = const { decode_chunk_len::<T>() };
    // SAFETY: caller guarantees one chunk of slack past max(pos, end).
    unsafe {
        let mut p = base.add(pos);
        let stop = base.add(end);
        for i in 0..chunk {
            p.add(i).write(MaybeUninit::new(value));
        }
        p = p.add(chunk);
        if p >= stop {
            return;
        }
        let len = end - pos;
        if len * size_of::<T>() >= LONG_RUN_FILL_BYTES {
            fill_run_shipped(base.add(pos), len, value);
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

/// Bench-local mirror of the shipped non-nullable decode (see above).
pub fn decode_v3_local<E: IntegerPType, T: NativePType>(
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
        unsafe { splat_run_shipped(base, pos, end, value) };
        pos = end;
    }
    // SAFETY: every element in 0..pos was initialized above.
    unsafe { decoded.set_len(pos) };
    (decoded.into(), Nullability::NonNullable.into())
}
