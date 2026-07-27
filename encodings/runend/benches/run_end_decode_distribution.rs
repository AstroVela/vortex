// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Distribution-sweep ablation: attributes each run-end decode change to the data
//! distribution it is sensitive to. Every stage kernel is shared with
//! `run_end_decode_ablation` via `shared/decode_variants.rs`, so the two benchmarks measure
//! byte-identical code.
//!
//! Throughput is reported in decoded elements/sec (`ItemsCount`), which normalizes across run
//! lengths: a per-run fixed cost shows up as throughput that climbs with run length, while a
//! per-element cost shows up as throughput flat across run length.
//!
//! Groups:
//! - `nonnull_{v0,v1,v2,v3}` — run-length sweep. Isolates the non-null structural wins
//!   (v0->v1 iterator/bookkeeping, v1->v2 scalar-fill -> chunk stores) against run length and
//!   element width.
//! - `nonnull_v3_{elem_fill,byte_splat}` — alternative fill strategies for the shipped kernel.
//! - `zeros_v3{,_prev}` / `rand_v3_prev` — byte-uniform versus arbitrary values, isolating the
//!   `memset` fast path.
//! - `nullable_{n0,n2,n3}` — run-length sweep at 90% valid. Isolates the nullable structural
//!   wins and the majority-prefill validity (n2->n3), including the long-run fill dispatch.
//! - `density_{n0,n2,n3}` — validity-density sweep, u32 at run length 8. Isolates the
//!   prefill validity win (n2->n3) against validity skew: prefill rewrites only the minority
//!   runs, so its advantage grows as validity moves away from 50/50.
//!
//! # Reading these numbers
//!
//! `nonnull_v3` and `zeros_v3` call the real `runend_decode_typed_primitive` across a crate
//! boundary; every other variant is a copy compiled into this binary and inlines more
//! aggressively. That difference is worth up to ~40% at short runs *on identical algorithms*,
//! and it consistently favours the bench-local copies. So:
//!
//! - Comparing a bench-local variant against another bench-local variant is apples to apples.
//! - Comparing `nonnull_v3`/`zeros_v3` against a bench-local variant understates the shipped
//!   kernel. A win measured that way is a lower bound; a small loss may be the boundary alone.
//! - To isolate a change cleanly, prefer a *within-function* contrast (the same bench over two
//!   data distributions, as `zeros_v3` versus `nonnull_v3`), which cancels the effect.

#![expect(clippy::cast_possible_truncation)]

use std::fmt;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use divan::Bencher;
use divan::counter::ItemsCount;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_mask::Mask;
use vortex_runend::compress::runend_decode_typed_primitive;
use vortex_runend::trimmed_ends_iter;

#[path = "shared/decode_variants.rs"]
mod decode_variants;

use decode_variants::decode_n2;
use decode_variants::decode_v0;
use decode_variants::decode_v1;
use decode_variants::decode_v2;
use decode_variants::decode_v3_byte_splat;
use decode_variants::decode_v3_elem_fill;
use decode_variants::decode_v3_local;
use decode_variants::decode_v3_no_memset;
use decode_variants::make_data_values;

fn main() {
    divan::main();
}

const SEED: u64 = 0x5eed;
const TOTAL_LENGTH: usize = 65_536;

/// Average run length. Run lengths are drawn uniformly from `1..=(2*avg-1)`, so every point
/// mixes short and long runs rather than repeating one length; `avg` is only the mean.
///
/// The range brackets the run lengths run-end encoding is actually chosen for, and straddles
/// the 2 KiB doubling-fill threshold from both sides for each element width (u32 crosses at an
/// average of 512, u64 at 256).
const RUN_LENGTHS: &[usize] = &[32, 64, 128, 256, 512];

/// Fixed short run length for the density sweep, where per-run validity work is visible.
const DENSITY_RUN_LENGTH: usize = 8;

fn max_run_len(avg: usize) -> usize {
    2 * avg - 1
}

/// Number of independently-seeded datasets cycled across benchmark iterations.
///
/// Re-using one dataset replays an identical run-length sequence every iteration, so the branch
/// predictor learns which runs take each length-dependent branch and the benchmark reports a
/// predictor state no real decode reaches. Cycling datasets multiplies the pattern length past
/// what the predictor retains. `predictor_{fixed,rotating}` measures what this is worth; every
/// other benchmark here rotates.
const DATASETS: u64 = 16;

/// Input provider cycling `DATASETS` independently-seeded datasets.
fn rotating<T: NativePType + From<u8>>(
    avg: usize,
    density: f64,
    zero_values: bool,
) -> impl Fn() -> (Buffer<u32>, Buffer<T>, BitBuffer) {
    let sets: Vec<_> = (0..DATASETS)
        .map(|k| {
            make_data_values::<T>(
                SEED.wrapping_add(k.wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                TOTAL_LENGTH,
                max_run_len(avg),
                density,
                zero_values,
            )
        })
        .collect();
    let next = AtomicUsize::new(0);
    move || {
        let i = next.fetch_add(1, Ordering::Relaxed);
        sets[i % sets.len()].clone()
    }
}

/// Input provider replaying a single dataset, for the `predictor_*` control only.
fn fixed<T: NativePType + From<u8>>(
    avg: usize,
    density: f64,
) -> impl Fn() -> (Buffer<u32>, Buffer<T>, BitBuffer) {
    let set = make_data_values::<T>(SEED, TOTAL_LENGTH, max_run_len(avg), density, false);
    move || set.clone()
}

// ---- Group A: non-nullable run-length sweep ----

#[divan::bench(types = [u8, u32, u64], args = RUN_LENGTHS)]
fn nonnull_v0<T: NativePType + From<u8>>(bencher: Bencher, avg: usize) {
    bencher
        .counter(ItemsCount::new(TOTAL_LENGTH))
        .with_inputs(rotating::<T>(avg, 1.0, false))
        .bench_refs(|(ends, values, _)| {
            let (buf, validity) = decode_v0(
                trimmed_ends_iter(ends.as_slice(), 0, TOTAL_LENGTH),
                values.as_slice(),
                Mask::new_true(values.len()),
                TOTAL_LENGTH,
            );
            PrimitiveArray::new(buf, validity)
        });
}

#[divan::bench(types = [u8, u32, u64], args = RUN_LENGTHS)]
fn nonnull_v1<T: NativePType + From<u8>>(bencher: Bencher, avg: usize) {
    bencher
        .counter(ItemsCount::new(TOTAL_LENGTH))
        .with_inputs(rotating::<T>(avg, 1.0, false))
        .bench_refs(|(ends, values, _)| {
            let (buf, validity) = decode_v1(ends.as_slice(), values.as_slice(), TOTAL_LENGTH);
            PrimitiveArray::new(buf, validity)
        });
}

#[divan::bench(types = [u8, u32, u64], args = RUN_LENGTHS)]
fn nonnull_v2<T: NativePType + From<u8>>(bencher: Bencher, avg: usize) {
    bencher
        .counter(ItemsCount::new(TOTAL_LENGTH))
        .with_inputs(rotating::<T>(avg, 1.0, false))
        .bench_refs(|(ends, values, _)| {
            let (buf, validity) = decode_v2(ends.as_slice(), values.as_slice(), TOTAL_LENGTH);
            PrimitiveArray::new(buf, validity)
        });
}

#[divan::bench(types = [u8, u16, u32, u64], args = RUN_LENGTHS)]
fn nonnull_v3<T: NativePType + From<u8>>(bencher: Bencher, avg: usize) {
    bencher
        .counter(ItemsCount::new(TOTAL_LENGTH))
        .with_inputs(rotating::<T>(avg, 1.0, false))
        .bench_refs(|(ends, values, _)| {
            runend_decode_typed_primitive(
                ends.as_slice(),
                0,
                values.as_slice(),
                Mask::new_true(values.len()),
                Nullability::NonNullable,
                TOTAL_LENGTH,
            )
        });
}

/// Bench-local mirror of the shipped kernel. Compare against `nonnull_v3_elem_fill` (also
/// bench-local) to isolate the long-run fill strategy without the crate-boundary confound that
/// makes `nonnull_v3` unusable for that comparison.
#[divan::bench(types = [u8, u32, u64], args = RUN_LENGTHS)]
fn nonnull_v3_local<T: NativePType + From<u8>>(bencher: Bencher, avg: usize) {
    bencher
        .counter(ItemsCount::new(TOTAL_LENGTH))
        .with_inputs(rotating::<T>(avg, 1.0, false))
        .bench_refs(|(ends, values, _)| {
            let (buf, validity) = decode_v3_local(ends.as_slice(), values.as_slice(), TOTAL_LENGTH);
            PrimitiveArray::new(buf, validity)
        });
}

/// The previous long-run fill (element loop, baseline SSE2 width) instead of the shipped
/// doubling `memcpy`. Identical to `nonnull_v3` below the 2 KiB doubling threshold; above it
/// the shipped kernel should win.
#[divan::bench(types = [u8, u16, u32, u64], args = RUN_LENGTHS)]
fn nonnull_v3_elem_fill<T: NativePType + From<u8>>(bencher: Bencher, avg: usize) {
    bencher
        .counter(ItemsCount::new(TOTAL_LENGTH))
        .with_inputs(rotating::<T>(avg, 1.0, false))
        .bench_refs(|(ends, values, _)| {
            let (buf, validity) =
                decode_v3_elem_fill(ends.as_slice(), values.as_slice(), TOTAL_LENGTH);
            PrimitiveArray::new(buf, validity)
        });
}

/// The rejected byte word-splat kernel. For widths > 1 this is identical to `nonnull_v3`;
/// only u8 differs, and there the shipped `nonnull_v3` (generic path) is faster across all run
/// lengths — which is why the shipped kernel carries no byte special case.
#[divan::bench(types = [u8, u16, u32, u64], args = RUN_LENGTHS)]
fn nonnull_v3_byte_splat<T: NativePType + From<u8>>(bencher: Bencher, avg: usize) {
    bencher
        .counter(ItemsCount::new(TOTAL_LENGTH))
        .with_inputs(rotating::<T>(avg, 1.0, false))
        .bench_refs(|(ends, values, _)| {
            let (buf, validity) =
                decode_v3_byte_splat(ends.as_slice(), values.as_slice(), TOTAL_LENGTH);
            PrimitiveArray::new(buf, validity)
        });
}

// ---- Group A2: byte-uniform values (zeros) vs arbitrary values ----
//
// A byte-uniform value fills with `memset`; an arbitrary one takes the doubling path.
// `zeros_v3` vs `zeros_v3_prev` isolates that fast path, and `rand_v3_prev` vs `nonnull_v3`
// confirms arbitrary values are unaffected by it.

fn zero_data<T: NativePType + From<u8>>(avg: usize) -> (Buffer<u32>, Buffer<T>) {
    let (ends, values, _) = make_data_values::<T>(SEED, TOTAL_LENGTH, max_run_len(avg), 1.0, true);
    (ends, values)
}

#[divan::bench(types = [u32, u64], args = RUN_LENGTHS)]
fn zeros_v3<T: NativePType + From<u8>>(bencher: Bencher, avg: usize) {
    let (ends, values) = zero_data::<T>(avg);
    bencher
        .counter(ItemsCount::new(TOTAL_LENGTH))
        .with_inputs(|| (ends.clone(), values.clone()))
        .bench_refs(|(ends, values)| {
            runend_decode_typed_primitive(
                ends.as_slice(),
                0,
                values.as_slice(),
                Mask::new_true(values.len()),
                Nullability::NonNullable,
                TOTAL_LENGTH,
            )
        });
}

#[divan::bench(types = [u32, u64], args = RUN_LENGTHS)]
fn zeros_v3_prev<T: NativePType + From<u8>>(bencher: Bencher, avg: usize) {
    let (ends, values) = zero_data::<T>(avg);
    bencher
        .counter(ItemsCount::new(TOTAL_LENGTH))
        .with_inputs(|| (ends.clone(), values.clone()))
        .bench_refs(|(ends, values)| {
            let (buf, validity) =
                decode_v3_no_memset(ends.as_slice(), values.as_slice(), TOTAL_LENGTH);
            PrimitiveArray::new(buf, validity)
        });
}

/// Arbitrary (not byte-uniform) values through the pre-`memset` kernel; compare against
/// `nonnull_v3` to confirm the fast path costs nothing when it does not apply.
#[divan::bench(types = [u32, u64], args = RUN_LENGTHS)]
fn rand_v3_prev<T: NativePType + From<u8>>(bencher: Bencher, avg: usize) {
    bencher
        .counter(ItemsCount::new(TOTAL_LENGTH))
        .with_inputs(rotating::<T>(avg, 1.0, false))
        .bench_refs(|(ends, values, _)| {
            let (buf, validity) =
                decode_v3_no_memset(ends.as_slice(), values.as_slice(), TOTAL_LENGTH);
            PrimitiveArray::new(buf, validity)
        });
}

// ---- Group B: nullable run-length sweep at 90% valid ----

#[divan::bench(types = [u8, u32, u64], args = RUN_LENGTHS)]
fn nullable_n0<T: NativePType + From<u8>>(bencher: Bencher, avg: usize) {
    bencher
        .counter(ItemsCount::new(TOTAL_LENGTH))
        .with_inputs(rotating::<T>(avg, 0.9, false))
        .bench_refs(|(ends, values, validity)| {
            let (buf, decoded_validity) = decode_v0(
                trimmed_ends_iter(ends.as_slice(), 0, TOTAL_LENGTH),
                values.as_slice(),
                Mask::from_buffer(validity.clone()),
                TOTAL_LENGTH,
            );
            PrimitiveArray::new(buf, decoded_validity)
        });
}

#[divan::bench(types = [u8, u32, u64], args = RUN_LENGTHS)]
fn nullable_n2<T: NativePType + From<u8>>(bencher: Bencher, avg: usize) {
    bencher
        .counter(ItemsCount::new(TOTAL_LENGTH))
        .with_inputs(rotating::<T>(avg, 0.9, false))
        .bench_refs(|(ends, values, validity)| {
            let (buf, decoded_validity) =
                decode_n2(ends.as_slice(), values.as_slice(), validity, TOTAL_LENGTH);
            PrimitiveArray::new(buf, decoded_validity)
        });
}

#[divan::bench(types = [u8, u32, u64], args = RUN_LENGTHS)]
fn nullable_n3<T: NativePType + From<u8>>(bencher: Bencher, avg: usize) {
    bencher
        .counter(ItemsCount::new(TOTAL_LENGTH))
        .with_inputs(rotating::<T>(avg, 0.9, false))
        .bench_refs(|(ends, values, validity)| {
            runend_decode_typed_primitive(
                ends.as_slice(),
                0,
                values.as_slice(),
                Mask::from_buffer(validity.clone()),
                Nullability::Nullable,
                TOTAL_LENGTH,
            )
        });
}

// ---- Group C: validity-density sweep (u32, run length 8) ----

fn density_inputs(density_pct: u32) -> impl Fn() -> (Buffer<u32>, Buffer<u32>, BitBuffer) {
    rotating::<u32>(DENSITY_RUN_LENGTH, f64::from(density_pct) / 100.0, false)
}

#[derive(Clone, Copy)]
struct Density(u32);

impl fmt::Display for Density {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "valid{:02}pct", self.0)
    }
}

/// Fraction-valid densities (percent). Spans balanced (50) to strongly skewed (99) validity,
/// plus a null-heavy point (10) to show the prefill win is symmetric about 50%.
const DENSITIES: &[Density] = &[
    Density(10),
    Density(50),
    Density(70),
    Density(90),
    Density(95),
    Density(99),
];

#[divan::bench(args = DENSITIES)]
fn density_n0(bencher: Bencher, density: Density) {
    bencher
        .counter(ItemsCount::new(TOTAL_LENGTH))
        .with_inputs(density_inputs(density.0))
        .bench_refs(|(ends, values, validity)| {
            let (buf, decoded_validity) = decode_v0(
                trimmed_ends_iter(ends.as_slice(), 0, TOTAL_LENGTH),
                values.as_slice(),
                Mask::from_buffer(validity.clone()),
                TOTAL_LENGTH,
            );
            PrimitiveArray::new(buf, decoded_validity)
        });
}

#[divan::bench(args = DENSITIES)]
fn density_n2(bencher: Bencher, density: Density) {
    bencher
        .counter(ItemsCount::new(TOTAL_LENGTH))
        .with_inputs(density_inputs(density.0))
        .bench_refs(|(ends, values, validity)| {
            let (buf, decoded_validity) =
                decode_n2(ends.as_slice(), values.as_slice(), validity, TOTAL_LENGTH);
            PrimitiveArray::new(buf, decoded_validity)
        });
}

#[divan::bench(args = DENSITIES)]
fn density_n3(bencher: Bencher, density: Density) {
    bencher
        .counter(ItemsCount::new(TOTAL_LENGTH))
        .with_inputs(density_inputs(density.0))
        .bench_refs(|(ends, values, validity)| {
            runend_decode_typed_primitive(
                ends.as_slice(),
                0,
                values.as_slice(),
                Mask::from_buffer(validity.clone()),
                Nullability::Nullable,
                TOTAL_LENGTH,
            )
        });
}

// ---- Group D: branch-predictor control ----
//
// Both benchmarks run the same shipped kernel in the same binary over the same distribution;
// only the input provider differs. `predictor_fixed` replays one dataset, so the predictor can
// learn its run-length sequence; `predictor_rotating` cycles `DATASETS` of them, as every other
// benchmark here does. The gap is the amount a single-dataset benchmark would have flattered
// the length-dependent branches. Rotating also spreads input reads over more memory, so treat
// the gap as an upper bound on the predictor component alone.

#[divan::bench(types = [u32, u64], args = RUN_LENGTHS)]
fn predictor_fixed<T: NativePType + From<u8>>(bencher: Bencher, avg: usize) {
    bencher
        .counter(ItemsCount::new(TOTAL_LENGTH))
        .with_inputs(fixed::<T>(avg, 1.0))
        .bench_refs(|(ends, values, _)| {
            runend_decode_typed_primitive(
                ends.as_slice(),
                0,
                values.as_slice(),
                Mask::new_true(values.len()),
                Nullability::NonNullable,
                TOTAL_LENGTH,
            )
        });
}

#[divan::bench(types = [u32, u64], args = RUN_LENGTHS)]
fn predictor_rotating<T: NativePType + From<u8>>(bencher: Bencher, avg: usize) {
    bencher
        .counter(ItemsCount::new(TOTAL_LENGTH))
        .with_inputs(rotating::<T>(avg, 1.0, false))
        .bench_refs(|(ends, values, _)| {
            runend_decode_typed_primitive(
                ends.as_slice(),
                0,
                values.as_slice(),
                Mask::new_true(values.len()),
                Nullability::NonNullable,
                TOTAL_LENGTH,
            )
        });
}
