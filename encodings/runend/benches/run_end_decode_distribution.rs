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
//! - `nonnull_{v0,v1,v2,v3}` — run-length sweep, u8/u32/u64. Isolates the non-null structural
//!   wins (v0->v1 iterator/bookkeeping, v1->v2 scalar-fill -> chunk stores) and the byte
//!   word/memset path (v2->v3 on u8) against run length and element width.
//! - `nullable_{n0,n2,n3}` — run-length sweep, u8/u32/u64 at 90% valid. Isolates the nullable
//!   structural wins and the majority-prefill validity (n2->n3), including the wide-element
//!   long-run fill dispatch.
//! - `density_{n0,n2,n3}` — validity-density sweep, u32 at run length 8. Isolates the
//!   prefill validity win (n2->n3) against validity skew: prefill rewrites only the minority
//!   runs, so its advantage grows as validity moves away from 50/50.

#![expect(clippy::cast_possible_truncation)]

use std::fmt;

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
use decode_variants::make_data;

fn main() {
    divan::main();
}

const SEED: u64 = 0x5eed;
const TOTAL_LENGTH: usize = 65_536;

/// Average run length. Uniform run lengths in `1..=(2*avg-1)` have this mean.
const RUN_LENGTHS: &[usize] = &[2, 4, 8, 16, 32, 64, 256, 1024];

/// Fixed short run length for the density sweep, where per-run validity work is visible.
const DENSITY_RUN_LENGTH: usize = 8;

fn max_run_len(avg: usize) -> usize {
    2 * avg - 1
}

// ---- Group A: non-nullable run-length sweep ----

fn nonnull_data<T: NativePType + From<u8>>(avg: usize) -> (Buffer<u32>, Buffer<T>) {
    let (ends, values, _) = make_data::<T>(SEED, TOTAL_LENGTH, max_run_len(avg), 1.0);
    (ends, values)
}

#[divan::bench(types = [u8, u32, u64], args = RUN_LENGTHS)]
fn nonnull_v0<T: NativePType + From<u8>>(bencher: Bencher, avg: usize) {
    let (ends, values) = nonnull_data::<T>(avg);
    bencher
        .counter(ItemsCount::new(TOTAL_LENGTH))
        .with_inputs(|| (ends.clone(), values.clone()))
        .bench_refs(|(ends, values)| {
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
    let (ends, values) = nonnull_data::<T>(avg);
    bencher
        .counter(ItemsCount::new(TOTAL_LENGTH))
        .with_inputs(|| (ends.clone(), values.clone()))
        .bench_refs(|(ends, values)| {
            let (buf, validity) = decode_v1(ends.as_slice(), values.as_slice(), TOTAL_LENGTH);
            PrimitiveArray::new(buf, validity)
        });
}

#[divan::bench(types = [u8, u32, u64], args = RUN_LENGTHS)]
fn nonnull_v2<T: NativePType + From<u8>>(bencher: Bencher, avg: usize) {
    let (ends, values) = nonnull_data::<T>(avg);
    bencher
        .counter(ItemsCount::new(TOTAL_LENGTH))
        .with_inputs(|| (ends.clone(), values.clone()))
        .bench_refs(|(ends, values)| {
            let (buf, validity) = decode_v2(ends.as_slice(), values.as_slice(), TOTAL_LENGTH);
            PrimitiveArray::new(buf, validity)
        });
}

#[divan::bench(types = [u8, u32, u64], args = RUN_LENGTHS)]
fn nonnull_v3<T: NativePType + From<u8>>(bencher: Bencher, avg: usize) {
    let (ends, values) = nonnull_data::<T>(avg);
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

// ---- Group B: nullable run-length sweep at 90% valid ----

fn nullable_data<T: NativePType + From<u8>>(avg: usize) -> (Buffer<u32>, Buffer<T>, BitBuffer) {
    make_data::<T>(SEED, TOTAL_LENGTH, max_run_len(avg), 0.9)
}

#[divan::bench(types = [u8, u32, u64], args = RUN_LENGTHS)]
fn nullable_n0<T: NativePType + From<u8>>(bencher: Bencher, avg: usize) {
    let (ends, values, validity) = nullable_data::<T>(avg);
    bencher
        .counter(ItemsCount::new(TOTAL_LENGTH))
        .with_inputs(|| (ends.clone(), values.clone(), validity.clone()))
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
    let (ends, values, validity) = nullable_data::<T>(avg);
    bencher
        .counter(ItemsCount::new(TOTAL_LENGTH))
        .with_inputs(|| (ends.clone(), values.clone(), validity.clone()))
        .bench_refs(|(ends, values, validity)| {
            let (buf, decoded_validity) =
                decode_n2(ends.as_slice(), values.as_slice(), validity, TOTAL_LENGTH);
            PrimitiveArray::new(buf, decoded_validity)
        });
}

#[divan::bench(types = [u8, u32, u64], args = RUN_LENGTHS)]
fn nullable_n3<T: NativePType + From<u8>>(bencher: Bencher, avg: usize) {
    let (ends, values, validity) = nullable_data::<T>(avg);
    bencher
        .counter(ItemsCount::new(TOTAL_LENGTH))
        .with_inputs(|| (ends.clone(), values.clone(), validity.clone()))
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

fn density_data(density_pct: u32) -> (Buffer<u32>, Buffer<u32>, BitBuffer) {
    make_data::<u32>(
        SEED,
        TOTAL_LENGTH,
        max_run_len(DENSITY_RUN_LENGTH),
        f64::from(density_pct) / 100.0,
    )
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
    let (ends, values, validity) = density_data(density.0);
    bencher
        .counter(ItemsCount::new(TOTAL_LENGTH))
        .with_inputs(|| (ends.clone(), values.clone(), validity.clone()))
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
    let (ends, values, validity) = density_data(density.0);
    bencher
        .counter(ItemsCount::new(TOTAL_LENGTH))
        .with_inputs(|| (ends.clone(), values.clone(), validity.clone()))
        .bench_refs(|(ends, values, validity)| {
            let (buf, decoded_validity) =
                decode_n2(ends.as_slice(), values.as_slice(), validity, TOTAL_LENGTH);
            PrimitiveArray::new(buf, decoded_validity)
        });
}

#[divan::bench(args = DENSITIES)]
fn density_n3(bencher: Bencher, density: Density) {
    let (ends, values, validity) = density_data(density.0);
    bencher
        .counter(ItemsCount::new(TOTAL_LENGTH))
        .with_inputs(|| (ends.clone(), values.clone(), validity.clone()))
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
