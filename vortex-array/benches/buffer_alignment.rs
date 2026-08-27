// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Measures what buffer base alignment is worth to the elementwise kernels.
//!
//! Every case slices its operands out of a single 256-byte aligned allocation, so the aligned
//! and misaligned variants differ **only** in the base pointer: same allocation, same length,
//! same values, same encoding. Building one operand with an aligned constructor and the other
//! with a growing one would also vary capacity, allocator state, and page locality, and the
//! difference could no longer be attributed to alignment.
//!
//! `slice(1..len + 1)` shifts the base by `size_of::<T>()` bytes, which is the misalignment a
//! sliced array actually carries: [`Buffer::slice`] only requires the offset to be aligned to
//! `Alignment::of::<T>()`, so any row range that is not a multiple of 8 elements lands here for
//! `i64`.
//!
//! The comparison cases carry `#[cpu_features]` because vector width decides how much a split
//! load costs: a 32-byte load crossing a 64-byte line is two L1 accesses, and a kernel compiled
//! to 128-bit vectors splits half as often as one compiled to 256-bit. The effect only appears
//! while the working set is cache-resident — once the kernel is bandwidth-bound the split loads
//! hide behind memory latency, which is why the sizes below straddle L2 rather than sitting in
//! one regime.

#![expect(clippy::unwrap_used)]

use divan::Bencher;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_buffer::Buffer;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

/// Element counts spanning the L2 boundary on a typical server core.
///
/// 8192 matches the writer's block length and the size the other comparison benchmarks use;
/// 100_000 is [`IDEAL_SPLIT_SIZE`], the row count a scan actually hands to these kernels.
///
/// [`IDEAL_SPLIT_SIZE`]: https://github.com/vortex-data/vortex/blob/develop/vortex-layout/src/scan/mod.rs
const SIZES: &[usize] = &[8_192, 65_536, 100_000];

/// Slack elements so the misaligned slice keeps the same length as the aligned one.
const PAD: usize = 8;

/// Build one 256-byte aligned allocation and slice `len` elements out of it.
///
/// `offset` is in elements: 0 leaves the 256-byte base alignment intact, 1 shifts it to 8 bytes.
fn operand(len: usize, offset: usize, seed: i64) -> ArrayRef {
    let values: Vec<i64> = (0..(len + PAD) as i64)
        .map(|index| (index.wrapping_mul(2_654_435_761).wrapping_add(seed)) % 100_000_000)
        .collect();
    // `copy_from` sizes the allocation up front, so it keeps the 256-byte over-alignment that a
    // reallocating constructor would drop.
    let base: Buffer<i64> = Buffer::copy_from(&values);
    let sliced = base.slice(offset..offset + len);

    // A silent regression here would leave the benchmark measuring nothing, so fail loudly if
    // the operand did not land on the alignment this case is meant to compare.
    let addr = sliced.as_ptr() as usize;
    if offset == 0 {
        assert_eq!(
            addr % 64,
            0,
            "aligned operand is only {}-byte aligned",
            addr % 64
        );
    } else {
        assert_ne!(
            addr % 64,
            0,
            "misaligned operand landed on a 64-byte boundary"
        );
    }

    sliced.into_array()
}

fn bench_binary(bencher: Bencher, len: usize, offset: usize, operator: Operator) {
    let session = vortex_array::array_session();
    let lhs = operand(len, offset, 0);
    let rhs = operand(len, offset, 12_345);

    bencher
        .counter(ItemsCount::new(len))
        .with_inputs(|| (&lhs, &rhs, session.create_execution_ctx()))
        .bench_refs(|input| {
            input
                .0
                .clone()
                .binary(input.1.clone(), operator)
                .unwrap()
                .execute::<Canonical>(&mut input.2)
        });
}

#[vortex_bench_support::cpu_features]
#[divan::bench(args = SIZES)]
fn compare_aligned(bencher: Bencher, len: usize) {
    bench_binary(bencher, len, 0, Operator::Lt);
}

#[vortex_bench_support::cpu_features]
#[divan::bench(args = SIZES)]
fn compare_misaligned(bencher: Bencher, len: usize) {
    bench_binary(bencher, len, 1, Operator::Lt);
}

/// Addition is the control: it reads two operands and writes a full-width third, so it leaves
/// cache sooner than comparison (which writes one bit per row) and goes bandwidth-bound at a
/// size where comparison is still issue-bound.
#[vortex_bench_support::cpu_features]
#[divan::bench(args = SIZES)]
fn add_aligned(bencher: Bencher, len: usize) {
    bench_binary(bencher, len, 0, Operator::Add);
}

#[vortex_bench_support::cpu_features]
#[divan::bench(args = SIZES)]
fn add_misaligned(bencher: Bencher, len: usize) {
    bench_binary(bencher, len, 1, Operator::Add);
}
