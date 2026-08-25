// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks comparing full decompression of a bit-packed array with exceptions across the
//! three patch representations:
//!
//! * `interior`: classic `BitPacked` with interior `Patches` — unpack everything, then walk the
//!   sorted absolute patch indices over the full-length output buffer.
//! * `patched`: the lane-transposed experimental `Patched(BitPacked)` array — unpack everything,
//!   then apply lane-grouped patches.
//! * `patches_fused`: the block-relative `Patches(BitPacked)` array — each 1024-element FastLanes
//!   chunk is unpacked and its (contiguous) patch run is applied while the chunk is cache-hot.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

use std::sync::LazyLock;

use divan::Bencher;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Patched;
use vortex_array::arrays::Patches;
use vortex_array::arrays::patches::PatchFn;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_buffer::BufferMut;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::BitPackedArray;
use vortex_fastlanes::BitPackedArrayExt;
use vortex_fastlanes::BitPackedData;
use vortex_mask::Mask;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

const ARRAY_LEN: usize = 1 << 20;
const BIT_WIDTH: u8 = 9;

/// Fraction of positions patched.
const BENCH_ARGS: &[f64] = &[0.001, 0.01, 0.05];

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_fastlanes::initialize(&session);
    session
});

/// Bit-pack values where a `fraction` of positions exceed the bit width and become patches.
fn bitpacked_with_patches(fraction: f64) -> BitPackedArray {
    let mut ctx = SESSION.create_execution_ctx();
    let mut rng = StdRng::seed_from_u64(42);
    let n_patched = (ARRAY_LEN as f64 * fraction) as usize;

    let mut values = BufferMut::from_iter((0..ARRAY_LEN).map(|i| (i % 512) as u32));
    for _ in 0..n_patched {
        let position = rng.random_range(0..ARRAY_LEN);
        values[position] = rng.random_range(100_000..200_000u32);
    }

    let array = BitPackedData::encode(&values.freeze().into_array(), BIT_WIDTH, &mut ctx).unwrap();
    assert!(array.patches().is_some());
    array
}

/// Rebuild the same values as a patchless `BitPacked` plus the extracted `Patches`.
fn split_patches(bitpacked: &BitPackedArray) -> (ArrayRef, vortex_array::patches::Patches) {
    let patches = bitpacked.patches().unwrap();
    let without_patches = BitPacked::try_new(
        bitpacked.packed().clone(),
        bitpacked.dtype().as_ptype(),
        bitpacked.validity().unwrap(),
        None,
        bitpacked.bit_width(),
        bitpacked.len(),
        bitpacked.offset(),
    )
    .unwrap()
    .into_array();
    (without_patches, patches)
}

fn bench_canonicalize(bencher: Bencher, array: ArrayRef) {
    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_local_values(|(array, mut ctx)| {
            divan::black_box(array.execute::<Canonical>(&mut ctx).unwrap());
        });
}

/// Classic pathway: unpack the whole array, then apply interior patches over the output.
#[divan::bench(args = BENCH_ARGS)]
fn canonicalize_interior(bencher: Bencher, fraction: f64) {
    let array = bitpacked_with_patches(fraction).into_array();
    bench_canonicalize(bencher, array);
}

/// Legacy experimental pathway: `Patched(BitPacked)` with lane-transposed patches.
#[divan::bench(args = BENCH_ARGS)]
fn canonicalize_patched(bencher: Bencher, fraction: f64) {
    let mut ctx = SESSION.create_execution_ctx();
    let bitpacked = bitpacked_with_patches(fraction);
    let (inner, patches) = split_patches(&bitpacked);
    let array = Patched::from_array_and_patches(inner, &patches, &mut ctx)
        .unwrap()
        .into_array();
    bench_canonicalize(bencher, array);
}

/// New pathway: `Patches(BitPacked)` with block-relative patches, executed by the fused
/// chunk-at-a-time kernel.
#[divan::bench(args = BENCH_ARGS)]
fn canonicalize_patches_fused(bencher: Bencher, fraction: f64) {
    let mut ctx = SESSION.create_execution_ctx();
    let bitpacked = bitpacked_with_patches(fraction);
    let (inner, patches) = split_patches(&bitpacked);
    let array = Patches::from_array_and_patches(inner, &patches, PatchFn::Overwrite, &mut ctx)
        .unwrap()
        .into_array();
    bench_canonicalize(bencher, array);
}

/// Random access through the new block-relative `Patches(BitPacked)` array.
#[divan::bench(args = BENCH_ARGS)]
fn scalar_at_patches(bencher: Bencher, fraction: f64) {
    let mut ctx = SESSION.create_execution_ctx();
    let bitpacked = bitpacked_with_patches(fraction);
    let (inner, patches) = split_patches(&bitpacked);
    let array = Patches::from_array_and_patches(inner, &patches, PatchFn::Overwrite, &mut ctx)
        .unwrap()
        .into_array();
    bench_scalar_at(bencher, array);
}

/// Random access through the legacy lane-transposed `Patched(BitPacked)` array.
#[divan::bench(args = BENCH_ARGS)]
fn scalar_at_patched(bencher: Bencher, fraction: f64) {
    let mut ctx = SESSION.create_execution_ctx();
    let bitpacked = bitpacked_with_patches(fraction);
    let (inner, patches) = split_patches(&bitpacked);
    let array = Patched::from_array_and_patches(inner, &patches, &mut ctx)
        .unwrap()
        .into_array();
    bench_scalar_at(bencher, array);
}

/// Random access through the classic `BitPacked` array with interior patches.
#[divan::bench(args = BENCH_ARGS)]
fn scalar_at_interior(bencher: Bencher, fraction: f64) {
    let array = bitpacked_with_patches(fraction).into_array();
    bench_scalar_at(bencher, array);
}

fn bench_scalar_at(bencher: Bencher, array: ArrayRef) {
    const NUM_QUERIES: usize = 1000;
    let mut rng = StdRng::seed_from_u64(7);
    let queries: Vec<usize> = (0..NUM_QUERIES)
        .map(|_| rng.random_range(0..ARRAY_LEN))
        .collect();
    bencher
        .with_inputs(|| {
            (
                array.clone(),
                SESSION.create_execution_ctx(),
                queries.clone(),
            )
        })
        .bench_local_values(|(array, mut ctx, queries)| {
            for q in queries {
                divan::black_box(array.execute_scalar(q, &mut ctx).unwrap());
            }
        });
}

/// A selective filter clustered into a narrow row range: block pruning should pay off.
fn clustered_mask() -> Mask {
    Mask::from_indices(ARRAY_LEN, 500_000..502_000)
}

/// A 1%-selectivity filter spread uniformly over the array.
fn spread_mask() -> Mask {
    let mut rng = StdRng::seed_from_u64(11);
    let mut positions: Vec<usize> = (0..ARRAY_LEN / 100)
        .map(|_| rng.random_range(0..ARRAY_LEN))
        .collect();
    positions.sort_unstable();
    positions.dedup();
    Mask::from_indices(ARRAY_LEN, positions)
}

fn bench_filter(bencher: Bencher, array: ArrayRef, mask: Mask) {
    bencher
        .with_inputs(|| (array.clone(), mask.clone(), SESSION.create_execution_ctx()))
        .bench_local_values(|(array, mask, mut ctx)| {
            let filtered = array.filter(mask).unwrap().optimize().unwrap();
            divan::black_box(filtered.execute::<Canonical>(&mut ctx).unwrap());
        });
}

/// Clustered selective filter over the new block-relative `Patches(BitPacked)` array.
#[divan::bench(args = BENCH_ARGS)]
fn filter_clustered_patches(bencher: Bencher, fraction: f64) {
    let mut ctx = SESSION.create_execution_ctx();
    let bitpacked = bitpacked_with_patches(fraction);
    let (inner, patches) = split_patches(&bitpacked);
    let array = Patches::from_array_and_patches(inner, &patches, PatchFn::Overwrite, &mut ctx)
        .unwrap()
        .into_array();
    bench_filter(bencher, array, clustered_mask());
}

/// Clustered selective filter over the legacy lane-transposed `Patched(BitPacked)` array.
#[divan::bench(args = BENCH_ARGS)]
fn filter_clustered_patched(bencher: Bencher, fraction: f64) {
    let mut ctx = SESSION.create_execution_ctx();
    let bitpacked = bitpacked_with_patches(fraction);
    let (inner, patches) = split_patches(&bitpacked);
    let array = Patched::from_array_and_patches(inner, &patches, &mut ctx)
        .unwrap()
        .into_array();
    bench_filter(bencher, array, clustered_mask());
}

/// Clustered selective filter over the classic `BitPacked` array with interior patches.
#[divan::bench(args = BENCH_ARGS)]
fn filter_clustered_interior(bencher: Bencher, fraction: f64) {
    let array = bitpacked_with_patches(fraction).into_array();
    bench_filter(bencher, array, clustered_mask());
}

/// Uniform 1% filter over the new block-relative `Patches(BitPacked)` array.
#[divan::bench(args = BENCH_ARGS)]
fn filter_spread_patches(bencher: Bencher, fraction: f64) {
    let mut ctx = SESSION.create_execution_ctx();
    let bitpacked = bitpacked_with_patches(fraction);
    let (inner, patches) = split_patches(&bitpacked);
    let array = Patches::from_array_and_patches(inner, &patches, PatchFn::Overwrite, &mut ctx)
        .unwrap()
        .into_array();
    bench_filter(bencher, array, spread_mask());
}

/// Uniform 1% filter over the legacy lane-transposed `Patched(BitPacked)` array.
#[divan::bench(args = BENCH_ARGS)]
fn filter_spread_patched(bencher: Bencher, fraction: f64) {
    let mut ctx = SESSION.create_execution_ctx();
    let bitpacked = bitpacked_with_patches(fraction);
    let (inner, patches) = split_patches(&bitpacked);
    let array = Patched::from_array_and_patches(inner, &patches, &mut ctx)
        .unwrap()
        .into_array();
    bench_filter(bencher, array, spread_mask());
}

/// Uniform 1% filter over the classic `BitPacked` array with interior patches.
#[divan::bench(args = BENCH_ARGS)]
fn filter_spread_interior(bencher: Bencher, fraction: f64) {
    let array = bitpacked_with_patches(fraction).into_array();
    bench_filter(bencher, array, spread_mask());
}
