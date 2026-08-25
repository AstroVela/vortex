// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks comparing the block-relative [`PatchesArray`] layout against the legacy patch
//! representations:
//!
//! * `search_kernel_*`: the raw lookup algorithms, isolating "one global binary search over
//!   absolute indices" from "constant-time skip-index load + in-block binary search over `u16`s".
//! * `search_legacy_*`: the legacy `Patches` struct's `search_index` (binary and chunked).
//! * `scalar_at_*`: random access through the array vtables (new `Patches` vs legacy `Patched`).
//! * `execute_*`: full patch application over a canonical primitive base.

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
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::patches::PATCH_BLOCK_SIZE;
use vortex_array::arrays::patches::PatchFn;
use vortex_array::arrays::patches::search_block;
use vortex_array::patches::Patches as PatchesStruct;
use vortex_buffer::Buffer;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

const ARRAY_LEN: usize = 1 << 20;
const NUM_QUERIES: usize = 1000;

/// Number of patches over the 1M-element array: ~1, ~10, and ~100 patches per 1024-block.
const BENCH_ARGS: &[usize] = &[1_000, 10_000, 100_000];

static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

/// Sorted unique random patch positions.
fn patch_positions(n_patches: usize) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut positions: Vec<u64> = (0..n_patches)
        .map(|_| rng.random_range(0..ARRAY_LEN as u64))
        .collect();
    positions.sort_unstable();
    positions.dedup();
    positions
}

fn queries() -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(7);
    (0..NUM_QUERIES)
        .map(|_| rng.random_range(0..ARRAY_LEN))
        .collect()
}

/// The block-relative layout: per-block skip offsets plus in-block u16 positions.
fn block_layout(positions: &[u64]) -> (Vec<u32>, Vec<u16>) {
    let n_blocks = ARRAY_LEN.div_ceil(PATCH_BLOCK_SIZE);
    let mut skip = Vec::with_capacity(n_blocks + 1);
    let mut rel = Vec::with_capacity(positions.len());
    let mut cursor = 0usize;
    for block in 0..n_blocks {
        skip.push(cursor as u32);
        let block_start = block * PATCH_BLOCK_SIZE;
        let block_end = block_start + PATCH_BLOCK_SIZE;
        while cursor < positions.len() && (positions[cursor] as usize) < block_end {
            rel.push((positions[cursor] as usize - block_start) as u16);
            cursor += 1;
        }
    }
    skip.push(positions.len() as u32);
    (skip, rel)
}

fn legacy_patches(n_patches: usize, chunked: bool) -> PatchesStruct {
    let positions = patch_positions(n_patches);
    let values: Buffer<i32> = (0..positions.len() as i32).collect();
    let chunk_offsets = chunked.then(|| {
        let offsets: Vec<u64> = (0..ARRAY_LEN)
            .step_by(PATCH_BLOCK_SIZE)
            .map(|start| positions.partition_point(|&idx| (idx as usize) < start) as u64)
            .collect();
        Buffer::from(offsets).into_array()
    });
    PatchesStruct::new(
        ARRAY_LEN,
        0,
        Buffer::from(positions).into_array(),
        values.into_array(),
        chunk_offsets,
    )
    .unwrap()
}

fn patches_array(n_patches: usize) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let inner = PrimitiveArray::from_iter((0..ARRAY_LEN).map(|i| i as i32)).into_array();
    Patches::from_array_and_patches(
        inner,
        &legacy_patches(n_patches, false),
        PatchFn::Overwrite,
        &mut ctx,
    )
    .unwrap()
    .into_array()
}

fn patched_array(n_patches: usize) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let inner = PrimitiveArray::from_iter((0..ARRAY_LEN).map(|i| i as i32)).into_array();
    Patched::from_array_and_patches(inner, &legacy_patches(n_patches, false), &mut ctx)
        .unwrap()
        .into_array()
}

/// One global binary search over sorted absolute u64 indices per query.
#[divan::bench(args = BENCH_ARGS)]
fn search_kernel_global_binary_u64(bencher: Bencher, n_patches: usize) {
    let positions = patch_positions(n_patches);
    let queries = queries();
    bencher
        .with_inputs(|| (&positions, &queries))
        .bench_refs(|(positions, queries)| {
            for &q in queries.iter() {
                divan::black_box(positions.binary_search(&(q as u64)).ok());
            }
        });
}

/// One global binary search over sorted absolute u32 indices per query (narrower index type).
#[divan::bench(args = BENCH_ARGS)]
fn search_kernel_global_binary_u32(bencher: Bencher, n_patches: usize) {
    let positions: Vec<u32> = patch_positions(n_patches)
        .into_iter()
        .map(|p| p as u32)
        .collect();
    let queries = queries();
    bencher
        .with_inputs(|| (&positions, &queries))
        .bench_refs(|(positions, queries)| {
            for &q in queries.iter() {
                divan::black_box(positions.binary_search(&(q as u32)).ok());
            }
        });
}

/// Constant-time skip-index load plus a binary search within one block's u16 indices.
#[divan::bench(args = BENCH_ARGS)]
fn search_kernel_skip_index(bencher: Bencher, n_patches: usize) {
    let (skip, rel) = block_layout(&patch_positions(n_patches));
    let queries = queries();
    bencher
        .with_inputs(|| (&skip, &rel, &queries))
        .bench_refs(|(skip, rel, queries)| {
            for &q in queries.iter() {
                let block = q / PATCH_BLOCK_SIZE;
                let range = skip[block] as usize..skip[block + 1] as usize;
                divan::black_box(search_block(rel, range, (q % PATCH_BLOCK_SIZE) as u16));
            }
        });
}

/// The legacy `Patches` struct: one binary search over the whole indices array.
#[divan::bench(args = BENCH_ARGS)]
fn search_legacy_binary(bencher: Bencher, n_patches: usize) {
    let patches = legacy_patches(n_patches, false);
    let queries = queries();
    bencher
        .with_inputs(|| (&patches, &queries))
        .bench_refs(|(patches, queries)| {
            for &q in queries.iter() {
                divan::black_box(patches.search_index(q).unwrap());
            }
        });
}

/// The legacy `Patches` struct with chunk offsets: chunk lookup plus in-chunk binary search.
#[divan::bench(args = BENCH_ARGS)]
fn search_legacy_chunked(bencher: Bencher, n_patches: usize) {
    let patches = legacy_patches(n_patches, true);
    let queries = queries();
    bencher
        .with_inputs(|| (&patches, &queries))
        .bench_refs(|(patches, queries)| {
            for &q in queries.iter() {
                divan::black_box(patches.search_index(q).unwrap());
            }
        });
}

/// Random access through the new block-relative `Patches` array vtable.
#[divan::bench(args = BENCH_ARGS)]
fn scalar_at_patches_array(bencher: Bencher, n_patches: usize) {
    let array = patches_array(n_patches);
    let queries = queries();
    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx(), &queries))
        .bench_local_values(|(array, mut ctx, queries)| {
            for &q in queries.iter() {
                divan::black_box(array.execute_scalar(q, &mut ctx).unwrap());
            }
        });
}

/// Random access through the legacy lane-transposed `Patched` array vtable.
#[divan::bench(args = BENCH_ARGS)]
fn scalar_at_patched_array(bencher: Bencher, n_patches: usize) {
    let array = patched_array(n_patches);
    let queries = queries();
    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx(), &queries))
        .bench_local_values(|(array, mut ctx, queries)| {
            for &q in queries.iter() {
                divan::black_box(array.execute_scalar(q, &mut ctx).unwrap());
            }
        });
}

/// Full patch application of the new block-relative layout over a canonical primitive base.
#[divan::bench(args = BENCH_ARGS)]
fn execute_patches_array(bencher: Bencher, n_patches: usize) {
    let array = patches_array(n_patches);
    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_local_values(|(array, mut ctx)| {
            divan::black_box(array.execute::<Canonical>(&mut ctx).unwrap());
        });
}

/// Full patch application of the legacy lane-transposed layout over a primitive base.
#[divan::bench(args = BENCH_ARGS)]
fn execute_patched_array(bencher: Bencher, n_patches: usize) {
    let array = patched_array(n_patches);
    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_local_values(|(array, mut ctx)| {
            divan::black_box(array.execute::<Canonical>(&mut ctx).unwrap());
        });
}
