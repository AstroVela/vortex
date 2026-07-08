// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for `take` on [`ChunkedArray`], materialized through the builder path.
//!
//! `take` is lazy (a `Dict` over the source), so the cost lands where the result is
//! materialized. These benches mirror the common consumer path of appending the taken
//! array into an [`ArrayBuilder`], and compare against `execute::<Canonical>`.
//!
//! Parameterized over:
//! - Number of indices to take
//! - Shuffled vs sorted indices (sortedness dominates `take_chunked`'s index grouping cost)
//! - Flat primitive chunks vs fixed-size-list shapes, where row indices are expanded into
//!   `list_size` element indices before reaching the chunked elements child

#![expect(clippy::cast_possible_truncation)]
#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use half::f16;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::index::sample;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builders::builder_with_capacity;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_session::VortexSession;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

/// Chunk length for the flat primitive benches.
const CHUNK_LEN: usize = 65_536;

/// Number of chunks for the flat primitive benches.
const NCHUNKS: usize = 64;

/// Number of indices to take in the flat primitive benches.
const NUM_INDICES: &[usize] = &[10_000, 100_000, 1_000_000];

/// Elements per list in the fixed-size-list benches.
const LIST_SIZE: usize = 128;

/// Rows per chunk in the fixed-size-list benches.
const FSL_CHUNK_ROWS: usize = 2_048;

/// Number of chunks in the fixed-size-list benches.
const FSL_NCHUNKS: usize = 32;

/// Number of row indices to take in the fixed-size-list benches.
const FSL_NUM_INDICES: &[usize] = &[4_096, 32_768];

fn make_chunked_f32(chunk_len: usize, nchunks: usize) -> ArrayRef {
    (0..nchunks)
        .map(|c| {
            PrimitiveArray::from_iter((0..chunk_len).map(|i| (c * chunk_len + i) as f32))
                .into_array()
        })
        .collect::<ChunkedArray>()
        .into_array()
}

fn f16_elements(num_rows: usize) -> Buffer<f16> {
    (0..num_rows * LIST_SIZE)
        .map(|i| f16::from_f32((i % 1024) as f32))
        .collect()
}

/// `Chunked(FixedSizeList(f16))`: the shape produced by reading a fixed-size-list column
/// as row-group chunks. Take runs against the chunked array at row granularity.
fn make_chunked_of_fsl() -> ArrayRef {
    (0..FSL_NCHUNKS)
        .map(|_| {
            FixedSizeListArray::new(
                f16_elements(FSL_CHUNK_ROWS).into_array(),
                LIST_SIZE as u32,
                Validity::NonNullable,
                FSL_CHUNK_ROWS,
            )
            .into_array()
        })
        .collect::<ChunkedArray>()
        .into_array()
}

/// `FixedSizeList(Chunked(f16))`: the shape produced by canonicalizing
/// `Chunked(FixedSizeList)`, which swizzles chunking down into the elements child. Take
/// expands row indices into `LIST_SIZE` element indices before hitting the chunked child.
fn make_fsl_of_chunked() -> ArrayRef {
    let elements = (0..FSL_NCHUNKS)
        .map(|_| f16_elements(FSL_CHUNK_ROWS).into_array())
        .collect::<ChunkedArray>()
        .into_array();
    FixedSizeListArray::new(
        elements,
        LIST_SIZE as u32,
        Validity::NonNullable,
        FSL_NCHUNKS * FSL_CHUNK_ROWS,
    )
    .into_array()
}

fn shuffled_indices(num_indices: usize, max_index: usize) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(42);
    (0..num_indices)
        .map(|_| rng.random_range(0..max_index) as u64)
        .collect::<Buffer<u64>>()
        .into_array()
}

fn sorted_indices(num_indices: usize, max_index: usize) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(42);
    let mut indices: Vec<u64> = (0..num_indices)
        .map(|_| rng.random_range(0..max_index) as u64)
        .collect();
    indices.sort_unstable();
    Buffer::from(indices).into_array()
}

/// Strictly increasing indices with no duplicates (samples without replacement), the shape
/// that hits the sorted-unique fast path in `take_chunked`.
fn sorted_unique_indices(num_indices: usize, max_index: usize) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(42);
    let mut indices: Vec<u64> = sample(&mut rng, max_index, num_indices)
        .into_iter()
        .map(|i| i as u64)
        .collect();
    indices.sort_unstable();
    Buffer::from(indices).into_array()
}

fn take_to_builder(array: &ArrayRef, indices: &ArrayRef, session: &VortexSession) -> ArrayRef {
    let mut ctx = session.create_execution_ctx();
    let taken = array.take(indices.clone()).unwrap();
    let mut builder = builder_with_capacity(taken.dtype(), taken.len());
    taken.append_to_builder(builder.as_mut(), &mut ctx).unwrap();
    builder.finish()
}

#[divan::bench(args = NUM_INDICES)]
fn take_chunked_f32_to_builder_shuffled(bencher: Bencher, num_indices: usize) {
    let array = make_chunked_f32(CHUNK_LEN, NCHUNKS);
    let indices = shuffled_indices(num_indices, CHUNK_LEN * NCHUNKS);

    bencher
        .with_inputs(|| (&array, &indices))
        .bench_refs(|(array, indices)| take_to_builder(array, indices, &SESSION));
}

#[divan::bench(args = NUM_INDICES)]
fn take_chunked_f32_to_builder_sorted(bencher: Bencher, num_indices: usize) {
    let array = make_chunked_f32(CHUNK_LEN, NCHUNKS);
    let indices = sorted_indices(num_indices, CHUNK_LEN * NCHUNKS);

    bencher
        .with_inputs(|| (&array, &indices))
        .bench_refs(|(array, indices)| take_to_builder(array, indices, &SESSION));
}

#[divan::bench(args = NUM_INDICES)]
fn take_chunked_f32_to_builder_sorted_unique(bencher: Bencher, num_indices: usize) {
    let array = make_chunked_f32(CHUNK_LEN, NCHUNKS);
    let indices = sorted_unique_indices(num_indices, CHUNK_LEN * NCHUNKS);

    bencher
        .with_inputs(|| (&array, &indices))
        .bench_refs(|(array, indices)| take_to_builder(array, indices, &SESSION));
}

#[divan::bench(args = NUM_INDICES)]
fn take_chunked_f32_canonical_shuffled(bencher: Bencher, num_indices: usize) {
    let array = make_chunked_f32(CHUNK_LEN, NCHUNKS);
    let indices = shuffled_indices(num_indices, CHUNK_LEN * NCHUNKS);

    bencher
        .with_inputs(|| (&array, &indices, SESSION.create_execution_ctx()))
        .bench_refs(|(array, indices, ctx)| {
            array
                .take((*indices).clone())
                .unwrap()
                .execute::<Canonical>(ctx)
                .unwrap()
        });
}

#[divan::bench(args = FSL_NUM_INDICES)]
fn take_chunked_of_fsl_to_builder_shuffled(bencher: Bencher, num_indices: usize) {
    let array = make_chunked_of_fsl();
    let indices = shuffled_indices(num_indices, FSL_NCHUNKS * FSL_CHUNK_ROWS);

    bencher
        .with_inputs(|| (&array, &indices))
        .bench_refs(|(array, indices)| take_to_builder(array, indices, &SESSION));
}

#[divan::bench(args = FSL_NUM_INDICES)]
fn take_fsl_of_chunked_to_builder_shuffled(bencher: Bencher, num_indices: usize) {
    let array = make_fsl_of_chunked();
    let indices = shuffled_indices(num_indices, FSL_NCHUNKS * FSL_CHUNK_ROWS);

    bencher
        .with_inputs(|| (&array, &indices))
        .bench_refs(|(array, indices)| take_to_builder(array, indices, &SESSION));
}

#[divan::bench(args = FSL_NUM_INDICES)]
fn take_fsl_of_chunked_canonical_shuffled(bencher: Bencher, num_indices: usize) {
    let array = make_fsl_of_chunked();
    let indices = shuffled_indices(num_indices, FSL_NCHUNKS * FSL_CHUNK_ROWS);

    bencher
        .with_inputs(|| (&array, &indices, SESSION.create_execution_ctx()))
        .bench_refs(|(array, indices, ctx)| {
            array
                .take((*indices).clone())
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
                .unwrap()
        });
}
