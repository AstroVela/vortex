// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Measures the composable chunked-decompression vtable path (`decompress_chunks`) for
//! FoR-over-BitPacked against:
//!
//! - `two_pass`: full fused decompression to a `PrimitiveArray` followed by a second pass over
//!   the materialized buffer (the pattern `decompress_chunks` is designed to replace).
//! - `hand_fused`: a monomorphized loop over `BitUnpackedChunks` with the reference folded in
//!   algebraically — the no-dispatch upper bound, equivalent to the fused `unpack_map` kernels.
//!
//! The gap between `chunked_vtable` and `hand_fused` is the exact price of the generic
//! mechanism: one virtual call per 1024-element chunk per encoding level plus one in-place
//! add pass over the L1-resident chunk at the FoR level.

// The `#[gat(Item)]` import expansion trips unused_imports even though the lending iterator
// machinery requires it.
#![allow(unused_imports)]

use std::hint::black_box;
use std::ops::Range;
use std::sync::LazyLock;

use divan::Bencher;
use lending_iterator::gat;
use lending_iterator::prelude::Item;
#[gat(Item)]
use lending_iterator::prelude::LendingIterator;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::chunk_iter::ChunkMut;
use vortex_array::chunk_iter::ChunkSink;
use vortex_array::scalar::Scalar;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_fastlanes::BitPackedArrayExt;
use vortex_fastlanes::FoR;
use vortex_fastlanes::FoRArraySlotsExt;
use vortex_fastlanes::bitpack_compress::bitpack_encode;
use vortex_fastlanes::unpack_iter::BitUnpackedChunks;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_fastlanes::initialize(&session);
    session
});

const LEN: usize = 4 * 1024 * 1024;
const REFERENCE: u32 = 1_000_000;

/// FoR(reference=1M) over BitPacked(bit_width=10), no patches.
fn make_for_bitpacked() -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let deltas = PrimitiveArray::from_iter(
        (0..LEN as u64).map(|i| u32::try_from((i * 7) % 1000).vortex_expect("fits")),
    );
    let bp = bitpack_encode(&deltas, 10, None, &mut ctx).vortex_expect("bench");
    FoR::try_new(bp.into_array(), Scalar::from(REFERENCE))
        .vortex_expect("bench")
        .into_array()
}

struct SumSink {
    total: u64,
}

impl ChunkSink for SumSink {
    #[inline]
    fn accept(&mut self, chunk: ChunkMut<'_>, _row_range: Range<usize>) -> VortexResult<()> {
        self.total = self
            .total
            .wrapping_add(chunk.as_slice::<u32>().iter().map(|&v| v as u64).sum());
        Ok(())
    }
}

/// Sum through the composable vtable path: BitPacked streams unpacked blocks, FoR shifts each
/// block in place, the sink folds it — one streaming pass, no materialization.
#[divan::bench]
fn chunked_vtable_sum(bencher: Bencher) {
    let array = make_for_bitpacked();
    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| {
            let mut sink = SumSink { total: 0 };
            array
                .decompress_chunks(&mut ctx, &mut sink)
                .vortex_expect("bench");
            black_box(sink.total)
        });
}

/// Sum by fully decompressing (the fused FoR+BitPacked execute path) and then re-reading the
/// materialized array: two passes over `LEN` values.
#[divan::bench]
fn two_pass_sum(bencher: Bencher) {
    let array = make_for_bitpacked();
    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| {
            let primitive = array
                .execute::<PrimitiveArray>(&mut ctx)
                .vortex_expect("bench");
            let total: u64 = primitive
                .as_slice::<u32>()
                .iter()
                .map(|&v| v as u64)
                .fold(0, u64::wrapping_add);
            black_box(total)
        });
}

/// Monomorphized upper bound: iterate the BitPacked child's unpacked blocks directly and fold
/// the FoR reference in algebraically — no dynamic dispatch, no extra in-place pass.
#[divan::bench]
fn hand_fused_sum(bencher: Bencher) {
    let array = make_for_bitpacked();
    bencher.with_inputs(|| array.clone()).bench_values(|array| {
        let for_ = array.as_::<FoR>();
        let bp = for_.encoded().as_::<vortex_fastlanes::BitPacked>();
        let mut chunks: BitUnpackedChunks<u32> = bp.unpacked_chunks().vortex_expect("bench");
        let mut total = 0u64;
        let mut count = 0u64;
        if let Some(initial) = chunks.initial() {
            total = total.wrapping_add(initial.iter().map(|&v| v as u64).sum());
            count += initial.len() as u64;
        }
        let mut iter = chunks.full_chunks();
        while let Some(chunk) = iter.next() {
            total = total.wrapping_add(chunk.iter().map(|&v| v as u64).sum());
            count += chunk.len() as u64;
        }
        if let Some(trailer) = chunks.trailer() {
            total = total.wrapping_add(trailer.iter().map(|&v| v as u64).sum());
            count += trailer.len() as u64;
        }
        black_box(total.wrapping_add(count * REFERENCE as u64))
    });
}

/// Monomorphized version of exactly what the vtable path does per chunk (unpack, in-place
/// reference add, then fold) but with zero dynamic dispatch. The gap between this and
/// `hand_fused_sum` is the cost of the extra in-place pass; the gap between this and
/// `chunked_vtable_sum` is the pure dynamic-dispatch cost (two virtual calls per 1024 values).
#[divan::bench]
fn hand_chunked_add_pass_sum(bencher: Bencher) {
    let array = make_for_bitpacked();
    bencher.with_inputs(|| array.clone()).bench_values(|array| {
        let for_ = array.as_::<FoR>();
        let bp = for_.encoded().as_::<vortex_fastlanes::BitPacked>();
        let mut chunks: BitUnpackedChunks<u32> = bp.unpacked_chunks().vortex_expect("bench");
        let mut total = 0u64;
        let mut fold = |chunk: &mut [u32]| {
            for v in chunk.iter_mut() {
                *v = v.wrapping_add(REFERENCE);
            }
            total = total.wrapping_add(chunk.iter().map(|&v| v as u64).sum());
        };
        if let Some(initial) = chunks.initial() {
            fold(initial);
        }
        let mut iter = chunks.full_chunks();
        while let Some(chunk) = iter.next() {
            fold(chunk);
        }
        if let Some(trailer) = chunks.trailer() {
            fold(trailer);
        }
        black_box(total)
    });
}

struct WriteSink<'a> {
    out: &'a mut Vec<u32>,
}

impl ChunkSink for WriteSink<'_> {
    #[inline]
    fn accept(&mut self, chunk: ChunkMut<'_>, _row_range: Range<usize>) -> VortexResult<()> {
        self.out.extend_from_slice(chunk.as_slice::<u32>());
        Ok(())
    }
}

/// Materialize through the chunked path (unpack to scratch, add reference in place, copy out) —
/// upper-bounds the cost of the non-fused composition when the consumer wants a full buffer.
#[divan::bench]
fn chunked_vtable_decompress_into(bencher: Bencher) {
    let array = make_for_bitpacked();
    bencher
        .with_inputs(|| {
            (
                array.clone(),
                SESSION.create_execution_ctx(),
                Vec::<u32>::with_capacity(LEN),
            )
        })
        .bench_values(|(array, mut ctx, mut out)| {
            let mut sink = WriteSink { out: &mut out };
            array
                .decompress_chunks(&mut ctx, &mut sink)
                .vortex_expect("bench");
            black_box(out.len())
        });
}

/// Materialize through the fused execute path (`decode_into` writes unpacked+shifted values
/// straight into the output buffer) — the specialized baseline for full decompression.
#[divan::bench]
fn fused_decompress(bencher: Bencher) {
    let array = make_for_bitpacked();
    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| {
            let primitive = array
                .execute::<PrimitiveArray>(&mut ctx)
                .vortex_expect("bench");
            black_box(primitive.len())
        });
}
