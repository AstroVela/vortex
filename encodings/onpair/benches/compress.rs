// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! Compress-path microbenchmarks for the OnPair Vortex array.
//!
//! `compress` covers the full [`onpair_compress`] entry point: gather the rows
//! the trainer needs, train + encode upstream, then wrap the result as a Vortex
//! array. `compress_nullable` runs the same shapes with 10 % nulls, which takes
//! a different gather loop.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::panic,
    clippy::tests_outside_test_module,
    clippy::unwrap_used,
    clippy::expect_used
)]

use std::sync::LazyLock;

use divan::Bencher;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::VarBinViewArray;
use vortex_onpair::DEFAULT_CONFIG;
use vortex_onpair::onpair_compress;
use vortex_session::VortexSession;

mod shared;

use shared::Shape;
use shared::corpus;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = array_session();
    vortex_onpair::initialize(&session);
    session
});

/// Build the `VarBinViewArray` the compressor is handed. `null_every` of 0
/// keeps every row valid.
fn input(n: usize, shape: Shape, null_every: usize) -> ArrayRef {
    let strings = corpus(n, shape);
    if null_every == 0 {
        VarBinViewArray::from_iter_str(strings).into_array()
    } else {
        VarBinViewArray::from_iter_nullable_str(
            strings
                .into_iter()
                .enumerate()
                .map(|(i, s)| (i % null_every != 0).then_some(s)),
        )
        .into_array()
    }
}

/// Total input bytes for a case, so divan can report throughput.
fn input_bytes(array: &ArrayRef, ctx: &mut ExecutionCtx) -> u64 {
    let view = array.clone().execute::<VarBinViewArray>(ctx).unwrap();
    view.views().iter().map(|v| v.len() as u64).sum()
}

const CASES: &[(Shape, usize)] = &[
    (Shape::UrlLog, 100_000),
    (Shape::UrlLog, 1_000_000),
    (Shape::Short, 1_000_000),
    (Shape::Long, 100_000),
    (Shape::HighCard, 100_000),
];

/// End-to-end `onpair_compress`: gather, upstream train + encode, and the
/// Vortex array wrapping (buffer alignment, offset children, metadata).
#[divan::bench(args = CASES)]
fn compress(bencher: Bencher, case: (Shape, usize)) {
    let mut ctx = SESSION.create_execution_ctx();
    let (shape, n) = case;
    let array = input(n, shape, 0);
    let bytes = input_bytes(&array, &mut ctx);
    bencher
        .counter(divan::counter::BytesCount::new(bytes))
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_local_values(|mut ctx| {
            divan::black_box(
                onpair_compress(&array, DEFAULT_CONFIG, &mut ctx)
                    .unwrap_or_else(|e| panic!("onpair_compress failed: {e}")),
            )
        });
}

/// Same, with 10 % nulls: exercises the `AllOr::Some` gather loop.
#[divan::bench(args = CASES)]
fn compress_nullable(bencher: Bencher, case: (Shape, usize)) {
    let mut ctx = SESSION.create_execution_ctx();
    let (shape, n) = case;
    let array = input(n, shape, 10);
    let bytes = input_bytes(&array, &mut ctx);
    bencher
        .counter(divan::counter::BytesCount::new(bytes))
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_local_values(|mut ctx| {
            divan::black_box(
                onpair_compress(&array, DEFAULT_CONFIG, &mut ctx)
                    .unwrap_or_else(|e| panic!("onpair_compress failed: {e}")),
            )
        });
}

fn main() {
    divan::main();
}
