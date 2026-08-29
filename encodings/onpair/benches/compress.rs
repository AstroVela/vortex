// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! Compress-path benchmarks for the OnPair Vortex array.
//!
//! `onpair_compress` flattens a [`VarBinViewArray`] into the contiguous
//! `(bytes, offsets)` pair `onpair` trains on, compresses, and lowers the result
//! back into Vortex children. The upstream `onpair` benches cover the codec
//! itself; this covers the Vortex-side work wrapped around it — the flatten, the
//! offset-width choice, and the child lowering.
//!
//! Shapes mirror `decode.rs` so the two paths are read against the same corpora.

#![allow(
    clippy::cast_possible_truncation,
    clippy::panic,
    clippy::tests_outside_test_module,
    clippy::unwrap_used,
    clippy::expect_used
)]

use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::BytesCount;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::VarBinViewArray;
use vortex_onpair::DEFAULT_CONFIG;
use vortex_onpair::onpair_compress;
use vortex_session::VortexSession;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_onpair::initialize(&session);
    session
});

#[derive(Copy, Clone, Debug)]
enum Shape {
    /// URL / HTTP-log shaped — high lexical overlap, ~35-45 bytes per row.
    UrlLog,
    /// Short uniform strings — 4-8 bytes per row. Below the 12-byte inline
    /// threshold, so every view is inlined and the flatten cannot be elided.
    Short,
    /// Long log-line shaped — ~120 bytes per row, more tokens per row.
    Long,
    /// High cardinality — every row unique.
    HighCard,
}

fn corpus(n: usize, shape: Shape) -> Vec<String> {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state
    };
    let mut out = Vec::with_capacity(n);
    match shape {
        Shape::UrlLog => {
            let templates: &[&str] = &[
                "https://www.example.com/products/{id}",
                "https://cdn.example.com/img/{id}.webp",
                "https://api.example.com/v2/orders/{id}",
                "INFO  request_id={id} status=200 method=GET",
                "ERROR request_id={id} status=500 method=PUT",
            ];
            for _ in 0..n {
                let s = next();
                out.push(
                    templates[(s as usize) % templates.len()]
                        .replace("{id}", &format!("{:08x}", s as u32)),
                );
            }
        }
        Shape::Short => {
            let words: &[&str] = &["alpha", "beta", "gamma", "delta", "eps"];
            for _ in 0..n {
                out.push(words[(next() as usize) % words.len()].to_string());
            }
        }
        Shape::Long => {
            for _ in 0..n {
                let s = next();
                out.push(format!(
                    "2026-08-29T12:00:00Z host=worker-{:03} svc=ingest span={:016x} \
                     msg=\"batch committed to the write-ahead log\" rows={} bytes={}",
                    s as u16 % 512,
                    s,
                    s as u16,
                    s as u32
                ));
            }
        }
        Shape::HighCard => {
            for _ in 0..n {
                out.push(format!("{:032x}", next() as u128 * 0x9e37_79b9));
            }
        }
    }
    out
}

const CASES: &[(Shape, usize)] = &[
    (Shape::UrlLog, 100_000),
    (Shape::UrlLog, 1_000_000),
    (Shape::Short, 100_000),
    (Shape::Long, 100_000),
    (Shape::HighCard, 100_000),
];

/// Full Vortex compress path: flatten, train, compress, lower to children.
#[divan::bench(args = CASES)]
fn compress(bencher: Bencher, case: (Shape, usize)) {
    let (shape, n) = case;
    let values = corpus(n, shape);
    let total_bytes: usize = values.iter().map(String::len).sum();
    let array = VarBinViewArray::from_iter_str(values).into_array();

    bencher
        .counter(BytesCount::new(total_bytes))
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_local_values(|mut ctx| {
            divan::black_box(
                onpair_compress(&array, DEFAULT_CONFIG, &mut ctx)
                    .unwrap_or_else(|e| panic!("compress failed: {e}")),
            )
        });
}

fn main() {
    divan::main();
}
