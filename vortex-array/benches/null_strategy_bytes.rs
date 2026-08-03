// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Null-strategy comparison for a *cheap* kernel: `byte_length` visited at the [`Bytes`] element
//! (which is not dense-safe, so it executes under `NullHandling::Filter`).
//!
//! Arms: `filter` and `branch` force one strategy through the test-harness seam
//! ([`execute_row_fn_with_strategy`]); `auto` executes the full pipeline and lets the per-batch
//! selection choose. `Bytes` decodes in bulk, so the selection should track the `branch` arm at
//! every mixed density. Null densities run 0/1/5/10/25/50/90 percent over 65536 rows of
//! non-inlined strings, nulls placed by a seeded splitmix hash.
//!
//! Run with `cargo bench -p vortex-array --bench null_strategy_bytes`.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::dtype::DType;
use vortex_array::scalar_fn::Bytes;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::ElementSink;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::NullStrategy;
use vortex_array::scalar_fn::RowFn;
use vortex_array::scalar_fn::RowVisitor;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::execute_row_fn_with_strategy;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

const ROWS: usize = 65536;

/// Null densities in percent.
const DENSITIES: &[usize] = &[0, 1, 5, 10, 25, 50, 90];

/// splitmix64, for seeded random null placement.
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

/// Byte length read from the resolved row, exactly `byte_length_element.rs`'s `LenFromSlice`:
/// because [`Bytes`] is not dense-safe this function is filtered rather than dense.
#[derive(Clone)]
struct LenFromSlice;

impl RowFn for LenFromSlice {
    type Options = EmptyOptions;
    type ArgsWitness = (Bytes,);

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("bench.null_strategy.len_from_slice");
        *ID
    }

    fn arg_name(&self, _idx: usize) -> ChildName {
        ChildName::from("input")
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::Out> {
        visitor.visit_prepared_into::<(Bytes,), ElementSink<u64>, _, _>(
            |_| (),
            |&(), (bytes,), output| output.write(bytes.len() as u64),
        )
    }
}

/// Non-inlined rows (every view resolves into a data buffer) with seeded nulls at `density`
/// percent. Zero density builds a non-nullable column.
fn long_strings(density: usize) -> ArrayRef {
    if density == 0 {
        return VarBinViewArray::from_iter_str(
            (0..ROWS).map(|i| format!("a string well past inline: {i}")),
        )
        .into_array();
    }

    VarBinViewArray::from_iter_nullable_str((0..ROWS).map(|i| {
        ((splitmix64(1 ^ i as u64) % 100) >= density as u64)
            .then(|| format!("a string well past inline: {i}"))
    }))
    .into_array()
}

/// One arm: `Some` forces a strategy through the harness seam, `None` runs the full pipeline
/// with the per-batch selection.
fn bench_len(bencher: Bencher, density: usize, strategy: Option<NullStrategy>) {
    let mut ctx = SESSION.create_execution_ctx();
    let input = long_strings(density);

    bencher
        .counter(ItemsCount::new(ROWS))
        .bench_local(|| match strategy {
            None => LenFromSlice
                .try_new_array(ROWS, EmptyOptions, [input.clone()])
                .unwrap()
                .execute::<Canonical>(&mut ctx)
                .unwrap(),
            Some(strategy) => execute_row_fn_with_strategy(
                &LenFromSlice,
                &EmptyOptions,
                vec![input.clone()],
                ROWS,
                strategy,
                &mut ctx,
            )
            .unwrap()
            .execute::<Canonical>(&mut ctx)
            .unwrap(),
        });
}

#[divan::bench(args = DENSITIES)]
fn filter(bencher: Bencher, density: usize) {
    bench_len(bencher, density, Some(NullStrategy::Filter));
}

#[divan::bench(args = DENSITIES)]
fn branch(bencher: Bencher, density: usize) {
    bench_len(bencher, density, Some(NullStrategy::BranchAndSkip));
}

#[divan::bench(args = DENSITIES)]
fn auto(bencher: Bencher, density: usize) {
    bench_len(bencher, density, None);
}
