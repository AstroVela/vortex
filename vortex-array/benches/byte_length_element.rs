// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks the two byte-string input elements a row function can ask for.
//!
//! `byte_length` only needs each row's length, which every view already stores. Asking for
//! [`BytesLen`] reads that field, while asking for [`Bytes`] instead resolves the row, which for a
//! non-inlined view means indexing the data buffers and building a slice per row. The `bytes_len`
//! arm is what `vortex.byte_length` does, and `bytes_slice` is the same function written over
//! [`Bytes`].
//!
//! Strings longer than 12 bytes are not inlined into the view, so `long_strings` is where the
//! difference shows up. `short_strings` stays inlined and should be close to even.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::dtype::DType;
use vortex_array::scalar_fn::Bytes;
use vortex_array::scalar_fn::BytesLen;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::RowFn;
use vortex_array::scalar_fn::RowVisitor;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

const SIZES: &[usize] = &[4096, 65536];

/// Byte length read from the view, as `vortex.byte_length` does.
#[derive(Clone)]
struct LenFromView;

impl RowFn for LenFromView {
    type Options = EmptyOptions;
    type ArgsWitness = (BytesLen,);
    type RetWitness = u64;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("bench.len_from_view");
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
        visitor.visit::<(BytesLen,), u64>(|(len,)| len as u64)
    }
}

/// Byte length read from the resolved row. Because `Bytes` is not dense-safe, this function is
/// filtered rather than dense, which follows from the argument type rather than from a choice here.
#[derive(Clone)]
struct LenFromSlice;

impl RowFn for LenFromSlice {
    type Options = EmptyOptions;
    type ArgsWitness = (Bytes,);
    type RetWitness = u64;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("bench.len_from_slice");
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
        visitor.visit::<(Bytes,), u64>(|(bytes,)| bytes.len() as u64)
    }
}

/// Non-inlined rows: every view points into a data buffer.
fn long_strings(len: usize) -> ArrayRef {
    VarBinViewArray::from_iter_str((0..len).map(|i| format!("a string well past inline: {i}")))
        .into_array()
}

/// Inlined rows: the bytes live in the view itself.
fn short_strings(len: usize) -> ArrayRef {
    VarBinViewArray::from_iter_str((0..len).map(|i| format!("{}", i % 1000))).into_array()
}

fn bench_len<F: RowFn<Options = EmptyOptions>>(bencher: Bencher, scalar_fn: F, input: ArrayRef) {
    let len = input.len();
    bencher
        .with_inputs(|| input.clone())
        .bench_local_values(|input| {
            let mut ctx = SESSION.create_execution_ctx();
            scalar_fn
                .clone()
                .try_new_array(len, EmptyOptions, [input])
                .unwrap()
                .execute::<Canonical>(&mut ctx)
                .unwrap()
        });
}

#[divan::bench(args = SIZES)]
fn long_strings_bytes_len(bencher: Bencher, len: usize) {
    bench_len(bencher, LenFromView, long_strings(len));
}

#[divan::bench(args = SIZES)]
fn long_strings_bytes_slice(bencher: Bencher, len: usize) {
    bench_len(bencher, LenFromSlice, long_strings(len));
}

#[divan::bench(args = SIZES)]
fn short_strings_bytes_len(bencher: Bencher, len: usize) {
    bench_len(bencher, LenFromView, short_strings(len));
}

#[divan::bench(args = SIZES)]
fn short_strings_bytes_slice(bencher: Bencher, len: usize) {
    bench_len(bencher, LenFromSlice, short_strings(len));
}
