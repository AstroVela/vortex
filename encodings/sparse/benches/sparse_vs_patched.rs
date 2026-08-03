// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compares the two representations of "a constant fill plus a few patched values":
//!
//! * `Sparse`, which executes by filling a buffer with the fill value and then scattering the
//!   patches in index order.
//! * `Patched(Constant, patches)`, produced by `SparsePatchedPlugin` at deserialize time, which
//!   executes by filling a buffer from the inner `Constant` and then scattering the patches from
//!   the lane-transposed layout.
//!
//! `transpose` measures the one-off conversion cost the plugin pays at deserialize time, so the
//! execute numbers can be read against how many executions amortize it.

#![expect(clippy::cast_possible_truncation)]

use std::sync::LazyLock;

use divan::Bencher;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Patched;
use vortex_array::patches::Patches;
use vortex_array::scalar::Scalar;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_session::VortexSession;
use vortex_sparse::Sparse;

fn main() {
    divan::main();
}

/// `(len, num_patches)` pairs spanning very sparse (0.1%) to patch-dominated (10%) arrays, at
/// two lengths so the fill cost can be separated from the per-patch scatter cost.
const ARGS: &[(usize, usize)] = &[
    (100_000, 100),
    (100_000, 1_000),
    (100_000, 10_000),
    (1_000_000, 1_000),
    (1_000_000, 10_000),
    (1_000_000, 100_000),
];

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_sparse::initialize(&session);
    session
});

/// Uniformly-spaced patch indices and values over an array of `len` `i32`s.
fn make_patches(len: usize, num_patches: usize) -> Patches {
    let stride = len / num_patches;
    let indices: Buffer<u32> = (0..num_patches).map(|i| (i * stride) as u32).collect();
    let values: Buffer<i32> = (0..num_patches).map(|i| 2 + i as i32).collect();
    Patches::new(len, 0, indices.into_array(), values.into_array(), None)
        .vortex_expect("valid patches")
}

fn make_sparse(len: usize, num_patches: usize) -> ArrayRef {
    Sparse::try_new_from_patches(make_patches(len, num_patches), Scalar::from(1i32))
        .vortex_expect("valid sparse")
        .into_array()
}

fn make_patched(len: usize, num_patches: usize) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let inner = ConstantArray::new(Scalar::from(1i32), len).into_array();
    Patched::from_array_and_patches(inner, &make_patches(len, num_patches), &mut ctx)
        .vortex_expect("valid patched")
        .into_array()
}

#[divan::bench(args = ARGS)]
fn sparse_execute(bencher: Bencher, (len, num_patches): (usize, usize)) {
    let sparse = make_sparse(len, num_patches);

    bencher
        .with_inputs(|| (sparse.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| {
            divan::black_box(
                array
                    .execute::<Canonical>(&mut ctx)
                    .vortex_expect("execute"),
            )
        });
}

#[divan::bench(args = ARGS)]
fn patched_execute(bencher: Bencher, (len, num_patches): (usize, usize)) {
    let patched = make_patched(len, num_patches);

    bencher
        .with_inputs(|| (patched.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| {
            divan::black_box(
                array
                    .execute::<Canonical>(&mut ctx)
                    .vortex_expect("execute"),
            )
        });
}

/// The one-off `Sparse` -> `Patched` conversion performed by `SparsePatchedPlugin::deserialize`.
#[divan::bench(args = ARGS)]
fn transpose(bencher: Bencher, (len, num_patches): (usize, usize)) {
    let patches = make_patches(len, num_patches);

    bencher
        .with_inputs(|| {
            (
                ConstantArray::new(Scalar::from(1i32), len).into_array(),
                patches.clone(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(inner, patches, mut ctx)| {
            divan::black_box(
                Patched::from_array_and_patches(inner, &patches, &mut ctx).vortex_expect("patched"),
            )
        });
}
