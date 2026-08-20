// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Baseline throughput for `l2_norm` over tensor columns.
//!
//! The arms vary vector width and input nullability. Their names are intended to remain stable
//! across scalar-function implementation changes so CodSpeed can compare them against `develop`.
//!
//! Rows are derived from a fixed element budget rather than fixed per arm, so widening a vector
//! trades rows for elements instead of multiplying the work. See [`ELEMENTS`].

#![expect(clippy::unwrap_used)]

use divan::Bencher;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::MaskedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_tensor::encodings::normalized::Normalized;
use vortex_tensor::scalar_fns::NormMode;
use vortex_tensor::scalar_fns::l2_norm::L2Norm;
use vortex_tensor::vector::Vector;

// Scalar function execution allocates its output inside the timed region, so use the vendored
// allocator instead of measuring glibc differences between CodSpeed runner images.
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

/// Total `f64` elements per operand, held constant across widths: the row count is
/// `ELEMENTS / width`. CodSpeed's CPU simulation charges memory traffic far more than a desktop
/// does, so this budget is what keeps every arm inside the 1 ms per-iteration limit from
/// `docs/developer-guide/benchmarking.md`.
const ELEMENTS: usize = 16_384;

const WIDTHS: &[usize] = &[2, 32, 256];

fn vectors(width: usize) -> ArrayRef {
    let elements: Buffer<f64> = (0..ELEMENTS).map(|i| ((i % 97) as f64) - 48.0).collect();
    let storage = FixedSizeListArray::new(
        elements.into_array(),
        u32::try_from(width).unwrap(),
        Validity::NonNullable,
        ELEMENTS / width,
    )
    .into_array();
    Vector::try_new_vector_array(storage).unwrap()
}

fn normalized_vectors(width: usize) -> ArrayRef {
    let row_count = ELEMENTS / width;
    let elements: Buffer<f64> = (0..ELEMENTS)
        .map(|i| if i % width == 0 { 1.0 } else { 0.0 })
        .collect();
    let storage = FixedSizeListArray::new(
        elements.into_array(),
        u32::try_from(width).unwrap(),
        Validity::NonNullable,
        row_count,
    )
    .into_array();
    let direction = Vector::try_new_vector_array(storage).unwrap();
    let norms = PrimitiveArray::from_iter((1..=row_count).map(|norm| norm as f64)).into_array();

    // SAFETY: Every direction row is unit length, and the non-negative norms have matching length.
    unsafe { Normalized::new_unchecked(direction, norms, Validity::NonNullable) }.into_array()
}

fn bench_l2_norm(bencher: Bencher, input: ArrayRef, mode: NormMode) {
    let session = vortex_array::array_session();
    bencher
        .counter(ItemsCount::new(input.len()))
        .with_inputs(|| {
            (
                L2Norm::try_new(input.clone(), mode).unwrap().into_array(),
                session.create_execution_ctx(),
            )
        })
        .bench_values(|(array, mut ctx)| array.execute::<PrimitiveArray>(&mut ctx).unwrap());
}

#[divan::bench(args = WIDTHS)]
fn non_nullable(bencher: Bencher, width: usize) {
    bench_l2_norm(bencher, vectors(width), NormMode::Exact);
}

#[divan::bench(args = WIDTHS)]
fn nullable(bencher: Bencher, width: usize) {
    let validity = Validity::from_iter((0..ELEMENTS / width).map(|i| i % 8 != 0));
    let input = MaskedArray::try_new(vectors(width), validity)
        .unwrap()
        .into_array();
    bench_l2_norm(bencher, input, NormMode::Exact);
}

/// Measures the physical norm of a [`Normalized`] input.
#[divan::bench(args = WIDTHS)]
fn normalized_exact(bencher: Bencher, width: usize) {
    bench_l2_norm(bencher, normalized_vectors(width), NormMode::Exact);
}

/// Reads the stored norm while trusting the normalized-direction claim.
#[divan::bench(args = WIDTHS)]
fn normalized_assume(bencher: Bencher, width: usize) {
    bench_l2_norm(
        bencher,
        normalized_vectors(width),
        NormMode::AssumeNormalized,
    );
}
