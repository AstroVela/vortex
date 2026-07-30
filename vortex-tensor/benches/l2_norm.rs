// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! What the row-function framework costs `l2_norm` relative to a hand-written kernel.
//!
//! Before the port, the kernel built its values buffer and paired it with the input's validity in one
//! step (`PrimitiveArray::new_unchecked(buffer, validity)`), so a nullable input cost it nothing extra.
//! The framework cannot do that: [`OutputElement::build`] returns a *non-nullable* column and the
//! strict lifting applies validity afterwards, which for `Validity::Array` means materializing a mask
//! and running a separate `mask` pass over the result.
//!
//! The non-nullable case is therefore the like-for-like comparison against the old kernel, and the
//! gap between the two cases below is what the framework adds on nullable input.

#![expect(clippy::unwrap_used)]

use divan::Bencher;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::MaskedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_tensor::scalar_fns::l2_norm::L2Norm;
use vortex_tensor::vector::Vector;

fn main() {
    divan::main();
}

const ROWS: usize = 16_384;

/// Widths chosen to separate the two costs. The kernel is `O(rows * width)` while the framework's
/// per-row and masking costs are `O(rows)`, so a wide tensor amortizes the framework away and only a
/// narrow one exposes it.
const WIDTHS: &[usize] = &[2, 32, 256];

/// [`ROWS`] vectors of `width` `f64` elements, non-nullable.
fn vectors(width: usize) -> ArrayRef {
    let elements: Buffer<f64> = (0..ROWS * width)
        .map(|i| ((i % 97) as f64) - 48.0)
        .collect();
    let storage = FixedSizeListArray::new(
        elements.into_array(),
        u32::try_from(width).unwrap(),
        Validity::NonNullable,
        ROWS,
    )
    .into_array();
    Vector::try_new_vector_array(storage).unwrap()
}

fn bench_l2_norm(bencher: Bencher, array: ArrayRef) {
    let session = vortex_array::array_session();
    let rows = array.len();
    bencher
        .with_inputs(|| {
            (
                L2Norm
                    .try_new_array(rows, EmptyOptions, [array.clone()])
                    .unwrap(),
                session.create_execution_ctx(),
            )
        })
        .bench_values(|(array, mut ctx)| array.execute::<PrimitiveArray>(&mut ctx).unwrap());
}

/// Like for like against the pre-port kernel: no validity to apply, so the lifting adds no pass.
#[divan::bench(args = WIDTHS)]
fn non_nullable(bencher: Bencher, width: usize) {
    bench_l2_norm(bencher, vectors(width));
}

/// One row in eight null, so the conjoined validity is a `Validity::Array` and the lifting
/// materializes a mask and applies it to the row loop's output.
#[divan::bench(args = WIDTHS)]
fn nullable(bencher: Bencher, width: usize) {
    let array = vectors(width);
    let validity = Validity::from_iter((0..ROWS).map(|i| i % 8 != 0));
    bench_l2_norm(
        bencher,
        MaskedArray::try_new(array, validity).unwrap().into_array(),
    );
}
