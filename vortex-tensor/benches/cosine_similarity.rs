// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! What a batch-constant operand costs `cosine_similarity`'s row loop.
//!
//! The row closure computes the dot product and *both* norms per row. When one operand is a
//! broadcast query vector, its norm is the same in every row, so the closure re-pays an
//! `O(width)` pass and a `sqrt` per row for a value that is constant across the batch. The
//! `column_x_constant` arm measures that; `column_x_column` is the control where no hoist is
//! possible.
//!
//! The constant operand here is a [`ConstantArray`] whose scalar is a [`Vector`] extension
//! scalar, the shape a `lit(query_vector)` literal expression produces. That is the constant
//! representation that reaches the row loop, decoded once and read at stride 0. An extension
//! array whose *storage* is constant (`Vector::constant_array`) never gets there:
//! `CosineSimilarity::reduce_encoded` rewrites it into an `L2Denorm` and answers through the
//! inner-product path instead.

#![expect(clippy::unwrap_used)]

use divan::Bencher;
use vortex_array::ArrayRef;
use vortex_array::EmptyMetadata;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_tensor::scalar_fns::cosine_similarity::CosineSimilarity;
use vortex_tensor::vector::Vector;

fn main() {
    divan::main();
}

const ROWS: usize = 16_384;

/// Widths chosen to separate the two costs, as in `l2_norm.rs`: the redundant norm pass is
/// `O(rows * width)`, one third of the closure's arithmetic, so wide tensors show the hoist
/// while a narrow one is dominated by per-row framework costs.
const WIDTHS: &[usize] = &[2, 32, 256];

/// [`ROWS`] vectors of `width` `f64` elements, non-nullable. `seed` offsets the values so the
/// two sides of the column arm are not the same array.
fn vectors(width: usize, seed: usize) -> ArrayRef {
    let elements: Buffer<f64> = (0..ROWS * width)
        .map(|i| (((i + seed) % 97) as f64) - 48.0)
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

/// One query vector of `width` `f64` elements broadcast to [`ROWS`] rows, as a
/// [`ConstantArray`] over a [`Vector`] extension scalar.
fn constant_vector(width: usize) -> ArrayRef {
    let element_dtype = DType::Primitive(PType::F64, Nullability::NonNullable);
    let children: Vec<Scalar> = (0..width)
        .map(|i| Scalar::primitive(((i % 97) as f64) - 48.0, Nullability::NonNullable))
        .collect();
    let fsl_scalar = Scalar::fixed_size_list(element_dtype, children, Nullability::NonNullable);
    let ext_scalar = Scalar::extension::<Vector>(EmptyMetadata, fsl_scalar);
    ConstantArray::new(ext_scalar, ROWS).into_array()
}

fn bench_cosine(bencher: Bencher, lhs: ArrayRef, rhs: ArrayRef) {
    let session = vortex_array::array_session();
    let rows = lhs.len();
    bencher
        .with_inputs(|| {
            (
                CosineSimilarity
                    .try_new_array(rows, EmptyOptions, [lhs.clone(), rhs.clone()])
                    .unwrap(),
                session.create_execution_ctx(),
            )
        })
        .bench_values(|(array, mut ctx)| array.execute::<PrimitiveArray>(&mut ctx).unwrap());
}

/// The control: both operands vary by row, so every norm must be computed in the row loop.
#[divan::bench(args = WIDTHS)]
fn column_x_column(bencher: Bencher, width: usize) {
    bench_cosine(bencher, vectors(width, 0), vectors(width, 31));
}

/// The rhs is a broadcast query vector, whose norm is the same in every row.
#[divan::bench(args = WIDTHS)]
fn column_x_constant(bencher: Bencher, width: usize) {
    bench_cosine(bencher, vectors(width, 0), constant_vector(width));
}

/// One query vector as an extension array over constant storage, the other spelling of a
/// broadcast operand (what `Vector::constant_array` builds before `ExtensionConstantRule`
/// normalizes it).
fn ext_constant_vector(width: usize) -> ArrayRef {
    let ext_dtype = vectors(width, 0).dtype().as_extension().clone();
    let element_dtype = DType::Primitive(PType::F64, Nullability::NonNullable);
    let children: Vec<Scalar> = (0..width)
        .map(|i| Scalar::primitive(((i % 97) as f64) - 48.0, Nullability::NonNullable))
        .collect();
    let fsl_scalar = Scalar::fixed_size_list(element_dtype, children, Nullability::NonNullable);
    ExtensionArray::new(ext_dtype, ConstantArray::new(fsl_scalar, ROWS).into_array()).into_array()
}

/// The rhs is the same broadcast query as `column_x_constant`, spelled as an extension array over
/// constant storage rather than a top-level constant.
#[divan::bench(args = WIDTHS)]
fn column_x_ext_constant(bencher: Bencher, width: usize) {
    bench_cosine(bencher, vectors(width, 0), ext_constant_vector(width));
}
