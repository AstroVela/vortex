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
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::MaskedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::expr::Expression;
use vortex_array::expr::union_child_validities;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;
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

/// The kernel as it was written before the port, for a like-for-like comparison.
///
/// It differs from [`L2Norm`] in exactly the two places the framework changed: the row loop indexes
/// the flat slice directly and collects into a [`Buffer`] rather than reading through the framework's
/// per-argument stride into a `Vec`, and it attaches the input's validity to that buffer in one step
/// rather than building non-nullable and masking afterwards. `PrimitiveArray::new` stands in for the
/// original `new_unchecked`, differing only by an `O(1)` length check.
///
/// Both sides reach the same decode, so the gap between them is the framework's own per-row and
/// output-construction cost rather than anything about canonicalization.
#[derive(Clone)]
struct PrePortL2Norm;

impl ScalarFnVTable for PrePortL2Norm {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("bench.pre_port_l2_norm");
        *ID
    }

    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        Ok(EmptyOptions)
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, _child_idx: usize) -> ChildName {
        ChildName::from("input")
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        Ok(DType::Primitive(PType::F64, arg_dtypes[0].nullability()))
    }

    fn execute(
        &self,
        _options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;
        let rows = args.row_count();
        let validity = input.validity()?;

        let ext: ExtensionArray = input.execute(ctx)?;
        let fsl: FixedSizeListArray = ext.storage_array().clone().execute(ctx)?;
        let width = fsl.list_size() as usize;
        let elements: PrimitiveArray = fsl.elements().clone().execute(ctx)?;
        let flat = elements.as_slice::<f64>();

        let norms: Buffer<f64> = (0..rows)
            .map(|i| l2_norm_row(&flat[i * width..(i + 1) * width]))
            .collect();

        Ok(PrimitiveArray::new(norms, validity).into_array())
    }

    fn validity(
        &self,
        _options: &Self::Options,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        union_child_validities(expression)
    }

    fn is_strict(&self, _options: &Self::Options) -> bool {
        true
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        false
    }
}

/// The same arithmetic the production kernel does, kept here so the comparison is of the machinery
/// around it rather than of two different formulas.
fn l2_norm_row(row: &[f64]) -> f64 {
    row.iter().fold(0.0f64, |acc, &x| acc + x * x).sqrt()
}

fn bench_pre_port(bencher: Bencher, array: ArrayRef) {
    let session = vortex_array::array_session();
    let rows = array.len();
    bencher
        .with_inputs(|| {
            (
                PrePortL2Norm
                    .try_new_array(rows, EmptyOptions, [array.clone()])
                    .unwrap(),
                session.create_execution_ctx(),
            )
        })
        .bench_values(|(array, mut ctx)| array.execute::<PrimitiveArray>(&mut ctx).unwrap());
}

#[divan::bench(args = WIDTHS)]
fn pre_port_non_nullable(bencher: Bencher, width: usize) {
    bench_pre_port(bencher, vectors(width));
}

#[divan::bench(args = WIDTHS)]
fn pre_port_nullable(bencher: Bencher, width: usize) {
    let array = vectors(width);
    let validity = Validity::from_iter((0..ROWS).map(|i| i % 8 != 0));
    bench_pre_port(
        bencher,
        MaskedArray::try_new(array, validity).unwrap().into_array(),
    );
}
