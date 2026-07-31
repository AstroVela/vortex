// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! What the [`OutputSink`] row layer costs `l2_denorm` relative to its hand-written kernel.
//!
//! The two kernels differ in exactly one way. The pre-port one builds its output by collecting a
//! `flat_map` over rows into a fresh [`Buffer`], never touching an element twice. The sink allocates a
//! zeroed buffer once and each row writes its own slice of it in place, which trades that iterator
//! chain for indexed writes into memory the allocator already zeroed. This measures whether the trade
//! is free.
//!
//! The pre-port side is a full [`ScalarFnVTable`](vortex_array::scalar_fn::ScalarFnVTable) and so
//! hand-writes the null handling that the row lifting derives for the ported one, but it hand-writes
//! the same two steps the lifting takes, so both sides pay them.
//!
//! Norms are a real column rather than a constant, since a constant norm is answered by
//! `reduce_encoded` and never reaches either row loop.

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
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
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
use vortex_session::registry::CachedId;
use vortex_tensor::scalar_fns::l2_denorm::L2Denorm;
use vortex_tensor::vector::Vector;

fn main() {
    divan::main();
}

const ROWS: usize = 16_384;

/// Widths chosen to separate the two costs, as in `l2_norm.rs`: the kernel is `O(rows * width)` while
/// any per-row framework cost is `O(rows)`, so only a narrow tensor exposes the latter.
const WIDTHS: &[usize] = &[2, 32, 256];

/// [`ROWS`] unit-norm-ish vectors of `width` `f64` elements, non-nullable.
///
/// The rows do not need to be exactly unit-norm: both kernels multiply through regardless, and the
/// scalar function array is built through the low-level factory, which does not check the invariant.
fn normalized(width: usize) -> ArrayRef {
    let elements: Buffer<f64> = (0..ROWS * width)
        .map(|i| ((i % 97) as f64) / 97.0)
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

/// One norm per row, deliberately not constant so the row loop actually runs.
fn norms() -> ArrayRef {
    let values: Buffer<f64> = (0..ROWS).map(|i| 1.0 + ((i % 13) as f64) / 13.0).collect();
    PrimitiveArray::new(values, Validity::NonNullable).into_array()
}

fn bench_denorm(bencher: Bencher, normalized: ArrayRef) {
    let session = vortex_array::array_session();
    let norms = norms();
    bencher
        .with_inputs(|| {
            (
                // `L2Denorm` has an inherent `try_new_array` with a different signature, so the
                // factory method needs naming explicitly.
                ScalarFnFactoryExt::try_new_array(
                    &L2Denorm,
                    ROWS,
                    EmptyOptions,
                    [normalized.clone(), norms.clone()],
                )
                .unwrap(),
                session.create_execution_ctx(),
            )
        })
        .bench_values(|(array, mut ctx)| array.execute::<ExtensionArray>(&mut ctx).unwrap());
}

/// Like for like against the pre-port kernel: no validity to apply, so the lifting adds no pass.
#[divan::bench(args = WIDTHS)]
fn non_nullable(bencher: Bencher, width: usize) {
    bench_denorm(bencher, normalized(width));
}

/// One row in eight null, so the conjoined validity is a `Validity::Array` and the lifting
/// materializes a mask and applies it to the row loop's output.
#[divan::bench(args = WIDTHS)]
fn nullable(bencher: Bencher, width: usize) {
    let array = normalized(width);
    let validity = Validity::from_iter((0..ROWS).map(|i| i % 8 != 0));
    bench_denorm(
        bencher,
        MaskedArray::try_new(array, validity).unwrap().into_array(),
    );
}

/// The kernel as it was written before the port, specialized to `f64`.
///
/// This is a full [`ScalarFnVTable`], exactly as `l2_norm.rs`'s control is, and hand-writes the null
/// handling that production `L2Denorm` derives: conjoin the input validities, build the values
/// all-valid, and mask them with the conjunction. That is what the lifting does around a dense
/// kernel, so the arms stay comparable.
#[derive(Clone)]
struct PrePortL2Denorm;

impl ScalarFnVTable for PrePortL2Denorm {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("bench.pre_port_l2_denorm");
        *ID
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(2)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("normalized"),
            _ => ChildName::from("norms"),
        }
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        Ok(arg_dtypes[0].union_nullability(arg_dtypes[1].nullability()))
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

    fn execute(
        &self,
        _options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let normalized_ref = args.get(0)?;
        let norms_ref = args.get(1)?;
        let row_count = args.row_count();

        let output_dtype = normalized_ref
            .dtype()
            .union_nullability(norms_ref.dtype().nullability());
        let conjoined = normalized_ref.validity()?.and(norms_ref.validity()?)?;
        let validity = if output_dtype.is_nullable() {
            Validity::AllValid
        } else {
            Validity::NonNullable
        };

        let ext: ExtensionArray = normalized_ref.execute(ctx)?;
        let fsl: FixedSizeListArray = ext.storage_array().clone().execute(ctx)?;
        let width = fsl.list_size() as usize;
        let elements: PrimitiveArray = fsl.elements().clone().execute(ctx)?;
        let flat = elements.as_slice::<f64>();

        let norms: PrimitiveArray = norms_ref.execute(ctx)?;
        let norms = norms.as_slice::<f64>();

        let scaled: Buffer<f64> = (0..row_count)
            .flat_map(|i| {
                let norm = norms[i];
                flat[i * width..(i + 1) * width]
                    .iter()
                    .map(move |&x| x * norm)
            })
            .collect();

        let storage = FixedSizeListArray::try_new(
            PrimitiveArray::new(scaled, Validity::NonNullable).into_array(),
            fsl.list_size(),
            validity,
            row_count,
        )?;

        let denormalized =
            ExtensionArray::new(output_dtype.as_extension().clone(), storage.into_array())
                .into_array();

        // The one thing the lifting used to add for this kernel, and still adds for the ported one:
        // mask the all-valid output with the conjoined input validity. Both arms therefore pay the
        // same masking pass, which leaves the output construction as the only difference.
        match conjoined {
            Validity::Array(valid) => denormalized.mask(valid),
            // Nothing to mask: these inputs are either wholly non-nullable or wholly valid, and
            // never wholly invalid.
            Validity::NonNullable | Validity::AllValid | Validity::AllInvalid => Ok(denormalized),
        }
    }
}

fn bench_pre_port(bencher: Bencher, normalized: ArrayRef) {
    let session = vortex_array::array_session();
    let norms = norms();
    bencher
        .with_inputs(|| {
            (
                PrePortL2Denorm
                    .try_new_array(ROWS, EmptyOptions, [normalized.clone(), norms.clone()])
                    .unwrap(),
                session.create_execution_ctx(),
            )
        })
        .bench_values(|(array, mut ctx)| array.execute::<ExtensionArray>(&mut ctx).unwrap());
}

#[divan::bench(args = WIDTHS)]
fn pre_port_non_nullable(bencher: Bencher, width: usize) {
    bench_pre_port(bencher, normalized(width));
}

#[divan::bench(args = WIDTHS)]
fn pre_port_nullable(bencher: Bencher, width: usize) {
    let array = normalized(width);
    let validity = Validity::from_iter((0..ROWS).map(|i| i % 8 != 0));
    bench_pre_port(
        bencher,
        MaskedArray::try_new(array, validity).unwrap().into_array(),
    );
}
