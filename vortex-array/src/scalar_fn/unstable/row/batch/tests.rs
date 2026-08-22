// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::registry::CachedId;

use super::finalize_kernel_output;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::PrimitiveArray;
use crate::assert_arrays_eq;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::scalar::Scalar;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::VecExecutionArgs;
use crate::scalar_fn::unstable::row::ArgView;
use crate::scalar_fn::unstable::row::DenseRows;
use crate::scalar_fn::unstable::row::InputElement;
use crate::scalar_fn::unstable::row::OutputElement;
use crate::scalar_fn::unstable::row::OutputSink;
use crate::scalar_fn::unstable::row::PackedBoolOutput;
use crate::scalar_fn::unstable::row::RowFn;
use crate::scalar_fn::unstable::row::RowKernel;
use crate::scalar_fn::unstable::row::RowKernelOutput;
use crate::scalar_fn::unstable::row::RowVisitor;
use crate::scalar_fn::unstable::row::VecOutput;
use crate::scalar_fn::unstable::row::execute_rows;
use crate::validity::Validity;

#[derive(Clone, Default)]
struct DeferredAdd {
    /// The number of preparations across the dense attempt and any valid-row retry.
    prepare_count: Arc<AtomicUsize>,
}

impl DeferredAdd {
    fn prepare_count(&self) -> usize {
        self.prepare_count.load(Ordering::Relaxed)
    }
}

#[derive(Clone)]
struct ValidOnlyIdentity;

#[derive(Clone)]
struct InvalidKernelOutput;

#[derive(Clone)]
struct PackedPositive;

struct PackedPositiveKernel;

#[derive(Clone)]
struct WrongLengthKernelOutput;

struct WrongLengthKernel;

struct WrongLengthOutput {
    row_count: usize,
}

#[derive(Clone)]
struct WrongDTypeKernelOutput;

struct WrongDTypeKernel;

struct WrongDTypeOutput {
    row_count: usize,
}

#[derive(Clone)]
struct RetainedViewIdentity;

struct RetainedViewIdentityKernel;

struct ChangingViewI64;

struct ChangingViewColumn {
    values: Buffer<i64>,
    view_count: AtomicUsize,
}

#[derive(Clone)]
struct ValidOnlyAssociatedOutput;

struct ValidOnlyAssociatedOutputKernel;

struct CountingBoolOutput(Vec<bool>);

static ASSOCIATED_OUTPUT_CONVERSIONS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: each returned slice is stable for its lifetime, and unchecked access is valid below that
// slice's length. Returning a shorter slice on later calls exercises the executor's obligation to
// retain the exact view whose length it validated.
unsafe impl InputElement for ChangingViewI64 {
    type Column = ChangingViewColumn;
    type View<'a> = &'a [i64];
    type Elem<'a> = i64;

    const DENSE_SAFE: bool = false;
    const DECODE_INFALLIBLE: bool = true;

    fn validate(dtype: &DType) -> VortexResult<()> {
        <i64 as InputElement>::validate(dtype)
    }

    fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Column> {
        Ok(ChangingViewColumn {
            values: <i64 as InputElement>::decode(array, ctx)?,
            view_count: AtomicUsize::new(0),
        })
    }

    fn can_decode_null_tolerant(_array: &ArrayRef) -> VortexResult<bool> {
        Ok(true)
    }

    fn get(column: &Self::Column, index: usize) -> Self::Elem<'_> {
        column.values[index]
    }

    fn view(column: &Self::Column) -> Self::View<'_> {
        let values = column.values.as_slice();
        if column.view_count.fetch_add(1, Ordering::Relaxed) == 0 {
            values
        } else {
            &values[..0]
        }
    }

    fn get_from_view<'a>(view: &Self::View<'a>, index: usize) -> Self::Elem<'a> {
        view[index]
    }
}

impl RowKernel<(ChangingViewI64,)> for RetainedViewIdentityKernel {
    type Element = i64;
    type Output = VecOutput<i64>;

    fn eval(&self, (value,): (i64,)) -> Self::Element {
        value
    }
}

impl RowKernel<(ChangingViewI64,)> for ValidOnlyAssociatedOutputKernel {
    type Element = bool;
    type Output = CountingBoolOutput;

    fn eval(&self, (value,): (i64,)) -> Self::Element {
        value > 0
    }
}

impl RowKernelOutput for CountingBoolOutput {
    type Element = bool;

    fn from_values(values: Vec<Self::Element>) -> VortexResult<Self> {
        ASSOCIATED_OUTPUT_CONVERSIONS.fetch_add(1, Ordering::Relaxed);
        Ok(Self(values))
    }

    fn finish(self) -> VortexResult<ArrayRef> {
        Ok(BoolArray::from_iter(self.0).into_array())
    }
}

impl RowKernel<(i64,)> for PackedPositiveKernel {
    type Element = bool;
    type Output = PackedBoolOutput;

    fn eval(&self, (value,): (i64,)) -> Self::Element {
        value > 0
    }

    fn collect_dense(&self, rows: DenseRows<'_, (i64,)>) -> VortexResult<Self::Output> {
        let mut output = PackedBoolOutput::zeroed(rows.len());

        match rows.inputs().0.view() {
            ArgView::Column(values) => {
                for (word_index, values) in values.chunks(64).enumerate() {
                    output.words_mut()[word_index] =
                        values.iter().enumerate().fold(0, |word, (bit, value)| {
                            word | (u64::from(*value > 0) << bit)
                        });
                }
            }
            ArgView::Constant(value) => {
                if value[0] > 0 {
                    output.words_mut().fill(u64::MAX);
                }
            }
        }

        Ok(output)
    }
}

impl RowKernel<(i64,)> for WrongLengthKernel {
    type Element = bool;
    type Output = WrongLengthOutput;

    fn eval(&self, (_value,): (i64,)) -> Self::Element {
        true
    }
}

impl RowKernelOutput for WrongLengthOutput {
    type Element = bool;

    fn from_values(values: Vec<Self::Element>) -> VortexResult<Self> {
        Ok(Self {
            row_count: values.len(),
        })
    }

    fn finish(self) -> VortexResult<ArrayRef> {
        Ok(
            BoolArray::from_iter(std::iter::repeat_n(true, self.row_count.saturating_sub(1)))
                .into_array(),
        )
    }
}

impl RowKernel<(i64,)> for WrongDTypeKernel {
    type Element = bool;
    type Output = WrongDTypeOutput;

    fn eval(&self, (_value,): (i64,)) -> Self::Element {
        true
    }
}

impl RowKernelOutput for WrongDTypeOutput {
    type Element = bool;

    fn from_values(values: Vec<Self::Element>) -> VortexResult<Self> {
        Ok(Self {
            row_count: values.len(),
        })
    }

    fn finish(self) -> VortexResult<ArrayRef> {
        Ok(PrimitiveArray::from_iter(std::iter::repeat_n(1_i64, self.row_count)).into_array())
    }
}

/// Produces a null row to exercise output validation at the row-function boundary.
#[derive(Default)]
struct NullProducingI64(i64);

impl OutputElement for NullProducingI64 {
    fn element_dtype() -> DType {
        DType::from(i64::PTYPE)
    }

    fn build(values: Vec<Self>) -> ArrayRef {
        let values: Vec<_> = values.into_iter().map(|value| value.0).collect();
        let validity = Validity::from_iter((0..values.len()).map(|index| index != 0));

        PrimitiveArray::new(values, validity).into_array()
    }
}

struct I64Sink(BufferMut<i64>);

// SAFETY: every row is initialized by `BufferMut::zeroed`, and the sink exposes exactly that
// initialized slice. The `()` write token therefore proves no additional invariant.
unsafe impl<Options> OutputSink<Options> for I64Sink {
    type Rows<'a> = &'a mut [i64];
    type Row<'a> = &'a mut i64;
    type WriteToken = ();

    fn return_dtype(_options: &Options) -> VortexResult<DType> {
        Ok(DType::from(i64::PTYPE))
    }

    fn with_capacity(rows: usize) -> VortexResult<Self> {
        Ok(Self(BufferMut::zeroed(rows)))
    }

    fn rows(&mut self) -> Self::Rows<'_> {
        self.0.as_mut_slice()
    }

    unsafe fn row_unchecked<'a>(rows: &'a mut Self::Rows<'_>, index: usize) -> Self::Row<'a> {
        // SAFETY: required by this method's contract.
        unsafe { rows.get_unchecked_mut(index) }
    }

    unsafe fn finish(self) -> VortexResult<ArrayRef> {
        Ok(PrimitiveArray::new(self.0.freeze(), Validity::NonNullable).into_array())
    }
}

impl RowFn for DeferredAdd {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["lhs", "rhs"];
    const INFALLIBLE: bool = false;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("test.deferred_add");
        *ID
    }

    fn dispatch<V: RowVisitor<Self::Options>>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        let prepare_count = Arc::clone(&self.prepare_count);

        visitor.visit_prepared_deferred::<(i64, i64), i64, (), bool>(
            move |_| {
                prepare_count.fetch_add(1, Ordering::Relaxed);
            },
            |&(), (lhs, rhs)| lhs.overflowing_add(rhs),
            |overflowed| {
                if overflowed {
                    vortex_bail!(InvalidArgument: "deferred addition overflowed");
                }

                Ok(())
            },
        )
    }
}

impl RowFn for ValidOnlyIdentity {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["value"];
    const INFALLIBLE: bool = false;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("test.valid_only_identity");
        *ID
    }

    fn dispatch<V: RowVisitor<Self::Options>>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit_into::<(i64,), I64Sink, VortexResult<()>>(|(value,), output| {
            *output = value;
            Ok(())
        })
    }
}

impl RowFn for InvalidKernelOutput {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["value"];
    const INFALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("test.invalid_kernel_output");
        *ID
    }

    fn dispatch<V: RowVisitor<Self::Options>>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit::<(i64,), NullProducingI64>(|(value,)| NullProducingI64(value))
    }
}

impl RowFn for PackedPositive {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["value"];
    const INFALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("test.packed_positive");
        *ID
    }

    fn dispatch<V: RowVisitor<Self::Options>>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit_kernel::<(i64,), _>(PackedPositiveKernel)
    }
}

impl RowFn for WrongLengthKernelOutput {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["value"];
    const INFALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("test.wrong_length_kernel_output");
        *ID
    }

    fn dispatch<V: RowVisitor<Self::Options>>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit_kernel::<(i64,), _>(WrongLengthKernel)
    }
}

impl RowFn for WrongDTypeKernelOutput {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["value"];
    const INFALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("test.wrong_dtype_kernel_output");
        *ID
    }

    fn dispatch<V: RowVisitor<Self::Options>>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit_kernel::<(i64,), _>(WrongDTypeKernel)
    }
}

impl RowFn for RetainedViewIdentity {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["value"];
    const INFALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("test.retained_view_identity");
        *ID
    }

    fn dispatch<V: RowVisitor<Self::Options>>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit_kernel::<(ChangingViewI64,), _>(RetainedViewIdentityKernel)
    }
}

impl RowFn for ValidOnlyAssociatedOutput {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["value"];
    const INFALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("test.valid_only_associated_output");
        *ID
    }

    fn dispatch<V: RowVisitor<Self::Options>>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit_kernel::<(ChangingViewI64,), _>(ValidOnlyAssociatedOutputKernel)
    }
}

#[test]
fn test_finalize_kernel_output_rejects_nested_dtype_mismatch() -> VortexResult<()> {
    static ID: CachedId = CachedId::new("test.finalize_kernel_output");

    let element_dtype = DType::Primitive(i64::PTYPE, Nullability::NonNullable);
    let values = ConstantArray::new(
        Scalar::list_empty(Arc::new(element_dtype), Nullability::NonNullable),
        2,
    )
    .into_array();
    let result_dtype = DType::List(
        Arc::new(DType::Primitive(i64::PTYPE, Nullability::Nullable)),
        Nullability::NonNullable,
    );
    let mut ctx = array_session().create_execution_ctx();

    assert!(finalize_kernel_output(*ID, &result_dtype, 2, values, &mut ctx).is_err());
    Ok(())
}

#[test]
fn test_kernel_output_rejects_nulls_at_function_boundary() -> VortexResult<()> {
    let input = PrimitiveArray::new(vec![1_i64, 2], Validity::NonNullable).into_array();
    let args = VecExecutionArgs::new(vec![input], 2);
    let mut ctx = array_session().create_execution_ctx();
    let execution = execute_rows(&InvalidKernelOutput, &EmptyOptions, &args, &mut ctx);
    let error = match execution {
        Err(error) => error,
        Ok(output) => match output.execute::<PrimitiveArray>(&mut ctx) {
            Err(error) => error,
            Ok(_) => vortex_bail!("an invalid row kernel output passed boundary validation"),
        },
    };
    let error = error.to_string();

    assert!(
        error.contains("test.invalid_kernel_output"),
        "the boundary error must name the function, got {error}",
    );
    assert!(
        error.contains("row kernel must produce only valid rows"),
        "the boundary error must identify invalid row output, got {error}",
    );
    Ok(())
}

#[test]
fn test_dense_kernel_writes_packed_output_words() -> VortexResult<()> {
    let input = PrimitiveArray::new(
        vec![1_i64, -1, 2, 0, 3],
        Validity::from_iter([true, true, false, true, true]),
    )
    .into_array();
    let args = VecExecutionArgs::new(vec![input], 5);
    let mut ctx = array_session().create_execution_ctx();

    let actual = execute_rows(&PackedPositive, &EmptyOptions, &args, &mut ctx)?;
    let expected =
        BoolArray::from_iter([Some(true), Some(false), None, Some(false), Some(true)]).into_array();

    assert_arrays_eq!(&actual, &expected, &mut ctx);
    Ok(())
}

#[test]
fn test_dense_kernel_handles_batch_constant_input() -> VortexResult<()> {
    let input = ConstantArray::new(7_i64, 65).into_array();
    let args = VecExecutionArgs::new(vec![input], 65);
    let mut ctx = array_session().create_execution_ctx();

    let actual = execute_rows(&PackedPositive, &EmptyOptions, &args, &mut ctx)?;
    let expected = BoolArray::from_iter(std::iter::repeat_n(true, 65)).into_array();

    assert_arrays_eq!(&actual, &expected, &mut ctx);
    Ok(())
}

#[test]
fn test_dense_kernel_retains_the_validated_views() -> VortexResult<()> {
    let input = PrimitiveArray::from_iter([1_i64, 2, 3]).into_array();
    let args = VecExecutionArgs::new(vec![input], 3);
    let mut ctx = array_session().create_execution_ctx();

    let actual = execute_rows(&RetainedViewIdentity, &EmptyOptions, &args, &mut ctx)?;
    let expected = PrimitiveArray::from_iter([1_i64, 2, 3]).into_array();

    assert_arrays_eq!(&actual, &expected, &mut ctx);
    Ok(())
}

#[test]
fn test_valid_only_kernel_uses_its_associated_output() -> VortexResult<()> {
    ASSOCIATED_OUTPUT_CONVERSIONS.store(0, Ordering::Relaxed);

    let input = PrimitiveArray::new(vec![1_i64, -1, 2], Validity::from_iter([true, false, true]))
        .into_array();
    let args = VecExecutionArgs::new(vec![input], 3);
    let mut ctx = array_session().create_execution_ctx();

    let actual = execute_rows(&ValidOnlyAssociatedOutput, &EmptyOptions, &args, &mut ctx)?;
    let expected = BoolArray::from_iter([Some(true), None, Some(true)]).into_array();

    assert_arrays_eq!(&actual, &expected, &mut ctx);
    assert_eq!(ASSOCIATED_OUTPUT_CONVERSIONS.load(Ordering::Relaxed), 1);
    Ok(())
}

#[test]
fn test_dense_kernel_rejects_wrong_output_length() -> VortexResult<()> {
    let input = PrimitiveArray::from_iter([1_i64, 2]).into_array();
    let args = VecExecutionArgs::new(vec![input], 2);
    let mut ctx = array_session().create_execution_ctx();

    let error = match execute_rows(&WrongLengthKernelOutput, &EmptyOptions, &args, &mut ctx) {
        Err(error) => error.to_string(),
        Ok(_) => vortex_bail!("a RowKernelOutput with the wrong length passed validation"),
    };

    assert!(
        error.contains("test.wrong_length_kernel_output"),
        "the boundary error must name the function, got {error}",
    );
    assert!(
        error.contains("must contain 2 rows, got 1"),
        "the boundary error must identify the wrong output length, got {error}",
    );
    Ok(())
}

#[test]
fn test_dense_kernel_rejects_wrong_output_dtype() -> VortexResult<()> {
    let input = PrimitiveArray::from_iter([1_i64, 2]).into_array();
    let args = VecExecutionArgs::new(vec![input], 2);
    let mut ctx = array_session().create_execution_ctx();

    let error = match execute_rows(&WrongDTypeKernelOutput, &EmptyOptions, &args, &mut ctx) {
        Err(error) => error.to_string(),
        Ok(_) => vortex_bail!("a RowKernelOutput with the wrong dtype passed validation"),
    };

    assert!(
        error.contains("test.wrong_dtype_kernel_output"),
        "the boundary error must name the function, got {error}",
    );
    assert!(
        error.contains("output dtype must match bool") && error.contains("got i64"),
        "the boundary error must identify the wrong output dtype, got {error}",
    );
    Ok(())
}

#[test]
fn test_deferred_owned_execution_retries_null_row_failure() -> VortexResult<()> {
    let function = DeferredAdd::default();
    let validity = Validity::from_iter([true, false]);
    let lhs = PrimitiveArray::new(vec![1_i64, i64::MAX], validity.clone()).into_array();
    let rhs = ConstantArray::new(1_i64, 2).into_array();
    let args = VecExecutionArgs::new(vec![lhs, rhs], 2);
    let mut ctx = array_session().create_execution_ctx();

    let actual = execute_rows(&function, &EmptyOptions, &args, &mut ctx)?;
    let expected = PrimitiveArray::new(vec![2_i64, 0], validity).into_array();

    assert_arrays_eq!(&actual, &expected, &mut ctx);
    assert_eq!(function.prepare_count(), 2);
    Ok(())
}

#[test]
fn test_deferred_owned_execution_does_not_retry_partially_valid_success() -> VortexResult<()> {
    let function = DeferredAdd::default();
    let validity = Validity::from_iter([true, false]);
    let lhs = PrimitiveArray::new(vec![1_i64, 2], validity.clone()).into_array();
    let rhs = ConstantArray::new(1_i64, 2).into_array();
    let args = VecExecutionArgs::new(vec![lhs, rhs], 2);
    let mut ctx = array_session().create_execution_ctx();

    let actual = execute_rows(&function, &EmptyOptions, &args, &mut ctx)?;
    let expected = PrimitiveArray::new(vec![2_i64, 0], validity).into_array();

    assert_arrays_eq!(&actual, &expected, &mut ctx);
    assert_eq!(function.prepare_count(), 1);
    Ok(())
}

#[test]
fn test_deferred_owned_execution_retries_and_reports_valid_row_failure() -> VortexResult<()> {
    let function = DeferredAdd::default();
    let validity = Validity::from_iter([true, false]);
    let lhs = PrimitiveArray::new(vec![i64::MAX, 1], validity).into_array();
    let rhs = ConstantArray::new(1_i64, 2).into_array();
    let args = VecExecutionArgs::new(vec![lhs, rhs], 2);
    let mut ctx = array_session().create_execution_ctx();

    let error = execute_rows(&function, &EmptyOptions, &args, &mut ctx)
        .expect_err("a valid-row overflow must remain observable");
    let error = error.to_string();

    assert!(
        error.contains("deferred addition overflowed"),
        "the valid-row retry must report its deferred error, got {error}",
    );
    assert_eq!(function.prepare_count(), 2);
    Ok(())
}

#[test]
fn test_deferred_owned_array_backed_all_valid_error_does_not_retry() -> VortexResult<()> {
    let function = DeferredAdd::default();
    let validity = Validity::Array(ConstantArray::new(true, 2).into_array());
    let lhs = PrimitiveArray::new(vec![i64::MAX, 1], validity).into_array();
    let rhs = ConstantArray::new(1_i64, 2).into_array();
    let args = VecExecutionArgs::new(vec![lhs, rhs], 2);
    let mut ctx = array_session().create_execution_ctx();

    let error = execute_rows(&function, &EmptyOptions, &args, &mut ctx)
        .expect_err("an all-valid deferred error must remain observable");
    let error = error.to_string();

    assert!(
        error.contains("deferred addition overflowed"),
        "the dense attempt must return its original deferred error, got {error}",
    );
    assert_eq!(function.prepare_count(), 1);
    Ok(())
}

#[test]
fn test_valid_only_empty_batch_preserves_nonnullable_dtype() -> VortexResult<()> {
    let input = PrimitiveArray::from_iter(std::iter::empty::<i64>()).into_array();
    let args = VecExecutionArgs::new(vec![input], 0);
    let mut ctx = array_session().create_execution_ctx();

    let actual = execute_rows(&ValidOnlyIdentity, &EmptyOptions, &args, &mut ctx)?;

    assert_eq!(actual.len(), 0);
    assert_eq!(actual.dtype(), &DType::from(i64::PTYPE));
    Ok(())
}
