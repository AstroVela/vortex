// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end tests for row function execution.

use rstest::rstest;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::ConstantArray;
use crate::arrays::MaskedArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::scalar_fn::ScalarFnFactoryExt;
use crate::assert_arrays_eq;
use crate::dtype::DType;
use crate::expr::root;
use crate::scalar::Scalar;
use crate::scalar_fn::row::execute::row_null_handling;
use crate::scalar_fn::*;

mod conformance;
mod constant_operands;
mod decode_fallibility;
mod dispatched;
mod lifting;
mod null_strategies;
mod nullable_outputs;
mod prepared;
mod sink;

/// Builds `scalar_fn` over `args` and executes it end to end, which is what every test below does.
fn apply<F: RowFn<Options = EmptyOptions>>(
    scalar_fn: F,
    args: impl IntoIterator<Item = ArrayRef>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let args = args.into_iter().collect::<Vec<_>>();
    let rows = args.first().map_or(0, |arg| arg.len());

    Ok(scalar_fn
        .try_new_array(rows, EmptyOptions, args)?
        .execute::<Canonical>(ctx)?
        .into_array())
}

/// A binary row function over fixed primitive types: `hypot(x, y)`.
#[derive(Clone)]
struct Hypot;

impl RowFn for Hypot {
    type Options = EmptyOptions;
    type ArgsWitness = (f64, f64);

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.test.hypot");
        *ID
    }

    fn arg_name(&self, idx: usize) -> ChildName {
        ChildName::from(["x", "y"][idx])
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::Out> {
        visitor.visit_prepared_into::<(f64, f64), ElementSink<f64>, _, _>(
            |_| (),
            |&(), (x, y), output| output.write(x.hypot(y)),
        )
    }
}

/// A unary row function over strings: uppercased text, exercising [`Bytes`] input and
/// [`String`] output.
#[derive(Clone)]
struct Shout;

impl RowFn for Shout {
    type Options = EmptyOptions;
    type ArgsWitness = (Bytes,);

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.test.shout");
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
        visitor.visit_prepared_into::<(Bytes,), ElementSink<String>, _, _>(
            |_| (),
            |&(), (text,), output| {
                output.write(String::from_utf8_lossy(text).to_uppercase());
            },
        )
    }
}

/// A fallible row function: integer division, undefined at a zero divisor.
#[derive(Clone)]
struct CheckedDiv;

impl RowFn for CheckedDiv {
    type Options = EmptyOptions;
    type ArgsWitness = (i64, i64);
    const FALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.test.checked_div");
        *ID
    }

    fn arg_name(&self, idx: usize) -> ChildName {
        ChildName::from(["lhs", "rhs"][idx])
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::Out> {
        visitor.visit_prepared_into::<(i64, i64), ElementSink<i64>, _, _>(
            |_| (),
            |&(), (lhs, rhs), output| {
                if rhs == 0 {
                    vortex_bail!("division by zero");
                }
                output.write(lhs / rhs);
                Ok(())
            },
        )
    }
}

#[test]
fn hypot_columns() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let x = buffer![3.0f64, 5.0].into_array();
    let y = buffer![4.0f64, 12.0].into_array();

    let result = apply(Hypot, [x, y], &mut ctx)?;

    assert_arrays_eq!(result, PrimitiveArray::from_iter([5.0f64, 13.0]), &mut ctx);
    Ok(())
}

#[test]
fn hypot_propagates_nulls_and_constants() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let x = PrimitiveArray::from_option_iter([Some(3.0f64), None, Some(8.0)]).into_array();
    let y = ConstantArray::new(Scalar::from(4.0f64), 3).into_array();

    let result = apply(Hypot, [x, y], &mut ctx)?;

    assert_arrays_eq!(
        result,
        PrimitiveArray::from_option_iter([Some(5.0f64), None, Some((80.0f64).sqrt())]),
        &mut ctx
    );
    Ok(())
}

#[test]
fn shout_strings() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let input =
        VarBinViewArray::from_iter_nullable_str([Some("hello"), None, Some("Vortex")]).into_array();

    let result = apply(Shout, [input], &mut ctx)?;

    let expected =
        VarBinViewArray::from_iter_nullable_str([Some("HELLO"), None, Some("VORTEX")]).into_array();
    assert_arrays_eq!(result, expected, &mut ctx);
    Ok(())
}

#[test]
fn display_names_the_function_id() {
    let expr = Hypot.new_expr(EmptyOptions, [root(), root()]);
    assert_eq!(expr.to_string(), "vortex.test.hypot($, $)");
}

#[derive(Clone)]
struct WrongLength;

impl RowFn for WrongLength {
    type Options = EmptyOptions;
    type ArgsWitness = (i64,);

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.test.wrong_length");
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
        visitor.visit_prepared_into::<(i64,), ElementSink<i64>, _, _>(
            |_| (),
            |&(), (value,), output| output.write(value),
        )
    }

    fn reduce_encoded(
        &self,
        _options: &Self::Options,
        _args: &[ArrayRef],
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(PrimitiveArray::from_iter([0i64]).into_array()))
    }
}

#[test]
fn kernel_result_length_is_validated() {
    let mut ctx = array_session().create_execution_ctx();
    let input = buffer![1i64, 2, 3].into_array();

    let error = apply(WrongLength, [input], &mut ctx).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("produced 1 rows for 3 input rows"),
        "{error}"
    );
}

#[derive(Clone)]
struct FortyTwo;

impl RowFn for FortyTwo {
    type Options = EmptyOptions;
    type ArgsWitness = ();

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.test.forty_two");
        *ID
    }

    fn arg_name(&self, _idx: usize) -> ChildName {
        ChildName::from("unused")
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::Out> {
        visitor.visit_prepared_into::<(), ElementSink<i64>, _, _>(
            |()| (),
            |&(), (), output| output.write(42),
        )
    }
}

#[test]
fn nullary_row_fn_executes_requested_rows() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let result = FortyTwo
        .try_new_array(3, EmptyOptions, [])?
        .execute::<Canonical>(&mut ctx)?;

    assert_arrays_eq!(result, PrimitiveArray::from_iter([42i64; 3]), &mut ctx);
    Ok(())
}

#[derive(Clone)]
struct SumFour;

impl RowFn for SumFour {
    type Options = EmptyOptions;
    type ArgsWitness = (i64, i64, i64, i64);

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.test.sum_four");
        *ID
    }

    fn arg_name(&self, idx: usize) -> ChildName {
        ChildName::from(["a", "b", "c", "d"][idx])
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::Out> {
        visitor.visit_prepared_into::<(i64, i64, i64, i64), ElementSink<i64>, _, _>(
            |_| (),
            |&(), (a, b, c, d), output| output.write(a + b + c + d),
        )
    }
}

#[test]
fn four_argument_row_fn_executes() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let result = apply(
        SumFour,
        [
            buffer![1i64, 2].into_array(),
            buffer![10i64, 20].into_array(),
            buffer![100i64, 200].into_array(),
            buffer![1000i64, 2000].into_array(),
        ],
        &mut ctx,
    )?;

    assert_arrays_eq!(result, PrimitiveArray::from_iter([1111i64, 2222]), &mut ctx);
    Ok(())
}

#[test]
fn tuples_are_supported_through_arity_twelve() {
    type TwelveI64s = (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64);

    assert_eq!(<() as ElementTuple>::ARITY, 0);
    assert_eq!(<TwelveI64s as ElementTuple>::ARITY, 12);
}

#[test]
fn kernel_flag_decides_fallibility() {
    assert!(!ScalarFnVTable::is_fallible(&Hypot, &EmptyOptions));
    assert!(ScalarFnVTable::is_fallible(&CheckedDiv, &EmptyOptions));
}

#[test]
fn fallible_apply_propagates_its_error() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let lhs = buffer![10i64, 10].into_array();
    let rhs = buffer![2i64, 0].into_array();

    let error = apply(CheckedDiv, [lhs, rhs], &mut ctx)
        .expect_err("a zero divisor must fail the execution");

    assert!(
        error.to_string().contains("division by zero"),
        "unexpected error: {error}"
    );
    Ok(())
}

/// The divisor's null slot holds a zero, which a dense pass would divide by. Filtering keeps the
/// fallible kernel away from it.
#[test]
fn fallible_apply_never_sees_rows_behind_nulls() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let lhs = buffer![10i64, 10].into_array();
    let rhs = PrimitiveArray::from_option_iter([Some(2i64), None]).into_array();

    let result = apply(CheckedDiv, [lhs, rhs], &mut ctx)?;

    assert_arrays_eq!(
        result,
        PrimitiveArray::from_option_iter([Some(5i64), None]),
        &mut ctx
    );
    Ok(())
}

/// The [`NullHandling`] the framework derives for `F`, which no other API exposes: a row function
/// never declares one.
fn null_handling<F: RowFn>() -> NullHandling {
    row_null_handling::<F::ArgsWitness>(F::FALLIBLE)
}

/// Neither `Dense` nor `Filter` is ever written down: the arguments and fallibility decide.
/// `Dense` is chosen whenever it is sound, because it is cheaper and preserves input encodings.
#[test]
fn null_handling_follows_from_args_and_fallibility() {
    // Primitive arguments, infallible: nothing behind a null row can fault.
    assert_eq!(null_handling::<Hypot>(), NullHandling::Dense);
    // `Bytes` resolves a view into a data buffer, which is only meaningful for valid rows.
    assert_eq!(null_handling::<Shout>(), NullHandling::Filter);
    // Fallible: a garbage row could raise an error of its own.
    assert_eq!(null_handling::<CheckedDiv>(), NullHandling::Filter);
}
