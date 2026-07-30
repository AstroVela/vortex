// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::ConstantArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::scalar_fn::ScalarFnFactoryExt;
use crate::assert_arrays_eq;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::expr::Expression;
use crate::expr::root;
use crate::expr::union_child_validities;
use crate::scalar::Scalar;
use crate::scalar_fn::*;
use crate::validity::Validity;

/// A strict addition over i32 used to exercise both null-handling strategies of the adapter.
#[derive(Clone, Debug)]
struct TestAdd {
    null_handling: NullHandling,
}

impl StrictScalarFnVTable for TestAdd {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.test.strict_add");
        *ID
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(2)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("lhs"),
            1 => ChildName::from("rhs"),
            _ => unreachable!("TestAdd has exactly two children"),
        }
    }

    fn return_element_dtype(
        &self,
        _options: &Self::Options,
        args: &[DType],
    ) -> VortexResult<DType> {
        let i32 = DType::Primitive(PType::I32, Nullability::NonNullable);
        vortex_ensure!(
            args.iter().all(|dtype| dtype.eq_ignore_nullability(&i32)),
            "test_add requires i32 inputs, got {args:?}",
        );
        Ok(i32)
    }

    fn null_handling(&self, _options: &Self::Options) -> NullHandling {
        self.null_handling
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        false
    }

    fn validity(
        &self,
        _options: &Self::Options,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        union_child_validities(expression)
    }

    fn execute_strict(
        &self,
        _options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let lhs = args.get(0)?.execute::<PrimitiveArray>(ctx)?;
        let rhs = args.get(1)?.execute::<PrimitiveArray>(ctx)?;
        let values: Buffer<i32> = lhs
            .as_slice::<i32>()
            .iter()
            .zip(rhs.as_slice::<i32>())
            .map(|(l, r)| l.wrapping_add(*r))
            .collect();
        Ok(PrimitiveArray::new(values, Validity::NonNullable).into_array())
    }
}

fn add(null_handling: NullHandling) -> TestAdd {
    TestAdd { null_handling }
}

fn execute_add(
    null_handling: NullHandling,
    lhs: ArrayRef,
    rhs: ArrayRef,
) -> VortexResult<ArrayRef> {
    let len = lhs.len();
    add(null_handling)
        .try_new_array(len, EmptyOptions, [lhs, rhs])?
        .execute::<Canonical>(&mut array_session().create_execution_ctx())
        .map(|c| c.into_array())
}

#[rstest]
#[case(NullHandling::Filter)]
#[case(NullHandling::Dense)]
fn no_nulls(#[case] null_handling: NullHandling) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let lhs = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
    let rhs = PrimitiveArray::from_iter([10i32, 20, 30]).into_array();
    let result = execute_add(null_handling, lhs, rhs)?;
    assert_arrays_eq!(result, PrimitiveArray::from_iter([11i32, 22, 33]), &mut ctx);
    Ok(())
}

#[rstest]
#[case(NullHandling::Filter)]
#[case(NullHandling::Dense)]
fn nulls_propagate(#[case] null_handling: NullHandling) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let lhs = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), None]).into_array();
    let rhs = PrimitiveArray::from_option_iter([Some(10i32), Some(20), None, None]).into_array();
    let result = execute_add(null_handling, lhs, rhs)?;
    assert_arrays_eq!(
        result,
        PrimitiveArray::from_option_iter([Some(11i32), None, None, None]),
        &mut ctx
    );
    Ok(())
}

#[rstest]
#[case(NullHandling::Filter)]
#[case(NullHandling::Dense)]
fn null_constant_short_circuits(#[case] null_handling: NullHandling) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let lhs = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
    let null = Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable));
    let rhs = ConstantArray::new(null, 3).into_array();
    let result = execute_add(null_handling, lhs, rhs)?;
    assert_arrays_eq!(
        result,
        PrimitiveArray::from_option_iter([Option::<i32>::None, None, None]),
        &mut ctx
    );
    Ok(())
}

#[test]
fn all_constants_broadcast() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let lhs = ConstantArray::new(Scalar::from(2i32), 4).into_array();
    let rhs = ConstantArray::new(Scalar::from(40i32), 4).into_array();
    let result = execute_add(NullHandling::Filter, lhs, rhs)?;
    assert_arrays_eq!(
        result,
        PrimitiveArray::from_iter([42i32, 42, 42, 42]),
        &mut ctx
    );
    Ok(())
}

#[rstest]
#[case(NullHandling::Filter)]
#[case(NullHandling::Dense)]
fn mixed_constant_and_column(#[case] null_handling: NullHandling) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let lhs = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array();
    let rhs = ConstantArray::new(Scalar::from(10i32), 3).into_array();
    let result = execute_add(null_handling, lhs, rhs)?;
    assert_arrays_eq!(
        result,
        PrimitiveArray::from_option_iter([Some(11i32), None, Some(13)]),
        &mut ctx
    );
    Ok(())
}

#[test]
fn empty_input_keeps_dtype() -> VortexResult<()> {
    let lhs = PrimitiveArray::from_iter(Vec::<i32>::new()).into_array();
    let rhs = PrimitiveArray::from_iter(Vec::<i32>::new()).into_array();

    let result = execute_add(NullHandling::Filter, lhs, rhs)?;

    assert_eq!(result.len(), 0);
    assert!(!result.dtype().is_nullable());
    Ok(())
}

#[test]
fn return_dtype_unions_nullability() -> VortexResult<()> {
    let non_nullable = DType::Primitive(PType::I32, Nullability::NonNullable);
    let nullable = non_nullable.as_nullable();

    let vtable = add(NullHandling::Filter);
    assert_eq!(
        vtable.return_dtype(&EmptyOptions, &[non_nullable.clone(), non_nullable.clone()])?,
        non_nullable
    );
    assert_eq!(
        vtable.return_dtype(&EmptyOptions, &[non_nullable, nullable.clone()])?,
        nullable
    );
    Ok(())
}

#[test]
fn adapter_reports_strict() {
    assert!(ScalarFnVTable::is_strict(
        &add(NullHandling::Filter),
        &EmptyOptions
    ));
}

/// Options serde comes from [`PersistableOptions`], so a strict function needs none of its own.
#[test]
fn options_round_trip_without_per_function_serde() -> VortexResult<()> {
    let vtable = add(NullHandling::Dense);
    let metadata =
        ScalarFnVTable::serialize(&vtable, &EmptyOptions)?.expect("EmptyOptions is serializable");
    let options = ScalarFnVTable::deserialize(&vtable, &metadata, &array_session())?;
    assert_eq!(options, EmptyOptions);
    Ok(())
}

/// A strict but *not* total function, standing in for `list_sum`: it propagates nulls like any strict
/// kernel, and additionally returns null for a *valid* row holding zero, the way summing a valid
/// empty list yields null.
#[derive(Clone, Debug)]
struct SumOrNull {
    /// Nullability this function declares for its output element. `NonNullable` is a lie for a
    /// non-total kernel, and the lifting **must** reject it.
    element_nullability: Nullability,
}

impl StrictScalarFnVTable for SumOrNull {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.test.sum_or_null");
        *ID
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, _child_idx: usize) -> ChildName {
        ChildName::from("input")
    }

    fn return_element_dtype(
        &self,
        _options: &Self::Options,
        _args: &[DType],
    ) -> VortexResult<DType> {
        Ok(DType::Primitive(PType::I32, self.element_nullability))
    }

    fn null_handling(&self, _options: &Self::Options) -> NullHandling {
        NullHandling::Dense
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        false
    }

    fn execute_strict(
        &self,
        _options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?.execute::<PrimitiveArray>(ctx)?;
        let summed = input
            .as_slice::<i32>()
            .iter()
            .map(|value| (*value != 0).then_some(*value));

        Ok(PrimitiveArray::from_option_iter(summed).into_array())
    }
}

/// The honest non-total function, declaring the nullable element dtype its kernel needs.
fn sum_or_null() -> SumOrNull {
    SumOrNull {
        element_nullability: Nullability::Nullable,
    }
}

/// A kernel that never turns a wholly non-null row into a null can answer `validity` with the child
/// conjunction, which is what lets the planner skip executing it.
#[test]
fn a_total_kernel_precomputes_validity() -> VortexResult<()> {
    let expr = add(NullHandling::Dense).new_expr(EmptyOptions, [root(), root()]);

    let vtable = add(NullHandling::Dense);
    assert!(ScalarFnVTable::validity(&vtable, &EmptyOptions, &expr)?.is_some());
    Ok(())
}

/// A kernel that can, like summing a valid empty list, leaves it at the default instead.
#[test]
fn a_non_total_kernel_declines_precomputed_validity() -> VortexResult<()> {
    let expr = sum_or_null().new_expr(EmptyOptions, [root()]);

    assert!(ScalarFnVTable::validity(&sum_or_null(), &EmptyOptions, &expr)?.is_none());
    Ok(())
}

/// Declining precomputed validity **must** not cost the function its strictness, which is the
/// property mask, filter and dictionary push-downs actually require.
#[test]
fn a_non_total_kernel_is_still_strict() {
    assert!(ScalarFnVTable::is_strict(&sum_or_null(), &EmptyOptions));
}

/// Row 1 is valid but sums to null, and row 2 is null in the input. Both have to survive to the
/// output, so the kernel's own nulls are unioned with the ones the lifting applies.
#[test]
fn a_non_total_kernel_keeps_its_own_nulls() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let input = PrimitiveArray::from_option_iter([Some(4i32), Some(0), None]).into_array();

    let result = sum_or_null()
        .try_new_array(3, EmptyOptions, [input])?
        .execute::<Canonical>(&mut ctx)?
        .into_array();

    assert_arrays_eq!(
        result,
        PrimitiveArray::from_option_iter([Some(4i32), None, None]),
        &mut ctx
    );
    Ok(())
}

/// An idempotent strict function, collapsing `Identity(Identity(x))` to `Identity(x)`. It exists to
/// pin that the `reduce` hook reaches a strict function through the blanket impl.
#[derive(Clone, Debug)]
struct Identity;

impl StrictScalarFnVTable for Identity {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.test.strict_identity");
        *ID
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, _child_idx: usize) -> ChildName {
        ChildName::from("input")
    }

    fn return_element_dtype(
        &self,
        _options: &Self::Options,
        args: &[DType],
    ) -> VortexResult<DType> {
        Ok(args[0].as_nonnullable())
    }

    fn null_handling(&self, _options: &Self::Options) -> NullHandling {
        NullHandling::Dense
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        false
    }

    fn execute_strict(
        &self,
        _options: &Self::Options,
        args: &dyn ExecutionArgs,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        args.get(0)
    }

    fn reduce(
        &self,
        _options: &Self::Options,
        node: &dyn ReduceNode,
        _ctx: &dyn ReduceCtx,
    ) -> VortexResult<Option<ReduceNodeRef>> {
        let child = node.child(0);
        let is_idempotent = child
            .scalar_fn()
            .is_some_and(|scalar_fn| scalar_fn.as_opt::<Identity>().is_some());

        // Drop this layer, keeping the equivalent inner one.
        Ok(is_idempotent.then_some(child))
    }
}

#[test]
fn reduce_reaches_a_strict_function() -> VortexResult<()> {
    let scope = DType::Primitive(PType::I32, Nullability::NonNullable);
    let once = Identity.new_expr(EmptyOptions, [root()]);
    let twice = Identity.new_expr(EmptyOptions, [once.clone()]);

    assert_eq!(twice.optimize(&scope)?, once);
    Ok(())
}
