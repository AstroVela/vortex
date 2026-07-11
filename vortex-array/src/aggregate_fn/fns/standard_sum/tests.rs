// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::CheckedAdd;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::DynAccumulator;
use crate::aggregate_fn::DynGroupedAccumulator;
use crate::aggregate_fn::GroupedAccumulator;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::standard_sum::StandardSum;
use crate::aggregate_fn::fns::standard_sum::standard_sum;
use crate::aggregate_fn::fns::sum::sum;
use crate::array_session;
use crate::arrays::BoolArray;
use crate::arrays::ChunkedArray;
use crate::arrays::ConstantArray;
use crate::arrays::DecimalArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::ListViewArray;
use crate::arrays::PrimitiveArray;
use crate::assert_arrays_eq;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::Nullability;
use crate::dtype::Nullability::Nullable;
use crate::dtype::PType;
use crate::dtype::i256;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProvider;
use crate::scalar::DecimalValue;
use crate::scalar::NumericOperator;
use crate::scalar::Scalar;
use crate::validity::Validity;

/// StandardSum an array with an initial value (test-only helper).
fn sum_with_accumulator(array: &ArrayRef, accumulator: &Scalar) -> VortexResult<Scalar> {
    let mut ctx = array_session().create_execution_ctx();
    if accumulator.is_null() {
        return Ok(accumulator.clone());
    }
    if accumulator.is_zero() == Some(true) {
        return standard_sum(array, &mut ctx);
    }

    let sum_dtype = Stat::Sum.dtype(array.dtype()).ok_or_else(|| {
        vortex_error::vortex_err!("StandardSum not supported for dtype: {}", array.dtype())
    })?;

    // For non-float types, try statistics short-circuit with accumulator.
    if !matches!(&sum_dtype, DType::Primitive(p, _) if p.is_float())
        && let Precision::Exact(sum_scalar) = array.statistics().get(Stat::Sum)
    {
        return add_scalars(&sum_dtype, &sum_scalar, accumulator);
    }

    // Compute array sum from zero (also caches stats).
    let array_sum = standard_sum(array, &mut ctx)?;

    // Combine with the accumulator.
    add_scalars(&sum_dtype, &array_sum, accumulator)
}

/// Add two sum scalars with overflow checking.
fn add_scalars(sum_dtype: &DType, lhs: &Scalar, rhs: &Scalar) -> VortexResult<Scalar> {
    if lhs.is_null() || rhs.is_null() {
        return Ok(Scalar::null(sum_dtype.as_nullable()));
    }

    Ok(match sum_dtype {
        DType::Primitive(ptype, _) if ptype.is_float() => {
            let lhs_val = f64::try_from(lhs)?;
            let rhs_val = f64::try_from(rhs)?;
            Scalar::primitive(lhs_val + rhs_val, Nullable)
        }
        DType::Primitive(..) => lhs
            .as_primitive()
            .checked_add(&rhs.as_primitive())
            .map(Scalar::from)
            .unwrap_or_else(|| Scalar::null(sum_dtype.as_nullable())),
        DType::Decimal(..) => lhs
            .as_decimal()
            .checked_binary_numeric(&rhs.as_decimal(), NumericOperator::Add)
            .map(Scalar::from)
            .unwrap_or_else(|| Scalar::null(sum_dtype.as_nullable())),
        _ => unreachable!("StandardSum will always be a decimal or a primitive dtype"),
    })
}

// Multi-batch and reset tests

#[test]
fn sum_multi_batch() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let mut acc = Accumulator::try_new(StandardSum, NumericalAggregateOpts::default(), dtype)?;

    let batch1 = PrimitiveArray::new(buffer![10i32, 20], Validity::NonNullable).into_array();
    acc.accumulate(&batch1, &mut ctx)?;

    let batch2 = PrimitiveArray::new(buffer![3i32, 6, 9], Validity::NonNullable).into_array();
    acc.accumulate(&batch2, &mut ctx)?;

    let result = acc.finish()?;
    assert_eq!(result.as_primitive().typed_value::<i64>(), Some(48));
    Ok(())
}

#[test]
fn sum_finish_resets_state() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let mut acc = Accumulator::try_new(StandardSum, NumericalAggregateOpts::default(), dtype)?;

    let batch1 = PrimitiveArray::new(buffer![10i32, 20], Validity::NonNullable).into_array();
    acc.accumulate(&batch1, &mut ctx)?;
    let result1 = acc.finish()?;
    assert_eq!(result1.as_primitive().typed_value::<i64>(), Some(30));

    let batch2 = PrimitiveArray::new(buffer![3i32, 6, 9], Validity::NonNullable).into_array();
    acc.accumulate(&batch2, &mut ctx)?;
    let result2 = acc.finish()?;
    assert_eq!(result2.as_primitive().typed_value::<i64>(), Some(18));
    Ok(())
}

// State merge tests (vtable-level)

#[test]
fn sum_state_empty_is_null() -> VortexResult<()> {
    // A state that never saw a valid value finalizes to null, and combining empty states
    // stays empty.
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let mut state = StandardSum.empty_partial(&NumericalAggregateOpts::default(), &dtype)?;
    let empty = StandardSum.to_scalar(&state)?;
    StandardSum.combine_partials(&mut state, empty)?;
    assert!(StandardSum.finalize_scalar(&state)?.is_null());
    Ok(())
}

#[test]
fn sum_state_empty_is_identity() -> VortexResult<()> {
    // Combining an empty state into a seen state changes nothing: `{0, false}` is the
    // identity of the `{sum, seen}` monoid.
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let mut state = StandardSum.empty_partial(&NumericalAggregateOpts::default(), &dtype)?;
    StandardSum.combine_partials(&mut state, Scalar::primitive(100i64, Nullable))?;

    let empty = StandardSum
        .to_scalar(&StandardSum.empty_partial(&NumericalAggregateOpts::default(), &dtype)?)?;
    StandardSum.combine_partials(&mut state, empty)?;

    let result = StandardSum.finalize_scalar(&state)?;
    assert_eq!(result.as_primitive().typed_value::<i64>(), Some(100));
    Ok(())
}

#[test]
fn sum_state_overflow_poisons_but_stays_seen() -> VortexResult<()> {
    // Overflow (a null `sum` field) poisons the merge even when combined with later
    // values: the result is null via the sum value, not via `seen`.
    let dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
    let mut overflowed = StandardSum.empty_partial(&NumericalAggregateOpts::default(), &dtype)?;
    StandardSum.combine_partials(&mut overflowed, Scalar::primitive(i64::MAX, Nullable))?;
    StandardSum.combine_partials(&mut overflowed, Scalar::primitive(1i64, Nullable))?;
    let overflowed = StandardSum.to_scalar(&overflowed)?;

    let mut state = StandardSum.empty_partial(&NumericalAggregateOpts::default(), &dtype)?;
    StandardSum.combine_partials(&mut state, Scalar::primitive(5i64, Nullable))?;
    StandardSum.combine_partials(&mut state, overflowed)?;
    StandardSum.combine_partials(&mut state, Scalar::primitive(7i64, Nullable))?;

    assert!(StandardSum.finalize_scalar(&state)?.is_null());
    Ok(())
}

#[test]
fn sum_all_nan_is_zero_not_null() -> VortexResult<()> {
    // NaNs are valid values: with the default `skip_nans` they contribute nothing, but
    // the sum is a genuine `0.0`, unlike an all-null array whose sum is null.
    let arr = PrimitiveArray::new(buffer![f64::NAN, f64::NAN], Validity::NonNullable).into_array();
    let result = standard_sum(&arr, &mut array_session().create_execution_ctx())?;
    assert_eq!(result.as_primitive().typed_value::<f64>(), Some(0.0));
    Ok(())
}

#[test]
fn sum_is_zero_while_standard_sum_is_null() -> VortexResult<()> {
    // The persisted statistic keeps the Sum semantics (zero for all-null) that zone
    // and chunk merging require, while the StandardSum  applies the null-for-empty rule.
    let mut ctx = array_session().create_execution_ctx();
    let arr = PrimitiveArray::from_option_iter([None::<i32>, None, None]).into_array();
    assert_eq!(
        sum(&arr, &mut ctx)?.as_primitive().typed_value::<i64>(),
        Some(0)
    );
    // The cached Sum statistic must not leak through StandardSum's cache short-circuit.
    assert!(standard_sum(&arr, &mut ctx)?.is_null());
    Ok(())
}

#[test]
fn grouped_sum_fallback_empty_and_all_null_groups() -> VortexResult<()> {
    // Bool elements are rejected by the primitive grouped kernel, forcing the generic
    // per-group fallback: empty and all-null groups have null sums there too.
    let mut ctx = array_session().create_execution_ctx();
    let elements = BoolArray::from_iter([Some(true), Some(true), None, None]).into_array();
    let groups = ListViewArray::try_new(
        elements,
        buffer![0i32, 2, 2].into_array(),
        buffer![2i32, 0, 2].into_array(),
        Validity::NonNullable,
    )?
    .into_array();

    let result = run_grouped_sum(&groups, &DType::Bool(Nullable))?;
    let expected = PrimitiveArray::from_option_iter([Some(2u64), None, None]).into_array();
    assert_arrays_eq!(&result, &expected, &mut ctx);
    Ok(())
}

#[test]
fn sum_state_merge() -> VortexResult<()> {
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let mut state = StandardSum.empty_partial(&NumericalAggregateOpts::default(), &dtype)?;

    let scalar1 = Scalar::primitive(100i64, Nullable);
    StandardSum.combine_partials(&mut state, scalar1)?;

    let scalar2 = Scalar::primitive(50i64, Nullable);
    StandardSum.combine_partials(&mut state, scalar2)?;

    let result = StandardSum.finalize_scalar(&state)?;
    StandardSum.reset(&mut state);
    assert_eq!(result.as_primitive().typed_value::<i64>(), Some(150));
    Ok(())
}

// Stats caching test

#[test]
fn sum_stats() -> VortexResult<()> {
    let array = ChunkedArray::try_new(
        vec![
            PrimitiveArray::from_iter([1, 1, 1]).into_array(),
            PrimitiveArray::from_iter([2, 2, 2]).into_array(),
        ],
        DType::Primitive(PType::I32, Nullability::NonNullable),
    )
    .vortex_expect("operation should succeed in test");
    let array = array.into_array();
    // compute sum with accumulator to populate stats
    sum_with_accumulator(&array, &Scalar::primitive(2i64, Nullable))?;

    let sum_without_acc = standard_sum(&array, &mut array_session().create_execution_ctx())?;
    assert_eq!(sum_without_acc, Scalar::primitive(9i64, Nullable));
    Ok(())
}

// Constant float non-multiply test

#[test]
fn sum_constant_float_non_multiply() -> VortexResult<()> {
    let acc = -2048669276050936500000000000f64;
    let array = ConstantArray::new(6.1811675e16f64, 25);
    let result = sum_with_accumulator(&array.into_array(), &Scalar::primitive(acc, Nullable))
        .vortex_expect("operation should succeed in test");
    assert_eq!(
        f64::try_from(&result).vortex_expect("operation should succeed in test"),
        -2048669274505644600000000000f64
    );
    Ok(())
}

// Grouped sum tests

fn run_grouped_sum(groups: &ArrayRef, elem_dtype: &DType) -> VortexResult<ArrayRef> {
    let mut acc = GroupedAccumulator::try_new(
        StandardSum,
        NumericalAggregateOpts::default(),
        elem_dtype.clone(),
    )?;
    let mut ctx = array_session().create_execution_ctx();
    acc.accumulate_list(groups, &mut ctx)?;
    acc.finish()
}

#[test]
fn grouped_sum_fixed_size_list() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let elements =
        PrimitiveArray::new(buffer![1i32, 2, 3, 4, 5, 6], Validity::NonNullable).into_array();
    let groups = FixedSizeListArray::try_new(elements, 3, Validity::NonNullable, 2)?;

    let elem_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let result = run_grouped_sum(&groups.into_array(), &elem_dtype)?;

    let expected = PrimitiveArray::from_option_iter([Some(6i64), Some(15i64)]).into_array();
    assert_arrays_eq!(&result, &expected, &mut ctx);
    Ok(())
}

#[test]
fn grouped_sum_with_null_elements() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let elements =
        PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), None, Some(5), Some(6)])
            .into_array();
    let groups = FixedSizeListArray::try_new(elements, 3, Validity::NonNullable, 2)?;

    let elem_dtype = DType::Primitive(PType::I32, Nullable);
    let result = run_grouped_sum(&groups.into_array(), &elem_dtype)?;

    let expected = PrimitiveArray::from_option_iter([Some(4i64), Some(11i64)]).into_array();
    assert_arrays_eq!(&result, &expected, &mut ctx);
    Ok(())
}

#[test]
fn grouped_sum_with_null_group() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let elements =
        PrimitiveArray::new(buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9], Validity::NonNullable)
            .into_array();
    let validity = Validity::from_iter([true, false, true]);
    let groups = FixedSizeListArray::try_new(elements, 3, validity, 3)?;

    let elem_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let result = run_grouped_sum(&groups.into_array(), &elem_dtype)?;

    let expected = PrimitiveArray::from_option_iter([Some(6i64), None, Some(24i64)]).into_array();
    assert_arrays_eq!(&result, &expected, &mut ctx);
    Ok(())
}

#[test]
fn grouped_sum_all_null_elements_in_group() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let elements =
        PrimitiveArray::from_option_iter([None::<i32>, None, Some(3), Some(4)]).into_array();
    let groups = FixedSizeListArray::try_new(elements, 2, Validity::NonNullable, 2)?;

    let elem_dtype = DType::Primitive(PType::I32, Nullable);
    let result = run_grouped_sum(&groups.into_array(), &elem_dtype)?;

    // The all-null group has a null sum
    let expected = PrimitiveArray::from_option_iter([None, Some(7i64)]).into_array();
    assert_arrays_eq!(&result, &expected, &mut ctx);
    Ok(())
}

#[test]
fn grouped_sum_bool() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let elements: BoolArray = [true, false, true, true, true, true].into_iter().collect();
    let groups = FixedSizeListArray::try_new(elements.into_array(), 3, Validity::NonNullable, 2)?;

    let elem_dtype = DType::Bool(Nullability::NonNullable);
    let result = run_grouped_sum(&groups.into_array(), &elem_dtype)?;

    let expected = PrimitiveArray::from_option_iter([Some(2u64), Some(3u64)]).into_array();
    assert_arrays_eq!(&result, &expected, &mut ctx);
    Ok(())
}

#[test]
fn grouped_sum_finish_resets() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let elem_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let mut acc =
        GroupedAccumulator::try_new(StandardSum, NumericalAggregateOpts::default(), elem_dtype)?;

    let elements1 = PrimitiveArray::new(buffer![1i32, 2, 3, 4], Validity::NonNullable).into_array();
    let groups1 = FixedSizeListArray::try_new(elements1, 2, Validity::NonNullable, 2)?;
    acc.accumulate_list(&groups1.into_array(), &mut ctx)?;
    let result1 = acc.finish()?;

    let expected1 = PrimitiveArray::from_option_iter([Some(3i64), Some(7i64)]).into_array();
    assert_arrays_eq!(&result1, &expected1, &mut ctx);

    let elements2 = PrimitiveArray::new(buffer![10i32, 20], Validity::NonNullable).into_array();
    let groups2 = FixedSizeListArray::try_new(elements2, 2, Validity::NonNullable, 1)?;
    acc.accumulate_list(&groups2.into_array(), &mut ctx)?;
    let result2 = acc.finish()?;

    let expected2 = PrimitiveArray::from_option_iter([Some(30i64)]).into_array();
    assert_arrays_eq!(&result2, &expected2, &mut ctx);
    Ok(())
}

#[test]
fn grouped_sum_listview_out_of_order_offsets_with_null_group() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let elements =
        PrimitiveArray::new(buffer![100i32, 200, 300], Validity::NonNullable).into_array();
    let offsets = PrimitiveArray::new(buffer![2i32, 0, 1], Validity::NonNullable).into_array();
    let sizes = PrimitiveArray::new(buffer![1i32, 1, 1], Validity::NonNullable).into_array();
    let validity = Validity::from_iter([true, false, true]);
    let groups = ListViewArray::try_new(elements, offsets, sizes, validity)?.into_array();

    let elem_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let result = run_grouped_sum(&groups, &elem_dtype)?;

    // group 0 -> elements[2..3] = 300; group 1 -> null; group 2 -> elements[1..2] = 200.
    let expected =
        PrimitiveArray::from_option_iter([Some(300i64), None, Some(200i64)]).into_array();
    assert_arrays_eq!(&result, &expected, &mut ctx);
    Ok(())
}

// Chunked array tests

#[test]
fn sum_chunked_floats_with_nulls() -> VortexResult<()> {
    let chunk1 = PrimitiveArray::from_option_iter(vec![Some(1.5f64), None, Some(3.2), Some(4.8)]);
    let chunk2 = PrimitiveArray::from_option_iter(vec![Some(2.1f64), Some(5.7), None]);
    let chunk3 = PrimitiveArray::from_option_iter(vec![None, Some(1.0f64), Some(2.5), None]);
    let dtype = chunk1.dtype().clone();
    let chunked = ChunkedArray::try_new(
        vec![
            chunk1.into_array(),
            chunk2.into_array(),
            chunk3.into_array(),
        ],
        dtype,
    )?;

    let result = standard_sum(
        &chunked.into_array(),
        &mut array_session().create_execution_ctx(),
    )?;
    assert_eq!(result.as_primitive().as_::<f64>(), Some(20.8));
    Ok(())
}

#[test]
fn sum_chunked_floats_all_nulls_is_null() -> VortexResult<()> {
    let chunk1 = PrimitiveArray::from_option_iter::<f32, _>(vec![None, None, None]);
    let chunk2 = PrimitiveArray::from_option_iter::<f32, _>(vec![None, None]);
    let dtype = chunk1.dtype().clone();
    let chunked = ChunkedArray::try_new(vec![chunk1.into_array(), chunk2.into_array()], dtype)?;
    let result = standard_sum(
        &chunked.into_array(),
        &mut array_session().create_execution_ctx(),
    )?;

    assert!(result.is_null());
    Ok(())
}

#[test]
fn sum_chunked_floats_empty_chunks() -> VortexResult<()> {
    let chunk1 = PrimitiveArray::from_option_iter(vec![Some(10.5f64), Some(20.3)]);
    let chunk2 = ConstantArray::new(Scalar::primitive(0f64, Nullable), 0);
    let chunk3 = PrimitiveArray::from_option_iter(vec![Some(5.2f64)]);
    let dtype = chunk1.dtype().clone();
    let chunked = ChunkedArray::try_new(
        vec![
            chunk1.into_array(),
            chunk2.into_array(),
            chunk3.into_array(),
        ],
        dtype,
    )?;

    let result = standard_sum(
        &chunked.into_array(),
        &mut array_session().create_execution_ctx(),
    )?;
    assert_eq!(result.as_primitive().as_::<f64>(), Some(36.0));
    Ok(())
}

#[test]
fn sum_chunked_int_almost_all_null() -> VortexResult<()> {
    let chunk1 = PrimitiveArray::from_option_iter::<u32, _>(vec![Some(1)]);
    let chunk2 = PrimitiveArray::from_option_iter::<u32, _>(vec![None]);
    let dtype = chunk1.dtype().clone();
    let chunked = ChunkedArray::try_new(vec![chunk1.into_array(), chunk2.into_array()], dtype)?;

    let result = standard_sum(
        &chunked.into_array(),
        &mut array_session().create_execution_ctx(),
    )?;
    assert_eq!(result.as_primitive().as_::<u64>(), Some(1));
    Ok(())
}

#[test]
fn sum_chunked_decimals() -> VortexResult<()> {
    let decimal_dtype = DecimalDType::new(10, 2);
    let chunk1 = DecimalArray::new(
        buffer![100i32, 100i32, 100i32, 100i32, 100i32],
        decimal_dtype,
        Validity::AllValid,
    );
    let chunk2 = DecimalArray::new(
        buffer![200i32, 200i32, 200i32],
        decimal_dtype,
        Validity::AllValid,
    );
    let chunk3 = DecimalArray::new(buffer![300i32, 300i32], decimal_dtype, Validity::AllValid);
    let dtype = chunk1.dtype().clone();
    let chunked = ChunkedArray::try_new(
        vec![
            chunk1.into_array(),
            chunk2.into_array(),
            chunk3.into_array(),
        ],
        dtype,
    )?;

    let result = standard_sum(
        &chunked.into_array(),
        &mut array_session().create_execution_ctx(),
    )?;
    let decimal_result = result.as_decimal();
    assert_eq!(
        decimal_result.decimal_value(),
        Some(DecimalValue::I256(i256::from_i128(1700)))
    );
    Ok(())
}

#[test]
fn sum_chunked_decimals_with_nulls() -> VortexResult<()> {
    let decimal_dtype = DecimalDType::new(10, 2);
    let chunk1 = DecimalArray::new(
        buffer![100i32, 100i32, 100i32],
        decimal_dtype,
        Validity::AllValid,
    );
    let chunk2 = DecimalArray::new(
        buffer![0i32, 0i32],
        decimal_dtype,
        Validity::from_iter([false, false]),
    );
    let chunk3 = DecimalArray::new(buffer![200i32, 200i32], decimal_dtype, Validity::AllValid);
    let dtype = chunk1.dtype().clone();
    let chunked = ChunkedArray::try_new(
        vec![
            chunk1.into_array(),
            chunk2.into_array(),
            chunk3.into_array(),
        ],
        dtype,
    )?;

    let result = standard_sum(
        &chunked.into_array(),
        &mut array_session().create_execution_ctx(),
    )?;
    let decimal_result = result.as_decimal();
    assert_eq!(
        decimal_result.decimal_value(),
        Some(DecimalValue::I256(i256::from_i128(700)))
    );
    Ok(())
}

#[test]
fn sum_chunked_decimals_large() -> VortexResult<()> {
    let decimal_dtype = DecimalDType::new(3, 0);
    let chunk1 = ConstantArray::new(
        Scalar::decimal(
            DecimalValue::I16(500),
            decimal_dtype,
            Nullability::NonNullable,
        ),
        1,
    );
    let chunk2 = ConstantArray::new(
        Scalar::decimal(
            DecimalValue::I16(600),
            decimal_dtype,
            Nullability::NonNullable,
        ),
        1,
    );
    let dtype = chunk1.dtype().clone();
    let chunked = ChunkedArray::try_new(vec![chunk1.into_array(), chunk2.into_array()], dtype)?;

    let result = standard_sum(
        &chunked.into_array(),
        &mut array_session().create_execution_ctx(),
    )?;
    let decimal_result = result.as_decimal();
    assert_eq!(
        decimal_result.decimal_value(),
        Some(DecimalValue::I256(i256::from_i128(1100)))
    );
    assert_eq!(
        result.dtype(),
        &DType::Decimal(DecimalDType::new(13, 0), Nullable)
    );
    Ok(())
}

mod bool_inputs {
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::aggregate_fn::Accumulator;
    use crate::aggregate_fn::AggregateFnVTable;
    use crate::aggregate_fn::DynAccumulator;
    use crate::aggregate_fn::NumericalAggregateOpts;
    use crate::aggregate_fn::fns::standard_sum::StandardSum;
    use crate::aggregate_fn::fns::standard_sum::standard_sum;
    use crate::array_session;
    use crate::arrays::BoolArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::executor::VortexSessionExecute;

    #[test]
    fn sum_bool_all_true() -> VortexResult<()> {
        let arr: BoolArray = [true, true, true].into_iter().collect();
        let result = standard_sum(
            &arr.into_array(),
            &mut array_session().create_execution_ctx(),
        )?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(3));
        Ok(())
    }

    #[test]
    fn sum_bool_mixed() -> VortexResult<()> {
        let arr: BoolArray = [true, false, true, false, true].into_iter().collect();
        let result = standard_sum(
            &arr.into_array(),
            &mut array_session().create_execution_ctx(),
        )?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(3));
        Ok(())
    }

    #[test]
    fn sum_bool_all_false() -> VortexResult<()> {
        let arr: BoolArray = [false, false, false].into_iter().collect();
        let result = standard_sum(
            &arr.into_array(),
            &mut array_session().create_execution_ctx(),
        )?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(0));
        Ok(())
    }

    #[test]
    fn sum_bool_with_nulls() -> VortexResult<()> {
        let arr = BoolArray::from_iter([Some(true), None, Some(true), Some(false)]);
        let result = standard_sum(
            &arr.into_array(),
            &mut array_session().create_execution_ctx(),
        )?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(2));
        Ok(())
    }

    #[test]
    fn sum_bool_all_null() -> VortexResult<()> {
        let arr = BoolArray::from_iter([None::<bool>, None, None]);
        let result = standard_sum(
            &arr.into_array(),
            &mut array_session().create_execution_ctx(),
        )?;

        assert!(result.is_null());
        Ok(())
    }

    #[test]
    fn sum_bool_empty_is_null() -> VortexResult<()> {
        let dtype = DType::Bool(Nullability::NonNullable);
        let mut acc = Accumulator::try_new(StandardSum, NumericalAggregateOpts::default(), dtype)?;
        let result = acc.finish()?;
        assert!(result.is_null());
        Ok(())
    }

    #[test]
    fn sum_bool_finish_resets_state() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let dtype = DType::Bool(Nullability::NonNullable);
        let mut acc = Accumulator::try_new(StandardSum, NumericalAggregateOpts::default(), dtype)?;

        let batch1: BoolArray = [true, true, false].into_iter().collect();
        acc.accumulate(&batch1.into_array(), &mut ctx)?;
        let result1 = acc.finish()?;
        assert_eq!(result1.as_primitive().typed_value::<u64>(), Some(2));

        let batch2: BoolArray = [false, true].into_iter().collect();
        acc.accumulate(&batch2.into_array(), &mut ctx)?;
        let result2 = acc.finish()?;
        assert_eq!(result2.as_primitive().typed_value::<u64>(), Some(1));
        Ok(())
    }

    #[test]
    fn sum_bool_return_dtype() -> VortexResult<()> {
        let dtype = StandardSum
            .return_dtype(
                &NumericalAggregateOpts::default(),
                &DType::Bool(Nullability::NonNullable),
            )
            .unwrap();
        assert_eq!(dtype, DType::Primitive(PType::U64, Nullability::Nullable));
        Ok(())
    }

    #[test]
    fn sum_boolean_from_iter() -> VortexResult<()> {
        let arr = BoolArray::from_iter([true, false, false, true]).into_array();
        let result = standard_sum(&arr, &mut array_session().create_execution_ctx())?;
        assert_eq!(result.as_primitive().as_::<i32>(), Some(2));
        Ok(())
    }
}

mod constant_inputs {
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::fns::standard_sum::standard_sum;
    use crate::array_session;
    use crate::arrays::ConstantArray;
    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::Nullability;
    use crate::dtype::Nullability::Nullable;
    use crate::dtype::PType;
    use crate::dtype::i256;
    use crate::expr::stats::Stat;
    use crate::scalar::DecimalValue;
    use crate::scalar::Scalar;

    #[test]
    fn sum_constant_unsigned() -> VortexResult<()> {
        let array = ConstantArray::new(5u64, 10).into_array();
        let result = standard_sum(&array, &mut array_session().create_execution_ctx())?;
        assert_eq!(result, 50u64.into());
        Ok(())
    }

    #[test]
    fn sum_constant_signed() -> VortexResult<()> {
        let array = ConstantArray::new(-5i64, 10).into_array();
        let result = standard_sum(&array, &mut array_session().create_execution_ctx())?;
        assert_eq!(result, (-50i64).into());
        Ok(())
    }

    #[test]
    fn sum_constant_nullable_value() -> VortexResult<()> {
        let array = ConstantArray::new(Scalar::null(DType::Primitive(PType::U32, Nullable)), 10)
            .into_array();
        let result = standard_sum(&array, &mut array_session().create_execution_ctx())?;

        assert_eq!(result, Scalar::null(DType::Primitive(PType::U64, Nullable)));
        Ok(())
    }

    #[test]
    fn sum_constant_bool_false() -> VortexResult<()> {
        let array = ConstantArray::new(false, 10).into_array();
        let result = standard_sum(&array, &mut array_session().create_execution_ctx())?;
        assert_eq!(result, 0u64.into());
        Ok(())
    }

    #[test]
    fn sum_constant_bool_true() -> VortexResult<()> {
        let array = ConstantArray::new(true, 10).into_array();
        let result = standard_sum(&array, &mut array_session().create_execution_ctx())?;
        assert_eq!(result, 10u64.into());
        Ok(())
    }

    #[test]
    fn sum_constant_bool_null() -> VortexResult<()> {
        let array = ConstantArray::new(Scalar::null(DType::Bool(Nullable)), 10).into_array();
        let result = standard_sum(&array, &mut array_session().create_execution_ctx())?;
        assert_eq!(result, Scalar::null(DType::Primitive(PType::U64, Nullable)));
        Ok(())
    }

    #[test]
    fn sum_constant_decimal() -> VortexResult<()> {
        let decimal_dtype = DecimalDType::new(10, 2);
        let array = ConstantArray::new(
            Scalar::decimal(
                DecimalValue::I64(100),
                decimal_dtype,
                Nullability::NonNullable,
            ),
            5,
        )
        .into_array();

        let result = standard_sum(&array, &mut array_session().create_execution_ctx())?;

        assert_eq!(
            result.as_decimal().decimal_value(),
            Some(DecimalValue::I256(i256::from_i128(500)))
        );
        assert_eq!(result.dtype(), &Stat::Sum.dtype(array.dtype()).unwrap());
        Ok(())
    }

    #[test]
    fn sum_constant_decimal_null() -> VortexResult<()> {
        let decimal_dtype = DecimalDType::new(10, 2);
        let array = ConstantArray::new(Scalar::null(DType::Decimal(decimal_dtype, Nullable)), 10)
            .into_array();

        let result = standard_sum(&array, &mut array_session().create_execution_ctx())?;
        assert_eq!(
            result,
            Scalar::null(DType::Decimal(DecimalDType::new(20, 2), Nullable))
        );
        Ok(())
    }

    #[test]
    fn sum_constant_decimal_large_value() -> VortexResult<()> {
        let decimal_dtype = DecimalDType::new(10, 2);
        let array = ConstantArray::new(
            Scalar::decimal(
                DecimalValue::I64(999_999_999),
                decimal_dtype,
                Nullability::NonNullable,
            ),
            100,
        )
        .into_array();

        let result = standard_sum(&array, &mut array_session().create_execution_ctx())?;
        assert_eq!(
            result.as_decimal().decimal_value(),
            Some(DecimalValue::I256(i256::from_i128(99_999_999_900)))
        );
        Ok(())
    }
}

mod decimal_inputs {
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::AggregateFnVTable;
    use crate::aggregate_fn::NumericalAggregateOpts;
    use crate::aggregate_fn::fns::standard_sum::StandardSum;
    use crate::aggregate_fn::fns::standard_sum::standard_sum;
    use crate::array_session;
    use crate::arrays::DecimalArray;
    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::Nullability;
    use crate::dtype::Nullability::Nullable;
    use crate::dtype::i256;
    use crate::scalar::DecimalValue;
    use crate::scalar::Scalar;
    use crate::scalar::ScalarValue;
    use crate::validity::Validity;

    #[test]
    fn sum_decimal_basic() -> VortexResult<()> {
        let decimal = DecimalArray::new(
            buffer![100i32, 200i32, 300i32],
            DecimalDType::new(4, 2),
            Validity::AllValid,
        );

        let result = standard_sum(
            &decimal.into_array(),
            &mut array_session().create_execution_ctx(),
        )?;

        let expected = Scalar::try_new(
            DType::Decimal(DecimalDType::new(14, 2), Nullability::NonNullable),
            Some(ScalarValue::from(DecimalValue::from(600i32))),
        )?;

        assert_eq!(result, expected);
        Ok(())
    }

    #[test]
    fn sum_decimal_with_nulls() -> VortexResult<()> {
        let decimal = DecimalArray::new(
            buffer![100i32, 200i32, 300i32, 400i32],
            DecimalDType::new(4, 2),
            Validity::from_iter([true, false, true, true]),
        );

        let result = standard_sum(
            &decimal.into_array(),
            &mut array_session().create_execution_ctx(),
        )?;

        let expected = Scalar::try_new(
            DType::Decimal(DecimalDType::new(14, 2), Nullable),
            Some(ScalarValue::from(DecimalValue::from(800i32))),
        )?;

        assert_eq!(result, expected);
        Ok(())
    }

    #[test]
    fn sum_decimal_negative_values() -> VortexResult<()> {
        let decimal = DecimalArray::new(
            buffer![100i32, -200i32, 300i32, -50i32],
            DecimalDType::new(4, 2),
            Validity::AllValid,
        );

        let result = standard_sum(
            &decimal.into_array(),
            &mut array_session().create_execution_ctx(),
        )?;

        let expected = Scalar::try_new(
            DType::Decimal(DecimalDType::new(14, 2), Nullability::NonNullable),
            Some(ScalarValue::from(DecimalValue::from(150i32))),
        )?;

        assert_eq!(result, expected);
        Ok(())
    }

    #[test]
    fn sum_decimal_near_i32_max() -> VortexResult<()> {
        let near_max = i32::MAX - 1000;
        let decimal = DecimalArray::new(
            buffer![near_max, 500i32, 400i32],
            DecimalDType::new(10, 2),
            Validity::AllValid,
        );

        let result = standard_sum(
            &decimal.into_array(),
            &mut array_session().create_execution_ctx(),
        )?;

        let expected_sum = near_max as i64 + 500 + 400;
        let expected = Scalar::try_new(
            DType::Decimal(DecimalDType::new(20, 2), Nullability::NonNullable),
            Some(ScalarValue::from(DecimalValue::from(expected_sum))),
        )?;

        assert_eq!(result, expected);
        Ok(())
    }

    #[test]
    fn sum_decimal_large_i64_values() -> VortexResult<()> {
        let large_val = i64::MAX / 4;
        let decimal = DecimalArray::new(
            buffer![large_val, large_val, large_val, large_val + 1],
            DecimalDType::new(19, 0),
            Validity::AllValid,
        );

        let result = standard_sum(
            &decimal.into_array(),
            &mut array_session().create_execution_ctx(),
        )?;

        let expected_sum = (large_val as i128) * 4 + 1;
        let expected = Scalar::try_new(
            DType::Decimal(DecimalDType::new(29, 0), Nullability::NonNullable),
            Some(ScalarValue::from(DecimalValue::from(expected_sum))),
        )?;

        assert_eq!(result, expected);
        Ok(())
    }

    #[test]
    fn sum_decimal_preserves_scale() -> VortexResult<()> {
        let decimal = DecimalArray::new(
            buffer![12345i32, 67890i32, 11111i32],
            DecimalDType::new(6, 4),
            Validity::AllValid,
        );

        let result = standard_sum(
            &decimal.into_array(),
            &mut array_session().create_execution_ctx(),
        )?;

        let expected = Scalar::try_new(
            DType::Decimal(DecimalDType::new(16, 4), Nullability::NonNullable),
            Some(ScalarValue::from(DecimalValue::from(91346i32))),
        )?;

        assert_eq!(result, expected);
        Ok(())
    }

    #[test]
    fn sum_decimal_single_value() -> VortexResult<()> {
        let decimal =
            DecimalArray::new(buffer![42i32], DecimalDType::new(3, 1), Validity::AllValid);

        let result = standard_sum(
            &decimal.into_array(),
            &mut array_session().create_execution_ctx(),
        )?;

        let expected = Scalar::try_new(
            DType::Decimal(DecimalDType::new(13, 1), Nullability::NonNullable),
            Some(ScalarValue::from(DecimalValue::from(42i32))),
        )?;

        assert_eq!(result, expected);
        Ok(())
    }

    #[test]
    fn sum_decimal_all_nulls_except_one() -> VortexResult<()> {
        let decimal = DecimalArray::new(
            buffer![100i32, 200i32, 300i32, 400i32],
            DecimalDType::new(4, 2),
            Validity::from_iter([false, false, true, false]),
        );

        let result = standard_sum(
            &decimal.into_array(),
            &mut array_session().create_execution_ctx(),
        )?;

        let expected = Scalar::try_new(
            DType::Decimal(DecimalDType::new(14, 2), Nullable),
            Some(ScalarValue::from(DecimalValue::from(300i32))),
        )?;

        assert_eq!(result, expected);
        Ok(())
    }

    #[test]
    fn sum_decimal_overflow_detection() -> VortexResult<()> {
        let max_val = i128::MAX / 2;
        let decimal = DecimalArray::new(
            buffer![max_val, max_val, max_val],
            DecimalDType::new(38, 0),
            Validity::AllValid,
        );

        let result = standard_sum(
            &decimal.into_array(),
            &mut array_session().create_execution_ctx(),
        )?;

        let expected_sum =
            i256::from_i128(max_val) + i256::from_i128(max_val) + i256::from_i128(max_val);
        let expected = Scalar::try_new(
            DType::Decimal(DecimalDType::new(48, 0), Nullability::NonNullable),
            Some(ScalarValue::from(DecimalValue::from(expected_sum))),
        )?;

        assert_eq!(result, expected);
        Ok(())
    }

    #[test]
    fn sum_decimal_i256_overflow() -> VortexResult<()> {
        let decimal_dtype = DecimalDType::new(76, 0);
        let decimal = DecimalArray::new(
            buffer![i256::MAX, i256::MAX, i256::MAX],
            decimal_dtype,
            Validity::AllValid,
        );

        assert_eq!(
            standard_sum(
                &decimal.into_array(),
                &mut array_session().create_execution_ctx()
            )
            .vortex_expect("operation should succeed in test"),
            Scalar::null(DType::Decimal(decimal_dtype, Nullable))
        );
        Ok(())
    }

    #[test]
    fn sum_decimal_near_precision_boundary() -> VortexResult<()> {
        // Input precision 4 → return precision min(76, 4+10) = 14.
        // Native type for precision 14 is I64 (max precision 18), so 14 < 18.
        // Use combine_partials to push state near (but under) 10^14.
        let input_dtype = DType::Decimal(DecimalDType::new(4, 0), Nullability::NonNullable);
        let mut state =
            StandardSum.empty_partial(&NumericalAggregateOpts::default(), &input_dtype)?;

        let near_limit = Scalar::decimal(
            DecimalValue::from(99_999_999_999_990i64),
            DecimalDType::new(14, 0),
            Nullable,
        );
        StandardSum.combine_partials(&mut state, near_limit)?;

        // Add a small value that keeps us just under 10^14.
        let small = Scalar::decimal(DecimalValue::from(9i64), DecimalDType::new(14, 0), Nullable);
        StandardSum.combine_partials(&mut state, small)?;

        let result = StandardSum.finalize_scalar(&state)?;
        assert!(!result.is_null());
        assert_eq!(
            result.as_decimal().decimal_value(),
            Some(DecimalValue::I256(i256::from_i128(99_999_999_999_999)))
        );
        Ok(())
    }

    #[test]
    fn sum_decimal_precision_overflow_within_i256() -> VortexResult<()> {
        // Input precision 4 → return precision 14. Native I64 (max 18).
        // The max representable value for precision 14 is 10^14 - 1.
        // When the sum reaches exactly 10^14, fits_in_precision fails even though
        // i256 arithmetic does not overflow. This tests the precision-based
        // saturation path in combine_partials.
        let input_dtype = DType::Decimal(DecimalDType::new(4, 0), Nullability::NonNullable);
        let mut state =
            StandardSum.empty_partial(&NumericalAggregateOpts::default(), &input_dtype)?;

        let near_limit = Scalar::decimal(
            DecimalValue::from(99_999_999_999_999i64),
            DecimalDType::new(14, 0),
            Nullable,
        );
        StandardSum.combine_partials(&mut state, near_limit)?;

        // Push the sum to exactly 10^14, exceeding precision 14.
        let one_more =
            Scalar::decimal(DecimalValue::from(1i64), DecimalDType::new(14, 0), Nullable);
        StandardSum.combine_partials(&mut state, one_more)?;

        let result = StandardSum.finalize_scalar(&state)?;
        assert!(result.is_null());
        assert_eq!(
            result.dtype(),
            &DType::Decimal(DecimalDType::new(14, 0), Nullable)
        );
        Ok(())
    }

    #[test]
    fn sum_decimal_precision_overflow_negative() -> VortexResult<()> {
        // Same setup but with negative values: sum reaches -10^14.
        let input_dtype = DType::Decimal(DecimalDType::new(4, 0), Nullability::NonNullable);
        let mut state =
            StandardSum.empty_partial(&NumericalAggregateOpts::default(), &input_dtype)?;

        let near_limit = Scalar::decimal(
            DecimalValue::from(-99_999_999_999_999i64),
            DecimalDType::new(14, 0),
            Nullable,
        );
        StandardSum.combine_partials(&mut state, near_limit)?;

        let one_more = Scalar::decimal(
            DecimalValue::from(-1i64),
            DecimalDType::new(14, 0),
            Nullable,
        );
        StandardSum.combine_partials(&mut state, one_more)?;

        let result = StandardSum.finalize_scalar(&state)?;
        assert!(result.is_null());
        Ok(())
    }

    #[test]
    fn sum_decimal_accumulate_precision_overflow() -> VortexResult<()> {
        // Test precision overflow via the accumulate_decimal path (not combine_partials).
        // Input precision 28 (I128 storage) → return precision min(76, 38) = 38.
        // Native for precision 38 is I128 (max 38), so 38 = 38.
        // Use precision 27 → return 37. Native for 37 is I128 (max 38), so 37 < 38.
        //
        // We use combine_partials to get the state close to 10^37, then accumulate
        // a real array that pushes it over.
        let input_dtype = DType::Decimal(DecimalDType::new(27, 0), Nullability::NonNullable);
        let return_dtype = DecimalDType::new(37, 0);
        let mut state =
            StandardSum.empty_partial(&NumericalAggregateOpts::default(), &input_dtype)?;

        // Set state to 10^37 - 1 via combine_partials.
        let near_limit_val: i128 = 10i128.pow(37) - 1;
        let near_limit =
            Scalar::decimal(DecimalValue::from(near_limit_val), return_dtype, Nullable);
        StandardSum.combine_partials(&mut state, near_limit)?;

        // Now accumulate a real i128 array with a single element = 1 to overflow precision.
        let decimal =
            DecimalArray::new(buffer![1i128], DecimalDType::new(27, 0), Validity::AllValid);

        // Drive accumulate through the vtable directly.
        let columnar = crate::Columnar::Canonical(crate::Canonical::Decimal(decimal));
        let mut ctx = array_session().create_execution_ctx();
        StandardSum.accumulate(&mut state, &columnar, &mut ctx)?;

        let result = StandardSum.finalize_scalar(&state)?;
        assert!(result.is_null());
        Ok(())
    }
}

mod primitive_inputs {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::Accumulator;
    use crate::aggregate_fn::DynAccumulator;
    use crate::aggregate_fn::NumericalAggregateOpts;
    use crate::aggregate_fn::fns::standard_sum::StandardSum;
    use crate::aggregate_fn::fns::standard_sum::standard_sum;
    use crate::array_session;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::Nullability::Nullable;
    use crate::dtype::PType;
    use crate::expr::stats::Precision;
    use crate::expr::stats::Stat;
    use crate::scalar::Scalar;
    use crate::scalar::ScalarValue;
    use crate::validity::Validity;

    #[test]
    fn sum_i32() -> VortexResult<()> {
        let arr = PrimitiveArray::new(buffer![1i32, 2, 3, 4], Validity::NonNullable).into_array();
        let result = standard_sum(&arr, &mut array_session().create_execution_ctx())?;
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(10));
        Ok(())
    }

    #[test]
    fn sum_u8() -> VortexResult<()> {
        let arr = PrimitiveArray::new(buffer![10u8, 20, 30], Validity::NonNullable).into_array();
        let result = standard_sum(&arr, &mut array_session().create_execution_ctx())?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(60));
        Ok(())
    }

    #[test]
    fn sum_f64() -> VortexResult<()> {
        let arr =
            PrimitiveArray::new(buffer![1.5f64, 2.5, 3.0], Validity::NonNullable).into_array();
        let result = standard_sum(&arr, &mut array_session().create_execution_ctx())?;
        assert_eq!(result.as_primitive().typed_value::<f64>(), Some(7.0));
        Ok(())
    }

    #[test]
    fn sum_with_nulls() -> VortexResult<()> {
        let arr = PrimitiveArray::from_option_iter([Some(2i32), None, Some(4)]).into_array();
        let result = standard_sum(&arr, &mut array_session().create_execution_ctx())?;
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(6));
        Ok(())
    }

    #[test]
    fn sum_multiple_null_runs() -> VortexResult<()> {
        // Several disjoint valid runs separated by nulls exercise the per-run fold.
        let arr = PrimitiveArray::from_option_iter([
            Some(1i32),
            Some(2),
            None,
            None,
            Some(3),
            None,
            Some(4),
            Some(5),
            Some(6),
        ])
        .into_array();
        let result = standard_sum(&arr, &mut array_session().create_execution_ctx())?;
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(21));
        Ok(())
    }

    #[test]
    fn sum_all_null() -> VortexResult<()> {
        let arr = PrimitiveArray::from_option_iter([None::<i32>, None, None]).into_array();
        let result = standard_sum(&arr, &mut array_session().create_execution_ctx())?;

        assert!(result.is_null());
        Ok(())
    }

    #[test]
    fn sum_all_invalid_float() -> VortexResult<()> {
        let arr = PrimitiveArray::from_option_iter::<f32, _>([None, None, None]).into_array();
        let result = standard_sum(&arr, &mut array_session().create_execution_ctx())?;
        assert_eq!(result, Scalar::null(DType::Primitive(PType::F64, Nullable)));
        Ok(())
    }

    #[test]
    fn sum_buffer_i32() -> VortexResult<()> {
        let arr = buffer![1, 1, 1, 1].into_array();
        let result = standard_sum(&arr, &mut array_session().create_execution_ctx())?;
        assert_eq!(result.as_primitive().as_::<i32>(), Some(4));
        Ok(())
    }

    #[test]
    fn sum_buffer_f64() -> VortexResult<()> {
        let arr = buffer![1., 1., 1., 1.].into_array();
        let result = standard_sum(&arr, &mut array_session().create_execution_ctx())?;
        assert_eq!(result.as_primitive().as_::<f32>(), Some(4.));
        Ok(())
    }

    #[test]
    fn sum_empty_is_null() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(StandardSum, NumericalAggregateOpts::default(), dtype)?;
        let result = acc.finish()?;

        assert!(result.is_null());
        Ok(())
    }

    #[test]
    fn sum_empty_f64_is_null() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::F64, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(StandardSum, NumericalAggregateOpts::default(), dtype)?;
        let result = acc.finish()?;
        assert!(result.is_null());
        Ok(())
    }

    #[test]
    fn sum_f64_with_nan() -> VortexResult<()> {
        let arr = PrimitiveArray::new(
            buffer![1.0f64, f64::NAN, 2.0, f64::NAN, 3.0],
            Validity::NonNullable,
        )
        .into_array();
        let result = standard_sum(&arr, &mut array_session().create_execution_ctx())?;
        assert_eq!(result.as_primitive().typed_value::<f64>(), Some(6.0));
        Ok(())
    }

    #[test]
    fn sum_f32_with_nan() -> VortexResult<()> {
        let arr =
            PrimitiveArray::new(buffer![1.0f32, f32::NAN, 4.0], Validity::NonNullable).into_array();
        let result = standard_sum(&arr, &mut array_session().create_execution_ctx())?;
        assert_eq!(result.as_primitive().typed_value::<f64>(), Some(5.0));
        Ok(())
    }

    #[test]
    fn sum_f64_with_nan_and_nulls() -> VortexResult<()> {
        let arr = PrimitiveArray::from_option_iter([Some(1.0f64), None, Some(f64::NAN), Some(3.0)])
            .into_array();
        let result = standard_sum(&arr, &mut array_session().create_execution_ctx())?;
        assert_eq!(result.as_primitive().typed_value::<f64>(), Some(4.0));
        Ok(())
    }

    #[test]
    fn sum_all_nan() -> VortexResult<()> {
        let arr =
            PrimitiveArray::new(buffer![f64::NAN, f64::NAN], Validity::NonNullable).into_array();
        let result = standard_sum(&arr, &mut array_session().create_execution_ctx())?;
        assert_eq!(result.as_primitive().typed_value::<f64>(), Some(0.0));
        Ok(())
    }

    /// StandardSum an array with explicit [`NumericalAggregateOpts`] (test-only helper).
    fn sum_with_options(
        arr: &crate::ArrayRef,
        options: NumericalAggregateOpts,
    ) -> VortexResult<Scalar> {
        let mut acc = Accumulator::try_new(StandardSum, options, arr.dtype().clone())?;
        acc.accumulate(arr, &mut array_session().create_execution_ctx())?;
        acc.finish()
    }

    #[test]
    fn sum_f64_with_nan_not_skipping() -> VortexResult<()> {
        let arr =
            PrimitiveArray::new(buffer![1.0f64, f64::NAN, 2.0], Validity::NonNullable).into_array();
        let result = sum_with_options(&arr, NumericalAggregateOpts::include_nans())?;
        assert!(result.as_primitive().typed_value::<f64>().unwrap().is_nan());
        Ok(())
    }

    #[test]
    fn sum_f64_without_nan_not_skipping() -> VortexResult<()> {
        let arr =
            PrimitiveArray::new(buffer![1.0f64, 2.0, 3.0], Validity::NonNullable).into_array();
        let result = sum_with_options(&arr, NumericalAggregateOpts::include_nans())?;
        assert_eq!(result.as_primitive().typed_value::<f64>(), Some(6.0));
        Ok(())
    }

    #[test]
    fn sum_not_skipping_shortcircuits_on_exact_nan_count_stat() -> VortexResult<()> {
        // The array has no NaNs; a planted exact NaNCount stat proves the NaN poisoning came
        // from the stat rather than a scan.
        let arr =
            PrimitiveArray::new(buffer![1.0f64, 2.0, 3.0], Validity::NonNullable).into_array();
        arr.statistics()
            .set(Stat::NaNCount, Precision::Exact(ScalarValue::from(1u64)));
        let result = sum_with_options(&arr, NumericalAggregateOpts::include_nans())?;
        assert!(result.as_primitive().typed_value::<f64>().unwrap().is_nan());
        Ok(())
    }

    #[test]
    fn sum_not_skipping_uses_cached_sum_when_nan_free() -> VortexResult<()> {
        // With an exact NaNCount of zero, the planted exact StandardSum stat is usable as-is.
        let arr =
            PrimitiveArray::new(buffer![1.0f64, 2.0, 3.0], Validity::NonNullable).into_array();
        arr.statistics()
            .set(Stat::NaNCount, Precision::Exact(ScalarValue::from(0u64)));
        arr.statistics()
            .set(Stat::Sum, Precision::Exact(ScalarValue::from(42.0f64)));
        arr.statistics()
            .set(Stat::NullCount, Precision::Exact(ScalarValue::from(0u64)));
        let result = sum_with_options(&arr, NumericalAggregateOpts::include_nans())?;
        assert_eq!(result.as_primitive().typed_value::<f64>(), Some(42.0));
        Ok(())
    }

    #[test]
    fn sum_constant_nan() -> VortexResult<()> {
        let arr = ConstantArray::new(f64::NAN, 4).into_array();
        // NaN constants are skipped by default and poison the sum otherwise.
        let result = sum_with_options(&arr, NumericalAggregateOpts::default())?;
        assert_eq!(result.as_primitive().typed_value::<f64>(), Some(0.0));

        let result = sum_with_options(&arr, NumericalAggregateOpts::include_nans())?;
        assert!(result.as_primitive().typed_value::<f64>().unwrap().is_nan());
        Ok(())
    }

    #[test]
    fn sum_f64_with_infinity() -> VortexResult<()> {
        let batch = PrimitiveArray::new(
            buffer![1.0f64, f64::INFINITY, f64::NEG_INFINITY, 2.0],
            Validity::NonNullable,
        )
        .into_array();
        let acc = standard_sum(&batch, &mut array_session().create_execution_ctx())?;
        // INFINITY + NEG_INFINITY = NaN, which is treated as saturated
        assert!(acc.as_primitive().typed_value::<f64>().unwrap().is_nan());

        let mut acc = Accumulator::try_new(
            StandardSum,
            NumericalAggregateOpts::default(),
            DType::Primitive(PType::F64, Nullability::NonNullable),
        )?;
        acc.accumulate(&batch, &mut array_session().create_execution_ctx())?;
        assert!(acc.is_saturated());
        Ok(())
    }

    #[test]
    fn sum_checked_overflow() -> VortexResult<()> {
        let arr = PrimitiveArray::new(buffer![i64::MAX, 1i64], Validity::NonNullable).into_array();
        let result = standard_sum(&arr, &mut array_session().create_execution_ctx())?;
        assert!(result.is_null());
        Ok(())
    }

    #[test]
    fn sum_checked_overflow_is_saturated() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(StandardSum, NumericalAggregateOpts::default(), dtype)?;
        assert!(!acc.is_saturated());

        let batch =
            PrimitiveArray::new(buffer![i64::MAX, 1i64], Validity::NonNullable).into_array();
        acc.accumulate(&batch, &mut array_session().create_execution_ctx())?;
        assert!(acc.is_saturated());

        // finish resets state, clearing saturation
        drop(acc.finish()?);
        assert!(!acc.is_saturated());
        Ok(())
    }
}
