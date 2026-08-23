// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::buffer;
use vortex_error::VortexResult;

use super::SumV2;
use super::sum_v2;
use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::DynAccumulator;
use crate::aggregate_fn::DynGroupedAccumulator;
use crate::aggregate_fn::GroupedAccumulator;
use crate::aggregate_fn::NumericalAggregateOpts;
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
use crate::scalar::DecimalValue;
use crate::scalar::Scalar;
use crate::validity::Validity;

fn run_sum_v2(array: &ArrayRef) -> VortexResult<Scalar> {
    sum_v2(array, &mut array_session().create_execution_ctx())
}

fn sum_v2_with_options(array: &ArrayRef, options: NumericalAggregateOpts) -> VortexResult<Scalar> {
    let mut acc = Accumulator::try_new(SumV2, options, array.dtype().clone())?;
    acc.accumulate(array, &mut array_session().create_execution_ctx())?;
    acc.finish()
}

// Scalar semantics

#[test]
fn sum_v2_i32() -> VortexResult<()> {
    let arr = PrimitiveArray::new(buffer![1i32, 2, 3, 4], Validity::NonNullable).into_array();
    let result = run_sum_v2(&arr)?;
    assert_eq!(result.as_primitive().typed_value::<i64>(), Some(10));
    Ok(())
}

#[test]
fn sum_v2_with_nulls() -> VortexResult<()> {
    let arr = PrimitiveArray::from_option_iter([Some(2i32), None, Some(4)]).into_array();
    let result = run_sum_v2(&arr)?;
    assert_eq!(result.as_primitive().typed_value::<i64>(), Some(6));
    Ok(())
}

#[test]
fn sum_v2_valid_zero_is_zero() -> VortexResult<()> {
    let arr = PrimitiveArray::from_option_iter([Some(1i32), None, Some(-1)]).into_array();
    let result = run_sum_v2(&arr)?;
    assert_eq!(result, Scalar::primitive(0i64, Nullable));
    Ok(())
}

#[test]
fn sum_v2_all_null_is_null() -> VortexResult<()> {
    let arr = PrimitiveArray::from_option_iter([None::<i32>, None, None]).into_array();
    let result = run_sum_v2(&arr)?;
    assert!(result.is_null());
    assert_eq!(result.dtype(), &DType::Primitive(PType::I64, Nullable));
    Ok(())
}

#[test]
fn sum_v2_all_null_float_is_null() -> VortexResult<()> {
    let arr = PrimitiveArray::from_option_iter::<f32, _>([None, None, None]).into_array();
    let result = run_sum_v2(&arr)?;
    assert_eq!(result, Scalar::null(DType::Primitive(PType::F64, Nullable)));
    Ok(())
}

#[test]
fn sum_v2_empty_is_null() -> VortexResult<()> {
    let arr = PrimitiveArray::new(buffer![1i32, 2], Validity::NonNullable)
        .into_array()
        .slice(0..0)?;
    let result = run_sum_v2(&arr)?;
    assert!(result.is_null());
    Ok(())
}

#[test]
fn sum_v2_no_batches_is_null() -> VortexResult<()> {
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let mut acc = Accumulator::try_new(SumV2, NumericalAggregateOpts::default(), dtype)?;
    let result = acc.finish()?;
    assert!(result.is_null());
    assert_eq!(result.dtype(), &DType::Primitive(PType::I64, Nullable));
    Ok(())
}

#[test]
fn sum_v2_no_batches_f64_is_null() -> VortexResult<()> {
    let dtype = DType::Primitive(PType::F64, Nullability::NonNullable);
    let mut acc = Accumulator::try_new(SumV2, NumericalAggregateOpts::default(), dtype)?;
    let result = acc.finish()?;
    assert!(result.is_null());
    Ok(())
}

#[test]
fn sum_v2_overflow_is_null() -> VortexResult<()> {
    let arr = PrimitiveArray::new(buffer![i64::MAX, 1i64], Validity::NonNullable).into_array();
    let result = run_sum_v2(&arr)?;
    assert!(result.is_null());
    Ok(())
}

#[test]
fn sum_v2_finish_resets_state() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DType::Primitive(PType::I32, Nullable);
    let mut acc = Accumulator::try_new(SumV2, NumericalAggregateOpts::default(), dtype)?;

    let batch1 = PrimitiveArray::from_option_iter([Some(10i32), Some(20)]).into_array();
    acc.accumulate(&batch1, &mut ctx)?;
    assert_eq!(acc.finish()?.as_primitive().typed_value::<i64>(), Some(30));

    // After the reset the accumulator is empty again, so an all-null batch sums to null.
    let batch2 = PrimitiveArray::from_option_iter([None::<i32>, None]).into_array();
    acc.accumulate(&batch2, &mut ctx)?;
    assert!(acc.finish()?.is_null());
    Ok(())
}

// NaN semantics

#[test]
fn sum_v2_all_nan_skipping_is_zero() -> VortexResult<()> {
    // NaNs are valid values even when skipped, so the sum is an empty-of-contributions zero
    // rather than null.
    let arr = PrimitiveArray::new(buffer![f64::NAN, f64::NAN], Validity::NonNullable).into_array();
    let result = run_sum_v2(&arr)?;
    assert_eq!(result.as_primitive().typed_value::<f64>(), Some(0.0));
    Ok(())
}

#[test]
fn sum_v2_nan_not_skipping_is_nan() -> VortexResult<()> {
    let arr =
        PrimitiveArray::new(buffer![1.0f64, f64::NAN, 2.0], Validity::NonNullable).into_array();
    let result = sum_v2_with_options(&arr, NumericalAggregateOpts::include_nans())?;
    assert!(
        result
            .as_primitive()
            .typed_value::<f64>()
            .is_some_and(f64::is_nan)
    );
    Ok(())
}

#[test]
fn sum_v2_nan_and_nulls_skipping() -> VortexResult<()> {
    let arr = PrimitiveArray::from_option_iter([Some(1.0f64), None, Some(f64::NAN), Some(3.0)])
        .into_array();
    let result = run_sum_v2(&arr)?;
    assert_eq!(result.as_primitive().typed_value::<f64>(), Some(4.0));
    Ok(())
}

// Chunked arrays (partial merges)

#[test]
fn sum_v2_chunked_null_chunk_is_merge_identity() -> VortexResult<()> {
    // The all-null chunk comes first: its empty partial must not clobber the later values.
    let chunk1 = PrimitiveArray::from_option_iter([None::<i32>, None]);
    let chunk2 = PrimitiveArray::from_option_iter([Some(3i32), Some(4)]);
    let dtype = chunk1.dtype().clone();
    let chunked = ChunkedArray::try_new(vec![chunk1.into_array(), chunk2.into_array()], dtype)?;
    let result = run_sum_v2(&chunked.into_array())?;
    assert_eq!(result.as_primitive().typed_value::<i64>(), Some(7));
    Ok(())
}

#[test]
fn sum_v2_chunked_all_nulls_is_null() -> VortexResult<()> {
    let chunk1 = PrimitiveArray::from_option_iter::<f32, _>([None, None, None]);
    let chunk2 = PrimitiveArray::from_option_iter::<f32, _>([None, None]);
    let dtype = chunk1.dtype().clone();
    let chunked = ChunkedArray::try_new(vec![chunk1.into_array(), chunk2.into_array()], dtype)?;
    let result = run_sum_v2(&chunked.into_array())?;
    assert!(result.is_null());
    Ok(())
}

// Constant arrays

#[test]
fn sum_v2_constant() -> VortexResult<()> {
    let array = ConstantArray::new(5u64, 10).into_array();
    let result = run_sum_v2(&array)?;
    assert_eq!(result.as_primitive().typed_value::<u64>(), Some(50));
    Ok(())
}

#[test]
fn sum_v2_constant_null_is_null() -> VortexResult<()> {
    let array =
        ConstantArray::new(Scalar::null(DType::Primitive(PType::U32, Nullable)), 10).into_array();
    let result = run_sum_v2(&array)?;
    assert!(result.is_null());
    Ok(())
}

#[test]
fn sum_v2_constant_bool_false_is_zero() -> VortexResult<()> {
    // False booleans are valid values, so the sum is zero rather than null.
    let array = ConstantArray::new(false, 10).into_array();
    let result = run_sum_v2(&array)?;
    assert_eq!(result.as_primitive().typed_value::<u64>(), Some(0));
    Ok(())
}

#[test]
fn sum_v2_constant_nan_skipping_is_zero() -> VortexResult<()> {
    let array = ConstantArray::new(f64::NAN, 4).into_array();
    let result = run_sum_v2(&array)?;
    assert_eq!(result.as_primitive().typed_value::<f64>(), Some(0.0));
    Ok(())
}

// Booleans

#[test]
fn sum_v2_bool_with_nulls() -> VortexResult<()> {
    let arr = BoolArray::from_iter([Some(true), None, Some(true), Some(false)]).into_array();
    let result = run_sum_v2(&arr)?;
    assert_eq!(result.as_primitive().typed_value::<u64>(), Some(2));
    Ok(())
}

#[test]
fn sum_v2_bool_all_null_is_null() -> VortexResult<()> {
    let arr = BoolArray::from_iter([None::<bool>, None, None]).into_array();
    let result = run_sum_v2(&arr)?;
    assert!(result.is_null());
    assert_eq!(result.dtype(), &DType::Primitive(PType::U64, Nullable));
    Ok(())
}

// Decimals

#[test]
fn sum_v2_decimal_with_nulls() -> VortexResult<()> {
    let decimal_dtype = DecimalDType::new(10, 2);
    let validity = Validity::from_iter([true, false, true]);
    let arr = DecimalArray::new(buffer![100i32, 0, 200], decimal_dtype, validity).into_array();
    let result = run_sum_v2(&arr)?;
    assert_eq!(
        result.as_decimal().decimal_value(),
        Some(DecimalValue::I256(i256::from_i128(300)))
    );
    Ok(())
}

#[test]
fn sum_v2_decimal_all_null_is_null() -> VortexResult<()> {
    let decimal_dtype = DecimalDType::new(10, 2);
    let validity = Validity::from_iter([false, false]);
    let arr = DecimalArray::new(buffer![0i32, 0], decimal_dtype, validity).into_array();
    let result = run_sum_v2(&arr)?;
    assert!(result.is_null());
    assert_eq!(
        result.dtype(),
        &DType::Decimal(DecimalDType::new(20, 2), Nullable)
    );
    Ok(())
}

// Partial merges (vtable-level)

#[test]
fn sum_v2_state_merge() -> VortexResult<()> {
    let dtype = DType::Primitive(PType::I32, Nullable);
    let options = NumericalAggregateOpts::default();
    let partial_dtype = SumV2.partial_dtype(&options, &dtype).unwrap();
    let partial_scalar = |sum: i64, is_empty: bool| {
        Scalar::struct_(
            partial_dtype.clone(),
            vec![
                Scalar::primitive(sum, Nullability::NonNullable),
                Scalar::bool(is_empty, Nullability::NonNullable),
            ],
        )
    };
    let mut state = SumV2.empty_partial(&options, &dtype)?;

    // Merging an empty partial keeps the state empty, whatever its sum field holds.
    SumV2.combine_partials(&mut state, partial_scalar(42, true))?;
    assert!(SumV2.finalize_scalar(&state)?.is_null());

    // Merging non-empty partials makes the state non-empty and sums their values.
    SumV2.combine_partials(&mut state, partial_scalar(100, false))?;
    SumV2.combine_partials(&mut state, partial_scalar(-100, false))?;
    let result = SumV2.finalize_scalar(&state)?;
    assert_eq!(result.as_primitive().typed_value::<i64>(), Some(0));

    // A null partial marks an overflow and is absorbing.
    SumV2.combine_partials(&mut state, Scalar::null(partial_dtype.clone()))?;
    SumV2.combine_partials(&mut state, partial_scalar(1, false))?;
    assert!(SumV2.finalize_scalar(&state)?.is_null());
    Ok(())
}

// Grouped aggregation

fn run_grouped_sum_v2(groups: &ArrayRef, elem_dtype: &DType) -> VortexResult<ArrayRef> {
    let mut acc =
        GroupedAccumulator::try_new(SumV2, NumericalAggregateOpts::default(), elem_dtype.clone())?;
    acc.accumulate_list(groups, &mut array_session().create_execution_ctx())?;
    acc.finish()
}

#[test]
fn grouped_sum_v2_fixed_size_list() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let elements =
        PrimitiveArray::new(buffer![1i32, 2, 3, 4, 5, 6], Validity::NonNullable).into_array();
    let groups = FixedSizeListArray::try_new(elements, 3, Validity::NonNullable, 2)?;

    let elem_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let result = run_grouped_sum_v2(&groups.into_array(), &elem_dtype)?;

    let expected = PrimitiveArray::from_option_iter([Some(6i64), Some(15i64)]).into_array();
    assert_arrays_eq!(&result, &expected, &mut ctx);
    Ok(())
}

#[test]
fn grouped_sum_v2_all_null_group_is_null() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let elements =
        PrimitiveArray::from_option_iter([None::<i32>, None, Some(3), Some(4)]).into_array();
    let groups = FixedSizeListArray::try_new(elements, 2, Validity::NonNullable, 2)?;

    let elem_dtype = DType::Primitive(PType::I32, Nullable);
    let result = run_grouped_sum_v2(&groups.into_array(), &elem_dtype)?;

    // Unlike `Sum`, the all-null group is null rather than zero.
    let expected = PrimitiveArray::from_option_iter([None, Some(7i64)]).into_array();
    assert_arrays_eq!(&result, &expected, &mut ctx);
    Ok(())
}

#[test]
fn grouped_sum_v2_null_and_empty_groups() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    // Group 0 has values, group 1 is invalid, group 2 is empty, group 3 sums to zero.
    let elements =
        PrimitiveArray::new(buffer![1i32, 2, 10, 20, 5, -5], Validity::NonNullable).into_array();
    let offsets = PrimitiveArray::new(buffer![0i32, 2, 4, 4], Validity::NonNullable).into_array();
    let sizes = PrimitiveArray::new(buffer![2i32, 2, 0, 2], Validity::NonNullable).into_array();
    let validity = Validity::from_iter([true, false, true, true]);
    let groups = ListViewArray::try_new(elements, offsets, sizes, validity)?.into_array();

    let elem_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let result = run_grouped_sum_v2(&groups, &elem_dtype)?;

    let expected =
        PrimitiveArray::from_option_iter([Some(3i64), None, None, Some(0i64)]).into_array();
    assert_arrays_eq!(&result, &expected, &mut ctx);
    Ok(())
}

#[test]
fn grouped_sum_v2_overflow_group_is_null() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let elements =
        PrimitiveArray::new(buffer![i64::MAX, 1, 2, 3], Validity::NonNullable).into_array();
    let groups = FixedSizeListArray::try_new(elements, 2, Validity::NonNullable, 2)?;

    let elem_dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
    let result = run_grouped_sum_v2(&groups.into_array(), &elem_dtype)?;

    let expected = PrimitiveArray::from_option_iter([None, Some(5i64)]).into_array();
    assert_arrays_eq!(&result, &expected, &mut ctx);
    Ok(())
}

// Return dtypes

#[test]
fn sum_v2_return_dtype_matches_sum() -> VortexResult<()> {
    use crate::aggregate_fn::fns::sum::Sum;

    let options = NumericalAggregateOpts::default();
    for dtype in [
        DType::Bool(Nullability::NonNullable),
        DType::Primitive(PType::U16, Nullable),
        DType::Primitive(PType::I32, Nullability::NonNullable),
        DType::Primitive(PType::F32, Nullable),
        DType::Decimal(DecimalDType::new(10, 2), Nullable),
    ] {
        assert_eq!(
            SumV2.return_dtype(&options, &dtype),
            Sum.return_dtype(&options, &dtype),
            "return dtype mismatch for {dtype}"
        );
    }
    assert_eq!(SumV2.return_dtype(&options, &DType::Utf8(Nullable)), None);
    Ok(())
}
