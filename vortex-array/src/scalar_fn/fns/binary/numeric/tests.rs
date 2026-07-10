// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_buffer::buffer;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::RecursiveCanonical;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::ConstantArray;
use crate::arrays::DecimalArray;
use crate::arrays::PrimitiveArray;
use crate::assert_arrays_eq;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::Nullability;
use crate::dtype::i256;
use crate::scalar::DecimalValue;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::operators::Operator;
use crate::validity::Validity;

fn sub_scalar(array: &ArrayRef, scalar: impl Into<Scalar>) -> VortexResult<ArrayRef> {
    array
        .binary(
            ConstantArray::new(scalar, array.len()).into_array(),
            Operator::Sub,
        )
        .and_then(|a| a.execute::<RecursiveCanonical>(&mut array_session().create_execution_ctx()))
        .map(|a| a.0.into_array())
}

#[test]
fn test_scalar_subtract_unsigned() {
    let mut ctx = array_session().create_execution_ctx();
    let values = buffer![1u16, 2, 3].into_array();
    let result = sub_scalar(&values, 1u16).unwrap();
    assert_arrays_eq!(result, PrimitiveArray::from_iter([0u16, 1, 2]), &mut ctx);
}

#[test]
fn test_scalar_subtract_signed() {
    let mut ctx = array_session().create_execution_ctx();
    let values = buffer![1i64, 2, 3].into_array();
    let result = sub_scalar(&values, -1i64).unwrap();
    assert_arrays_eq!(result, PrimitiveArray::from_iter([2i64, 3, 4]), &mut ctx);
}

#[test]
fn test_scalar_subtract_nullable() {
    let mut ctx = array_session().create_execution_ctx();
    let values = PrimitiveArray::from_option_iter([Some(1u16), Some(2), None, Some(3)]);
    let result = sub_scalar(&values.into_array(), Some(1u16)).unwrap();
    assert_arrays_eq!(
        result,
        PrimitiveArray::from_option_iter([Some(0u16), Some(1), None, Some(2)]),
        &mut ctx
    );
}

#[test]
fn test_scalar_subtract_float() {
    let mut ctx = array_session().create_execution_ctx();
    let values = buffer![1.0f64, 2.0, 3.0].into_array();
    let result = sub_scalar(&values, -1f64).unwrap();
    assert_arrays_eq!(
        result,
        PrimitiveArray::from_iter([2.0f64, 3.0, 4.0]),
        &mut ctx
    );
}

#[test]
fn test_scalar_subtract_float_underflow_is_ok() {
    let values = buffer![f32::MIN, 2.0, 3.0].into_array();
    let _results = sub_scalar(&values, 1.0f32).unwrap();
    let _results = sub_scalar(&values, f32::MAX).unwrap();
}

#[test]
fn test_float_divide_by_zero_is_ok() {
    let mut ctx = array_session().create_execution_ctx();
    let values = buffer![1.0f64, -1.0].into_array();
    let result = values
        .binary(
            ConstantArray::new(0.0f64, values.len()).into_array(),
            Operator::Div,
        )
        .and_then(|a| a.execute::<PrimitiveArray>(&mut array_session().create_execution_ctx()))
        .unwrap();

    assert_arrays_eq!(
        result,
        PrimitiveArray::from_iter([f64::INFINITY, f64::NEG_INFINITY]),
        &mut ctx
    );
}

#[test]
fn test_integer_overflow_errors() {
    let values = buffer![u8::MAX].into_array();
    let result = values
        .binary(
            ConstantArray::new(1u8, values.len()).into_array(),
            Operator::Add,
        )
        .and_then(|a| a.execute::<PrimitiveArray>(&mut array_session().create_execution_ctx()));

    assert!(result.is_err());
}

#[test]
fn test_integer_divide_by_zero_errors() {
    let values = buffer![1i32].into_array();
    let result = values
        .binary(
            ConstantArray::new(0i32, values.len()).into_array(),
            Operator::Div,
        )
        .and_then(|a| a.execute::<PrimitiveArray>(&mut array_session().create_execution_ctx()));

    assert!(result.is_err());
}

#[test]
fn test_integer_divide_overflow_errors() {
    let values = buffer![i64::MIN].into_array();
    let result = values
        .binary(
            ConstantArray::new(-1i64, values.len()).into_array(),
            Operator::Div,
        )
        .and_then(|a| a.execute::<PrimitiveArray>(&mut array_session().create_execution_ctx()));

    assert!(result.is_err());
}

#[test]
fn test_integer_divide_errors_ignore_null_lanes() {
    let mut ctx = array_session().create_execution_ctx();
    let lhs =
        PrimitiveArray::new(buffer![10i32, 10], Validity::from_iter([false, true])).into_array();
    let rhs = buffer![0i32, 2].into_array();
    let result = lhs
        .binary(rhs, Operator::Div)
        .and_then(|a| a.execute::<RecursiveCanonical>(&mut array_session().create_execution_ctx()))
        .map(|a| a.0.into_array())
        .unwrap();

    assert_arrays_eq!(
        result,
        PrimitiveArray::from_option_iter([None, Some(5i32)]),
        &mut ctx
    );
}

#[test]
fn test_integer_errors_ignore_null_lanes() {
    let mut ctx = array_session().create_execution_ctx();
    let values =
        PrimitiveArray::new(buffer![u8::MAX, 1], Validity::from_iter([false, true])).into_array();
    let result = values
        .binary(
            ConstantArray::new(1u8, values.len()).into_array(),
            Operator::Add,
        )
        .and_then(|a| a.execute::<RecursiveCanonical>(&mut array_session().create_execution_ctx()))
        .map(|a| a.0.into_array())
        .unwrap();

    assert_arrays_eq!(
        result,
        PrimitiveArray::from_option_iter([None, Some(2u8)]),
        &mut ctx
    );
}

#[test]
fn test_integer_array_array_errors_on_valid_lanes() {
    let lhs = PrimitiveArray::new(
        buffer![u8::MAX, 1, u8::MAX],
        Validity::from_iter([false, true, true]),
    )
    .into_array();
    let rhs = buffer![1u8, 1, 1].into_array();
    let result = lhs
        .binary(rhs, Operator::Add)
        .and_then(|a| a.execute::<PrimitiveArray>(&mut array_session().create_execution_ctx()));

    assert!(result.is_err());
}

#[test]
fn test_present_nullable_constant_preserves_nullable_output() {
    let mut ctx = array_session().create_execution_ctx();
    let values = buffer![1u8, 2].into_array();
    let result = values
        .binary(
            ConstantArray::new(Some(1u8), values.len()).into_array(),
            Operator::Add,
        )
        .and_then(|a| a.execute::<PrimitiveArray>(&mut array_session().create_execution_ctx()))
        .unwrap();

    assert_arrays_eq!(
        result,
        PrimitiveArray::from_option_iter([Some(2u8), Some(3)]),
        &mut ctx
    );
}

// -- Decimal arithmetic --

fn decimal_binary(lhs: ArrayRef, rhs: ArrayRef, op: Operator) -> VortexResult<ArrayRef> {
    lhs.binary(rhs, op)
        .and_then(|a| a.execute::<RecursiveCanonical>(&mut array_session().create_execution_ctx()))
        .map(|a| a.0.into_array())
}

fn decimal_constant(value: impl Into<DecimalValue>, dtype: DecimalDType, len: usize) -> ArrayRef {
    ConstantArray::new(
        Scalar::decimal(value.into(), dtype, Nullability::NonNullable),
        len,
    )
    .into_array()
}

#[rstest]
#[case::add(Operator::Add, [150i64, 225], [1050i64, 1225])]
#[case::sub(Operator::Sub, [150i64, 225], [750i64, 775])]
#[case::mul(Operator::Mul, [50i64, 200], [450i64, 2000])] // 9.00 * 0.50, 10.00 * 2.00
#[case::div(Operator::Div, [300i64, 400], [300i64, 250])] // 9.00 / 3.00, 10.00 / 4.00
fn test_decimal_array_array(
    #[case] op: Operator,
    #[case] rhs: [i64; 2],
    #[case] expected: [i64; 2],
) {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DecimalDType::new(10, 2);
    let lhs = DecimalArray::from_iter::<i64, _>([900, 1000], dtype).into_array();
    let rhs = DecimalArray::from_iter::<i64, _>(rhs, dtype).into_array();

    let result = decimal_binary(lhs, rhs, op).unwrap();
    assert_arrays_eq!(
        result,
        DecimalArray::from_iter::<i64, _>(expected, dtype),
        &mut ctx
    );
}

#[test]
fn test_decimal_mixed_storage_widths() {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DecimalDType::new(10, 2);
    let lhs = DecimalArray::from_iter::<i32, _>([100, 250], dtype).into_array();
    let rhs = DecimalArray::from_iter::<i128, _>([200, 250], dtype).into_array();

    let result = decimal_binary(lhs, rhs, Operator::Add).unwrap();
    assert_arrays_eq!(
        result,
        DecimalArray::from_iter::<i64, _>([300, 500], dtype),
        &mut ctx
    );
}

#[test]
fn test_decimal_nullable_lanes() {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DecimalDType::new(10, 2);
    let lhs =
        DecimalArray::from_option_iter::<i64, _>([Some(100), None, Some(300)], dtype).into_array();
    let rhs = DecimalArray::from_iter::<i64, _>([50, 50, 50], dtype).into_array();

    let result = decimal_binary(lhs, rhs, Operator::Add).unwrap();
    assert_arrays_eq!(
        result,
        DecimalArray::from_option_iter::<i64, _>([Some(150), None, Some(350)], dtype),
        &mut ctx
    );
}

#[test]
fn test_decimal_overflow_on_valid_lane_errors() {
    let dtype = DecimalDType::new(3, 0);
    let lhs = DecimalArray::from_iter::<i16, _>([999], dtype).into_array();
    let rhs = DecimalArray::from_iter::<i16, _>([2], dtype).into_array();

    assert!(decimal_binary(lhs, rhs, Operator::Add).is_err());
}

#[test]
fn test_decimal_overflow_on_null_lane_ignored() {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DecimalDType::new(3, 0);
    // The null lane holds 999, so adding 500 overflows the precision there but is ignored.
    let lhs = DecimalArray::new(
        buffer![999i16, 1],
        dtype,
        Validity::from_iter([false, true]),
    )
    .into_array();
    let rhs = decimal_constant(500i16, dtype, 2);

    let result = decimal_binary(lhs, rhs, Operator::Add).unwrap();
    assert_arrays_eq!(
        result,
        DecimalArray::from_option_iter::<i16, _>([None, Some(501)], dtype),
        &mut ctx
    );
}

/// A value can fit the storage width while violating the dtype precision: 60 + 60 fits an i8 but
/// exceeds precision 2.
#[test]
fn test_decimal_precision_stricter_than_width() {
    let dtype = DecimalDType::new(2, 0);
    let lhs = DecimalArray::from_iter::<i8, _>([60], dtype).into_array();
    let rhs = DecimalArray::from_iter::<i8, _>([60], dtype).into_array();

    assert!(decimal_binary(lhs, rhs, Operator::Add).is_err());
}

#[rstest]
#[case::rescales([50i64], [10i64], [5i64])] // 0.50 * 0.10 = 0.05
#[case::truncates([15i64], [15i64], [2i64])] // 0.15 * 0.15 = 0.0225 -> 0.02
#[case::truncates_toward_zero([-15i64], [15i64], [-2i64])]
fn test_decimal_mul_fixed_point(
    #[case] lhs: [i64; 1],
    #[case] rhs: [i64; 1],
    #[case] expected: [i64; 1],
) {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DecimalDType::new(10, 2);
    let lhs = DecimalArray::from_iter::<i64, _>(lhs, dtype).into_array();
    let rhs = DecimalArray::from_iter::<i64, _>(rhs, dtype).into_array();

    let result = decimal_binary(lhs, rhs, Operator::Mul).unwrap();
    assert_arrays_eq!(
        result,
        DecimalArray::from_iter::<i64, _>(expected, dtype),
        &mut ctx
    );
}

/// The raw product may overflow the operand storage width while the rescaled result fits:
/// 3000.00 * 3000.00 = 9,000,000.00 needs a wider intermediate than the i32 storage.
#[test]
fn test_decimal_mul_wider_than_operand_storage() {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DecimalDType::new(10, 2);
    let values = DecimalArray::from_iter::<i32, _>([300_000], dtype).into_array();

    let result = decimal_binary(values.clone(), values, Operator::Mul).unwrap();
    assert_arrays_eq!(
        result,
        DecimalArray::from_iter::<i64, _>([900_000_000], dtype),
        &mut ctx
    );
}

#[rstest]
#[case::rescales([1000i64], [10i64], [10000i64])] // 10.00 / 0.10 = 100.00
#[case::truncates([1000i64], [300i64], [333i64])] // 10.00 / 3.00 = 3.33...
#[case::truncates_toward_zero([-1000i64], [300i64], [-333i64])]
fn test_decimal_div_fixed_point(
    #[case] lhs: [i64; 1],
    #[case] rhs: [i64; 1],
    #[case] expected: [i64; 1],
) {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DecimalDType::new(10, 2);
    let lhs = DecimalArray::from_iter::<i64, _>(lhs, dtype).into_array();
    let rhs = DecimalArray::from_iter::<i64, _>(rhs, dtype).into_array();

    let result = decimal_binary(lhs, rhs, Operator::Div).unwrap();
    assert_arrays_eq!(
        result,
        DecimalArray::from_iter::<i64, _>(expected, dtype),
        &mut ctx
    );
}

#[test]
fn test_decimal_divide_by_zero_errors() {
    let dtype = DecimalDType::new(10, 2);
    let lhs = DecimalArray::from_iter::<i64, _>([100], dtype).into_array();
    let rhs = decimal_constant(0i64, dtype, 1);

    assert!(decimal_binary(lhs, rhs, Operator::Div).is_err());
}

#[test]
fn test_decimal_divide_by_zero_on_null_lane_ignored() {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DecimalDType::new(10, 2);
    let lhs = DecimalArray::from_option_iter::<i64, _>([None, Some(1000)], dtype).into_array();
    let rhs = DecimalArray::from_iter::<i64, _>([0, 500], dtype).into_array();

    let result = decimal_binary(lhs, rhs, Operator::Div).unwrap();
    assert_arrays_eq!(
        result,
        DecimalArray::from_option_iter::<i64, _>([None, Some(200)], dtype),
        &mut ctx
    );
}

#[test]
fn test_decimal_constant_lhs_non_commutative() {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DecimalDType::new(10, 2);
    let lhs = decimal_constant(1000i64, dtype, 2);
    let rhs = DecimalArray::from_iter::<i64, _>([250, 400], dtype).into_array();

    let result = decimal_binary(lhs, rhs, Operator::Sub).unwrap();
    assert_arrays_eq!(
        result,
        DecimalArray::from_iter::<i64, _>([750, 600], dtype),
        &mut ctx
    );
}

#[test]
fn test_decimal_nullable_constant_preserves_nullable_output() {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DecimalDType::new(10, 2);
    let values = DecimalArray::from_iter::<i64, _>([100, 200], dtype).into_array();
    let constant = ConstantArray::new(
        Scalar::decimal(DecimalValue::from(50i64), dtype, Nullability::Nullable),
        2,
    )
    .into_array();

    let result = decimal_binary(values, constant, Operator::Add).unwrap();
    assert_arrays_eq!(
        result,
        DecimalArray::from_option_iter::<i64, _>([Some(150), Some(250)], dtype),
        &mut ctx
    );
}

#[test]
fn test_decimal_null_constant_yields_all_null() {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DecimalDType::new(10, 2);
    let values = DecimalArray::from_iter::<i64, _>([100, 200], dtype).into_array();
    let null_constant = ConstantArray::new(
        Scalar::null(DType::Decimal(dtype, Nullability::Nullable)),
        2,
    )
    .into_array();

    let result = decimal_binary(values, null_constant, Operator::Add).unwrap();
    assert_arrays_eq!(
        result,
        DecimalArray::from_option_iter::<i64, _>([None, None], dtype),
        &mut ctx
    );
}

/// A constant stored in a wider variant than the array storage participates through the widened
/// working type.
#[test]
fn test_decimal_constant_wider_than_array_storage() {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DecimalDType::new(20, 0);
    let values = DecimalArray::from_iter::<i8, _>([1, 2], dtype).into_array();
    let constant = decimal_constant(10_000_000_000i64, dtype, 2);

    let result = decimal_binary(values, constant, Operator::Add).unwrap();
    assert_arrays_eq!(
        result,
        DecimalArray::from_iter::<i64, _>([10_000_000_001, 10_000_000_002], dtype),
        &mut ctx
    );
}

/// p=38 multiplication requires a 256-bit intermediate product.
#[test]
fn test_decimal_mul_i256_working_width() {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DecimalDType::new(38, 2);
    let big = 10_i128.pow(19);
    let values = DecimalArray::from_iter::<i128, _>([big], dtype).into_array();

    // (10^17).00 * (10^17).00 = 10^34.00, stored as 10^36.
    let result = decimal_binary(values.clone(), values, Operator::Mul).unwrap();
    assert_arrays_eq!(
        result,
        DecimalArray::from_iter::<i128, _>([10_i128.pow(36)], dtype),
        &mut ctx
    );
}

/// Near the 76-digit cap, a raw product that overflows i256 errors even if the rescaled result
/// would fit. This is a documented limitation of the 256-bit intermediate.
#[test]
fn test_decimal_mul_i256_intermediate_overflow_errors() {
    let dtype = DecimalDType::new(76, 76);
    let big = i256::from_i128(10).checked_pow(75).unwrap();
    let values = DecimalArray::from_iter::<i256, _>([big], dtype).into_array();

    assert!(decimal_binary(values.clone(), values, Operator::Mul).is_err());
}

#[rstest]
#[case::mul(Operator::Mul, [5i64], [3i64], [1500i64])] // 500 * 300 = 150,000, stored 1500
#[case::div(Operator::Div, [600i64], [3i64], [2i64])] // 60,000 / 300 = 200, stored 2
#[case::div_truncates(Operator::Div, [5000i64], [3i64], [16i64])] // 500,000 / 300 = 1666.67 -> 1600
fn test_decimal_negative_scale(
    #[case] op: Operator,
    #[case] lhs: [i64; 1],
    #[case] rhs: [i64; 1],
    #[case] expected: [i64; 1],
) {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DecimalDType::new(10, -2);
    let lhs = DecimalArray::from_iter::<i64, _>(lhs, dtype).into_array();
    let rhs = DecimalArray::from_iter::<i64, _>(rhs, dtype).into_array();

    let result = decimal_binary(lhs, rhs, op).unwrap();
    assert_arrays_eq!(
        result,
        DecimalArray::from_iter::<i64, _>(expected, dtype),
        &mut ctx
    );
}

#[test]
fn test_decimal_empty() {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DecimalDType::new(10, 2);
    let empty = DecimalArray::from_iter::<i64, _>([], dtype).into_array();

    let result = decimal_binary(empty.clone(), empty, Operator::Add).unwrap();
    assert_arrays_eq!(
        result,
        DecimalArray::from_iter::<i64, _>([], dtype),
        &mut ctx
    );
}

#[test]
fn test_decimal_constant_constant_folds() {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DecimalDType::new(10, 2);
    let lhs = decimal_constant(150i64, dtype, 3);
    let rhs = decimal_constant(50i64, dtype, 3);

    let result = decimal_binary(lhs, rhs, Operator::Mul).unwrap();
    assert_arrays_eq!(
        result,
        DecimalArray::from_iter::<i64, _>([75, 75, 75], dtype),
        &mut ctx
    );
}
