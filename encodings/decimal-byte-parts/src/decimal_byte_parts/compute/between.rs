// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_array::scalar_fn::fns::between::BetweenKernel;
use vortex_array::scalar_fn::fns::between::BetweenOptions;
use vortex_array::scalar_fn::fns::between::StrictComparison;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::DecimalByteParts;
use crate::decimal_byte_parts::DecimalBytePartsArrayExt;
use crate::decimal_byte_parts::compute::compare::Sign;
use crate::decimal_byte_parts::compute::compare::decimal_value_to_i128;
use crate::decimal_byte_parts::compute::compare::decimal_value_wrapper_to_primitive;
use crate::decimal_byte_parts::compute::two_limb::between_bits;
use crate::decimal_byte_parts::compute::two_limb::eval;

impl BetweenKernel for DecimalByteParts {
    fn between(
        arr: ArrayView<'_, Self>,
        lower: &ArrayRef,
        upper: &ArrayRef,
        options: &BetweenOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // We can only push the comparison down into the limbs when both bounds are constant.
        let (Some(lower_const), Some(upper_const)) = (lower.as_constant(), upper.as_constant())
        else {
            return Ok(None);
        };

        // NOTE: the `between` entrypoint precondition already replaced null bounds with an
        // all-null result, so both bounds are guaranteed to be non-null here.
        let lower_decimal = lower_const
            .as_decimal()
            .decimal_value()
            .vortex_expect("checked for null in entry func");
        let upper_decimal = upper_const
            .as_decimal()
            .decimal_value()
            .vortex_expect("checked for null in entry func");

        let nullability =
            arr.dtype().nullability() | lower.dtype().nullability() | upper.dtype().nullability();

        if arr.lower().is_some() {
            // Two-limb representation: a lexicographic comparison over the (signed high, unsigned
            // low) limbs. A bound that does not fit in i128 lies outside every two-limb value's
            // range: clamp it to the i128 domain boundary non-strictly (always satisfied), or
            // resolve the whole `between` to all-false when it excludes the entire domain.
            let lower_bound = match decimal_value_to_i128(lower_decimal) {
                Ok(v) => Some((v, options.lower_strict)),
                Err(Sign::Negative) => Some((i128::MIN, StrictComparison::NonStrict)),
                Err(Sign::Positive) => None,
            };
            let upper_bound = match decimal_value_to_i128(upper_decimal) {
                Ok(v) => Some((v, options.upper_strict)),
                Err(Sign::Positive) => Some((i128::MAX, StrictComparison::NonStrict)),
                Err(Sign::Negative) => None,
            };
            let (Some((lower_i128, lower_strict)), Some((upper_i128, upper_strict))) =
                (lower_bound, upper_bound)
            else {
                return never_satisfied(&arr, lower, upper, nullability, ctx);
            };
            let options = BetweenOptions {
                lower_strict,
                upper_strict,
            };
            return Ok(Some(eval(&arr, nullability, ctx, |high, low| {
                between_bits(high, low, lower_i128, upper_i128, &options)
            })?));
        }

        let scalar_type = arr.msp().dtype().with_nullability(nullability);
        let msp_ptype = arr.msp().dtype().as_ptype();

        // A bound outside the MSP's physical integer range lies outside every stored value:
        // clamp it to the type's boundary non-strictly (always satisfied), or resolve the whole
        // `between` to all-false when it excludes the entire domain.
        let lower_bound = match decimal_value_wrapper_to_primitive(lower_decimal, msp_ptype) {
            Ok(v) => Some((v, options.lower_strict)),
            Err(Sign::Negative) => Some((min_scalar_value(msp_ptype), StrictComparison::NonStrict)),
            Err(Sign::Positive) => None,
        };
        let upper_bound = match decimal_value_wrapper_to_primitive(upper_decimal, msp_ptype) {
            Ok(v) => Some((v, options.upper_strict)),
            Err(Sign::Positive) => Some((max_scalar_value(msp_ptype), StrictComparison::NonStrict)),
            Err(Sign::Negative) => None,
        };
        let (Some((lower_value, lower_strict)), Some((upper_value, upper_strict))) =
            (lower_bound, upper_bound)
        else {
            return never_satisfied(&arr, lower, upper, nullability, ctx);
        };

        let lower_const = ConstantArray::new(
            Scalar::try_new(scalar_type.clone(), Some(lower_value))?,
            arr.len(),
        );
        let upper_const =
            ConstantArray::new(Scalar::try_new(scalar_type, Some(upper_value))?, arr.len());

        arr.msp()
            .clone()
            .between(
                lower_const.into_array(),
                upper_const.into_array(),
                BetweenOptions {
                    lower_strict,
                    upper_strict,
                },
            )
            .map(Some)
    }
}

/// The `between` result when a bound excludes the array's entire value domain: constant false,
/// provided every input is non-null. Otherwise returns `None` to fall back to the canonicalized
/// implementation, which does per-row null-checking instead.
fn never_satisfied(
    arr: &ArrayView<'_, DecimalByteParts>,
    lower: &ArrayRef,
    upper: &ArrayRef,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    if arr.array().all_valid(ctx)? && lower.all_valid(ctx)? && upper.all_valid(ctx)? {
        Ok(Some(
            ConstantArray::new(Scalar::bool(false, nullability), arr.len()).into_array(),
        ))
    } else {
        Ok(None)
    }
}

fn min_scalar_value(ptype: PType) -> ScalarValue {
    match_each_integer_ptype!(ptype, |P| { ScalarValue::from(P::MIN) })
}

fn max_scalar_value(ptype: PType) -> ScalarValue {
    match_each_integer_ptype!(ptype, |P| { ScalarValue::from(P::MAX) })
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::DecimalArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::i256;
    use vortex_array::scalar::DecimalValue;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::fns::between::BetweenKernel;
    use vortex_array::scalar_fn::fns::between::BetweenOptions;
    use vortex_array::scalar_fn::fns::between::StrictComparison;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::super::two_limb::two_limb_array;
    use crate::DecimalByteParts;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    fn decimal_const(value: DecimalValue, decimal_type: DecimalDType, len: usize) -> ArrayRef {
        ConstantArray::new(
            Scalar::decimal(value, decimal_type, Nullability::NonNullable),
            len,
        )
        .into_array()
    }

    fn two_limb(values: &[i128], decimal_type: DecimalDType) -> ArrayRef {
        two_limb_array(values, Validity::NonNullable, decimal_type).into_array()
    }

    /// The two-limb `between` pushdown must agree with the canonical i128 implementation across
    /// values spanning the low-limb wraparound, the high limb, and negatives, for every strictness.
    #[rstest]
    #[case(StrictComparison::NonStrict, StrictComparison::NonStrict)]
    #[case(StrictComparison::Strict, StrictComparison::NonStrict)]
    #[case(StrictComparison::NonStrict, StrictComparison::Strict)]
    #[case(StrictComparison::Strict, StrictComparison::Strict)]
    fn two_limb_between_matches_canonical(
        #[case] lower_strict: StrictComparison,
        #[case] upper_strict: StrictComparison,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let decimal_type = DecimalDType::new(38, 0);
        let values: Vec<i128> = vec![
            0,
            1,
            -1,
            i128::from(i64::MAX),
            i128::from(i64::MAX) + 1,
            (5i128 << 64) | 3,
            (5i128 << 64) | 5,
            (5i128 << 64) | 9,
            (4i128 << 64) | i128::from(u64::MAX),
            (6i128 << 64),
            -(7i128 << 64) | 11,
        ];
        let lower = (5i128 << 64) | 3;
        let upper = (5i128 << 64) | 9;
        let len = values.len();
        let options = BetweenOptions {
            lower_strict,
            upper_strict,
        };

        let lower_arr = decimal_const(DecimalValue::I128(lower), decimal_type, len);
        let upper_arr = decimal_const(DecimalValue::I128(upper), decimal_type, len);

        let got = two_limb(&values, decimal_type)
            .between(lower_arr.clone(), upper_arr.clone(), options.clone())?
            .execute::<BoolArray>(&mut ctx)?;

        let canonical = DecimalArray::new(
            values.iter().copied().collect::<Buffer<i128>>(),
            decimal_type,
            Validity::NonNullable,
        )
        .into_array();
        let want = canonical
            .between(lower_arr, upper_arr, options)?
            .execute::<BoolArray>(&mut ctx)?;

        assert_arrays_eq!(got, want, &mut ctx);
        Ok(())
    }

    /// A two-limb array must canonicalize to the same values as a canonical i128 `DecimalArray`.
    #[test]
    fn two_limb_canonicalizes_to_i128() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let decimal_type = DecimalDType::new(38, 0);
        let values: Vec<i128> = vec![
            0,
            -1,
            i128::from(i64::MIN),
            (3i128 << 64) | 42,
            -(9i128 << 64) | 17,
        ];

        let got = two_limb(&values, decimal_type).execute::<DecimalArray>(&mut ctx)?;
        let want = DecimalArray::new(
            values.iter().copied().collect::<Buffer<i128>>(),
            decimal_type,
            Validity::NonNullable,
        );
        assert_arrays_eq!(got.into_array(), want.into_array(), &mut ctx);
        Ok(())
    }

    #[test]
    fn between_decimal_const() -> VortexResult<()> {
        let decimal_type = DecimalDType::new(8, 2);
        let arr = DecimalByteParts::try_new(
            PrimitiveArray::new(buffer![100i32, 200, 300, 400, 500], Validity::AllValid)
                .into_array(),
            decimal_type,
        )?
        .into_array();

        let lower = decimal_const(DecimalValue::I64(200), decimal_type, arr.len());
        let upper = decimal_const(DecimalValue::I64(400), decimal_type, arr.len());

        // 200 <= value <= 400
        let res = arr.clone().between(
            lower.clone(),
            upper.clone(),
            BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::NonStrict,
            },
        )?;
        assert_arrays_eq!(
            res,
            BoolArray::from_iter([Some(false), Some(true), Some(true), Some(true), Some(false)]),
            &mut SESSION.create_execution_ctx()
        );

        // 200 < value < 400
        let res = arr.between(
            lower,
            upper,
            BetweenOptions {
                lower_strict: StrictComparison::Strict,
                upper_strict: StrictComparison::Strict,
            },
        )?;
        assert_arrays_eq!(
            res,
            BoolArray::from_iter([
                Some(false),
                Some(false),
                Some(true),
                Some(false),
                Some(false)
            ]),
            &mut SESSION.create_execution_ctx()
        );

        Ok(())
    }

    #[test]
    fn between_decimal_nullable() -> VortexResult<()> {
        let decimal_type = DecimalDType::new(8, 2);
        let arr = DecimalByteParts::try_new(
            PrimitiveArray::new(
                buffer![100i32, 200, 300, 400],
                Validity::Array(BoolArray::from_iter([false, true, true, true]).into_array()),
            )
            .into_array(),
            decimal_type,
        )?
        .into_array();

        let lower = decimal_const(DecimalValue::I64(100), decimal_type, arr.len());
        let upper = decimal_const(DecimalValue::I64(300), decimal_type, arr.len());

        let res = arr.between(
            lower,
            upper,
            BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::NonStrict,
            },
        )?;
        assert_arrays_eq!(
            res,
            BoolArray::from_iter([None, Some(true), Some(true), Some(false)]),
            &mut SESSION.create_execution_ctx()
        );

        Ok(())
    }

    /// Bounds outside the representable domain must resolve inside the kernel (clamped to an
    /// always-satisfied constraint, or constant false) rather than declining the pushdown.
    #[test]
    fn between_out_of_range_bounds_stay_pushed_down() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let nonstrict = BetweenOptions {
            lower_strict: StrictComparison::NonStrict,
            upper_strict: StrictComparison::NonStrict,
        };

        // Two-limb array with bounds beyond the i128 range (precision 39 holds +/- 2^128).
        let dt = DecimalDType::new(38, 0);
        let arr = two_limb_array(
            &[0, 1i128 << 64, (1i128 << 64) | 5, -(1i128 << 64)],
            Validity::NonNullable,
            dt,
        );
        let wide = DecimalDType::new(39, 0);
        let below = decimal_const(DecimalValue::I256(i256::from_parts(0, -1)), wide, arr.len());
        let above = decimal_const(DecimalValue::I256(i256::from_parts(0, 1)), wide, arr.len());
        let mid = decimal_const(DecimalValue::I128(1i128 << 64), dt, arr.len());

        // -2^128 <= v <= 2^64 reduces to v <= 2^64.
        let res = BetweenKernel::between(arr.as_view(), &below, &mid, &nonstrict, &mut ctx)?
            .vortex_expect("kernel must clamp an out-of-range lower bound");
        assert_arrays_eq!(
            res,
            BoolArray::from_iter([true, true, false, true]),
            &mut ctx
        );

        // 2^128 <= v <= 2^128 excludes every i128: constant false.
        let res = BetweenKernel::between(arr.as_view(), &above, &above, &nonstrict, &mut ctx)?
            .vortex_expect("kernel must resolve a never-satisfied between");
        assert_arrays_eq!(
            res,
            BoolArray::from_iter([false, false, false, false]),
            &mut ctx
        );

        // Single-limb i32 msp with bounds beyond the i32 range.
        let dt = DecimalDType::new(38, 2);
        let arr = DecimalByteParts::try_new(buffer![100i32, 200, 300].into_array(), dt)?;
        let below = decimal_const(DecimalValue::I64(i64::from(i32::MIN) - 1), dt, arr.len());
        let mid = decimal_const(DecimalValue::I64(200), dt, arr.len());

        // below-i32-range <= v <= 200 reduces to v <= 200.
        let res = BetweenKernel::between(arr.as_view(), &below, &mid, &nonstrict, &mut ctx)?
            .vortex_expect("kernel must clamp a lower bound below the msp range");
        assert_arrays_eq!(res, BoolArray::from_iter([true, true, false]), &mut ctx);

        // v <= below-i32-range excludes every stored value: constant false.
        let res = BetweenKernel::between(arr.as_view(), &below, &below, &nonstrict, &mut ctx)?
            .vortex_expect("kernel must resolve a never-satisfied between");
        assert_arrays_eq!(res, BoolArray::from_iter([false, false, false]), &mut ctx);

        Ok(())
    }

    /// End-to-end: an upper bound that only fits in i128 over i32 storage is always satisfied,
    /// so the result reduces to the lower constraint alone.
    #[test]
    fn between_decimal_unconvertible_bound() -> VortexResult<()> {
        let decimal_type = DecimalDType::new(38, 2);
        let arr = DecimalByteParts::try_new(
            PrimitiveArray::new(buffer![100i32, 200, 300], Validity::AllValid).into_array(),
            decimal_type,
        )?
        .into_array();

        let lower = decimal_const(DecimalValue::I64(150), decimal_type, arr.len());
        let upper = decimal_const(
            DecimalValue::I128(9_999_999_999_999_999_999),
            decimal_type,
            arr.len(),
        );

        let res = arr.between(
            lower,
            upper,
            BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::NonStrict,
            },
        )?;
        assert_arrays_eq!(
            res,
            BoolArray::from_iter([Some(false), Some(true), Some(true)]),
            &mut SESSION.create_execution_ctx()
        );

        Ok(())
    }
}
