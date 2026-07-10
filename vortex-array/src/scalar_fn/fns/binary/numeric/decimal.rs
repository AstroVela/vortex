// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Native execution of the arithmetic operators over decimal arrays.
//!
//! Both operands share a logical [`DecimalDType`] (equal precision and scale) and the result
//! keeps that dtype: fixed-point arithmetic at the shared scale, truncating toward zero. Add and
//! Sub apply directly to the unscaled stored integers; Mul rescales the raw product from twice
//! the scale back to the scale; Div pre-scales the dividend (or, for negative scales, the
//! divisor).
//!
//! Lanes execute in a working width chosen so that in-precision inputs cannot spuriously
//! overflow an intermediate value (capped at 256 bits, where an overflowing intermediate is
//! reported as an error even if the rescaled result would fit). An operation that overflows the
//! result precision or divides by zero on a valid lane is an error; invalid lanes never error.

use num_traits::CheckedAdd;
use num_traits::CheckedDiv;
use num_traits::CheckedMul;
use num_traits::CheckedSub;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use super::CheckedValues;
use super::check_numeric_errors;
use super::checked_all_lanes;
use super::checked_valid_lanes;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::DecimalArray;
use crate::arrays::decimal::DecimalArrayExt;
use crate::arrays::decimal::widened_buffer;
use crate::dtype::BigCast;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::DecimalType;
use crate::dtype::NativeDecimalType;
use crate::dtype::i256;
use crate::match_each_decimal_value_type;
use crate::scalar::DecimalValue;
use crate::scalar::NumericOperator;
use crate::scalar::Scalar;
use crate::validity::Validity;

/// Execute a numeric operation between two decimal arrays sharing a decimal dtype.
pub(super) fn execute_numeric_decimal(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: NumericOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let DType::Decimal(decimal_dtype, _) = lhs.dtype() else {
        vortex_bail!("expected a decimal dtype, got {}", lhs.dtype());
    };
    let decimal_dtype = *decimal_dtype;
    let result_dtype = lhs
        .dtype()
        .with_nullability(lhs.dtype().nullability() | rhs.dtype().nullability());

    let lhs = DecimalOperand::try_new(lhs, ctx)?;
    let rhs = DecimalOperand::try_new(rhs, ctx)?;
    let len = lhs.len();
    if len != rhs.len() {
        vortex_bail!(
            "numeric operator requires equal lengths, got {} and {}",
            len,
            rhs.len()
        );
    }

    let validity = lhs.validity().and(rhs.validity())?;
    let valid_rows = validity.execute_mask(len, ctx)?;

    let work = working_type(&lhs, &rhs, decimal_dtype, op);
    match_each_decimal_value_type!(work, |W| {
        match op {
            NumericOperator::Add => execute_decimal_typed::<W, DecimalAdd>(
                &lhs,
                &rhs,
                decimal_dtype,
                &result_dtype,
                validity,
                &valid_rows,
            ),
            NumericOperator::Sub => execute_decimal_typed::<W, DecimalSub>(
                &lhs,
                &rhs,
                decimal_dtype,
                &result_dtype,
                validity,
                &valid_rows,
            ),
            NumericOperator::Mul => execute_decimal_typed::<W, DecimalMul>(
                &lhs,
                &rhs,
                decimal_dtype,
                &result_dtype,
                validity,
                &valid_rows,
            ),
            NumericOperator::Div => execute_decimal_typed::<W, DecimalDiv>(
                &lhs,
                &rhs,
                decimal_dtype,
                &result_dtype,
                validity,
                &valid_rows,
            ),
        }
    })
}

/// A decimal binary-operator operand: a canonical decimal array, a non-null constant, or an
/// all-null constant.
enum DecimalOperand {
    Array {
        values: DecimalArray,
        validity: Validity,
    },
    Constant {
        value: DecimalValue,
        len: usize,
        validity: Validity,
    },
    Null(usize),
}

impl DecimalOperand {
    fn try_new(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        if let Some(constant) = array.as_opt::<Constant>() {
            return Ok(match constant.scalar().as_decimal().decimal_value() {
                Some(value) => Self::Constant {
                    value,
                    len: array.len(),
                    validity: if constant.scalar().dtype().is_nullable() {
                        Validity::AllValid
                    } else {
                        Validity::NonNullable
                    },
                },
                None => Self::Null(array.len()),
            });
        }

        let values = array.clone().execute::<DecimalArray>(ctx)?;
        let validity = values.validity()?;
        Ok(Self::Array { values, validity })
    }

    fn len(&self) -> usize {
        match self {
            Self::Array { values, .. } => values.len(),
            Self::Constant { len, .. } | Self::Null(len) => *len,
        }
    }

    fn validity(&self) -> Validity {
        match self {
            Self::Array { validity, .. } | Self::Constant { validity, .. } => validity.clone(),
            Self::Null(_) => Validity::AllInvalid,
        }
    }
}

/// Choose the lane width for executing `op`: wide enough for the operand storage and for every
/// intermediate value that in-precision inputs can produce, capped at 256 bits.
fn working_type(
    lhs: &DecimalOperand,
    rhs: &DecimalOperand,
    dtype: DecimalDType,
    op: NumericOperator,
) -> DecimalType {
    let precision = dtype.precision() as u16;
    let scale_digits = dtype.scale().unsigned_abs() as u16;
    let needed_digits = match op {
        // The sum of two p-digit values has at most p + 1 digits.
        NumericOperator::Add | NumericOperator::Sub => precision + 1,
        // The raw product has at most 2p digits; a negative scale then multiplies it by 10^|s|.
        NumericOperator::Mul => 2 * precision + if dtype.scale() < 0 { scale_digits } else { 0 },
        // The pre-scaled dividend (or divisor) has at most p + |s| digits.
        NumericOperator::Div => precision + scale_digits,
    };
    let needed_digits = needed_digits.min(<i256 as NativeDecimalType>::MAX_PRECISION as u16);
    let needed =
        DecimalType::smallest_decimal_value_type(&DecimalDType::new(needed_digits as u8, 0));

    let operand_type = |operand: &DecimalOperand| match operand {
        DecimalOperand::Array { values, .. } => values.values_type(),
        DecimalOperand::Constant { value, .. } => smallest_value_type(value),
        DecimalOperand::Null(_) => DecimalType::I8,
    };

    needed.max(operand_type(lhs)).max(operand_type(rhs))
}

/// The smallest decimal value type that can represent `value`, regardless of its stored width.
fn smallest_value_type(value: &DecimalValue) -> DecimalType {
    if value.cast::<i8>().is_some() {
        DecimalType::I8
    } else if value.cast::<i16>().is_some() {
        DecimalType::I16
    } else if value.cast::<i32>().is_some() {
        DecimalType::I32
    } else if value.cast::<i64>().is_some() {
        DecimalType::I64
    } else if value.cast::<i128>().is_some() {
        DecimalType::I128
    } else {
        DecimalType::I256
    }
}

/// Per-execution constants for checked decimal lane operations at working width `W`.
struct DecimalOpPlan<W> {
    /// Inclusive stored-value bounds implied by the result precision.
    prec_min: W,
    prec_max: W,
    /// `10^|scale|` in `W`, or `None` if unrepresentable (only for scales below -76, where any
    /// nonzero rescale overflows anyway).
    scale_factor: Option<W>,
    /// The shared operand/result scale.
    scale: i8,
}

impl<W: NativeDecimalType> DecimalOpPlan<W> {
    fn new(dtype: DecimalDType) -> Self {
        let precision = dtype.precision() as usize;
        let scale = dtype.scale();
        let scale_factor = i256::from_i128(10)
            .checked_pow(scale.unsigned_abs() as u32)
            .and_then(<W as BigCast>::from);
        Self {
            prec_min: W::MIN_BY_PRECISION[precision],
            prec_max: W::MAX_BY_PRECISION[precision],
            scale_factor,
            scale,
        }
    }

    /// Bounds-check a candidate result against the result precision.
    #[inline(always)]
    fn in_precision(&self, value: W) -> Option<W> {
        (self.prec_min <= value && value <= self.prec_max).then_some(value)
    }
}

/// A checked fixed-point decimal operation on unscaled values at working width `W`.
trait CheckedDecimalOp {
    const ERROR: &'static str;

    fn checked<W>(lhs: W, rhs: W, plan: &DecimalOpPlan<W>) -> Option<W>
    where
        W: NativeDecimalType + CheckedAdd + CheckedSub + CheckedMul + CheckedDiv;
}

struct DecimalAdd;

struct DecimalSub;

struct DecimalMul;

struct DecimalDiv;

impl CheckedDecimalOp for DecimalAdd {
    const ERROR: &'static str = "decimal overflow in checked add";

    #[inline(always)]
    fn checked<W>(lhs: W, rhs: W, plan: &DecimalOpPlan<W>) -> Option<W>
    where
        W: NativeDecimalType + CheckedAdd + CheckedSub + CheckedMul + CheckedDiv,
    {
        plan.in_precision(lhs.checked_add(&rhs)?)
    }
}

impl CheckedDecimalOp for DecimalSub {
    const ERROR: &'static str = "decimal overflow in checked sub";

    #[inline(always)]
    fn checked<W>(lhs: W, rhs: W, plan: &DecimalOpPlan<W>) -> Option<W>
    where
        W: NativeDecimalType + CheckedAdd + CheckedSub + CheckedMul + CheckedDiv,
    {
        plan.in_precision(lhs.checked_sub(&rhs)?)
    }
}

impl CheckedDecimalOp for DecimalMul {
    const ERROR: &'static str = "decimal overflow in checked mul";

    #[inline(always)]
    fn checked<W>(lhs: W, rhs: W, plan: &DecimalOpPlan<W>) -> Option<W>
    where
        W: NativeDecimalType + CheckedAdd + CheckedSub + CheckedMul + CheckedDiv,
    {
        let product = lhs.checked_mul(&rhs)?;
        // The raw product has twice the operand scale; bring it back, truncating toward zero.
        let rescaled = if product == W::default() || plan.scale == 0 {
            product
        } else if plan.scale > 0 {
            product.checked_div(&plan.scale_factor?)?
        } else {
            product.checked_mul(&plan.scale_factor?)?
        };
        plan.in_precision(rescaled)
    }
}

impl CheckedDecimalOp for DecimalDiv {
    const ERROR: &'static str = "decimal division by zero or overflow in checked div";

    #[inline(always)]
    fn checked<W>(lhs: W, rhs: W, plan: &DecimalOpPlan<W>) -> Option<W>
    where
        W: NativeDecimalType + CheckedAdd + CheckedSub + CheckedMul + CheckedDiv,
    {
        if rhs == W::default() {
            return None;
        }
        if lhs == W::default() {
            return Some(lhs);
        }
        // Pre-scaling the dividend (or, for negative scales, the divisor) by 10^|s| yields a
        // quotient at the operand scale, truncating toward zero.
        let quotient = if plan.scale >= 0 {
            lhs.checked_mul(&plan.scale_factor?)?.checked_div(&rhs)?
        } else {
            lhs.checked_div(&rhs.checked_mul(&plan.scale_factor?)?)?
        };
        plan.in_precision(quotient)
    }
}

fn execute_decimal_typed<W, Op>(
    lhs: &DecimalOperand,
    rhs: &DecimalOperand,
    decimal_dtype: DecimalDType,
    result_dtype: &DType,
    validity: Validity,
    valid_rows: &Mask,
) -> VortexResult<ArrayRef>
where
    W: NativeDecimalType + CheckedAdd + CheckedSub + CheckedMul + CheckedDiv,
    DecimalValue: From<W>,
    Op: CheckedDecimalOp,
{
    let len = lhs.len();
    let plan = DecimalOpPlan::<W>::new(decimal_dtype);

    let checked = match (lhs, rhs) {
        (DecimalOperand::Array { values: lhs, .. }, DecimalOperand::Array { values: rhs, .. }) => {
            let lhs = widened_buffer::<W>(lhs);
            let rhs = widened_buffer::<W>(rhs);
            checked_decimal_lanes(len, valid_rows, |idx| {
                Op::checked(lhs[idx], rhs[idx], &plan)
            })
        }
        (DecimalOperand::Array { values: lhs, .. }, DecimalOperand::Constant { value, .. }) => {
            let lhs = widened_buffer::<W>(lhs);
            let rhs = typed_constant::<W>(value);
            checked_decimal_lanes(len, valid_rows, |idx| Op::checked(lhs[idx], rhs, &plan))
        }
        (DecimalOperand::Constant { value, .. }, DecimalOperand::Array { values: rhs, .. }) => {
            let lhs = typed_constant::<W>(value);
            let rhs = widened_buffer::<W>(rhs);
            checked_decimal_lanes(len, valid_rows, |idx| Op::checked(lhs, rhs[idx], &plan))
        }
        (
            DecimalOperand::Constant { value: lhs, .. },
            DecimalOperand::Constant { value: rhs, .. },
        ) => {
            let lhs = typed_constant::<W>(lhs);
            let rhs = typed_constant::<W>(rhs);
            let value = Op::checked(lhs, rhs, &plan)
                .ok_or_else(|| vortex_err!(InvalidArgument: "{}", Op::ERROR))?;
            return Ok(ConstantArray::new(
                Scalar::decimal(
                    DecimalValue::from(value),
                    decimal_dtype,
                    result_dtype.nullability(),
                ),
                len,
            )
            .into_array());
        }
        (DecimalOperand::Null(_), _) | (_, DecimalOperand::Null(_)) => CheckedValues::zeroed(len),
    };
    check_numeric_errors(checked.failed, Op::ERROR)?;

    Ok(DecimalArray::new(
        checked.values,
        decimal_dtype,
        validity.union_nullability(result_dtype.nullability()),
    )
    .into_array())
}

fn checked_decimal_lanes<W, F>(len: usize, valid_rows: &Mask, checked_at: F) -> CheckedValues<W>
where
    W: NativeDecimalType,
    F: FnMut(usize) -> Option<W>,
{
    match valid_rows.bit_buffer() {
        AllOr::All => checked_all_lanes(len, checked_at),
        AllOr::None => CheckedValues::zeroed(len),
        AllOr::Some(valid_bits) => checked_valid_lanes(len, valid_bits, checked_at),
    }
}

fn typed_constant<W: NativeDecimalType>(value: &DecimalValue) -> W {
    value
        .cast::<W>()
        .vortex_expect("the working width must be able to represent the constant")
}
