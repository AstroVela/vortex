// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pushdown planning for DataFusion decimal arithmetic.
//!
//! DataFusion evaluates decimal `+`/`-` with the arrow-rs `numeric` kernels, which rescale both
//! operands to the wider scale and derive the result type with the Hive rule:
//!
//! ```text
//! s = max(s1, s2)
//! p = max(p1 - s1, p2 - s2) + s + 1    (saturated at the physical type's max precision)
//! ```
//!
//! Vortex's decimal kernels instead require both operands to share one decimal dtype and widen
//! the result by a single carry digit, capped at Vortex's maximum precision. Vortex's own
//! coercion pass ([`coerce_expression`]) aligns mismatched operands to their least supertype —
//! the same `(p - 1, s)` dtype arrow rescales to internally — so a converted expression run
//! through that pass reproduces DataFusion's result type and values exactly, as long as arrow
//! does not saturate.
//!
//! When arrow saturates (e.g. two `Decimal128(38, 0)` inputs keep precision 38 while Vortex
//! would widen to 39), the result types and overflow behaviour diverge, so such expressions —
//! and decimal Mul/Div, which have no Vortex kernels — must not be pushed down and are instead
//! evaluated by DataFusion.

use arrow_schema::DECIMAL32_MAX_PRECISION;
use arrow_schema::DECIMAL64_MAX_PRECISION;
use arrow_schema::DECIMAL128_MAX_PRECISION;
use arrow_schema::DECIMAL256_MAX_PRECISION;
use arrow_schema::DataType;
use datafusion_common::Result as DFResult;
use datafusion_common::exec_datafusion_err;
use datafusion_expr::Operator as DFOperator;
use vortex::dtype::DType;
use vortex::expr::Expression;
use vortex::expr::transform::coerce_expression;
use vortex::scalar_fn::fns::binary::Binary;

/// Decide whether a DataFusion decimal arithmetic expression can be pushed down to Vortex.
///
/// Returns `false` when the expression must stay in DataFusion: non-`+`/`-` operators (Vortex
/// has no decimal Mul/Div kernels yet), operands of different physical decimal widths, or a
/// result precision that arrow saturates at the physical type's ceiling.
pub(crate) fn decimal_arithmetic_is_pushable(
    op: &DFOperator,
    lhs: &DataType,
    rhs: &DataType,
) -> bool {
    if !matches!(op, DFOperator::Plus | DFOperator::Minus) {
        return false;
    }

    let (Some((p1, s1, lhs_max_precision)), Some((p2, s2, rhs_max_precision))) =
        (decimal_parts(lhs), decimal_parts(rhs))
    else {
        return false;
    };
    // DataFusion coerces mixed physical widths before execution; reject them if they show up
    // here rather than guessing at arrow's cross-width semantics.
    if lhs_max_precision != rhs_max_precision {
        return false;
    }

    let scale = s1.max(s2);
    let integer_digits = (i32::from(p1) - i32::from(s1)).max(i32::from(p2) - i32::from(s2));
    let precision = integer_digits + i32::from(scale) + 1;
    // Arrow saturates the result precision at the ceiling; Vortex would widen instead.
    precision <= i32::from(lhs_max_precision)
}

/// Align decimal arithmetic operands in a converted expression by running Vortex's coercion
/// pass ([`coerce_expression`]) over it.
///
/// The pass is applied only when the expression contains decimal arithmetic: for everything
/// else the converted operand dtypes already match (DataFusion coerces them), and skipping the
/// pass avoids inserting nullability-only casts into unrelated expressions.
pub(crate) fn coerce_decimal_arithmetic(expr: Expression, scope: &DType) -> DFResult<Expression> {
    if !contains_decimal_arithmetic(&expr, scope) {
        return Ok(expr);
    }
    coerce_expression(expr, scope)
        .map_err(|e| exec_datafusion_err!("Failed to coerce decimal arithmetic expression: {e}"))
}

/// Whether any arithmetic [`Binary`] node in the tree has a decimal-typed child. A child whose
/// dtype cannot be resolved yet (e.g. arithmetic over still-unaligned operands) counts as
/// decimal, since only coercion can make it resolvable.
fn contains_decimal_arithmetic(expr: &Expression, scope: &DType) -> bool {
    if expr.is::<Binary>()
        && expr.as_::<Binary>().is_arithmetic()
        && expr.children().iter().any(|child| {
            child
                .return_dtype(scope)
                .map(|dtype| matches!(dtype, DType::Decimal(..)))
                .unwrap_or(true)
        })
    {
        return true;
    }
    expr.children()
        .iter()
        .any(|child| contains_decimal_arithmetic(child, scope))
}

fn decimal_parts(dt: &DataType) -> Option<(u8, i8, u8)> {
    match dt {
        DataType::Decimal32(p, s) => Some((*p, *s, DECIMAL32_MAX_PRECISION)),
        DataType::Decimal64(p, s) => Some((*p, *s, DECIMAL64_MAX_PRECISION)),
        DataType::Decimal128(p, s) => Some((*p, *s, DECIMAL128_MAX_PRECISION)),
        DataType::Decimal256(p, s) => Some((*p, *s, DECIMAL256_MAX_PRECISION)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::same_dtype(DataType::Decimal128(10, 2), DataType::Decimal128(10, 2), true)]
    #[case::mixed_scale(DataType::Decimal128(10, 2), DataType::Decimal128(12, 4), true)]
    #[case::mixed_precision(DataType::Decimal128(10, 2), DataType::Decimal128(20, 2), true)]
    #[case::decimal128_ceiling(DataType::Decimal128(38, 0), DataType::Decimal128(38, 0), false)]
    #[case::decimal128_ceiling_by_rescale(
        DataType::Decimal128(38, 0),
        DataType::Decimal128(38, 10),
        false
    )]
    #[case::decimal256_below_ceiling(
        DataType::Decimal256(38, 0),
        DataType::Decimal256(38, 0),
        true
    )]
    #[case::decimal256_ceiling(DataType::Decimal256(76, 0), DataType::Decimal256(76, 0), false)]
    #[case::negative_scale(DataType::Decimal128(10, -2), DataType::Decimal128(10, -2), true)]
    #[case::mixed_width(DataType::Decimal128(10, 2), DataType::Decimal256(10, 2), false)]
    #[case::non_decimal(DataType::Decimal128(10, 2), DataType::Int64, false)]
    fn test_decimal_arithmetic_is_pushable(
        #[case] lhs: DataType,
        #[case] rhs: DataType,
        #[case] expected: bool,
    ) {
        assert_eq!(
            decimal_arithmetic_is_pushable(&DFOperator::Plus, &lhs, &rhs),
            expected
        );
        assert_eq!(
            decimal_arithmetic_is_pushable(&DFOperator::Minus, &lhs, &rhs),
            expected
        );
    }

    #[rstest]
    #[case::mul(DFOperator::Multiply)]
    #[case::div(DFOperator::Divide)]
    #[case::modulo(DFOperator::Modulo)]
    fn test_decimal_arithmetic_unsupported_ops(#[case] op: DFOperator) {
        let dt = DataType::Decimal128(10, 2);
        assert!(!decimal_arithmetic_is_pushable(&op, &dt, &dt));
    }
}
