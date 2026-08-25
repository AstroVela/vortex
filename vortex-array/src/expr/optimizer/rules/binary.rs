// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::RuleRegistry;
use super::preserve_dtype;
use crate::expr::BoundExpression;
use crate::expr::ExpressionId;
use crate::expr::bound;
use crate::expr::optimizer::BoundExpressionRewriteRule;
use crate::scalar::Scalar;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::fns::between::Between;
use crate::scalar_fn::fns::between::BetweenOptions;
use crate::scalar_fn::fns::between::StrictComparison;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::get_item::GetItem;
use crate::scalar_fn::fns::literal::Literal;
use crate::scalar_fn::fns::operators::Operator;

pub(super) fn register(registry: &mut RuleRegistry) {
    registry.register(BinaryBoolean);
    registry.register(BinaryNullComparison);
    registry.register(FindBetween);
}

/// Simplifies `AND` and `OR` with literal operands using Kleene boolean semantics.
///
/// # Example
///
/// ```text
/// original: and(value, lit(true))
/// rewritten: value
/// ```
#[derive(Debug)]
struct BinaryBoolean;

impl BoundExpressionRewriteRule for BinaryBoolean {
    fn expression_id(&self) -> ExpressionId {
        Binary.id()
    }

    fn rewrite(&self, expr: &BoundExpression) -> VortexResult<Option<BoundExpression>> {
        let operator = expr.as_::<Binary>();
        let lhs = expr.child(0);
        let rhs = expr.child(1);
        let bool_literal = |expr: &BoundExpression| {
            expr.as_opt::<Literal>()?
                .as_bool_opt()
                .map(|value| value.value())
        };

        let replacement = match operator {
            Operator::And => match (bool_literal(lhs), bool_literal(rhs)) {
                (Some(Some(false)), _) | (_, Some(Some(false))) => {
                    Some(bound::lit(Scalar::bool(false, expr.dtype().nullability())))
                }
                (Some(Some(true)), _) => Some(preserve_dtype(rhs.clone(), expr.dtype())?),
                (_, Some(Some(true))) => Some(preserve_dtype(lhs.clone(), expr.dtype())?),
                (Some(None), Some(None)) => Some(lhs.clone()),
                _ => None,
            },
            Operator::Or => match (bool_literal(lhs), bool_literal(rhs)) {
                (Some(Some(true)), _) | (_, Some(Some(true))) => {
                    Some(bound::lit(Scalar::bool(true, expr.dtype().nullability())))
                }
                (Some(Some(false)), _) => Some(preserve_dtype(rhs.clone(), expr.dtype())?),
                (_, Some(Some(false))) => Some(preserve_dtype(lhs.clone(), expr.dtype())?),
                (Some(None), Some(None)) => Some(lhs.clone()),
                _ => None,
            },
            _ => None,
        };
        Ok(replacement)
    }
}

/// Replaces a comparison against a null literal with a null boolean literal.
///
/// # Example
///
/// ```text
/// original: eq(value, lit(Scalar::null(nullable_i32)))
/// rewritten: lit(Scalar::null(nullable_bool))
/// ```
#[derive(Debug)]
struct BinaryNullComparison;

impl BoundExpressionRewriteRule for BinaryNullComparison {
    fn expression_id(&self) -> ExpressionId {
        Binary.id()
    }

    fn rewrite(&self, expr: &BoundExpression) -> VortexResult<Option<BoundExpression>> {
        if !expr.as_::<Binary>().is_comparison() {
            return Ok(None);
        }
        let is_null_literal =
            |child: &BoundExpression| child.as_opt::<Literal>().is_some_and(Scalar::is_null);
        if !is_null_literal(expr.child(0)) && !is_null_literal(expr.child(1)) {
            return Ok(None);
        }

        Ok(Some(bound::lit(Scalar::null(expr.dtype().clone()))))
    }
}

/// Combines compatible lower- and upper-bound conjuncts into `between` expressions.
///
/// # Example
///
/// ```text
/// original: and(gt_eq(x, lit(1)), lt(x, lit(10)))
/// rewritten: between(x, lit(1), lit(10), BetweenOptions { lower_strict: NonStrict, upper_strict: Strict })
/// ```
#[derive(Debug)]
struct FindBetween;

impl BoundExpressionRewriteRule for FindBetween {
    fn expression_id(&self) -> ExpressionId {
        Binary.id()
    }

    fn rewrite(&self, expr: &BoundExpression) -> VortexResult<Option<BoundExpression>> {
        if expr.as_opt::<Binary>() != Some(&Operator::And) {
            return Ok(None);
        }

        let mut conjuncts = collect_conjuncts(expr);
        let mut rewritten = Vec::with_capacity(conjuncts.len());
        let mut changed = false;
        for idx in 0..conjuncts.len() {
            let Some(conjunct) = conjuncts.get(idx).cloned() else {
                continue;
            };
            let mut matched = false;
            for other_idx in (idx + 1)..conjuncts.len() {
                let Some(other) = conjuncts.get(other_idx) else {
                    continue;
                };
                if let Some(between) = match_between(&conjunct, other)? {
                    rewritten.push(between);
                    conjuncts.remove(other_idx);
                    matched = true;
                    changed = true;
                    break;
                }
            }
            if !matched {
                rewritten.push(conjunct);
            }
        }

        if !changed {
            return Ok(None);
        }
        Ok(bound::and_collect(rewritten))
    }
}

fn collect_conjuncts(expr: &BoundExpression) -> Vec<BoundExpression> {
    let mut stack = vec![expr];
    let mut conjuncts = Vec::new();
    while let Some(expr) = stack.pop() {
        if expr.as_opt::<Binary>() == Some(&Operator::And) {
            stack.push(expr.child(1));
            stack.push(expr.child(0));
        } else {
            conjuncts.push(expr.clone());
        }
    }
    conjuncts
}

fn match_between(
    lhs: &BoundExpression,
    rhs: &BoundExpression,
) -> VortexResult<Option<BoundExpression>> {
    let (Some(lhs_op), Some(rhs_op)) = (lhs.as_opt::<Binary>(), rhs.as_opt::<Binary>()) else {
        return Ok(None);
    };
    if lhs.child(0) == lhs.child(1) || rhs.child(0) == rhs.child(1) {
        return Ok(None);
    }

    let lhs = normalize_get_item_comparison(lhs, *lhs_op)?;
    let rhs = normalize_get_item_comparison(rhs, *rhs_op)?;
    let (Some(lhs), Some(rhs)) = (lhs, rhs) else {
        return Ok(None);
    };
    if lhs.child(0) != rhs.child(0) {
        return Ok(None);
    }

    let (lower, upper) = match (lhs.as_::<Binary>(), rhs.as_::<Binary>()) {
        (Operator::Lt | Operator::Lte, Operator::Gt | Operator::Gte) => (rhs, lhs),
        (Operator::Gt | Operator::Gte, Operator::Lt | Operator::Lte) => (lhs, rhs),
        _ => return Ok(None),
    };
    let lower_lit = lower.child(1).as_opt::<Literal>();
    let upper_lit = upper.child(1).as_opt::<Literal>();
    if lower_lit.is_none_or(Scalar::is_null) || upper_lit.is_none_or(Scalar::is_null) {
        return Ok(None);
    }

    let lower_strict = comparison_strictness(*lower.as_::<Binary>())?;
    let upper_strict = comparison_strictness(*upper.as_::<Binary>())?;
    Ok(Some(Between.try_new_bound_expr(
        BetweenOptions {
            lower_strict,
            upper_strict,
        },
        [
            lower.child(0).clone(),
            lower.child(1).clone(),
            upper.child(1).clone(),
        ],
    )?))
}

fn normalize_get_item_comparison(
    expr: &BoundExpression,
    operator: Operator,
) -> VortexResult<Option<BoundExpression>> {
    match (expr.child(0).is::<GetItem>(), expr.child(1).is::<GetItem>()) {
        (true, false) => Ok(Some(expr.clone())),
        (false, true) => {
            let Some(swapped) = operator.swap() else {
                return Ok(None);
            };
            Ok(Some(Binary.try_new_bound_expr(
                swapped,
                [expr.child(1).clone(), expr.child(0).clone()],
            )?))
        }
        _ => Ok(None),
    }
}

fn comparison_strictness(operator: Operator) -> VortexResult<StrictComparison> {
    match operator {
        Operator::Lt | Operator::Gt => Ok(StrictComparison::Strict),
        Operator::Lte | Operator::Gte => Ok(StrictComparison::NonStrict),
        _ => Err(vortex_err!(
            "expected an inequality operator, got {operator}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::BoundExpression;
    use crate::expr::bound;
    use crate::expr::optimizer::BoundExpressionOptimizer;
    use crate::scalar::Scalar;
    use crate::scalar_fn::fns::between::Between;
    use crate::scalar_fn::fns::between::BetweenOptions;
    use crate::scalar_fn::fns::between::StrictComparison;

    fn optimize(expr: &BoundExpression) -> VortexResult<BoundExpression> {
        BoundExpressionOptimizer::default().optimize(expr)
    }

    #[test]
    fn boolean_annihilator_preserves_nullable_dtype() -> VortexResult<()> {
        let nullable_bool = Scalar::bool(true, Nullability::Nullable);
        let expr = bound::and(bound::lit(false), bound::lit(nullable_bool));

        assert_eq!(
            optimize(&expr)?,
            bound::lit(Scalar::bool(false, Nullability::Nullable))
        );
        Ok(())
    }

    #[test]
    fn null_comparison_folds_to_nullable_null() -> VortexResult<()> {
        let nullable_i32 = DType::Primitive(PType::I32, Nullability::Nullable);
        let expr = bound::eq(
            bound::root(nullable_i32.clone()),
            bound::lit(Scalar::null(nullable_i32)),
        );

        assert_eq!(
            optimize(&expr)?,
            bound::lit(Scalar::null(DType::Bool(Nullability::Nullable)))
        );
        Ok(())
    }

    fn comparison_scope() -> DType {
        DType::struct_(
            [("x", DType::Primitive(PType::I32, Nullability::NonNullable))],
            Nullability::NonNullable,
        )
    }

    #[test]
    fn comparison_pair_lowers_to_between() -> VortexResult<()> {
        let scope = comparison_scope();
        let x = bound::col("x", scope);
        let expr = bound::and(
            bound::gt_eq(x.clone(), bound::lit(2i32)),
            bound::lt(x.clone(), bound::lit(5i32)),
        );

        assert_eq!(
            optimize(&expr)?,
            bound::between(
                x,
                bound::lit(2i32),
                bound::lit(5i32),
                BetweenOptions {
                    lower_strict: StrictComparison::NonStrict,
                    upper_strict: StrictComparison::Strict,
                }
            )
        );
        Ok(())
    }

    #[test]
    fn null_bound_does_not_lower_to_between() -> VortexResult<()> {
        let scope = comparison_scope();
        let x = bound::col("x", scope);
        let null = bound::lit(Scalar::null(DType::Primitive(
            PType::I32,
            Nullability::Nullable,
        )));
        let expr = bound::and(
            bound::gt_eq(x.clone(), null),
            bound::lt(x, bound::lit(5i32)),
        );

        assert!(!optimize(&expr)?.contains::<Between>()?);
        Ok(())
    }
}
