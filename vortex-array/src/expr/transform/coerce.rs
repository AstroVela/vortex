// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Expression-level type coercion pass.

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::dtype::DType;
use crate::expr::Expression;
use crate::expr::Scope;
use crate::expr::cast;

/// Rewrite an expression tree to insert casts where a scalar function's `coerce_args` demands
/// a different type than what the child currently produces.
///
/// The rewrite is bottom-up: children are coerced first, then each parent node checks whether
/// its children match the coerced argument types.
pub fn coerce_expression(expr: Expression, scope: &DType) -> VortexResult<Expression> {
    coerce_expression_scope(expr, &Scope::new(scope.clone()))
}

/// Rewrite an expression against a lexical scope, inserting casts demanded by scalar functions
/// and by the ordinary inputs of higher-order functions.
///
/// A higher-order function's lambdas are bound only after its ordinary inputs have been coerced:
/// those final input dtypes establish the lexical parameter scope of each lambda.
pub fn coerce_expression_scope(expr: Expression, scope: &Scope) -> VortexResult<Expression> {
    coerce_expression_inner(expr, scope)
}

fn coerce_expression_inner(expr: Expression, scope: &Scope) -> VortexResult<Expression> {
    match &expr {
        Expression::Root | Expression::Variable(_) | Expression::Lambda(_) => Ok(expr),
        Expression::Scalar {
            scalar_fn,
            children,
        } => {
            let children = children
                .iter()
                .cloned()
                .map(|child| coerce_expression_inner(child, scope))
                .collect::<VortexResult<Vec<_>>>()?;
            let children =
                coerce_children(children, scope, |dtypes| scalar_fn.coerce_args(dtypes))?;
            Expression::try_new(scalar_fn.clone(), children)
        }
        Expression::HigherOrder {
            higher_order_fn,
            children,
            lambdas,
        } => {
            let children = children
                .iter()
                .cloned()
                .map(|child| coerce_expression_inner(child, scope))
                .collect::<VortexResult<Vec<_>>>()?;
            let children = coerce_children(children, scope, |dtypes| {
                higher_order_fn.coerce_input_args(dtypes)
            })?;

            let arg_dtypes = children
                .iter()
                .map(|child| child.return_dtype_scope(scope))
                .collect::<VortexResult<Vec<_>>>()?;
            let lambda_syntax = lambdas
                .iter()
                .map(|lambda| {
                    lambda.as_lambda().ok_or_else(|| {
                        vortex_err!(
                            "higher-order expression '{}' contained a non-lambda argument",
                            higher_order_fn.id()
                        )
                    })
                })
                .collect::<VortexResult<Vec<_>>>()?;
            let signatures = higher_order_fn.lambda_signatures(&arg_dtypes, &lambda_syntax)?;
            vortex_ensure!(
                signatures.len() == lambda_syntax.len(),
                "higher-order function '{}' produced {} lambda signatures for {} lambda arguments",
                higher_order_fn.id(),
                signatures.len(),
                lambda_syntax.len(),
            );
            let lambdas = lambda_syntax
                .into_iter()
                .zip(signatures.iter())
                .map(|(lambda, signature)| {
                    let body = coerce_expression_inner(
                        lambda.body().clone(),
                        &signature.scope(lambda, scope)?,
                    )?;
                    Ok(Expression::Lambda(lambda.clone().with_body(body)))
                })
                .collect::<VortexResult<Vec<_>>>()?;

            Expression::try_new_higher_order(higher_order_fn.clone(), children, lambdas)
        }
    }
}

fn coerce_children(
    children: Vec<Expression>,
    scope: &Scope,
    coerce: impl FnOnce(&[DType]) -> VortexResult<Vec<DType>>,
) -> VortexResult<Vec<Expression>> {
    let child_dtypes = children
        .iter()
        .map(|child| child.return_dtype_scope(scope))
        .collect::<VortexResult<Vec<_>>>()?;
    let coerced_dtypes = coerce(&child_dtypes)?;
    vortex_ensure!(
        child_dtypes.len() == coerced_dtypes.len(),
        "argument coercion returned {} types for {} children",
        coerced_dtypes.len(),
        child_dtypes.len(),
    );

    children
        .into_iter()
        .zip(child_dtypes.into_iter().zip(coerced_dtypes))
        .map(|(child, (child_dtype, target))| {
            if child_dtype.eq_ignore_nullability(&target)
                && child_dtype.nullability() == target.nullability()
            {
                Ok(child)
            } else {
                Ok(cast(child, target))
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::expr::col;
    use crate::expr::lit;
    use crate::expr::transform::coerce::coerce_expression;
    use crate::scalar::Scalar;
    use crate::scalar_fn::ScalarFnVTableExt;
    use crate::scalar_fn::fns::binary::Binary;
    use crate::scalar_fn::fns::cast::Cast;
    use crate::scalar_fn::fns::operators::Operator;

    fn test_scope() -> DType {
        DType::Struct(
            StructFields::new(
                ["x", "y"].into(),
                vec![
                    DType::Primitive(PType::I32, NonNullable),
                    DType::Primitive(PType::I64, NonNullable),
                ],
            ),
            NonNullable,
        )
    }

    #[test]
    fn mixed_type_comparison_inserts_cast() -> VortexResult<()> {
        let scope = test_scope();
        // x (I32) < y (I64) => should cast x to I64
        let expr = Binary.new_expr(Operator::Lt, [col("x"), col("y")]);
        let coerced = coerce_expression(expr, &scope)?;

        // The LHS child should now be a cast expression
        assert!(coerced.child(0).is::<Cast>());
        // The coerced LHS should return I64
        assert_eq!(
            coerced.child(0).return_dtype(&scope)?,
            DType::Primitive(PType::I64, NonNullable)
        );
        // The RHS should be unchanged
        assert!(!coerced.child(1).is::<Cast>());
        Ok(())
    }

    #[test]
    fn same_type_comparison_no_cast() -> VortexResult<()> {
        let scope = test_scope();
        // x (I32) < x (I32) => no cast needed
        let expr = Binary.new_expr(Operator::Lt, [col("x"), col("x")]);
        let coerced = coerce_expression(expr, &scope)?;

        // Neither child should be a cast
        assert!(!coerced.child(0).is::<Cast>());
        assert!(!coerced.child(1).is::<Cast>());
        Ok(())
    }

    #[test]
    fn mixed_type_arithmetic_coerces_both() -> VortexResult<()> {
        let scope = DType::Struct(
            StructFields::new(
                ["a", "b"].into(),
                vec![
                    DType::Primitive(PType::U8, NonNullable),
                    DType::Primitive(PType::I32, NonNullable),
                ],
            ),
            NonNullable,
        );
        // a (U8) + b (I32) => both should be coerced to I32
        // U8 + I32: unsigned_signed_supertype(U8, I32) => max(1,4)=4 => I64
        let expr = Binary.new_expr(Operator::Add, [col("a"), col("b")]);
        let coerced = coerce_expression(expr, &scope)?;

        // LHS (U8) should be cast
        assert!(coerced.child(0).is::<Cast>());
        // Both should return the same supertype
        let lhs_dt = coerced.child(0).return_dtype(&scope)?;
        let rhs_dt = coerced.child(1).return_dtype(&scope)?;
        assert_eq!(lhs_dt, rhs_dt);
        Ok(())
    }

    #[test]
    fn decimal_arithmetic_coerces_precision_and_scale() -> VortexResult<()> {
        let common_dtype = DType::Decimal(DecimalDType::new(4, 2), NonNullable);
        let result_dtype = DType::Decimal(DecimalDType::new(5, 2), NonNullable);
        let scope = DType::Struct(
            StructFields::new(
                ["a", "b"].into(),
                vec![
                    DType::Decimal(DecimalDType::new(3, 1), NonNullable),
                    common_dtype,
                ],
            ),
            NonNullable,
        );
        let expr = Binary.new_expr(Operator::Add, [col("a"), col("b")]);

        let coerced = coerce_expression(expr, &scope)?;

        assert!(coerced.child(0).is::<Cast>());
        assert!(!coerced.child(1).is::<Cast>());
        assert_eq!(coerced.return_dtype(&scope)?, result_dtype);
        Ok(())
    }

    #[test]
    fn boolean_operators_no_coercion() -> VortexResult<()> {
        let scope = DType::Struct(
            StructFields::new(
                ["p", "q"].into(),
                vec![DType::Bool(NonNullable), DType::Bool(NonNullable)],
            ),
            NonNullable,
        );
        let expr = Binary.new_expr(Operator::And, [col("p"), col("q")]);
        let coerced = coerce_expression(expr, &scope)?;

        assert!(!coerced.child(0).is::<Cast>());
        assert!(!coerced.child(1).is::<Cast>());
        Ok(())
    }

    #[test]
    fn literal_coercion() -> VortexResult<()> {
        let scope = DType::Struct(
            StructFields::new(
                ["x"].into(),
                vec![DType::Primitive(PType::I64, NonNullable)],
            ),
            NonNullable,
        );
        // x (I64) + 1i32 => literal should be cast to I64
        let expr = Binary.new_expr(Operator::Add, [col("x"), lit(Scalar::from(1i32))]);
        let coerced = coerce_expression(expr, &scope)?;

        // The RHS (literal) should be cast to I64
        assert!(coerced.child(1).is::<Cast>());
        assert_eq!(
            coerced.child(1).return_dtype(&scope)?,
            DType::Primitive(PType::I64, NonNullable)
        );
        Ok(())
    }
}
