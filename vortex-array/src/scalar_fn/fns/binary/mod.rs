// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

#[expect(deprecated)]
pub use boolean::and_kleene;
#[expect(deprecated)]
pub use boolean::or_kleene;
use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_proto::expr as pb;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::expr::and;
use crate::expr::display::ExprDisplay;
use crate::expr::expression::Expression;
use crate::expr::lit;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ReduceNode;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::SimplifyCtx;
use crate::scalar_fn::fns::literal::Literal;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::scalar_fn::fns::operators::Operator;

pub mod boolean;
pub use boolean::BooleanExecuteAdaptor;
use boolean::BooleanFold;
pub use boolean::BooleanKernel;
pub(crate) use boolean::execute_boolean;
use boolean::fold_boolean_constants;
pub use boolean::kleene_boolean_buffer_scalar;
pub use boolean::kleene_boolean_buffers;
mod compare;
pub use compare::*;
mod numeric;
pub(crate) use numeric::*;
mod primitive_operand;

use crate::scalar::NumericOperator;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

#[derive(Clone)]
pub struct Binary;

impl ScalarFnVTable for Binary {
    type Options = Operator;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.binary");
        *ID
    }

    fn serialize(&self, instance: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            pb::BinaryOpts {
                op: (*instance).into(),
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        let opts = pb::BinaryOpts::decode(_metadata)?;
        Operator::try_from(opts.op)
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(2)
    }

    fn child_name(&self, _instance: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("lhs"),
            1 => ChildName::from("rhs"),
            _ => unreachable!("Binary has only two children"),
        }
    }

    fn fmt_sql(
        &self,
        operator: &Operator,
        expr: &dyn ExprDisplay,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "(")?;
        Display::fmt(expr.display_child(0), f)?;
        write!(f, " {} ", operator)?;
        Display::fmt(expr.display_child(1), f)?;
        write!(f, ")")
    }

    fn coerce_args(&self, operator: &Self::Options, args: &[DType]) -> VortexResult<Vec<DType>> {
        let lhs = &args[0];
        let rhs = &args[1];
        if operator.is_arithmetic() || operator.is_comparison() {
            let supertype = lhs.least_supertype(rhs).ok_or_else(|| {
                vortex_error::vortex_err!("No common supertype for {} and {}", lhs, rhs)
            })?;
            Ok(vec![supertype.clone(), supertype])
        } else {
            // Boolean And/Or: no coercion
            Ok(args.to_vec())
        }
    }

    fn return_dtype(&self, operator: &Operator, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let lhs = &arg_dtypes[0];
        let rhs = &arg_dtypes[1];

        if operator.is_arithmetic() {
            if lhs.is_primitive() && lhs.eq_ignore_nullability(rhs) {
                return Ok(lhs.with_nullability(lhs.nullability() | rhs.nullability()));
            }

            if let DType::Decimal(decimal_dtype, _) = lhs
                && lhs.eq_ignore_nullability(rhs)
            {
                let numeric_op = NumericOperator::try_from(*operator)?;
                return Ok(DType::Decimal(
                    numeric_op_result_decimal_dtype(*decimal_dtype, numeric_op)?,
                    lhs.nullability() | rhs.nullability(),
                ));
            }
            vortex_bail!(
                "incompatible types for arithmetic operation: {} {}",
                lhs,
                rhs
            );
        }

        if operator.is_comparison()
            && !lhs.eq_ignore_nullability(rhs)
            && !lhs.is_extension()
            && !rhs.is_extension()
        {
            vortex_bail!("Cannot compare different DTypes {} and {}", lhs, rhs);
        }

        Ok(DType::Bool((lhs.is_nullable() || rhs.is_nullable()).into()))
    }

    fn execute(
        &self,
        op: &Operator,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let lhs = args.get(0)?;
        let rhs = args.get(1)?;

        match op {
            Operator::Eq => execute_compare(&lhs, &rhs, CompareOperator::Eq, ctx),
            Operator::NotEq => execute_compare(&lhs, &rhs, CompareOperator::NotEq, ctx),
            Operator::Lt => execute_compare(&lhs, &rhs, CompareOperator::Lt, ctx),
            Operator::Lte => execute_compare(&lhs, &rhs, CompareOperator::Lte, ctx),
            Operator::Gt => execute_compare(&lhs, &rhs, CompareOperator::Gt, ctx),
            Operator::Gte => execute_compare(&lhs, &rhs, CompareOperator::Gte, ctx),
            Operator::And => execute_boolean(lhs, rhs, Operator::And, ctx),
            Operator::Or => execute_boolean(lhs, rhs, Operator::Or, ctx),
            Operator::Add => execute_numeric(&lhs, &rhs, NumericOperator::Add, ctx),
            Operator::Sub => execute_numeric(&lhs, &rhs, NumericOperator::Sub, ctx),
            Operator::Mul => execute_numeric(&lhs, &rhs, NumericOperator::Mul, ctx),
            Operator::Div => execute_numeric(&lhs, &rhs, NumericOperator::Div, ctx),
        }
    }

    fn reduce<T: ReduceNode>(&self, operator: &Operator, node: &T) -> VortexResult<Option<T>> {
        let bool_constant = |node: &T| node.as_constant()?.as_bool_opt().map(|value| value.value());

        let Some(fold) = fold_boolean_constants(
            *operator,
            bool_constant(&node.child(0)),
            bool_constant(&node.child(1)),
        ) else {
            return Ok(None);
        };

        let dtype = node.node_dtype()?;
        Ok(match fold {
            BooleanFold::Constant(value) => {
                let scalar = Scalar::try_new(dtype, value.map(ScalarValue::from))?;
                Some(node.new_node(Literal.bind(scalar), &[])?)
            }
            BooleanFold::Child(idx) => {
                // Only collapse into the child when it already carries the operator's result
                // dtype. A nullable constant widens the result, and dropping this node would
                // narrow it back, so leave the widening case for execution to handle.
                let child = node.child(idx);
                (child.node_dtype()? == dtype).then_some(child)
            }
        })
    }

    fn simplify_untyped(
        &self,
        operator: &Operator,
        expr: &Expression,
    ) -> VortexResult<Option<Expression>> {
        let bool_literal = |expr: &Expression| {
            expr.as_opt::<Literal>()?
                .as_bool_opt()
                .map(|value| value.value())
        };

        let Some(fold) = fold_boolean_constants(
            *operator,
            bool_literal(expr.child(0)),
            bool_literal(expr.child(1)),
        ) else {
            return Ok(None);
        };

        Ok(Some(match fold {
            // Without type information the folded literal cannot inherit the operator's
            // nullability, so a non-null result is spelled as a non-nullable literal. A null
            // boolean literal is always nullable, so it needs no such widening.
            BooleanFold::Constant(Some(value)) => lit(value),
            BooleanFold::Constant(None) => lit(Scalar::null(DType::Bool(Nullability::Nullable))),
            BooleanFold::Child(idx) => expr.child(idx).clone(),
        }))
    }

    fn simplify(
        &self,
        operator: &Operator,
        expr: &Expression,
        ctx: &dyn SimplifyCtx,
    ) -> VortexResult<Option<Expression>> {
        let is_literal_null =
            |expr: &Expression| expr.as_opt::<Literal>().is_some_and(Scalar::is_null);

        if operator.is_comparison()
            && (is_literal_null(expr.child(0)) || is_literal_null(expr.child(1)))
        {
            // Validate the comparison before reducing it. This preserves type
            // errors for expressions like `int_col = null_utf8`.
            ctx.return_dtype(expr)?;
            return Ok(Some(lit(Scalar::null(DType::Bool(Nullability::Nullable)))));
        }

        Ok(None)
    }

    fn validity(
        &self,
        operator: &Operator,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        let lhs = expression.child(0).validity()?;
        let rhs = expression.child(1).validity()?;

        Ok(match operator {
            // AND and OR are kleene logic.
            Operator::And => None,
            Operator::Or => None,
            _ => {
                // All other binary operators are null if either side is null.
                Some(and(lhs, rhs))
            }
        })
    }

    fn is_strict(&self, operator: &Operator) -> bool {
        // Kleene AND/OR is not strict (`false AND null = false`, `true OR null = true`), which is
        // consistent with `validity` returning `None` for these operators above.
        !matches!(operator, Operator::And | Operator::Or)
    }

    fn is_fallible(&self, operator: &Operator) -> bool {
        // Opt-in not out for fallibility.
        // Arithmetic operations could be better modelled here.
        let infallible = matches!(
            operator,
            Operator::Eq
                | Operator::NotEq
                | Operator::Gt
                | Operator::Gte
                | Operator::Lt
                | Operator::Lte
                | Operator::And
                | Operator::Or
        );

        !infallible
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;

    use super::*;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::Expression;
    use crate::expr::and_collect;
    use crate::expr::col;
    use crate::expr::eq;
    use crate::expr::gt;
    use crate::expr::gt_eq;
    use crate::expr::lit;
    use crate::expr::lt;
    use crate::expr::lt_eq;
    use crate::expr::not_eq;
    use crate::expr::or;
    use crate::expr::or_collect;
    use crate::expr::test_harness;
    use crate::scalar::Scalar;
    #[test]
    fn and_collect_balanced() {
        let values = vec![lit(1), lit(2), lit(3), lit(4), lit(5)];

        insta::assert_snapshot!(and_collect(values.into_iter()).unwrap().display_tree(), @r"
        vortex.binary(and)
        ├── lhs: vortex.binary(and)
        │   ├── lhs: vortex.literal(1i32)
        │   └── rhs: vortex.literal(2i32)
        └── rhs: vortex.binary(and)
            ├── lhs: vortex.binary(and)
            │   ├── lhs: vortex.literal(3i32)
            │   └── rhs: vortex.literal(4i32)
            └── rhs: vortex.literal(5i32)
        ");

        // 4 elements: and(and(1, 2), and(3, 4)) - perfectly balanced
        let values = vec![lit(1), lit(2), lit(3), lit(4)];
        insta::assert_snapshot!(and_collect(values.into_iter()).unwrap().display_tree(), @r"
        vortex.binary(and)
        ├── lhs: vortex.binary(and)
        │   ├── lhs: vortex.literal(1i32)
        │   └── rhs: vortex.literal(2i32)
        └── rhs: vortex.binary(and)
            ├── lhs: vortex.literal(3i32)
            └── rhs: vortex.literal(4i32)
        ");

        // 1 element: just the element
        let values = vec![lit(1)];
        insta::assert_snapshot!(and_collect(values.into_iter()).unwrap().display_tree(), @"vortex.literal(1i32)");

        // 0 elements: None
        let values: Vec<Expression> = vec![];
        assert!(and_collect(values.into_iter()).is_none());
    }

    #[test]
    fn or_collect_balanced() {
        // 4 elements: or(or(1, 2), or(3, 4)) - perfectly balanced
        let values = vec![lit(1), lit(2), lit(3), lit(4)];
        insta::assert_snapshot!(or_collect(values.into_iter()).unwrap().display_tree(), @r"
        vortex.binary(or)
        ├── lhs: vortex.binary(or)
        │   ├── lhs: vortex.literal(1i32)
        │   └── rhs: vortex.literal(2i32)
        └── rhs: vortex.binary(or)
            ├── lhs: vortex.literal(3i32)
            └── rhs: vortex.literal(4i32)
        ");
    }

    #[test]
    fn dtype() {
        let dtype = test_harness::struct_dtype();
        let bool1: Expression = col("bool1");
        let bool2: Expression = col("bool2");
        assert_eq!(
            and(bool1.clone(), bool2.clone())
                .return_dtype(&dtype)
                .unwrap(),
            DType::Bool(Nullability::NonNullable)
        );
        assert_eq!(
            or(bool1, bool2).return_dtype(&dtype).unwrap(),
            DType::Bool(Nullability::NonNullable)
        );

        let col1: Expression = col("col1");
        let col2: Expression = col("col2");

        assert_eq!(
            eq(col1.clone(), col2.clone()).return_dtype(&dtype).unwrap(),
            DType::Bool(Nullability::Nullable)
        );
        assert_eq!(
            not_eq(col1.clone(), col2.clone())
                .return_dtype(&dtype)
                .unwrap(),
            DType::Bool(Nullability::Nullable)
        );
        assert_eq!(
            gt(col1.clone(), col2.clone()).return_dtype(&dtype).unwrap(),
            DType::Bool(Nullability::Nullable)
        );
        assert_eq!(
            gt_eq(col1.clone(), col2.clone())
                .return_dtype(&dtype)
                .unwrap(),
            DType::Bool(Nullability::Nullable)
        );
        assert_eq!(
            lt(col1.clone(), col2.clone()).return_dtype(&dtype).unwrap(),
            DType::Bool(Nullability::Nullable)
        );
        assert_eq!(
            lt_eq(col1.clone(), col2.clone())
                .return_dtype(&dtype)
                .unwrap(),
            DType::Bool(Nullability::Nullable)
        );

        assert_eq!(
            or(lt(col1.clone(), col2.clone()), not_eq(col1, col2))
                .return_dtype(&dtype)
                .unwrap(),
            DType::Bool(Nullability::Nullable)
        );
    }

    #[test]
    fn comparison_with_typed_null_simplifies_after_type_check() -> VortexResult<()> {
        let dtype = test_harness::struct_dtype();

        let expr = eq(
            col("col1"),
            lit(Scalar::null(DType::Primitive(
                PType::U16,
                Nullability::Nullable,
            ))),
        );

        assert_eq!(
            expr.optimize_recursive(&dtype)?,
            lit(Scalar::null(DType::Bool(Nullability::Nullable)))
        );
        Ok(())
    }

    #[test]
    fn comparison_with_incompatible_null_still_type_checks() {
        let dtype = test_harness::struct_dtype();
        let expr = eq(
            col("col1"),
            lit(Scalar::null(DType::Utf8(Nullability::Nullable))),
        );

        assert!(expr.optimize_recursive(&dtype).is_err());
    }

    #[test]
    fn test_display_print() {
        let expr = gt(lit(1), lit(2));
        assert_eq!(format!("{expr}"), "(1i32 > 2i32)");
    }

    /// Regression test for GitHub issue #5947: struct comparison in filter expressions should work
    /// using `make_comparator` instead of Arrow's `cmp` functions which don't support nested types.
    #[test]
    fn test_struct_comparison() {
        use crate::IntoArray;
        use crate::arrays::StructArray;

        // Create a struct array with one element for testing.
        let lhs_struct = StructArray::from_fields(&[
            (
                "a",
                crate::arrays::PrimitiveArray::from_iter([1i32]).into_array(),
            ),
            (
                "b",
                crate::arrays::PrimitiveArray::from_iter([3i32]).into_array(),
            ),
        ])
        .unwrap()
        .into_array();

        let rhs_struct_equal = StructArray::from_fields(&[
            (
                "a",
                crate::arrays::PrimitiveArray::from_iter([1i32]).into_array(),
            ),
            (
                "b",
                crate::arrays::PrimitiveArray::from_iter([3i32]).into_array(),
            ),
        ])
        .unwrap()
        .into_array();

        let rhs_struct_different = StructArray::from_fields(&[
            (
                "a",
                crate::arrays::PrimitiveArray::from_iter([1i32]).into_array(),
            ),
            (
                "b",
                crate::arrays::PrimitiveArray::from_iter([4i32]).into_array(),
            ),
        ])
        .unwrap()
        .into_array();

        // Test using binary method directly
        let result_equal = lhs_struct.binary(rhs_struct_equal, Operator::Eq).unwrap();
        assert_eq!(
            result_equal
                .execute_scalar(0, &mut array_session().create_execution_ctx())
                .vortex_expect("value"),
            Scalar::bool(true, Nullability::NonNullable),
            "Equal structs should be equal"
        );

        let result_different = lhs_struct
            .binary(rhs_struct_different, Operator::Eq)
            .unwrap();
        assert_eq!(
            result_different
                .execute_scalar(0, &mut array_session().create_execution_ctx())
                .vortex_expect("value"),
            Scalar::bool(false, Nullability::NonNullable),
            "Different structs should not be equal"
        );
    }

    /// A single non-nullable boolean field, for reducing expressions over an array tree.
    fn bool_array() -> VortexResult<ArrayRef> {
        use crate::IntoArray;
        use crate::arrays::BoolArray;
        use crate::arrays::StructArray;

        let field = BoolArray::from_iter([true, false, true]).into_array();
        Ok(StructArray::from_fields(&[("a", field)])?.into_array())
    }

    #[test]
    fn boolean_reduce_drops_annihilated_operand() -> VortexResult<()> {
        use crate::arrays::BoolArray;

        let mut ctx = array_session().create_execution_ctx();
        let array = bool_array()?;

        // `false AND x` is `false` for every `x`, so `x` must not survive in the array tree.
        let reduced = array.clone().apply(&and(lit(false), col("a")))?;
        assert_eq!(reduced.nchildren(), 0, "operand should have been dropped");
        assert_eq!(reduced.dtype(), &DType::Bool(Nullability::NonNullable));
        assert_arrays_eq!(
            reduced,
            BoolArray::from_iter([false, false, false]),
            &mut ctx
        );

        // `true OR x` is `true` for every `x`, from either side.
        let reduced = array.clone().apply(&or(col("a"), lit(true)))?;
        assert_eq!(reduced.nchildren(), 0, "operand should have been dropped");
        assert_arrays_eq!(reduced, BoolArray::from_iter([true, true, true]), &mut ctx);

        // The identity element collapses to the other operand.
        let reduced = array.apply(&and(lit(true), col("a")))?;
        assert_arrays_eq!(reduced, BoolArray::from_iter([true, false, true]), &mut ctx);
        Ok(())
    }

    #[test]
    fn boolean_reduce_collapses_transitively() -> VortexResult<()> {
        use crate::arrays::BoolArray;

        let mut ctx = array_session().create_execution_ctx();

        // Pruning falsifiers nest a guard above the predicate it guards, so a constant guard
        // has to collapse the operands above it too, not just its own node.
        let guarded = and(
            and(col("a"), lit(false)),
            or(col("a"), not_eq(col("a"), col("a"))),
        );
        let reduced = bool_array()?.apply(&guarded)?;

        assert_eq!(reduced.nchildren(), 0, "whole tree should have collapsed");
        assert_arrays_eq!(
            reduced,
            BoolArray::from_iter([false, false, false]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn boolean_reduce_preserves_nullable_result_dtype() -> VortexResult<()> {
        use crate::arrays::BoolArray;

        let mut ctx = array_session().create_execution_ctx();

        // A nullable `true` widens the result to nullable, so collapsing into the non-nullable
        // operand would narrow the dtype of the node being replaced.
        let expr = and(lit(Scalar::bool(true, Nullability::Nullable)), col("a"));
        let reduced = bool_array()?.apply(&expr)?;

        assert_eq!(reduced.dtype(), &DType::Bool(Nullability::Nullable));
        assert_arrays_eq!(
            reduced,
            BoolArray::from_iter([Some(true), Some(false), Some(true)]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn boolean_reduce_folds_two_null_operands() -> VortexResult<()> {
        let null_bool = || lit(Scalar::null(DType::Bool(Nullability::Nullable)));
        let reduced = bool_array()?.apply(&and(null_bool(), null_bool()))?;

        assert_eq!(reduced.dtype(), &DType::Bool(Nullability::Nullable));
        assert_eq!(reduced.nchildren(), 0, "operands should have been dropped");
        Ok(())
    }

    #[test]
    fn boolean_reduce_keeps_value_dependent_null() -> VortexResult<()> {
        // `null AND x` is `false` where `x` is false and `null` elsewhere, so it cannot fold.
        let expr = and(
            lit(Scalar::null(DType::Bool(Nullability::Nullable))),
            col("a"),
        );
        let reduced = bool_array()?.apply(&expr)?;

        assert_eq!(reduced.nchildren(), 2, "neither operand should be dropped");
        Ok(())
    }

    #[test]
    fn test_or_kleene_validity() {
        let mut ctx = array_session().create_execution_ctx();
        use crate::IntoArray;
        use crate::arrays::BoolArray;
        use crate::arrays::StructArray;
        use crate::expr::col;

        let struct_arr = StructArray::from_fields(&[
            ("a", BoolArray::from_iter([Some(true)]).into_array()),
            (
                "b",
                BoolArray::from_iter([Option::<bool>::None]).into_array(),
            ),
        ])
        .unwrap()
        .into_array();

        let expr = or(col("a"), col("b"));
        let result = struct_arr.apply(&expr).unwrap();

        assert_arrays_eq!(
            result,
            BoolArray::from_iter([Some(true)]).into_array(),
            &mut ctx
        )
    }

    #[test]
    fn test_scalar_subtract_unsigned() {
        let mut ctx = array_session().create_execution_ctx();
        use vortex_buffer::buffer;

        use crate::IntoArray;
        use crate::arrays::ConstantArray;
        use crate::arrays::PrimitiveArray;

        let values = buffer![1u16, 2, 3].into_array();
        let rhs = ConstantArray::new(Scalar::from(1u16), 3).into_array();
        let result = values.binary(rhs, Operator::Sub).unwrap();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([0u16, 1, 2]), &mut ctx);
    }

    #[test]
    fn test_scalar_subtract_signed() {
        let mut ctx = array_session().create_execution_ctx();
        use vortex_buffer::buffer;

        use crate::IntoArray;
        use crate::arrays::ConstantArray;
        use crate::arrays::PrimitiveArray;

        let values = buffer![1i64, 2, 3].into_array();
        let rhs = ConstantArray::new(Scalar::from(-1i64), 3).into_array();
        let result = values.binary(rhs, Operator::Sub).unwrap();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([2i64, 3, 4]), &mut ctx);
    }

    #[test]
    fn test_scalar_subtract_nullable() {
        let mut ctx = array_session().create_execution_ctx();
        use crate::IntoArray;
        use crate::arrays::ConstantArray;
        use crate::arrays::PrimitiveArray;

        let values = PrimitiveArray::from_option_iter([Some(1u16), Some(2), None, Some(3)]);
        let rhs = ConstantArray::new(Scalar::from(Some(1u16)), 4).into_array();
        let result = values.into_array().binary(rhs, Operator::Sub).unwrap();
        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(0u16), Some(1), None, Some(2)]),
            &mut ctx
        );
    }

    #[test]
    fn test_scalar_subtract_float() {
        let mut ctx = array_session().create_execution_ctx();
        use vortex_buffer::buffer;

        use crate::IntoArray;
        use crate::arrays::ConstantArray;
        use crate::arrays::PrimitiveArray;

        let values = buffer![1.0f64, 2.0, 3.0].into_array();
        let rhs = ConstantArray::new(Scalar::from(-1f64), 3).into_array();
        let result = values.binary(rhs, Operator::Sub).unwrap();
        assert_arrays_eq!(
            result,
            PrimitiveArray::from_iter([2.0f64, 3.0, 4.0]),
            &mut ctx
        );
    }

    #[test]
    fn test_scalar_subtract_float_underflow_is_ok() {
        use vortex_buffer::buffer;

        use crate::IntoArray;
        use crate::arrays::ConstantArray;

        let values = buffer![f32::MIN, 2.0, 3.0].into_array();
        let rhs1 = ConstantArray::new(Scalar::from(1.0f32), 3).into_array();
        let _results = values.binary(rhs1, Operator::Sub).unwrap();
        let values = buffer![f32::MIN, 2.0, 3.0].into_array();
        let rhs2 = ConstantArray::new(Scalar::from(f32::MAX), 3).into_array();
        let _results = values.binary(rhs2, Operator::Sub).unwrap();
    }
}
