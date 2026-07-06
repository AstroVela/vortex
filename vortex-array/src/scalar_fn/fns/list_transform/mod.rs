// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_proto::expr as pb;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::FixedSizeList;
use crate::arrays::FixedSizeListArray;
use crate::arrays::List;
use crate::arrays::ListArray;
use crate::arrays::ListView;
use crate::arrays::ListViewArray;
use crate::arrays::fixed_size_list::FixedSizeListArrayExt;
use crate::arrays::list::ListArrayExt;
use crate::arrays::listview::ListViewArrayExt;
use crate::dtype::DType;
use crate::expr::Expression;
use crate::expr::label_is_fallible;
use crate::expr::list_transform;
use crate::expr::proto::ExprSerializeProtoExt;
use crate::expr::root;
use crate::expr::transform::replace;
use crate::scalar::Scalar;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::list_length::AnyList;
use crate::scalar_fn::fns::root::Root;

/// Options for [`ListTransform`], holding the lambda body.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ListTransformOptions {
    /// The lambda body, evaluated with the list's *elements* array as its root scope: within
    /// this expression, `root()` refers to the element rather than the enclosing row.
    pub body: Expression,
}

impl Display for ListTransformOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "-> {}", self.body)
    }
}

/// Applies an expression to every element of a `List`, `ListView`, or `FixedSizeList` typed
/// array, preserving the list structure.
///
/// This is the Vortex equivalent of DuckDB's `list_transform(l, lambda x: ...)`: offsets (or
/// the fixed size) and list-level validity pass through untouched, and only the elements child
/// is rewritten by applying the body expression with the elements array as its root scope.
/// Null lists stay null, empty lists stay empty, and null elements flow through the body's own
/// null semantics.
#[derive(Clone)]
pub struct ListTransform;

impl ScalarFnVTable for ListTransform {
    type Options = ListTransformOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.list.transform");
        *ID
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(options.body.serialize_proto()?.encode_to_vec()))
    }

    fn deserialize(&self, metadata: &[u8], session: &VortexSession) -> VortexResult<Self::Options> {
        let body = pb::Expr::decode(metadata)?;
        Ok(ListTransformOptions {
            body: Expression::from_proto(&body, session)?,
        })
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            _ => unreachable!("Invalid child index {child_idx} for list_transform()"),
        }
    }

    fn fmt_sql(
        &self,
        options: &Self::Options,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "{}(", self.id())?;
        expr.child(0).fmt_sql(f)?;
        write!(f, ", -> ")?;
        options.body.fmt_sql(f)?;
        write!(f, ")")
    }

    fn return_dtype(&self, options: &Self::Options, args: &[DType]) -> VortexResult<DType> {
        match &args[0] {
            DType::List(elem, nullable) => Ok(DType::List(
                options.body.return_dtype(elem)?.into(),
                *nullable,
            )),
            DType::FixedSizeList(elem, size, nullable) => Ok(DType::FixedSizeList(
                options.body.return_dtype(elem)?.into(),
                *size,
                *nullable,
            )),
            other => vortex_bail!("list_transform() requires List or FixedSizeList, got {other}"),
        }
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;

        if let Some(scalar) = input.as_constant() {
            let transformed = transform_scalar(&scalar, options, ctx)?;
            return Ok(ConstantArray::new(transformed, args.row_count()).into_array());
        }

        let any_list = input.execute_until::<AnyList>(ctx)?;
        transform_list(&any_list, &options.body)
    }

    fn simplify_untyped(
        &self,
        options: &Self::Options,
        expr: &Expression,
    ) -> VortexResult<Option<Expression>> {
        // Identity: list_transform(l, x -> x) == l.
        if options.body.is::<Root>() {
            return Ok(Some(expr.child(0).clone()));
        }

        // Fusion: list_transform(list_transform(l, f), g) == list_transform(l, g[root := f]).
        // Substitution only touches the outer body's children, so a nested transform inside `g`
        // keeps its own body (and root scope) intact while its input is still rewritten.
        let input = expr.child(0);
        if let Some(inner) = input.as_opt::<ListTransform>() {
            let fused = replace(options.body.clone(), &root(), inner.body.clone());
            return Ok(Some(list_transform(input.child(0).clone(), fused)));
        }

        Ok(None)
    }

    fn validity(
        &self,
        _options: &Self::Options,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        Ok(Some(expression.child(0).validity()?))
    }

    fn is_null_sensitive(&self, _options: &Self::Options) -> bool {
        false
    }

    fn is_fallible(&self, options: &Self::Options) -> bool {
        // This node evaluates the whole body, so its fallibility is the body's. The body lives
        // in the options rather than as a child, so generic tree walks cannot see it.
        label_is_fallible(&options.body)
            .get(&options.body)
            .copied()
            .unwrap_or(true)
    }
}

/// Transform each list of `any_list` (a `List`, `ListView`, or `FixedSizeList` array) by
/// applying `body` to its elements child, preserving the list structure.
///
/// The elements are wrapped in a deferred expression array, so no element values are computed
/// here. Note that this evaluates `body` over every position of the elements child, including
/// positions referenced only by null lists.
// TODO(#design §4.3): fallible bodies should not be evaluated over elements referenced only by
// null lists, which Arrow permits to hold arbitrary values.
fn transform_list(any_list: &ArrayRef, body: &Expression) -> VortexResult<ArrayRef> {
    if let Some(fsl) = any_list.as_opt::<FixedSizeList>() {
        let elements = fsl.elements().clone().apply(body)?;
        Ok(
            FixedSizeListArray::new(elements, fsl.list_size(), fsl.validity()?, fsl.len())
                .into_array(),
        )
    } else if let Some(lv) = any_list.as_opt::<ListView>() {
        let elements = lv.elements().clone().apply(body)?;
        Ok(ListViewArray::new(
            elements,
            lv.offsets().clone(),
            lv.sizes().clone(),
            lv.listview_validity(),
        )
        .into_array())
    } else if let Some(l) = any_list.as_opt::<List>() {
        let elements = l.elements().clone().apply(body)?;
        Ok(ListArray::try_new(elements, l.offsets().clone(), l.list_validity())?.into_array())
    } else {
        let dtype = any_list.dtype();
        vortex_bail!("list_transform() requires List, ListView, or FixedSizeList but got {dtype}")
    }
}

/// Transform a constant list scalar by transforming a single-row list array and extracting the
/// resulting scalar.
fn transform_scalar(
    scalar: &Scalar,
    options: &ListTransformOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Scalar> {
    let out_dtype = ListTransform.return_dtype(options, std::slice::from_ref(scalar.dtype()))?;
    if scalar.is_null() {
        return Ok(Scalar::null(out_dtype));
    }

    let one_row = ConstantArray::new(scalar.clone(), 1)
        .into_array()
        .execute_until::<AnyList>(ctx)?;
    transform_list(&one_row, &options.body)?.execute_scalar(0, ctx)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::BoolArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::FixedSizeListArray;
    use crate::arrays::ListArray;
    use crate::arrays::ListViewArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::checked_add;
    use crate::expr::get_item;
    use crate::expr::list_length;
    use crate::expr::list_transform;
    use crate::expr::lit;
    use crate::expr::pack;
    use crate::expr::root;
    use crate::validity::Validity;

    fn create_list_elements() -> ArrayRef {
        PrimitiveArray::from_option_iter::<i32, _>([
            Some(1),
            Some(2),
            Some(3),
            Some(4),
            Some(5),
            Some(6),
            None,
        ])
        .into_array()
    }

    fn incremented_elements() -> ArrayRef {
        PrimitiveArray::from_option_iter::<i32, _>([
            Some(2),
            Some(3),
            Some(4),
            Some(5),
            Some(6),
            Some(7),
            None,
        ])
        .into_array()
    }

    #[rstest]
    #[case(buffer![0u32, 2, 5, 5, 7].into_array())]
    #[case(buffer![0u64, 2, 5, 5, 7].into_array())]
    fn test_list_transform(#[case] offsets: ArrayRef) -> VortexResult<()> {
        let list = ListArray::try_new(
            create_list_elements(),
            offsets.clone(),
            Validity::NonNullable,
        )?
        .into_array();
        let result = list.apply(&list_transform(root(), checked_add(root(), lit(1))))?;

        let expected = ListArray::try_new(incremented_elements(), offsets, Validity::NonNullable)?
            .into_array();

        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_nullable_list_transform() -> VortexResult<()> {
        let validity =
            Validity::Array(BoolArray::from_iter([true, false, true, false]).into_array());
        let list = ListArray::try_new(
            create_list_elements(),
            buffer![0u32, 2, 5, 5, 7].into_array(),
            validity.clone(),
        )?
        .into_array();
        let result = list.apply(&list_transform(root(), checked_add(root(), lit(1))))?;

        let expected = ListArray::try_new(
            incremented_elements(),
            buffer![0u32, 2, 5, 5, 7].into_array(),
            validity,
        )?
        .into_array();

        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_listview_transform() -> VortexResult<()> {
        let lv = ListViewArray::new(
            create_list_elements(),
            buffer![5u32, 0, 4, 1].into_array(),
            buffer![2u32, 3, 0, 2].into_array(),
            Validity::NonNullable,
        )
        .into_array();
        let result = lv.apply(&list_transform(root(), checked_add(root(), lit(1))))?;

        let expected = ListViewArray::new(
            incremented_elements(),
            buffer![5u32, 0, 4, 1].into_array(),
            buffer![2u32, 3, 0, 2].into_array(),
            Validity::NonNullable,
        )
        .into_array();

        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_fixed_size_list_transform() -> VortexResult<()> {
        let elements = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5, 6, 7, 8]).into_array();
        let fsl = FixedSizeListArray::new(elements, 2, Validity::NonNullable, 4).into_array();
        let result = fsl.apply(&list_transform(root(), checked_add(root(), lit(1))))?;

        // The fixed-size structure is preserved.
        assert!(matches!(result.dtype(), DType::FixedSizeList(_, 2, _)));

        let expected_elements = PrimitiveArray::from_iter([2i32, 3, 4, 5, 6, 7, 8, 9]).into_array();
        let expected =
            FixedSizeListArray::new(expected_elements, 2, Validity::NonNullable, 4).into_array();

        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_constant_list_transform() -> VortexResult<()> {
        let list = ListArray::try_new(
            PrimitiveArray::from_iter([1i32, 2, 3]).into_array(),
            buffer![0u32, 3].into_array(),
            Validity::NonNullable,
        )?
        .into_array();

        let mut ctx = array_session().create_execution_ctx();
        let scalar = list.execute_scalar(0, &mut ctx)?;
        let constant = ConstantArray::new(scalar, 4).into_array();

        let result = constant.apply(&list_transform(root(), checked_add(root(), lit(1))))?;

        let expected = ListArray::try_new(
            PrimitiveArray::from_iter([2i32, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4]).into_array(),
            buffer![0u32, 3, 6, 9, 12].into_array(),
            Validity::NonNullable,
        )?
        .into_array();

        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_null_constant_list_transform() -> VortexResult<()> {
        let null_scalar = crate::scalar::Scalar::null(DType::List(
            std::sync::Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            Nullability::Nullable,
        ));
        let constant = ConstantArray::new(null_scalar, 2).into_array();
        let result = constant.apply(&list_transform(root(), checked_add(root(), lit(1))))?;

        let mut ctx = array_session().create_execution_ctx();
        assert!(!result.is_valid(0, &mut ctx)?);
        assert!(!result.is_valid(1, &mut ctx)?);
        Ok(())
    }

    #[test]
    fn test_list_transform_take() -> VortexResult<()> {
        let list = ListArray::try_new(
            create_list_elements(),
            buffer![0u32, 2, 5, 5, 7].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let taken = list.take(buffer![3u64, 0, 2].into_array())?;

        let result = taken.apply(&list_transform(root(), checked_add(root(), lit(1))))?;

        // Transform-then-take equals take-then-transform.
        let expected = list
            .apply(&list_transform(root(), checked_add(root(), lit(1))))?
            .take(buffer![3u64, 0, 2].into_array())?;

        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_nested_list_transform() -> VortexResult<()> {
        // [[1, 2], [3]], [], [[4, 5]]
        let inner = ListArray::try_new(
            PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]).into_array(),
            buffer![0u32, 2, 3, 5].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let outer = ListArray::try_new(
            inner,
            buffer![0u32, 2, 2, 3].into_array(),
            Validity::NonNullable,
        )?
        .into_array();

        // Increment every innermost element: the outer body's root is the inner list, and the
        // inner body's root is the innermost element.
        let expr = list_transform(root(), list_transform(root(), checked_add(root(), lit(1))));
        let result = outer.apply(&expr)?;

        let expected_inner = ListArray::try_new(
            PrimitiveArray::from_iter([2i32, 3, 4, 5, 6]).into_array(),
            buffer![0u32, 2, 3, 5].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let expected = ListArray::try_new(
            expected_inner,
            buffer![0u32, 2, 2, 3].into_array(),
            Validity::NonNullable,
        )?
        .into_array();

        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_struct_element_transform() -> VortexResult<()> {
        // Transform a list of structs into a list of one of its fields: x -> x.a
        let a = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
        let b = PrimitiveArray::from_iter([10i32, 20, 30]).into_array();
        let elements = crate::arrays::StructArray::try_new(
            ["a", "b"].into(),
            vec![a.clone(), b],
            3,
            Validity::NonNullable,
        )?
        .into_array();
        let list = ListArray::try_new(
            elements,
            buffer![0u32, 1, 3].into_array(),
            Validity::NonNullable,
        )?
        .into_array();

        let result = list.apply(&list_transform(root(), get_item("a", root())))?;

        let expected =
            ListArray::try_new(a, buffer![0u32, 1, 3].into_array(), Validity::NonNullable)?
                .into_array();

        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_identity_simplification() -> VortexResult<()> {
        let input = get_item("tags", root());
        let expr = list_transform(input.clone(), root());
        assert_eq!(expr.scalar_fn().simplify_untyped(&expr)?, Some(input));
        Ok(())
    }

    #[test]
    fn test_fusion_simplification() -> VortexResult<()> {
        let input = get_item("tags", root());
        let f = checked_add(root(), lit(1));
        let g = checked_add(root(), lit(2));
        let expr = list_transform(list_transform(input.clone(), f.clone()), g);

        let expected = list_transform(input, checked_add(f, lit(2)));
        assert_eq!(expr.scalar_fn().simplify_untyped(&expr)?, Some(expected));
        Ok(())
    }

    #[test]
    fn test_fusion_preserves_nested_scope() -> VortexResult<()> {
        // The outer body contains a nested transform: its input (a child) is in the outer
        // scope and must be substituted, while its own body must be left untouched.
        let f = get_item("a", root());
        let g = list_transform(root(), checked_add(root(), lit(1)));
        let expr = list_transform(list_transform(get_item("tags", root()), f.clone()), g);

        let expected = list_transform(
            get_item("tags", root()),
            list_transform(f, checked_add(root(), lit(1))),
        );
        assert_eq!(expr.scalar_fn().simplify_untyped(&expr)?, Some(expected));
        Ok(())
    }

    #[test]
    fn test_list_length_of_transform_simplification() -> VortexResult<()> {
        let input = get_item("tags", root());
        let expr = list_length(list_transform(input.clone(), checked_add(root(), lit(1))));
        assert_eq!(
            expr.scalar_fn().simplify_untyped(&expr)?,
            Some(list_length(input))
        );
        Ok(())
    }

    #[test]
    fn test_serde_roundtrip() -> VortexResult<()> {
        use prost::Message;

        use crate::expr::proto::ExprSerializeProtoExt;

        let expr = list_transform(
            get_item("tags", root()),
            checked_add(get_item("a", root()), lit(1)),
        );

        let buf = expr.serialize_proto()?.encode_to_vec();
        let decoded = vortex_proto::expr::Expr::decode(buf.as_slice())?;
        let roundtrip = crate::expr::Expression::from_proto(&decoded, &array_session())?;

        assert_eq!(roundtrip, expr);
        Ok(())
    }

    #[test]
    fn test_return_dtype() -> VortexResult<()> {
        let expr = list_transform(root(), pack([("x", root())], Nullability::NonNullable));
        let scope = DType::List(
            std::sync::Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            Nullability::Nullable,
        );
        let dtype = expr.return_dtype(&scope)?;

        // List nullability passes through; the element dtype is the body's return dtype.
        let DType::List(elem, Nullability::Nullable) = dtype else {
            panic!("expected nullable list dtype");
        };
        assert!(matches!(elem.as_ref(), DType::Struct(..)));
        Ok(())
    }

    #[test]
    fn test_display() {
        let body = checked_add(root(), lit(1));
        let expr = list_transform(root(), body.clone());
        assert_eq!(
            expr.to_string(),
            format!("vortex.list.transform($, -> {body})")
        );
    }
}
