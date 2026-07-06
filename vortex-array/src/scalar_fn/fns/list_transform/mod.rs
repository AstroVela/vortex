// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

use prost::Message;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_proto::expr as pb;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::FixedSizeList;
use crate::arrays::FixedSizeListArray;
use crate::arrays::List;
use crate::arrays::ListArray;
use crate::arrays::ListView;
use crate::arrays::ListViewArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::fixed_size_list::FixedSizeListArrayExt;
use crate::arrays::list::ListArrayExt;
use crate::arrays::listview::ListViewArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::expr::Expression;
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
use crate::scalar_fn::SimplifyCtx;
use crate::scalar_fn::fns::list_length::AnyList;
use crate::scalar_fn::fns::root::Root;
use crate::validity::Validity;

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
        Ok(options
            .body
            .try_serialize_proto()?
            .map(|body| body.encode_to_vec()))
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
        write!(f, ", {})", options)
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

        // A zero-row batch must not evaluate the body: with a constant input, the eager scalar
        // path below could error on element values no row observes.
        if args.row_count() == 0 {
            let out_dtype = self.return_dtype(options, std::slice::from_ref(input.dtype()))?;
            return Ok(Canonical::empty(&out_dtype).into_array());
        }

        if let Some(scalar) = input.as_constant() {
            let transformed = transform_scalar(&scalar, options, ctx)?;
            return Ok(ConstantArray::new(transformed, args.row_count()).into_array());
        }

        let any_list = input.execute_until::<AnyList>(ctx)?;
        transform_list(&any_list, &options.body, ctx)
    }

    fn simplify_untyped(
        &self,
        options: &Self::Options,
        expr: &Expression,
    ) -> VortexResult<Option<Expression>> {
        // Fusion: list_transform(list_transform(l, f), g) == list_transform(l, g[root := f]).
        // Substitution only touches the outer body's children, so a nested transform inside `g`
        // keeps its own body (and root scope) intact while its input is still rewritten.
        //
        // The whole transform chain is collapsed in a single rewrite so that deep chains do not
        // exhaust the optimizer's per-node iteration budget, and only bodies with at most one
        // root reference are fused, since each substitution duplicates the inner body once per
        // root occurrence (multiplying across fusion steps).
        let mut input = expr.child(0);
        let mut body = options.body.clone();
        let mut fused = false;
        while let Some(inner) = input.as_opt::<ListTransform>() {
            if count_root_refs(&body) > 1 {
                break;
            }
            body = replace(body, &root(), inner.body.clone());
            input = input.child(0);
            fused = true;
        }

        Ok(fused.then(|| list_transform(input.clone(), body)))
    }

    fn simplify(
        &self,
        options: &Self::Options,
        expr: &Expression,
        ctx: &dyn SimplifyCtx,
    ) -> VortexResult<Option<Expression>> {
        // If the input is not list-typed, leave the node intact so the type error still
        // surfaces through return_dtype rather than being rewritten away.
        let input_dtype = ctx.return_dtype(expr.child(0))?;
        let element_dtype = match &input_dtype {
            DType::List(elem, _) | DType::FixedSizeList(elem, ..) => elem.as_ref().clone(),
            _ => return Ok(None),
        };

        // Identity: list_transform(l, x -> x) == l.
        if options.body.is::<Root>() {
            return Ok(Some(expr.child(0).clone()));
        }

        // The body is invisible to the optimizer's children traversal, so optimize it here
        // against its own scope: the element dtype.
        let optimized = options.body.optimize_recursive(&element_dtype)?;
        if optimized != options.body {
            return Ok(Some(list_transform(expr.child(0).clone(), optimized)));
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
        expr_is_fallible(&options.body)
    }
}

/// Whether any node of the expression is fallible. A nested `list_transform` inside the tree
/// reports its own body's fallibility through its signature.
fn expr_is_fallible(expr: &Expression) -> bool {
    expr.signature().is_fallible() || expr.children().iter().any(expr_is_fallible)
}

/// The number of `root()` references in the expression. Does not descend into options-embedded
/// bodies of nested `list_transform` nodes, whose root refers to their own scope.
fn count_root_refs(expr: &Expression) -> usize {
    if expr.is::<Root>() {
        return 1;
    }
    expr.children().iter().map(count_root_refs).sum()
}

/// Transform each list of `any_list` (a `List`, `ListView`, or `FixedSizeList` array) by
/// applying `body` to its elements child, preserving the list structure.
///
/// For infallible bodies the elements are wrapped in a deferred expression array, so no element
/// values are computed here. A fallible body must only be evaluated over elements some visible
/// list references — ListView gaps, and ranges under null lists (which Arrow permits to hold
/// arbitrary values), could otherwise raise errors from data that is not logically part of the
/// array — so that path first gathers the referenced elements into a compact `List`.
///
/// A fallible body over a `FixedSizeList` with null lists still evaluates the null rows' element
/// strides: the fixed-size structure cannot drop them. Canonical null FSL rows hold default
/// values rather than garbage, so this only matters for externally-created arrays.
fn transform_list(
    any_list: &ArrayRef,
    body: &Expression,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    if let Some(fsl) = any_list.as_opt::<FixedSizeList>() {
        let elements = fsl.elements().clone().apply(body)?;
        Ok(
            FixedSizeListArray::new(elements, fsl.list_size(), fsl.validity()?, fsl.len())
                .into_array(),
        )
    } else if let Some(lv) = any_list.as_opt::<ListView>() {
        if expr_is_fallible(body) {
            let validity = lv.listview_validity();
            let (elements, offsets) = gather_referenced_elements(
                lv.elements(),
                lv.offsets(),
                Some(lv.sizes()),
                &validity,
                lv.len(),
                ctx,
            )?;
            return Ok(ListArray::try_new(elements.apply(body)?, offsets, validity)?.into_array());
        }
        let elements = lv.elements().clone().apply(body)?;
        Ok(ListViewArray::new(
            elements,
            lv.offsets().clone(),
            lv.sizes().clone(),
            lv.listview_validity(),
        )
        .into_array())
    } else if let Some(l) = any_list.as_opt::<List>() {
        let validity = l.list_validity();
        // A canonical List's offsets tile the referenced range contiguously, so unreferenced
        // positions exist only under null lists.
        if expr_is_fallible(body) && !matches!(validity, Validity::NonNullable | Validity::AllValid)
        {
            let (elements, offsets) = gather_referenced_elements(
                l.elements(),
                l.offsets(),
                None,
                &validity,
                l.len(),
                ctx,
            )?;
            return Ok(ListArray::try_new(elements.apply(body)?, offsets, validity)?.into_array());
        }
        let elements = l.elements().clone().apply(body)?;
        Ok(ListArray::try_new(elements, l.offsets().clone(), l.list_validity())?.into_array())
    } else {
        let dtype = any_list.dtype();
        vortex_bail!("list_transform() requires List, ListView, or FixedSizeList but got {dtype}")
    }
}

/// Gather exactly the elements referenced by valid lists into a compact elements array with
/// dense offsets, so a fallible body never observes unreferenced positions.
///
/// `sizes` is `Some` for ListView semantics (`elements[offsets[i]..offsets[i] + sizes[i]]`) and
/// `None` for List semantics (`elements[offsets[i]..offsets[i + 1]]`). Null lists contribute
/// zero-length ranges; the caller reattaches the original validity.
fn gather_referenced_elements(
    elements: &ArrayRef,
    offsets: &ArrayRef,
    sizes: Option<&ArrayRef>,
    validity: &Validity,
    len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(ArrayRef, ArrayRef)> {
    let u64_dtype = DType::Primitive(PType::U64, Nullability::NonNullable);
    let offsets = offsets
        .cast(u64_dtype.clone())?
        .execute::<PrimitiveArray>(ctx)?;
    let offsets = offsets.as_slice::<u64>();
    let sizes = sizes
        .map(|sizes| sizes.cast(u64_dtype)?.execute::<PrimitiveArray>(ctx))
        .transpose()?;
    let sizes = sizes.as_ref().map(|sizes| sizes.as_slice::<u64>());

    let mask = validity.execute_mask(len, ctx)?;

    let mut indices = BufferMut::<u64>::empty();
    let mut new_offsets = BufferMut::<u64>::with_capacity(len + 1);
    new_offsets.push(0);
    for row in 0..len {
        if mask.value(row) {
            let start = offsets[row];
            let end = match sizes {
                Some(sizes) => start + sizes[row],
                None => offsets[row + 1],
            };
            indices.extend(start..end);
        }
        new_offsets.push(indices.len() as u64);
    }

    let gathered = elements.take(indices.freeze().into_array())?;
    Ok((gathered, new_offsets.freeze().into_array()))
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
    transform_list(&one_row, &options.body, ctx)?.execute_scalar(0, ctx)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use prost::Message;
    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;
    use vortex_proto::expr as pb;

    use super::ListTransform;
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
    use crate::arrays::StructArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::Expression;
    use crate::expr::and;
    use crate::expr::cast;
    use crate::expr::checked_add;
    use crate::expr::get_item;
    use crate::expr::list_length;
    use crate::expr::list_transform;
    use crate::expr::lit;
    use crate::expr::pack;
    use crate::expr::proto::ExprSerializeProtoExt;
    use crate::expr::root;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    fn i32_list_dtype() -> DType {
        DType::List(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            Nullability::NonNullable,
        )
    }

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
        let null_scalar = Scalar::null(DType::List(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
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
        let elements = StructArray::try_new(
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
        let expr = list_transform(root(), root());
        assert_eq!(expr.optimize_recursive(&i32_list_dtype())?, root());
        Ok(())
    }

    #[test]
    fn test_identity_preserves_type_error() -> VortexResult<()> {
        // The identity rewrite is type-gated: an ill-typed non-list input must keep failing
        // return_dtype instead of being rewritten into a well-typed non-list expression.
        let expr = list_transform(lit(5), root());
        let scope = DType::Primitive(PType::I32, Nullability::NonNullable);
        let optimized = expr.optimize_recursive(&scope)?;
        assert!(optimized.return_dtype(&scope).is_err());
        Ok(())
    }

    #[test]
    fn test_body_is_optimized() -> VortexResult<()> {
        // The body simplifies to root() against its element scope, after which the identity
        // rule collapses the whole transform.
        let expr = list_transform(root(), and(root(), lit(true)));
        let scope = DType::List(
            Arc::new(DType::Bool(Nullability::NonNullable)),
            Nullability::NonNullable,
        );
        assert_eq!(expr.optimize_recursive(&scope)?, root());
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
        let expr = list_transform(
            get_item("tags", root()),
            checked_add(get_item("a", root()), lit(1)),
        );

        let buf = expr.serialize_proto()?.encode_to_vec();
        let decoded = pb::Expr::decode(buf.as_slice())?;
        let roundtrip = Expression::from_proto(&decoded, &array_session())?;

        assert_eq!(roundtrip, expr);
        Ok(())
    }

    #[test]
    fn test_return_dtype() -> VortexResult<()> {
        let expr = list_transform(root(), pack([("x", root())], Nullability::NonNullable));
        let scope = DType::List(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            Nullability::Nullable,
        );
        let dtype = expr.return_dtype(&scope)?;

        // List nullability passes through; the element dtype is the body's return dtype.
        let DType::List(elem, Nullability::Nullable) = &dtype else {
            vortex_bail!("expected nullable list dtype, got {dtype}");
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

    #[test]
    fn test_fallible_body_skips_listview_gaps() -> VortexResult<()> {
        // Visible data is [[1], [3]]; the gap element (i32::MAX) is referenced by no view and
        // must not be evaluated by the fallible body.
        let lv = ListViewArray::new(
            PrimitiveArray::from_iter([1i32, i32::MAX, 3]).into_array(),
            buffer![0u32, 2].into_array(),
            buffer![1u32, 1].into_array(),
            Validity::NonNullable,
        )
        .into_array();
        let result = lv.apply(&list_transform(root(), checked_add(root(), lit(1))))?;

        let expected = ListArray::try_new(
            PrimitiveArray::from_iter([2i32, 4]).into_array(),
            buffer![0u64, 1, 2].into_array(),
            Validity::NonNullable,
        )?
        .into_array();

        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_fallible_body_skips_null_list_garbage() -> VortexResult<()> {
        // Arrow permits a null list to reference arbitrary element values; the fallible body
        // must not error on them.
        let list = ListArray::try_new(
            PrimitiveArray::from_iter([1i32, i32::MAX]).into_array(),
            buffer![0u32, 1, 2].into_array(),
            Validity::Array(BoolArray::from_iter([true, false]).into_array()),
        )?
        .into_array();
        let result = list.apply(&list_transform(root(), checked_add(root(), lit(1))))?;

        let expected = ListArray::try_new(
            PrimitiveArray::from_iter([2i32]).into_array(),
            buffer![0u64, 1, 1].into_array(),
            Validity::Array(BoolArray::from_iter([true, false]).into_array()),
        )?
        .into_array();

        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_zero_row_constant_fallible_body() -> VortexResult<()> {
        let list = ListArray::try_new(
            PrimitiveArray::from_iter([i32::MAX]).into_array(),
            buffer![0u32, 1].into_array(),
            Validity::NonNullable,
        )?
        .into_array();

        let mut ctx = array_session().create_execution_ctx();
        let scalar = list.execute_scalar(0, &mut ctx)?;
        let constant = ConstantArray::new(scalar, 0).into_array();

        // A zero-row batch must not evaluate the body over the constant's elements.
        let result = constant.apply(&list_transform(root(), checked_add(root(), lit(1))))?;
        let result = result.execute::<ArrayRef>(&mut ctx)?;
        assert_eq!(result.len(), 0);
        Ok(())
    }

    #[test]
    fn test_null_fsl_constant_fallible_body() -> VortexResult<()> {
        // A null FixedSizeList constant canonicalizes to default-filled placeholder elements;
        // a fallible body must not run over them.
        let null_scalar = Scalar::null(DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
            2,
            Nullability::Nullable,
        ));
        let constant = ConstantArray::new(null_scalar, 3).into_array();

        let body = cast(
            root(),
            DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        let result = constant.apply(&list_transform(root(), body))?;

        let mut ctx = array_session().create_execution_ctx();
        assert!(!result.is_valid(0, &mut ctx)?);
        assert!(!result.is_valid(2, &mut ctx)?);
        Ok(())
    }

    #[test]
    fn test_fusion_skips_multi_root_bodies() -> VortexResult<()> {
        // Fusing a body with multiple root references would duplicate the inner body per
        // occurrence, so it is skipped.
        let inner = list_transform(get_item("tags", root()), checked_add(root(), lit(1)));
        let expr = list_transform(inner, checked_add(root(), root()));
        assert_eq!(expr.scalar_fn().simplify_untyped(&expr)?, None);
        Ok(())
    }

    #[test]
    fn test_deep_fusion_chain() -> VortexResult<()> {
        // The whole chain must collapse in a single rewrite rather than consuming one optimizer
        // iteration per level (the per-node budget is 100).
        let mut expr = root();
        for _ in 0..150 {
            expr = list_transform(expr, checked_add(root(), lit(1)));
        }

        let optimized = expr.optimize_recursive(&i32_list_dtype())?;
        assert!(optimized.is::<ListTransform>());
        assert_eq!(optimized.child(0), &root());
        Ok(())
    }

    #[test]
    fn test_is_fallible_delegates_to_body() {
        let fallible = list_transform(root(), checked_add(root(), lit(1)));
        assert!(fallible.signature().is_fallible());

        let infallible = list_transform(root(), get_item("a", root()));
        assert!(!infallible.signature().is_fallible());
    }
}
