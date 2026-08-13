// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::Canonical;
use crate::IntoArray;
use crate::arrays::FixedSizeList;
use crate::arrays::FixedSizeListArray;
use crate::arrays::List;
use crate::arrays::ListArray;
use crate::arrays::ListView;
use crate::arrays::ListViewArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::fixed_size_list::FixedSizeListArrayExt;
use crate::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use crate::arrays::list::ListArrayExt;
use crate::arrays::list::ListArraySlotsExt;
use crate::arrays::listview::ListViewArrayExt;
use crate::arrays::listview::ListViewArraySlotsExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::expr::Frame;
use crate::expr::Lambda;
use crate::expr::Scope;
use crate::expr::TypedLambda;
use crate::higher_order_fn::HigherOrderFunctionId;
use crate::higher_order_fn::HigherOrderFunctionVTable;
use crate::higher_order_fn::LambdaCall;
use crate::matcher::Matcher;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::validity::Validity;

/// Applies a single-argument lambda to every element of a list.
///
/// The input's list structure and list-level validity are retained while the lambda transforms
/// the elements array. Nested invocation rebases captured outer parameters from list rows to
/// element rows before evaluating the lambda body.
#[derive(Clone, Debug)]
pub struct ListTransform;

impl HigherOrderFunctionVTable for ListTransform {
    fn id(&self) -> HigherOrderFunctionId {
        static ID: CachedId = CachedId::new("vortex.list.transform");
        *ID
    }

    fn serialize(&self) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(&self, metadata: &[u8], _session: &VortexSession) -> VortexResult<()> {
        if metadata.is_empty() {
            Ok(())
        } else {
            vortex_bail!("list_transform() does not accept metadata")
        }
    }

    fn arity(&self) -> Arity {
        Arity::Exact(1)
    }

    fn lambda_arity(&self) -> usize {
        1
    }

    fn child_name(&self, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            _ => unreachable!("invalid child index {child_idx} for list_transform()"),
        }
    }

    fn bind_lambdas(
        &self,
        scope: &Scope,
        args: &[DType],
        lambdas: &[Lambda],
    ) -> VortexResult<Box<[TypedLambda]>> {
        let [lambda] = lambdas else {
            vortex_bail!("list_transform() requires exactly one lambda");
        };
        vortex_ensure!(
            lambda.params().len() == 1,
            "list_transform() lambda must take exactly one parameter, got {}",
            lambda.params().len()
        );
        let element = match &args[0] {
            DType::List(element, _) | DType::FixedSizeList(element, ..) => element,
            dtype => vortex_bail!("list_transform() requires List or FixedSizeList, got {dtype}"),
        };
        Ok(Box::new([bind_lambda(lambda, scope, element)?]))
    }

    fn return_dtype(&self, args: &[DType], lambdas: &[TypedLambda]) -> VortexResult<DType> {
        let [lambda] = lambdas else {
            vortex_bail!("list_transform() requires exactly one lambda");
        };
        match &args[0] {
            DType::List(_, nullability) => Ok(DType::List(
                lambda.body_dtype().clone().into(),
                *nullability,
            )),
            DType::FixedSizeList(_, size, nullability) => Ok(DType::FixedSizeList(
                lambda.body_dtype().clone().into(),
                *size,
                *nullability,
            )),
            dtype => vortex_bail!("list_transform() requires List or FixedSizeList, got {dtype}"),
        }
    }

    fn validity(&self, args: &[ArrayRef], _lambdas: &[TypedLambda]) -> VortexResult<Validity> {
        let [input] = args else {
            vortex_bail!("list_transform() requires exactly one input");
        };
        input.validity()
    }

    fn execute(
        &self,
        args: &[ArrayRef],
        lambdas: &[LambdaCall<'_>],
        ctx: &mut crate::ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let [input] = args else {
            vortex_bail!("list_transform() requires exactly one input");
        };
        let [lambda] = lambdas else {
            vortex_bail!("list_transform() requires exactly one lambda");
        };
        let lambda_definitions = [lambda.lambda().clone()];
        let output_dtype =
            self.return_dtype(std::slice::from_ref(input.dtype()), &lambda_definitions)?;
        if input.is_empty() {
            return Ok(Canonical::empty(&output_dtype).into_array());
        }

        let input = input.clone().execute_until::<AnyList>(ctx)?;
        transform_list(&input, lambda, lambda_is_fallible(lambda.lambda()), ctx)
    }
}

fn bind_lambda(
    lambda: &Lambda,
    parent_scope: &Scope,
    element_dtype: &DType,
) -> VortexResult<TypedLambda> {
    let frame = Frame::try_new(
        lambda
            .params()
            .iter()
            .cloned()
            .map(|parameter| (parameter, element_dtype.clone())),
    )?;
    let lambda_scope = parent_scope
        .with_root(element_dtype.clone())
        .push_frame(frame);
    TypedLambda::bind(lambda, &lambda_scope)
}

fn transform_list(
    input: &ArrayRef,
    lambda: &LambdaCall<'_>,
    lambda_is_fallible: bool,
    ctx: &mut crate::ExecutionCtx,
) -> VortexResult<ArrayRef> {
    if let Some(fixed_size_list) = input.as_opt::<FixedSizeList>() {
        let parent_indices = lambda.has_captures().then(|| {
            fixed_size_list_parent_indices(
                fixed_size_list.list_size() as usize,
                fixed_size_list.len(),
            )
        });
        let elements = fixed_size_list.elements();
        return Ok(FixedSizeListArray::new(
            lambda.apply(
                elements.clone(),
                std::slice::from_ref(elements),
                parent_indices.as_ref(),
            )?,
            fixed_size_list.list_size(),
            fixed_size_list.fixed_size_list_validity(),
            fixed_size_list.len(),
        )
        .into_array());
    }

    if let Some(list) = input.as_opt::<List>() {
        let validity = list.list_validity();
        if lambda.has_captures()
            || (lambda_is_fallible
                && !matches!(validity, Validity::NonNullable | Validity::AllValid))
        {
            let (elements, offsets, parent_indices) = gather_referenced_elements(
                list.elements(),
                list.offsets(),
                None,
                &validity,
                list.len(),
                ctx,
            )?;
            return Ok(ListArray::try_new(
                lambda.apply(
                    elements.clone(),
                    std::slice::from_ref(&elements),
                    lambda.has_captures().then_some(&parent_indices),
                )?,
                offsets,
                validity,
            )?
            .into_array());
        }

        return Ok(ListArray::try_new(
            lambda.apply(
                list.elements().clone(),
                std::slice::from_ref(list.elements()),
                None,
            )?,
            list.offsets().clone(),
            validity,
        )?
        .into_array());
    }

    if let Some(list_view) = input.as_opt::<ListView>() {
        let validity = list_view.listview_validity();
        if lambda_is_fallible || lambda.has_captures() {
            let (elements, offsets, parent_indices) = gather_referenced_elements(
                list_view.elements(),
                list_view.offsets(),
                Some(list_view.sizes()),
                &validity,
                list_view.len(),
                ctx,
            )?;
            return Ok(ListArray::try_new(
                lambda.apply(
                    elements.clone(),
                    std::slice::from_ref(&elements),
                    lambda.has_captures().then_some(&parent_indices),
                )?,
                offsets,
                validity,
            )?
            .into_array());
        }

        return Ok(ListViewArray::new(
            lambda.apply(
                list_view.elements().clone(),
                std::slice::from_ref(list_view.elements()),
                None,
            )?,
            list_view.offsets().clone(),
            list_view.sizes().clone(),
            validity,
        )
        .into_array());
    }

    let dtype = input.dtype();
    vortex_bail!("list_transform() requires List, ListView, or FixedSizeList, got {dtype}")
}

fn lambda_is_fallible(lambda: &TypedLambda) -> bool {
    match lambda.body() {
        crate::expr::BoundExpression::Scalar {
            scalar_fn,
            children,
            ..
        } => scalar_fn.signature().is_fallible() || children.iter().any(bound_is_fallible),
        crate::expr::BoundExpression::HigherOrder { .. } => true,
        crate::expr::BoundExpression::Root { .. }
        | crate::expr::BoundExpression::Variable { .. } => false,
    }
}

fn bound_is_fallible(expr: &crate::expr::BoundExpression) -> bool {
    match expr {
        crate::expr::BoundExpression::Scalar {
            scalar_fn,
            children,
            ..
        } => scalar_fn.signature().is_fallible() || children.iter().any(bound_is_fallible),
        crate::expr::BoundExpression::HigherOrder { .. } => true,
        crate::expr::BoundExpression::Root { .. }
        | crate::expr::BoundExpression::Variable { .. } => false,
    }
}

/// Gather elements referenced by valid lists into a dense `List` layout.
///
/// A fallible lambda must only see elements in valid list rows. `ListView` can also leave gaps in
/// its element array, so its ranges are gathered even when every list row is valid.
fn gather_referenced_elements(
    elements: &ArrayRef,
    offsets: &ArrayRef,
    sizes: Option<&ArrayRef>,
    validity: &Validity,
    len: usize,
    ctx: &mut crate::ExecutionCtx,
) -> VortexResult<(ArrayRef, ArrayRef, ArrayRef)> {
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
    let mut parent_indices = BufferMut::<u64>::empty();
    let mut new_offsets = BufferMut::<u64>::with_capacity(len + 1);
    new_offsets.push(0);
    for row in 0..len {
        if mask.value(row) {
            let start = offsets[row];
            let end = match sizes {
                Some(sizes) => start + sizes[row],
                None => offsets[row + 1],
            };
            for index in start..end {
                indices.push(index);
                parent_indices.push(row as u64);
            }
        }
        new_offsets.push(indices.len() as u64);
    }

    Ok((
        elements.take(indices.freeze().into_array())?,
        new_offsets.freeze().into_array(),
        parent_indices.freeze().into_array(),
    ))
}

fn fixed_size_list_parent_indices(list_size: usize, len: usize) -> ArrayRef {
    let mut indices = BufferMut::<u64>::with_capacity(list_size * len);
    for row in 0..len {
        indices.extend(std::iter::repeat_n(row as u64, list_size));
    }
    indices.freeze().into_array()
}

/// Matches a concrete variable- or fixed-size list encoding.
struct AnyList;

impl Matcher for AnyList {
    type Match<'a> = ();

    fn try_match(array: &ArrayRef) -> Option<Self::Match<'_>> {
        (array.as_opt::<List>().is_some()
            || array.as_opt::<ListView>().is_some()
            || array.as_opt::<FixedSizeList>().is_some())
        .then_some(())
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::*;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::BoolArray;
    use crate::arrays::HigherOrderFn;
    use crate::arrays::ListArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::expr::Expression;
    use crate::expr::cast;
    use crate::expr::checked_add;
    use crate::expr::lambda as lambda_expr;
    use crate::expr::list_length;
    use crate::expr::list_transform;
    use crate::expr::lit;
    use crate::expr::proto::ExprSerializeProtoExt;
    use crate::expr::root;
    use crate::expr::var;
    use crate::validity::Validity;

    #[test]
    fn transforms_elements_with_the_lambda_parameter() -> VortexResult<()> {
        let input = ListArray::try_new(
            PrimitiveArray::from_iter([1_i32, 2, 3]).into_array(),
            buffer![0_u32, 2, 3].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let expression = list_transform(
            root(),
            lambda_expr(["element"], checked_add(var("element"), lit(1_i32)))?,
        )?;

        let result = input.apply(&expression)?;
        assert!(result.is::<HigherOrderFn>());
        let expected = ListArray::try_new(
            PrimitiveArray::from_iter([2_i32, 3, 4]).into_array(),
            buffer![0_u32, 2, 3].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn rejects_a_lambda_with_the_wrong_arity() -> VortexResult<()> {
        let expression = list_transform(root(), lambda_expr(["x", "i"], var("x"))?)?;
        let dtype = DType::List(
            std::sync::Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            Nullability::NonNullable,
        );
        assert!(expression.return_dtype(&dtype).is_err());
        Ok(())
    }

    #[test]
    fn nested_transforms_capture_outer_parameters() -> VortexResult<()> {
        let inner_lists = ListArray::try_new(
            PrimitiveArray::from_iter([1_i32, 2, 3, 4]).into_array(),
            buffer![0_u32, 2, 3, 4].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let input = ListArray::try_new(
            inner_lists,
            buffer![0_u32, 2, 3].into_array(),
            Validity::NonNullable,
        )?
        .into_array();

        let inner = list_transform(
            var("outer"),
            lambda_expr(
                ["inner"],
                checked_add(
                    var("inner"),
                    cast(
                        list_length(var("outer")),
                        DType::Primitive(PType::I32, Nullability::NonNullable),
                    ),
                ),
            )?,
        )?;
        let expression = list_transform(root(), lambda_expr(["outer"], inner)?)?;

        let result = input.apply(&expression)?;
        let expected_inner_lists = ListArray::try_new(
            PrimitiveArray::from_iter([3_i32, 4, 4, 5]).into_array(),
            buffer![0_u32, 2, 3, 4].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let expected = ListArray::try_new(
            expected_inner_lists,
            buffer![0_u32, 2, 3].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn inner_parameters_shadow_outer_parameters() -> VortexResult<()> {
        let inner_lists = ListArray::try_new(
            PrimitiveArray::from_iter([1_i32, 2, 3, 4]).into_array(),
            buffer![0_u32, 2, 3, 4].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let input = ListArray::try_new(
            inner_lists,
            buffer![0_u32, 2, 3].into_array(),
            Validity::NonNullable,
        )?
        .into_array();

        let inner = list_transform(
            var("value"),
            lambda_expr(["value"], checked_add(var("value"), lit(1_i32)))?,
        )?;
        let expression = list_transform(root(), lambda_expr(["value"], inner)?)?;

        let result = input.apply(&expression)?;
        let expected_inner_lists = ListArray::try_new(
            PrimitiveArray::from_iter([2_i32, 3, 4, 5]).into_array(),
            buffer![0_u32, 2, 3, 4].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let expected = ListArray::try_new(
            expected_inner_lists,
            buffer![0_u32, 2, 3].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn fallible_lambdas_skip_elements_of_null_lists() -> VortexResult<()> {
        let input = ListArray::try_new(
            PrimitiveArray::from_iter([1_i32, i32::MAX]).into_array(),
            buffer![0_u32, 1, 2].into_array(),
            Validity::Array(BoolArray::from_iter([true, false]).into_array()),
        )?
        .into_array();
        let expression = list_transform(
            root(),
            lambda_expr(["element"], checked_add(var("element"), lit(1_i32)))?,
        )?;

        let result = input.apply(&expression)?;
        let expected = ListArray::try_new(
            PrimitiveArray::from_iter([2_i32]).into_array(),
            buffer![0_u32, 1, 1].into_array(),
            Validity::Array(BoolArray::from_iter([true, false]).into_array()),
        )?
        .into_array();
        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn higher_order_expression_round_trips() -> VortexResult<()> {
        let expression = list_transform(
            root(),
            lambda_expr(["element"], checked_add(var("element"), lit(1_i32)))?,
        )?;
        let session = array_session();

        let round_tripped = Expression::from_proto(&expression.serialize_proto()?, &session)?;
        assert_eq!(round_tripped, expression);
        Ok(())
    }

    #[test]
    fn nested_higher_order_expression_round_trips() -> VortexResult<()> {
        let inner = list_transform(
            var("outer"),
            lambda_expr(["inner"], checked_add(var("inner"), lit(1_i32)))?,
        )?;
        let expression = list_transform(root(), lambda_expr(["outer"], inner)?)?;
        let session = array_session();

        let round_tripped = Expression::from_proto(&expression.serialize_proto()?, &session)?;
        assert_eq!(round_tripped, expression);
        Ok(())
    }
}
