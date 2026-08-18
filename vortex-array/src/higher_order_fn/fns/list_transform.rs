// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::Canonical;
use crate::IntoArray;
use crate::arrays::BoolArray;
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
use crate::expr::BoundLambda;
use crate::expr::BoundLambdaArgs;
use crate::expr::Lambda;
use crate::higher_order_fn::EmptyOptions;
use crate::higher_order_fn::HigherOrderFunctionId;
use crate::higher_order_fn::HigherOrderFunctionVTable;
use crate::higher_order_fn::LambdaCall;
use crate::higher_order_fn::LambdaSignature;
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
    type Options = EmptyOptions;

    fn id(&self) -> HigherOrderFunctionId {
        static ID: CachedId = CachedId::new("vortex.list.transform");
        *ID
    }

    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &vortex_session::VortexSession,
    ) -> VortexResult<Self::Options> {
        if metadata.is_empty() {
            Ok(EmptyOptions)
        } else {
            vortex_bail!("list_transform() does not accept metadata")
        }
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn lambda_arity(&self, _options: &Self::Options) -> usize {
        1
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            _ => unreachable!("invalid child index {child_idx} for list_transform()"),
        }
    }

    fn lambda_signatures(
        &self,
        _options: &Self::Options,
        args: &[DType],
        lambdas: &[&Lambda],
    ) -> VortexResult<Box<[LambdaSignature]>> {
        let [lambda] = lambdas else {
            vortex_bail!("list_transform() requires exactly one lambda");
        };
        vortex_ensure!(
            lambda.params().len() == 1,
            "list_transform() lambda must take exactly one parameter, got {}",
            lambda.params().len()
        );
        let (element, list_nullability) = match &args[0] {
            DType::List(element, nullability) | DType::FixedSizeList(element, _, nullability) => {
                (element, *nullability)
            }
            dtype => vortex_bail!("list_transform() requires List or FixedSizeList, got {dtype}"),
        };
        // A null list owns no observable elements, but its physical elements are still present in
        // the child array. Bind nullable list inputs as nullable lambda arguments so lowering can
        // push the list validity into the scalar-function child before it is evaluated.
        let element = match list_nullability {
            Nullability::NonNullable => element.as_ref().clone(),
            Nullability::Nullable => element.as_nullable(),
        };
        Ok(Box::new([LambdaSignature::new(element.clone(), [element])]))
    }

    fn return_dtype(
        &self,
        _options: &Self::Options,
        args: &[DType],
        lambdas: BoundLambdaArgs<'_>,
    ) -> VortexResult<DType> {
        list_transform_dtype(args, lambdas.get(0))
    }

    fn validity(
        &self,
        _options: &Self::Options,
        args: &[ArrayRef],
        _lambdas: BoundLambdaArgs<'_>,
    ) -> VortexResult<Validity> {
        let [input] = args else {
            vortex_bail!("list_transform() requires exactly one input");
        };
        input.validity()
    }

    fn apply(
        &self,
        _options: &Self::Options,
        args: &[ArrayRef],
        lambdas: &[LambdaCall],
        execution_ctx: &mut crate::ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let [input] = args else {
            vortex_bail!("list_transform() requires exactly one input");
        };
        let [lambda] = lambdas else {
            vortex_bail!("list_transform() requires exactly one lambda");
        };
        let output_dtype =
            list_transform_dtype(std::slice::from_ref(input.dtype()), Some(lambda.lambda()))?;
        if input.is_empty() {
            return Ok(Canonical::empty(&output_dtype).into_array());
        }

        // Do not execute a concrete list encoding: `execute_until` may canonicalize a ListView
        // into a List, which would discard the structural layout this lowering can preserve.
        let input = if input.as_opt::<List>().is_some()
            || input.as_opt::<ListView>().is_some()
            || input.as_opt::<FixedSizeList>().is_some()
        {
            input.clone()
        } else {
            input.clone().execute_until::<AnyList>(execution_ctx)?
        };
        transform_list(&input, lambda, execution_ctx)
    }
}

fn list_transform_dtype(args: &[DType], lambda: Option<&BoundLambda>) -> VortexResult<DType> {
    let Some(lambda) = lambda else {
        vortex_bail!("list_transform() requires exactly one lambda");
    };
    let [input] = args else {
        vortex_bail!("list_transform() requires exactly one input");
    };
    match input {
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

fn transform_list(
    input: &ArrayRef,
    lambda: &LambdaCall,
    ctx: &mut crate::ExecutionCtx,
) -> VortexResult<ArrayRef> {
    if let Some(fixed_size_list) = input.as_opt::<FixedSizeList>() {
        let validity = fixed_size_list.fixed_size_list_validity();
        let parent_indices = lambda.has_captures().then(|| {
            fixed_size_list_parent_indices(
                fixed_size_list.list_size() as usize,
                fixed_size_list.len(),
            )
        });
        let elements = fixed_size_list.elements();
        return Ok(FixedSizeListArray::new(
            apply_lambda(
                lambda,
                elements.clone(),
                fixed_size_list_element_mask(
                    lambda,
                    elements,
                    fixed_size_list.list_size() as usize,
                    &validity,
                    fixed_size_list.len(),
                    ctx,
                )?,
                parent_indices.as_ref(),
                ctx,
            )?,
            fixed_size_list.list_size(),
            validity,
            fixed_size_list.len(),
        )
        .into_array());
    }

    if input.as_opt::<ListView>().is_none()
        && let Some(list) = input.as_opt::<List>()
    {
        let validity = list.list_validity();
        if lambda.has_captures() {
            let (elements, offsets, parent_indices) = gather_referenced_elements(
                list.elements(),
                list.offsets(),
                None,
                &validity,
                list.len(),
                ctx,
            )?;
            return Ok(ListArray::try_new(
                apply_lambda(
                    lambda,
                    elements,
                    None,
                    lambda.has_captures().then_some(&parent_indices),
                    ctx,
                )?,
                offsets,
                validity,
            )?
            .into_array());
        }

        return Ok(ListArray::try_new(
            apply_lambda(
                lambda,
                list.elements().clone(),
                list_element_mask(
                    lambda,
                    list.elements(),
                    list.offsets(),
                    &validity,
                    list.len(),
                    ctx,
                )?,
                None,
                ctx,
            )?,
            list.offsets().clone(),
            validity,
        )?
        .into_array());
    }

    if let Some(list_view) = input.as_opt::<ListView>() {
        let validity = list_view.listview_validity();
        if lambda.has_captures() {
            let (elements, offsets, parent_indices) = gather_referenced_elements(
                list_view.elements(),
                list_view.offsets(),
                Some(list_view.sizes()),
                &validity,
                list_view.len(),
                ctx,
            )?;
            return Ok(ListArray::try_new(
                apply_lambda(
                    lambda,
                    elements,
                    None,
                    lambda.has_captures().then_some(&parent_indices),
                    ctx,
                )?,
                offsets,
                validity,
            )?
            .into_array());
        }

        return Ok(ListViewArray::new(
            apply_lambda(
                lambda,
                list_view.elements().clone(),
                list_view_element_mask(
                    lambda,
                    list_view.elements(),
                    list_view.offsets(),
                    list_view.sizes(),
                    &validity,
                    list_view.len(),
                    ctx,
                )?,
                None,
                ctx,
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

/// Apply `lambda` to its element argument, optionally masking unobservable child values first.
fn apply_lambda(
    lambda: &LambdaCall,
    elements: ArrayRef,
    element_mask: Option<ArrayRef>,
    parent_indices: Option<&ArrayRef>,
    ctx: &mut crate::ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let [param_dtype] = lambda.lambda().param_dtypes() else {
        unreachable!("list_transform binds exactly one lambda parameter")
    };
    let elements = match element_mask {
        Some(mask) => elements.mask(mask)?,
        None if elements.dtype() != param_dtype => elements.cast(param_dtype.clone())?,
        None => elements,
    };
    lambda.apply(
        elements.clone(),
        std::slice::from_ref(&elements),
        parent_indices,
        ctx,
    )
}

fn requires_element_mask(lambda: &LambdaCall, elements: &ArrayRef) -> bool {
    let [param_dtype] = lambda.lambda().param_dtypes() else {
        unreachable!("list_transform binds exactly one lambda parameter")
    };
    elements.dtype() != param_dtype
}

fn fixed_size_list_element_mask(
    lambda: &LambdaCall,
    elements: &ArrayRef,
    list_size: usize,
    validity: &Validity,
    len: usize,
    ctx: &mut crate::ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    if !requires_element_mask(lambda, elements) {
        return Ok(None);
    }

    let parent_mask = validity.execute_mask(len, ctx)?;
    Ok(Some(
        BoolArray::from_iter(
            (0..len).flat_map(|row| std::iter::repeat_n(parent_mask.value(row), list_size)),
        )
        .into_array(),
    ))
}

fn list_element_mask(
    lambda: &LambdaCall,
    elements: &ArrayRef,
    offsets: &ArrayRef,
    validity: &Validity,
    len: usize,
    ctx: &mut crate::ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    if !requires_element_mask(lambda, elements) {
        return Ok(None);
    }

    let offsets = offsets
        .cast(DType::Primitive(PType::U64, Nullability::NonNullable))?
        .execute::<PrimitiveArray>(ctx)?;
    let offsets = offsets.as_slice::<u64>();
    let parent_mask = validity.execute_mask(len, ctx)?;
    let mut mask = Vec::with_capacity(elements.len());
    for row in 0..len {
        let start = usize_offset(offsets[row])?;
        let end = usize_offset(offsets[row + 1])?;
        vortex_ensure!(
            end <= elements.len(),
            "list offset {end} exceeds elements length {}",
            elements.len()
        );
        mask.extend(std::iter::repeat_n(parent_mask.value(row), end - start));
    }
    debug_assert_eq!(mask.len(), elements.len());
    Ok(Some(BoolArray::from_iter(mask).into_array()))
}

fn list_view_element_mask(
    lambda: &LambdaCall,
    elements: &ArrayRef,
    offsets: &ArrayRef,
    sizes: &ArrayRef,
    validity: &Validity,
    len: usize,
    ctx: &mut crate::ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    if !requires_element_mask(lambda, elements) {
        return Ok(None);
    }

    let dtype = DType::Primitive(PType::U64, Nullability::NonNullable);
    let offsets = offsets
        .cast(dtype.clone())?
        .execute::<PrimitiveArray>(ctx)?;
    let sizes = sizes.cast(dtype)?.execute::<PrimitiveArray>(ctx)?;
    let offsets = offsets.as_slice::<u64>();
    let sizes = sizes.as_slice::<u64>();
    let parent_mask = validity.execute_mask(len, ctx)?;
    let mut mask = vec![false; elements.len()];
    for row in 0..len {
        if parent_mask.value(row) {
            let start = usize_offset(offsets[row])?;
            let size = usize_offset(sizes[row])?;
            let end = start
                .checked_add(size)
                .ok_or_else(|| vortex_err!("list view range overflow"))?;
            vortex_ensure!(
                end <= elements.len(),
                "list view range {start}..{end} exceeds elements length {}",
                elements.len()
            );
            mask[start..end].fill(true);
        }
    }
    Ok(Some(BoolArray::from_iter(mask).into_array()))
}

fn usize_offset(offset: u64) -> VortexResult<usize> {
    usize::try_from(offset).map_err(|_| vortex_err!("offset {offset} does not fit in usize"))
}

/// Gather elements referenced by valid lists into a dense `List` layout.
///
/// Captures live at list-row granularity, while lambda parameters live at element granularity.
/// `ListView` can also leave gaps in its element array, so its referenced ranges are gathered
/// before captures are rebased to the lambda's elements.
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
    use crate::arrays::ListArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::ScalarFn;
    use crate::arrays::StructArray;
    use crate::assert_arrays_eq;
    use crate::dtype::FieldNames;
    use crate::expr::Expression;
    use crate::expr::cast;
    use crate::expr::checked_add;
    use crate::expr::col;
    use crate::expr::lambda as lambda_expr;
    use crate::expr::list_length;
    use crate::expr::list_transform;
    use crate::expr::lit;
    use crate::expr::not;
    use crate::expr::proto::ExprSerializeProtoExt;
    use crate::expr::root;
    use crate::expr::var;
    use crate::validity::Validity;

    #[test]
    fn lowers_to_a_list_with_scalar_function_elements() -> VortexResult<()> {
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

        let mut ctx = array_session().create_execution_ctx();
        let result = input.apply_with_ctx(&expression, &mut ctx)?;
        assert!(result.is::<List>());
        assert!(result.as_::<List>().elements().is::<ScalarFn>());
        let expected = ListArray::try_new(
            PrimitiveArray::from_iter([2_i32, 3, 4]).into_array(),
            buffer![0_u32, 2, 3].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn preserves_a_fixed_size_list_with_scalar_function_elements() -> VortexResult<()> {
        let input = FixedSizeListArray::new(
            PrimitiveArray::from_iter([1_i32, 2, 3, 4]).into_array(),
            2,
            Validity::NonNullable,
            2,
        )
        .into_array();
        let expression = list_transform(
            root(),
            lambda_expr(["element"], checked_add(var("element"), lit(1_i32)))?,
        )?;

        let result = input.apply(&expression)?;
        assert!(result.is::<FixedSizeList>());
        assert!(result.as_::<FixedSizeList>().elements().is::<ScalarFn>());
        let expected = FixedSizeListArray::new(
            PrimitiveArray::from_iter([2_i32, 3, 4, 5]).into_array(),
            2,
            Validity::NonNullable,
            2,
        )
        .into_array();
        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn masks_garbage_elements_of_null_fixed_size_lists() -> VortexResult<()> {
        let input = FixedSizeListArray::new(
            PrimitiveArray::from_iter([i32::MAX, i32::MAX, 1, 2]).into_array(),
            2,
            Validity::from_iter([false, true]),
            2,
        )
        .into_array();
        let expression = list_transform(
            root(),
            lambda_expr(["element"], checked_add(var("element"), lit(1_i32)))?,
        )?;

        let result = input.apply(&expression)?;
        assert!(result.is::<FixedSizeList>());
        assert!(result.as_::<FixedSizeList>().elements().is::<ScalarFn>());
        let expected = FixedSizeListArray::new(
            PrimitiveArray::from_option_iter([None, None, Some(2_i32), Some(3)]).into_array(),
            2,
            Validity::from_iter([false, true]),
            2,
        )
        .into_array();
        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn lowers_a_list_column_to_scalar_function_elements() -> VortexResult<()> {
        let lists = ListArray::try_new(
            PrimitiveArray::from_iter([1_i32, 2, 3]).into_array(),
            buffer![0_u32, 2, 3].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let input = StructArray::try_new(
            FieldNames::from(["lists"]),
            vec![lists],
            2,
            Validity::NonNullable,
        )?
        .into_array();
        let expression = list_transform(
            col("lists"),
            lambda_expr(["element"], checked_add(var("element"), lit(1_i32)))?,
        )?;

        let result = input.apply(&expression)?;
        assert!(result.is::<List>());
        assert!(result.as_::<List>().elements().is::<ScalarFn>());
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
    fn preserves_a_list_view_with_scalar_function_elements() -> VortexResult<()> {
        let input = ListViewArray::new(
            BoolArray::from_iter([true, false, true, false]).into_array(),
            buffer![2_u32, 0].into_array(),
            buffer![2_u32, 2].into_array(),
            Validity::NonNullable,
        )
        .into_array();
        assert!(input.is::<ListView>());
        let expression = list_transform(root(), lambda_expr(["element"], not(var("element")))?)?;

        let result = input.apply(&expression)?;
        assert!(
            result.is::<ListView>(),
            "expected a ListView result, got {}",
            result.encoding_id()
        );
        assert!(result.as_::<ListView>().elements().is::<ScalarFn>());
        let expected = ListViewArray::new(
            BoolArray::from_iter([false, true, false, true]).into_array(),
            buffer![2_u32, 0].into_array(),
            buffer![2_u32, 2].into_array(),
            Validity::NonNullable,
        )
        .into_array();
        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn masks_garbage_elements_of_null_list_views() -> VortexResult<()> {
        let input = ListViewArray::new(
            PrimitiveArray::from_iter([i32::MAX, i32::MAX, 1, 2]).into_array(),
            buffer![2_u32, 0].into_array(),
            buffer![2_u32, 2].into_array(),
            Validity::from_iter([true, false]),
        )
        .into_array();
        let expression = list_transform(
            root(),
            lambda_expr(["element"], checked_add(var("element"), lit(1_i32)))?,
        )?;

        let result = input.apply(&expression)?;
        assert!(result.is::<ListView>());
        assert!(result.as_::<ListView>().elements().is::<ScalarFn>());
        let expected = ListViewArray::new(
            PrimitiveArray::from_option_iter([None, None, Some(2_i32), Some(3)]).into_array(),
            buffer![2_u32, 0].into_array(),
            buffer![2_u32, 2].into_array(),
            Validity::from_iter([true, false]),
        )
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
    fn masks_garbage_elements_of_null_lists_before_lambda_application() -> VortexResult<()> {
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
        assert!(result.is::<List>());
        assert!(result.as_::<List>().elements().is::<ScalarFn>());
        let expected = ListArray::try_new(
            PrimitiveArray::from_option_iter([Some(2_i32), None]).into_array(),
            buffer![0_u32, 1, 2].into_array(),
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
