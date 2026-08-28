// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ArrayView;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::Primitive;
use crate::dtype::DType;
use crate::match_each_integer_ptype;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::list_contains::IntegerMembership;
use crate::scalar_fn::fns::list_contains::ListContainsElementKernel;

impl ListContainsElementKernel for Primitive {
    fn list_contains(
        list: &ArrayRef,
        element: ArrayView<'_, Self>,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(list_scalar) = list.as_constant() else {
            return Ok(None);
        };
        let DType::List(member_dtype, _) = list.dtype() else {
            return Ok(None);
        };
        if !member_dtype.eq_ignore_nullability(element.dtype()) || !element.ptype().is_int() {
            return Ok(None);
        }

        let nullability = list.dtype().nullability() | element.dtype().nullability();
        let Some(elements) = list_scalar.as_list().elements() else {
            return Ok(Some(
                ConstantArray::new(Scalar::null(DType::Bool(nullability)), element.len())
                    .into_array(),
            ));
        };
        if elements.is_empty() {
            return Ok(Some(
                ConstantArray::new(Scalar::bool(false, nullability), element.len()).into_array(),
            ));
        }

        let result = match_each_integer_ptype!(element.ptype(), |T| {
            let members = elements
                .iter()
                .map(|value| {
                    value
                        .as_primitive_opt()
                        .vortex_expect("list dtype was checked before member extraction")
                        .try_typed_value::<T>()
                })
                .collect::<VortexResult<Vec<Option<T>>>>()?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();

            IntegerMembership::new(members).evaluate_primitive(element, nullability)?
        });

        Ok(Some(result))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_buffer::BitBuffer;

    use super::*;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::dtype::Nullability;
    use crate::dtype::PType::I32;

    fn list(values: impl IntoIterator<Item = i32>, len: usize) -> ArrayRef {
        ConstantArray::new(
            Scalar::list(
                Arc::new(DType::Primitive(I32, Nullability::NonNullable)),
                values
                    .into_iter()
                    .map(|value| Scalar::primitive(value, Nullability::NonNullable))
                    .collect(),
                Nullability::NonNullable,
            ),
            len,
        )
        .into_array()
    }

    #[rstest]
    #[case::empty(vec![])]
    #[case::one(vec![3])]
    #[case::four(vec![3, 7, 11, 15])]
    #[case::dense((0..32).map(|value| value * 3).collect())]
    #[case::sparse((0..32).map(|value| value * 10_000).collect())]
    fn test_membership_plans(#[case] members: Vec<i32>) -> VortexResult<()> {
        let mut ctx = crate::array_session().create_execution_ctx();
        let values = [0, 3, 7, 15, 31, 90_000, 310_000];
        let element = PrimitiveArray::from_iter(values);
        let expected = BoolArray::from_iter(values.map(|value| members.contains(&value)));

        let actual = <Primitive as ListContainsElementKernel>::list_contains(
            &list(members, element.len()),
            element.as_view(),
            &mut ctx,
        )?
        .vortex_expect("integer constant-list membership is supported");

        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_null_needles() -> VortexResult<()> {
        let mut ctx = crate::array_session().create_execution_ctx();
        let element = PrimitiveArray::from_option_iter([Some(1), None, Some(2)]);
        let expected = BoolArray::from_iter([Some(true), None, Some(false)]);

        let actual = <Primitive as ListContainsElementKernel>::list_contains(
            &list([1, 3], element.len()),
            element.as_view(),
            &mut ctx,
        )?
        .vortex_expect("integer constant-list membership is supported");

        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_empty_list_ignores_needle_validity() -> VortexResult<()> {
        let mut ctx = crate::array_session().create_execution_ctx();
        let element = PrimitiveArray::from_option_iter([Some(1i32), None, Some(2)]);
        let expected = BoolArray::new(BitBuffer::new_unset(3), crate::validity::Validity::AllValid);

        let actual = <Primitive as ListContainsElementKernel>::list_contains(
            &list([], element.len()),
            element.as_view(),
            &mut ctx,
        )?
        .vortex_expect("integer constant-list membership is supported");

        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }
}
