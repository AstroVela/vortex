// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::list_contains::ListContainsElementKernel;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::compare_fused::stream_compare_fused;
use crate::BitPacked;

#[derive(Clone, Copy)]
enum SearchStrategy {
    Linear,
    Binary,
}

impl ListContainsElementKernel for BitPacked {
    fn list_contains(
        list: &ArrayRef,
        element: ArrayView<'_, Self>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        list_contains_with_strategy(list, element, SearchStrategy::Binary, ctx)
    }
}

fn list_contains_with_strategy(
    list: &ArrayRef,
    element: ArrayView<'_, BitPacked>,
    strategy: SearchStrategy,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let Some(list_scalar) = list.as_constant() else {
        return Ok(None);
    };
    let DType::List(member_dtype, _) = list.dtype() else {
        return Ok(None);
    };
    if !member_dtype.eq_ignore_nullability(element.dtype()) {
        return Ok(None);
    }

    let nullability = list.dtype().nullability() | element.dtype().nullability();
    let Some(elements) = list_scalar.as_list().elements() else {
        return Ok(Some(
            ConstantArray::new(Scalar::null(DType::Bool(nullability)), element.len()).into_array(),
        ));
    };

    let result = match_each_integer_ptype!(element.dtype().as_ptype(), |T| {
        let mut members = elements
            .iter()
            .map(|value| {
                value
                    .as_primitive_opt()
                    .ok_or_else(|| vortex_err!("List member is not a primitive scalar"))?
                    .try_typed_value::<T>()
            })
            .collect::<VortexResult<Vec<Option<T>>>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        match members.as_slice() {
            [] => ConstantArray::new(Scalar::bool(false, nullability), element.len()).into_array(),
            [member] => {
                let member = *member;
                stream_compare_fused::<T, _>(element, member, nullability, NativePType::is_eq, ctx)?
            }
            [first, second] => {
                let (first, second) = (*first, *second);
                stream_compare_fused::<T, _>(
                    element,
                    first,
                    nullability,
                    move |value, _| value.is_eq(first) | value.is_eq(second),
                    ctx,
                )?
            }
            _ if matches!(strategy, SearchStrategy::Linear) => stream_compare_fused::<T, _>(
                element,
                members[0],
                nullability,
                |value, _| members.contains(&value),
                ctx,
            )?,
            _ => {
                members.sort_unstable();
                members.dedup();
                stream_compare_fused::<T, _>(
                    element,
                    members[0],
                    nullability,
                    |value, _| members.binary_search(&value).is_ok(),
                    ctx,
                )?
            }
        }
    });
    Ok(Some(result))
}

#[cfg(feature = "_test-harness")]
pub mod test_harness {
    use vortex_array::ArrayRef;
    use vortex_array::ArrayView;
    use vortex_array::ExecutionCtx;
    use vortex_error::VortexResult;

    use super::SearchStrategy;
    use super::list_contains_with_strategy;
    use crate::BitPacked;

    /// Selects the membership lookup strategy for a benchmark invocation.
    #[derive(Clone, Copy)]
    pub enum MembershipSearch {
        /// Scans list members in order.
        Linear,
        /// Sorts list members and uses binary search.
        Binary,
    }

    /// Executes the BitPacked membership kernel with a fixed lookup strategy.
    pub fn list_contains(
        list: &ArrayRef,
        element: ArrayView<'_, BitPacked>,
        strategy: MembershipSearch,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let strategy = match strategy {
            MembershipSearch::Linear => SearchStrategy::Linear,
            MembershipSearch::Binary => SearchStrategy::Binary,
        };
        list_contains_with_strategy(list, element, strategy, ctx)
    }
}

#[cfg(test)]
mod tests;
