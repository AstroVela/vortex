// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::Canonical;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::EmptyOptions;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::min_max::MinMaxResult;
use crate::aggregate_fn::fns::min_max::min_max;
use crate::arrays::ListViewArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::fixed_size_list::FixedSizeListArrayExt;
use crate::arrays::listview::ListViewArrayExt;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::dtype::DType;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::StructFields;
use crate::scalar::Scalar;

/// Field containing the minimum length among non-null lists.
pub const MIN_LENGTH: &str = "min_length";
/// Field containing the maximum length among non-null lists.
pub const MAX_LENGTH: &str = "max_length";
static NAMES: LazyLock<FieldNames> = LazyLock::new(|| FieldNames::from([MIN_LENGTH, MAX_LENGTH]));

/// Creates the struct dtype used by [`ListLengthMinMax`].
pub fn list_length_min_max_dtype() -> DType {
    DType::Struct(
        StructFields::new(
            NAMES.clone(),
            vec![
                DType::Primitive(PType::U64, Nullability::NonNullable),
                DType::Primitive(PType::U64, Nullability::NonNullable),
            ],
        ),
        Nullability::Nullable,
    )
}

/// Minimum and maximum lengths among the non-null values of a list or fixed size list array.
///
/// Null lists are ignored. Empty lists contribute a length of zero and participate in the minimum
/// and maximum. The result is null when there are no non-null lists. Fixed-size-list inputs read
/// both bounds from the dtype without accessing their elements.
#[derive(Clone, Debug)]
pub struct ListLengthMinMax;

/// Partial accumulator state for [`ListLengthMinMax`].
pub struct ListLengthMinMaxPartial {
    min: Option<u64>,
    max: Option<u64>,
    fixed_size: bool,
}

impl ListLengthMinMaxPartial {
    fn observe(&mut self, length: u64) {
        self.min = Some(self.min.map_or(length, |current| current.min(length)));
        self.max = Some(self.max.map_or(length, |current| current.max(length)));
    }

    fn merge_bounds(&mut self, min: Option<u64>, max: Option<u64>) {
        if let Some(min_length) = min {
            self.min = Some(
                self.min
                    .map_or(min_length, |current| current.min(min_length)),
            );
        }
        if let Some(max_length) = max {
            self.max = Some(
                self.max
                    .map_or(max_length, |current| current.max(max_length)),
            );
        }
    }
}

impl AggregateFnVTable for ListLengthMinMax {
    type Options = EmptyOptions;
    type Partial = ListLengthMinMaxPartial;

    fn id(&self) -> AggregateFnId {
        static ID: CachedId = CachedId::new("vortex.list_length_min_max");
        *ID
    }

    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        Ok(EmptyOptions)
    }

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        matches!(input_dtype, DType::List(..) | DType::FixedSizeList(..))
            .then(list_length_min_max_dtype)
    }

    fn partial_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        self.return_dtype(options, input_dtype)
    }

    fn empty_partial(
        &self,
        _options: &Self::Options,
        input_dtype: &DType,
    ) -> VortexResult<Self::Partial> {
        Ok(ListLengthMinMaxPartial {
            min: None,
            max: None,
            fixed_size: matches!(input_dtype, DType::FixedSizeList(..)),
        })
    }

    fn combine_partials(&self, partial: &mut Self::Partial, other: Scalar) -> VortexResult<()> {
        if other.is_null() {
            return Ok(());
        }

        let other = other.as_struct();
        let min_length = other
            .field(MIN_LENGTH)
            .vortex_expect("list length min/max min_length field")
            .as_primitive()
            .typed_value::<u64>()
            .vortex_expect("list length min/max min_length must be non-null");
        let max_length = other
            .field(MAX_LENGTH)
            .vortex_expect("list length min/max max_length field")
            .as_primitive()
            .typed_value::<u64>()
            .vortex_expect("list length min/max max_length must be non-null");
        partial.merge_bounds(Some(min_length), Some(max_length));
        Ok(())
    }

    fn to_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        let dtype = list_length_min_max_dtype();
        Ok(match (partial.min, partial.max) {
            (Some(min_length), Some(max_length)) => Scalar::struct_(
                dtype,
                vec![
                    Scalar::primitive(min_length, Nullability::NonNullable),
                    Scalar::primitive(max_length, Nullability::NonNullable),
                ],
            ),
            _ => Scalar::null(dtype),
        })
    }

    fn reset(&self, partial: &mut Self::Partial) {
        partial.min = None;
        partial.max = None;
    }

    #[inline]
    fn is_saturated(&self, partial: &Self::Partial) -> bool {
        partial.fixed_size && partial.min.is_some()
    }

    fn try_accumulate(
        &self,
        partial: &mut Self::Partial,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool> {
        let DType::FixedSizeList(_, size, _) = batch.dtype() else {
            return Ok(false);
        };

        if !batch.is_empty() && !batch.all_invalid(ctx)? {
            partial.observe(u64::from(*size));
        }
        Ok(true)
    }

    fn accumulate(
        &self,
        partial: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        match batch {
            Columnar::Constant(constant) => {
                if constant.is_empty() || constant.scalar().is_null() {
                    return Ok(());
                }

                let length = u64::try_from(constant.scalar().as_list().len())?;
                partial.observe(length);
                Ok(())
            }
            Columnar::Canonical(Canonical::List(list)) => {
                let sizes = build_sizes_with_validity(list, ctx)?;

                if let Some(MinMaxResult {
                    min: min_scalar,
                    max: max_scalar,
                }) = min_max(&sizes, ctx, NumericalAggregateOpts::default())?
                {
                    let min = size_scalar_as_u64(min_scalar);
                    let max = size_scalar_as_u64(max_scalar);

                    partial.merge_bounds(Some(min), Some(max));
                }
                Ok(())
            }
            Columnar::Canonical(Canonical::FixedSizeList(list)) => {
                if !list.is_empty() && !list.all_invalid(ctx)? {
                    partial.observe(u64::from(list.list_size()));
                }
                Ok(())
            }
            _ => unreachable!(),
        }
    }

    fn finalize(&self, partials: ArrayRef) -> VortexResult<ArrayRef> {
        Ok(partials)
    }

    fn finalize_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        self.to_scalar(partial)
    }
}

fn size_scalar_as_u64(scalar: Scalar) -> u64 {
    scalar
        .as_primitive()
        .as_::<u64>()
        .vortex_expect("size must fit in u64")
}

/// Extracts sizes and validity children from a `ListViewArray` and combines
/// them into a nullable `PrimitiveArray`.
fn build_sizes_with_validity(
    listview: &ListViewArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let sizes = listview.sizes().clone().execute::<PrimitiveArray>(ctx)?;
    let validity = listview.validity()?;

    // SAFETY: A canonical ListViewArray guarantees that its sizes and validity children both have
    // `listview.len()` entries.
    Ok(unsafe {
        PrimitiveArray::new_unchecked_from_handle(
            sizes.buffer_handle().clone(),
            sizes.ptype(),
            validity,
        )
    }
    .into_array())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::*;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::Accumulator;
    use crate::aggregate_fn::DynAccumulator;
    use crate::array_session;
    use crate::arrays::FixedSizeListArray;
    use crate::arrays::ListArray;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::validity::Validity;

    fn compute(array: ArrayRef) -> VortexResult<Scalar> {
        let mut accumulator =
            Accumulator::try_new(ListLengthMinMax, EmptyOptions, array.dtype().clone())?;
        accumulator.accumulate(&array, &mut array_session().create_execution_ctx())?;
        accumulator.finish()
    }

    fn field(stats: &Scalar, name: &str) -> Scalar {
        stats
            .as_struct()
            .field(name)
            .vortex_expect("list length min/max field")
    }

    #[test]
    fn computes_length_bounds_and_ignores_null_lists() -> VortexResult<()> {
        let array = ListArray::try_new(
            buffer![1i32, 2, 3, 4, 5, 6].into_array(),
            buffer![0u32, 2, 5, 5, 6].into_array(),
            Validity::from_iter([true, false, true, true]),
        )?
        .into_array();

        let stats = compute(array)?;
        assert_eq!(
            field(&stats, MIN_LENGTH).as_primitive().as_::<u64>(),
            Some(0)
        );
        assert_eq!(
            field(&stats, MAX_LENGTH).as_primitive().as_::<u64>(),
            Some(2)
        );
        Ok(())
    }

    #[test]
    fn all_null_lists_have_no_length_bounds() -> VortexResult<()> {
        let array = ListArray::try_new(
            buffer![1i32, 2, 3].into_array(),
            buffer![0u32, 1, 3].into_array(),
            Validity::AllInvalid,
        )?
        .into_array();

        let stats = compute(array)?;
        assert!(stats.is_null());
        Ok(())
    }

    #[test]
    fn constant_list_has_length_bounds() -> VortexResult<()> {
        let scalar = Scalar::list(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            vec![1i32.into(), 2i32.into()],
            Nullability::NonNullable,
        );
        let stats = compute(crate::arrays::ConstantArray::new(scalar, 3).into_array())?;

        assert_eq!(
            field(&stats, MIN_LENGTH).as_primitive().as_::<u64>(),
            Some(2)
        );
        assert_eq!(
            field(&stats, MAX_LENGTH).as_primitive().as_::<u64>(),
            Some(2)
        );
        Ok(())
    }

    #[test]
    fn empty_constant_list_has_no_length_bounds() -> VortexResult<()> {
        let scalar = Scalar::list(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            vec![1i32.into(), 2i32.into()],
            Nullability::NonNullable,
        );

        assert!(compute(crate::arrays::ConstantArray::new(scalar, 0).into_array())?.is_null());
        Ok(())
    }

    #[test]
    fn combines_partial_length_bounds() -> VortexResult<()> {
        let first = ListArray::try_new(
            buffer![1i32, 2].into_array(),
            buffer![0u32, 2, 2].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let second = ListArray::try_new(
            buffer![3i32, 4, 5].into_array(),
            buffer![0u32, 3].into_array(),
            Validity::NonNullable,
        )?
        .into_array();

        let mut first_accumulator =
            Accumulator::try_new(ListLengthMinMax, EmptyOptions, first.dtype().clone())?;
        first_accumulator.accumulate(&first, &mut array_session().create_execution_ctx())?;

        let mut combined =
            Accumulator::try_new(ListLengthMinMax, EmptyOptions, second.dtype().clone())?;
        combined.accumulate(&second, &mut array_session().create_execution_ctx())?;
        combined.combine_partials(first_accumulator.partial_scalar()?)?;
        let stats = combined.finish()?;

        assert_eq!(
            field(&stats, MIN_LENGTH).as_primitive().as_::<u64>(),
            Some(0)
        );
        assert_eq!(
            field(&stats, MAX_LENGTH).as_primitive().as_::<u64>(),
            Some(3)
        );
        Ok(())
    }

    #[test]
    fn fixed_size_lists_use_the_static_length() -> VortexResult<()> {
        let array = FixedSizeListArray::new(
            buffer![1i32, 2, 3, 4, 5, 6].into_array(),
            2,
            Validity::from_iter([false, true, false]),
            3,
        )
        .into_array();
        let mut accumulator =
            Accumulator::try_new(ListLengthMinMax, EmptyOptions, array.dtype().clone())?;
        assert!(!accumulator.is_saturated());
        accumulator.accumulate(&array, &mut array_session().create_execution_ctx())?;
        assert!(accumulator.is_saturated());

        let stats = accumulator.finish()?;
        assert_eq!(
            field(&stats, MIN_LENGTH).as_primitive().as_::<u64>(),
            Some(2)
        );
        assert_eq!(
            field(&stats, MAX_LENGTH).as_primitive().as_::<u64>(),
            Some(2)
        );
        Ok(())
    }

    #[test]
    fn all_null_fixed_size_lists_have_no_length_bounds() -> VortexResult<()> {
        let array = FixedSizeListArray::new(
            buffer![1i32, 2, 3, 4].into_array(),
            2,
            Validity::AllInvalid,
            2,
        )
        .into_array();

        let mut accumulator =
            Accumulator::try_new(ListLengthMinMax, EmptyOptions, array.dtype().clone())?;
        accumulator.accumulate(&array, &mut array_session().create_execution_ctx())?;

        assert!(!accumulator.is_saturated());
        assert!(accumulator.finish()?.is_null());
        Ok(())
    }

    #[test]
    fn variable_lists_do_not_saturate() -> VortexResult<()> {
        let array = ListArray::try_new(
            buffer![1i32, 2, 3, 4].into_array(),
            buffer![0u32, 2, 4].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let mut accumulator =
            Accumulator::try_new(ListLengthMinMax, EmptyOptions, array.dtype().clone())?;
        accumulator.accumulate(&array, &mut array_session().create_execution_ctx())?;

        assert!(!accumulator.is_saturated());
        Ok(())
    }

    #[test]
    fn only_supports_list_dtypes() {
        let element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
        assert!(
            ListLengthMinMax
                .return_dtype(
                    &EmptyOptions,
                    &DType::List(Arc::clone(&element_dtype), Nullability::Nullable)
                )
                .is_some()
        );
        assert!(
            ListLengthMinMax
                .return_dtype(
                    &EmptyOptions,
                    &DType::FixedSizeList(Arc::clone(&element_dtype), 2, Nullability::Nullable)
                )
                .is_some()
        );
        assert!(
            ListLengthMinMax
                .return_dtype(
                    &EmptyOptions,
                    &DType::Primitive(PType::I32, Nullability::Nullable)
                )
                .is_none()
        );
    }
}
