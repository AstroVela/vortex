// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod bool;
mod constant;
mod decimal;
mod grouped;
mod primitive;
pub(crate) use grouped::PrimitiveGroupedStandardSumEncodingKernel;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use self::bool::accumulate_bool;
use self::constant::multiply_constant;
use self::decimal::accumulate_decimal;
use self::primitive::accumulate_primitive;
use crate::ArrayRef;
use crate::Canonical;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::DynAccumulator;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::sum::SumState;
use crate::aggregate_fn::fns::sum::checked_add_i64;
use crate::aggregate_fn::fns::sum::checked_add_u64;
use crate::aggregate_fn::fns::sum::make_zero_state;
use crate::aggregate_fn::fns::sum::sum_decimal_dtype;
use crate::arrays::ConstantArray;
use crate::arrays::scalar_fn::ScalarFnFactoryExt;
use crate::dtype::DType;
use crate::dtype::FieldName;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::StructFields;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProvider;
use crate::expr::stats::StatsProviderExt;
use crate::scalar::Scalar;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::fns::fill_null::FillNull;
use crate::scalar_fn::fns::get_item::GetItem;
use crate::scalar_fn::fns::mask::Mask;

/// Return the SQL sum of an array: null when the array has no valid values or the sum
/// overflows.
///
/// See [`StandardSum`] for details.
pub fn standard_sum(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Scalar> {
    // Short-circuit using cached array statistics. `Stat::Sum` is the monoid sum (zero for an
    // array with no valid values), so the SQL empty-sum rule needs the null count: when it is
    // unknown for a nullable array, fall through and compute rather than trust the cache.
    if let Precision::Exact(sum_scalar) = array.statistics().get(Stat::Sum) {
        if !array.dtype().is_nullable() {
            return Ok(sum_scalar);
        }
        match array.statistics().get_as::<u64>(Stat::NullCount) {
            Precision::Exact(null_count) if null_count == array.len() as u64 => {
                return Ok(Scalar::null(sum_scalar.dtype().as_nullable()));
            }
            Precision::Exact(_) => return Ok(sum_scalar),
            _ => {}
        }
    }

    // Compute using Accumulator<StandardSum>.
    // TODO(ngates): we may want to wrap this three-step dance up into an extension crate maybe.
    let mut acc = Accumulator::try_new(
        StandardSum,
        NumericalAggregateOpts::default(),
        array.dtype().clone(),
    )?;
    acc.accumulate(array, ctx)?;
    let result = acc.finish()?;

    // Cache the computed sum as a statistic (only if non-null, i.e. no overflow and at least
    // one valid value).
    if let Some(val) = result.value().cloned() {
        array.statistics().set(Stat::Sum, Precision::Exact(val));
    }

    Ok(result)
}

/// Sum an array following SQL `SUM` semantics.
///
/// A sum over zero valid values yields null (the SQL rule: nulls are eliminated, and the sum
/// of an empty set is null), and a sum that overflows yields null. This is a distinct
/// aggregate from the monoid [`Sum`](crate::aggregate_fn::fns::sum::Sum), whose empty sum is
/// zero and whose plain partial is the persisted form of sum statistics; `StandardSum`'s
/// `{sum, seen}` partial is never written to statistics.
///
/// The partial state is a `{sum, seen}` struct: `sum` is the running monoid value (null once
/// saturated by overflow, which poisons merges), and `seen` records whether any valid value
/// contributed (merged with OR). `finalize` maps `seen == false` to null. `seen` is decided by
/// validity alone: NaN values are valid, so with `skip_nans` a sum over only NaNs is `0`, not
/// null.
///
/// NaN handling for float inputs is controlled by [`NumericalAggregateOpts`]: with `skip_nans` (the
/// default) NaN values contribute nothing, otherwise any NaN value poisons the sum to NaN.
#[derive(Clone, Debug)]
pub struct StandardSum;

impl AggregateFnVTable for StandardSum {
    type Options = NumericalAggregateOpts;
    type Partial = StandardSumPartial;

    fn id(&self) -> AggregateFnId {
        static ID: CachedId = CachedId::new("vortex.standard_sum");
        *ID
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(options.serialize()))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        NumericalAggregateOpts::deserialize(metadata)
    }

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        // When a sum overflows, we return a sum _value_ of null. Therefore, we all return dtypes
        // are nullable.
        use Nullability::Nullable;

        Some(match input_dtype {
            DType::Bool(_) => DType::Primitive(PType::U64, Nullable),
            DType::Primitive(ptype, _) => match ptype {
                PType::U8 | PType::U16 | PType::U32 | PType::U64 => {
                    DType::Primitive(PType::U64, Nullable)
                }
                PType::I8 | PType::I16 | PType::I32 | PType::I64 => {
                    DType::Primitive(PType::I64, Nullable)
                }
                PType::F16 | PType::F32 | PType::F64 => {
                    // Float sums cannot overflow, but all null floats still end up as null
                    DType::Primitive(PType::F64, Nullable)
                }
            },
            DType::Decimal(decimal_dtype, _) => {
                DType::Decimal(sum_decimal_dtype(decimal_dtype), Nullable)
            }
            // Unsupported types
            _ => return None,
        })
    }

    fn partial_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        Some(sum_partial_dtype(self.return_dtype(options, input_dtype)?))
    }

    fn empty_partial(
        &self,
        options: &Self::Options,
        input_dtype: &DType,
    ) -> VortexResult<Self::Partial> {
        let return_dtype = self
            .return_dtype(options, input_dtype)
            .ok_or_else(|| vortex_err!("Unsupported sum dtype: {}", input_dtype))?;
        let initial = make_zero_state(&return_dtype);

        Ok(StandardSumPartial {
            return_dtype,
            current: Some(initial),
            seen: false,
            skip_nans: options.skip_nans,
        })
    }

    fn combine_partials(&self, partial: &mut Self::Partial, other: Scalar) -> VortexResult<()> {
        // Partials are `{sum, seen}` structs. A plain (non-struct) scalar is the legacy or
        // statistic form: a monoid sum value from an older writer or a cached `Stat::Sum`. It
        // cannot distinguish an empty sum from a zero sum, so it is treated as seen.
        let (sum_value, other_seen) = if matches!(other.dtype(), DType::Struct(..)) {
            if other.is_null() {
                // A null struct partial carries no recoverable state; treat as saturated.
                partial.seen = true;
                partial.current = None;
                return Ok(());
            }
            let fields = other.as_struct();
            let sum_value = fields
                .field("sum")
                .ok_or_else(|| vortex_err!("StandardSum partial is missing the `sum` field"))?;
            let other_seen = fields
                .field("seen")
                .and_then(|seen| seen.as_bool().value())
                .unwrap_or(true);
            (sum_value, other_seen)
        } else {
            (other, true)
        };

        partial.seen |= other_seen;
        if sum_value.is_null() {
            // A null sum value means the sub-accumulator saturated (overflow).
            partial.current = None;
            return Ok(());
        }
        let Some(ref mut inner) = partial.current else {
            return Ok(());
        };
        let other = sum_value;
        let saturated = match inner {
            SumState::Unsigned(acc) => {
                let val = other
                    .as_primitive()
                    .typed_value::<u64>()
                    .vortex_expect("checked non-null");
                checked_add_u64(acc, val)
            }
            SumState::Signed(acc) => {
                let val = other
                    .as_primitive()
                    .typed_value::<i64>()
                    .vortex_expect("checked non-null");
                checked_add_i64(acc, val)
            }
            SumState::Float(acc) => {
                let val = other
                    .as_primitive()
                    .typed_value::<f64>()
                    .vortex_expect("checked non-null");
                *acc += val;
                false
            }
            SumState::Decimal { value, dtype } => {
                let val = other
                    .as_decimal()
                    .decimal_value()
                    .vortex_expect("checked non-null");
                match value.checked_add(&val) {
                    Some(r) => {
                        *value = r;
                        !value.fits_in_precision(*dtype)
                    }
                    None => true,
                }
            }
        };
        if saturated {
            partial.current = None;
        }
        Ok(())
    }

    fn to_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        Ok(Scalar::struct_(
            sum_partial_dtype(partial.return_dtype.as_nullable()),
            vec![
                sum_value_scalar(partial),
                Scalar::bool(partial.seen, Nullability::NonNullable),
            ],
        ))
    }

    fn reset(&self, partial: &mut Self::Partial) {
        partial.current = Some(make_zero_state(&partial.return_dtype));
        partial.seen = false;
    }

    #[inline]
    fn is_saturated(&self, partial: &Self::Partial) -> bool {
        match partial.current.as_ref() {
            None => true,
            Some(SumState::Float(v)) => v.is_nan(),
            Some(_) => false,
        }
    }

    fn try_accumulate(
        &self,
        partial: &mut Self::Partial,
        batch: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool> {
        // `Stat::Sum` is the monoid (NaN-skipping) sum, so the default NaN-skipping path can
        // consume it directly. `StandardSum` has no Stat slot of its own — the shortcut lives here
        // rather than in the accumulator's stats bridge.
        if partial.skip_nans {
            return try_accumulate_cached_sum(self, partial, batch);
        }
        // NaN-including float sums need a NaN-free batch before the cached sum applies;
        // everything else takes the default dispatch path.
        if !matches!(partial.current, Some(SumState::Float(_))) {
            return Ok(false);
        }
        match batch.statistics().get_as::<u64>(Stat::NaNCount) {
            Precision::Exact(0) => {
                // NaN-free batch: the cached NaN-skipping sum (if any) equals the
                // NaN-including sum.
                try_accumulate_cached_sum(self, partial, batch)
            }
            Precision::Exact(_) => {
                // At least one NaN value (a valid value): the sum is NaN without scanning.
                partial.seen = true;
                if let Some(SumState::Float(acc)) = partial.current.as_mut() {
                    *acc = f64::NAN;
                }
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    fn accumulate(
        &self,
        partial: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        // Constants compute scalar * len and combine via combine_partials.
        if let Columnar::Constant(c) = batch {
            // Any valid value counts as seen, including NaN and `false` constants that
            // contribute nothing to the running sum.
            partial.seen |= !c.scalar().is_null() && !c.is_empty();
            // NaN constants are treated as missing when skipping NaNs.
            if partial.skip_nans && c.scalar().as_primitive_opt().is_some_and(|p| p.is_nan()) {
                return Ok(());
            }
            if let Some(product) = multiply_constant(c.scalar(), c.len(), &partial.return_dtype)? {
                self.combine_partials(partial, product)?;
            }
            return Ok(());
        }

        let skip_nans = partial.skip_nans;
        let mut inner = match partial.current.take() {
            Some(inner) => inner,
            None => return Ok(()),
        };

        let result = match batch {
            Columnar::Canonical(c) => match c {
                Canonical::Primitive(p) => {
                    accumulate_primitive(&mut inner, p, ctx, skip_nans, &mut partial.seen)
                }
                Canonical::Bool(b) => accumulate_bool(&mut inner, b, ctx, &mut partial.seen),
                Canonical::Decimal(d) => accumulate_decimal(&mut inner, d, ctx, &mut partial.seen),
                _ => vortex_bail!("Unsupported canonical type for sum: {}", batch.dtype()),
            },
            Columnar::Constant(_) => unreachable!(),
        };

        match result {
            Ok(false) => partial.current = Some(inner),
            Ok(true) => {} // saturated: current stays None
            Err(e) => {
                partial.current = Some(inner);
                return Err(e);
            }
        }
        Ok(())
    }

    fn finalize(&self, partials: ArrayRef) -> VortexResult<ArrayRef> {
        // Entries that saw no valid values finalize to null (SQL `SUM`), while a null `sum`
        // field (overflow) and null partial rows (e.g. null groups) stay null via the mask's
        // validity intersection.
        //
        // The expressions are built unoptimized: `optimize` costs multiples of the whole
        // aggregation at small group counts, and the caller's execution evaluates the lazy
        // expression as-is.
        let len = partials.len();
        let sum = GetItem.try_new_array(len, FieldName::from("sum"), [partials.clone()])?;
        let seen = GetItem.try_new_array(len, FieldName::from("seen"), [partials])?;
        let seen = FillNull.try_new_array(
            len,
            EmptyOptions,
            [
                seen,
                ConstantArray::new(Scalar::bool(false, Nullability::NonNullable), len).into_array(),
            ],
        )?;
        Mask.try_new_array(len, EmptyOptions, [sum, seen])
    }

    fn finalize_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        if !partial.seen {
            return Ok(Scalar::null(partial.return_dtype.as_nullable()));
        }
        Ok(sum_value_scalar(partial))
    }
}

/// Consume a batch's cached monoid `Stat::Sum` instead of scanning it. The cached sum cannot
/// carry `seen`, so the batch's null count decides between the identity (all null) and a seen
/// contribution; when either statistic is missing the caller falls through to a real scan.
fn try_accumulate_cached_sum(
    vtable: &StandardSum,
    partial: &mut StandardSumPartial,
    batch: &ArrayRef,
) -> VortexResult<bool> {
    let Precision::Exact(sum) = batch.statistics().get(Stat::Sum) else {
        return Ok(false);
    };
    match batch.statistics().get_as::<u64>(Stat::NullCount) {
        Precision::Exact(null_count) if null_count == batch.len() as u64 => {
            // No valid values: the batch is the identity.
            return Ok(true);
        }
        Precision::Exact(_) => partial.seen = true,
        _ => return Ok(false),
    }
    let sum = if sum.dtype() == &partial.return_dtype {
        sum
    } else {
        sum.cast(&partial.return_dtype)?
    };
    vtable.combine_partials(partial, sum)?;
    Ok(true)
}

/// The group state for a sum aggregate, containing the accumulated value and configuration
/// needed for reset/result without external context.
pub struct StandardSumPartial {
    return_dtype: DType,
    /// The current accumulated state, or `None` if saturated (checked overflow).
    current: Option<SumState>,
    /// Whether at least one valid value has been accumulated. A sum over zero valid values
    /// finalizes to null (SQL `SUM` semantics) rather than the monoid zero.
    seen: bool,
    /// Whether NaN values in float inputs are skipped.
    skip_nans: bool,
}

/// The partial dtype for a sum whose result is `sum_dtype`: a `{sum, seen}` struct, where
/// `sum` is the running monoid value (null once saturated by overflow, poisoning merges) and
/// `seen` records whether any valid value contributed (merged with OR). Keeping the flag
/// separate from the sum lets merges stay a monoid — the identity is `{0, false}` — while
/// `finalize` maps unseen sums to null.
fn sum_partial_dtype(sum_dtype: DType) -> DType {
    DType::Struct(
        StructFields::new(
            FieldNames::from_iter([FieldName::from("sum"), FieldName::from("seen")]),
            vec![sum_dtype, DType::Bool(Nullability::NonNullable)],
        ),
        Nullability::Nullable,
    )
}

/// The running sum as a nullable scalar of the return dtype (null when saturated by overflow).
fn sum_value_scalar(partial: &StandardSumPartial) -> Scalar {
    match &partial.current {
        None => Scalar::null(partial.return_dtype.as_nullable()),
        Some(SumState::Unsigned(v)) => Scalar::primitive(*v, Nullability::Nullable),
        Some(SumState::Signed(v)) => Scalar::primitive(*v, Nullability::Nullable),
        Some(SumState::Float(v)) => Scalar::primitive(*v, Nullability::Nullable),
        Some(SumState::Decimal { value, .. }) => {
            let decimal_dtype = *partial
                .return_dtype
                .as_decimal_opt()
                .vortex_expect("return dtype must be decimal");
            Scalar::decimal(*value, decimal_dtype, Nullability::Nullable)
        }
    }
}

#[cfg(test)]
mod tests {
    use num_traits::CheckedAdd;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::Accumulator;
    use crate::aggregate_fn::AggregateFnVTable;
    use crate::aggregate_fn::DynAccumulator;
    use crate::aggregate_fn::DynGroupedAccumulator;
    use crate::aggregate_fn::GroupedAccumulator;
    use crate::aggregate_fn::NumericalAggregateOpts;
    use crate::aggregate_fn::fns::standard_sum::StandardSum;
    use crate::aggregate_fn::fns::standard_sum::standard_sum;
    use crate::aggregate_fn::fns::sum::sum as monoid_sum;
    use crate::array_session;
    use crate::arrays::BoolArray;
    use crate::arrays::ChunkedArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::DecimalArray;
    use crate::arrays::FixedSizeListArray;
    use crate::arrays::ListViewArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::Nullability;
    use crate::dtype::Nullability::Nullable;
    use crate::dtype::PType;
    use crate::dtype::i256;
    use crate::expr::stats::Precision;
    use crate::expr::stats::Stat;
    use crate::expr::stats::StatsProvider;
    use crate::scalar::DecimalValue;
    use crate::scalar::NumericOperator;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    /// StandardSum an array with an initial value (test-only helper).
    fn sum_with_accumulator(array: &ArrayRef, accumulator: &Scalar) -> VortexResult<Scalar> {
        let mut ctx = array_session().create_execution_ctx();
        if accumulator.is_null() {
            return Ok(accumulator.clone());
        }
        if accumulator.is_zero() == Some(true) {
            return standard_sum(array, &mut ctx);
        }

        let sum_dtype = Stat::Sum.dtype(array.dtype()).ok_or_else(|| {
            vortex_error::vortex_err!("StandardSum not supported for dtype: {}", array.dtype())
        })?;

        // For non-float types, try statistics short-circuit with accumulator.
        if !matches!(&sum_dtype, DType::Primitive(p, _) if p.is_float())
            && let Precision::Exact(sum_scalar) = array.statistics().get(Stat::Sum)
        {
            return add_scalars(&sum_dtype, &sum_scalar, accumulator);
        }

        // Compute array sum from zero (also caches stats).
        let array_sum = standard_sum(array, &mut ctx)?;

        // Combine with the accumulator.
        add_scalars(&sum_dtype, &array_sum, accumulator)
    }

    /// Add two sum scalars with overflow checking.
    fn add_scalars(sum_dtype: &DType, lhs: &Scalar, rhs: &Scalar) -> VortexResult<Scalar> {
        if lhs.is_null() || rhs.is_null() {
            return Ok(Scalar::null(sum_dtype.as_nullable()));
        }

        Ok(match sum_dtype {
            DType::Primitive(ptype, _) if ptype.is_float() => {
                let lhs_val = f64::try_from(lhs)?;
                let rhs_val = f64::try_from(rhs)?;
                Scalar::primitive(lhs_val + rhs_val, Nullable)
            }
            DType::Primitive(..) => lhs
                .as_primitive()
                .checked_add(&rhs.as_primitive())
                .map(Scalar::from)
                .unwrap_or_else(|| Scalar::null(sum_dtype.as_nullable())),
            DType::Decimal(..) => lhs
                .as_decimal()
                .checked_binary_numeric(&rhs.as_decimal(), NumericOperator::Add)
                .map(Scalar::from)
                .unwrap_or_else(|| Scalar::null(sum_dtype.as_nullable())),
            _ => unreachable!("StandardSum will always be a decimal or a primitive dtype"),
        })
    }

    // Multi-batch and reset tests

    #[test]
    fn sum_multi_batch() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(StandardSum, NumericalAggregateOpts::default(), dtype)?;

        let batch1 = PrimitiveArray::new(buffer![10i32, 20], Validity::NonNullable).into_array();
        acc.accumulate(&batch1, &mut ctx)?;

        let batch2 = PrimitiveArray::new(buffer![3i32, 6, 9], Validity::NonNullable).into_array();
        acc.accumulate(&batch2, &mut ctx)?;

        let result = acc.finish()?;
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(48));
        Ok(())
    }

    #[test]
    fn sum_finish_resets_state() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(StandardSum, NumericalAggregateOpts::default(), dtype)?;

        let batch1 = PrimitiveArray::new(buffer![10i32, 20], Validity::NonNullable).into_array();
        acc.accumulate(&batch1, &mut ctx)?;
        let result1 = acc.finish()?;
        assert_eq!(result1.as_primitive().typed_value::<i64>(), Some(30));

        let batch2 = PrimitiveArray::new(buffer![3i32, 6, 9], Validity::NonNullable).into_array();
        acc.accumulate(&batch2, &mut ctx)?;
        let result2 = acc.finish()?;
        assert_eq!(result2.as_primitive().typed_value::<i64>(), Some(18));
        Ok(())
    }

    // State merge tests (vtable-level)

    #[test]
    fn sum_state_empty_is_null() -> VortexResult<()> {
        // A state that never saw a valid value finalizes to null, and combining empty states
        // stays empty.
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut state = StandardSum.empty_partial(&NumericalAggregateOpts::default(), &dtype)?;
        let empty = StandardSum.to_scalar(&state)?;
        StandardSum.combine_partials(&mut state, empty)?;
        assert!(StandardSum.finalize_scalar(&state)?.is_null());
        Ok(())
    }

    #[test]
    fn sum_state_empty_is_identity() -> VortexResult<()> {
        // Combining an empty state into a seen state changes nothing: `{0, false}` is the
        // identity of the `{sum, seen}` monoid.
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut state = StandardSum.empty_partial(&NumericalAggregateOpts::default(), &dtype)?;
        StandardSum.combine_partials(&mut state, Scalar::primitive(100i64, Nullable))?;

        let empty = StandardSum
            .to_scalar(&StandardSum.empty_partial(&NumericalAggregateOpts::default(), &dtype)?)?;
        StandardSum.combine_partials(&mut state, empty)?;

        let result = StandardSum.finalize_scalar(&state)?;
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(100));
        Ok(())
    }

    #[test]
    fn sum_state_overflow_poisons_but_stays_seen() -> VortexResult<()> {
        // Overflow (a null `sum` field) poisons the merge even when combined with later
        // values: the result is null via the sum value, not via `seen`.
        let dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
        let mut overflowed =
            StandardSum.empty_partial(&NumericalAggregateOpts::default(), &dtype)?;
        StandardSum.combine_partials(&mut overflowed, Scalar::primitive(i64::MAX, Nullable))?;
        StandardSum.combine_partials(&mut overflowed, Scalar::primitive(1i64, Nullable))?;
        let overflowed = StandardSum.to_scalar(&overflowed)?;

        let mut state = StandardSum.empty_partial(&NumericalAggregateOpts::default(), &dtype)?;
        StandardSum.combine_partials(&mut state, Scalar::primitive(5i64, Nullable))?;
        StandardSum.combine_partials(&mut state, overflowed)?;
        StandardSum.combine_partials(&mut state, Scalar::primitive(7i64, Nullable))?;

        assert!(StandardSum.finalize_scalar(&state)?.is_null());
        Ok(())
    }

    #[test]
    fn sum_all_nan_is_zero_not_null() -> VortexResult<()> {
        // NaNs are valid values: with the default `skip_nans` they contribute nothing, but
        // the sum is a genuine `0.0`, unlike an all-null array whose sum is null.
        let arr =
            PrimitiveArray::new(buffer![f64::NAN, f64::NAN], Validity::NonNullable).into_array();
        let result = standard_sum(&arr, &mut array_session().create_execution_ctx())?;
        assert_eq!(result.as_primitive().typed_value::<f64>(), Some(0.0));
        Ok(())
    }

    #[test]
    fn sum_is_monoid_while_standard_sum_is_sql() -> VortexResult<()> {
        // The persisted statistic keeps the monoid semantics (zero for all-null) that zone
        // and chunk merging require, while the SQL `sum` applies the null-for-empty rule.
        let mut ctx = array_session().create_execution_ctx();
        let arr = PrimitiveArray::from_option_iter([None::<i32>, None, None]).into_array();
        assert_eq!(
            monoid_sum(&arr, &mut ctx)?
                .as_primitive()
                .typed_value::<i64>(),
            Some(0)
        );
        // The cached monoid statistic must not leak through `sum`'s cache short-circuit.
        assert!(standard_sum(&arr, &mut ctx)?.is_null());
        Ok(())
    }

    #[test]
    fn grouped_sum_fallback_empty_and_all_null_groups() -> VortexResult<()> {
        // Bool elements are rejected by the primitive grouped kernel, forcing the generic
        // per-group fallback: empty and all-null groups have null sums there too.
        let mut ctx = array_session().create_execution_ctx();
        let elements = BoolArray::from_iter([Some(true), Some(true), None, None]).into_array();
        let groups = ListViewArray::try_new(
            elements,
            buffer![0i32, 2, 2].into_array(),
            buffer![2i32, 0, 2].into_array(),
            Validity::NonNullable,
        )?
        .into_array();

        let result = run_grouped_sum(&groups, &DType::Bool(Nullable))?;
        let expected = PrimitiveArray::from_option_iter([Some(2u64), None, None]).into_array();
        assert_arrays_eq!(&result, &expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn sum_state_merge() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut state = StandardSum.empty_partial(&NumericalAggregateOpts::default(), &dtype)?;

        let scalar1 = Scalar::primitive(100i64, Nullable);
        StandardSum.combine_partials(&mut state, scalar1)?;

        let scalar2 = Scalar::primitive(50i64, Nullable);
        StandardSum.combine_partials(&mut state, scalar2)?;

        let result = StandardSum.finalize_scalar(&state)?;
        StandardSum.reset(&mut state);
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(150));
        Ok(())
    }

    // Stats caching test

    #[test]
    fn sum_stats() -> VortexResult<()> {
        let array = ChunkedArray::try_new(
            vec![
                PrimitiveArray::from_iter([1, 1, 1]).into_array(),
                PrimitiveArray::from_iter([2, 2, 2]).into_array(),
            ],
            DType::Primitive(PType::I32, Nullability::NonNullable),
        )
        .vortex_expect("operation should succeed in test");
        let array = array.into_array();
        // compute sum with accumulator to populate stats
        sum_with_accumulator(&array, &Scalar::primitive(2i64, Nullable))?;

        let sum_without_acc = standard_sum(&array, &mut array_session().create_execution_ctx())?;
        assert_eq!(sum_without_acc, Scalar::primitive(9i64, Nullable));
        Ok(())
    }

    // Constant float non-multiply test

    #[test]
    fn sum_constant_float_non_multiply() -> VortexResult<()> {
        let acc = -2048669276050936500000000000f64;
        let array = ConstantArray::new(6.1811675e16f64, 25);
        let result = sum_with_accumulator(&array.into_array(), &Scalar::primitive(acc, Nullable))
            .vortex_expect("operation should succeed in test");
        assert_eq!(
            f64::try_from(&result).vortex_expect("operation should succeed in test"),
            -2048669274505644600000000000f64
        );
        Ok(())
    }

    // Grouped sum tests

    fn run_grouped_sum(groups: &ArrayRef, elem_dtype: &DType) -> VortexResult<ArrayRef> {
        let mut acc = GroupedAccumulator::try_new(
            StandardSum,
            NumericalAggregateOpts::default(),
            elem_dtype.clone(),
        )?;
        let mut ctx = array_session().create_execution_ctx();
        acc.accumulate_list(groups, &mut ctx)?;
        acc.finish()
    }

    #[test]
    fn grouped_sum_fixed_size_list() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let elements =
            PrimitiveArray::new(buffer![1i32, 2, 3, 4, 5, 6], Validity::NonNullable).into_array();
        let groups = FixedSizeListArray::try_new(elements, 3, Validity::NonNullable, 2)?;

        let elem_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let result = run_grouped_sum(&groups.into_array(), &elem_dtype)?;

        let expected = PrimitiveArray::from_option_iter([Some(6i64), Some(15i64)]).into_array();
        assert_arrays_eq!(&result, &expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn grouped_sum_with_null_elements() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let elements =
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), None, Some(5), Some(6)])
                .into_array();
        let groups = FixedSizeListArray::try_new(elements, 3, Validity::NonNullable, 2)?;

        let elem_dtype = DType::Primitive(PType::I32, Nullable);
        let result = run_grouped_sum(&groups.into_array(), &elem_dtype)?;

        let expected = PrimitiveArray::from_option_iter([Some(4i64), Some(11i64)]).into_array();
        assert_arrays_eq!(&result, &expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn grouped_sum_with_null_group() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let elements =
            PrimitiveArray::new(buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9], Validity::NonNullable)
                .into_array();
        let validity = Validity::from_iter([true, false, true]);
        let groups = FixedSizeListArray::try_new(elements, 3, validity, 3)?;

        let elem_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let result = run_grouped_sum(&groups.into_array(), &elem_dtype)?;

        let expected =
            PrimitiveArray::from_option_iter([Some(6i64), None, Some(24i64)]).into_array();
        assert_arrays_eq!(&result, &expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn grouped_sum_all_null_elements_in_group() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let elements =
            PrimitiveArray::from_option_iter([None::<i32>, None, Some(3), Some(4)]).into_array();
        let groups = FixedSizeListArray::try_new(elements, 2, Validity::NonNullable, 2)?;

        let elem_dtype = DType::Primitive(PType::I32, Nullable);
        let result = run_grouped_sum(&groups.into_array(), &elem_dtype)?;

        // The all-null group has a null sum (SQL `SUM` semantics).
        let expected = PrimitiveArray::from_option_iter([None, Some(7i64)]).into_array();
        assert_arrays_eq!(&result, &expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn grouped_sum_bool() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let elements: BoolArray = [true, false, true, true, true, true].into_iter().collect();
        let groups =
            FixedSizeListArray::try_new(elements.into_array(), 3, Validity::NonNullable, 2)?;

        let elem_dtype = DType::Bool(Nullability::NonNullable);
        let result = run_grouped_sum(&groups.into_array(), &elem_dtype)?;

        let expected = PrimitiveArray::from_option_iter([Some(2u64), Some(3u64)]).into_array();
        assert_arrays_eq!(&result, &expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn grouped_sum_finish_resets() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let elem_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut acc = GroupedAccumulator::try_new(
            StandardSum,
            NumericalAggregateOpts::default(),
            elem_dtype,
        )?;

        let elements1 =
            PrimitiveArray::new(buffer![1i32, 2, 3, 4], Validity::NonNullable).into_array();
        let groups1 = FixedSizeListArray::try_new(elements1, 2, Validity::NonNullable, 2)?;
        acc.accumulate_list(&groups1.into_array(), &mut ctx)?;
        let result1 = acc.finish()?;

        let expected1 = PrimitiveArray::from_option_iter([Some(3i64), Some(7i64)]).into_array();
        assert_arrays_eq!(&result1, &expected1, &mut ctx);

        let elements2 = PrimitiveArray::new(buffer![10i32, 20], Validity::NonNullable).into_array();
        let groups2 = FixedSizeListArray::try_new(elements2, 2, Validity::NonNullable, 1)?;
        acc.accumulate_list(&groups2.into_array(), &mut ctx)?;
        let result2 = acc.finish()?;

        let expected2 = PrimitiveArray::from_option_iter([Some(30i64)]).into_array();
        assert_arrays_eq!(&result2, &expected2, &mut ctx);
        Ok(())
    }

    #[test]
    fn grouped_sum_listview_out_of_order_offsets_with_null_group() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let elements =
            PrimitiveArray::new(buffer![100i32, 200, 300], Validity::NonNullable).into_array();
        let offsets = PrimitiveArray::new(buffer![2i32, 0, 1], Validity::NonNullable).into_array();
        let sizes = PrimitiveArray::new(buffer![1i32, 1, 1], Validity::NonNullable).into_array();
        let validity = Validity::from_iter([true, false, true]);
        let groups = ListViewArray::try_new(elements, offsets, sizes, validity)?.into_array();

        let elem_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let result = run_grouped_sum(&groups, &elem_dtype)?;

        // group 0 -> elements[2..3] = 300; group 1 -> null; group 2 -> elements[1..2] = 200.
        let expected =
            PrimitiveArray::from_option_iter([Some(300i64), None, Some(200i64)]).into_array();
        assert_arrays_eq!(&result, &expected, &mut ctx);
        Ok(())
    }

    // Chunked array tests

    #[test]
    fn sum_chunked_floats_with_nulls() -> VortexResult<()> {
        let chunk1 =
            PrimitiveArray::from_option_iter(vec![Some(1.5f64), None, Some(3.2), Some(4.8)]);
        let chunk2 = PrimitiveArray::from_option_iter(vec![Some(2.1f64), Some(5.7), None]);
        let chunk3 = PrimitiveArray::from_option_iter(vec![None, Some(1.0f64), Some(2.5), None]);
        let dtype = chunk1.dtype().clone();
        let chunked = ChunkedArray::try_new(
            vec![
                chunk1.into_array(),
                chunk2.into_array(),
                chunk3.into_array(),
            ],
            dtype,
        )?;

        let result = standard_sum(
            &chunked.into_array(),
            &mut array_session().create_execution_ctx(),
        )?;
        assert_eq!(result.as_primitive().as_::<f64>(), Some(20.8));
        Ok(())
    }

    #[test]
    fn sum_chunked_floats_all_nulls_is_null() -> VortexResult<()> {
        let chunk1 = PrimitiveArray::from_option_iter::<f32, _>(vec![None, None, None]);
        let chunk2 = PrimitiveArray::from_option_iter::<f32, _>(vec![None, None]);
        let dtype = chunk1.dtype().clone();
        let chunked = ChunkedArray::try_new(vec![chunk1.into_array(), chunk2.into_array()], dtype)?;
        let result = standard_sum(
            &chunked.into_array(),
            &mut array_session().create_execution_ctx(),
        )?;
        // SQL `SUM`: no valid values across any chunk yields null.
        assert!(result.is_null());
        Ok(())
    }

    #[test]
    fn sum_chunked_floats_empty_chunks() -> VortexResult<()> {
        let chunk1 = PrimitiveArray::from_option_iter(vec![Some(10.5f64), Some(20.3)]);
        let chunk2 = ConstantArray::new(Scalar::primitive(0f64, Nullable), 0);
        let chunk3 = PrimitiveArray::from_option_iter(vec![Some(5.2f64)]);
        let dtype = chunk1.dtype().clone();
        let chunked = ChunkedArray::try_new(
            vec![
                chunk1.into_array(),
                chunk2.into_array(),
                chunk3.into_array(),
            ],
            dtype,
        )?;

        let result = standard_sum(
            &chunked.into_array(),
            &mut array_session().create_execution_ctx(),
        )?;
        assert_eq!(result.as_primitive().as_::<f64>(), Some(36.0));
        Ok(())
    }

    #[test]
    fn sum_chunked_int_almost_all_null() -> VortexResult<()> {
        let chunk1 = PrimitiveArray::from_option_iter::<u32, _>(vec![Some(1)]);
        let chunk2 = PrimitiveArray::from_option_iter::<u32, _>(vec![None]);
        let dtype = chunk1.dtype().clone();
        let chunked = ChunkedArray::try_new(vec![chunk1.into_array(), chunk2.into_array()], dtype)?;

        let result = standard_sum(
            &chunked.into_array(),
            &mut array_session().create_execution_ctx(),
        )?;
        assert_eq!(result.as_primitive().as_::<u64>(), Some(1));
        Ok(())
    }

    #[test]
    fn sum_chunked_decimals() -> VortexResult<()> {
        let decimal_dtype = DecimalDType::new(10, 2);
        let chunk1 = DecimalArray::new(
            buffer![100i32, 100i32, 100i32, 100i32, 100i32],
            decimal_dtype,
            Validity::AllValid,
        );
        let chunk2 = DecimalArray::new(
            buffer![200i32, 200i32, 200i32],
            decimal_dtype,
            Validity::AllValid,
        );
        let chunk3 = DecimalArray::new(buffer![300i32, 300i32], decimal_dtype, Validity::AllValid);
        let dtype = chunk1.dtype().clone();
        let chunked = ChunkedArray::try_new(
            vec![
                chunk1.into_array(),
                chunk2.into_array(),
                chunk3.into_array(),
            ],
            dtype,
        )?;

        let result = standard_sum(
            &chunked.into_array(),
            &mut array_session().create_execution_ctx(),
        )?;
        let decimal_result = result.as_decimal();
        assert_eq!(
            decimal_result.decimal_value(),
            Some(DecimalValue::I256(i256::from_i128(1700)))
        );
        Ok(())
    }

    #[test]
    fn sum_chunked_decimals_with_nulls() -> VortexResult<()> {
        let decimal_dtype = DecimalDType::new(10, 2);
        let chunk1 = DecimalArray::new(
            buffer![100i32, 100i32, 100i32],
            decimal_dtype,
            Validity::AllValid,
        );
        let chunk2 = DecimalArray::new(
            buffer![0i32, 0i32],
            decimal_dtype,
            Validity::from_iter([false, false]),
        );
        let chunk3 = DecimalArray::new(buffer![200i32, 200i32], decimal_dtype, Validity::AllValid);
        let dtype = chunk1.dtype().clone();
        let chunked = ChunkedArray::try_new(
            vec![
                chunk1.into_array(),
                chunk2.into_array(),
                chunk3.into_array(),
            ],
            dtype,
        )?;

        let result = standard_sum(
            &chunked.into_array(),
            &mut array_session().create_execution_ctx(),
        )?;
        let decimal_result = result.as_decimal();
        assert_eq!(
            decimal_result.decimal_value(),
            Some(DecimalValue::I256(i256::from_i128(700)))
        );
        Ok(())
    }

    #[test]
    fn sum_chunked_decimals_large() -> VortexResult<()> {
        let decimal_dtype = DecimalDType::new(3, 0);
        let chunk1 = ConstantArray::new(
            Scalar::decimal(
                DecimalValue::I16(500),
                decimal_dtype,
                Nullability::NonNullable,
            ),
            1,
        );
        let chunk2 = ConstantArray::new(
            Scalar::decimal(
                DecimalValue::I16(600),
                decimal_dtype,
                Nullability::NonNullable,
            ),
            1,
        );
        let dtype = chunk1.dtype().clone();
        let chunked = ChunkedArray::try_new(vec![chunk1.into_array(), chunk2.into_array()], dtype)?;

        let result = standard_sum(
            &chunked.into_array(),
            &mut array_session().create_execution_ctx(),
        )?;
        let decimal_result = result.as_decimal();
        assert_eq!(
            decimal_result.decimal_value(),
            Some(DecimalValue::I256(i256::from_i128(1100)))
        );
        assert_eq!(
            result.dtype(),
            &DType::Decimal(DecimalDType::new(13, 0), Nullable)
        );
        Ok(())
    }
}
