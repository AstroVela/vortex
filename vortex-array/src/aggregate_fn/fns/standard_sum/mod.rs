// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod grouped;
pub(crate) use grouped::PrimitiveGroupedStandardSumEncodingKernel;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

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
use crate::aggregate_fn::fns::sum::accumulate_bool;
use crate::aggregate_fn::fns::sum::accumulate_decimal;
use crate::aggregate_fn::fns::sum::accumulate_primitive;
use crate::aggregate_fn::fns::sum::checked_add_i64;
use crate::aggregate_fn::fns::sum::checked_add_u64;
use crate::aggregate_fn::fns::sum::make_zero_state;
use crate::aggregate_fn::fns::sum::multiply_constant;
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
use crate::validity::Validity;

/// Return the sum of an array. The result is null when the array has no valid values or the sum
/// overflows.
///
/// See [`StandardSum`] for details.
pub fn standard_sum(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Scalar> {
    // Short-circuit using cached array statistics. `Stat::Sum` is zero for an
    // array with no valid values instead of null, so we need the null count for a nullable array.
    // When it is unknown, fall through and compute rather than trust the cache.
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

    // Compute using Accumulator<StandardSum>
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

/// The same as [`Sum`](crate::aggregate_fn::fns::sum::Sum), except that arrays with no valid elements yield null.
///
/// Sum aggregates typically follow this behavior. See:
/// - [DuckDB](https://duckdb.org/docs/stable/sql/functions/aggregates.html)
/// - [Arrow](https://docs.rs/arrow/latest/arrow/compute/fn.sum.html)
/// - [DataFusion](https://github.com/apache/datafusion/blob/4153adf2c0f6e317ef476febfdc834208bd46622/datafusion/functions-aggregate/src/sum.rs#L370)
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
        // Partials are `{sum, seen}` structs. A plain scalar is a cached `Stat::Sum` value,
        // which cannot distinguish an empty sum from a zero sum, so it is treated as seen.
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
        // `Stat::Sum` is produced by `Sum` aggregate, so the default NaN-skipping path can
        // consume it directly.
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

        // `seen` is decided by validity alone (NaNs are valid values), so it is tracked here
        // and the summation reuses [`Sum`]'s accumulation kernels unchanged.
        let result = match batch {
            Columnar::Canonical(c) => match c {
                Canonical::Primitive(p) => {
                    partial.seen |= any_valid(p.as_ref().validity()?, p.as_ref().len(), ctx)?;
                    accumulate_primitive(&mut inner, p, ctx, skip_nans)
                }
                Canonical::Bool(b) => {
                    partial.seen |= any_valid(b.as_ref().validity()?, b.as_ref().len(), ctx)?;
                    accumulate_bool(&mut inner, b, ctx)
                }
                Canonical::Decimal(d) => {
                    partial.seen |= any_valid(d.as_ref().validity()?, d.as_ref().len(), ctx)?;
                    accumulate_decimal(&mut inner, d, ctx)
                }
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
        // Entries that saw no valid values finalize to null, while a null `sum`
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

/// Consume a batch's cached `Stat::Sum` instead of scanning it. The cached sum cannot
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

/// Whether a batch contains at least one valid element.
fn any_valid(validity: Validity, len: usize, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
    Ok(validity.execute_mask(len, ctx)?.true_count() > 0)
}

/// The group state for a sum aggregate, containing the accumulated value and configuration
/// needed for reset/result without external context.
pub struct StandardSumPartial {
    return_dtype: DType,
    /// The current accumulated state, or `None` if saturated (checked overflow).
    current: Option<SumState>,
    /// Whether at least one valid value has been accumulated. A sum over zero valid values
    /// finalizes to null rather than zero.
    seen: bool,
    /// Whether NaN values in float inputs are skipped.
    skip_nans: bool,
}

/// The partial dtype for a sum whose result is `sum_dtype`: a `{sum, seen}` struct, where
/// `sum` is the running sum, and is null once saturated by overflow, poisoning merges, and
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
mod tests;
