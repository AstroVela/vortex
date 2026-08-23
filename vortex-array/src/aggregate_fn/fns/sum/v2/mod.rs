// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(test)]
mod tests;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use super::Sum;
use super::SumState;
use super::bool::accumulate_bool;
use super::checked_add_i64;
use super::checked_add_u64;
use super::constant::multiply_constant;
use super::decimal::accumulate_decimal;
use super::make_zero_state;
use super::primitive::accumulate_primitive;
use crate::ArrayRef;
use crate::Canonical;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::DynAccumulator;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::FieldName;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;
use crate::dtype::StructFields;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProviderExt;
use crate::scalar::Scalar;

/// Return the sum of an array using SQL-style null-on-empty semantics.
///
/// See [`SumV2`] for details.
pub fn sum_v2(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Scalar> {
    let mut acc = Accumulator::try_new(
        SumV2,
        NumericalAggregateOpts::default(),
        array.dtype().clone(),
    )?;
    acc.accumulate(array, ctx)?;
    acc.finish()
}

/// Sum an array with SQL-style null-on-empty semantics.
///
/// Unlike [`Sum`], which sums an empty or all-null input to zero, this aggregate returns null
/// when no valid values are observed:
///
/// - No rows, or all values null: null.
/// - Valid values summing to zero: `0`.
/// - Integer or decimal overflow: null.
/// - All NaNs with the default `skip_nans`: `0`, since NaNs count as valid values even when
///   their contribution is skipped.
/// - Any NaN with `include_nans`: NaN.
///
/// NaN handling for float inputs is controlled by [`NumericalAggregateOpts`], exactly as for
/// [`Sum`].
#[derive(Clone, Debug)]
pub struct SumV2;

/// Field name of the accumulated sum in the partial struct.
const SUM_FIELD: &str = "sum";
/// Field name of the empty flag in the partial struct.
const IS_EMPTY_FIELD: &str = "is_empty";

impl AggregateFnVTable for SumV2 {
    type Options = NumericalAggregateOpts;
    type Partial = SumV2Partial;

    fn id(&self) -> AggregateFnId {
        static ID: CachedId = CachedId::new("vortex.sum_v2");
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

    fn return_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        // The sum value dtypes match `Sum` exactly; only the empty-input semantics differ.
        Sum.return_dtype(options, input_dtype)
    }

    fn partial_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        let sum_dtype = self.return_dtype(options, input_dtype)?;
        Some(DType::Struct(
            StructFields::new(
                FieldNames::from_iter([
                    FieldName::from(SUM_FIELD),
                    FieldName::from(IS_EMPTY_FIELD),
                ]),
                vec![
                    sum_dtype.as_nonnullable(),
                    DType::Bool(Nullability::NonNullable),
                ],
            ),
            // The outer struct is nullable: a null partial marks a saturated (overflowed) sum
            // or an invalid group. An empty partial is instead flagged through `is_empty`, so
            // that it stays a merge identity while overflow stays absorbing.
            Nullability::Nullable,
        ))
    }

    fn empty_partial(
        &self,
        options: &Self::Options,
        input_dtype: &DType,
    ) -> VortexResult<Self::Partial> {
        let sum_dtype = self
            .return_dtype(options, input_dtype)
            .ok_or_else(|| vortex_err!("Unsupported sum dtype: {}", input_dtype))?;
        let partial_dtype = self
            .partial_dtype(options, input_dtype)
            .ok_or_else(|| vortex_err!("Unsupported sum dtype: {}", input_dtype))?;
        let initial = make_zero_state(&sum_dtype);

        Ok(SumV2Partial {
            sum_dtype,
            partial_dtype,
            current: Some(initial),
            is_empty: true,
            skip_nans: options.skip_nans,
        })
    }

    fn combine_partials(&self, partial: &mut Self::Partial, other: Scalar) -> VortexResult<()> {
        if other.is_null() {
            // A null partial means the sub-accumulator saturated (overflow).
            partial.current = None;
            return Ok(());
        }
        let other = other.as_struct();
        let is_empty = other
            .field(IS_EMPTY_FIELD)
            .ok_or_else(|| vortex_err!("SumV2 partial missing `{}` field", IS_EMPTY_FIELD))?
            .as_bool()
            .value()
            .vortex_expect("non-null partial has a non-null is_empty field");
        if is_empty {
            // An empty partial is a merge identity.
            return Ok(());
        }
        let value = other
            .field(SUM_FIELD)
            .ok_or_else(|| vortex_err!("SumV2 partial missing `{}` field", SUM_FIELD))?;
        partial.is_empty = false;
        add_value(partial, &value);
        Ok(())
    }

    fn to_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        let Some(current) = &partial.current else {
            return Ok(Scalar::null(partial.partial_dtype.clone()));
        };
        let sum = sum_state_scalar(current, &partial.sum_dtype, Nullability::NonNullable);
        Ok(Scalar::struct_(
            partial.partial_dtype.clone(),
            vec![
                sum,
                Scalar::bool(partial.is_empty, Nullability::NonNullable),
            ],
        ))
    }

    fn reset(&self, partial: &mut Self::Partial) {
        partial.current = Some(make_zero_state(&partial.sum_dtype));
        partial.is_empty = true;
    }

    #[inline]
    fn is_saturated(&self, partial: &Self::Partial) -> bool {
        match partial.current.as_ref() {
            None => true,
            // A NaN sum implies a valid value was observed, so the NaN result is final.
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
        // NaN-aware shortcircuit for NaN-including float sums: any NaN poisons the sum, and a
        // NaN is a valid value, so the sum is also non-empty. Unlike `Sum`, the NaN-free
        // cached-sum shortcut is unsound here: a cached batch sum alone cannot prove the batch
        // contains a valid value.
        if partial.skip_nans || !matches!(partial.current, Some(SumState::Float(_))) {
            return Ok(false);
        }
        if let Precision::Exact(nan_count) = batch.statistics().get_as::<u64>(Stat::NaNCount)
            && nan_count > 0
        {
            partial.is_empty = false;
            if let Some(SumState::Float(acc)) = partial.current.as_mut() {
                *acc = f64::NAN;
            }
            return Ok(true);
        }
        Ok(false)
    }

    fn accumulate(
        &self,
        partial: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        // Constants compute scalar * len and fold the product into the state.
        if let Columnar::Constant(c) = batch {
            if c.is_empty() || c.scalar().is_null() {
                // No valid values: the state stays empty.
                return Ok(());
            }
            partial.is_empty = false;
            // NaN constants are valid values but contribute nothing when skipping NaNs.
            if partial.skip_nans && c.scalar().as_primitive_opt().is_some_and(|p| p.is_nan()) {
                return Ok(());
            }
            if let Some(product) = multiply_constant(c.scalar(), c.len(), &partial.sum_dtype)? {
                if product.is_null() {
                    // The product overflowed.
                    partial.current = None;
                } else {
                    add_value(partial, &product);
                }
            }
            return Ok(());
        }

        // Any valid element makes the sum non-empty, including skipped NaNs.
        if has_valid_values(batch, ctx)? {
            partial.is_empty = false;
        }

        let skip_nans = partial.skip_nans;
        let mut inner = match partial.current.take() {
            Some(inner) => inner,
            None => return Ok(()),
        };

        let result = match batch {
            Columnar::Canonical(c) => match c {
                Canonical::Primitive(p) => accumulate_primitive(&mut inner, p, ctx, skip_nans),
                Canonical::Bool(b) => accumulate_bool(&mut inner, b, ctx),
                Canonical::Decimal(d) => accumulate_decimal(&mut inner, d, ctx),
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

    fn finalize(&self, states: ArrayRef) -> VortexResult<ArrayRef> {
        // A null partial (overflow or an invalid group) projects to a null sum directly; an
        // empty partial is masked out to null.
        let sums = states.get_item(SUM_FIELD)?;
        let non_empty = states.get_item(IS_EMPTY_FIELD)?.fill_null(true)?.not()?;
        sums.mask(non_empty)
    }

    fn finalize_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        let Some(current) = &partial.current else {
            return Ok(Scalar::null(partial.sum_dtype.clone()));
        };
        if partial.is_empty {
            return Ok(Scalar::null(partial.sum_dtype.clone()));
        }
        Ok(sum_state_scalar(
            current,
            &partial.sum_dtype,
            Nullability::Nullable,
        ))
    }
}

/// The accumulator state for a [`SumV2`] aggregate.
pub struct SumV2Partial {
    /// The nullable dtype of the final sum, which is also the dtype of the non-nullable `sum`
    /// field of the partial struct.
    sum_dtype: DType,
    /// The struct dtype of the partial state.
    partial_dtype: DType,
    /// The current accumulated state, or `None` if saturated (checked overflow).
    current: Option<SumState>,
    /// Whether no valid values have been observed yet.
    is_empty: bool,
    /// Whether NaN values in float inputs are skipped.
    skip_nans: bool,
}

/// Convert the accumulated state into a sum scalar of the given nullability.
fn sum_state_scalar(state: &SumState, sum_dtype: &DType, nullability: Nullability) -> Scalar {
    match state {
        SumState::Unsigned(v) => Scalar::primitive(*v, nullability),
        SumState::Signed(v) => Scalar::primitive(*v, nullability),
        SumState::Float(v) => Scalar::primitive(*v, nullability),
        SumState::Decimal { value, .. } => {
            let decimal_dtype = *sum_dtype
                .as_decimal_opt()
                .vortex_expect("sum dtype must be decimal");
            Scalar::decimal(*value, decimal_dtype, nullability)
        }
    }
}

/// Add a non-null sum value into the partial state, saturating on checked overflow.
fn add_value(partial: &mut SumV2Partial, value: &Scalar) {
    let Some(ref mut inner) = partial.current else {
        return;
    };
    let saturated = match inner {
        SumState::Unsigned(acc) => {
            let val = value
                .as_primitive()
                .typed_value::<u64>()
                .vortex_expect("checked non-null");
            checked_add_u64(acc, val)
        }
        SumState::Signed(acc) => {
            let val = value
                .as_primitive()
                .typed_value::<i64>()
                .vortex_expect("checked non-null");
            checked_add_i64(acc, val)
        }
        SumState::Float(acc) => {
            let val = value
                .as_primitive()
                .typed_value::<f64>()
                .vortex_expect("checked non-null");
            *acc += val;
            false
        }
        SumState::Decimal { value: acc, dtype } => {
            let val = value
                .as_decimal()
                .decimal_value()
                .vortex_expect("checked non-null");
            match acc.checked_add(&val) {
                Some(r) => {
                    *acc = r;
                    !acc.fits_in_precision(*dtype)
                }
                None => true,
            }
        }
    };
    if saturated {
        partial.current = None;
    }
}

/// Whether the canonical batch contains at least one valid element.
fn has_valid_values(batch: &Columnar, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
    let array = match batch {
        Columnar::Canonical(c) => match c {
            Canonical::Primitive(p) => p.as_ref(),
            Canonical::Bool(b) => b.as_ref(),
            Canonical::Decimal(d) => d.as_ref(),
            _ => vortex_bail!("Unsupported canonical type for sum: {}", batch.dtype()),
        },
        Columnar::Constant(_) => unreachable!("constants are handled before dispatch"),
    };
    let mask = array.validity()?.execute_mask(array.len(), ctx)?;
    Ok(mask.true_count() > 0)
}
