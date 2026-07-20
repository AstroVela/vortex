// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::Canonical;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::DynGroupedAccumulator;
use crate::aggregate_fn::GroupedAccumulator;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::max::Max;
use crate::aggregate_fn::fns::min::Min;
use crate::arrays::ConstantArray;
use crate::dtype::DType;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;

/// Minimum of the non-null elements in each `List`, `ListView`, or `FixedSizeList` value.
///
/// Null lists, empty lists, and lists without a participating element produce null. Null elements
/// are ignored. With the default [`NumericalAggregateOpts`], float NaNs are also ignored; with
/// [`NumericalAggregateOpts::include_nans`], any NaN poisons its list's result to NaN.
#[derive(Clone)]
pub struct ListMin;

/// Maximum of the non-null elements in each `List`, `ListView`, or `FixedSizeList` value.
///
/// Null lists, empty lists, and lists without a participating element produce null. Null elements
/// are ignored. With the default [`NumericalAggregateOpts`], float NaNs are also ignored; with
/// [`NumericalAggregateOpts::include_nans`], any NaN poisons its list's result to NaN.
#[derive(Clone)]
pub struct ListMax;

impl ScalarFnVTable for ListMin {
    type Options = NumericalAggregateOpts;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.list.min");
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

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            _ => unreachable!("Invalid child index {child_idx} for list_min()"),
        }
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        list_extreme_return_dtype("list_min", &arg_dtypes[0])
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        execute_list_extreme(Min, "list_min", options, args, ctx)
    }

    fn is_null_sensitive(&self, _options: &Self::Options) -> bool {
        false
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        false
    }
}

impl ScalarFnVTable for ListMax {
    type Options = NumericalAggregateOpts;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.list.max");
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

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            _ => unreachable!("Invalid child index {child_idx} for list_max()"),
        }
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        list_extreme_return_dtype("list_max", &arg_dtypes[0])
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        execute_list_extreme(Max, "list_max", options, args, ctx)
    }

    fn is_null_sensitive(&self, _options: &Self::Options) -> bool {
        false
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        false
    }
}

/// Validate the list and element dtypes and derive the nullable scalar output dtype.
///
/// The result is nullable even when both input levels are non-nullable because an empty list has no
/// minimum or maximum.
fn list_extreme_return_dtype(function: &str, input_dtype: &DType) -> VortexResult<DType> {
    let element_dtype = list_element_dtype(function, input_dtype)?;
    Ok(element_dtype.as_nullable())
}

/// Return the comparable element dtype accepted by the list-extrema implementation.
///
/// Nested and extension values are rejected until their aggregate ordering and nested-null
/// semantics are explicitly defined. In particular, an extension's storage order is not
/// necessarily its logical order.
fn list_element_dtype(function: &str, input_dtype: &DType) -> VortexResult<DType> {
    let element_dtype = match input_dtype {
        DType::List(element_dtype, _) | DType::FixedSizeList(element_dtype, ..) => {
            element_dtype.as_ref()
        }
        other => vortex_bail!("{function}() requires List or FixedSizeList, got {other}"),
    };

    if !matches!(
        element_dtype,
        DType::Bool(_)
            | DType::Primitive(..)
            | DType::Decimal(..)
            | DType::Utf8(_)
            | DType::Binary(_)
    ) {
        vortex_bail!("{function}() cannot compare elements of type {element_dtype}")
    }

    Ok(element_dtype.clone())
}

/// Execute a list extremum while preserving the constant-array fast path.
///
/// Non-constant inputs are executed to a columnar representation and delegated to the grouped
/// aggregate machinery. A constant list is reduced once and its scalar result is broadcast to the
/// original row count, avoiding repeated work for both its elements and validity.
fn execute_list_extreme<V: AggregateFnVTable<Options = NumericalAggregateOpts>>(
    aggregate: V,
    function: &str,
    options: &NumericalAggregateOpts,
    args: &dyn ExecutionArgs,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let input = args.get(0)?;
    let element_dtype = list_element_dtype(function, input.dtype())?;

    match input.execute::<Columnar>(ctx)? {
        Columnar::Constant(constant) => {
            // Evaluate a single list and broadcast its extremum rather than expanding all of the
            // constant's repeated elements.
            let one_row = ConstantArray::new(constant.scalar().clone(), 1)
                .into_array()
                .execute::<Canonical>(ctx)?
                .into_array();
            let extreme = list_extreme_impl(aggregate, one_row, element_dtype, options, ctx)?
                .execute_scalar(0, ctx)?;
            Ok(ConstantArray::new(extreme, constant.len()).into_array())
        }
        Columnar::Canonical(canonical) => list_extreme_impl(
            aggregate,
            canonical.into_array(),
            element_dtype,
            options,
            ctx,
        ),
    }
}

/// Treat every outer list value as one group and run the supplied `Min` or `Max` aggregate.
///
/// The grouped accumulator preserves explicit ListView ranges, so repeated, overlapping, gapped,
/// and out-of-order views are reduced according to their logical rows rather than their physical
/// element order.
fn list_extreme_impl<V: AggregateFnVTable<Options = NumericalAggregateOpts>>(
    aggregate: V,
    canonical: ArrayRef,
    element_dtype: DType,
    options: &NumericalAggregateOpts,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let mut accumulator = GroupedAccumulator::try_new(aggregate, *options, element_dtype)?;
    accumulator.accumulate_list(&canonical, ctx)?;
    accumulator.finish()
}

#[cfg(test)]
mod tests;
