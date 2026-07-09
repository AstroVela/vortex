// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reduce and execute adaptors for contiguous-run take operations.
//!
//! `TakeSlicesArray` represents a gather of contiguous child runs: the output is the
//! concatenation of `values[starts[i]..starts[i] + lengths[i]]` for each selector row. Runs may
//! overlap, repeat, and appear in any order. Encodings that know how to serve those runs
//! efficiently implement
//! [`TakeSlicesReduce`] or [`TakeSlicesExecute`].

mod array;
mod rules;
mod vtable;

pub use array::TakeSlicesArrayExt;
pub use array::TakeSlicesData;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
pub use vtable::*;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::dtype::DType;
use crate::dtype::IntegerPType;
use crate::kernel::ExecuteParentKernel;
use crate::legacy_session;
use crate::match_each_unsigned_integer_ptype;
use crate::matcher::Matcher;
use crate::optimizer::rules::ArrayParentReduceRule;

/// Metadata-only implementation hook for taking a sequence of child runs.
pub trait TakeSlicesReduce: VTable {
    /// Take a sequence of contiguous runs from an array without reading value buffers.
    ///
    /// Implementations should return `None` if serving the runs requires value buffer access.
    ///
    /// # Preconditions
    ///
    /// `starts` and `lengths` are guaranteed to be equal-length, non-nullable unsigned integer
    /// arrays. Each run is in bounds for `array`.
    fn take_slices(
        array: ArrayView<'_, Self>,
        starts: &ArrayRef,
        lengths: &ArrayRef,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Execution implementation hook for taking a sequence of child runs.
pub trait TakeSlicesExecute: VTable {
    /// Take a sequence of contiguous runs from an array, potentially reading value buffers.
    ///
    /// # Preconditions
    ///
    /// `starts` and `lengths` are guaranteed to be equal-length, non-nullable unsigned integer
    /// arrays. Each run is in bounds for `array`.
    fn take_slices(
        array: ArrayView<'_, Self>,
        starts: &ArrayRef,
        lengths: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

fn trivial_take_slices(
    child: &ArrayRef,
    starts: &ArrayRef,
    lengths: &ArrayRef,
) -> VortexResult<Option<ArrayRef>> {
    let mut ctx = legacy_execution_ctx();
    trivial_take_slices_with_ctx(child, starts, lengths, &mut ctx)
}

fn trivial_take_slices_with_ctx(
    child: &ArrayRef,
    starts: &ArrayRef,
    lengths: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let slices = selector_slices(child.len(), starts, lengths, ctx)?;
    trivial_take_slices_from_ranges(child, &slices)
}

fn trivial_take_slices_from_ranges(
    child: &ArrayRef,
    slices: &[(usize, usize)],
) -> VortexResult<Option<ArrayRef>> {
    if slices.is_empty() {
        return Ok(Some(Canonical::empty(child.dtype()).into_array()));
    }

    if let [(start, end)] = slices {
        if *start == 0 && *end == child.len() {
            return Ok(Some(child.clone()));
        }
        return child.slice(*start..*end).map(Some);
    }

    Ok(None)
}

pub(super) fn selector_output_len(
    child_len: usize,
    starts: &ArrayRef,
    lengths: &ArrayRef,
) -> VortexResult<usize> {
    let mut ctx = legacy_execution_ctx();
    let slices = selector_slices(child_len, starts, lengths, &mut ctx)?;
    slices.iter().try_fold(0usize, |len, &(start, end)| {
        len.checked_add(end - start)
            .ok_or_else(|| vortex_err!("TakeSlicesArray length overflow"))
    })
}

#[allow(clippy::disallowed_methods)]
fn legacy_execution_ctx() -> ExecutionCtx {
    legacy_session().create_execution_ctx()
}

pub(super) fn selector_slices(
    child_len: usize,
    starts: &ArrayRef,
    lengths: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Vec<(usize, usize)>> {
    check_selector_arrays(starts, lengths)?;
    let starts = starts.clone().execute::<PrimitiveArray>(ctx)?;
    let lengths = lengths.clone().execute::<PrimitiveArray>(ctx)?;
    primitive_selector_slices(child_len, starts.as_view(), lengths.as_view())
}

pub(super) fn check_selector_arrays(starts: &ArrayRef, lengths: &ArrayRef) -> VortexResult<()> {
    check_selector_dtype("starts", starts)?;
    check_selector_dtype("lengths", lengths)?;
    vortex_ensure!(
        starts.len() == lengths.len(),
        "TakeSlicesArray selectors must have equal length, got starts {} and lengths {}",
        starts.len(),
        lengths.len()
    );
    Ok(())
}

fn check_selector_dtype(name: &str, selector: &ArrayRef) -> VortexResult<()> {
    match selector.dtype() {
        DType::Primitive(ptype, nullability) if ptype.is_unsigned_int() => {
            vortex_ensure!(
                !nullability.is_nullable(),
                "TakeSlicesArray {name} must be non-nullable, got {}",
                selector.dtype()
            );
            Ok(())
        }
        other => vortex_bail!(
            "TakeSlicesArray {name} must be a non-nullable unsigned integer, got {other}"
        ),
    }
}

fn primitive_selector_slices(
    child_len: usize,
    starts: ArrayView<'_, Primitive>,
    lengths: ArrayView<'_, Primitive>,
) -> VortexResult<Vec<(usize, usize)>> {
    vortex_ensure!(
        starts.len() == lengths.len(),
        "TakeSlicesArray selectors must have equal length, got starts {} and lengths {}",
        starts.len(),
        lengths.len()
    );
    match_each_unsigned_integer_ptype!(starts.ptype(), |S| {
        match_each_unsigned_integer_ptype!(lengths.ptype(), |L| {
            selector_slices_typed::<S, L>(
                child_len,
                starts.as_slice::<S>(),
                lengths.as_slice::<L>(),
            )
        })
    })
}

fn selector_slices_typed<S: IntegerPType, L: IntegerPType>(
    child_len: usize,
    starts: &[S],
    lengths: &[L],
) -> VortexResult<Vec<(usize, usize)>> {
    let mut slices = Vec::with_capacity(starts.len());
    for (&start, &length) in starts.iter().zip(lengths) {
        let start = selector_to_usize("start", start)?;
        let length = selector_to_usize("length", length)?;
        let end = start.checked_add(length).ok_or_else(|| {
            vortex_err!("TakeSlicesArray run overflow for start {start} and length {length}")
        })?;
        vortex_ensure!(
            end <= child_len,
            "TakeSlicesArray run {start}..{end} exceeds child array length {child_len}"
        );
        if length != 0 {
            slices.push((start, end));
        }
    }
    Ok(slices)
}

fn selector_to_usize<T: IntegerPType>(name: &str, value: T) -> VortexResult<usize> {
    value
        .to_usize()
        .ok_or_else(|| vortex_err!("TakeSlicesArray {name} selector {value} does not fit in usize"))
}

/// Adaptor that wraps a [`TakeSlicesReduce`] impl as an [`ArrayParentReduceRule`].
#[derive(Default, Debug)]
pub struct TakeSlicesReduceAdaptor<V>(pub V);

impl<V> ArrayParentReduceRule<V> for TakeSlicesReduceAdaptor<V>
where
    V: TakeSlicesReduce,
{
    type Parent = TakeSlices;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: <Self::Parent as Matcher>::Match<'_>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        if child_idx != 0 {
            return Ok(None);
        }
        if let Some(result) = trivial_take_slices(array.array(), parent.starts(), parent.lengths())?
        {
            return Ok(Some(result));
        }
        <V as TakeSlicesReduce>::take_slices(array, parent.starts(), parent.lengths())
    }
}

/// Adaptor that wraps a [`TakeSlicesExecute`] impl as an [`ExecuteParentKernel`].
#[derive(Default, Debug)]
pub struct TakeSlicesExecuteAdaptor<V>(pub V);

impl<V> ExecuteParentKernel<V> for TakeSlicesExecuteAdaptor<V>
where
    V: TakeSlicesExecute,
{
    type Parent = TakeSlices;

    fn execute_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: <Self::Parent as Matcher>::Match<'_>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if child_idx != 0 {
            return Ok(None);
        }
        if let Some(result) =
            trivial_take_slices_with_ctx(array.array(), parent.starts(), parent.lengths(), ctx)?
        {
            return Ok(Some(result));
        }
        <V as TakeSlicesExecute>::take_slices(array, parent.starts(), parent.lengths(), ctx)
    }
}

#[cfg(test)]
mod tests;
