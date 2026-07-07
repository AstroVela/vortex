// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reduce and execute adaptors for ordered slice-take operations.
//!
//! `TakeSlicesArray` represents concatenating an ordered list of non-empty child ranges. The
//! ranges preserve caller order and may overlap. Encodings that know how to serve those ranges
//! efficiently implement [`TakeSlicesReduce`] or [`TakeSlicesExecute`].

mod array;
mod rules;
mod vtable;

pub use array::TakeSlicesArrayExt;
pub use array::TakeSlicesData;
use vortex_error::VortexResult;
pub use vtable::*;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::kernel::ExecuteParentKernel;
use crate::matcher::Matcher;
use crate::optimizer::rules::ArrayParentReduceRule;

/// Metadata-only implementation hook for taking ordered child ranges.
pub trait TakeSlicesReduce: VTable {
    /// Take ordered slices from an array without reading buffers.
    ///
    /// Implementations should return `None` if serving the ranges requires buffer access.
    ///
    /// # Preconditions
    ///
    /// `slices` is guaranteed to contain only non-empty ranges in bounds for `array`.
    fn take_slices(
        array: ArrayView<'_, Self>,
        slices: &[(usize, usize)],
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Execution implementation hook for taking ordered child ranges.
pub trait TakeSlicesExecute: VTable {
    /// Take ordered slices from an array, potentially reading buffers.
    ///
    /// # Preconditions
    ///
    /// `slices` is guaranteed to contain only non-empty ranges in bounds for `array`.
    fn take_slices(
        array: ArrayView<'_, Self>,
        slices: &[(usize, usize)],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

fn trivial_take_slices(
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
        assert_eq!(child_idx, 0);
        if let Some(result) = trivial_take_slices(array.array(), parent.slices())? {
            return Ok(Some(result));
        }
        <V as TakeSlicesReduce>::take_slices(array, parent.slices())
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
        assert_eq!(child_idx, 0);
        if let Some(result) = trivial_take_slices(array.array(), parent.slices())? {
            return Ok(Some(result));
        }
        <V as TakeSlicesExecute>::take_slices(array, parent.slices(), ctx)
    }
}

#[cfg(test)]
mod tests;
