// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reduce and execute adaptors for `TakeSlices` parent operations.

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::TakeSlices;
use crate::arrays::take_slices::TakeSlicesArrayExt;
use crate::arrays::take_slices::array::CHILD_SLOT;
use crate::kernel::ExecuteParentKernel;
use crate::matcher::Matcher;
use crate::optimizer::rules::ArrayParentReduceRule;

/// Metadata-only rewrite for child encodings that can push a `TakeSlices` parent through
/// themselves without reading buffers or executing child arrays.
pub trait TakeSlicesReduce: VTable {
    /// Rewrite a contiguous-run gather from `array` to an equivalent array.
    ///
    /// Implementations must not inspect the values of `starts` or `lengths`; range-value errors
    /// remain deferred to execution of the rewritten child `TakeSlicesArray`s.
    fn take_slices(
        array: ArrayView<'_, Self>,
        starts: &ArrayRef,
        lengths: &ArrayRef,
        output_len: usize,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Adapter that wraps a [`TakeSlicesReduce`] impl as an [`ArrayParentReduceRule`].
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
        if child_idx != CHILD_SLOT {
            return Ok(None);
        }

        <V as TakeSlicesReduce>::take_slices(array, parent.starts(), parent.lengths(), parent.len())
    }
}

/// Execution kernel for child encodings that can handle a `TakeSlices` parent directly.
///
/// Implementations may either materialize the gathered values or return another equivalent
/// encoding that makes progress, such as a canonical nested array whose child is another
/// `TakeSlicesArray`. The executor will continue executing the returned array as needed.
pub trait TakeSlicesKernel: VTable {
    /// Gather contiguous runs from `array`, or rewrite that gather to an equivalent array.
    ///
    /// `starts` and `lengths` are non-nullable unsigned integer arrays of equal length. `output_len`
    /// is the declared length of the `TakeSlices` parent and must match the sum of selected lengths.
    fn take_slices(
        array: ArrayView<'_, Self>,
        starts: &ArrayRef,
        lengths: &ArrayRef,
        output_len: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Adapter that exposes a [`TakeSlicesKernel`] implementation as an [`ExecuteParentKernel`].
#[derive(Default, Debug)]
pub struct TakeSlicesExecuteAdaptor<V>(pub V);

impl<V> ExecuteParentKernel<V> for TakeSlicesExecuteAdaptor<V>
where
    V: TakeSlicesKernel,
{
    type Parent = TakeSlices;

    fn execute_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: <Self::Parent as Matcher>::Match<'_>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if child_idx != CHILD_SLOT {
            return Ok(None);
        }

        <V as TakeSlicesKernel>::take_slices(
            array,
            parent.starts(),
            parent.lengths(),
            parent.len(),
            ctx,
        )
    }
}
