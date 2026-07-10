// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execute adaptor for `TakeSlices` parent operations.

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

/// Execution kernel for child encodings that can materialize a `TakeSlices` parent directly.
pub trait TakeSlicesKernel: VTable {
    /// Gather contiguous runs from `array`.
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
