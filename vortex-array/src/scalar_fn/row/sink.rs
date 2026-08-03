// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The column builders a row function can write its output into.

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::dtype::DType;

/// A column allocated once per batch that a row closure writes into, one row at a time.
///
/// This is the second of the two ways a [`RowFn`](crate::scalar_fn::RowFn) produces output, and it
/// exists for what the first cannot express. An [`OutputElement`](crate::scalar_fn::OutputElement) is
/// one owned value per row whose dtype is a property of its Rust type. That rules out an output whose
/// *width* is runtime data, and makes builder-backed output such as a transformed string allocate an
/// owned value per row. A sink resolves both: it is created knowing the output dtype and owns the
/// batch-wide builder while handing out a place to write rather than taking a value back.
///
/// Two properties of the contract are worth stating, since both are load-bearing:
///
/// - **The row loop, not the closure, holds the sink.** [`row`](Self::row) is called by the framework
///   and its result passed in, so a writing closure stays [`Fn`] and captures nothing mutable.
///   Relaxing the row closure to `FnMut` instead was measured at 8 to 11%, because a captured `&mut`
///   inhibits vectorization of the loop.
/// - **[`sink_dtype`](Self::sink_dtype) sees the input dtypes**, unlike
///   [`OutputElement::element_dtype`](crate::scalar_fn::OutputElement::element_dtype), which takes
///   none. That is the whole reason a runtime-shaped output fits here: the width comes out of the
///   arguments.
///
/// Rows arrive in order, `0..row_count`, exactly once each.
pub trait OutputSink: 'static + Sized {
    /// A loop-local view of all output rows.
    ///
    /// Borrowed once before execution so the sink's buffer descriptor and shape become loop
    /// invariants rather than being re-read through `&mut Self` for every row.
    type Rows<'a>
    where
        Self: 'a;

    /// The place a row closure writes one row through, borrowed from the sink.
    type Row<'a>
    where
        Self: 'a;

    /// The dtype of the column this sink builds, given the function's input dtypes.
    ///
    /// Must be non-nullable: nullability is derived from the inputs by the lifting, which
    /// widens the result and masks the null rows itself.
    fn sink_dtype(args: &[DType]) -> VortexResult<DType>;

    /// Allocate a sink for `rows` rows of `dtype`, which is this sink's own
    /// [`sink_dtype`](Self::sink_dtype). Called once per batch.
    fn with_capacity(rows: usize, dtype: &DType) -> VortexResult<Self>;

    /// Borrow all output rows for the hot loop.
    fn rows(&mut self) -> Self::Rows<'_>;

    /// Whether every index in `0..row_count` is addressable through [`row`](Self::row).
    ///
    /// Called once before the hot loop. Besides validating the sink contract, this gives the
    /// optimizer the output bounds it needs to remove the bounds check hidden in each row accessor.
    fn row_count_matches(rows: &Self::Rows<'_>, row_count: usize) -> bool;

    /// Hand out the place to write row `index`. Must be `O(1)`: it is called in the row loop.
    fn row<'a>(rows: &'a mut Self::Rows<'_>, index: usize) -> Self::Row<'a>;

    /// Finish into the built column, whose dtype **must** be this sink's
    /// [`sink_dtype`](Self::sink_dtype). Called once per batch.
    fn finish(self) -> VortexResult<ArrayRef>;
}
