// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! What a row closure may return.
//!
//! A row function produces its output one of two ways, and each has its own return shape: a closure
//! that *returns* a value per row returns an [`ApplyResult`], and one that *writes* into an
//! [`OutputSink`](crate::scalar_fn::OutputSink) returns a [`SinkResult`]. Both come in an infallible
//! and a [`VortexResult`] form.
//!
//! [`RowResult`] is what the two share, and exists because it is all the framework can read *before*
//! it knows which of the two a dispatch will use. See [`RowFn::RetWitness`](crate::scalar_fn::RowFn::
//! RetWitness) for why that has to be readable up front.

use vortex_error::VortexResult;

use crate::scalar_fn::OutputElement;

/// The one fact the framework needs from a row closure's return type regardless of which visit it
/// belongs to.
///
/// Both [`ApplyResult`] and [`SinkResult`] refine this. Keeping it separate is what lets
/// [`RowFn::RetWitness`](crate::scalar_fn::RowFn::RetWitness) name either shape: the witness is read
/// before `dispatch` runs, and at that point the framework cannot know whether the output will be
/// returned or written.
pub trait RowResult: 'static {
    /// Whether this return type can carry an error.
    const FALLIBLE: bool;
}

/// What a row computation that *returns* its value may produce: an [`OutputElement`] directly, or a
/// [`VortexResult`] of one.
///
/// Implementing it for both forms is what lets one row function trait serve infallible and fallible
/// kernels without a second trait or a wrapper. A kernel returning `f64` and one returning
/// `VortexResult<f64>` agree on [`Out`](Self::Out) and differ only in
/// [`FALLIBLE`](RowResult::FALLIBLE), which is what the framework reads.
pub trait ApplyResult: RowResult {
    /// The element this computation produces.
    type Out: OutputElement;

    /// Convert into a result, so one code path can handle both forms.
    fn into_result(self) -> VortexResult<Self::Out>;
}

/// What a row computation that *writes* into an [`OutputSink`](crate::scalar_fn::OutputSink) may
/// produce: nothing, or a [`VortexResult`] of nothing.
///
/// The value is already in the sink by the time the closure returns, so the only thing left to report
/// is failure.
pub trait SinkResult: RowResult {
    /// Convert into a result, so one code path can handle both forms.
    fn into_result(self) -> VortexResult<()>;
}

impl<T: OutputElement> RowResult for T {
    const FALLIBLE: bool = false;
}

impl<T: OutputElement> RowResult for VortexResult<T> {
    const FALLIBLE: bool = true;
}

impl<T: OutputElement> ApplyResult for T {
    type Out = T;

    fn into_result(self) -> VortexResult<T> {
        Ok(self)
    }
}

impl<T: OutputElement> ApplyResult for VortexResult<T> {
    type Out = T;

    fn into_result(self) -> VortexResult<T> {
        self
    }
}

// `()` is not an `OutputElement`, and cannot become one outside this crate, so these do not overlap
// the blanket impls above.
impl RowResult for () {
    const FALLIBLE: bool = false;
}

impl RowResult for VortexResult<()> {
    const FALLIBLE: bool = true;
}

impl SinkResult for () {
    fn into_result(self) -> VortexResult<()> {
        Ok(())
    }
}

impl SinkResult for VortexResult<()> {
    fn into_result(self) -> VortexResult<()> {
        self
    }
}
