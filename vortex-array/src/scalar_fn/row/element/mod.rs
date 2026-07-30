// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The element types a row function can read and produce.
//!
//! Both traits are open, and this module holds one file per type family, so covering a new one is a
//! sibling file and every row function gains it. The families are not confined to this crate:
//! `vortex-tensor`'s `TensorRow` drills through an extension wrapper into its storage.
//!
//! The two directions are deliberately asymmetric. [`InputElement::Elem`] is a GAT, so an input row
//! can borrow out of the decoded column, while an [`OutputElement`] is one owned value per row. When
//! that owned value is the wrong shape for an output, the row function writes into an
//! [`OutputSink`](crate::scalar_fn::OutputSink) instead. See
//! [choosing a trait](crate::scalar_fn#choosing-a-trait) for which to reach for.

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;

mod bool;

mod bytes;
pub use bytes::Bytes;
pub use bytes::BytesColumn;
pub use bytes::BytesLen;

#[cfg(any(test, feature = "_test-harness"))]
mod conformance;
#[cfg(any(test, feature = "_test-harness"))]
pub use conformance::assert_element_conforms;

mod primitive;

mod tuple;
pub use tuple::ArgColumn;
pub use tuple::ElementTuple;

/// An element type that can be read row-wise out of an input column.
pub trait InputElement: 'static {
    /// The decoded column representation supporting `O(1)` row access.
    type Column;

    /// The borrowed element value handed to the row closure a [`RowFn`](crate::scalar_fn::RowFn)
    /// visits with.
    type Elem<'a>;

    /// Whether [`get`](Self::get) may be called for a row that is null in the input.
    ///
    /// Arrays only guarantee their contents for *valid* rows, so this is `false` for any element
    /// that follows an offset or pointer stored in the array: behind a null row that value is
    /// arbitrary and may not address anything. Reading a whole value out of a flat buffer is `true`,
    /// since the value is garbage but the read cannot fault.
    ///
    /// [`NullHandling::Dense`](crate::scalar_fn::NullHandling::Dense) requires this of every
    /// argument, and the row layers reject the combination when it does not hold.
    const DENSE_SAFE: bool;

    /// Whether [`decode`](Self::decode) can fail on *legal* input data.
    ///
    /// `false` for an element read straight out of a buffer: decoding can still fail for
    /// infrastructural reasons (IO, allocation), but never because of the values. `true` for an
    /// element that parses its bytes, since a malformed WKB geometry in a *valid* row is a domain
    /// error, which makes a function over that element
    /// [fallible](crate::scalar_fn::ScalarFnVTable::is_fallible) however infallible its own row
    /// computation is.
    const DECODE_FALLIBLE: bool;

    /// Validate that `dtype` is an acceptable input column dtype for this element type.
    fn validate(dtype: &DType) -> VortexResult<()>;

    /// Decode `array` into its column representation. Called once per batch.
    ///
    /// This is where every per-batch cost belongs: resolving the dtype, downcasting the buffer,
    /// checking the ptype, and anything else that does not vary by row. [`Column`](Self::Column) is
    /// the type to widen if that means carrying more, since it is chosen by the element.
    fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Column>;

    /// Read the element at `index`, the one function called once per row.
    ///
    /// `O(1)` is necessary but **not sufficient**: this must not repeat work that is constant across
    /// the batch, however cheap that work looks per call. An `O(1)` ptype check and buffer downcast
    /// per row cost `l2_norm` 2x at width 2, invisible in the call because it read like a getter. Do
    /// that work in [`decode`](Self::decode) and leave this an offset computation.
    fn get(column: &Self::Column, index: usize) -> Self::Elem<'_>;
}

/// An element type that a row computation can produce, buildable into an all-valid column.
pub trait OutputElement: 'static + Sized {
    /// The dtype of columns built from this element type. Must be non-nullable: nullability is
    /// derived from the inputs by the strict lifting.
    ///
    /// Taking no arguments confines an element's dtype to a property of its Rust type, so an output
    /// whose dtype depends on runtime data (a tensor, whose dtype carries its shape) cannot be an
    /// element. Such an output uses an [`OutputSink`](crate::scalar_fn::OutputSink), whose
    /// [`sink_dtype`](crate::scalar_fn::OutputSink::sink_dtype) does see the input dtypes.
    fn element_dtype() -> DType;

    /// Build a column from one value per row. Called once per batch.
    fn build(values: Vec<Self>) -> ArrayRef;
}
