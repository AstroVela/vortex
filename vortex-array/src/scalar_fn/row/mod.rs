// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Defining scalar functions one row at a time.
//!
//! This is the most derived of the three scalar function traits, and the right default for a kernel
//! that has to read every row anyway. See [choosing a trait](crate::scalar_fn#choosing-a-trait) for
//! when to drop to [`StrictScalarFnVTable`](crate::scalar_fn::StrictScalarFnVTable) instead.
//!
//! [`RowFn`] asks for two things: a witness argument tuple and return type, and a
//! [`dispatch`](RowFn::dispatch) that picks the concrete element types for a batch and visits the
//! framework with a row closure. Everything structural (arity, dtype checks, null handling,
//! fallibility, constants, validity, array serde) is derived from the witnesses and from whatever
//! `dispatch` visits.
//!
//! When the element types are fixed, `dispatch` is a single visit at those types. When one function
//! ID has to cover several (`l2_norm` accepts `f16`, `f32` and `f64` columns), `dispatch` matches on
//! the input dtypes and visits at the chosen width. The witnesses name one representative choice,
//! and the framework checks at compile time that every visit agrees with them.
//!
//! [`RowFn`] does not say how a row is *stored*, which is the element's job: `vortex-tensor` adds a
//! `TensorRow<T>` [`InputElement`] and writes ordinary kernels over it.
//!
//! Output comes in two forms, and a dispatch picks one per visit. [`RowVisitor::visit`] takes a
//! closure that *returns* an [`OutputElement`] per row, which is the common case.
//! [`RowVisitor::visit_into`] takes one that *writes* its row into an [`OutputSink`], which carries
//! what an owned per-row value cannot: a runtime-shaped row such as a tensor, or bytes appended to a
//! buffer shared by the whole batch.
//!
//! [`RowVisitor::visit_prepared`] adds a per-batch prepare step to the returning form: it is handed
//! the element value of every argument whose operand is constant for the batch, and whatever it
//! returns reaches every row by shared reference, so work that depends only on a constant argument
//! runs once per batch instead of once per row.
//!
//! Null handling is derived and executed by the strict lifting, never by the row closure, which
//! only ever computes rows valid in every argument. A batch with a mixed validity mask executes by
//! one of two strategies, selected per batch: *branch-and-skip* (decode the unfiltered columns
//! null-tolerantly via [`InputElement::decode_null_tolerant`], compute only the valid rows a word
//! of the mask at a time, mask the result) whenever it can, and *filter* (shrink every input to
//! the surviving rows, compute, scatter back) when an argument has no null-tolerant decode for its
//! array or when a per-row decode makes filtering cheaper at sparse validity. Authors do nothing;
//! an element whose decode does expensive per-row work opts its sparse batches back into filtering
//! by setting [`InputElement::DECODE_SHRINKS_WHEN_FILTERED`]. Sink dispatches
//! ([`RowVisitor::visit_into`]) always use the dense or filter paths: a sink has no notion of a
//! skipped row.

mod element;
pub use element::ArgColumn;
pub use element::Bytes;
pub use element::BytesColumn;
pub use element::BytesLen;
pub use element::ElementTuple;
pub use element::InputElement;
pub use element::OutputElement;
#[cfg(any(test, feature = "_test-harness"))]
pub use element::assert_element_conforms;

mod result;
pub use result::ApplyResult;
pub use result::RowResult;
pub use result::SinkResult;

mod sink;
pub use sink::OutputSink;

mod execute;

mod row_fn;
pub use row_fn::RowFn;
pub use row_fn::RowVisitor;

#[cfg(test)]
mod tests;
