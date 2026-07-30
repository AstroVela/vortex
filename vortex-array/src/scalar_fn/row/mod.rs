// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Defining scalar functions one row at a time.
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

mod element;
pub use element::ApplyResult;
pub use element::ArgColumn;
pub use element::Bytes;
pub use element::BytesColumn;
pub use element::BytesLen;
pub use element::ElementTuple;
pub use element::InputElement;
pub use element::OutputElement;

mod execute;

mod row_fn;
pub use row_fn::RowFn;
pub use row_fn::RowVisitor;

#[cfg(test)]
mod tests;
