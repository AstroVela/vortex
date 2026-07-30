// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scalar function vtable machinery.
//!
//! This module contains the [`ScalarFnVTable`] trait and all built-in scalar function
//! implementations. Expressions ([`crate::expr::Expression`]) reference scalar functions
//! at each node.
//!
//! # Choosing a trait
//!
//! Three traits reach this vtable, each deriving more of it than the last. Implement the most
//! derived one that the function fits.
//!
//! [`RowFn`] is for a kernel whose value at a row is determined by that row alone, and which has to
//! read every row anyway: `vortex.byte_length`, `vortex.tensor.l2_norm`, `vortex.geo.distance`.
//! Name the element types and write the row closure, and the rest is derived, including which rows
//! get visited.
//!
//! Its *input* side is open. [`InputElement::Elem`] is a GAT, so an element can hand the closure
//! borrowed variable-length data ([`Bytes`] yields `&[u8]`) or drill through a wrapper
//! (`vortex-tensor`'s `TensorRow` yields a slice of an extension array's storage). Covering a new
//! type family, a list row included, is one impl.
//!
//! The *output* side is what narrows it. [`OutputElement::build`] takes one owned value per row and
//! [`OutputElement::element_dtype`] takes no arguments, which together rule out three things:
//!
//! - **A result that borrows from an input.** A row closure returns an [`ApplyResult`], which is
//!   `'static`, so a function whose output is a *slice* of its input cannot avoid copying it.
//!   Trimming strings is the example: the ideal kernel keeps the input's data buffer and writes
//!   new views over it, copying no bytes, which only a columnar kernel can express.
//! - **An output dtype that depends on runtime data.** `element_dtype` is a property of the Rust
//!   type, and a tensor's dtype carries its shape. This one is a signature choice rather than a law,
//!   since the blanket path already holds the input dtypes when it asks; what actually keeps
//!   `vortex.tensor.l2_denorm` columnar is `build` taking one owned value per row, which for a tensor
//!   row means an allocation per row against a kernel that scales the flat buffer in one pass.
//! - **A null result for a non-null row.** Every output element builds an all-valid column, so
//!   `vortex.list.sum` cannot be a row function: a valid empty list sums to null.
//!
//! [`StrictScalarFnVTable`] takes the whole column instead. Besides the three cases above, reach for
//! it when a row loop *could* express the function but would do avoidable work:
//!
//! - **The answer is already an array, or is one value for the whole column.**
//!   `vortex.list.length` hands back a `ListViewArray`'s sizes child, and a single `ConstantArray`
//!   for a `FixedSizeListArray`. A row loop would rebuild that one `u64` at a time, even given a
//!   list-length element in the style of [`BytesLen`].
//! - **A row is not the natural unit of work.** `vortex.not` is one `!` per 64-bit word, in place
//!   when the bit buffer is unshared, against 64 loop iterations and 64 bit writes.
//!
//! [`ScalarFnVTable`] itself is for a function that is not unconditionally strict, such as Kleene
//! logic, or one whose strictness depends on its options.

use vortex_session::registry::Id;

use crate::scalar_fn::fns::byte_length::ByteLength;
use crate::scalar_fn::fns::ext_storage::ExtStorage;
use crate::scalar_fn::fns::get_item::GetItem;
use crate::scalar_fn::fns::literal::Literal;

mod vtable;
pub use vtable::*;

mod array_metadata;
pub use array_metadata::*;

mod plugin;
pub use plugin::*;

mod foreign;
pub use foreign::*;

mod typed;
pub use typed::*;

mod erased;
pub use erased::*;

mod options;
pub use options::*;

mod signature;
pub use signature::*;

mod strict;
pub use strict::*;

mod row;
pub use row::*;

pub mod fns;
pub mod internal;
pub mod session;

/// A unique identifier for a scalar function.
pub type ScalarFnId = Id;

/// Private module to seal [`typed::DynScalarFn`].
mod sealed {
    use crate::scalar_fn::ScalarFnVTable;
    use crate::scalar_fn::typed::TypedScalarFnInstance;

    /// Marker trait to prevent external implementations of [`super::typed::DynScalarFn`].
    pub(crate) trait Sealed {}

    /// This can be the **only** implementor for [`super::typed::DynScalarFn`].
    impl<V: ScalarFnVTable> Sealed for TypedScalarFnInstance<V> {}
}

/// A scalar function has a negative cost if applying it to an array and
/// canonicalizing is cheaper than canonicalizing an array and applying it.
///
/// Example of negative cost expressions are byte_length(), ext_storage(), and get_item() since
/// they don't depend on input size.
///
/// Example of non-negative cost expression is like() as it's linear over
/// individual input.
pub fn is_negative_cost(id: ScalarFnId) -> bool {
    id == ScalarFnVTable::id(&ByteLength)
        || id == ScalarFnVTable::id(&ExtStorage)
        || id == ScalarFnVTable::id(&GetItem)
        || id == ScalarFnVTable::id(&Literal)
}
