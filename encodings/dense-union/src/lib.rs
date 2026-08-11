// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A dense physical encoding for Vortex union arrays.
//!
//! [`DenseUnionArray`] stores one type ID and one child offset per logical row. Variant children
//! are compact: unlike the canonical sparse union, they do not contain placeholders for rows that
//! select a different variant. The array still has the logical
//! [`DType::Union`](vortex_array::dtype::DType::Union) dtype.

mod array;
mod canonical;
mod compute;
mod rules;

pub use array::*;
use vortex_array::session::ArraySessionExt;
use vortex_session::VortexSession;

/// Register the dense union encoding in a Vortex session.
pub fn initialize(session: &VortexSession) {
    session.arrays().register(DenseUnion);
}

#[cfg(test)]
mod tests;
