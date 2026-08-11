// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Argument lists built from [`InputElement`]s, and the per-argument decode behind them.

mod element_tuple;
pub use element_tuple::ElementTuple;
pub use element_tuple::batch_constant;

mod indexed;
pub use indexed::IndexedElementTuple;

#[cfg(test)]
mod tests;
