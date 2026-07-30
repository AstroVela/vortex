// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A vtable for scalar functions that are unconditionally [strict].
//!
//! A strict function maps a null input row to a null output row, and computes non-null outputs
//! from non-null inputs alone. That makes null propagation, constant folding, validity, and
//! nullability identical across implementations, so [`StrictScalarFnVTable`] asks only for the
//! kernel over non-null values and a blanket impl derives the rest:
//!
//! - `is_strict` is `true`, and a kernel that never turns a wholly non-null row into a null can
//!   answer [`validity`] with the conjunction of its child validities, so the planner never has to
//!   execute the function to know which rows are null.
//! - `return_dtype` widens [`return_element_dtype`] to nullable iff any input is nullable, making
//!   the strictness dtype contract hold by construction.
//! - `execute` handles the shared cases before the kernel runs: a null constant input short
//!   circuits to an all-null constant, all-constant inputs evaluate one row and broadcast, and
//!   partially-null inputs are handled per [`NullHandling`].
//! - `serialize` and `deserialize` come from [`PersistableOptions`] on the options type.
//!
//! A strict function whose kernel is happy to read one row at a time should implement
//! [`RowFn`] instead, which derives this whole trait in turn. See
//! [choosing a trait](crate::scalar_fn#choosing-a-trait).
//!
//! Functions with options-dependent strictness (`Binary`) or Kleene logic should implement
//! [`ScalarFnVTable`] directly. Of that trait's optional methods only [`reduce`] and [`validity`] are
//! mirrored here, because a strict function cannot implement [`ScalarFnVTable`] itself to override
//! one. Mirror another method when a function actually needs it.
//!
//! [`PersistableOptions`]: crate::scalar_fn::PersistableOptions
//! [`RowFn`]: crate::scalar_fn::RowFn
//! [`ScalarFnVTable`]: crate::scalar_fn::ScalarFnVTable
//! [`reduce`]: StrictScalarFnVTable::reduce
//! [`return_element_dtype`]: StrictScalarFnVTable::return_element_dtype
//! [`validity`]: StrictScalarFnVTable::validity
//! [strict]: crate::scalar_fn::ScalarFnVTable::is_strict

mod execute;

mod vtable;
pub use vtable::StrictScalarFnVTable;

#[cfg(test)]
mod tests;

/// How [`execute_strict`] sees rows that are null in some input.
///
/// [`execute_strict`]: crate::scalar_fn::StrictScalarFnVTable::execute_strict
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NullHandling {
    /// Evaluate every row, including rows behind nulls, then mask the result.
    ///
    /// Cheapest, and the only option that leaves inputs at their original encoding. Requires that
    /// the function is infallible and total over whatever sits behind a null, which holds for any
    /// flat fixed-width payload since a null row is just unused bytes. Asking for it without meeting
    /// that is rejected rather than trusted: a fallible function is refused here rather than being
    /// taken at its word.
    Dense,

    /// Filter the inputs to the rows valid in every input, evaluate those, and scatter the results
    /// back.
    ///
    /// Always sound. Use it when the function is fallible, or when decoding a row behind a null
    /// could itself fail (a dictionary code or string view only meaningful for valid rows).
    ///
    /// Not for encoding-aware kernels: the inputs are *filtered copies*, and filtering only pushes
    /// through an extension array or a `ScalarFnArray` with at most one non-constant child. With
    /// two or more, the filter stays on top, so `array.is::<ExactScalarFn<Foo>>()` stops matching
    /// once any row is null and the kernel silently takes its generic path.
    Filter,
}
