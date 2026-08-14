// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The [`RowFn`] contract for scalar functions whose natural kernel computes one row at a time.
//!
//! Implementations declare their arity and fallibility, then use [`RowFn::dispatch`] to select the
//! typed row signature for each supported dtype combination. Optional hooks provide serialization
//! and encoding-aware execution without putting columnar plumbing in the row kernel.

use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use super::visitor::RowVisitor;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::scalar_fn::ScalarFnId;

/// A strict scalar function whose row kernel cannot produce null from valid inputs.
///
/// This is stronger than
/// [`ScalarFnVTable::is_strict`](crate::scalar_fn::ScalarFnVTable::is_strict), which requires null
/// propagation but permits valid inputs to produce null. The framework derives output validity
/// only from input validity.
///
/// A dispatched [`OutputElement`] or [`OutputSink`] describes the non-nullable values produced for
/// valid rows. The framework widens that dtype when an input dtype is nullable, attaches the
/// input-derived validity, and casts the finished array to the widened dtype. Implementations do
/// not construct nullable placeholders for invalid rows.
///
/// Declare argument names and use [`dispatch`](Self::dispatch) to select element and output types.
/// Every implementation receives the standard [`ScalarFnVTable`]. A public type that needs custom
/// vtable hooks can delegate its row kernel through [`row_fn_return_dtype`] and [`execute_rows`].
///
/// [`OutputElement`]: crate::scalar_fn::unstable::row::OutputElement
/// [`OutputSink`]: crate::scalar_fn::unstable::row::OutputSink
/// [`ScalarFnVTable`]: crate::scalar_fn::ScalarFnVTable
/// [`execute_rows`]: crate::scalar_fn::unstable::row::execute_rows
/// [`row_fn_return_dtype`]: crate::scalar_fn::unstable::row::row_fn_return_dtype
pub trait RowFn: 'static + Sized + Clone + Send + Sync {
    /// Options for this function, or [`EmptyOptions`](crate::scalar_fn::EmptyOptions) for none.
    type Options: 'static + Send + Sync + Clone + Debug + Display + PartialEq + Eq + Hash;

    /// The arguments in display order. Its length is the function's exact arity.
    const ARG_NAMES: &'static [&'static str];

    /// Whether every dispatch and encoded reduction is infallible.
    ///
    /// See [`ScalarFnVTable::is_infallible`](crate::scalar_fn::ScalarFnVTable::is_infallible) for
    /// a more detailed explanation of semantic errors.
    ///
    /// The framework checks dispatched element and result types, but cannot inspect
    /// [`reduce_encoded`](Self::reduce_encoded). Set this to `false` when that hook can return a
    /// semantic error. A conservative `false` is allowed.
    const INFALLIBLE: bool;

    /// Returns the ID of the scalar function.
    fn id(&self) -> ScalarFnId;

    /// Serialize this function's options, or return `None` when the function is not serializable.
    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        _ = options;
        Ok(None)
    }

    /// Restore options written by [`serialize`](Self::serialize).
    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        vortex_bail!("Expression {} is not deserializable", self.id())
    }

    /// Choose element types for these input dtypes and visit the framework with them.
    ///
    /// Planning and execution both call this method, so its result **must** depend only on
    /// `options` and `args`. Cross-argument dtype validation belongs here.
    fn dispatch<V: RowVisitor<Self::Options>>(
        &self,
        options: &Self::Options,
        args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult>;

    /// Try an encoding-aware implementation before decoding the inputs into row elements.
    ///
    /// `None` continues to the row loop. The returned array can remain encoded or lazy. Batch
    /// execution calls this hook at most once with the original nonempty inputs. Nullary functions,
    /// empty batches, slices, and compacted retries skip it.
    ///
    /// Like a dense row closure, this hook must be total over every stored payload, including
    /// payloads behind null rows. An `Err` is immediately user-visible and is never suppressed or
    /// retried through the row layer.
    ///
    /// # Requirements
    ///
    /// - `output.len()` **must** equal `args[0].len()`.
    /// - The output dtype **must** match the planned dtype when ignoring outer nullability.
    /// - The output **must not** introduce a null where every input is valid.
    fn reduce_encoded(
        &self,
        options: &Self::Options,
        args: &[ArrayRef],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        _ = (options, args, ctx);
        Ok(None)
    }
}
