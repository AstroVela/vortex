// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scalar functions computed one row at a time.

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ElementTuple;
use crate::scalar_fn::OutputSink;
use crate::scalar_fn::PersistableOptions;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::SinkResult;

/// A scalar function computed one row at a time.
///
/// An implementor names a representative argument tuple, then in [`dispatch`](Self::dispatch) picks
/// the concrete element and sink types for a batch. Everything structural (arity, dtype validation,
/// the return dtype, null handling, and execution) is derived through the blanket impls, so the
/// implementing type _is_ the scalar function, with no wrapper.
///
/// The argument witness names one representative choice. Its arity and decode properties steer the
/// framework _before_ `dispatch` runs. They therefore **must** not vary between dispatches, which
/// the following pins: the witness promises the dense-safe
/// [`BytesLen`](crate::scalar_fn::BytesLen), the dispatch visits the non-dense-safe
/// [`Bytes`](crate::scalar_fn::Bytes), and the build fails.
///
/// ```compile_fail
/// # use vortex_array::dtype::{DType, Nullability};
/// # use vortex_array::scalar_fn::*;
/// # use vortex_error::VortexResult;
/// # use vortex_session::registry::CachedId;
/// #[derive(Clone)]
/// struct Lie;
/// impl RowFn for Lie {
///     type Options = EmptyOptions;
///     type ArgsWitness = (BytesLen,);
///     fn id(&self) -> ScalarFnId {
///         static ID: CachedId = CachedId::new("example.lie");
///         *ID
///     }
///     fn arg_name(&self, _idx: usize) -> ChildName {
///         ChildName::from("input")
///     }
///     fn dispatch<V: RowVisitor>(
///         &self,
///         _options: &Self::Options,
///         _args: &[DType],
///         visitor: V,
///     ) -> VortexResult<V::Out> {
///         visitor.visit_prepared_into::<(Bytes,), ElementSink<u64>, _, _>(
///             |_| (),
///             |&(), (b,), output| output.write(b.len() as u64),
///         )
///     }
/// }
/// // Instantiating a vtable method that dispatches evaluates the compile-time witness check.
/// let dtype = DType::Utf8(Nullability::NonNullable);
/// let _ = ScalarFnVTable::return_dtype(&Lie, &EmptyOptions, &[dtype]);
/// ```
///
/// A function whose kernel is columnar rather than row-at-a-time (negating a whole bit buffer, a
/// zero-copy unwrap) is not a `RowFn`, and implements
/// [`ScalarFnVTable`](crate::scalar_fn::ScalarFnVTable) directly.
pub trait RowFn: 'static + Sized + Clone + Send + Sync {
    /// Options for this function, if any. Use [`EmptyOptions`](crate::scalar_fn::EmptyOptions)
    /// for none.
    type Options: PersistableOptions;

    /// Any one argument tuple [`dispatch`](Self::dispatch) can choose.
    ///
    /// The framework reads the arity, and whether every argument is readable behind a null row, off
    /// this witness before dispatching. A dispatch that visits at a tuple disagreeing with it on
    /// either does not compile.
    ///
    /// **Why a witness exists at all**, rather than the framework asking `dispatch`:
    /// [`arity`](crate::scalar_fn::ScalarFnVTable::arity) and
    /// [`is_fallible`](crate::scalar_fn::ScalarFnVTable::is_fallible) take no input dtypes, while
    /// `dispatch` needs dtypes to choose. The argument properties must therefore be
    /// dtype-independent. The compile-time check on every visit stops a dispatch from contradicting
    /// this witness.
    type ArgsWitness: ElementTuple;

    /// Whether the row computation can fail on legal input values.
    ///
    /// Fallible input decoding is derived separately from [`ArgsWitness`](Self::ArgsWitness). This
    /// constant exists because [`ScalarFnVTable::is_fallible`] is queried without input dtypes, so
    /// the framework cannot run [`dispatch`](Self::dispatch) to inspect its [`SinkResult`].
    const FALLIBLE: bool = false;

    /// Returns the ID of the scalar function.
    fn id(&self) -> ScalarFnId;

    /// The display name of the `idx`-th argument.
    fn arg_name(&self, idx: usize) -> ChildName;

    /// Choose element types for these input dtypes and visit the framework with them.
    ///
    /// This is where a per-batch width match lives (`match_each_float_ptype!` and friends panic
    /// outside their width class, so check the class first), and where cross-argument dtype
    /// constraints belong, since per-argument validation runs inside the visit. Plan time and run
    /// time both come through here, so the choice **must** be a pure function of `options` and
    /// `args`.
    fn dispatch<V: RowVisitor>(
        &self,
        options: &Self::Options,
        args: &[DType],
        visitor: V,
    ) -> VortexResult<V::Out>;

    /// An encoding-aware rewrite, tried on the input arrays before the row loop.
    ///
    /// `Some` skips the row loop entirely, which makes this the escape hatch for a function that is
    /// row-shaped in general but has a bulk answer for some encodings: reading stored values back out
    /// of a wrapper encoding, or handing back a child array whole. The result may be lazy and
    /// nullable, but its nulls **must** be a subset of the rows the lifting will mask, and it
    /// **must** have one row per row of `args`, which on the filter strategy is the _filtered_ count
    /// rather than the original one.
    ///
    /// Whether the arrays still carry their original encoding depends on the path above:
    ///
    /// - [`Dense`](crate::scalar_fn::NullHandling::Dense) always passes them through untouched.
    /// - [`Filter`](crate::scalar_fn::NullHandling::Filter) passes them through untouched when no
    ///   row is null. For a mixed mask, the branch-and-skip strategy also passes them through
    ///   untouched (full length, the result masked afterwards), while the filter strategy hands
    ///   over filtered copies, which are canonical and so match no encoding fast path.
    ///
    /// A non-nullable operand therefore reaches an encoding fast path under either. Note also that
    /// filtering a constant yields a constant, so a fast path keyed on
    /// [`as_constant`](ArrayRef::as_constant) still fires even for a filtered batch.
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

/// One use of a [`RowFn`] at concrete element types.
///
/// The framework hands a visitor to [`RowFn::dispatch`], which calls one of the visit methods with
/// the element types it chose: at plan time the visit validates dtypes, at run time it executes the row
/// loop. Only the framework implements this trait, and a function only ever _calls_ a visit.
///
/// The function names one output sink and one preparation step. Passing `|_| ()` is the no-prepare
/// case.
pub trait RowVisitor: private::Sealed {
    /// What this visit produces.
    type Out;

    /// Visit at argument tuple `A`, preparing shared state once and writing every output row into
    /// sink `S`.
    ///
    /// `prepare` receives [`A::ConstElems`](ElementTuple::ConstElems): the element value of every
    /// argument whose operand is constant for the batch, and `None` for each one that varies by
    /// row. Whatever it returns is handed to every `apply` call by shared reference.
    ///
    /// `A` **must** agree with the [`RowFn`]'s argument witness. A fallible or deferred-error `R`
    /// also requires [`RowFn::FALLIBLE`] to be `true`; the reverse is not required. A deferred
    /// result must be paired with a sink whose [`OutputSink::ERRORS_ARE_DEFERRED`] is `true`. A visit
    /// that violates any of these conditions is a compile error.
    fn visit_prepared_into<A: ElementTuple, S: OutputSink, P, R: SinkResult>(
        self,
        prepare: impl FnOnce(A::ConstElems<'_>) -> P,
        apply: impl Fn(&P, A::Elems<'_>, S::Row<'_>) -> R,
    ) -> VortexResult<Self::Out>;
}

pub(super) mod private {
    pub trait Sealed {}
}
