// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scalar functions computed one row at a time.

use std::marker::PhantomData;

use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::arrays::scalar_fn::plugin::ScalarFnArrayParts;
use crate::arrays::scalar_fn::plugin::ScalarFnArrayVTable;
use crate::dtype::DType;
use crate::expr::Expression;
use crate::expr::union_child_validities;
use crate::scalar_fn::ApplyResult;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ElementTuple;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::NullHandling;
use crate::scalar_fn::OutputSink;
use crate::scalar_fn::PersistableOptions;
use crate::scalar_fn::RowResult;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::SinkResult;
use crate::scalar_fn::StrictScalarFnVTable;
use crate::scalar_fn::decode_scalar_fn_array;
use crate::scalar_fn::encode_children_and_options;
use crate::scalar_fn::row::execute::execute_row_loop_branch;
use crate::scalar_fn::row::execute::execute_row_loop_prepared;
use crate::scalar_fn::row::execute::execute_row_sink;
use crate::scalar_fn::row::execute::row_is_fallible;
use crate::scalar_fn::row::execute::row_null_handling;
use crate::scalar_fn::row::execute::validate_row_args;
use crate::scalar_fn::row::execute::validate_row_sink;
use crate::serde::ArrayChildren;

/// A scalar function computed one row at a time.
///
/// An implementor names a representative argument tuple and return type as witnesses, then in
/// [`dispatch`](Self::dispatch) picks the concrete element types for a batch and hands the framework
/// a row closure. Everything structural (arity, dtype validation, the return dtype, null handling,
/// fallibility, execution, and array serde) is derived through the blanket impls, so the
/// implementing type *is* the scalar function, with no wrapper. See `byte_length` for a function at
/// fixed element types, or `vortex-tensor`'s `L2Norm` for one that dispatches over widths.
///
/// The witnesses name one representative choice, and the arity, dense-safety and fallibility read off
/// them steer the framework *before* `dispatch` runs. They therefore **must** not vary between
/// dispatches, which the following pins: the witnesses promise the dense-safe
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
///     type RetWitness = u64;
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
///         visitor.visit::<(Bytes,), u64>(|(b,)| b.len() as u64)
///     }
/// }
/// // Instantiating a vtable method that dispatches evaluates the compile-time witness check.
/// let dtype = DType::Utf8(Nullability::NonNullable);
/// let _ = StrictScalarFnVTable::return_element_dtype(&Lie, &EmptyOptions, &[dtype]);
/// ```
///
/// A function whose kernel is columnar rather than row-at-a-time (negating a whole bit buffer, a
/// zero-copy unwrap) is not a `RowFn`, and implements
/// [`StrictScalarFnVTable`](crate::scalar_fn::StrictScalarFnVTable) directly.
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
    /// [`arity`](StrictScalarFnVTable::arity),
    /// [`null_handling`](StrictScalarFnVTable::null_handling) and
    /// [`is_fallible`](StrictScalarFnVTable::is_fallible) all take only the options, with no input
    /// dtypes, while `dispatch` needs dtypes to choose. So those three answers **must** be
    /// dtype-independent, and cannot be read off whichever element types a batch happens to pick.
    /// The witness is where they are stated once, and the compile-time check on every visit is what
    /// stops a dispatch from contradicting them.
    ///
    /// Naming *types* rather than three constants is deliberate: it means dense-safety and
    /// fallibility are derived from the element types instead of hand-declared, so the only mistake
    /// available is a witness that disagrees with the dispatch, which is a build error.
    type ArgsWitness: ElementTuple;

    /// The return type paired with [`ArgsWitness`](Self::ArgsWitness).
    ///
    /// Together with [`ArgsWitness`](Self::ArgsWitness) this fixes whether the function is fallible,
    /// either from the return type here or from an argument whose decode can fail. The framework
    /// also reads that before dispatching, so it too **must** not vary between the choices.
    ///
    /// Fallibility is all it fixes, which is why the bound is [`RowResult`] and not
    /// [`ApplyResult`](crate::scalar_fn::ApplyResult): the output dtype is read off whatever a visit
    /// chooses, not off the witness, so a returning dispatch names an
    /// [`OutputElement`](crate::scalar_fn::OutputElement) (or a [`VortexResult`] of one) here while a
    /// sink-writing one names `()` (or `VortexResult<()>`).
    type RetWitness: RowResult;

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
    /// nullable, but its nulls **must** be a subset of the rows the strict lifting will mask, and it
    /// **must** have one row per row of `args`, which on the filter strategy is the *filtered* count
    /// rather than the original one.
    ///
    /// Whether the arrays still carry their original encoding depends on the path above:
    ///
    /// - [`NullHandling::Dense`] always passes them through untouched.
    /// - [`NullHandling::Filter`] passes them through untouched when no row is null. For a mixed
    ///   mask, the branch-and-skip strategy also passes them through untouched (full length, the
    ///   result masked afterwards), while the filter strategy hands over filtered copies, which
    ///   are canonical and so match no encoding fast path.
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
/// loop. Only the framework implements this trait, and a function only ever *calls* a visit.
///
/// The methods differ only in how the row closure delivers its output and in whether a per-batch
/// prepare step runs before the loop, and a dispatch picks whichever fits. All apply the same
/// witness check.
pub trait RowVisitor {
    /// What this visit produces.
    type Out;

    /// Visit at argument tuple `A`, with `apply` *returning* one `R` from one row.
    ///
    /// This is [`visit_prepared`](Self::visit_prepared) with nothing to prepare: the unit state is
    /// zero-sized, so the derived call monomorphizes to the same row loop and the derivation is
    /// not overridable.
    ///
    /// `A` and `R` **must** agree with the [`RowFn`]'s witnesses on arity, dense-safety and
    /// fallibility. A visit that does not is a compile error.
    fn visit<A: ElementTuple, R: ApplyResult>(
        self,
        apply: impl Fn(A::Elems<'_>) -> R,
    ) -> VortexResult<Self::Out>
    where
        Self: Sized,
    {
        self.visit_prepared::<A, (), R>(|_| (), move |&(), elems| apply(elems))
    }

    /// Visit at argument tuple `A`, with a `prepare` step run once per batch before the row loop.
    ///
    /// `prepare` receives [`A::ConstElems`](ElementTuple::ConstElems): the element value of every
    /// argument whose operand is constant for the batch, and `None` for each one that varies by
    /// row. Whatever it returns is handed to every `apply` call by shared reference, so work that
    /// depends only on a constant argument (the norm of a broadcast query vector, say) is paid
    /// once per batch instead of once per row. `apply` stays [`Fn`] and the state arrives behind
    /// `&P`, which keeps the row loop's shape identical to [`visit`](Self::visit).
    ///
    /// Constness is invisible to [`RowFn::dispatch`], which sees only dtypes: the same dispatch
    /// serves a batch where an operand is constant and one where it is not, and `prepare` finds
    /// out which at run time. At plan time neither closure runs.
    ///
    /// `prepare` is infallible by design: it refines values the row loop could compute itself,
    /// and fallibility is read off the witnesses *before* dispatch, so a failing prepare would
    /// have nowhere to be declared. A fallible-prepare variant (`prepare` returning
    /// `VortexResult<P>`, surfaced through the witnesses so it forces
    /// [`is_fallible`](crate::scalar_fn::ScalarFnVTable::is_fallible) and
    /// [`NullHandling::Filter`]) is a possible extension, deliberately left out of this method.
    ///
    /// `A` and `R` **must** agree with the [`RowFn`]'s witnesses on arity, dense-safety and
    /// fallibility, exactly as for [`visit`](Self::visit).
    fn visit_prepared<A: ElementTuple, P, R: ApplyResult>(
        self,
        prepare: impl FnOnce(A::ConstElems<'_>) -> P,
        apply: impl Fn(&P, A::Elems<'_>) -> R,
    ) -> VortexResult<Self::Out>;

    /// Visit at argument tuple `A`, with `apply` *writing* one row of sink `S` and returning only
    /// whether that failed.
    ///
    /// Use this when an owned value per row is the wrong shape for the output: a row whose width is
    /// runtime data, or bytes appended to one buffer for the whole batch. See [`OutputSink`].
    ///
    /// `A` and `R` **must** agree with the [`RowFn`]'s witnesses on arity, dense-safety and
    /// fallibility, exactly as for [`visit`](Self::visit). A sink-writing dispatch therefore names
    /// `()` or `VortexResult<()>` as its [`RetWitness`](RowFn::RetWitness).
    fn visit_into<A: ElementTuple, S: OutputSink, R: SinkResult>(
        self,
        apply: impl Fn(A::Elems<'_>, S::Row<'_>) -> R,
    ) -> VortexResult<Self::Out>;
}

/// Compile-time check that a dispatched `(A, R)` agrees with `F`'s witnesses on everything the
/// framework reads *without* input dtypes, and therefore before dispatching: the arity (which the
/// expression layer and array serde use), and the dense-safety and fallibility that decide null
/// handling. Evaluated by monomorphizing a [`visit`](RowVisitor::visit), so even a dispatch arm that
/// never runs is checked.
///
/// Comparing the raw properties rather than the derived [`NullHandling`] is deliberate: null handling
/// collapses dense-safety and fallibility together, so an arm that flipped both would slip past a
/// check on the derived value.
const fn assert_witness_agrees<F: RowFn, A: ElementTuple, R: RowResult>() {
    assert!(
        A::ARITY == <F::ArgsWitness as ElementTuple>::ARITY,
        "dispatch visited a tuple whose arity differs from ArgsWitness",
    );
    assert!(
        A::DENSE_SAFE == <F::ArgsWitness as ElementTuple>::DENSE_SAFE,
        "dispatch visited an argument that differs from ArgsWitness in whether it is readable \
         behind a null row",
    );
    assert!(
        A::DECODE_SHRINKS_WHEN_FILTERED
            == <F::ArgsWitness as ElementTuple>::DECODE_SHRINKS_WHEN_FILTERED,
        "dispatch visited an argument that differs from ArgsWitness in whether its decode \
         shrinks when filtered",
    );
    assert!(
        row_is_fallible::<A, R>() == row_is_fallible::<F::ArgsWitness, F::RetWitness>(),
        "dispatch visited types whose fallibility differs from the witnesses",
    );
}

/// The plan-time visit: validate the input dtypes and name the output element dtype.
struct ValidateRows<'a, F> {
    args: &'a [DType],

    /// The visited function, carried only so the witness check can name its witnesses.
    row_fn: PhantomData<F>,
}

impl<F: RowFn> RowVisitor for ValidateRows<'_, F> {
    type Out = DType;

    fn visit_prepared<A: ElementTuple, P, R: ApplyResult>(
        self,
        _prepare: impl FnOnce(A::ConstElems<'_>) -> P,
        _apply: impl Fn(&P, A::Elems<'_>) -> R,
    ) -> VortexResult<DType> {
        const { assert_witness_agrees::<F, A, R>() };
        validate_row_args::<A, R>(self.args)
    }

    fn visit_into<A: ElementTuple, S: OutputSink, R: SinkResult>(
        self,
        _apply: impl Fn(A::Elems<'_>, S::Row<'_>) -> R,
    ) -> VortexResult<DType> {
        const { assert_witness_agrees::<F, A, R>() };
        validate_row_sink::<A, S>(self.args)
    }
}

/// The run-time visit: decode every column once and run the row loop.
struct ExecuteRows<'a, 'b, F> {
    args: &'a dyn ExecutionArgs,

    /// The input dtypes, which a sink needs to size and name its output column. Carried rather than
    /// re-derived from `args`, since [`execute_strict`](StrictScalarFnVTable::execute_strict) has
    /// already collected them to dispatch on.
    arg_dtypes: &'a [DType],

    ctx: &'b mut ExecutionCtx,

    /// The visited function, carried only so the witness check can name its witnesses.
    row_fn: PhantomData<F>,
}

impl<F: RowFn> RowVisitor for ExecuteRows<'_, '_, F> {
    type Out = ArrayRef;

    fn visit_prepared<A: ElementTuple, P, R: ApplyResult>(
        self,
        prepare: impl FnOnce(A::ConstElems<'_>) -> P,
        apply: impl Fn(&P, A::Elems<'_>) -> R,
    ) -> VortexResult<ArrayRef> {
        const { assert_witness_agrees::<F, A, R>() };
        execute_row_loop_prepared::<A, P, R>(self.args, self.ctx, prepare, apply)
    }

    fn visit_into<A: ElementTuple, S: OutputSink, R: SinkResult>(
        self,
        apply: impl Fn(A::Elems<'_>, S::Row<'_>) -> R,
    ) -> VortexResult<ArrayRef> {
        const { assert_witness_agrees::<F, A, R>() };
        execute_row_sink::<A, S, R>(self.args, self.arg_dtypes, self.ctx, apply)
    }
}

/// The run-time visit for the branch-and-skip null strategy: compute only the conjoined-valid
/// rows over unfiltered columns.
///
/// `Ok(None)` means the visit cannot take that strategy (a sink dispatch, or an argument with no
/// null-tolerant decode for its array), and the strict lifting falls back to the filter strategy.
struct ExecuteRowsBranch<'a, 'b, F> {
    args: &'a dyn ExecutionArgs,

    /// The conjoined validity, materialized by the strict lifting and guaranteed mixed.
    valid: &'a Mask,

    ctx: &'b mut ExecutionCtx,

    /// The visited function, carried only so the witness check can name its witnesses.
    row_fn: PhantomData<F>,
}

impl<F: RowFn> RowVisitor for ExecuteRowsBranch<'_, '_, F> {
    type Out = Option<ArrayRef>;

    fn visit_prepared<A: ElementTuple, P, R: ApplyResult>(
        self,
        prepare: impl FnOnce(A::ConstElems<'_>) -> P,
        apply: impl Fn(&P, A::Elems<'_>) -> R,
    ) -> VortexResult<Option<ArrayRef>> {
        const { assert_witness_agrees::<F, A, R>() };
        execute_row_loop_branch::<A, P, R>(self.args, self.valid, self.ctx, prepare, apply)
    }

    fn visit_into<A: ElementTuple, S: OutputSink, R: SinkResult>(
        self,
        _apply: impl Fn(A::Elems<'_>, S::Row<'_>) -> R,
    ) -> VortexResult<Option<ArrayRef>> {
        const { assert_witness_agrees::<F, A, R>() };
        // Sink dispatches stay on the dense and filter paths: a sink is allocated for a known row
        // count and has no notion of a skipped row, so `None` sends the batch to the filter
        // strategy. See the module docs.
        Ok(None)
    }
}

/// Every [`RowFn`] is a [`StrictScalarFnVTable`], and hence a full
/// [`ScalarFnVTable`](crate::scalar_fn::ScalarFnVTable).
impl<F: RowFn> StrictScalarFnVTable for F {
    type Options = F::Options;

    fn id(&self) -> ScalarFnId {
        RowFn::id(self)
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(F::ArgsWitness::ARITY)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        self.arg_name(child_idx)
    }

    fn return_element_dtype(&self, options: &Self::Options, args: &[DType]) -> VortexResult<DType> {
        self.dispatch(
            options,
            args,
            ValidateRows::<F> {
                args,
                row_fn: PhantomData,
            },
        )
    }

    fn null_handling(&self, _options: &Self::Options) -> NullHandling {
        row_null_handling::<F::ArgsWitness, F::RetWitness>()
    }

    /// Both output forms build an all-valid column, so a row kernel cannot turn a wholly non-null row
    /// into a null and the output validity is exactly the conjunction of the inputs'. Letting either
    /// an [`OutputElement`](crate::scalar_fn::OutputElement) or an [`OutputSink`] produce nulls would
    /// invalidate this.
    fn validity(
        &self,
        _options: &Self::Options,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        union_child_validities(expression)
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        row_is_fallible::<F::ArgsWitness, F::RetWitness>()
    }

    fn execute_strict(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let inputs = (0..args.num_inputs())
            .map(|i| args.get(i))
            .collect::<VortexResult<Vec<_>>>()?;

        if let Some(reduced) = self.reduce_encoded(options, &inputs, ctx)? {
            return Ok(reduced);
        }

        let arg_dtypes = inputs
            .iter()
            .map(|input| input.dtype().clone())
            .collect::<Vec<_>>();

        self.dispatch(
            options,
            &arg_dtypes,
            ExecuteRows::<F> {
                args,
                arg_dtypes: &arg_dtypes,
                ctx,
                row_fn: PhantomData,
            },
        )
    }

    fn execute_strict_branch(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        valid: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let inputs = (0..args.num_inputs())
            .map(|i| args.get(i))
            .collect::<VortexResult<Vec<_>>>()?;

        // The encoding-aware rewrite runs before the row loop exactly as in
        // [`execute_strict`](StrictScalarFnVTable::execute_strict). Here it sees the original
        // (unfiltered) encodings, and its full-length result is masked by the caller like any
        // other branch result.
        if let Some(reduced) = self.reduce_encoded(options, &inputs, ctx)? {
            return Ok(Some(reduced));
        }

        let arg_dtypes = inputs
            .iter()
            .map(|input| input.dtype().clone())
            .collect::<Vec<_>>();

        self.dispatch(
            options,
            &arg_dtypes,
            ExecuteRowsBranch::<F> {
                args,
                valid,
                ctx,
                row_fn: PhantomData,
            },
        )
    }

    fn decode_shrinks_when_filtered(&self, _options: &Self::Options) -> bool {
        F::ArgsWitness::DECODE_SHRINKS_WHEN_FILTERED
    }
}

/// Every [`RowFn`] can be persisted as an array: all child dtypes are stored, since the output
/// dtype generally cannot recover them, then re-validated through the function's own dtype rules.
impl<F: RowFn> ScalarFnArrayVTable for F {
    fn serialize(
        &self,
        view: &ScalarFnArrayView<Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        encode_children_and_options(view, F::ArgsWitness::ARITY).map(Some)
    }

    fn deserialize(
        &self,
        _dtype: &DType,
        len: usize,
        metadata: &[u8],
        children: &dyn ArrayChildren,
        session: &VortexSession,
    ) -> VortexResult<ScalarFnArrayParts<Self>> {
        decode_scalar_fn_array(
            self,
            F::ArgsWitness::ARITY,
            len,
            metadata,
            children,
            session,
        )
    }
}
