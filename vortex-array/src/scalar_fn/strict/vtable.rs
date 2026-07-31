// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The strict scalar function contract.

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::expr::Expression;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::NullHandling;
use crate::scalar_fn::PersistableOptions;
use crate::scalar_fn::ReduceCtx;
use crate::scalar_fn::ReduceNode;
use crate::scalar_fn::ReduceNodeRef;
use crate::scalar_fn::ScalarFnId;

/// A vtable for scalar functions that are unconditionally [strict].
///
/// The implementor writes the structural metadata plus one columnar kernel over non-null values, and
/// a blanket impl derives null propagation, constant folding, nullability, and options serde. See
/// `not`, `list_length`, or `list_sum` for implementations.
///
/// Beyond the members here, the blanket impl leaves every optional [`ScalarFnVTable`] method at that
/// trait's own default. [`reduce`](Self::reduce) and [`validity`](Self::validity) are mirrored,
/// because a strict function cannot implement [`ScalarFnVTable`] itself to override one: mirror
/// another method here when a function actually needs it.
///
/// Strictness here is null *propagation* only, which is the property push-downs need. A kernel that
/// turns a wholly non-null row into a null is still welcome, and just leaves
/// [`validity`](Self::validity) at its default.
///
/// [strict]: crate::scalar_fn::ScalarFnVTable::is_strict
/// [`ScalarFnVTable`]: crate::scalar_fn::ScalarFnVTable
pub trait StrictScalarFnVTable: 'static + Sized + Clone + Send + Sync {
    /// Options for this function, which know how to persist themselves. Use
    /// [`EmptyOptions`](crate::scalar_fn::EmptyOptions) for none.
    type Options: PersistableOptions;

    /// Returns the ID of the scalar function.
    fn id(&self) -> ScalarFnId;

    /// Returns the arity of this function.
    fn arity(&self, options: &Self::Options) -> Arity;

    /// Returns the name of the nth child of the function.
    fn child_name(&self, options: &Self::Options, child_idx: usize) -> ChildName;

    /// The [`DType`] the kernel produces over non-null values.
    ///
    /// Nullability is managed by the blanket impl, which widens this to nullable iff any input is
    /// nullable.
    fn return_element_dtype(&self, options: &Self::Options, args: &[DType]) -> VortexResult<DType>;

    /// How the kernel sees rows that are null in some input. See [`NullHandling`].
    fn null_handling(&self, options: &Self::Options) -> NullHandling;

    /// Whether the kernel is semantically fallible. See [`ScalarFnVTable::is_fallible`] for more
    /// information.
    ///
    /// A fallible kernel **must** use [`NullHandling::Filter`]. Pairing it with
    /// [`NullHandling::Dense`] is an error, since the kernel would also run on the arbitrary values
    /// behind null rows.
    ///
    /// [`ScalarFnVTable::is_fallible`]: crate::scalar_fn::ScalarFnVTable::is_fallible
    fn is_fallible(&self, options: &Self::Options) -> bool;

    /// Evaluate the kernel columnwise.
    ///
    /// The blanket impl guarantees that no input is a null constant, and that the inputs are not all
    /// constant. Under [`NullHandling::Filter`] every row of every input is valid. Under
    /// [`NullHandling::Dense`] rows behind nulls hold arbitrary values and their results are
    /// discarded.
    ///
    /// Either way the kernel can ignore input validity, and its result **must** equal
    /// [`return_element_dtype`](Self::return_element_dtype) up to nullability. A kernel that returns
    /// nulls of its own keeps them, unioned with the ones the lifting applies, which requires
    /// `return_element_dtype` to be nullable.
    fn execute_strict(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef>;

    /// Evaluate the kernel over the full-length, *unfiltered* inputs, computing only the rows set
    /// in `valid` and writing an arbitrary placeholder to every other output slot.
    ///
    /// This is the branch-and-skip null strategy, which the lifting may pick for a batch with a
    /// mixed validity mask instead of literally filtering (see [`NullHandling::Filter`]). The
    /// contract mirrors [`execute_strict`](Self::execute_strict), except that the inputs still
    /// contain their null rows: the kernel **must not** run its row computation (nor any per-row
    /// fallible decode) on a row unset in `valid`, because such rows hold arbitrary values and a
    /// fallible kernel would spuriously fail on them. The caller masks the result with `valid`
    /// afterwards, exactly as the dense path does, so the placeholders are never observed.
    ///
    /// `valid` is guaranteed mixed (neither all-true nor all-false). The default returns
    /// `Ok(None)`: this kernel has no branch execution, and the lifting uses the filter strategy.
    /// The hook lives on this trait, rather than the machinery staying inside the lifting,
    /// because only the implementor knows how to compute a subset of rows; the row layer
    /// overrides it once for every [`RowFn`](crate::scalar_fn::RowFn), and a hand-written strict
    /// function normally leaves it alone.
    fn execute_strict_branch(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        valid: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        _ = (options, args, valid, ctx);
        Ok(None)
    }

    /// Whether this function's per-batch decode does per-row work whose cost shrinks
    /// proportionally when the inputs are filtered first.
    ///
    /// This is [`InputElement::DECODE_SHRINKS_WHEN_FILTERED`] aggregated over the arguments,
    /// which is how the row layer answers it. It steers the per-batch choice between the
    /// branch-and-skip and filter strategies, so it is only consulted when
    /// [`execute_strict_branch`](Self::execute_strict_branch) is implemented; a function that
    /// leaves that at its default leaves this alone too.
    ///
    /// [`InputElement::DECODE_SHRINKS_WHEN_FILTERED`]:
    ///     crate::scalar_fn::InputElement::DECODE_SHRINKS_WHEN_FILTERED
    fn decode_shrinks_when_filtered(&self, options: &Self::Options) -> bool {
        _ = options;
        false
    }

    /// An expression for the output validity, or `None` to read it off the executed result. See
    /// [`ScalarFnVTable::validity`] for more information.
    ///
    /// Strictness bounds the output validity from above, giving
    /// `valid(f(a1, .., ak)) ⊆ valid(a1) ∧ .. ∧ valid(ak)`, so a kernel that never turns a wholly
    /// non-null row into a null may return that conjunction with
    /// [`union_child_validities`](crate::expr::union_child_validities). One that can, like summing a
    /// valid *empty* list, **must** leave this at the default, since the conjunction would report a
    /// row valid where the kernel yields null.
    ///
    /// [`ScalarFnVTable::validity`]: crate::scalar_fn::ScalarFnVTable::validity
    fn validity(
        &self,
        options: &Self::Options,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        _ = (options, expression);
        Ok(None)
    }

    /// Apply a reduction rule over a tree of scalar functions. See [`ScalarFnVTable::reduce`] for
    /// more information.
    ///
    /// [`ScalarFnVTable::reduce`]: crate::scalar_fn::ScalarFnVTable::reduce
    fn reduce(
        &self,
        options: &Self::Options,
        node: &dyn ReduceNode,
        ctx: &dyn ReduceCtx,
    ) -> VortexResult<Option<ReduceNodeRef>> {
        _ = (options, node, ctx);
        Ok(None)
    }
}
