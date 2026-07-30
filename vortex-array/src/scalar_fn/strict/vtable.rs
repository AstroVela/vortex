// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The strict scalar function contract.

use vortex_error::VortexResult;

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
