// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::type_name;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_utils::debug_with::DebugWith;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::expr::BoundLambdaArgs;
use crate::expr::Lambda;
use crate::expr::display::HigherOrderExprDisplay;
use crate::higher_order_fn::HigherOrderFunctionId;
use crate::higher_order_fn::HigherOrderFunctionOptions;
use crate::higher_order_fn::HigherOrderFunctionVTable;
use crate::higher_order_fn::LambdaCall;
use crate::higher_order_fn::LambdaSignature;
use crate::higher_order_fn::TypedHigherOrderFunctionInstance;
use crate::higher_order_fn::typed::DynHigherOrderFunction;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::validity::Validity;

/// A type-erased higher-order function, pairing a vtable with bound per-call options.
#[derive(Clone)]
pub struct HigherOrderFunctionRef(pub(super) Arc<dyn DynHigherOrderFunction>);

impl HigherOrderFunctionRef {
    /// Bind `vtable` with its per-call `options`.
    pub fn new<V: HigherOrderFunctionVTable>(vtable: V, options: V::Options) -> Self {
        TypedHigherOrderFunctionInstance::new(vtable, options).erased()
    }

    /// The function's global identifier.
    pub fn id(&self) -> HigherOrderFunctionId {
        self.0.id()
    }

    /// Whether this function uses vtable `V`.
    pub fn is<V: HigherOrderFunctionVTable>(&self) -> bool {
        self.0.as_any().is::<TypedHigherOrderFunctionInstance<V>>()
    }

    /// Return typed options when this function uses vtable `V`.
    pub fn as_opt<V: HigherOrderFunctionVTable>(&self) -> Option<&V::Options> {
        self.0
            .as_any()
            .downcast_ref::<TypedHigherOrderFunctionInstance<V>>()
            .map(TypedHigherOrderFunctionInstance::options)
    }

    /// Return typed options for vtable `V`.
    ///
    /// # Panics
    ///
    /// Panics if this function does not use vtable `V`.
    pub fn as_<V: HigherOrderFunctionVTable>(&self) -> &V::Options {
        self.as_opt::<V>()
            .vortex_expect("higher-order function options type mismatch")
    }

    /// Return these options behind an opaque type-erased handle.
    pub fn options(&self) -> HigherOrderFunctionOptions<'_> {
        HigherOrderFunctionOptions { inner: &*self.0 }
    }

    /// Return the arity of the ordinary expression children.
    pub fn arity(&self) -> Arity {
        self.0.arity()
    }

    /// Return the number of lambda arguments.
    pub fn lambda_arity(&self) -> usize {
        self.0.lambda_arity()
    }

    /// Return the name of an ordinary expression child.
    pub fn child_name(&self, child_idx: usize) -> ChildName {
        self.0.child_name(child_idx)
    }

    /// Serialize the per-call options.
    pub fn serialize(&self) -> VortexResult<Option<Vec<u8>>> {
        self.0.options_serialize()
    }

    /// Coerce ordinary input arguments before the lambdas are bound.
    pub fn coerce_input_args(&self, arg_dtypes: &[DType]) -> VortexResult<Vec<DType>> {
        self.0.coerce_input_args(arg_dtypes)
    }

    pub(crate) fn lambda_signatures(
        &self,
        arg_dtypes: &[DType],
        lambdas: &[&Lambda],
    ) -> VortexResult<Box<[LambdaSignature]>> {
        self.0.lambda_signatures(arg_dtypes, lambdas)
    }

    /// Compute this higher-order expression's result dtype.
    pub fn return_dtype(
        &self,
        args: &[DType],
        lambdas: BoundLambdaArgs<'_>,
    ) -> VortexResult<DType> {
        self.0.return_dtype(args, lambdas)
    }

    /// Compute this higher-order expression's result validity.
    pub fn validity(
        &self,
        args: &[ArrayRef],
        lambdas: BoundLambdaArgs<'_>,
    ) -> VortexResult<Validity> {
        self.0.validity(args, lambdas)
    }

    /// Lower this higher-order expression over applied inputs and closed lambdas.
    pub fn apply(
        &self,
        args: &[ArrayRef],
        lambdas: &[LambdaCall],
        execution_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        self.0.apply(args, lambdas, execution_ctx)
    }

    /// Whether evaluation can raise a semantic error.
    pub fn is_fallible(&self, lambdas: BoundLambdaArgs<'_>) -> bool {
        self.0.is_fallible(lambdas)
    }

    /// Format an unbound or bound higher-order expression in SQL-style notation.
    pub(crate) fn fmt_sql(
        &self,
        expr: &dyn HigherOrderExprDisplay,
        f: &mut Formatter<'_>,
    ) -> fmt::Result {
        self.0.fmt_sql(expr, f)
    }
}

impl Debug for HigherOrderFunctionRef {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("HigherOrderFunctionRef")
            .field("vtable", &self.id())
            .field("options", &DebugWith(|fmt| self.0.options_debug(fmt)))
            .finish()
    }
}

impl Display for HigherOrderFunctionRef {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}(", self.id())?;
        self.0.options_display(f)?;
        write!(f, ")")
    }
}

impl PartialEq for HigherOrderFunctionRef {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id() && self.0.options_eq(other.0.options_any())
    }
}

impl Eq for HigherOrderFunctionRef {}

impl Hash for HigherOrderFunctionRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id().hash(state);
        self.0.options_hash(state);
    }
}

impl HigherOrderFunctionRef {
    /// Downcast this function to its typed representation.
    pub fn try_downcast<V: HigherOrderFunctionVTable>(
        self,
    ) -> Result<Arc<TypedHigherOrderFunctionInstance<V>>, Self> {
        if self.is::<V>() {
            let ptr = Arc::into_raw(self.0) as *const TypedHigherOrderFunctionInstance<V>;
            Ok(unsafe { Arc::from_raw(ptr) })
        } else {
            Err(self)
        }
    }

    /// Downcast this function to its typed representation.
    ///
    /// # Panics
    ///
    /// Panics if this function does not use vtable `V`.
    pub fn downcast<V: HigherOrderFunctionVTable>(
        self,
    ) -> Arc<TypedHigherOrderFunctionInstance<V>> {
        self.try_downcast::<V>()
            .map_err(|function| {
                vortex_err!(
                    "failed to downcast higher-order function {} to {}",
                    function.id(),
                    type_name::<V>(),
                )
            })
            .vortex_expect("failed to downcast higher-order function")
    }
}
