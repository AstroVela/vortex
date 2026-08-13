// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::dtype::DType;
use crate::expr::Lambda;
use crate::expr::Scope;
use crate::expr::TypedLambda;
use crate::higher_order_fn::HigherOrderFunctionId;
use crate::higher_order_fn::HigherOrderFunctionVTable;
use crate::higher_order_fn::LambdaCall;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::validity::Validity;

trait DynHigherOrderFunction: 'static + Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn id(&self) -> HigherOrderFunctionId;
    fn arity(&self) -> Arity;
    fn lambda_arity(&self) -> usize;
    fn child_name(&self, child_idx: usize) -> ChildName;
    fn serialize(&self) -> VortexResult<Option<Vec<u8>>>;
    fn deserialize(&self, metadata: &[u8], session: &VortexSession) -> VortexResult<()>;
    fn bind_lambdas(
        &self,
        scope: &Scope,
        arg_dtypes: &[DType],
        lambdas: &[Lambda],
    ) -> VortexResult<Box<[TypedLambda]>>;
    fn return_dtype(&self, args: &[DType], lambdas: &[TypedLambda]) -> VortexResult<DType>;
    fn validity(&self, args: &[ArrayRef], lambdas: &[TypedLambda]) -> VortexResult<Validity>;
    fn execute(
        &self,
        args: &[ArrayRef],
        lambdas: &[LambdaCall<'_>],
        ctx: &mut crate::ExecutionCtx,
    ) -> VortexResult<ArrayRef>;
    fn is_fallible(&self, lambdas: &[TypedLambda]) -> bool;
}

impl<V: HigherOrderFunctionVTable> DynHigherOrderFunction for V {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn id(&self) -> HigherOrderFunctionId {
        V::id(self)
    }

    fn arity(&self) -> Arity {
        V::arity(self)
    }

    fn lambda_arity(&self) -> usize {
        V::lambda_arity(self)
    }

    fn child_name(&self, child_idx: usize) -> ChildName {
        V::child_name(self, child_idx)
    }

    fn serialize(&self) -> VortexResult<Option<Vec<u8>>> {
        V::serialize(self)
    }

    fn deserialize(&self, metadata: &[u8], session: &VortexSession) -> VortexResult<()> {
        V::deserialize(self, metadata, session)
    }

    fn bind_lambdas(
        &self,
        scope: &Scope,
        arg_dtypes: &[DType],
        lambdas: &[Lambda],
    ) -> VortexResult<Box<[TypedLambda]>> {
        V::bind_lambdas(self, scope, arg_dtypes, lambdas)
    }

    fn return_dtype(&self, args: &[DType], lambdas: &[TypedLambda]) -> VortexResult<DType> {
        V::return_dtype(self, args, lambdas)
    }

    fn validity(&self, args: &[ArrayRef], lambdas: &[TypedLambda]) -> VortexResult<Validity> {
        V::validity(self, args, lambdas)
    }

    fn execute(
        &self,
        args: &[ArrayRef],
        lambdas: &[LambdaCall<'_>],
        ctx: &mut crate::ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        V::execute(self, args, lambdas, ctx)
    }

    fn is_fallible(&self, lambdas: &[TypedLambda]) -> bool {
        V::is_fallible(self, lambdas)
    }
}

/// A type-erased higher-order function vtable.
#[derive(Clone)]
pub struct HigherOrderFunctionRef(Arc<dyn DynHigherOrderFunction>);

impl HigherOrderFunctionRef {
    /// Create a reference to `vtable`.
    pub fn new<V: HigherOrderFunctionVTable>(vtable: V) -> Self {
        Self(Arc::new(vtable))
    }

    /// The function's global identifier.
    pub fn id(&self) -> HigherOrderFunctionId {
        self.0.id()
    }

    /// Whether this is an instance of `V`.
    pub fn is<V: HigherOrderFunctionVTable>(&self) -> bool {
        self.0.as_any().is::<V>()
    }

    pub fn arity(&self) -> Arity {
        self.0.arity()
    }

    pub fn lambda_arity(&self) -> usize {
        self.0.lambda_arity()
    }

    pub fn child_name(&self, child_idx: usize) -> ChildName {
        self.0.child_name(child_idx)
    }

    pub fn serialize(&self) -> VortexResult<Option<Vec<u8>>> {
        self.0.serialize()
    }

    pub(crate) fn deserialize(&self, metadata: &[u8], session: &VortexSession) -> VortexResult<()> {
        self.0.deserialize(metadata, session)
    }

    pub(crate) fn bind_lambdas(
        &self,
        scope: &Scope,
        arg_dtypes: &[DType],
        lambdas: &[Lambda],
    ) -> VortexResult<Box<[TypedLambda]>> {
        self.0.bind_lambdas(scope, arg_dtypes, lambdas)
    }

    pub fn return_dtype(&self, args: &[DType], lambdas: &[TypedLambda]) -> VortexResult<DType> {
        self.0.return_dtype(args, lambdas)
    }

    pub fn validity(&self, args: &[ArrayRef], lambdas: &[TypedLambda]) -> VortexResult<Validity> {
        self.0.validity(args, lambdas)
    }

    pub fn execute(
        &self,
        args: &[ArrayRef],
        lambdas: &[LambdaCall<'_>],
        ctx: &mut crate::ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        self.0.execute(args, lambdas, ctx)
    }

    pub fn is_fallible(&self, lambdas: &[TypedLambda]) -> bool {
        self.0.is_fallible(lambdas)
    }
}

impl Debug for HigherOrderFunctionRef {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_tuple("HigherOrderFunctionRef")
            .field(&self.id())
            .finish()
    }
}

impl Display for HigherOrderFunctionRef {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.id(), f)
    }
}

impl PartialEq for HigherOrderFunctionRef {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}

impl Eq for HigherOrderFunctionRef {}

impl Hash for HigherOrderFunctionRef {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id().hash(state);
    }
}
