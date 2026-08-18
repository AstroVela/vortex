// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hasher;
use std::sync::Arc;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::expr::BoundLambdaArgs;
use crate::expr::Lambda;
use crate::expr::display::HigherOrderExprDisplay;
use crate::higher_order_fn::HigherOrderFunctionId;
use crate::higher_order_fn::HigherOrderFunctionRef;
use crate::higher_order_fn::HigherOrderFunctionVTable;
use crate::higher_order_fn::LambdaCall;
use crate::higher_order_fn::LambdaSignature;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::validity::Validity;

/// A typed higher-order-function instance, pairing a vtable with its per-call options.
pub struct TypedHigherOrderFunctionInstance<V: HigherOrderFunctionVTable> {
    vtable: V,
    options: V::Options,
}

impl<V: HigherOrderFunctionVTable> TypedHigherOrderFunctionInstance<V> {
    /// Create a typed higher-order-function instance.
    pub fn new(vtable: V, options: V::Options) -> Self {
        Self { vtable, options }
    }

    /// Return the vtable.
    pub fn vtable(&self) -> &V {
        &self.vtable
    }

    /// Return the typed options.
    pub fn options(&self) -> &V::Options {
        &self.options
    }

    /// Erase the concrete type information.
    pub fn erased(self) -> HigherOrderFunctionRef {
        HigherOrderFunctionRef(Arc::new(self))
    }
}

pub(super) trait DynHigherOrderFunction: 'static + Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn id(&self) -> HigherOrderFunctionId;
    fn options_any(&self) -> &dyn Any;
    fn arity(&self) -> Arity;
    fn lambda_arity(&self) -> usize;
    fn child_name(&self, child_idx: usize) -> ChildName;
    fn coerce_input_args(&self, args: &[DType]) -> VortexResult<Vec<DType>>;
    fn lambda_signatures(
        &self,
        arg_dtypes: &[DType],
        lambdas: &[&Lambda],
    ) -> VortexResult<Box<[LambdaSignature]>>;
    fn return_dtype(&self, args: &[DType], lambdas: BoundLambdaArgs<'_>) -> VortexResult<DType>;
    fn validity(&self, args: &[ArrayRef], lambdas: BoundLambdaArgs<'_>) -> VortexResult<Validity>;
    fn apply(
        &self,
        args: &[ArrayRef],
        lambdas: &[LambdaCall],
        execution_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef>;
    fn is_fallible(&self, lambdas: BoundLambdaArgs<'_>) -> bool;
    fn fmt_sql(&self, expr: &dyn HigherOrderExprDisplay, f: &mut Formatter<'_>) -> fmt::Result;
    fn options_serialize(&self) -> VortexResult<Option<Vec<u8>>>;
    fn options_eq(&self, other_options: &dyn Any) -> bool;
    fn options_hash(&self, hasher: &mut dyn Hasher);
    fn options_display(&self, f: &mut Formatter<'_>) -> fmt::Result;
    fn options_debug(&self, f: &mut Formatter<'_>) -> fmt::Result;
}

impl<V: HigherOrderFunctionVTable> DynHigherOrderFunction for TypedHigherOrderFunctionInstance<V> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn id(&self) -> HigherOrderFunctionId {
        V::id(&self.vtable)
    }

    fn options_any(&self) -> &dyn Any {
        &self.options
    }

    fn arity(&self) -> Arity {
        V::arity(&self.vtable, &self.options)
    }

    fn lambda_arity(&self) -> usize {
        V::lambda_arity(&self.vtable, &self.options)
    }

    fn child_name(&self, child_idx: usize) -> ChildName {
        V::child_name(&self.vtable, &self.options, child_idx)
    }

    fn coerce_input_args(&self, args: &[DType]) -> VortexResult<Vec<DType>> {
        V::coerce_input_args(&self.vtable, &self.options, args)
    }

    fn lambda_signatures(
        &self,
        arg_dtypes: &[DType],
        lambdas: &[&Lambda],
    ) -> VortexResult<Box<[LambdaSignature]>> {
        V::lambda_signatures(&self.vtable, &self.options, arg_dtypes, lambdas)
    }

    fn return_dtype(&self, args: &[DType], lambdas: BoundLambdaArgs<'_>) -> VortexResult<DType> {
        V::return_dtype(&self.vtable, &self.options, args, lambdas)
    }

    fn validity(&self, args: &[ArrayRef], lambdas: BoundLambdaArgs<'_>) -> VortexResult<Validity> {
        V::validity(&self.vtable, &self.options, args, lambdas)
    }

    fn apply(
        &self,
        args: &[ArrayRef],
        lambdas: &[LambdaCall],
        execution_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        V::apply(&self.vtable, &self.options, args, lambdas, execution_ctx)
    }

    fn is_fallible(&self, lambdas: BoundLambdaArgs<'_>) -> bool {
        V::is_fallible(&self.vtable, &self.options, lambdas)
    }

    fn fmt_sql(&self, expr: &dyn HigherOrderExprDisplay, f: &mut Formatter<'_>) -> fmt::Result {
        V::fmt_sql(&self.vtable, &self.options, expr, f)
    }

    fn options_serialize(&self) -> VortexResult<Option<Vec<u8>>> {
        V::serialize(&self.vtable, &self.options)
    }

    fn options_eq(&self, other_options: &dyn Any) -> bool {
        other_options
            .downcast_ref::<V::Options>()
            .is_some_and(|options| self.options == *options)
    }

    fn options_hash(&self, mut hasher: &mut dyn Hasher) {
        std::hash::Hash::hash(&self.options, &mut hasher);
    }

    fn options_display(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.options, f)
    }

    fn options_debug(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.options, f)
    }
}
