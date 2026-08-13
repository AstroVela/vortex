// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::dtype::DType;
use crate::expr::Lambda;
use crate::expr::Scope;
use crate::expr::TypedLambda;
use crate::higher_order_fn::HigherOrderFunctionId;
use crate::higher_order_fn::LambdaCall;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::validity::Validity;

/// The implementation contract for a higher-order function.
///
/// Unlike a [`ScalarFnVTable`](crate::scalar_fn::ScalarFnVTable), a higher-order function owns
/// one or more lambdas in addition to its ordinary expression children. It determines the lambda
/// parameter types while binding, then receives the resulting [`TypedLambda`]s at execution.
pub trait HigherOrderFunctionVTable: 'static + Sized + Clone + Debug + Send + Sync {
    /// The globally unique identifier for this higher-order function.
    fn id(&self) -> HigherOrderFunctionId;

    /// The arity of the ordinary expression children.
    fn arity(&self) -> Arity;

    /// The number of lambda arguments.
    fn lambda_arity(&self) -> usize;

    /// The name of an ordinary expression child.
    fn child_name(&self, child_idx: usize) -> ChildName;

    /// Serialize any vtable-level metadata.
    ///
    /// The lambda syntax is serialized by [`Expression`](crate::expr::Expression), not here.
    fn serialize(&self) -> VortexResult<Option<Vec<u8>>> {
        Ok(None)
    }

    /// Validate metadata while deserializing an instance of this vtable.
    fn deserialize(&self, _metadata: &[u8], _session: &VortexSession) -> VortexResult<()> {
        vortex_bail!("higher-order function {} is not deserializable", self.id());
    }

    /// Bind `lambdas` after the ordinary children have been bound.
    ///
    /// The implementation establishes each lambda's parameter frame and root scope from
    /// `scope` and `arg_dtypes`.
    fn bind_lambdas(
        &self,
        scope: &Scope,
        arg_dtypes: &[DType],
        lambdas: &[Lambda],
    ) -> VortexResult<Box<[TypedLambda]>>;

    /// Return the dtype after both ordinary children and lambdas have been bound.
    fn return_dtype(&self, arg_dtypes: &[DType], lambdas: &[TypedLambda]) -> VortexResult<DType>;

    /// Return the result validity without executing the function.
    ///
    /// Most functions only know that a nullable result is potentially valid everywhere, but a
    /// function such as `list_transform` can preserve validity from one of its inputs exactly.
    fn validity(&self, args: &[ArrayRef], lambdas: &[TypedLambda]) -> VortexResult<Validity> {
        let arg_dtypes = args
            .iter()
            .map(|arg| arg.dtype().clone())
            .collect::<Vec<_>>();
        Ok(Validity::from(
            self.return_dtype(&arg_dtypes, lambdas)?.nullability(),
        ))
    }

    /// Execute the function over its already-applied ordinary arguments and closed lambdas.
    ///
    /// Lambdas build their bodies lazily when invoked. The returned array is then executed by the
    /// normal array scheduler, so expression application remains distinct from execution.
    fn execute(
        &self,
        args: &[ArrayRef],
        lambdas: &[LambdaCall<'_>],
        ctx: &mut crate::ExecutionCtx,
    ) -> VortexResult<ArrayRef>;

    /// Whether evaluation of the function can raise a semantic error.
    fn is_fallible(&self, _lambdas: &[TypedLambda]) -> bool {
        true
    }
}
