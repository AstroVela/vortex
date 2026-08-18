// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_utils::aliases::hash_map::HashMap;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::ConstantArray;
use crate::arrays::ScalarFnArray;
use crate::expr::BoundExpression;
use crate::expr::BoundLambda;
use crate::expr::Expression;
use crate::expr::VariableRef;
use crate::higher_order_fn::LambdaCall;
use crate::optimizer::ArrayOptimizer;
use crate::scalar_fn::fns::literal::Literal;

/// State shared while an expression is applied to an array.
///
/// The context tracks the current root array, lexical bindings, and the execution context used by
/// structural lowerings. It is an implementation detail of [`ArrayRef::apply`] rather than an
/// alternative evaluation layer.
pub(crate) struct ApplyCtx<'a> {
    root: ArrayRef,
    bindings: HashMap<VariableRef, ArrayRef>,
    execution_ctx: Option<ApplyExecutionCtx<'a>>,
}

enum ApplyExecutionCtx<'a> {
    Borrowed(&'a mut ExecutionCtx),
    Owned(ExecutionCtx),
}

impl<'a> ApplyCtx<'a> {
    fn new(root: ArrayRef) -> Self {
        Self {
            root,
            bindings: HashMap::new(),
            execution_ctx: None,
        }
    }

    fn with_execution_ctx(root: ArrayRef, execution_ctx: &'a mut ExecutionCtx) -> Self {
        Self {
            root,
            bindings: HashMap::new(),
            execution_ctx: Some(ApplyExecutionCtx::Borrowed(execution_ctx)),
        }
    }

    pub(crate) fn with_bindings(
        root: ArrayRef,
        bindings: HashMap<VariableRef, ArrayRef>,
        execution_ctx: &'a mut ExecutionCtx,
    ) -> Self {
        Self {
            root,
            bindings,
            execution_ctx: Some(ApplyExecutionCtx::Borrowed(execution_ctx)),
        }
    }

    fn binding(&self, variable_ref: VariableRef) -> Option<&ArrayRef> {
        self.bindings.get(&variable_ref)
    }

    fn execution_ctx(&mut self) -> &mut ExecutionCtx {
        if self.execution_ctx.is_none() {
            self.execution_ctx = Some(ApplyExecutionCtx::Owned(
                array_session().create_execution_ctx(),
            ));
        }
        match self.execution_ctx.as_mut() {
            Some(ApplyExecutionCtx::Borrowed(ctx)) => ctx,
            Some(ApplyExecutionCtx::Owned(ctx)) => ctx,
            None => unreachable!("an execution context was initialized above"),
        }
    }

    fn close_lambda(&self, lambda: BoundLambda) -> VortexResult<LambdaCall> {
        let captures = lambda
            .free_variables()
            .into_iter()
            .map(|variable_ref| {
                self.binding(variable_ref)
                    .cloned()
                    .map(|array| (variable_ref, array))
                    .ok_or_else(|| {
                        vortex_err!(
                            "cannot capture missing lexical binding at frame {}, slot {}",
                            variable_ref.frame(),
                            variable_ref.slot()
                        )
                    })
            })
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(LambdaCall::new(lambda, captures))
    }
}

impl ArrayRef {
    /// Apply a bound expression to this array, producing a new array in constant time.
    pub fn apply_bound(self, expr: &BoundExpression) -> VortexResult<ArrayRef> {
        expr.apply(&mut ApplyCtx::new(self))
    }

    /// Apply a bound expression using an explicit execution context for structural lowerings.
    pub fn apply_bound_with_ctx(
        self,
        expr: &BoundExpression,
        execution_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        expr.apply(&mut ApplyCtx::with_execution_ctx(self, execution_ctx))
    }

    /// Apply the expression to this array, producing a new array in constant time.
    pub fn apply(self, expr: &Expression) -> VortexResult<ArrayRef> {
        let bound = expr.bind(self.dtype())?;
        self.apply_bound(&bound)
    }

    /// Apply an expression using an explicit execution context for structural lowerings.
    pub fn apply_with_ctx(
        self,
        expr: &Expression,
        execution_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let bound = expr.bind(self.dtype())?;
        self.apply_bound_with_ctx(&bound, execution_ctx)
    }
}

impl BoundExpression {
    /// Apply this expression in `ctx`'s current lexical and row domain.
    pub(crate) fn apply(&self, ctx: &mut ApplyCtx<'_>) -> VortexResult<ArrayRef> {
        let (scalar_fn, children) = match self {
            BoundExpression::Root { .. } => return Ok(ctx.root.clone()),
            BoundExpression::Variable {
                variable,
                variable_ref,
                ..
            } => {
                return ctx
                    .binding(*variable_ref)
                    .cloned()
                    .ok_or_else(|| vortex_err!("cannot apply unbound variable '{variable}'"));
            }
            BoundExpression::HigherOrder {
                higher_order_fn,
                children,
                lambdas,
                ..
            } => {
                let children: Vec<_> = children
                    .iter()
                    .map(|child| child.apply(ctx))
                    .try_collect()?;
                let lambdas = lambdas
                    .iter()
                    .map(|lambda| {
                        lambda
                            .as_lambda()
                            .cloned()
                            .ok_or_else(|| {
                                vortex_err!(
                                    "higher-order expression contained a non-lambda argument"
                                )
                            })
                            .and_then(|lambda| ctx.close_lambda(lambda))
                    })
                    .collect::<VortexResult<Vec<_>>>()?;
                return higher_order_fn.apply(&children, &lambdas, ctx.execution_ctx());
            }
            BoundExpression::Lambda { .. } => {
                return Err(vortex_err!(
                    "a lambda can be applied only by a higher-order function"
                ));
            }
            BoundExpression::Scalar {
                scalar_fn,
                children,
                ..
            } => (scalar_fn, children),
        };

        if let Some(scalar) = scalar_fn.as_opt::<Literal>() {
            return Ok(ConstantArray::new(scalar.clone(), ctx.root.len()).into_array());
        }

        let children = children
            .iter()
            .map(|child| child.apply(ctx))
            .try_collect()?;
        let array = ScalarFnArray::try_new_with_len(scalar_fn.clone(), children, ctx.root.len())?
            .into_array();

        array.optimize()
    }
}
