// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_utils::aliases::hash_map::HashMap;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::HigherOrderFnArray;
use crate::arrays::ScalarFnArray;
use crate::expr::BoundExpression;
use crate::expr::TypedLambda;
use crate::expr::VariableRef;
use crate::higher_order_fn::LambdaClosure;
use crate::optimizer::ArrayOptimizer;
use crate::scalar_fn::fns::literal::Literal;

/// Applies bound expressions against their root array and lexical bindings.
///
/// Application constructs lazy arrays. A higher-order expression therefore produces a
/// [`HigherOrderFnArray`], retaining lambda closures for the array vtable to execute later.
pub struct ExprApplyCtx {
    root: ArrayRef,
    bindings: HashMap<VariableRef, ArrayRef>,
}

impl ExprApplyCtx {
    /// Create an application context for a top-level expression.
    pub fn new(root: ArrayRef) -> Self {
        Self::with_bindings(root, HashMap::new())
    }

    /// Create an application context for a lambda invocation.
    pub(crate) fn with_bindings(root: ArrayRef, bindings: HashMap<VariableRef, ArrayRef>) -> Self {
        Self { root, bindings }
    }

    fn binding(&self, variable_ref: VariableRef) -> Option<&ArrayRef> {
        self.bindings.get(&variable_ref)
    }

    fn close_lambda(
        &self,
        lambda: TypedLambda,
        capture_slots: &mut Vec<ArrayRef>,
    ) -> VortexResult<LambdaClosure> {
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
        Ok(LambdaClosure::new(lambda, captures, capture_slots))
    }
}

impl BoundExpression {
    /// Apply this expression in `ctx`'s current lexical and row domain.
    pub(crate) fn apply(&self, ctx: &mut ExprApplyCtx) -> VortexResult<ArrayRef> {
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
                let children = children
                    .iter()
                    .map(|child| child.apply(ctx))
                    .try_collect()?;
                let mut capture_slots = Vec::new();
                let lambdas = lambdas
                    .iter()
                    .cloned()
                    .map(|lambda| ctx.close_lambda(lambda, &mut capture_slots))
                    .collect::<VortexResult<Vec<_>>>()?;
                return Ok(HigherOrderFnArray::try_new_with_len(
                    higher_order_fn.clone(),
                    children,
                    lambdas,
                    capture_slots,
                    ctx.root.len(),
                )?
                .into_array());
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
