// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use itertools::Itertools;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::dtype::DType;
use crate::expr::Expression;
use crate::expr::display::DisplayTreeExpr;
use crate::expr::scope::Scope;
use crate::expr::scope::VariableRef;
use crate::expr::traversal::TraversalOrder;
use crate::expr::traversal::pre_order_visit_down;
use crate::expr::variable::Variable;
use crate::higher_order_fn::HigherOrderFunctionRef;
use crate::scalar_fn::ScalarFnRef;
use crate::scalar_fn::ScalarFnVTable;
use crate::stats::rewrite::StatsRewriteCtx;

/// An [`Expression`] that has been type-checked against a [`Scope`].
///
/// Every node carries its own dtype, so reading one is a field access rather than a walk of the
/// subtree. Holding a `BoundExpression` is proof that the whole tree type-checked.
///
/// Binding is purely logical: it deals only in [`DType`]s and never sees an array, a length, or an
/// encoding.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum BoundExpression {
    /// A scalar function applied to bound children.
    Scalar {
        /// The dtype this node evaluates to.
        dtype: DType,
        /// The scalar function for this node.
        scalar_fn: ScalarFnRef,
        /// The bound children, in argument order.
        ///
        /// Sharing keeps clones cheap even though the iterative [`Drop`] implementation prevents
        /// consumers from destructuring a `BoundExpression` by value.
        children: Arc<Vec<BoundExpression>>,
    },
    /// A type-checked higher-order function call.
    HigherOrder {
        /// The dtype this node evaluates to.
        dtype: DType,
        /// The higher-order function implementation.
        higher_order_fn: HigherOrderFunctionRef,
        /// Bound ordinary expression children.
        children: Arc<Vec<BoundExpression>>,
        /// Bound lambda arguments.
        lambdas: Arc<[TypedLambda]>,
    },
    /// The scope itself. Its dtype is the scope's root dtype.
    Root {
        /// The dtype this node evaluates to.
        dtype: DType,
    },
    /// A resolved reference to a bound variable.
    Variable {
        /// The dtype this node evaluates to.
        dtype: DType,
        /// The source-level name, retained for display and diagnostics.
        variable: Variable,
        /// The lexical location resolved while binding.
        variable_ref: VariableRef,
    },
}

/// A type-checked lambda expression.
///
/// A higher-order function's type-checked lambda argument.
///
/// This is deliberately separate from [`BoundExpression`]: a lambda is not a value and has no
/// dtype. The higher-order function establishes the parameter bindings, binds the body, and stores
/// the resulting typed lambda in its own state.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TypedLambda {
    /// The parameters and their dtypes: the argument side of the function type.
    params: Box<[Variable]>,
    param_dtypes: Box<[DType]>,
    /// The lexical locations assigned to parameters during binding.
    param_refs: Box<[VariableRef]>,
    /// The frame containing this lambda's parameters.
    parameter_frame: usize,
    body: Arc<BoundExpression>,
}

impl TypedLambda {
    /// Bind `lambda` against a scope containing its parameter bindings.
    ///
    /// The higher-order function determines each parameter type and installs it in `scope`
    /// before binding. Keeping the bindings in one scope makes that scope the single source of
    /// truth for both the lambda body and its function signature.
    pub fn bind(lambda: &crate::expr::Lambda, scope: &Scope) -> VortexResult<Self> {
        vortex_ensure!(
            scope.depth() > 0,
            "lambda parameters must be bound in a lexical frame"
        );
        let parameter_frame = scope.depth() - 1;
        let parameter_bindings = lambda
            .params()
            .iter()
            .map(|param| {
                scope
                    .resolve(param)
                    .map(|(dtype, variable_ref)| (dtype.clone(), variable_ref))
                    .ok_or_else(|| {
                        vortex_err!("lambda parameter '{param}' is not bound in its scope")
                    })
            })
            .collect::<VortexResult<Vec<_>>>()?;
        vortex_ensure!(
            parameter_bindings
                .iter()
                .all(|(_, variable_ref)| variable_ref.frame() == parameter_frame),
            "lambda parameters must be bound in the innermost lexical frame"
        );
        let body = lambda.body().bind_scope(scope)?;

        Ok(Self {
            params: lambda.params().to_vec().into_boxed_slice(),
            param_dtypes: parameter_bindings
                .iter()
                .map(|(dtype, _)| dtype.clone())
                .collect(),
            param_refs: parameter_bindings
                .into_iter()
                .map(|(_, variable_ref)| variable_ref)
                .collect(),
            parameter_frame,
            body: Arc::new(body),
        })
    }

    /// The variables this lambda binds, in declaration order.
    pub fn params(&self) -> &[Variable] {
        &self.params
    }

    /// The dtypes of the parameters, in declaration order.
    pub fn param_dtypes(&self) -> &[DType] {
        &self.param_dtypes
    }

    /// The lexical locations of this lambda's parameters.
    pub(crate) fn param_refs(&self) -> &[VariableRef] {
        &self.param_refs
    }

    /// The bound body.
    pub fn body(&self) -> &BoundExpression {
        &self.body
    }

    /// The dtype the body evaluates to — the result side of the function type.
    pub fn body_dtype(&self) -> &DType {
        self.body.dtype()
    }

    /// The outer lexical bindings read directly by this lambda body.
    ///
    /// Nested lambdas are deliberately not traversed: they become closures when their enclosing
    /// higher-order expression is applied, at which point their own free bindings are available.
    pub(crate) fn free_variables(&self) -> Vec<VariableRef> {
        fn collect(
            expr: &BoundExpression,
            parameter_frame: usize,
            variables: &mut Vec<VariableRef>,
        ) {
            match expr {
                BoundExpression::Variable { variable_ref, .. }
                    if variable_ref.frame() < parameter_frame
                        && !variables.contains(variable_ref) =>
                {
                    variables.push(*variable_ref);
                }
                BoundExpression::Scalar { children, .. }
                | BoundExpression::HigherOrder { children, .. } => {
                    for child in children.iter() {
                        collect(child, parameter_frame, variables);
                    }
                }
                BoundExpression::Root { .. } | BoundExpression::Variable { .. } => {}
            }
        }

        let mut variables = Vec::new();
        collect(&self.body, self.parameter_frame, &mut variables);
        variables.sort_by_key(|variable_ref| (variable_ref.frame(), variable_ref.slot()));
        variables
    }

    /// Validate arrays supplied to this lambda's parameter frame.
    pub(crate) fn validate_arguments(
        &self,
        root: &crate::ArrayRef,
        args: &[crate::ArrayRef],
    ) -> VortexResult<()> {
        vortex_ensure!(
            args.len() == self.params.len(),
            "lambda takes {} parameters but was applied with {} arguments",
            self.params.len(),
            args.len()
        );
        for ((param, dtype), arg) in self.params.iter().zip(&self.param_dtypes).zip(args) {
            vortex_ensure!(
                arg.dtype() == dtype,
                "lambda parameter '{param}' expects dtype {dtype}, got {}",
                arg.dtype()
            );
            vortex_ensure!(
                arg.len() == root.len(),
                "lambda parameter '{param}' has length {}, expected {}",
                arg.len(),
                root.len()
            );
        }
        Ok(())
    }
}

impl Display for TypedLambda {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "({}) -> {}", self.params.iter().join(", "), self.body)
    }
}

/// A bound-expression wrapper that compares shared tree identity instead of structure.
#[derive(Clone, Debug)]
pub struct ExactBoundExpr(pub BoundExpression);

impl PartialEq for ExactBoundExpr {
    fn eq(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (
                BoundExpression::Root { dtype: lhs_dtype },
                BoundExpression::Root { dtype: rhs_dtype },
            ) => lhs_dtype == rhs_dtype,
            (
                BoundExpression::Scalar {
                    dtype: lhs_dtype,
                    scalar_fn: lhs_fn,
                    children: lhs_children,
                },
                BoundExpression::Scalar {
                    dtype: rhs_dtype,
                    scalar_fn: rhs_fn,
                    children: rhs_children,
                },
            ) => {
                lhs_fn == rhs_fn
                    && Arc::ptr_eq(lhs_children, rhs_children)
                    && lhs_dtype == rhs_dtype
            }
            (
                BoundExpression::HigherOrder {
                    dtype: lhs_dtype,
                    higher_order_fn: lhs_fn,
                    children: lhs_children,
                    lambdas: lhs_lambdas,
                },
                BoundExpression::HigherOrder {
                    dtype: rhs_dtype,
                    higher_order_fn: rhs_fn,
                    children: rhs_children,
                    lambdas: rhs_lambdas,
                },
            ) => {
                lhs_fn == rhs_fn
                    && Arc::ptr_eq(lhs_children, rhs_children)
                    && Arc::ptr_eq(lhs_lambdas, rhs_lambdas)
                    && lhs_dtype == rhs_dtype
            }
            // No catch-all: a new variant must state its own identity rather than silently
            // comparing unequal, which would put `eq` out of step with `hash`.
            (
                BoundExpression::Variable {
                    dtype: lhs_dtype,
                    variable: lhs_var,
                    variable_ref: lhs_ref,
                },
                BoundExpression::Variable {
                    dtype: rhs_dtype,
                    variable: rhs_var,
                    variable_ref: rhs_ref,
                },
            ) => lhs_var == rhs_var && lhs_ref == rhs_ref && lhs_dtype == rhs_dtype,
            // No catch-all: a new variant must state its own identity, or `eq` drifts out of step
            // with `hash` and keys stop equalling themselves.
            (BoundExpression::Root { .. }, _)
            | (BoundExpression::Scalar { .. }, _)
            | (BoundExpression::HigherOrder { .. }, _)
            | (BoundExpression::Variable { .. }, _) => false,
        }
    }
}

impl Eq for ExactBoundExpr {}

impl Hash for ExactBoundExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // DType differences are resolved by equality. Omitting the potentially lazy dtype keeps
        // identity-keyed cache lookups from deserializing an entire schema just to compute a hash.
        match &self.0 {
            BoundExpression::Root { .. } => state.write_u8(0),
            BoundExpression::Variable {
                variable,
                variable_ref,
                ..
            } => {
                state.write_u8(2);
                variable.hash(state);
                variable_ref.hash(state);
            }
            BoundExpression::Scalar {
                scalar_fn,
                children,
                ..
            } => {
                state.write_u8(1);
                scalar_fn.hash(state);
                Arc::as_ptr(children).hash(state);
            }
            BoundExpression::HigherOrder {
                higher_order_fn,
                children,
                lambdas,
                ..
            } => {
                state.write_u8(3);
                higher_order_fn.hash(state);
                Arc::as_ptr(children).hash(state);
                Arc::as_ptr(lambdas).hash(state);
            }
        }
    }
}

impl BoundExpression {
    /// Create a bound root expression with the given dtype.
    pub fn new_root(dtype: DType) -> Self {
        Self::Root { dtype }
    }

    /// Create a bound scalar node from a scalar function and already-bound children.
    pub fn try_new(
        scalar_fn: ScalarFnRef,
        children: impl IntoIterator<Item = BoundExpression>,
    ) -> VortexResult<Self> {
        let children = Vec::from_iter(children);
        vortex_ensure!(
            scalar_fn.signature().arity().matches(children.len()),
            "Expression arity mismatch: expected {} children but got {}",
            scalar_fn.signature().arity(),
            children.len()
        );

        let arg_dtypes = children
            .iter()
            .map(|child| child.dtype().clone())
            .collect_vec();
        let dtype = scalar_fn.return_dtype(&arg_dtypes)?;

        Ok(Self::Scalar {
            dtype,
            scalar_fn,
            children: children.into(),
        })
    }

    /// Create a bound higher-order node from its bound children and lambdas.
    pub fn try_new_higher_order(
        higher_order_fn: HigherOrderFunctionRef,
        children: impl IntoIterator<Item = BoundExpression>,
        lambdas: impl Into<Box<[TypedLambda]>>,
    ) -> VortexResult<Self> {
        let children = Vec::from_iter(children);
        let lambdas: Arc<[TypedLambda]> = Arc::from(lambdas.into());
        vortex_ensure!(
            higher_order_fn.arity().matches(children.len()),
            "Higher-order expression arity mismatch: expected {} children but got {}",
            higher_order_fn.arity(),
            children.len()
        );
        vortex_ensure!(
            higher_order_fn.lambda_arity() == lambdas.len(),
            "Higher-order expression lambda arity mismatch: expected {} lambdas but got {}",
            higher_order_fn.lambda_arity(),
            lambdas.len()
        );
        let arg_dtypes = children
            .iter()
            .map(|child| child.dtype().clone())
            .collect_vec();
        let dtype = higher_order_fn.return_dtype(&arg_dtypes, &lambdas)?;
        Ok(Self::HigherOrder {
            dtype,
            higher_order_fn,
            children: children.into(),
            lambdas,
        })
    }

    /// Rebuild this node with new bound children, recomputing its dtype.
    pub fn with_children(
        self,
        children: impl IntoIterator<Item = BoundExpression>,
    ) -> VortexResult<Self> {
        let children = Vec::from_iter(children);
        match &self {
            BoundExpression::Scalar { scalar_fn, .. } => Self::try_new(scalar_fn.clone(), children),
            BoundExpression::HigherOrder {
                higher_order_fn,
                lambdas,
                ..
            } => Self::try_new_higher_order(higher_order_fn.clone(), children, lambdas.to_vec()),
            BoundExpression::Root { .. } | BoundExpression::Variable { .. } => {
                vortex_ensure!(
                    children.is_empty(),
                    "{self} cannot have {} children",
                    children.len()
                );
                Ok(self)
            }
        }
    }

    /// The dtype this expression evaluates to.
    pub fn dtype(&self) -> &DType {
        match self {
            Self::Scalar { dtype, .. }
            | Self::HigherOrder { dtype, .. }
            | Self::Root { dtype }
            | Self::Variable { dtype, .. } => dtype,
        }
    }

    /// The bound children of this node, in argument order. Empty for [`BoundExpression::Root`].
    pub fn children(&self) -> &[BoundExpression] {
        match self {
            Self::Scalar { children, .. } | Self::HigherOrder { children, .. } => {
                children.as_slice()
            }
            Self::Root { .. } | Self::Variable { .. } => &[],
        }
    }

    /// Return the child at `index`.
    pub fn child(&self, index: usize) -> &BoundExpression {
        &self.children()[index]
    }

    /// The scalar function for this node, or `None` if it is the scope root.
    pub fn as_scalar(&self) -> Option<&ScalarFnRef> {
        match self {
            Self::Scalar { scalar_fn, .. } => Some(scalar_fn),
            Self::HigherOrder { .. } | Self::Root { .. } | Self::Variable { .. } => None,
        }
    }

    /// The higher-order function for this node, if any.
    pub fn as_higher_order(&self) -> Option<&HigherOrderFunctionRef> {
        match self {
            Self::HigherOrder {
                higher_order_fn, ..
            } => Some(higher_order_fn),
            Self::Scalar { .. } | Self::Root { .. } | Self::Variable { .. } => None,
        }
    }

    /// The bound lambda arguments of this node.
    pub fn lambdas(&self) -> &[TypedLambda] {
        match self {
            Self::HigherOrder { lambdas, .. } => lambdas,
            Self::Scalar { .. } | Self::Root { .. } | Self::Variable { .. } => &[],
        }
    }

    /// Return whether this node uses the given scalar-function vtable.
    pub fn is<V: ScalarFnVTable>(&self) -> bool {
        self.as_scalar().is_some_and(ScalarFnRef::is::<V>)
    }

    /// Return whether this expression tree contains a node using the given scalar-function vtable.
    pub fn contains<V: ScalarFnVTable>(&self) -> VortexResult<bool> {
        let mut contains = false;
        pre_order_visit_down(self, |node| {
            if node.is::<V>() {
                contains = true;
                return Ok(TraversalOrder::Stop);
            }
            Ok(TraversalOrder::Continue)
        })?;
        Ok(contains)
    }

    /// Return the typed scalar-function options when this node uses the given vtable.
    pub fn as_opt<V: ScalarFnVTable>(&self) -> Option<&V::Options> {
        self.as_scalar().and_then(ScalarFnRef::as_opt::<V>)
    }

    /// Return the typed scalar-function options for this node.
    ///
    /// # Panics
    ///
    /// Panics when this node is the scope root or uses a different scalar-function vtable.
    pub fn as_<V: ScalarFnVTable>(&self) -> &V::Options {
        self.as_opt::<V>()
            .vortex_expect("Bound expression options type mismatch")
    }

    /// The variable this node resolves to, if this node is a variable reference.
    pub fn as_variable(&self) -> Option<&Variable> {
        match self {
            Self::Variable { variable, .. } => Some(variable),
            Self::Scalar { .. } | Self::HigherOrder { .. } | Self::Root { .. } => None,
        }
    }

    /// Whether this node is the scope root.
    pub fn is_root(&self) -> bool {
        matches!(self, Self::Root { .. })
    }

    /// Return whether every scope root in this expression has `dtype`.
    ///
    /// Expressions without a scope root, such as literals, match every dtype.
    pub fn is_root_bound_to(&self, dtype: &DType) -> bool {
        let mut is_bound_to = true;
        pre_order_visit_down(self, |node| {
            if node.is_root() && node.dtype() != dtype {
                is_bound_to = false;
                return Ok(TraversalOrder::Stop);
            }
            Ok(TraversalOrder::Continue)
        })
        .vortex_expect("bound expression traversal cannot not fail");
        is_bound_to
    }

    /// Return an expression that proves this predicate is definitely false from statistics.
    pub fn falsify(&self, session: &VortexSession) -> VortexResult<Option<BoundExpression>> {
        StatsRewriteCtx::new(session).falsify(self)
    }

    /// Return an expression that proves this predicate is definitely true from statistics.
    pub fn satisfy(&self, session: &VortexSession) -> VortexResult<Option<BoundExpression>> {
        StatsRewriteCtx::new(session).satisfy(self)
    }

    /// Display the bound expression as a formatted tree structure.
    pub fn display_tree(&self) -> impl Display {
        DisplayTreeExpr(self)
    }
}

impl Display for BoundExpression {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scalar { scalar_fn, .. } => scalar_fn.fmt_sql(self, f),
            Self::HigherOrder {
                higher_order_fn,
                children,
                lambdas,
                ..
            } => {
                write!(f, "{higher_order_fn}(")?;
                for (index, child) in children.iter().enumerate() {
                    if index > 0 {
                        write!(f, ", ")?;
                    }
                    Display::fmt(child, f)?;
                }
                for (index, lambda) in lambdas.iter().enumerate() {
                    if !children.is_empty() || index > 0 {
                        write!(f, ", ")?;
                    }
                    Display::fmt(lambda, f)?;
                }
                write!(f, ")")
            }
            Self::Root { .. } => f.write_str("$"),
            Self::Variable { variable, .. } => write!(f, "${variable}"),
        }
    }
}

impl Expression {
    /// Bind this expression against a root dtype, type-checking every node in a single walk.
    ///
    /// The returned tree carries a dtype on each node, so callers needing types at more than one
    /// node should bind once and read fields rather than calling
    /// [`return_dtype`](Expression::return_dtype) repeatedly.
    pub fn bind(&self, dtype: &DType) -> VortexResult<BoundExpression> {
        self.bind_scope(&Scope::new(dtype.clone()))
    }

    /// Bind this expression against an explicit [`Scope`].
    pub fn bind_scope(&self, scope: &Scope) -> VortexResult<BoundExpression> {
        match self {
            Expression::Root => Ok(BoundExpression::new_root(scope.root().clone())),
            Expression::Variable(variable) => {
                let Some((dtype, variable_ref)) = scope.resolve(variable) else {
                    vortex_bail!("unbound variable '{variable}'");
                };
                Ok(BoundExpression::Variable {
                    dtype: dtype.clone(),
                    variable: variable.clone(),
                    variable_ref,
                })
            }
            Expression::Scalar {
                scalar_fn,
                children,
            } => {
                let children: Vec<_> = children
                    .iter()
                    .map(|child| child.bind_scope(scope))
                    .try_collect()?;
                BoundExpression::try_new(scalar_fn.clone(), children)
            }
            Expression::HigherOrder {
                higher_order_fn,
                children,
                lambdas,
            } => {
                let children: Vec<_> = children
                    .iter()
                    .map(|child| child.bind_scope(scope))
                    .try_collect()?;
                let dtypes = children
                    .iter()
                    .map(|child| child.dtype().clone())
                    .collect_vec();
                let lambdas = higher_order_fn.bind_lambdas(scope, &dtypes, lambdas)?;
                BoundExpression::try_new_higher_order(higher_order_fn.clone(), children, lambdas)
            }
        }
    }
}

/// Iterative drop to avoid stack overflows on deep trees.
impl Drop for BoundExpression {
    fn drop(&mut self) {
        let (Self::Scalar { children, .. } | Self::HigherOrder { children, .. }) = self else {
            return;
        };
        let Some(children) = Arc::get_mut(children) else {
            return;
        };

        let mut to_drop = std::mem::take(children);
        while let Some(mut child) = to_drop.pop() {
            if let BoundExpression::Scalar { children, .. }
            | BoundExpression::HigherOrder { children, .. } = &mut child
                && let Some(grandchildren) = Arc::get_mut(children)
            {
                to_drop.append(grandchildren);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use super::*;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::checked_add;
    use crate::expr::col;
    use crate::expr::eq;
    use crate::expr::lambda;
    use crate::expr::lit;
    use crate::expr::root;
    use crate::expr::test_harness::struct_dtype;
    use crate::expr::var;
    use crate::scalar_fn::fns::is_not_null::IsNotNull;
    use crate::scalar_fn::fns::literal::Literal;

    fn scope() -> Scope {
        Scope::new(struct_dtype())
    }

    #[test]
    fn root_binds_to_the_scope() -> VortexResult<()> {
        let bound = root().bind_scope(&scope())?;
        assert!(bound.is_root());
        assert_eq!(bound.dtype(), &struct_dtype());
        assert_eq!(bound, BoundExpression::new_root(struct_dtype()));
        Ok(())
    }

    #[test]
    fn every_node_carries_its_dtype() -> VortexResult<()> {
        let expr = eq(col("a"), lit(1_i32));
        let bound = expr.bind_scope(&scope())?;

        assert_eq!(bound.dtype(), &DType::Bool(Nullability::NonNullable));

        let lhs = &bound.children()[0];
        assert_eq!(
            lhs.dtype(),
            &DType::Primitive(PType::I32, Nullability::NonNullable)
        );
        assert_eq!(lhs.children()[0].dtype(), &struct_dtype());
        Ok(())
    }

    #[test]
    fn bind_agrees_with_return_dtype() -> VortexResult<()> {
        for expr in [root(), col("a"), eq(col("a"), lit(1_i32)), lit(true)] {
            assert_eq!(
                expr.bind(&struct_dtype())?.dtype(),
                &expr.return_dtype(&struct_dtype())?,
                "disagreement for {expr}"
            );
        }
        Ok(())
    }

    #[test]
    fn contains_scalar_function() -> VortexResult<()> {
        let bound = eq(col("a"), lit(1_i32)).bind_scope(&scope())?;
        assert!(bound.contains::<Literal>()?);
        assert!(!root().bind_scope(&scope())?.contains::<Literal>()?);
        Ok(())
    }

    #[test]
    fn bound_to_checks_every_root() -> VortexResult<()> {
        let dtype = struct_dtype();
        let bound = eq(col("a"), col("a")).bind(&dtype)?;
        assert!(bound.is_root_bound_to(&dtype));
        assert!(!bound.is_root_bound_to(&DType::Bool(Nullability::NonNullable)));
        assert!(
            lit(true)
                .bind(&dtype)?
                .is_root_bound_to(&DType::Bool(Nullability::NonNullable))
        );
        Ok(())
    }

    #[test]
    fn bound_display_matches_unbound() -> VortexResult<()> {
        for expr in [root(), col("a"), eq(col("a"), lit(1_i32)), lit(true)] {
            let bound = expr.bind_scope(&scope())?;
            assert_eq!(bound.to_string(), expr.to_string());
            assert_eq!(
                bound.display_tree().to_string(),
                expr.display_tree().to_string()
            );
        }
        Ok(())
    }

    #[test]
    fn variable_binds_to_its_scope() -> VortexResult<()> {
        let scope = scope().with_bindings([(
            Variable::new("value"),
            DType::Primitive(PType::I64, Nullability::Nullable),
        )])?;

        let bound = var("value").bind_scope(&scope)?;
        let variable = bound
            .as_variable()
            .vortex_expect("variable must remain bound");

        assert_eq!(variable, &Variable::new("value"));
        assert_eq!(
            bound.dtype(),
            &DType::Primitive(PType::I64, Nullability::Nullable)
        );
        Ok(())
    }

    #[test]
    fn variable_validity_is_deferred_to_its_bound_array() -> VortexResult<()> {
        let scope = scope().with_bindings([(
            Variable::new("value"),
            DType::Primitive(PType::I32, Nullability::Nullable),
        )])?;
        let expression = checked_add(var("value"), lit(1_i32));

        let validity = expression.validity()?;
        assert!(validity.contains::<IsNotNull>()?);
        assert_eq!(
            validity.bind_scope(&scope)?.dtype(),
            &DType::Bool(Nullability::NonNullable)
        );
        Ok(())
    }

    #[test]
    fn duplicate_lambda_parameters_are_rejected() {
        assert!(lambda(["x", "x"], var("x")).is_err());
    }

    #[test]
    fn lambda_signature_comes_from_its_scope() -> VortexResult<()> {
        let lambda = lambda(["value"], var("value"))?;
        let value_dtype = DType::Primitive(PType::I64, Nullability::Nullable);
        let lambda_scope =
            scope().with_bindings([(Variable::new("value"), value_dtype.clone())])?;

        let bound = TypedLambda::bind(&lambda, &lambda_scope)?;
        assert_eq!(bound.body_dtype(), &value_dtype);
        assert_eq!(bound.param_dtypes(), &[value_dtype]);
        assert!(TypedLambda::bind(&lambda, &scope()).is_err());
        Ok(())
    }

    #[test]
    fn lambda_captures_only_direct_free_variables() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let lambda_scope = scope()
            .with_bindings([(Variable::new("outer"), dtype.clone())])?
            .with_bindings([(Variable::new("middle"), dtype.clone())])?
            .with_bindings([(Variable::new("value"), dtype)])?;
        let lambda = lambda(
            ["value"],
            checked_add(checked_add(var("value"), var("middle")), var("outer")),
        )?;

        let typed = TypedLambda::bind(&lambda, &lambda_scope)?;
        let free_variables = typed.free_variables();

        assert_eq!(free_variables.len(), 2);
        assert_eq!(
            free_variables
                .iter()
                .map(|variable_ref| (variable_ref.frame(), variable_ref.slot()))
                .collect_vec(),
            vec![(0, 0), (1, 0)]
        );
        Ok(())
    }

    #[test]
    fn structural_and_exact_equality_are_distinct() -> VortexResult<()> {
        let expr = eq(col("a"), lit(1_i32));
        let bound = expr.bind_scope(&scope())?;
        let independently_bound = expr.bind_scope(&scope())?;

        assert_eq!(bound, independently_bound);
        assert_eq!(ExactBoundExpr(bound.clone()), ExactBoundExpr(bound.clone()));
        assert_ne!(ExactBoundExpr(bound), ExactBoundExpr(independently_bound));
        Ok(())
    }

    #[test]
    fn binding_reports_a_type_error() {
        assert!(eq(col("a"), lit("nope")).bind_scope(&scope()).is_err());
    }
}
