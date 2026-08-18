// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::dtype::DType;
use crate::expr::display::DisplayTreeExpr;
use crate::expr::is_not_null;
use crate::expr::lambda::Lambda;
use crate::expr::scope::Scope;
use crate::expr::traversal::TraversalOrder;
use crate::expr::traversal::pre_order_visit_down;
use crate::expr::variable::Variable;
use crate::higher_order_fn::HigherOrderFunctionRef;
use crate::scalar_fn::ScalarFnRef;
use crate::scalar_fn::ScalarFnVTable;

/// An empty child slice, returned by [`Expression::children`] for childless variants.
const NO_CHILDREN: &[Expression] = &[];

/// A node in a Vortex expression tree.
///
/// Most nodes are a scalar function applied to child expressions. [`Expression::Root`] is the scope
/// itself: a language primitive rather than a registered function, because its dtype comes from the
/// scope rather than from children and it is not executable. A [`ScalarFnVTable`] can answer neither
/// of those, so `Root` is a variant instead.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Expression {
    /// A scalar function applied to child expressions.
    Scalar {
        /// The scalar fn for this node.
        scalar_fn: ScalarFnRef,
        /// Any children of this expression.
        children: Arc<Vec<Expression>>,
    },
    /// A higher-order function applied to ordinary children and owned lambda syntax.
    HigherOrder {
        higher_order_fn: HigherOrderFunctionRef,
        children: Arc<Vec<Expression>>,
        lambdas: Arc<Vec<Expression>>,
    },
    /// Lambda syntax. A lambda is a binder, not an array-valued expression.
    ///
    /// It can be bound only by a higher-order function that supplies its parameter signature.
    Lambda(Lambda),
    /// The full scope of the expression evaluation.
    Root,
    /// A name to be bound to a value.
    Variable(Variable),
}

impl Expression {
    /// Create a new expression node from a scalar_fn expression and its children.
    pub fn try_new(
        scalar_fn: ScalarFnRef,
        children: impl IntoIterator<Item = Expression>,
    ) -> VortexResult<Self> {
        let children = Vec::from_iter(children);

        vortex_ensure!(
            scalar_fn.signature().arity().matches(children.len()),
            "Expression arity mismatch: expected {} children but got {}",
            scalar_fn.signature().arity(),
            children.len()
        );
        vortex_ensure!(
            children.iter().all(|child| !child.is_lambda()),
            "a scalar function cannot take a lambda as an ordinary argument"
        );

        Ok(Self::Scalar {
            scalar_fn,
            children: children.into(),
        })
    }

    /// Create a higher-order function from ordinary children and owned lambdas.
    pub fn try_new_higher_order(
        higher_order_fn: HigherOrderFunctionRef,
        children: impl IntoIterator<Item = Expression>,
        lambdas: impl IntoIterator<Item = Expression>,
    ) -> VortexResult<Self> {
        let children = Vec::from_iter(children);
        let lambdas = Vec::from_iter(lambdas);
        vortex_ensure!(
            higher_order_fn.arity().matches(children.len()),
            "Higher-order expression arity mismatch: expected {} children but got {}",
            higher_order_fn.arity(),
            children.len()
        );
        vortex_ensure!(
            children.iter().all(|child| !child.is_lambda()),
            "a higher-order function cannot take a lambda as an ordinary argument"
        );
        vortex_ensure!(
            higher_order_fn.lambda_arity() == lambdas.len(),
            "Higher-order expression lambda arity mismatch: expected {} lambdas but got {}",
            higher_order_fn.lambda_arity(),
            lambdas.len()
        );
        vortex_ensure!(
            lambdas.iter().all(Expression::is_lambda),
            "Higher-order expression '{}' requires lambda arguments",
            higher_order_fn.id(),
        );
        Ok(Self::HigherOrder {
            higher_order_fn,
            children: children.into(),
            lambdas: lambdas.into(),
        })
    }

    /// Whether this expression is the scope root.
    pub fn is_root(&self) -> bool {
        matches!(self, Self::Root)
    }

    /// The variable this expression references, if it is a variable.
    pub fn as_variable(&self) -> Option<&Variable> {
        match self {
            Self::Variable(variable) => Some(variable),
            _ => None,
        }
    }

    pub fn as_higher_order(&self) -> Option<&HigherOrderFunctionRef> {
        match self {
            Self::HigherOrder {
                higher_order_fn, ..
            } => Some(higher_order_fn),
            _ => None,
        }
    }

    /// Return this node's lambda syntax if it is a lambda.
    pub fn as_lambda(&self) -> Option<&Lambda> {
        match self {
            Self::Lambda(lambda) => Some(lambda),
            Self::Scalar { .. } | Self::HigherOrder { .. } | Self::Root | Self::Variable(_) => None,
        }
    }

    /// Whether this node is lambda syntax.
    pub fn is_lambda(&self) -> bool {
        self.as_lambda().is_some()
    }

    /// Return the lambda arguments owned by this higher-order expression.
    ///
    /// Each returned node is guaranteed to be [`Expression::Lambda`]. They are deliberately
    /// separate from [`Self::children`], which is the row-domain value tree.
    pub fn lambdas(&self) -> &[Expression] {
        match self {
            Self::HigherOrder { lambdas, .. } => lambdas,
            _ => &[],
        }
    }

    /// Returns the scalar fn for this expression, or `None` if it is not a scalar node.
    pub fn as_scalar(&self) -> Option<&ScalarFnRef> {
        match self {
            Self::Scalar { scalar_fn, .. } => Some(scalar_fn),
            _ => None,
        }
    }

    /// Whether this expression's scalar fn is of the given vtable type.
    pub fn is<V: ScalarFnVTable>(&self) -> bool {
        self.as_scalar().is_some_and(|sf| sf.is::<V>())
    }

    /// The typed options for this expression if its scalar fn matches the given vtable type.
    pub fn as_opt<V: ScalarFnVTable>(&self) -> Option<&V::Options> {
        self.as_scalar().and_then(|sf| sf.as_opt::<V>())
    }

    /// The typed options for this expression.
    ///
    /// # Panics
    ///
    /// Panics if the vtable type does not match.
    pub fn as_<V: ScalarFnVTable>(&self) -> &V::Options {
        self.as_opt::<V>()
            .vortex_expect("Expression options type mismatch")
    }

    /// Returns the sub-expressions of this node.
    pub fn children(&self) -> &[Expression] {
        match self {
            Self::Scalar { children, .. } | Self::HigherOrder { children, .. } => {
                children.as_slice()
            }
            Self::Lambda(_) | Self::Root | Self::Variable(_) => NO_CHILDREN,
        }
    }

    /// Returns the n'th child of this expression.
    pub fn child(&self, n: usize) -> &Expression {
        &self.children()[n]
    }

    /// Replace the children of this expression with the provided new children.
    pub fn with_children(
        self,
        children: impl IntoIterator<Item = Expression>,
    ) -> VortexResult<Self> {
        let children = Vec::from_iter(children);
        match &self {
            Self::Scalar { scalar_fn, .. } => Self::try_new(scalar_fn.clone(), children),
            Self::HigherOrder {
                higher_order_fn,
                lambdas,
                ..
            } => Self::try_new_higher_order(higher_order_fn.clone(), children, lambdas.to_vec()),
            Self::Lambda(_) | Self::Root | Self::Variable(_) => {
                vortex_ensure!(
                    children.is_empty(),
                    "Expression arity mismatch: a leaf expects 0 children but got {}",
                    children.len()
                );
                Ok(self)
            }
        }
    }

    /// Replace the lambda arguments of a higher-order expression.
    pub(crate) fn with_lambdas(
        self,
        lambdas: impl IntoIterator<Item = Expression>,
    ) -> VortexResult<Self> {
        let lambdas = Vec::from_iter(lambdas);
        match &self {
            Self::HigherOrder {
                higher_order_fn,
                children,
                ..
            } => Self::try_new_higher_order(higher_order_fn.clone(), children.to_vec(), lambdas),
            Self::Scalar { .. } | Self::Lambda(_) | Self::Root | Self::Variable(_) => {
                vortex_ensure!(
                    lambdas.is_empty(),
                    "only a higher-order expression can have lambda arguments"
                );
                Ok(self)
            }
        }
    }

    /// Computes the return dtype of this expression given the input dtype.
    pub fn return_dtype(&self, scope: &DType) -> VortexResult<DType> {
        self.return_dtype_scope(&Scope::new(scope.clone()))
    }

    /// Computes the return dtype of this expression against a lexical scope.
    pub fn return_dtype_scope(&self, scope: &Scope) -> VortexResult<DType> {
        Ok(self.bind_scope(scope)?.dtype().clone())
    }

    /// Returns a new expression representing the validity mask output of this expression.
    ///
    /// The returned expression evaluates to a non-nullable boolean array.
    pub fn validity(&self) -> VortexResult<Expression> {
        match self {
            // The scope is exactly as valid as itself.
            Self::Root => Ok(Self::Root),
            // This is evaluated later against the array bound to the variable by a higher-order
            // function, yielding that array's validity as a non-nullable boolean mask.
            Self::Variable(_) => Ok(is_not_null(self.clone())),
            Self::HigherOrder { .. } => Ok(is_not_null(self.clone())),
            Self::Scalar { scalar_fn, .. } => scalar_fn.validity(self),
            Self::Lambda(_) => vortex_error::vortex_bail!(
                "a lambda has no standalone validity expression; it must be applied by a higher-order function"
            ),
        }
    }

    /// Format the expression as a compact string.
    ///
    /// Since this is a recursive formatter, it is exposed on the public Expression type.
    /// See fmt_data that is only implemented on the vtable trait.
    pub fn fmt_sql(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Root => write!(f, "$"),
            Self::Variable(variable) => write!(f, "${variable}"),
            Self::Lambda(lambda) => Display::fmt(lambda, f),
            Self::HigherOrder {
                higher_order_fn, ..
            } => higher_order_fn.fmt_sql(self, f),
            Self::Scalar { scalar_fn, .. } => scalar_fn.fmt_sql(self, f),
        }
    }

    /// Display the expression as a formatted tree structure.
    ///
    /// This provides a hierarchical view of the expression that shows the relationships
    /// between parent and child expressions, making complex nested expressions easier
    /// to understand and debug.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use vortex_array::dtype::{DType, Nullability, PType};
    /// # use vortex_array::scalar_fn::fns::like::{Like, LikeOptions};
    /// # use vortex_array::scalar_fn::ScalarFnVTableExt;
    /// # use vortex_array::expr::{and, cast, eq, get_item, gt, lit, not, root, select};
    /// // Build a complex nested expression
    /// let complex_expr = select(
    ///     ["result"],
    ///     and(
    ///         not(eq(get_item("status", root()), lit("inactive"))),
    ///         and(
    ///             Like.new_expr(LikeOptions::default(), [get_item("name", root()), lit("%admin%")]),
    ///             gt(
    ///                 cast(get_item("score", root()), DType::Primitive(PType::F64, Nullability::NonNullable)),
    ///                 lit(75.0)
    ///             )
    ///         )
    ///     )
    /// );
    ///
    /// println!("{}", complex_expr.display_tree());
    /// ```
    ///
    /// This produces output like:
    ///
    /// ```text
    /// Select(include): {result}
    /// └── Binary(and)
    ///     ├── lhs: Not
    ///     │   └── Binary(=)
    ///     │       ├── lhs: GetItem(status)
    ///     │       │   └── Root
    ///     │       └── rhs: Literal(value: "inactive", dtype: utf8)
    ///     └── rhs: Binary(and)
    ///         ├── lhs: Like
    ///         │   ├── child: GetItem(name)
    ///         │   │   └── Root
    ///         │   └── pattern: Literal(value: "%admin%", dtype: utf8)
    ///         └── rhs: Binary(>)
    ///             ├── lhs: Cast(target: f64)
    ///             │   └── GetItem(score)
    ///             │       └── Root
    ///             └── rhs: Literal(value: 75f64, dtype: f64)
    /// ```
    pub fn display_tree(&self) -> impl Display {
        DisplayTreeExpr(self)
    }

    /// Returns true if this expression contains expression E inside.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use vortex_array::scalar_fn::fns::literal::Literal;
    /// # use vortex_array::expr::{eq, lit, root};
    /// let expression = &eq(root(), lit(3u64));
    /// assert!(expression.contains::<Literal>().unwrap());
    /// let expression = root();
    /// assert!(!expression.contains::<Literal>().unwrap());
    /// ```
    pub fn contains<E: ScalarFnVTable>(&self) -> VortexResult<bool> {
        let mut contains = false;
        pre_order_visit_down(self, |node| {
            if node.is::<E>() {
                contains = true;
                return Ok(TraversalOrder::Stop);
            }
            Ok(TraversalOrder::Continue)
        })?;
        Ok(contains)
    }
}

/// The default display implementation for expressions uses the 'SQL'-style format.
impl Display for Expression {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.fmt_sql(f)
    }
}

fn drain_drop_children(expression: &mut Expression, to_drop: &mut Vec<Expression>) {
    match expression {
        Expression::Scalar { children, .. } => {
            if let Some(children) = Arc::get_mut(children) {
                to_drop.append(children);
            }
        }
        Expression::HigherOrder {
            children, lambdas, ..
        } => {
            if let Some(children) = Arc::get_mut(children) {
                to_drop.append(children);
            }
            if let Some(lambdas) = Arc::get_mut(lambdas) {
                for lambda in lambdas {
                    if let Expression::Lambda(lambda) = lambda
                        && let Some(body) = lambda.take_body()
                    {
                        to_drop.push(body);
                    }
                }
            }
        }
        Expression::Lambda(lambda) => {
            if let Some(body) = lambda.take_body() {
                to_drop.push(body);
            }
        }
        Expression::Root | Expression::Variable(_) => {}
    }
}

/// Iterative drop for expression to avoid stack overflows, including binder-owned lambda bodies.
impl Drop for Expression {
    fn drop(&mut self) {
        let mut to_drop = Vec::new();
        drain_drop_children(self, &mut to_drop);
        while let Some(mut expression) = to_drop.pop() {
            drain_drop_children(&mut expression, &mut to_drop);
        }
    }
}
