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
use vortex_session::VortexSession;

use crate::dtype::DType;
use crate::expr::display::DisplayTreeExpr;
use crate::expr::scope::Scope;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnOptions;
use crate::scalar_fn::ScalarFnRef;
use crate::scalar_fn::ScalarFnSignature;
use crate::scalar_fn::ScalarFnVTable;

/// An empty child slice, returned by [`Expression::children`] for childless variants.
const NO_CHILDREN: &[Expression] = &[];

/// A node in a Vortex expression tree.
///
/// Expressions represent scalar computations performed over a scope. Most nodes are a scalar
/// function applied to child expressions; [`Expression::Root`] is the scope itself.
///
/// `Root` is a distinct variant rather than a scalar function because its dtype comes from the
/// scope rather than from its children, and it is not executable — a scalar function vtable cannot
/// answer either question honestly.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Expression {
    /// A scalar function applied to child expressions.
    Scalar {
        /// The scalar fn for this node.
        scalar_fn: ScalarFnRef,
        /// Any children of this expression.
        children: Arc<Vec<Expression>>,
    },
    /// The full scope of the expression evaluation.
    Root,
}

impl Expression {
    /// Create a new expression node from a scalar fn and its children.
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

        Ok(Self::Scalar {
            scalar_fn,
            children: children.into(),
        })
    }

    /// Whether this expression is the scope root.
    pub fn is_root(&self) -> bool {
        matches!(self, Self::Root)
    }

    /// Returns the scalar fn for this expression, or `None` if it is not a scalar node.
    pub fn as_scalar(&self) -> Option<&ScalarFnRef> {
        match self {
            Self::Scalar { scalar_fn, .. } => Some(scalar_fn),
            Self::Root => None,
        }
    }

    /// Returns the id of this expression's scalar fn, or `None` for [`Expression::Root`].
    pub fn scalar_fn_id(&self) -> Option<ScalarFnId> {
        self.as_scalar().map(|sf| sf.id())
    }

    /// Signature information for this expression's scalar fn, or `None` for
    /// [`Expression::Root`].
    pub fn signature(&self) -> Option<ScalarFnSignature<'_>> {
        self.as_scalar().map(|sf| sf.signature())
    }

    /// The type-erased options for this expression's scalar fn, or `None` for
    /// [`Expression::Root`].
    pub fn options(&self) -> Option<ScalarFnOptions<'_>> {
        self.as_scalar().map(|sf| sf.options())
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

    /// Returns the children of this expression.
    pub fn children(&self) -> &[Expression] {
        match self {
            Self::Scalar { children, .. } => children.as_slice(),
            Self::Root => NO_CHILDREN,
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
            Self::Root => {
                vortex_ensure!(
                    children.is_empty(),
                    "Expression arity mismatch: root expects 0 children but got {}",
                    children.len()
                );
                Ok(Self::Root)
            }
            Self::Scalar { scalar_fn, .. } => {
                vortex_ensure!(
                    scalar_fn.signature().arity().matches(children.len()),
                    "Expression arity mismatch: expected {} children but got {}",
                    scalar_fn.signature().arity(),
                    children.len()
                );
                Ok(Self::Scalar {
                    scalar_fn: scalar_fn.clone(),
                    children: children.into(),
                })
            }
        }
    }

    /// Computes the return dtype of this expression given the input dtype.
    ///
    /// This binds the expression and discards everything but the root dtype. Callers needing types
    /// at more than one node should use [`Expression::bind`] once and read the dtypes off the bound
    /// tree instead of calling this repeatedly.
    pub fn return_dtype(&self, scope: &DType) -> VortexResult<DType> {
        Ok(self.bind(&Scope::new(scope.clone()))?.dtype().clone())
    }

    /// Returns a new expression representing the validity mask output of this expression.
    ///
    /// The returned expression evaluates to a non-nullable boolean array.
    pub fn validity(&self) -> VortexResult<Expression> {
        match self {
            // The scope is as valid as itself.
            Self::Root => Ok(Self::Root),
            Self::Scalar { scalar_fn, .. } => scalar_fn.validity(self),
        }
    }

    /// Returns an expression that proves this predicate is definitely false from stats.
    ///
    /// `scope` is the dtype of the row this expression evaluates over.
    ///
    /// If the returned expression evaluates to `true` for a stats scope, this expression is
    /// guaranteed to be false for every row in that scope. `false` and `null` are unknown.
    pub fn falsify(
        &self,
        scope: &DType,
        session: &VortexSession,
    ) -> VortexResult<Option<Expression>> {
        crate::stats::rewrite::StatsRewriteCtx::new(session, scope).falsify(self)
    }

    /// Returns an expression that proves this predicate is definitely true from stats.
    ///
    /// `scope` is the dtype of the row this expression evaluates over.
    ///
    /// If the returned expression evaluates to `true` for a stats scope, this expression is
    /// guaranteed to be true for every row in that scope. `false` and `null` are unknown.
    pub fn satisfy(
        &self,
        scope: &DType,
        session: &VortexSession,
    ) -> VortexResult<Option<Expression>> {
        crate::stats::rewrite::StatsRewriteCtx::new(session, scope).satisfy(self)
    }

    /// Format the expression as a compact string.
    ///
    /// Since this is a recursive formatter, it is exposed on the public Expression type.
    /// See fmt_data that is only implemented on the vtable trait.
    pub fn fmt_sql(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Root => write!(f, "$"),
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
    pub fn display_tree(&self) -> impl Display {
        DisplayTreeExpr(self)
    }
}

/// The default display implementation for expressions uses the 'SQL'-style format.
impl Display for Expression {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.fmt_sql(f)
    }
}

/// Iterative drop for expression to avoid stack overflows.
impl Drop for Expression {
    fn drop(&mut self) {
        let Self::Scalar { children, .. } = self else {
            return;
        };
        let Some(children) = Arc::get_mut(children) else {
            return;
        };

        let mut children_to_drop = std::mem::take(children);
        while let Some(mut child) = children_to_drop.pop() {
            if let Self::Scalar { children, .. } = &mut child
                && let Some(expr_children) = Arc::get_mut(children)
            {
                children_to_drop.append(expr_children);
            }
        }
    }
}
