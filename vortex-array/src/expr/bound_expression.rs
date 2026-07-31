// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use itertools::Itertools;
use vortex_error::VortexResult;

use crate::dtype::DType;
use crate::expr::Expression;
use crate::expr::scope::Scope;
use crate::scalar_fn::ScalarFnRef;

/// An [`Expression`] that has been type-checked against a [`Scope`].
///
/// Every node carries its own dtype, so reading one is a field access rather than a walk of the
/// subtree. Holding a `BoundExpression` is proof that the whole tree type-checked.
///
/// Binding is purely logical: it deals only in [`DType`]s and never sees an array, a length or an
/// encoding.
#[derive(Clone, Debug)]
pub struct BoundExpression {
    kind: BoundKind,
    dtype: DType,
}

/// The per-variant contents of a [`BoundExpression`], mirroring [`Expression`].
#[derive(Clone, Debug)]
pub enum BoundKind {
    /// A scalar function applied to bound children.
    Scalar {
        /// The scalar function for this node.
        scalar_fn: ScalarFnRef,
        /// The bound children, in argument order.
        ///
        /// Shared behind an `Arc` for the same reason as [`Expression::Scalar`]: it keeps clones
        /// cheap, which matters because the iterative `Drop` impl below makes a
        /// `BoundExpression` non-destructurable by value, so consumers that rebuild a tree must
        /// clone rather than move.
        children: Arc<Vec<BoundExpression>>,
    },
    /// The scope itself. Its dtype is the scope's root dtype.
    Root,
}

impl BoundExpression {
    /// The dtype this expression evaluates to.
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    /// The per-variant contents of this node.
    pub fn kind(&self) -> &BoundKind {
        &self.kind
    }

    /// The bound children of this node, in argument order. Empty for [`BoundKind::Root`].
    pub fn children(&self) -> &[BoundExpression] {
        match &self.kind {
            BoundKind::Scalar { children, .. } => children.as_slice(),
            BoundKind::Root => &[],
        }
    }

    /// The scalar function for this node, or `None` if it is not a scalar node.
    pub fn as_scalar(&self) -> Option<&ScalarFnRef> {
        match &self.kind {
            BoundKind::Scalar { scalar_fn, .. } => Some(scalar_fn),
            BoundKind::Root => None,
        }
    }

    /// Whether this node is the scope root.
    pub fn is_root(&self) -> bool {
        matches!(self.kind, BoundKind::Root)
    }
}

impl Expression {
    /// Bind this expression against `scope`, type-checking every node in a single walk.
    ///
    /// The returned tree carries a dtype on each node, so callers needing types at more than one
    /// node should bind once and read fields rather than calling
    /// [`return_dtype`](Expression::return_dtype) repeatedly.
    pub fn bind(&self, scope: &Scope) -> VortexResult<BoundExpression> {
        match self {
            Expression::Root => Ok(BoundExpression {
                kind: BoundKind::Root,
                dtype: scope.root().clone(),
            }),
            Expression::Scalar {
                scalar_fn,
                children,
            } => {
                let children: Vec<_> = children
                    .iter()
                    .map(|child| child.bind(scope))
                    .try_collect()?;
                let arg_dtypes: Vec<_> = children.iter().map(|c| c.dtype().clone()).collect();
                let dtype = scalar_fn.return_dtype(&arg_dtypes)?;
                Ok(BoundExpression {
                    kind: BoundKind::Scalar {
                        scalar_fn: scalar_fn.clone(),
                        children: children.into(),
                    },
                    dtype,
                })
            }
        }
    }
}

/// Iterative drop to avoid stack overflows on deep trees.
impl Drop for BoundExpression {
    fn drop(&mut self) {
        let BoundKind::Scalar { children, .. } = &mut self.kind else {
            return;
        };
        let Some(children) = Arc::get_mut(children) else {
            return;
        };

        let mut to_drop = std::mem::take(children);
        while let Some(mut child) = to_drop.pop() {
            if let BoundKind::Scalar { children, .. } = &mut child.kind
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
    use crate::expr::col;
    use crate::expr::eq;
    use crate::expr::lit;
    use crate::expr::root;
    use crate::expr::test_harness::struct_dtype;

    fn scope() -> Scope {
        Scope::new(struct_dtype())
    }

    #[test]
    fn root_binds_to_the_scope() -> VortexResult<()> {
        let bound = root().bind(&scope())?;
        assert!(bound.is_root());
        assert_eq!(bound.dtype(), &struct_dtype());
        Ok(())
    }

    #[test]
    fn every_node_carries_its_dtype() -> VortexResult<()> {
        let expr = eq(col("a"), lit(1i32));
        let bound = expr.bind(&scope())?;

        assert_eq!(bound.dtype(), &DType::Bool(Nullability::NonNullable));

        // get_item("a", root()) -> i32, and its child is the struct scope.
        let lhs = &bound.children()[0];
        assert_eq!(
            lhs.dtype(),
            &DType::Primitive(PType::I32, Nullability::NonNullable)
        );
        assert_eq!(lhs.children()[0].dtype(), &struct_dtype());
        Ok(())
    }

    /// `return_dtype` is a shim over `bind`, so the two must never disagree.
    #[test]
    fn bind_agrees_with_return_dtype() -> VortexResult<()> {
        for expr in [root(), col("a"), eq(col("a"), lit(1i32)), lit(true)] {
            assert_eq!(
                expr.bind(&scope())?.dtype(),
                &expr.return_dtype(&struct_dtype())?,
                "disagreement for {expr}"
            );
        }
        Ok(())
    }

    /// Cloning a bound tree must not deep-copy it: the children `Arc` is shared, which is what
    /// keeps rebuilding a tree affordable despite the `Drop` impl blocking moves.
    #[test]
    fn clone_shares_children() -> VortexResult<()> {
        let bound = eq(col("a"), lit(1i32)).bind(&scope())?;
        let cloned = bound.clone();

        let (BoundKind::Scalar { children: a, .. }, BoundKind::Scalar { children: b, .. }) =
            (bound.kind(), cloned.kind())
        else {
            unreachable!("eq is a scalar node")
        };
        assert!(Arc::ptr_eq(a, b));
        Ok(())
    }

    /// A subtree reused at two positions is bound once per occurrence, so each gets its own node.
    #[test]
    fn repeated_subtree_is_bound_per_occurrence() -> VortexResult<()> {
        let shared = col("a");
        let bound = eq(shared.clone(), shared).bind(&scope())?;
        let children = bound.children();
        assert_eq!(children[0].dtype(), children[1].dtype());
        Ok(())
    }

    #[test]
    fn binding_reports_a_type_error() {
        // `a` is i32, so comparing it against a string cannot type-check.
        let expr = eq(col("a"), lit("nope"));
        assert!(expr.bind(&scope()).is_err());
    }
}

#[cfg(test)]
mod size_probe {
    use super::*;

    #[test]
    fn print_sizes() {
        for (name, size) in [
            ("DType", std::mem::size_of::<DType>()),
            ("Arc<DType>", std::mem::size_of::<std::sync::Arc<DType>>()),
            ("ScalarFnRef", std::mem::size_of::<ScalarFnRef>()),
            ("Expression", std::mem::size_of::<Expression>()),
            ("BoundKind", std::mem::size_of::<BoundKind>()),
            ("BoundExpression", std::mem::size_of::<BoundExpression>()),
        ] {
            println!("{name:20} {size:>4} bytes");
        }
    }
}
