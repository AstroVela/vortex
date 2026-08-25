// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Rule-driven optimization for [`BoundExpression`] trees.
//!
//! This optimizer is independent from the legacy [`Expression`](super::Expression) optimizer.
//! Rules operate only on already-bound expressions, which makes every node's dtype available in
//! constant time and lets the driver verify that rewrites preserve the tree's type proof.
//!
//! # How optimization works
//!
//! [`BoundExpressionOptimizer`] holds the reusable rule registry and rewrite limit. Each call to
//! [`BoundExpressionOptimizer::try_optimize`] creates an `OptimizationRun` containing the mutable
//! state for that traversal.
//!
//! A run walks the tree bottom-up without recursion, using one stack of pending tasks and another
//! stack of completed node results:
//!
//! 1. A `Visit` task schedules the node to be finished, then schedules its children in evaluation
//!    order.
//! 2. A `Finish` task collects the optimized children and rebuilds the node only if a child
//!    changed.
//! 3. Rules registered for the node's expression ID run in registration order. The first matching
//!    rule wins.
//! 4. A replacement is visited as a new subtree so that rewrites introduced by other rewrites are
//!    also optimized. `FinishRewrite` then records that the original subtree changed.
//!
//! Every replacement must preserve the node's dtype and differ from the expression it replaces.
//! A per-run rewrite limit terminates rule cycles.

use std::fmt::Debug;
use std::sync::Arc;

use smallvec::SmallVec;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_utils::aliases::hash_map::HashMap;

use crate::expr::BoundExpression;
use crate::expr::ExpressionId;

mod rules;

const DEFAULT_MAX_REWRITES: usize = 10_000;

/// Shared reference to a bound-expression rewrite rule.
pub type BoundExpressionRewriteRuleRef = Arc<dyn BoundExpressionRewriteRule>;

/// An equivalence rewrite for bound expressions with a particular root node implementation.
///
/// The optimizer invokes a rule only when the expression's root ID equals
/// [`Self::expression_id`]. Returning `None` means the rule does not match. A replacement must be
/// semantically equivalent to the input and have exactly the same dtype, including nullability.
/// The optimizer verifies the dtype and rejects unchanged replacements.
pub trait BoundExpressionRewriteRule: Debug + Send + Sync + 'static {
    /// Returns a diagnostic name for this rule.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    /// Returns the expression node ID handled by this rule.
    fn expression_id(&self) -> ExpressionId;

    /// Try to rewrite `expr` to a semantically equivalent bound expression.
    fn rewrite(&self, expr: &BoundExpression) -> VortexResult<Option<BoundExpression>>;
}

/// A deterministic, bottom-up optimizer for [`BoundExpression`] trees.
///
/// Rules are grouped by root expression ID and run in registration order. After the first
/// matching rule rewrites a node, the replacement subtree is optimized from the bottom up before
/// its parent is revisited. A global rewrite budget terminates cyclic rule sets.
#[derive(Clone, Debug)]
pub struct BoundExpressionOptimizer {
    rules: HashMap<ExpressionId, Vec<BoundExpressionRewriteRuleRef>>,
    max_rewrites: usize,
}

impl Default for BoundExpressionOptimizer {
    fn default() -> Self {
        let mut optimizer = Self::empty();
        rules::register(&mut optimizer);
        optimizer
    }
}

impl BoundExpressionOptimizer {
    /// Create an optimizer containing the built-in Vortex expression rewrites.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an optimizer with no registered rules.
    pub fn empty() -> Self {
        Self {
            rules: HashMap::default(),
            max_rewrites: DEFAULT_MAX_REWRITES,
        }
    }

    /// Set the maximum number of successful rewrites allowed in one optimization.
    pub fn with_max_rewrites(mut self, max_rewrites: usize) -> Self {
        self.max_rewrites = max_rewrites;
        self
    }

    /// Register a rewrite after existing rules for the same expression node ID.
    pub fn register_rule<R: BoundExpressionRewriteRule>(&mut self, rule: R) {
        self.register_rule_ref(Arc::new(rule));
    }

    /// Register a shared rewrite after existing rules for the same expression node ID.
    pub fn register_rule_ref(&mut self, rule: BoundExpressionRewriteRuleRef) {
        self.rules
            .entry(rule.expression_id())
            .or_default()
            .push(rule);
    }

    /// Optimize an entire bound expression tree, cloning the input when it remains unchanged.
    pub fn optimize(&self, expr: &BoundExpression) -> VortexResult<BoundExpression> {
        Ok(self.try_optimize(expr)?.unwrap_or_else(|| expr.clone()))
    }

    /// Optimize an entire bound expression tree, returning `None` when no subtree changed.
    pub fn try_optimize(&self, expr: &BoundExpression) -> VortexResult<Option<BoundExpression>> {
        OptimizationRun::new(self, expr).run()
    }

    /// Run rules for the expression's root ID and return the first successful rewrite.
    fn rewrite_root(
        &self,
        expr: &BoundExpression,
    ) -> VortexResult<Option<(&'static str, BoundExpression)>> {
        let Some(rules) = self.rules.get(&expr.id()) else {
            return Ok(None);
        };

        for rule in rules {
            if let Some(replacement) = rule.rewrite(expr)? {
                return Ok(Some((rule.name(), replacement)));
            }
        }
        Ok(None)
    }
}

/// Mutable state for one optimizer invocation.
struct OptimizationRun<'a> {
    optimizer: &'a BoundExpressionOptimizer,
    tasks: TaskStack,
    results: ResultStack,
    rewrite_count: usize,
}

impl<'a> OptimizationRun<'a> {
    /// Create a run with the root expression as its first pending visit.
    fn new(optimizer: &'a BoundExpressionOptimizer, expr: &BoundExpression) -> Self {
        let mut tasks = TaskStack::new();
        tasks.push(Task::Visit(expr.clone()));
        Self {
            optimizer,
            tasks,
            results: ResultStack::new(),
            rewrite_count: 0,
        }
    }

    /// Execute pending tasks until the root produces its single optimization result.
    fn run(mut self) -> VortexResult<Option<BoundExpression>> {
        while let Some(task) = self.tasks.pop() {
            match task {
                Task::Visit(expr) => self.visit(expr)?,
                Task::Finish { expr, child_count } => self.finish(expr, child_count)?,
                Task::FinishRewrite => self.finish_rewrite()?,
            }
        }

        vortex_ensure!(
            self.results.len() == 1,
            "bound-expression optimizer produced {} root results",
            self.results.len()
        );
        let result = self
            .results
            .pop()
            .ok_or_else(|| vortex_err!("bound-expression optimizer produced no root result"))?;
        Ok(result.changed.then_some(result.expr))
    }

    /// Schedule an expression's children followed by the task that finishes the expression.
    ///
    /// A leaf has no child work to schedule, so it is finished immediately.
    fn visit(&mut self, expr: BoundExpression) -> VortexResult<()> {
        let child_count = expr.children().len();
        if child_count == 0 {
            return self.finish_node(expr, OptimizedChildren::Unchanged);
        }

        self.tasks.reserve(child_count + 1);
        self.results.reserve(child_count);
        self.tasks.push(Task::Finish {
            expr: expr.clone(),
            child_count,
        });
        self.tasks
            .extend(expr.children().iter().rev().cloned().map(Task::Visit));
        Ok(())
    }

    /// Collect an expression's child results and finish the expression itself.
    ///
    /// When every child is unchanged, their results are discarded and the original expression can
    /// be reused. Otherwise, all optimized children are collected to rebuild the expression.
    fn finish(&mut self, expr: BoundExpression, child_count: usize) -> VortexResult<()> {
        vortex_ensure!(
            self.results.len() >= child_count,
            "bound-expression optimizer lost a child result"
        );
        let first_child = self.results.len() - child_count;
        let children = if self.results[first_child..]
            .iter()
            .any(|result| result.changed)
        {
            let mut children = Vec::with_capacity(child_count);
            children.extend(self.results.drain(first_child..).map(|result| result.expr));
            OptimizedChildren::Changed(children)
        } else {
            self.results.truncate(first_child);
            OptimizedChildren::Unchanged
        };
        self.finish_node(expr, children)
    }

    /// Mark an optimized replacement subtree as changed from the expression it replaced.
    ///
    /// Optimizing the replacement may itself report no changes, but applying the rule that produced
    /// it must remain visible to the original expression's parent.
    fn finish_rewrite(&mut self) -> VortexResult<()> {
        let Some(mut result) = self.results.pop() else {
            vortex_bail!("bound-expression optimizer lost a rewrite result")
        };
        result.changed = true;
        self.results.push(result);
        Ok(())
    }

    /// Rebuild a node when necessary, then apply the first matching root rewrite.
    ///
    /// A node with no matching rule is pushed onto the result stack. A successful replacement is
    /// validated and scheduled for its own bottom-up optimization.
    fn finish_node(
        &mut self,
        original: BoundExpression,
        children: OptimizedChildren,
    ) -> VortexResult<()> {
        let (current, children_changed) = match children {
            OptimizedChildren::Unchanged => (original, false),
            OptimizedChildren::Changed(children) => {
                let original_dtype = original.dtype().clone();
                let rebuilt = original.with_children(children)?;
                vortex_ensure!(
                    rebuilt.dtype() == &original_dtype,
                    "optimizing children changed a node dtype from {original_dtype} to {}",
                    rebuilt.dtype()
                );
                (rebuilt, true)
            }
        };

        let Some((rule_name, replacement)) = self.optimizer.rewrite_root(&current)? else {
            self.results.push(OptimizationResult {
                expr: current,
                changed: children_changed,
            });
            return Ok(());
        };

        vortex_ensure!(
            replacement.dtype() == current.dtype(),
            "bound-expression rewrite rule {rule_name} changed dtype from {} to {}",
            current.dtype(),
            replacement.dtype()
        );
        vortex_ensure!(
            replacement != current,
            "bound-expression rewrite rule {rule_name} returned an unchanged expression"
        );
        if self.rewrite_count >= self.optimizer.max_rewrites {
            vortex_bail!(
                "Exceeded bound-expression rewrite limit of {} while applying {rule_name} \
                 (possible rewrite cycle)",
                self.optimizer.max_rewrites
            );
        }
        self.rewrite_count += 1;

        self.tasks.reserve(2);
        self.tasks.push(Task::FinishRewrite);
        self.tasks.push(Task::Visit(replacement));
        Ok(())
    }
}

enum Task {
    /// Visit an expression by scheduling its children followed by [`Task::Finish`].
    ///
    /// Childless expressions are finished immediately without scheduling another task.
    Visit(BoundExpression),
    /// Finish an original expression after all of its children have been optimized.
    ///
    /// The `child_count` most recent results belong to this expression. They are used to rebuild
    /// the node when any child changed, after which the optimizer tries to rewrite the node itself.
    Finish {
        expr: BoundExpression,
        child_count: usize,
    },
    /// Finish a successful root rewrite after its replacement subtree has been optimized.
    ///
    /// Unlike [`Task::Finish`], this task has no children to collect or node to rebuild: the
    /// replacement's result is already on the result stack. It marks that result as changed so the
    /// rewrite remains visible even when optimizing the replacement made no further changes.
    FinishRewrite,
}

/// Optimized children, distinguishing reusable originals from a changed child list.
enum OptimizedChildren {
    Unchanged,
    Changed(Vec<BoundExpression>),
}

/// A completed subtree and whether any rewrite changed it.
struct OptimizationResult {
    expr: BoundExpression,
    changed: bool,
}

type TaskStack = SmallVec<[Task; 16]>;
type ResultStack = SmallVec<[OptimizationResult; 16]>;

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use super::BoundExpressionOptimizer;
    use super::BoundExpressionRewriteRule;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::BoundExpression;
    use crate::expr::ExpressionId;
    use crate::expr::bound;
    use crate::scalar::Scalar;
    use crate::scalar_fn::ScalarFnVTable;
    use crate::scalar_fn::fns::binary::Binary;
    use crate::scalar_fn::fns::is_null::IsNull;
    use crate::scalar_fn::fns::literal::Literal;

    #[derive(Debug)]
    struct IsNullTo(bool);

    impl BoundExpressionRewriteRule for IsNullTo {
        fn expression_id(&self) -> ExpressionId {
            IsNull.id()
        }

        fn rewrite(&self, _expr: &BoundExpression) -> VortexResult<Option<BoundExpression>> {
            Ok(Some(bound::lit(self.0)))
        }
    }

    #[test]
    fn rules_run_in_registration_order() -> VortexResult<()> {
        let input = bound::is_null(bound::lit(1i32));
        let mut optimizer = BoundExpressionOptimizer::empty();
        optimizer.register_rule(IsNullTo(true));
        optimizer.register_rule(IsNullTo(false));

        assert_eq!(optimizer.optimize(&input)?, bound::lit(true));
        Ok(())
    }

    #[derive(Debug)]
    struct IsNullToReducibleTree;

    impl BoundExpressionRewriteRule for IsNullToReducibleTree {
        fn expression_id(&self) -> ExpressionId {
            IsNull.id()
        }

        fn rewrite(&self, _expr: &BoundExpression) -> VortexResult<Option<BoundExpression>> {
            Ok(Some(bound::and(bound::lit(false), bound::lit(true))))
        }
    }

    #[derive(Debug)]
    struct AndFalse;

    impl BoundExpressionRewriteRule for AndFalse {
        fn expression_id(&self) -> ExpressionId {
            Binary.id()
        }

        fn rewrite(&self, expr: &BoundExpression) -> VortexResult<Option<BoundExpression>> {
            Ok(
                (expr.child(0).as_opt::<Literal>() == Some(&Scalar::from(false)))
                    .then(|| bound::lit(false)),
            )
        }
    }

    #[test]
    fn optimizes_subtrees_introduced_by_rules() -> VortexResult<()> {
        let input = bound::is_null(bound::lit(1i32));
        let mut optimizer = BoundExpressionOptimizer::empty();
        optimizer.register_rule(IsNullToReducibleTree);
        optimizer.register_rule(AndFalse);

        assert_eq!(optimizer.optimize(&input)?, bound::lit(false));
        Ok(())
    }

    #[derive(Debug)]
    struct WrongDType;

    impl BoundExpressionRewriteRule for WrongDType {
        fn expression_id(&self) -> ExpressionId {
            IsNull.id()
        }

        fn rewrite(&self, _expr: &BoundExpression) -> VortexResult<Option<BoundExpression>> {
            Ok(Some(bound::lit(1i32)))
        }
    }

    #[test]
    fn rejects_dtype_changes() {
        let input = bound::is_null(bound::lit(1i32));
        let mut optimizer = BoundExpressionOptimizer::empty();
        optimizer.register_rule(WrongDType);

        assert!(optimizer.optimize(&input).is_err());
    }

    #[derive(Debug)]
    struct Unchanged;

    impl BoundExpressionRewriteRule for Unchanged {
        fn expression_id(&self) -> ExpressionId {
            IsNull.id()
        }

        fn rewrite(&self, expr: &BoundExpression) -> VortexResult<Option<BoundExpression>> {
            Ok(Some(expr.clone()))
        }
    }

    #[test]
    fn rejects_unchanged_replacements() {
        let input = bound::is_null(bound::lit(1i32));
        let mut optimizer = BoundExpressionOptimizer::empty();
        optimizer.register_rule(Unchanged);

        assert!(optimizer.optimize(&input).is_err());
    }

    #[derive(Debug)]
    struct ToggleBoolean;

    impl BoundExpressionRewriteRule for ToggleBoolean {
        fn expression_id(&self) -> ExpressionId {
            Literal.id()
        }

        fn rewrite(&self, expr: &BoundExpression) -> VortexResult<Option<BoundExpression>> {
            let Some(value) = expr
                .as_opt::<Literal>()
                .and_then(|scalar| scalar.as_bool_opt())
            else {
                return Ok(None);
            };
            Ok(value.value().map(|value| bound::lit(!value)))
        }
    }

    #[test]
    fn rewrite_budget_terminates_cycles() {
        let mut optimizer = BoundExpressionOptimizer::empty().with_max_rewrites(4);
        optimizer.register_rule(ToggleBoolean);

        assert!(optimizer.optimize(&bound::lit(true)).is_err());
    }

    #[derive(Debug)]
    struct RootToOne(ExpressionId);

    impl BoundExpressionRewriteRule for RootToOne {
        fn expression_id(&self) -> ExpressionId {
            self.0
        }

        fn rewrite(&self, _expr: &BoundExpression) -> VortexResult<Option<BoundExpression>> {
            Ok(Some(bound::lit(1i32)))
        }
    }

    #[test]
    fn rules_can_target_non_scalar_expression_nodes() -> VortexResult<()> {
        let input = bound::root(DType::Primitive(PType::I32, Nullability::NonNullable));
        let mut optimizer = BoundExpressionOptimizer::empty();
        optimizer.register_rule(RootToOne(input.id()));

        assert_eq!(optimizer.optimize(&input)?, bound::lit(1i32));
        Ok(())
    }

    #[test]
    fn unchanged_deep_tree_does_not_recurse() -> VortexResult<()> {
        let mut expr = bound::root(DType::Bool(Nullability::NonNullable));
        for _ in 0..20_000 {
            expr = bound::not(expr);
        }

        assert!(
            BoundExpressionOptimizer::empty()
                .try_optimize(&expr)?
                .is_none()
        );
        Ok(())
    }

    #[test]
    fn default_optimizer_folds_literal_cast() -> VortexResult<()> {
        let target = DType::Primitive(PType::I64, Nullability::NonNullable);
        let expr = bound::cast(bound::lit(1i32), target);

        assert_eq!(
            BoundExpressionOptimizer::default().optimize(&expr)?,
            bound::lit(1i64)
        );
        Ok(())
    }
}
