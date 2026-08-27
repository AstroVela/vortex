// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session state for stats APIs.

use std::any::Any;
use std::sync::Arc;

use parking_lot::RwLock;
use vortex_session::SessionExt;
use vortex_session::SessionGuard;
use vortex_session::SessionVar;
use vortex_utils::aliases::hash_map::HashMap;

use crate::expr::BoundExpression;
use crate::expr::ExactBoundExpr;
use crate::scalar_fn::ScalarFnId;
use crate::stats::rewrite::StatsRewriteRule;
use crate::stats::rewrite::StatsRewriteRuleRef;
use crate::stats::rewrite::register_builtins;

type StatsRewriteRuleSet = Arc<[StatsRewriteRuleRef]>;

/// Upper bound on cached rewrites per kind before the cache is reset.
///
/// Keys are identity-based, so a long-lived session that binds many distinct predicates would
/// otherwise grow without bound. A full reset is simpler than LRU bookkeeping and the cache is
/// only an accelerator: a miss recomputes the rewrite.
const REWRITE_CACHE_CAPACITY: usize = 4096;

/// The kind of stats proof a rewrite produces.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum StatsRewriteKind {
    /// A proof that the predicate is false for every row in scope.
    Falsify,
    /// A proof that the predicate is true for every row in scope.
    Satisfy,
}

/// Session-scoped cache of lowered stats proofs.
///
/// Keys compare shared-tree identity rather than structure. Every entry holds a clone of the
/// keyed expression, which keeps its allocation alive so the pointer cannot be reused by an
/// unrelated expression while the entry exists.
type StatsRewriteCache = HashMap<(StatsRewriteKind, ExactBoundExpr), Option<BoundExpression>>;

/// Session state for stats APIs.
#[derive(Clone, Debug)]
pub struct StatsSession {
    rewrite_rules: Arc<RwLock<HashMap<ScalarFnId, StatsRewriteRuleSet>>>,
    rewrite_cache: Arc<RwLock<StatsRewriteCache>>,
}

impl Default for StatsSession {
    fn default() -> Self {
        let this = Self {
            rewrite_rules: Arc::new(RwLock::new(HashMap::default())),
            rewrite_cache: Arc::new(RwLock::new(HashMap::default())),
        };
        register_builtins(&this);
        this
    }
}

impl StatsSession {
    /// Register a stats rewrite rule.
    pub fn register_rewrite<R: StatsRewriteRule>(&self, rule: R) {
        self.register_rewrite_ref(Arc::new(rule));
    }

    /// Register a shared stats rewrite rule.
    ///
    /// Registering a rule invalidates every cached rewrite, since a new rule can strengthen the
    /// proof produced for expressions that were already lowered.
    pub fn register_rewrite_ref(&self, rule: StatsRewriteRuleRef) {
        let mut rules = self.rewrite_rules.write();
        let rule_id = rule.scalar_fn_id();
        let mut updated_rules = rules
            .get(&rule_id)
            .map(|rules| rules.iter().cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        updated_rules.push(rule);
        rules.insert(rule_id, updated_rules.into());

        self.rewrite_cache.write().clear();
    }

    /// Return the cached rewrite of `expr`, if one exists.
    ///
    /// `Some(None)` records that no rule could prove anything for `expr`, which is just as
    /// valuable to remember as a successful proof.
    pub(crate) fn cached_rewrite(
        &self,
        kind: StatsRewriteKind,
        expr: &BoundExpression,
    ) -> Option<Option<BoundExpression>> {
        self.rewrite_cache
            .read()
            .get(&(kind, ExactBoundExpr(expr.clone())))
            .cloned()
    }

    /// Cache the rewrite of `expr` and return the entry that is now shared by the session.
    ///
    /// If another thread cached the same expression first, its result wins so that every caller
    /// observes one shared proof tree.
    pub(crate) fn cache_rewrite(
        &self,
        kind: StatsRewriteKind,
        expr: &BoundExpression,
        rewrite: Option<BoundExpression>,
    ) -> Option<BoundExpression> {
        let mut cache = self.rewrite_cache.write();

        if cache.len() >= REWRITE_CACHE_CAPACITY {
            cache.clear();
        }

        cache
            .entry((kind, ExactBoundExpr(expr.clone())))
            .or_insert(rewrite)
            .clone()
    }

    /// Return the rewrite rules registered for `scalar_fn_id`.
    pub(crate) fn rewrite_rules_for(
        &self,
        scalar_fn_id: ScalarFnId,
    ) -> Option<StatsRewriteRuleSet> {
        self.rewrite_rules.read().get(&scalar_fn_id).cloned()
    }
}

impl SessionVar for StatsSession {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Extension trait for accessing stats session data.
pub trait StatsSessionExt: SessionExt {
    /// Returns the stats session state.
    fn stats(&self) -> SessionGuard<'_, StatsSession> {
        self.get::<StatsSession>()
    }
}
impl<S: SessionExt> StatsSessionExt for S {}
