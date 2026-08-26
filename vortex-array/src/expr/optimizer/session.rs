// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session state for bound-expression rewrite rules.

use std::any::Any;
use std::sync::Arc;

use vortex_session::ArcSwapMap;
use vortex_session::SessionExt;
use vortex_session::SessionGuard;
use vortex_session::SessionVar;

use super::rules::BinaryBoolean;
use super::rules::BinaryNullComparison;
use super::rules::BoundExpressionRewriteRule;
use super::rules::BoundExpressionRewriteRuleRef;
use super::rules::CaseWhenToFillNull;
use super::rules::CastLiteralOrIdentity;
use super::rules::ConstantMask;
use super::rules::ConstantZip;
use super::rules::FindBetween;
use super::rules::GetItemFromPack;
use super::rules::MergeToPack;
use super::rules::RemoveRedundantFillNull;
use super::rules::SelectFromPack;
use crate::expr::ExpressionId;

/// Ordered rewrite rules registered for one expression node ID.
pub type ExpressionOptimizerRuleSet = Arc<[BoundExpressionRewriteRuleRef]>;

/// Session-scoped registry of bound-expression rewrite rules.
pub type ExpressionOptimizerRuleRegistry = ArcSwapMap<ExpressionId, ExpressionOptimizerRuleSet>;

/// Session state for bound-expression optimization.
#[derive(Clone, Debug)]
pub struct ExpressionOptimizerSession {
    registry: ExpressionOptimizerRuleRegistry,
}

impl ExpressionOptimizerSession {
    /// Create a session with no rewrite rules.
    pub fn empty() -> Self {
        Self {
            registry: ExpressionOptimizerRuleRegistry::default(),
        }
    }

    /// Return the bound-expression rewrite rule registry.
    pub fn registry(&self) -> &ExpressionOptimizerRuleRegistry {
        &self.registry
    }

    /// Register a rewrite after existing rules for the same expression node ID.
    pub fn register<R: BoundExpressionRewriteRule>(&self, rule: R) {
        self.registry.push(
            rule.expression_id(),
            Arc::new(rule) as BoundExpressionRewriteRuleRef,
        );
    }
}

impl Default for ExpressionOptimizerSession {
    fn default() -> Self {
        let session = Self::empty();

        session.register(BinaryBoolean);
        session.register(BinaryNullComparison);
        session.register(FindBetween);
        session.register(CastLiteralOrIdentity);
        session.register(GetItemFromPack);
        session.register(MergeToPack);
        session.register(SelectFromPack);
        session.register(RemoveRedundantFillNull);
        session.register(CaseWhenToFillNull);
        session.register(ConstantMask);
        session.register(ConstantZip);

        session
    }
}

impl SessionVar for ExpressionOptimizerSession {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Extension trait for accessing bound-expression optimizer session state.
pub trait ExpressionOptimizerSessionExt: SessionExt {
    /// Return the bound-expression optimizer session.
    fn expression_optimizer(&self) -> SessionGuard<'_, ExpressionOptimizerSession> {
        self.get::<ExpressionOptimizerSession>()
    }
}

impl<S: SessionExt> ExpressionOptimizerSessionExt for S {}
