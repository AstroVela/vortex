// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Heap-allocated physical plans for layout scans.

mod children;
mod display;
pub mod optimizer;
mod plans;

use std::any::Any;
use std::borrow::Cow;
use std::sync::Arc;

pub(crate) use children::LazyPlanChildren;
pub use display::PlanExpressionExtractor;
pub use display::PlanIndentedFormatter;
pub use display::PlanSummaryExtractor;
pub use display::PlanTreeContext;
pub use display::PlanTreeDisplay;
pub use display::PlanTreeExtractor;
pub use plans::ChunkedPlan;
pub use plans::DictPlan;
pub use plans::ExpressionPlan;
pub use plans::FlatPlan;
pub use plans::ListPlan;
pub use plans::RowIdxPlan;
pub use plans::StructPlan;
use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::LayoutRef;
use crate::layouts::chunked::Chunked;
use crate::layouts::dict::Dict;
use crate::layouts::flat::Flat;
use crate::layouts::list::List;
use crate::layouts::struct_::Struct;

/// Shared handle to a heap-allocated physical plan.
pub type PlanRef = Arc<dyn Plan>;

/// A heap-allocated physical plan over a row domain.
///
/// Layout plans expose their optimizer-facing children in a stable logical order. Optional child
/// slots count toward [`child_count`](Self::child_count) and are returned as `None` by
/// [`child`](Self::child) when absent. Accessing a child may initialize and cache its plan.
/// Parent-child rewrites are expressed as [`optimizer::PlanParentReduceRule`]s and collected in a
/// static [`optimizer::PlanParentRuleSet`].
pub trait Plan: Any + Send + Sync {
    /// Returns the display name of this plan kind.
    ///
    /// Plans should override this when the fully qualified Rust type name is not appropriate.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    /// Optimizes this plan while preserving its dtype and row domain.
    ///
    /// Implementations may defer child optimization until the child is accessed.
    fn optimize(&self) -> VortexResult<PlanRef>;

    /// Attempts to rewrite `expression` through this plan.
    ///
    /// Returns `None` when this plan has no applicable expression rewrite. Implementations may
    /// request only the children needed by the rewrite and should preserve all other lazy child
    /// slots.
    fn optimize_expression(&self, expression: &Expression) -> VortexResult<Option<PlanRef>> {
        let _ = expression;
        Ok(None)
    }

    /// Returns the dtype produced by this plan.
    fn dtype(&self) -> &DType;

    /// Returns the number of rows in this plan's row domain.
    fn row_count(&self) -> u64;

    /// Returns the number of ordered logical child slots.
    fn child_count(&self) -> usize {
        0
    }

    /// Returns the plan in logical child slot `index`, or `None` when that optional slot is absent.
    fn child(&self, index: usize) -> VortexResult<Option<PlanRef>> {
        vortex_bail!("Plan has no child {index}")
    }

    /// Returns the display name of logical child slot `index`.
    fn child_name(&self, index: usize) -> Cow<'_, str> {
        Cow::Owned(format!("child[{index}]"))
    }
}

/// Constructs a physical plan without changing the layout or scan APIs.
///
/// Known layouts are represented by optimizer-visible plan nodes, which may defer constructing
/// their children. Unsupported layout kinds return an error when their plan is requested.
pub fn new_plan(layout: &LayoutRef) -> VortexResult<PlanRef> {
    if let Some(layout) = layout.as_opt::<Chunked>() {
        return Ok(Arc::new(ChunkedPlan::new(layout)));
    }
    if let Some(layout) = layout.as_opt::<Dict>() {
        return Ok(Arc::new(DictPlan::try_new(layout)?));
    }
    if let Some(layout) = layout.as_opt::<Flat>() {
        return Ok(Arc::new(FlatPlan::new(layout)));
    }
    if let Some(layout) = layout.as_opt::<List>() {
        return Ok(Arc::new(ListPlan::try_new(layout)?));
    }
    if let Some(layout) = layout.as_opt::<Struct>() {
        return Ok(Arc::new(StructPlan::new(layout)));
    }
    vortex_bail!(
        "No physical plan implementation for layout '{}'",
        layout.encoding_id()
    )
}

impl dyn Plan + '_ {
    /// Returns whether this plan has concrete type `T`.
    pub fn is<T: Plan>(&self) -> bool {
        (self as &dyn Any).is::<T>()
    }

    /// Downcasts this plan to concrete type `T`.
    pub fn downcast_ref<T: Plan>(&self) -> Option<&T> {
        (self as &dyn Any).downcast_ref::<T>()
    }

    /// Displays this plan and its descendants with the default plan extractors.
    pub fn display_tree(&self) -> PlanTreeDisplay<'_> {
        PlanTreeDisplay::default_display(self)
    }

    /// Displays this plan and its descendants with the default plan extractors.
    pub fn tree_display(&self) -> PlanTreeDisplay<'_> {
        PlanTreeDisplay::default_display(self)
    }

    /// Creates a composable tree display with no extractors.
    pub fn tree_display_builder(&self) -> PlanTreeDisplay<'_> {
        PlanTreeDisplay::new(self)
    }
}

#[cfg(test)]
mod tests;
