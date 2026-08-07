// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Heap-allocated physical plans for layout scans.

mod children;
mod display;
mod execution;
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
pub use execution::PlanArrayFuture;
pub use execution::PlanExecutionContext;
pub use plans::ChunkedPlan;
pub use plans::DictPlan;
pub use plans::ExpressionPlan;
pub use plans::FlatPlan;
pub use plans::ListPlan;
pub use plans::RowIdxPartitionPlan;
pub use plans::RowIdxPlan;
pub use plans::RowIdxValuesPlan;
pub use plans::StructPlan;
pub use plans::ZonedPlan;
use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::LayoutRef;
use crate::layouts::chunked::Chunked;
use crate::layouts::dict::Dict;
use crate::layouts::flat::Flat;
use crate::layouts::list::List;
use crate::layouts::struct_::Struct;
use crate::layouts::zoned::LegacyStats;
use crate::layouts::zoned::Zoned;

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

    /// Recursively optimizes this plan and all of its children while preserving its dtype and row
    /// domain.
    fn optimize(&self) -> VortexResult<PlanRef>;

    /// Returns whether [`optimize`](Self::optimize) could rewrite this subtree.
    ///
    /// Rules only fire at [`ExpressionPlan`] boundaries, so a subtree built purely from a stored
    /// layout by [`new_plan`] contains nothing to rewrite; optimizing it is the identity. Plans
    /// that know their subtree is expression-free return `false` so parents can reuse the shared
    /// subtree, keeping lazily initialized children cached across optimize passes. The
    /// conservative default is `true`.
    fn needs_optimize(&self) -> bool {
        true
    }

    /// Executes this plan for `row_range`, returning values selected by `mask`.
    ///
    /// The row range is expressed in this plan's row domain. The returned array has one row for
    /// every true value in `mask`.
    fn execute(
        &self,
        _ctx: &PlanExecutionContext,
        _row_range: &std::ops::Range<u64>,
        _mask: vortex_array::MaskFuture,
    ) -> VortexResult<PlanArrayFuture> {
        vortex_bail!("Plan execution is not implemented for '{}'", self.name())
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

/// Optimizes `child` unless its subtree is expression-free.
///
/// Reusing the shared plan keeps lazily initialized children cached across optimize passes
/// instead of rebuilding identical subtrees.
pub(crate) fn optimize_child(child: &PlanRef) -> VortexResult<PlanRef> {
    if child.needs_optimize() {
        child.optimize()
    } else {
        Ok(Arc::clone(child))
    }
}

/// Constructs a physical plan for a stored layout tree.
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
    if layout.is::<Zoned>() || layout.is::<LegacyStats>() {
        return Ok(Arc::new(ZonedPlan::try_new(layout)?));
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
