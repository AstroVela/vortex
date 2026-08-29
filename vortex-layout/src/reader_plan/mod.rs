// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Heap-allocated physical plans for layout scans.

mod display;
mod plans;
mod scan;

use std::any::Any;
use std::borrow::Cow;
use std::sync::Arc;

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
pub(crate) use plans::LayoutPlan;
pub use plans::ListPlan;
pub(crate) use plans::RowIdxPlan;
pub use plans::StructPlan;
pub use scan::ReaderPlanScan;
use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use crate::LayoutReaderContext;
use crate::LayoutReaderRef;
use crate::segments::SegmentSource;

/// Shared handle to a heap-allocated physical plan.
pub type PlanRef = Arc<dyn Plan>;

/// A heap-allocated physical plan over a row domain.
///
/// Layout plans expose their optimizer-facing children in a stable logical order. Optional child
/// slots count toward [`child_count`](Self::child_count) and are returned as `None` by
/// [`child`](Self::child) when absent. Once optimization is complete, [`new_reader`](Self::new_reader)
/// recursively materializes the corresponding [`crate::LayoutReader`] tree.
pub trait Plan: 'static + Send + Sync {
    /// Returns this plan as [`Any`] for plan-specific optimization rules.
    fn as_any(&self) -> &dyn Any;

    /// Returns the display name of this plan kind.
    ///
    /// Plans should override this when the fully qualified Rust type name is not appropriate.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    /// Recursively optimizes this plan.
    fn optimize(&self) -> VortexResult<PlanRef>;

    /// Constructs the reader corresponding to this optimized plan node.
    fn new_reader(
        &self,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef>;

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

impl dyn Plan + '_ {
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
