// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![deny(missing_docs)]

//! An experimental morsel-driven scan executor for Vortex layouts.
//!
//! This crate is the P1 spine of the design recorded in
//! `docs/developer-guide/internals/scan-execution-models/morsel-based-plan-execution.md`: the scan
//! is cut into *morsels* (contiguous root row ranges), and each morsel is driven by a tree of
//! stateful [`ExecNode`] state machines that pull values from their children.
//!
//! The two halves of the contract are:
//!
//! * [`ExecNode::next_plan`] — planning. A node *names* the IO it will need by registering
//!   [`IoUse`](io::IoUse)s against the [`IoPlane`](io::IoPlane), which hands back tickets. Nodes
//!   never perform IO themselves. Planning is budget-bounded and resumable: a node that exhausts
//!   its quantum yields [`PlanItem::Plan`] and resumes from its own cursor on the next call.
//! * [`ExecNode::execute`] — value production. A node may only wait on tickets its own planning
//!   stream already emitted; it consumes the cell behind a ticket and produces
//!   [`ValueBatch`]es covering a dense range of input rows.
//!
//! Compared to the V1 `LayoutReader` path this executor differs in three measurable ways:
//!
//! 1. There is no future per evaluation. A morsel is driven inline, depth-first, on one thread.
//! 2. Decoded segments are cached per thread, so a chunk straddling several morsels is decoded
//!    once rather than once per morsel that touches it.
//! 3. Morsels are self-scheduled off one atomic cursor; emission order is restored by index.
//!
//! Only the FLAT, CHUNKED and STRUCT layout nodes are supported, plus the FILTER and
//! CONJUNCT_PARALLEL operators. Anything else is rejected at build time by [`build::build_plan`].

pub mod build;
pub mod cache;
pub mod driver;
#[cfg(any(test, feature = "_test-harness"))]
pub mod fixtures;
#[cfg(any(test, feature = "_test-harness"))]
pub mod harness;
pub mod io;
pub mod node;
pub mod nodes;
pub mod stats;

pub use build::ExecPlan;
pub use build::build_plan;
pub use driver::MorselScan;
pub use driver::morsels;
pub use node::ExecCx;
pub use node::ExecNode;
pub use node::ExecPoll;
pub use node::PlanCx;
pub use node::PlanItem;
pub use node::PlanPoll;
pub use node::Value;
pub use node::ValueBatch;
pub use stats::ScanStats;

#[cfg(test)]
mod tests;
