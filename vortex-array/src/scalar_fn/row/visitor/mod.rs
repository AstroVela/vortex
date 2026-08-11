// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Visits that plan or execute the concrete row signature selected by [`RowFn::dispatch`].
//!
//! [`RowFn::dispatch`]: crate::scalar_fn::RowFn::dispatch

mod check;
pub(super) use check::assert_owned_output_needs_no_drop;

mod execute;
pub(super) use execute::ExecuteRows;
pub(super) use execute::ExecuteValidRows;

mod plan;
pub(super) use plan::PlanRows;

mod row_visitor;
pub use row_visitor::RowVisitor;
