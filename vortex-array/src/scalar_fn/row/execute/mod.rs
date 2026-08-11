// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Row-loop execution for owned outputs and output sinks.
//!
//! [`owned`] stores one independent value per row and can reduce failure evidence. [`sink`]
//! drives output builders whose row handles may refer to shared batch state.

mod owned;
pub(super) use owned::execute_owned;
pub(super) use owned::execute_owned_infallible;

mod outcome;
pub use outcome::RowExecution;

mod sink;
pub(super) use sink::execute_sink;
pub(super) use sink::execute_sink_valid_rows;
