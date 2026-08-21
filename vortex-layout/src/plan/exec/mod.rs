// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Experimental self-paced plan execution.
//!
//! This module intentionally supports only `Struct<Chunked<Flat<i64>>>`. It is an executable
//! control-plane experiment and does not replace [`super::PlanVTable::execute`].

mod baseline;
mod evaluate;
mod graph;
mod model;
mod reactor;
mod slots;

pub use baseline::*;
pub use evaluate::*;
pub use graph::*;
pub use model::*;
pub use reactor::*;

#[cfg(test)]
mod tests;
