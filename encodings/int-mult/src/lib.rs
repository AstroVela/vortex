// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Integer multiplication transform with independent multiplied and additive children.

mod array;
mod rules;
mod slice;

pub use array::*;
use vortex_array::session::ArraySessionExt;
use vortex_session::VortexSession;

/// Register the integer multiplication encoding in one session.
pub fn initialize(session: &VortexSession) {
    session.arrays().register(IntMult);
}
