// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Two-dimensional tiled encoding for primitive Vortex fixed-size lists.

mod array;
mod gather;
mod geometry;
mod kernel;
mod operations;
mod rules;
mod slice;
mod transpose;

pub use array::*;
pub use geometry::TileBounds;
pub use geometry::TileBoundsIter;
pub use geometry::TileGeometry;
use vortex_array::session::ArraySessionExt;
use vortex_session::VortexSession;

/// Registers the tiled fixed-size-list array encoding in `session`.
pub fn initialize(session: &VortexSession) {
    session.arrays().register(TiledFixedSizeList);
    kernel::initialize(session);
}

#[cfg(test)]
mod tests;
