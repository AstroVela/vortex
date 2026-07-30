//! Two-dimensional tiled encoding for primitive Vortex fixed-size lists.

mod array;
mod geometry;
mod operations;
mod transpose;

pub use array::*;
pub use geometry::TileBounds;
pub use geometry::TileGeometry;
use vortex_array::session::ArraySessionExt;
use vortex_session::VortexSession;

/// Registers the tiled fixed-size-list array encoding in `session`.
pub fn initialize(session: &VortexSession) {
    session.arrays().register(TiledFixedSizeList);
}

#[cfg(test)]
mod tests;
