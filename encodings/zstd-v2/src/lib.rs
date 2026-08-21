// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Zstd compression for variable-width Vortex arrays, storing lengths apart from value bytes.
//!
//! [`ZstdV2Array`] compresses the value bytes into one or more zstd frames, and the length of
//! every value into a frame of its own. `vortex.zstd` interleaves a length prefix with each value
//! instead, which means finding the *n*th value costs a walk through every value before it, and
//! nothing can be located without decompressing the data that holds it.
//!
//! Keeping the lengths apart buys three things:
//!
//! - Value offsets come from a prefix sum over the lengths rather than a pointer chase.
//! - The views of the canonical form point straight into the decompressed frames, so decoding
//!   copies no value bytes at all.
//! - A slice or filter decompresses only the frames holding the values it asks for, because the
//!   lengths alone say where each value lives.

pub use array::*;
use vortex_array::session::ArraySessionExt;
use vortex_session::VortexSession;

mod array;
mod compute;
mod kernel;
mod rules;

#[cfg(test)]
mod test;

/// Registers the `vortex.zstd.v2` encoding and its kernels in `session`.
pub fn initialize(session: &VortexSession) {
    session.arrays().register(ZstdV2);
    kernel::initialize(session);
}

#[derive(Clone, prost::Message)]
/// Metadata for one frame of value bytes.
pub struct ZstdV2FrameMetadata {
    /// Uncompressed byte size of this frame.
    #[prost(uint64, tag = "1")]
    pub uncompressed_size: u64,
    /// Number of stored values whose bytes live in this frame.
    #[prost(uint64, tag = "2")]
    pub n_values: u64,
}

#[derive(Clone, prost::Message)]
/// Serialized metadata for a [`ZstdV2Array`].
pub struct ZstdV2Metadata {
    /// Uncompressed byte size of the lengths frame, which holds one `u32` per stored value.
    #[prost(uint64, tag = "1")]
    pub lengths_uncompressed_size: u64,
    /// Metadata for each frame of value bytes, in order.
    #[prost(message, repeated, tag = "2")]
    pub frames: Vec<ZstdV2FrameMetadata>,
}

impl ZstdV2Metadata {
    /// Total number of stored values across every frame.
    pub fn n_values(&self) -> u64 {
        self.frames.iter().map(|frame| frame.n_values).sum()
    }
}
