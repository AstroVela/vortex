// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Native range-bin and ANS entropy codec for Vortex numeric arrays.

mod array;
mod bit_split;
mod block_residual_array;
mod codec;
mod ordered_float_array;
mod patched_for;
mod rules;

pub use array::*;
pub use bit_split::BitSplitCodec;
pub use block_residual_array::*;
pub use codec::RangeEntropyCodec;
pub use codec::RangeEntropyParts;
pub use codec::RangeGroupedCodec;
pub use codec::RangePackedCodec;
pub use codec::RangeTwoLevelCodec;
pub use ordered_float_array::*;
pub use patched_for::BlockResidualParts;
pub use patched_for::PatchedFoRCodec;
use vortex_array::session::ArraySessionExt;
use vortex_session::VortexSession;

/// Register the range entropy encoding in one session.
pub fn initialize(session: &VortexSession) {
    session.arrays().register(RangeEntropy);
    session.arrays().register(BlockResidual);
    session.arrays().register(OrderedFloat);
}
