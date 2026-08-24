// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Ordered-float and block-residual array encodings.

mod block_residual_array;
mod codec;
mod ordered_float_array;
mod scalar_fn;

pub use block_residual_array::*;
pub(crate) use codec::BlockResidualCodec;
pub(crate) use codec::BlockResidualParts;
pub use ordered_float_array::*;
pub use scalar_fn::*;
use vortex_array::scalar_fn::session::ScalarFnSessionExt;
use vortex_array::session::ArraySessionExt;
use vortex_session::VortexSession;

/// Register the ordered-float and block-residual encodings in one session.
pub fn initialize(session: &VortexSession) {
    session.arrays().register(BlockResidual);
    session.arrays().register(OrderedFloat);
    session.scalar_fns().register(OrderedFloatDecode);
}
