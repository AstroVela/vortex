// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::BitPackedV2ArrayExt;
pub use array::BitPackedV2ArraySlotsExt;
pub use array::BitPackedV2Data;
pub use array::BitPackedV2DataParts;
pub use array::BitPackedV2Slots;
pub use array::compress as bitpack_v2_compress;
pub use array::decompress as bitpack_v2_decompress;

pub(crate) mod compute;

mod vtable;
pub use vtable::BitPackedV2;
pub use vtable::BitPackedV2Array;

#[cfg(test)]
mod tests;

pub(crate) fn initialize(session: &vortex_session::VortexSession) {
    vtable::initialize(session);
}
