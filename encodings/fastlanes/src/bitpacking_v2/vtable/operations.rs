// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_error::VortexResult;

use crate::BitPackedV2;
use crate::BitPackedV2ArrayExt;
use crate::bitpacking_v2::array::decompress::unpack_v2_single;

impl OperationsVTable<BitPackedV2> for BitPackedV2 {
    fn scalar_at(
        array: ArrayView<'_, BitPackedV2>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Ok(
            if let Some(patches) = array.patches()
                && let Some(patch) = patches.get_patched(index, ctx)?
            {
                patch
            } else {
                unpack_v2_single(array, index)
            },
        )
    }
}
