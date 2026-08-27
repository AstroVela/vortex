// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::slice::SliceKernel;
use vortex_array::arrays::slice::SliceReduce;
use vortex_array::patches::Patches;
use vortex_error::VortexResult;

use crate::BitPackedV2;
use crate::BitPackedV2ArrayExt;
use crate::FL_CHUNK_SIZE;

impl SliceReduce for BitPackedV2 {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        // We cannot access buffers (to slice the patches).
        if array.patches().is_some() {
            return Ok(None);
        }

        Ok(Some(slice_bitpacked_v2(array, range, None)?))
    }
}

impl SliceKernel for BitPackedV2 {
    fn slice(
        array: ArrayView<'_, Self>,
        range: Range<usize>,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let patches = array
            .patches()
            .map(|p| p.slice(range.clone()))
            .transpose()?
            .flatten();

        Ok(Some(slice_bitpacked_v2(array, range, patches)?))
    }
}

/// Slice by dropping whole chunks: the retained chunks keep their own bit widths, and the
/// leading values of the first retained chunk are skipped through the array's offset.
fn slice_bitpacked_v2(
    array: ArrayView<'_, BitPackedV2>,
    range: Range<usize>,
    patches: Option<Patches>,
) -> VortexResult<ArrayRef> {
    let offset_start = range.start + array.offset() as usize;
    let offset_stop = range.end + array.offset() as usize;
    let offset = offset_start % FL_CHUNK_SIZE;
    let first_chunk = offset_start / FL_CHUNK_SIZE;
    let chunk_end = offset_stop.div_ceil(FL_CHUNK_SIZE);

    let chunk_byte_offsets = array.chunk_byte_offsets();
    let packed_start = chunk_byte_offsets[first_chunk] as usize;
    let packed_stop = chunk_byte_offsets[chunk_end] as usize;

    Ok(BitPackedV2::try_new(
        array.packed().slice(packed_start..packed_stop),
        array.bit_widths_buffer().slice(first_chunk..chunk_end),
        array.dtype().as_ptype(),
        array.validity()?.slice(range.clone())?,
        patches,
        range.len(),
        offset as u16,
    )?
    .into_array())
}
