// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Patches;
use crate::arrays::patches::PATCH_BLOCK_SIZE;
use crate::arrays::patches::PatchesArrayExt;
use crate::arrays::patches::PatchesArraySlotsExt;
use crate::arrays::patches::PatchesSlots;
use crate::arrays::slice::SliceReduce;

impl SliceReduce for Patches {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        if range.is_empty() {
            // Let the generic pathway produce an empty result.
            return Ok(None);
        }

        // We **always** slice the patches at 1024-element block boundaries. We keep the offset
        // around so that when we execute we know how much to chop off. The skip_indices values
        // stay absolute into the (unsliced) indices/values children.
        let new_offset = (range.start + array.offset()) % PATCH_BLOCK_SIZE;
        let block_start = (range.start + array.offset()) / PATCH_BLOCK_SIZE;
        let block_stop = (range.end + array.offset()).div_ceil(PATCH_BLOCK_SIZE);
        let sliced_skip_indices = array.skip_indices().slice(block_start..block_stop + 1)?;

        // Unlike the patches, we slice the inner to the exact range. This is handled at execution
        // time by skipping patch positions that are < offset or >= offset + len.
        let inner = array.inner().slice(range.start..range.end)?;
        let len = inner.len();

        let slots = PatchesSlots {
            inner,
            skip_indices: sliced_skip_indices,
            indices: array.indices().clone(),
            values: array.values().clone(),
        }
        .into_slots();

        Ok(Some(
            unsafe {
                Patches::new_unchecked(
                    array.dtype().clone(),
                    len,
                    slots,
                    new_offset,
                    array.patch_fn(),
                )
            }
            .into_array(),
        ))
    }
}
