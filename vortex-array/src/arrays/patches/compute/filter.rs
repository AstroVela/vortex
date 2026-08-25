// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::FilterArray;
use crate::arrays::Patches;
use crate::arrays::filter::FilterReduce;
use crate::arrays::patches::PATCH_BLOCK_SIZE;
use crate::arrays::patches::PatchesArrayExt;

impl FilterReduce for Patches {
    fn filter(array: ArrayView<'_, Self>, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        // Find the contiguous block range that the mask covers. We use this to slice the inner
        // components, then wrap the rest up with another FilterArray.
        //
        // This is helpful when we have a very selective filter that is clustered to a small
        // range.
        let (block_start, block_stop) = match mask.slices() {
            AllOr::All | AllOr::None => {
                // This is handled as the precondition to this method, see the FilterReduce
                // documentation.
                unreachable!("mask must be a MaskValues here")
            }
            AllOr::Some(slices) => {
                let (first, _) = slices[0];
                let (_, last) = slices[slices.len() - 1];

                // Convert mask indices to padded positions by adding offset.
                (
                    (array.offset() + first) / PATCH_BLOCK_SIZE,
                    (array.offset() + last).div_ceil(PATCH_BLOCK_SIZE),
                )
            }
        };

        let n_blocks = (array.offset() + array.len()).div_ceil(PATCH_BLOCK_SIZE);

        // If all blocks are already covered, there is nothing to do.
        if block_start == 0 && block_stop == n_blocks {
            return Ok(None);
        }

        let sliced = array.slice_blocks(block_start..block_stop)?;

        // Slice the mask according to if the block is sliced.
        // Convert block bounds back to mask indices by subtracting offset.
        let mask_start = (block_start * PATCH_BLOCK_SIZE).saturating_sub(array.offset());
        let mask_end = (block_stop * PATCH_BLOCK_SIZE)
            .saturating_sub(array.offset())
            .min(array.len());
        let remainder = mask.slice(mask_start..mask_end);

        Ok(Some(
            FilterArray::new(sliced.into_array(), remainder).into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::Patches;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::patches::PatchFn;
    use crate::assert_arrays_eq;
    use crate::optimizer::ArrayOptimizer;

    fn patches_array(
        inner_len: usize,
        patch_indices: &[u32],
        patch_values: &[u16],
    ) -> VortexResult<crate::ArrayRef> {
        let mut ctx = crate::array_session().create_execution_ctx();
        let inner = PrimitiveArray::from_iter((0..inner_len).map(|_| u16::MIN)).into_array();
        let patches = crate::patches::Patches::new(
            inner_len,
            0,
            PrimitiveArray::from_iter(patch_indices.iter().copied()).into_array(),
            PrimitiveArray::from_iter(patch_values.iter().copied()).into_array(),
            None,
        )?;
        Ok(
            Patches::from_array_and_patches(inner, &patches, PatchFn::Overwrite, &mut ctx)?
                .into_array(),
        )
    }

    #[test]
    fn test_filter_noop() -> VortexResult<()> {
        // Filter that doesn't prune any blocks (all data fits in one block).
        let mut ctx = crate::array_session().create_execution_ctx();
        let array = patches_array(5, &[3, 4], &[u16::MAX, u16::MAX])?;

        let mask = Mask::from_iter([true, false, false, false, true]);
        let filtered = array
            .filter(mask)?
            .optimize()?
            .execute::<PrimitiveArray>(&mut ctx)?;

        let expected = PrimitiveArray::from_iter([u16::MIN, u16::MAX]);
        assert_arrays_eq!(expected, filtered, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_filter_prunes_blocks() -> VortexResult<()> {
        let mut ctx = crate::array_session().create_execution_ctx();
        let array = patches_array(4096, &[1024, 1025], &[u16::MAX, u16::MAX])?;

        // Filter that only touches the middle 2 blocks.
        let mask = Mask::from_indices(4096, vec![1024, 1025, 3000]);
        let filtered = array
            .filter(mask)?
            .optimize()?
            .execute::<PrimitiveArray>(&mut ctx)?;

        let expected = PrimitiveArray::from_iter([u16::MAX, u16::MAX, u16::MIN]);
        assert_arrays_eq!(expected, filtered, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_filter_sliced() -> VortexResult<()> {
        // Filter on a sliced PatchesArray to exercise the codepath where offset > 0.
        let mut ctx = crate::array_session().create_execution_ctx();
        let array = patches_array(6144, &[2048, 2049], &[u16::MAX, u16::MAX])?;

        // Slice mid-block to create offset > 0. After slicing [1000..5120], patches are at
        // relative indices 1048 and 1049.
        let sliced = array.slice(1000..5120)?;
        assert_eq!(sliced.len(), 4120);

        let mask = Mask::from_indices(4120, vec![1048, 1049, 3000]);
        let filtered = sliced
            .filter(mask)?
            .optimize()?
            .execute::<PrimitiveArray>(&mut ctx)?;

        let expected = PrimitiveArray::from_iter([u16::MAX, u16::MAX, u16::MIN]);
        assert_arrays_eq!(expected, filtered, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_filter_last_blocks() -> VortexResult<()> {
        let mut ctx = crate::array_session().create_execution_ctx();
        let array = patches_array(6144, &[5000, 6000], &[u16::MAX, u16::MAX])?;

        let sliced = array.slice(1024..6144)?.optimize()?;
        assert_eq!(sliced.len(), 5120);

        // Filter that touches only the last 2 blocks.
        let mask = Mask::from_indices(5120, vec![3976, 4976, 5119]);
        let filtered = sliced
            .filter(mask)?
            .optimize()?
            .execute::<PrimitiveArray>(&mut ctx)?;

        let expected = PrimitiveArray::from_iter([u16::MAX, u16::MAX, u16::MIN]);
        assert_arrays_eq!(expected, filtered, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_filter_all_indices() -> VortexResult<()> {
        let mut ctx = crate::array_session().create_execution_ctx();
        let array = patches_array(4096, &[5, 4090], &[u16::MAX, u16::MAX])?;

        // Mask spanning all blocks: filter reduce declines, generic pathway still works.
        let mask = Mask::from_indices(4096, vec![5, 2000, 4090]);
        let filtered = array
            .filter(mask)?
            .optimize()?
            .execute::<PrimitiveArray>(&mut ctx)?;

        let expected = PrimitiveArray::from_iter([u16::MAX, u16::MIN, u16::MAX]);
        assert_arrays_eq!(expected, filtered, &mut ctx);
        Ok(())
    }
}
