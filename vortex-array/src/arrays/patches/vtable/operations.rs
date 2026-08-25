// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::arrays::PrimitiveArray;
use crate::arrays::patches::PATCH_BLOCK_SIZE;
use crate::arrays::patches::PatchCombine;
use crate::arrays::patches::PatchFn;
use crate::arrays::patches::Patches;
use crate::arrays::patches::PatchesArrayExt;
use crate::arrays::patches::PatchesArraySlotsExt;
use crate::match_each_native_ptype;
use crate::optimizer::ArrayOptimizer;
use crate::scalar::Scalar;

impl OperationsVTable<Patches> for Patches {
    fn scalar_at(
        array: ArrayView<'_, Patches>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let padded = index + array.offset();
        let block = padded / PATCH_BLOCK_SIZE;

        #[expect(
            clippy::cast_possible_truncation,
            reason = "N % 1024 always fits in u16"
        )]
        let rel = (padded % PATCH_BLOCK_SIZE) as u16;

        // Constant-time lookup of this block's contiguous patch range.
        let range = array.block_patch_range(block)?;

        if range.is_empty() {
            return array.inner().execute_scalar(index, ctx);
        }

        // Get this block's indices, potentially decoding them to avoid the overhead of repeated
        // scalar_at calls. The in-block indices are sorted, so a binary search over at most 1024
        // u16 values (1-2 cache lines for typical patch counts) finds the patch.
        let block_indices = array
            .indices()
            .slice(range.clone())?
            .optimize()?
            .execute::<PrimitiveArray>(ctx)?;

        let Ok(found) = block_indices.as_slice::<u16>().binary_search(&rel) else {
            // No patch at this position: access the underlying value.
            return array.inner().execute_scalar(index, ctx);
        };

        let patch = array
            .values()
            .execute_scalar(range.start + found, ctx)?
            .cast(array.dtype())?;

        match array.patch_fn() {
            PatchFn::Overwrite => Ok(patch),
            patch_fn => {
                let base = array.inner().execute_scalar(index, ctx)?;
                combine_scalars(patch_fn, &base, &patch)
            }
        }
    }
}

/// Combine two primitive scalars of the array's dtype according to `patch_fn`.
fn combine_scalars(patch_fn: PatchFn, base: &Scalar, patch: &Scalar) -> VortexResult<Scalar> {
    let dtype = base.dtype();
    match_each_native_ptype!(dtype.as_ptype(), |T| {
        // A null base value stays null: patch validity always comes from the inner array.
        let Some(base_value) = base.as_primitive().typed_value::<T>() else {
            return Ok(Scalar::null(dtype.clone()));
        };
        let patch_value = patch
            .as_primitive()
            .typed_value::<T>()
            .ok_or_else(|| vortex_err!("patch value must be non-null"))?;
        Ok(Scalar::primitive(
            T::combine(patch_fn, base_value, patch_value),
            dtype.nullability(),
        ))
    })
}
