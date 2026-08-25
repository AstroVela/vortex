// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rustc_hash::FxHashMap;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Patches;
use crate::arrays::PrimitiveArray;
use crate::arrays::dict::TakeExecute;
use crate::arrays::patches::PATCH_BLOCK_SIZE;
use crate::arrays::patches::PatchCombine;
use crate::arrays::patches::PatchFn;
use crate::arrays::patches::PatchesArrayExt;
use crate::arrays::patches::PatchesArraySlotsExt;
use crate::arrays::primitive::PrimitiveDataParts;
use crate::dtype::IntegerPType;
use crate::match_each_native_ptype;
use crate::match_each_unsigned_integer_ptype;

impl TakeExecute for Patches {
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only pushdown take when we have primitive types.
        if !array.dtype().is_primitive() {
            return Ok(None);
        }

        // Perform take on the inner array, including the placeholders.
        let inner = array
            .inner()
            .take(indices.clone())?
            .execute::<PrimitiveArray>(ctx)?;

        let PrimitiveDataParts {
            buffer,
            validity,
            ptype,
        } = inner.into_data_parts();

        let indices_ptype = indices.dtype().as_ptype();

        match_each_unsigned_integer_ptype!(indices_ptype, |I| {
            match_each_native_ptype!(ptype, |V| {
                let indices = indices.clone().execute::<PrimitiveArray>(ctx)?;
                let skip_indices = array
                    .skip_indices()
                    .clone()
                    .execute::<PrimitiveArray>(ctx)?;
                let patch_indices = array.indices().clone().execute::<PrimitiveArray>(ctx)?;
                let patch_values = array.values().clone().execute::<PrimitiveArray>(ctx)?;
                let mut output = Buffer::<V>::from_byte_buffer(buffer.unwrap_host()).into_mut();
                take_map(
                    output.as_mut(),
                    indices.as_slice::<I>(),
                    array.offset(),
                    array.len(),
                    skip_indices.as_slice::<u32>(),
                    patch_indices.as_slice::<u16>(),
                    patch_values.as_slice::<V>(),
                    array.patch_fn(),
                );

                // SAFETY: output and validity still have same length after take_map returns.
                unsafe {
                    Ok(Some(
                        PrimitiveArray::new_unchecked(output.freeze(), validity).into_array(),
                    ))
                }
            })
        })
    }
}

/// Take patches for the given `indices` and combine them onto an `output` using a hash map.
///
/// First builds a hashmap from logical index to patch value, then uses the hashmap in a loop to
/// combine the values. Each block's patches are a contiguous run, so the map is built with one
/// linear pass over `skip_indices`.
#[expect(clippy::too_many_arguments)]
fn take_map<I: IntegerPType, V: PatchCombine>(
    output: &mut [V],
    indices: &[I],
    offset: usize,
    len: usize,
    skip_indices: &[u32],
    patch_index: &[u16],
    patch_value: &[V],
    patch_fn: PatchFn,
) {
    let n_blocks = (offset + len).div_ceil(PATCH_BLOCK_SIZE);
    // Build a hashmap of logical index -> patch value.
    let mut index_map = FxHashMap::with_capacity_and_hasher(patch_index.len(), Default::default());
    for block in 0..n_blocks {
        let start = skip_indices[block] as usize;
        let stop = skip_indices[block + 1] as usize;
        for patch_idx in start..stop {
            let index = block * PATCH_BLOCK_SIZE + patch_index[patch_idx] as usize;
            if index >= offset && index < offset + len {
                index_map.insert(index - offset, patch_value[patch_idx]);
            }
        }
    }

    // Now, iterate the take indices using the prebuilt hashmap.
    // Undefined/null indices will miss the hash map, which we can ignore.
    for (output_index, index) in indices.iter().enumerate() {
        let index = index.as_();
        if let Some(&patch) = index_map.get(&index) {
            let out = &mut output[output_index];
            *out = V::combine(patch_fn, *out, patch);
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::Patches;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::dict::TakeExecute;
    use crate::arrays::patches::PatchFn;
    use crate::assert_arrays_eq;

    fn make_array(patch_fn: PatchFn) -> VortexResult<crate::ArrayRef> {
        let mut ctx = crate::array_session().create_execution_ctx();
        let inner = PrimitiveArray::from_iter((0..3000u32).map(|i| i % 8)).into_array();
        let patches = crate::patches::Patches::new(
            3000,
            0,
            buffer![5u32, 1030, 2900].into_array(),
            buffer![1000u32, 2000, 3000].into_array(),
            None,
        )?;
        Ok(Patches::from_array_and_patches(inner, &patches, patch_fn, &mut ctx)?.into_array())
    }

    #[test]
    fn test_take_overwrite() -> VortexResult<()> {
        let mut ctx = crate::array_session().create_execution_ctx();
        let array = make_array(PatchFn::Overwrite)?;
        let array = array.as_opt::<Patches>().unwrap();

        let take_indices = buffer![5u64, 6, 1030, 2900].into_array();
        let taken = Patches::take(array, &take_indices, &mut ctx)?.unwrap();

        let expected = PrimitiveArray::from_iter([1000u32, 6, 2000, 3000]);
        assert_arrays_eq!(expected, taken, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_take_add() -> VortexResult<()> {
        let mut ctx = crate::array_session().create_execution_ctx();
        let array = make_array(PatchFn::Add)?;
        let array = array.as_opt::<Patches>().unwrap();

        let take_indices = buffer![5u64, 6, 1030].into_array();
        let taken = Patches::take(array, &take_indices, &mut ctx)?.unwrap();

        // Base at 5 is 5 % 8 = 5, at 1030 is 1030 % 8 = 6.
        let expected = PrimitiveArray::from_iter([1005u32, 6, 2006]);
        assert_arrays_eq!(expected, taken, &mut ctx);
        Ok(())
    }
}
