// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::Canonical;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Chunked;
use crate::arrays::ChunkedArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::chunked::ChunkedArrayExt;
use crate::arrays::dict::TakeExecute;
use crate::builders::builder_with_capacity_in;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::executor::ExecutionCtx;
use crate::validity::Validity;

/// A bucket is "dense" in its chunk when building a chunk-length bitmap is cheaper than
/// comparison-sorting the bucket. Bitmap cost is O(chunk_len / 64) words; sort cost is
/// O(k log k) element moves, so the crossover sits around one hit per 64 rows.
const DENSE_BUCKET_DIVISOR: usize = 64;

fn take_chunked(
    array: ArrayView<'_, Chunked>,
    indices: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let indices = indices
        .cast(DType::Primitive(PType::U64, indices.dtype().nullability()))?
        .execute::<PrimitiveArray>(ctx)?;

    let indices_mask = indices
        .as_ref()
        .validity()?
        .execute_mask(indices.as_ref().len(), ctx)?;
    let indices_values = indices.as_slice::<u64>();
    let n = indices_values.len();

    // Fast path: strictly increasing non-nullable indices contain no duplicates and preserve
    // order, so the per-chunk filters concatenate directly into the result with no reorder.
    if indices.dtype().nullability() == Nullability::NonNullable
        && indices_values.is_sorted_by(|a, b| a < b)
    {
        return take_chunked_sorted_unique(array, indices_values);
    }

    let chunk_offsets = array.chunk_offsets();
    let nchunks = array.nchunks();

    // 1. Resolve each valid index to its chunk and count per-chunk occupancy. Grouping by
    //    chunk only needs a stable O(n) bucket partition, not a comparison sort of all
    //    indices. Null indices are skipped — their final_take slots stay 0 and are masked
    //    null by validity.
    let mut chunk_ids = vec![0u32; n];
    let mut counts = vec![0usize; nchunks];
    if nchunks > 0 {
        // Uniform chunk sizing (every chunk but the last has the same length) resolves with
        // a divide; otherwise fall back to a binary search over the chunk offsets.
        let chunk_len0 = chunk_offsets[1] - chunk_offsets[0];
        let uniform = chunk_len0 > 0
            && chunk_offsets[..nchunks]
                .windows(2)
                .all(|w| w[1] - w[0] == chunk_len0);
        for i in 0..n {
            if !indices_mask.value(i) {
                continue;
            }
            let v = usize::try_from(indices_values[i])?;
            let chunk_id = if uniform {
                (v / chunk_len0).min(nchunks - 1)
            } else {
                chunk_offsets.partition_point(|&off| off <= v) - 1
            };
            chunk_ids[i] = u32::try_from(chunk_id)?;
            counts[chunk_id] += 1;
        }
    }

    // 2. Scatter (local index, original position) into per-chunk buckets, preserving the
    //    original relative order within each chunk.
    let mut bucket_starts = vec![0usize; nchunks + 1];
    for chunk_id in 0..nchunks {
        bucket_starts[chunk_id + 1] = bucket_starts[chunk_id] + counts[chunk_id];
    }
    let total_valid = bucket_starts[nchunks];
    let mut bucket_local = vec![0usize; total_valid];
    let mut bucket_pos = vec![0usize; total_valid];
    let mut fill = bucket_starts.clone();
    for i in 0..n {
        if !indices_mask.value(i) {
            continue;
        }
        let chunk_id = chunk_ids[i] as usize;
        let slot = fill[chunk_id];
        fill[chunk_id] += 1;
        bucket_local[slot] = usize::try_from(indices_values[i])? - chunk_offsets[chunk_id];
        bucket_pos[slot] = i;
    }

    // 3. Per chunk: build a dedup filter mask and scatter final_take[orig_pos] = dedup_idx.
    //    Dense buckets use an idempotent bitmap (no sort at all); sparse buckets sort only
    //    their own few entries.
    let mut chunks = Vec::with_capacity(nchunks);
    let mut final_take = BufferMut::<u64>::with_capacity(n);
    final_take.push_n(0u64, n);
    let mut dedup_base = 0u64;
    let mut rank_scratch: Vec<u64> = Vec::new();

    for chunk_id in 0..nchunks {
        let bucket = bucket_starts[chunk_id]..bucket_starts[chunk_id + 1];
        if bucket.is_empty() {
            continue;
        }
        let locals = &bucket_local[bucket.clone()];
        let positions = &bucket_pos[bucket];
        let chunk_len = chunk_offsets[chunk_id + 1] - chunk_offsets[chunk_id];

        let filter_mask = if locals.len() >= chunk_len / DENSE_BUCKET_DIVISOR {
            let mut bits = BitBufferMut::new_unset(chunk_len);
            for &local in locals {
                bits.set(local);
            }
            let bits = bits.freeze();
            if rank_scratch.len() < chunk_len {
                rank_scratch.resize(chunk_len, 0);
            }
            let mut next_rank = dedup_base;
            bits.for_each_set_index(|local| {
                rank_scratch[local] = next_rank;
                next_rank += 1;
            });
            for (&local, &pos) in locals.iter().zip(positions) {
                final_take[pos] = rank_scratch[local];
            }
            dedup_base = next_rank;
            Mask::from_buffer(bits)
        } else {
            let mut pairs: Vec<(usize, usize)> = locals
                .iter()
                .copied()
                .zip(positions.iter().copied())
                .collect();
            pairs.sort_unstable();
            let mut unique: Vec<usize> = Vec::with_capacity(pairs.len());
            for &(local, pos) in &pairs {
                if unique.last() != Some(&local) {
                    unique.push(local);
                }
                final_take[pos] = dedup_base + (unique.len() - 1) as u64;
            }
            dedup_base += unique.len() as u64;
            Mask::from_indices(chunk_len, unique)
        };

        chunks.push(array.chunk(chunk_id).filter(filter_mask)?);
    }

    // 4. Flatten the filtered chunks through a builder. Unlike execute::<Canonical>, this
    //    produces truly flat leaves for nested dtypes (Struct/List/FSL), so the reorder take
    //    below stays a flat gather instead of cascading a fresh chunked take into every leaf.
    let flat = if chunks.is_empty() {
        // SAFETY: an empty chunk list trivially satisfies the dtype invariant.
        unsafe { ChunkedArray::new_unchecked(chunks, array.dtype().clone()) }
            .into_array()
            .execute::<Canonical>(ctx)?
            .into_array()
    } else {
        let total: usize = chunks.iter().map(|c| c.len()).sum();
        let mut builder = builder_with_capacity_in(ctx.allocator(), array.dtype(), total);
        for chunk in &chunks {
            chunk.append_to_builder(builder.as_mut(), ctx)?;
        }
        builder.finish_into_canonical(ctx).into_array()
    };

    // 5. Single take to restore original order and expand duplicates.
    //    Carry the original index validity so null indices produce null outputs.
    let take_validity = Validity::from_mask(indices_mask, indices.dtype().nullability());
    flat.take(PrimitiveArray::new(final_take.freeze(), take_validity).into_array())
}

/// Take with strictly increasing, non-nullable indices: filter each chunk with the sorted
/// in-range indices and return the filtered chunks directly — order is already correct and
/// there are no duplicates, so no reorder take is needed.
fn take_chunked_sorted_unique(
    array: ArrayView<'_, Chunked>,
    indices_values: &[u64],
) -> VortexResult<ArrayRef> {
    let chunk_offsets = array.chunk_offsets();
    let nchunks = array.nchunks();
    let mut chunks = Vec::new();
    let mut cursor = 0usize;

    for chunk_id in 0..nchunks {
        let chunk_start = chunk_offsets[chunk_id];
        let chunk_end_u64 = u64::try_from(chunk_offsets[chunk_id + 1])?;
        let range_end = cursor + indices_values[cursor..].partition_point(|&v| v < chunk_end_u64);
        if range_end > cursor {
            let chunk_len = chunk_offsets[chunk_id + 1] - chunk_start;
            let locals = indices_values[cursor..range_end]
                .iter()
                .map(|&v| usize::try_from(v).map(|v| v - chunk_start))
                .collect::<Result<Vec<usize>, _>>()?;
            let filter_mask = Mask::from_indices(chunk_len, locals);
            chunks.push(array.chunk(chunk_id).filter(filter_mask)?);
        }
        cursor = range_end;
    }

    if chunks.is_empty() {
        return Ok(Canonical::empty(array.dtype()).into_array());
    }

    // SAFETY: every chunk came from a filter on a chunk with the same dtype, and the index
    // nullability is NonNullable so the result dtype is unchanged. The result stays lazy;
    // the executor drives it further as needed.
    Ok(unsafe { ChunkedArray::new_unchecked(chunks, array.dtype().clone()) }.into_array())
}

impl TakeExecute for Chunked {
    fn take(
        array: ArrayView<'_, Chunked>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        take_chunked(array, indices, ctx).map(Some)
    }
}

#[cfg(test)]
mod test {
    use vortex_buffer::bitbuffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::BoolArray;
    use crate::arrays::ChunkedArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::arrays::chunked::ChunkedArrayExt;
    use crate::assert_arrays_eq;
    use crate::compute::conformance::take::test_take_conformance;
    use crate::dtype::FieldNames;
    use crate::dtype::Nullability;
    use crate::validity::Validity;

    #[test]
    fn test_take() {
        let mut ctx = array_session().create_execution_ctx();
        let a = buffer![1i32, 2, 3].into_array();
        let arr = ChunkedArray::try_new(vec![a.clone(), a.clone(), a.clone()], a.dtype().clone())
            .unwrap();
        assert_eq!(arr.nchunks(), 3);
        assert_eq!(arr.len(), 9);
        let indices = buffer![0u64, 0, 6, 4].into_array();

        let result = arr.take(indices).unwrap();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([1i32, 1, 1, 2]), &mut ctx);
    }

    #[test]
    fn test_take_nullable_values() {
        let mut ctx = array_session().create_execution_ctx();
        let a = PrimitiveArray::new(buffer![1i32, 2, 3], Validity::AllValid).into_array();
        let arr = ChunkedArray::try_new(vec![a.clone(), a.clone(), a.clone()], a.dtype().clone())
            .unwrap();
        assert_eq!(arr.nchunks(), 3);
        assert_eq!(arr.len(), 9);
        let indices = PrimitiveArray::new(buffer![0u64, 0, 6, 4], Validity::NonNullable);

        let result = arr.take(indices.into_array()).unwrap();
        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([1i32, 1, 1, 2].map(Some)),
            &mut ctx
        );
    }

    #[test]
    fn test_take_nullable_indices() {
        let mut ctx = array_session().create_execution_ctx();
        let a = buffer![1i32, 2, 3].into_array();
        let arr = ChunkedArray::try_new(vec![a.clone(), a.clone(), a.clone()], a.dtype().clone())
            .unwrap();
        assert_eq!(arr.nchunks(), 3);
        assert_eq!(arr.len(), 9);
        let indices = PrimitiveArray::new(
            buffer![0u64, 0, 6, 4],
            Validity::Array(bitbuffer![1 0 0 1].into_array()),
        );

        let result = arr.take(indices.into_array()).unwrap();
        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(1i32), None, None, Some(2)]),
            &mut ctx
        );
    }

    #[test]
    fn test_take_nullable_struct() {
        let mut ctx = array_session().create_execution_ctx();
        let struct_array =
            StructArray::try_new(FieldNames::default(), vec![], 100, Validity::NonNullable)
                .unwrap();

        let arr = ChunkedArray::from_iter(vec![
            struct_array.clone().into_array(),
            struct_array.into_array(),
        ]);

        let result = arr
            .take(PrimitiveArray::from_option_iter(vec![Some(0), None, Some(101)]).into_array())
            .unwrap();

        let expect = StructArray::try_new(
            FieldNames::default(),
            vec![],
            3,
            Validity::Array(BoolArray::from_iter(vec![true, false, true]).into_array()),
        )
        .unwrap();
        assert_arrays_eq!(result, expect, &mut ctx);
    }

    #[test]
    fn test_empty_take() {
        let mut ctx = array_session().create_execution_ctx();
        let a = buffer![1i32, 2, 3].into_array();
        let arr = ChunkedArray::try_new(vec![a.clone(), a.clone(), a.clone()], a.dtype().clone())
            .unwrap();
        assert_eq!(arr.nchunks(), 3);
        assert_eq!(arr.len(), 9);

        let indices = PrimitiveArray::empty::<u64>(Nullability::NonNullable);
        let result = arr.take(indices.into_array()).unwrap();

        assert!(result.is_empty());
        assert_eq!(result.dtype(), arr.dtype());
        assert_arrays_eq!(
            result,
            PrimitiveArray::empty::<i32>(Nullability::NonNullable),
            &mut ctx
        );
    }

    #[test]
    fn test_take_shuffled_indices() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let c0 = buffer![0i32, 1, 2].into_array();
        let c1 = buffer![3i32, 4, 5].into_array();
        let c2 = buffer![6i32, 7, 8].into_array();
        let arr = ChunkedArray::try_new(
            vec![c0, c1, c2],
            PrimitiveArray::empty::<i32>(Nullability::NonNullable)
                .dtype()
                .clone(),
        )?;

        // Fully shuffled indices that cross every chunk boundary.
        let indices = buffer![8u64, 0, 5, 3, 2, 7, 1, 6, 4].into_array();
        let result = arr.take(indices)?;

        assert_arrays_eq!(
            result,
            PrimitiveArray::from_iter([8i32, 0, 5, 3, 2, 7, 1, 6, 4]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn test_take_shuffled_large() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let nchunks: i32 = 100;
        let chunk_len: i32 = 1_000;
        let total = nchunks * chunk_len;

        let chunks: Vec<_> = (0..nchunks)
            .map(|c| {
                let start = c * chunk_len;
                PrimitiveArray::from_iter(start..start + chunk_len).into_array()
            })
            .collect();
        let dtype = chunks[0].dtype().clone();
        let arr = ChunkedArray::try_new(chunks, dtype)?;

        // Fisher-Yates shuffle with a fixed seed for determinism.
        let mut indices: Vec<u64> = (0..u64::try_from(total)?).collect();
        let mut seed: u64 = 0xdeadbeef;
        for i in (1..indices.len()).rev() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (seed >> 33) as usize % (i + 1);
            indices.swap(i, j);
        }

        let indices_arr = PrimitiveArray::new(
            vortex_buffer::Buffer::from(indices.clone()),
            Validity::NonNullable,
        );
        let result = arr.take(indices_arr.into_array())?;

        // Verify every element.
        let result = result.execute::<PrimitiveArray>(&mut ctx)?;
        let result_vals = result.as_slice::<i32>();
        for (pos, &idx) in indices.iter().enumerate() {
            assert_eq!(
                result_vals[pos],
                i32::try_from(idx)?,
                "mismatch at position {pos}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_take_null_indices() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let c0 = buffer![10i32, 20, 30].into_array();
        let c1 = buffer![40i32, 50, 60].into_array();
        let arr = ChunkedArray::try_new(
            vec![c0, c1],
            PrimitiveArray::empty::<i32>(Nullability::NonNullable)
                .dtype()
                .clone(),
        )?;

        // Indices with nulls scattered across chunk boundaries.
        let indices =
            PrimitiveArray::from_option_iter([Some(5u64), None, Some(0), Some(3), None, Some(2)]);
        let result = arr.take(indices.into_array())?;

        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([
                Some(60i32),
                None,
                Some(10),
                Some(40),
                None,
                Some(30)
            ]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn test_take_shuffled_with_duplicates_and_nulls() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let c0 = buffer![10i32, 20, 30].into_array();
        let c1 = buffer![40i32, 50, 60].into_array();
        let c2 = buffer![70i32, 80, 90].into_array();
        let arr = ChunkedArray::try_new(
            vec![c0, c1, c2],
            PrimitiveArray::empty::<i32>(Nullability::NonNullable)
                .dtype()
                .clone(),
        )?;

        // Duplicates within and across chunks, nulls interleaved, order shuffled.
        let indices = PrimitiveArray::from_option_iter([
            Some(8u64),
            Some(2),
            None,
            Some(8),
            Some(0),
            Some(5),
            None,
            Some(2),
            Some(4),
        ]);
        let result = arr.take(indices.into_array())?;

        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([
                Some(90i32),
                Some(30),
                None,
                Some(90),
                Some(10),
                Some(60),
                None,
                Some(30),
                Some(50)
            ]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn test_take_sorted_unique_fast_path() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let c0 = buffer![0i32, 1, 2, 3].into_array();
        let c1 = buffer![4i32, 5, 6, 7].into_array();
        let c2 = buffer![8i32, 9, 10, 11].into_array();
        let arr = ChunkedArray::try_new(
            vec![c0, c1, c2],
            PrimitiveArray::empty::<i32>(Nullability::NonNullable)
                .dtype()
                .clone(),
        )?;

        // Strictly increasing indices spanning some (not all) chunks.
        let indices = buffer![1u64, 3, 4, 10, 11].into_array();
        let result = arr.take(indices)?;

        assert_arrays_eq!(
            result,
            PrimitiveArray::from_iter([1i32, 3, 4, 10, 11]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn test_take_sparse_over_large_chunks() -> VortexResult<()> {
        // Few indices over large chunks exercises the sparse per-bucket sort path.
        let mut ctx = array_session().create_execution_ctx();
        let chunk_len = 10_000i32;
        let chunks: Vec<_> = (0..3)
            .map(|c| {
                let start = c * chunk_len;
                PrimitiveArray::from_iter(start..start + chunk_len).into_array()
            })
            .collect();
        let dtype = chunks[0].dtype().clone();
        let arr = ChunkedArray::try_new(chunks, dtype)?;

        let indices = buffer![29_999u64, 5, 29_999, 10_001, 5].into_array();
        let result = arr.take(indices)?;

        assert_arrays_eq!(
            result,
            PrimitiveArray::from_iter([29_999i32, 5, 29_999, 10_001, 5]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn test_take_non_uniform_chunks() -> VortexResult<()> {
        // Uneven chunk lengths force the binary-search chunk resolution path.
        let mut ctx = array_session().create_execution_ctx();
        let c0 = buffer![0i32, 1].into_array();
        let c1 = buffer![2i32, 3, 4, 5, 6].into_array();
        let c2 = buffer![7i32].into_array();
        let arr = ChunkedArray::try_new(
            vec![c0, c1, c2],
            PrimitiveArray::empty::<i32>(Nullability::NonNullable)
                .dtype()
                .clone(),
        )?;

        let indices = buffer![7u64, 0, 3, 7, 1, 6].into_array();
        let result = arr.take(indices)?;

        assert_arrays_eq!(
            result,
            PrimitiveArray::from_iter([7i32, 0, 3, 7, 1, 6]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn test_take_chunked_conformance() {
        let a = buffer![1i32, 2, 3].into_array();
        let b = buffer![4i32, 5].into_array();
        let arr = ChunkedArray::try_new(
            vec![a, b],
            PrimitiveArray::empty::<i32>(Nullability::NonNullable)
                .dtype()
                .clone(),
        )
        .unwrap();
        test_take_conformance(
            &arr.into_array(),
            &mut array_session().create_execution_ctx(),
        );

        // Test with nullable chunked array
        let a = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]);
        let b = PrimitiveArray::from_option_iter([Some(4i32), Some(5)]);
        let dtype = a.dtype().clone();
        let arr = ChunkedArray::try_new(vec![a.into_array(), b.into_array()], dtype).unwrap();
        test_take_conformance(
            &arr.into_array(),
            &mut array_session().create_execution_ctx(),
        );

        // Test with multiple identical chunks
        let chunk = buffer![10i32, 20, 30, 40, 50].into_array();
        let arr = ChunkedArray::try_new(
            vec![chunk.clone(), chunk.clone(), chunk.clone()],
            chunk.dtype().clone(),
        )
        .unwrap();
        test_take_conformance(
            &arr.into_array(),
            &mut array_session().create_execution_ctx(),
        );
    }
}
