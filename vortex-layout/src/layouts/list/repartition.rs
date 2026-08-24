// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::num::NonZeroU64;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Chunked;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::chunked::ChunkedArrayExt;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

/// Element arrays and their interior ends, chosen on the enclosing list-offset grid.
pub(super) struct RepartitionedListElements {
    pub(super) arrays: Vec<ArrayRef>,
    pub(super) boundaries: Vec<u64>,
}

/// Repartition list elements at leaf fields, snapping every chunk end to a list boundary.
pub(super) fn repartition_list_elements(
    elements: ArrayRef,
    offsets: &[u64],
    target_element_bytes: NonZeroU64,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<RepartitionedListElements> {
    let elements = elements.execute::<Canonical>(exec_ctx)?.into_array();
    let offset_base = offsets.first().copied().unwrap_or(0);
    let element_end = offsets.last().copied().unwrap_or(offset_base);
    let elements = elements.slice(
        usize::try_from(offset_base).vortex_expect("list offset must fit usize")
            ..usize::try_from(element_end).vortex_expect("list offset must fit usize"),
    )?;

    if elements.dtype().is_struct() {
        let (elements, mut boundaries) =
            chunk_struct_fields(elements, offsets, target_element_bytes.get(), exec_ctx)?;
        let elements = elements.into_array();
        if boundaries.is_empty() {
            boundaries =
                chunk_boundaries_at_list_offsets(&elements, offsets, target_element_bytes.get());
        }
        return Ok(RepartitionedListElements {
            arrays: split_at_boundaries(elements, &boundaries)?,
            boundaries,
        });
    }
    if elements.dtype().is_list() {
        // Retain outer list fences even when a nested list refines the chunks inside them.
        let boundaries =
            chunk_boundaries_at_list_offsets(&elements, offsets, target_element_bytes.get());
        return Ok(RepartitionedListElements {
            arrays: split_at_boundaries(elements, &boundaries)?,
            boundaries,
        });
    }

    let elements = chunk_leaf_field(elements, offsets, target_element_bytes.get(), exec_ctx)?;
    let arrays = if let Some(chunked) = elements.as_opt::<Chunked>() {
        chunked.chunks()
    } else {
        vec![elements]
    };
    let boundaries = arrays
        .iter()
        .map(|array| array.len() as u64)
        .scan(0, |row_end, len| {
            *row_end += len;
            Some(*row_end)
        })
        .take(arrays.len().saturating_sub(1))
        .collect();
    Ok(RepartitionedListElements { arrays, boundaries })
}

/// Recursively descend through structs and chunk each non-struct field independently.
fn chunk_struct_fields(
    array: ArrayRef,
    offsets: &[u64],
    target_element_bytes: u64,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<(StructArray, Vec<u64>)> {
    let struct_array = array.execute::<StructArray>(exec_ctx)?;
    let mut fields = Vec::with_capacity(struct_array.struct_fields().nfields());
    let mut boundaries = Vec::new();
    for field in struct_array.iter_unmasked_fields() {
        let (field, field_boundaries) = if field.dtype().is_struct() {
            let (field, boundaries) =
                chunk_struct_fields(field.clone(), offsets, target_element_bytes, exec_ctx)?;
            (field.into_array(), boundaries)
        } else if field.dtype().is_list() {
            // A nested list has its own finer fences. The enclosing struct gets a fallback fence
            // schedule below if it has no non-list leaves.
            (field.clone(), Vec::new())
        } else {
            let field = chunk_leaf_field(field.clone(), offsets, target_element_bytes, exec_ctx)?;
            let boundaries = chunk_boundaries(&field);
            (field, boundaries)
        };
        fields.push(field);
        boundaries.extend(field_boundaries);
    }
    boundaries.sort_unstable();
    boundaries.dedup();

    Ok((
        StructArray::try_new_with_dtype(
            fields,
            struct_array.struct_fields().clone(),
            struct_array.len(),
            struct_array.validity()?,
        )?,
        boundaries,
    ))
}

fn chunk_boundaries(array: &ArrayRef) -> Vec<u64> {
    let Some(chunked) = array.as_opt::<Chunked>() else {
        return Vec::new();
    };
    chunked
        .iter_chunks()
        .map(|chunk| chunk.len() as u64)
        .scan(0, |row_end, len| {
            *row_end += len;
            Some(*row_end)
        })
        .take(chunked.nchunks().saturating_sub(1))
        .collect()
}

/// Canonicalize one leaf field and split it only at enclosing list boundaries.
fn chunk_leaf_field(
    field: ArrayRef,
    offsets: &[u64],
    target_element_bytes: u64,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let field = field.execute::<Canonical>(exec_ctx)?.into_array();
    let row_count = offsets.len().saturating_sub(1);
    if row_count == 0 {
        return Ok(field);
    }

    let boundaries = chunk_boundaries_at_list_offsets(&field, offsets, target_element_bytes);
    let chunks = split_at_boundaries(field.clone(), &boundaries)?;
    if chunks.len() == 1 {
        Ok(chunks.into_iter().next().vortex_expect("one leaf chunk"))
    } else {
        ChunkedArray::try_new(chunks, field.dtype().clone()).map(IntoArray::into_array)
    }
}

/// Choose interior element ends at enclosing list offsets using an array's proportional byte
/// contribution. The caller turns these ends into physical chunks appropriate for its dtype.
fn chunk_boundaries_at_list_offsets(
    elements: &ArrayRef,
    offsets: &[u64],
    target_element_bytes: u64,
) -> Vec<u64> {
    let row_count = offsets.len().saturating_sub(1);
    if row_count == 0 {
        return Vec::new();
    }

    let offset_base = offsets[0];
    let element_count = offsets[row_count] - offset_base;
    let element_bytes = estimated_element_bytes(elements);
    let mut boundaries = Vec::new();
    let mut range_bytes = 0u64;
    for row in 0..row_count {
        range_bytes = range_bytes.saturating_add(estimated_range_bytes(
            offsets[row] - offset_base,
            offsets[row + 1] - offset_base,
            element_count,
            element_bytes,
        ));
        let boundary = offsets[row + 1] - offset_base;
        if range_bytes >= target_element_bytes && boundary < element_count {
            boundaries.push(boundary);
            range_bytes = 0;
        }
    }
    boundaries
}

fn split_at_boundaries(array: ArrayRef, boundaries: &[u64]) -> VortexResult<Vec<ArrayRef>> {
    let mut chunks = Vec::with_capacity(boundaries.len() + 1);
    let mut start = 0;
    for &end in boundaries {
        let end = usize::try_from(end).vortex_expect("list offset must fit usize");
        chunks.push(array.slice(start..end)?);
        start = end;
    }
    chunks.push(array.slice(start..array.len())?);
    Ok(chunks)
}

/// Estimate the flattened elements' contribution to the repartition target.
fn estimated_element_bytes(elements: &ArrayRef) -> u64 {
    elements
        .dtype()
        .element_size()
        .and_then(|element_size| {
            u64::try_from(element_size)
                .ok()?
                .checked_mul(elements.len() as u64)
        })
        .unwrap_or_else(|| elements.nbytes())
}

/// Estimate a range's bytes by its proportional position in the flattened element array.
/// Taking the difference between two prefix estimates preserves the total byte count exactly.
fn estimated_range_bytes(start: u64, end: u64, element_count: u64, nbytes: u64) -> u64 {
    if element_count == 0 {
        return 0;
    }
    let prefix = |offset: u64| {
        u64::try_from(u128::from(offset) * u128::from(nbytes) / u128::from(element_count))
            .vortex_expect("estimated prefix bytes cannot exceed the element array size")
    };
    prefix(end) - prefix(start)
}

#[cfg(test)]
mod tests {
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::StructArray;
    use vortex_buffer::buffer;

    use super::*;

    fn target(target_element_bytes: u64) -> NonZeroU64 {
        NonZeroU64::new(target_element_bytes).vortex_expect("test target is non-zero")
    }

    #[test]
    fn keeps_sublists_whole_at_chunk_boundaries() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let chunks = repartition_list_elements(
            buffer![0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9].into_array(),
            &[0, 2, 4, 9, 10],
            target(16),
            &mut ctx,
        )?;

        assert_eq!(chunks.arrays.len(), 3);
        assert_eq!(chunks.arrays[0].len(), 4);
        assert_eq!(chunks.arrays[1].len(), 5);
        assert_eq!(chunks.arrays[2].len(), 1);
        assert_eq!(chunk_boundaries_from_arrays(&chunks.arrays), [4, 9]);
        Ok(())
    }

    #[test]
    fn chunks_struct_fields_independently() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let elements = StructArray::from_fields(
            [
                (
                    "wide",
                    buffer![0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9].into_array(),
                ),
                (
                    "narrow",
                    buffer![0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9].into_array(),
                ),
            ]
            .as_slice(),
        )?
        .into_array();

        let chunks = repartition_list_elements(elements, &[0, 2, 4, 9, 10], target(16), &mut ctx)?;

        assert_eq!(chunks.boundaries, [4, 9]);
        assert_eq!(chunk_boundaries_from_arrays(&chunks.arrays), [4, 9]);
        Ok(())
    }

    fn chunk_boundaries_from_arrays(arrays: &[ArrayRef]) -> Vec<u64> {
        arrays
            .iter()
            .map(|array| array.len() as u64)
            .scan(0, |row_end, len| {
                *row_end += len;
                Some(*row_end)
            })
            .take(arrays.len().saturating_sub(1))
            .collect()
    }
}
