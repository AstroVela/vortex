// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::num::NonZeroU64;
use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

/// Canonical list parts whose sublists are kept within one output chunk.
pub(super) struct ListChunk {
    pub(super) elements: ArrayRef,
    pub(super) offsets: ArrayRef,
    pub(super) validity: Validity,
}

/// Groups canonical list parts into target-sized chunks without splitting sublists.
pub(super) struct ListChunker {
    target_element_bytes: u64,
    buffer: ListChunkBuffer,
}

impl ListChunker {
    /// Create a chunker targeting the given number of element bytes per output chunk.
    pub(super) fn new(target_element_bytes: NonZeroU64) -> Self {
        Self {
            target_element_bytes: target_element_bytes.get(),
            buffer: ListChunkBuffer::default(),
        }
    }

    /// Buffer canonical list parts and return any completed list-aware chunks.
    pub(super) fn push_chunk(
        &mut self,
        elements: ArrayRef,
        offsets: &[u64],
        validity: Validity,
    ) -> VortexResult<Vec<ListChunk>> {
        let row_count = offsets.len().saturating_sub(1);
        if row_count == 0 {
            self.buffer.push_empty(elements, validity);
            return Ok(Vec::new());
        }

        let offset_base = offsets[0];
        let element_count = offsets[row_count] - offset_base;
        let element_bytes = estimated_element_bytes(&elements);
        let mut output = Vec::new();
        let mut range_start = 0;
        let mut range_bytes: u64 = 0;

        for row in 0..row_count {
            range_bytes = range_bytes.saturating_add(estimated_range_bytes(
                offsets[row] - offset_base,
                offsets[row + 1] - offset_base,
                element_count,
                element_bytes,
            ));

            if self.buffer.element_bytes.saturating_add(range_bytes) >= self.target_element_bytes {
                self.buffer.push_range(
                    &elements,
                    offsets,
                    &validity,
                    range_start..row + 1,
                    range_bytes,
                )?;
                output.push(self.buffer.take()?);
                range_start = row + 1;
                range_bytes = 0;
            }
        }

        if range_start < row_count {
            self.buffer.push_range(
                &elements,
                offsets,
                &validity,
                range_start..row_count,
                range_bytes,
            )?;
        }

        Ok(output)
    }

    /// Flush any remaining buffered list parts.
    pub(super) fn finish(&mut self) -> VortexResult<Option<ListChunk>> {
        if self.buffer.is_empty() {
            Ok(None)
        } else {
            Ok(Some(self.buffer.take()?))
        }
    }
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

#[derive(Default)]
struct ListChunkBuffer {
    elements: Vec<ArrayRef>,
    offsets: Vec<u64>,
    validities: Vec<(Validity, usize)>,
    element_count: u64,
    element_bytes: u64,
}

impl ListChunkBuffer {
    /// Return whether the buffer contains no list parts.
    fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Append a contiguous range of sublists from one canonical input chunk.
    fn push_range(
        &mut self,
        elements: &ArrayRef,
        offsets: &[u64],
        validity: &Validity,
        rows: Range<usize>,
        element_bytes: u64,
    ) -> VortexResult<()> {
        debug_assert!(rows.start < rows.end);
        let offset_base = offsets[0];
        let first_offset = offsets[rows.start];
        let last_offset = offsets[rows.end];
        let element_start =
            usize::try_from(first_offset - offset_base).vortex_expect("list offset must fit usize");
        let element_end =
            usize::try_from(last_offset - offset_base).vortex_expect("list offset must fit usize");

        self.elements
            .push(elements.slice(element_start..element_end)?);
        if self.offsets.is_empty() {
            self.offsets.push(0);
        }
        for offset in &offsets[rows.start + 1..=rows.end] {
            self.offsets
                .push(self.element_count + (*offset - first_offset));
        }
        self.validities
            .push((validity.slice(rows.clone())?, rows.len()));
        self.element_count += last_offset - first_offset;
        self.element_bytes = self.element_bytes.saturating_add(element_bytes);
        Ok(())
    }

    /// Preserve an empty input chunk until the buffer is flushed.
    fn push_empty(&mut self, elements: ArrayRef, validity: Validity) {
        debug_assert!(elements.is_empty());
        self.elements.push(elements);
        if self.offsets.is_empty() {
            self.offsets.push(0);
        }
        self.validities.push((validity, 0));
    }

    /// Materialize the buffered parts as one chunk and reset the buffer.
    fn take(&mut self) -> VortexResult<ListChunk> {
        let element_chunks = std::mem::take(&mut self.elements);
        let elements = if element_chunks.len() == 1 {
            element_chunks
                .into_iter()
                .next()
                .vortex_expect("one buffered element chunk")
        } else {
            let dtype = element_chunks
                .first()
                .vortex_expect("at least one buffered element chunk")
                .dtype()
                .clone();
            ChunkedArray::try_new(element_chunks, dtype)?.into_array()
        };
        let offsets = PrimitiveArray::from_iter(std::mem::take(&mut self.offsets)).into_array();
        let validity = Validity::concat(std::mem::take(&mut self.validities))
            .vortex_expect("at least one buffered validity");

        self.element_count = 0;
        self.element_bytes = 0;
        Ok(ListChunk {
            elements,
            offsets,
            validity,
        })
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use super::*;

    fn chunker(target_element_bytes: u64) -> ListChunker {
        ListChunker::new(
            NonZeroU64::new(target_element_bytes).vortex_expect("test target is non-zero"),
        )
    }

    #[test]
    fn keeps_sublists_whole_at_chunk_boundaries() -> VortexResult<()> {
        let mut chunker = chunker(16);
        let mut chunks = chunker.push_chunk(
            buffer![0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9].into_array(),
            &[0, 2, 4, 9, 10],
            Validity::NonNullable,
        )?;
        chunks.extend(chunker.finish()?);

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].elements.len(), 4);
        assert_eq!(chunks[1].elements.len(), 5);
        assert_eq!(chunks[2].elements.len(), 1);
        Ok(())
    }

    #[test]
    fn coalesces_sublists_across_input_chunks() -> VortexResult<()> {
        let mut chunker = chunker(16);
        let mut chunks = chunker.push_chunk(
            buffer![0i32, 1].into_array(),
            &[0, 2],
            Validity::NonNullable,
        )?;
        chunks.extend(chunker.push_chunk(
            buffer![2i32, 3, 4].into_array(),
            &[0, 2, 3],
            Validity::NonNullable,
        )?);
        chunks.extend(chunker.finish()?);

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].elements.len(), 4);
        assert_eq!(chunks[0].offsets.len(), 3);
        assert_eq!(chunks[1].elements.len(), 1);
        Ok(())
    }
}
