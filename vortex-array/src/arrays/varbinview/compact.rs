// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Defines a compaction operation for VarBinViewArrays that evicts unused buffers so they can
//! be dropped.

use std::ops::Range;
use std::sync::Arc;

use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ExecutionCtx;
use crate::arrays::VarBinViewArray;
use crate::arrays::varbinview::BinaryView;
use crate::arrays::varbinview::Ref;
use crate::arrays::varbinview::array::for_each_validity_run;
use crate::buffer::BufferHandle;
use crate::builders::VarBinViewBuilder;

const DEFAULT_COMPACTION_THRESHOLD: f64 = 0.5;
const MIN_RETAINED_BYTES_PER_ROW_TO_CHECK_COMPACTION: u64 = 128;

impl VarBinViewArray {
    /// Returns a compacted copy of the input array: the views of null slots are emptied, and then
    /// wasted buffer space is evicted.
    ///
    /// The two steps are ordered, not independent. Compaction decides which bytes of a buffer are
    /// live from the views of the non-null slots alone, so it is free to
    /// slice a buffer down, drop it, or renumber it out from under a view left behind in a null
    /// slot: emptying afterwards would mean compacting an array whose null views may be dangling,
    /// and remapping such a view indexes the buffer lookup by its (arbitrary) buffer index.
    /// Emptying first removes those views before compaction ever looks at them, and costs
    /// compaction nothing, since its measurements ignore null slots either way.
    pub fn compact(&self, ctx: &mut ExecutionCtx) -> VortexResult<VarBinViewArray> {
        self.empty_null_views(ctx)?.compact_buffers(ctx)
    }

    /// Returns a copy of the input array with the view of every null slot replaced by an empty
    /// view.
    ///
    /// Producers are free to leave anything in the view of a null slot, and Vortex itself does:
    /// masking an array keeps the views of the rows it nulls out. Those views are never read as
    /// values, but they are still followed by anything that walks the views buffer, they keep the
    /// bytes of the rows they point at alive, and they are written to disk as they are.
    ///
    /// The array is returned untouched when every null slot already holds an empty view, so the
    /// common case neither copies the views nor touches the data buffers.
    pub fn empty_null_views(&self, ctx: &mut ExecutionCtx) -> VortexResult<VarBinViewArray> {
        // Device views are left to the caller: they cannot be inspected without a copy back.
        if self.views_handle().is_on_device() {
            return Ok(self.clone());
        }

        let validity = self.as_ref().validity()?;
        let mask = validity.execute_mask(self.len(), ctx)?;
        if mask.all_true() {
            return Ok(self.clone());
        }

        let views = self.views();
        let empty = BinaryView::empty_view();

        // Find the first view to empty, walking a run of equal validity at a time so that the scan
        // itself never branches on validity.
        let mut first_dirty = None;
        for_each_validity_run(&mask, 0, |range, valid| {
            if !valid && first_dirty.is_none() {
                first_dirty = views[range.clone()]
                    .iter()
                    .position(|view| *view != empty)
                    .map(|idx| range.start + idx);
            }
            Ok(())
        })?;
        let Some(first_dirty) = first_dirty else {
            return Ok(self.clone());
        };

        // Non-null runs are copied over as they are, null runs are filled with empty views.
        let mut emptied = BufferMut::with_capacity(views.len());
        emptied.extend_from_slice(&views[..first_dirty]);
        for_each_validity_run(&mask, first_dirty, |range, valid| {
            if valid {
                emptied.extend_from_slice(&views[range]);
            } else {
                emptied.push_n(empty, range.len());
            }
            Ok(())
        })?;

        // SAFETY: only the views of null slots changed, and an empty view is always valid.
        Ok(unsafe {
            VarBinViewArray::new_handle_unchecked(
                BufferHandle::new_host(emptied.freeze().into_byte_buffer()),
                Arc::clone(self.data_buffers()),
                self.dtype().clone(),
                validity,
            )
        })
    }

    /// Returns a compacted copy of the input array, where all wasted space has been cleaned up. This
    /// operation can be very expensive, in the worst case copying all existing string data into
    /// a new allocation.
    ///
    /// After slicing/taking operations `VarBinViewArray`s can continue to hold references to buffers
    /// that are no longer visible. We detect when there is wasted space in any of the buffers, and if
    /// so, will aggressively compact all visible outlined string data into new buffers while keeping
    /// well-utilized buffers unchanged.
    pub fn compact_buffers(&self, ctx: &mut ExecutionCtx) -> VortexResult<VarBinViewArray> {
        // If there is nothing to be gained by compaction, return the original array untouched.
        if !self.should_compact(ctx)? {
            return Ok(self.clone());
        }

        self.compact_with_threshold(DEFAULT_COMPACTION_THRESHOLD, ctx)
    }

    fn should_compact(&self, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
        let nbuffers = self.data_buffers().len();

        // If the array is entirely inlined strings, do not attempt to compact.
        if nbuffers == 0 {
            return Ok(false);
        }

        // These will fail to write, so in most cases we want to compact this.
        if nbuffers > u16::MAX as usize {
            return Ok(true);
        }

        let buffer_total_bytes: u64 = self.buffers.iter().map(|buf| buf.len() as u64).sum();
        if buffer_total_bytes == 0 {
            return Ok(true);
        }

        let len = u64::try_from(self.len()).unwrap_or(u64::MAX);
        if len > 0 && buffer_total_bytes / len <= MIN_RETAINED_BYTES_PER_ROW_TO_CHECK_COMPACTION {
            return Ok(false);
        }

        let bytes_referenced: u64 = self.count_referenced_bytes(ctx)?;
        Ok((bytes_referenced as f64 / buffer_total_bytes as f64) < DEFAULT_COMPACTION_THRESHOLD)
    }

    /// Iterates over all valid, non-inlined views, calling the provided
    /// closure for each one.
    #[inline(always)]
    fn iter_valid_views<F>(&self, ctx: &mut ExecutionCtx, mut f: F) -> VortexResult<()>
    where
        F: FnMut(&Ref),
    {
        match self
            .as_ref()
            .validity()?
            .execute_mask(self.as_ref().len(), ctx)?
        {
            Mask::AllTrue(_) => {
                for &view in self.views().iter() {
                    if !view.is_inlined() {
                        f(view.as_view());
                    }
                }
            }
            Mask::AllFalse(_) => {}
            Mask::Values(v) => {
                for (&view, is_valid) in self.views().iter().zip(v.bit_buffer().iter()) {
                    if is_valid && !view.is_inlined() {
                        f(view.as_view());
                    }
                }
            }
        }
        Ok(())
    }

    /// Count the number of bytes addressed by the views, not including null
    /// values or any inlined strings.
    fn count_referenced_bytes(&self, ctx: &mut ExecutionCtx) -> VortexResult<u64> {
        let mut total = 0u64;
        self.iter_valid_views(ctx, |view| total += view.size as u64)?;
        Ok(total)
    }

    pub(crate) fn buffer_utilizations(
        &self,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Vec<BufferUtilization>> {
        let mut utilizations: Vec<BufferUtilization> = self
            .data_buffers()
            .iter()
            .map(|buf| {
                let len = u32::try_from(buf.len()).vortex_expect("buffer sizes must fit in u32");
                BufferUtilization::zero(len)
            })
            .collect();

        self.iter_valid_views(ctx, |view| {
            utilizations[view.buffer_index as usize].add(view.offset, view.size);
        })?;

        Ok(utilizations)
    }

    /// Returns a compacted copy of the input array using selective buffer compaction.
    ///
    /// This method analyzes each buffer's utilization and applies one of three strategies:
    /// - **KeepFull** (zero-copy): Well-utilized buffers are kept unchanged
    /// - **Slice** (zero-copy): Buffers with contiguous ranges of used data are sliced to that range
    /// - **Rewrite**: Poorly-utilized buffers have their data copied to new compact buffers
    ///
    /// By preserving or slicing well-utilized buffers, compaction becomes zero-copy in many cases.
    ///
    /// # Arguments
    ///
    /// * `buffer_utilization_threshold` - Threshold in range [0, 1]. Buffers with utilization
    ///   below this value will be compacted. Use 0.0 for no compaction, 1.0 for aggressive
    ///   compaction of any buffer with wasted space.
    pub fn compact_with_threshold(
        &self,
        buffer_utilization_threshold: f64, // [0, 1]
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<VarBinViewArray> {
        let mut builder = VarBinViewBuilder::with_compaction(
            self.dtype().clone(),
            self.len(),
            buffer_utilization_threshold,
        );
        builder.append_varbinview_array(self, ctx)?;
        Ok(builder.finish_into_varbinview())
    }
}

pub(crate) struct BufferUtilization {
    len: u32,
    used: u32,
    min_offset: u32,
    max_offset_end: u32,
}

impl BufferUtilization {
    pub(crate) fn zero(len: u32) -> Self {
        BufferUtilization {
            len,
            used: 0u32,
            min_offset: u32::MAX,
            max_offset_end: 0,
        }
    }

    pub(crate) fn add(&mut self, offset: u32, size: u32) {
        self.used += size;
        self.min_offset = self.min_offset.min(offset);
        self.max_offset_end = self.max_offset_end.max(offset + size);
    }

    pub fn overall_utilization(&self) -> f64 {
        match self.len {
            0 => 0.0,
            len => self.used as f64 / len as f64,
        }
    }

    pub fn range_utilization(&self) -> f64 {
        match self.range_span() {
            0 => 0.0,
            span => self.used as f64 / span as f64,
        }
    }

    pub fn range(&self) -> Range<u32> {
        self.min_offset..self.max_offset_end
    }

    fn range_span(&self) -> u32 {
        self.max_offset_end.saturating_sub(self.min_offset)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_buffer::Buffer;
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::VarBinArray;
    use crate::arrays::VarBinViewArray;
    use crate::arrays::varbinview::BinaryView;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::validity::Validity;

    /// An array of three rows whose middle row is null, `null_view` being the view left in it.
    /// The two non-null rows reference the last 20 bytes of a 1KiB buffer, which is wasteful
    /// enough for compaction to slice the buffer down to just that range.
    fn array_with_null_view(null_view: BinaryView) -> VarBinViewArray {
        let data = ByteBuffer::copy_from(vec![b'a'; 1024]);
        let valid_view = BinaryView::new_ref(20, *b"aaaa", 0, 900);
        let views = Buffer::copy_from(vec![valid_view, null_view, valid_view]);

        // SAFETY: the view of the null slot is deliberately left as the caller wants it, mimicking
        // an array written by another producer. It is never read as a value.
        unsafe {
            VarBinViewArray::new_unchecked(
                views,
                Arc::from([data]),
                DType::Utf8(Nullability::Nullable),
                Validity::from_iter([true, false, true]),
            )
        }
    }

    /// Every shape of leftover view must come out empty. Note that emptying has to happen before
    /// compaction, not after: compacting this array on its own panics remapping the null view,
    /// whose buffer index is out of range for the buffers compaction keeps.
    #[rstest]
    #[case::out_of_range_buffer(BinaryView::new_ref(13, *b"AAAA", 7, 0))]
    #[case::leftover_reference(BinaryView::new_ref(20, *b"aaaa", 0, 0))]
    #[case::leftover_inlined(BinaryView::new_inlined(b"leftover"))]
    fn compact_empties_null_views(#[case] null_view: BinaryView) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let array = array_with_null_view(null_view);

        let compacted = array.compact(&mut ctx)?;

        assert_eq!(compacted.views()[1], BinaryView::empty_view());
        let compacted_bytes: usize = compacted.data_buffers().iter().map(|buf| buf.len()).sum();
        assert!(compacted_bytes < 1024, "buffers were not compacted");
        assert_arrays_eq!(
            compacted,
            VarBinViewArray::from_iter_nullable_str([
                Some("aaaaaaaaaaaaaaaaaaaa"),
                None,
                Some("aaaaaaaaaaaaaaaaaaaa"),
            ]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn empty_null_views_leaves_clean_arrays_untouched() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let array = array_with_null_view(BinaryView::empty_view());

        let emptied = array.empty_null_views(&mut ctx)?;

        assert_eq!(
            emptied.views().as_ptr(),
            array.views().as_ptr(),
            "clean views must not be copied"
        );
        Ok(())
    }

    #[test]
    fn test_optimize_compacts_buffers() {
        let mut ctx = array_session().create_execution_ctx();
        // Create a VarBinViewArray with some long strings that will create multiple buffers
        let original = VarBinViewArray::from_iter_nullable_str([
            Some("short"),
            Some("this is a longer string that will be stored in a buffer"),
            Some("medium length string"),
            Some("another very long string that definitely needs a buffer to store it"),
            Some("tiny"),
        ]);

        // Verify it has buffers
        assert!(!original.data_buffers().is_empty());
        let original_buffers = original.data_buffers().len();

        // Take only the first and last elements (indices 0 and 4)
        let indices = buffer![0u32, 4u32].into_array();
        let taken = original.take(indices).unwrap();
        let taken = taken.execute::<VarBinViewArray>(&mut ctx).unwrap();
        // The taken array should still have the same number of buffers
        assert_eq!(taken.data_buffers().len(), original_buffers);

        // Now optimize the taken array
        let optimized_array = taken.compact_buffers(&mut ctx).unwrap();

        // The optimized array should have compacted buffers
        // Since both remaining strings are short, they should be inlined
        // so we might have 0 buffers, or 1 buffer if any were not inlined
        assert!(optimized_array.data_buffers().len() <= 1);

        // Verify the data is still correct
        assert_arrays_eq!(
            optimized_array,
            <VarBinArray as FromIterator<_>>::from_iter([Some("short"), Some("tiny")]),
            &mut ctx
        );
    }

    #[test]
    fn test_optimize_with_long_strings() {
        let mut ctx = array_session().create_execution_ctx();
        // Create strings that are definitely longer than 12 bytes
        let long_string_1 = "this is definitely a very long string that exceeds the inline limit";
        let long_string_2 = "another extremely long string that also needs external buffer storage";
        let long_string_3 = "yet another long string for testing buffer compaction functionality";

        let original = VarBinViewArray::from_iter_str([
            long_string_1,
            long_string_2,
            long_string_3,
            "short1",
            "short2",
        ]);

        // Take only the first and third long strings (indices 0 and 2)
        let indices = buffer![0u32, 2u32].into_array();
        let taken = original.take(indices).unwrap();
        let taken_array = taken
            .execute::<VarBinViewArray>(&mut array_session().create_execution_ctx())
            .unwrap();

        let optimized_array = taken_array.compact_with_threshold(1.0, &mut ctx).unwrap();

        // The optimized array should have exactly 1 buffer (consolidated)
        assert_eq!(optimized_array.data_buffers().len(), 1);

        // Verify the data is still correct
        assert_arrays_eq!(
            optimized_array,
            VarBinArray::from(vec![long_string_1, long_string_3]),
            &mut ctx
        );
    }

    #[test]
    fn test_optimize_no_buffers() {
        let mut ctx = array_session().create_execution_ctx();
        // Create an array with only short strings (all inlined)
        let original = VarBinViewArray::from_iter_str(["a", "bb", "ccc", "dddd"]);

        // This should have no buffers
        assert_eq!(original.data_buffers().len(), 0);

        // Optimize should return the same array
        let optimized_array = original.compact_buffers(&mut ctx).unwrap();

        assert_eq!(optimized_array.data_buffers().len(), 0);

        assert_arrays_eq!(optimized_array, original, &mut ctx);
    }

    #[test]
    fn test_optimize_single_buffer() {
        let mut ctx = array_session().create_execution_ctx();
        // Create an array that naturally has only one buffer
        let str1 = "this is a long string that goes into a buffer";
        let str2 = "another long string in the same buffer";
        let original = VarBinViewArray::from_iter_str([str1, str2]);

        // Should have 1 compact buffer
        assert_eq!(original.data_buffers().len(), 1);
        assert_eq!(original.buffer(0).len(), str1.len() + str2.len());

        // Optimize should return the same array (no change needed)
        let optimized_array = original.compact_buffers(&mut ctx).unwrap();

        assert_eq!(optimized_array.data_buffers().len(), 1);

        assert_arrays_eq!(optimized_array, original, &mut ctx);
    }

    #[test]
    fn test_selective_compaction_with_threshold_zero() {
        let mut ctx = array_session().create_execution_ctx();
        // threshold=0 should keep all buffers (no compaction)
        let original = VarBinViewArray::from_iter_str([
            "this is a longer string that will be stored in a buffer",
            "another very long string that definitely needs a buffer to store it",
        ]);

        let original_buffers = original.data_buffers().len();
        assert!(original_buffers > 0);

        // Take only first element
        let indices = buffer![0u32].into_array();
        let taken = original.take(indices).unwrap();
        let taken = taken
            .execute::<VarBinViewArray>(&mut array_session().create_execution_ctx())
            .unwrap();
        // Compact with threshold=0 (should not compact)
        let compacted = taken.compact_with_threshold(0.0, &mut ctx).unwrap();

        // Should still have the same number of buffers as the taken array
        assert_eq!(compacted.data_buffers().len(), taken.data_buffers().len());

        // Verify correctness
        assert_arrays_eq!(compacted, taken, &mut ctx);
    }

    #[test]
    fn test_selective_compaction_with_high_threshold() {
        let mut ctx = array_session().create_execution_ctx();
        // threshold=1.0 should compact any buffer with waste
        let original = VarBinViewArray::from_iter_str([
            "this is a longer string that will be stored in a buffer",
            "another very long string that definitely needs a buffer to store it",
            "yet another long string",
        ]);

        // Take only first and last elements
        let indices = buffer![0u32, 2u32].into_array();
        let taken = original.take(indices).unwrap();
        let taken = taken
            .execute::<VarBinViewArray>(&mut array_session().create_execution_ctx())
            .unwrap();

        let original_buffers = taken.data_buffers().len();

        // Compact with threshold=1.0 (aggressive compaction)
        let compacted = taken.compact_with_threshold(1.0, &mut ctx).unwrap();

        // Should have compacted buffers
        assert!(compacted.data_buffers().len() <= original_buffers);

        // Verify correctness
        assert_arrays_eq!(compacted, taken, &mut ctx);
    }

    #[test]
    fn test_selective_compaction_preserves_well_utilized_buffers() {
        let mut ctx = array_session().create_execution_ctx();
        // Create an array with multiple strings in one buffer (well-utilized)
        let str1 = "first long string that needs external buffer storage";
        let str2 = "second long string also in buffer";
        let str3 = "third long string in same buffer";

        let original = VarBinViewArray::from_iter_str([str1, str2, str3]);

        // All strings should be in one well-utilized buffer
        assert_eq!(original.data_buffers().len(), 1);

        // Compact with high threshold
        let compacted = original.compact_with_threshold(0.8, &mut ctx).unwrap();

        // Well-utilized buffer should be preserved
        assert_eq!(compacted.data_buffers().len(), 1);

        // Verify all data is correct
        assert_arrays_eq!(compacted, original, &mut ctx);
    }

    #[test]
    fn test_selective_compaction_with_mixed_utilization() {
        let mut ctx = array_session().create_execution_ctx();
        // Create array with some long strings
        let strings: Vec<String> = (0..10)
            .map(|i| {
                format!(
                    "this is a long string number {} that needs buffer storage",
                    i
                )
            })
            .collect();

        let original = VarBinViewArray::from_iter_str(strings.iter().map(|s| s.as_str()));

        // Take every other element to create mixed utilization
        let indices_array = buffer![0u32, 2u32, 4u32, 6u32, 8u32].into_array();
        let taken = original.take(indices_array).unwrap();
        let taken = taken
            .execute::<VarBinViewArray>(&mut array_session().create_execution_ctx())
            .unwrap();

        // Compact with moderate threshold
        let compacted = taken.compact_with_threshold(0.7, &mut ctx).unwrap();

        let expected = VarBinViewArray::from_iter(
            [0, 2, 4, 6, 8].map(|i| Some(strings[i].as_str())),
            DType::Utf8(Nullability::NonNullable),
        );
        assert_arrays_eq!(expected, compacted, &mut ctx);
    }

    #[test]
    fn test_slice_strategy_with_contiguous_range() {
        let mut ctx = array_session().create_execution_ctx();
        // Create array with strings that will be in one buffer
        let strings: Vec<String> = (0..20)
            .map(|i| format!("this is a long string number {} for slice test", i))
            .collect();

        let original = VarBinViewArray::from_iter_str(strings.iter().map(|s| s.as_str()));

        // Take only the first 5 elements - they should be in a contiguous range at the start
        let indices_array = buffer![0u32, 1u32, 2u32, 3u32, 4u32].into_array();
        let taken = original.take(indices_array).unwrap();
        let taken = taken
            .execute::<VarBinViewArray>(&mut array_session().create_execution_ctx())
            .unwrap();
        // Get buffer stats before compaction
        let utils_before = taken.buffer_utilizations(&mut ctx).unwrap();
        let original_buffer_count = taken.data_buffers().len();

        // Compact with a threshold that should trigger slicing
        // The range utilization should be high even if overall utilization is low
        let compacted = taken.compact_with_threshold(0.8, &mut ctx).unwrap();

        // After compaction, we should still have buffers (sliced, not rewritten)
        assert!(
            !compacted.data_buffers().is_empty(),
            "Should have buffers after slice compaction"
        );

        // Verify correctness
        assert_arrays_eq!(&compacted, taken, &mut ctx);

        // Verify that if there was only one buffer, the compacted version also has one
        // (it was sliced, not rewritten into multiple buffers)
        if original_buffer_count == 1 && utils_before[0].range_utilization() >= 0.8 {
            assert_eq!(
                compacted.data_buffers().len(),
                1,
                "Slice strategy should maintain single buffer"
            );
        }
    }

    const LONG1: &str = "long string one!";
    const LONG2: &str = "long string two!";
    const SHORT: &str = "x";
    const EXPECTED_BYTES: u64 = (LONG1.len() + LONG2.len()) as u64;

    fn mixed_array() -> VarBinViewArray {
        VarBinViewArray::from_iter_nullable_str([Some(LONG1), None, Some(LONG2), Some(SHORT)])
    }

    #[rstest]
    #[case::non_nullable(VarBinViewArray::from_iter_str([LONG1, LONG2, SHORT]), EXPECTED_BYTES, &[1.0])]
    #[case::all_valid(VarBinViewArray::from_iter_nullable_str([Some(LONG1), Some(LONG2), Some(SHORT)]), EXPECTED_BYTES, &[1.0])]
    #[case::all_invalid(VarBinViewArray::from_iter_nullable_str([None::<&str>, None]), 0, &[])]
    #[case::mixed_validity(mixed_array(), EXPECTED_BYTES, &[1.0])]
    fn test_validity_code_paths(
        #[case] arr: VarBinViewArray,
        #[case] expected_bytes: u64,
        #[case] expected_utils: &[f64],
    ) {
        let mut ctx = array_session().create_execution_ctx();
        assert_eq!(
            arr.count_referenced_bytes(&mut ctx).unwrap(),
            expected_bytes
        );
        let utils: Vec<f64> = arr
            .buffer_utilizations(&mut ctx)
            .unwrap()
            .iter()
            .map(|u| u.overall_utilization())
            .collect();
        assert_eq!(utils, expected_utils);
    }
}
