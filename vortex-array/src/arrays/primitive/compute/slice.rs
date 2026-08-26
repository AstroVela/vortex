// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_buffer::Alignment;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::slice::SliceReduce;

impl SliceReduce for Primitive {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let byte_width = array.ptype().byte_width();
        let byte_range = range.start * byte_width..range.end * byte_width;
        let values = array
            .buffer_handle()
            .slice_with_alignment(byte_range, Alignment::new(byte_width))?;
        let validity = array.validity()?.slice(range)?;

        // SAFETY:
        // slicing an existing PrimitiveArray on element boundaries preserves the buffer
        // alignment, ptype, length, and validity invariants.
        let array = unsafe {
            PrimitiveArray::new_unchecked_from_handle(values, array.ptype(), validity).into_array()
        };

        Ok(Some(array))
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::Alignment;
    use vortex_buffer::Buffer;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::primitive::PrimitiveArrayExt;
    use crate::validity::Validity;

    #[test]
    fn slice_over_aligned_f32_buffer_at_f32_aligned_offset() -> vortex_error::VortexResult<()> {
        let values: Vec<f32> = (0..4096).map(|value| value as f32).collect();
        let buffer = Buffer::copy_from_aligned(values, Alignment::DEFAULT_ALIGNMENT);
        let array = PrimitiveArray::new(buffer, Validity::NonNullable).into_array();

        let sliced = array.slice(3127..3130)?;
        let mut ctx = array_session().create_execution_ctx();
        let sliced = sliced.execute::<PrimitiveArray>(&mut ctx)?;

        assert_eq!(sliced.buffer_handle().alignment(), Alignment::of::<f32>());
        assert_eq!(sliced.as_slice::<f32>(), &[3127.0, 3128.0, 3129.0]);

        Ok(())
    }
}
