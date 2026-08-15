// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_buffer::ArrowNativeType;
use arrow_buffer::MutableBuffer;
use arrow_buffer::OffsetBuffer;
use vortex_error::vortex_panic;

use crate::Alignment;
use crate::Buffer;
use crate::ByteBuffer;
use crate::ByteBufferMut;

impl<T: ArrowNativeType> Buffer<T> {
    /// Converts the buffer zero-copy into a `arrow_buffer::Buffer`.
    pub fn into_arrow_scalar_buffer(self) -> arrow_buffer::ScalarBuffer<T> {
        let buffer = arrow_buffer::Buffer::from(self.into_bytes());
        arrow_buffer::ScalarBuffer::from(buffer)
    }

    /// Convert an Arrow scalar buffer into a Vortex scalar buffer.
    ///
    /// ## Panics
    ///
    /// Panics if the Arrow buffer is not aligned to the requested alignment, or if the requested
    /// alignment is not sufficient for type T.
    pub fn from_arrow_scalar_buffer(arrow: arrow_buffer::ScalarBuffer<T>) -> Self {
        let buffer = ByteBuffer::from_arrow_buffer(arrow.into_inner(), Alignment::of::<T>());
        Self::from_byte_buffer(buffer)
    }

    /// Converts the buffer zero-copy into a `arrow_buffer::OffsetBuffer`.
    ///
    /// SAFETY: The caller should ensure that the buffer contains monotonically increasing values
    /// greater than or equal to zero.
    pub fn into_arrow_offset_buffer(self) -> OffsetBuffer<T> {
        unsafe { OffsetBuffer::new_unchecked(self.into_arrow_scalar_buffer()) }
    }
}

impl ByteBuffer {
    /// Converts the buffer zero-copy into a `arrow_buffer::Buffer`.
    pub fn into_arrow_buffer(self) -> arrow_buffer::Buffer {
        arrow_buffer::Buffer::from(self.into_bytes())
    }

    /// Convert an Arrow buffer into a Vortex byte buffer, without copying.
    ///
    /// When the Arrow buffer is the sole owner of its allocation, the resulting Vortex buffer
    /// keeps it mutable: `try_into_mut` will then succeed rather than copying.
    ///
    /// ## Panics
    ///
    /// Panics if the Arrow buffer is not sufficiently aligned.
    pub fn from_arrow_buffer(arrow: arrow_buffer::Buffer, alignment: Alignment) -> Self {
        let buffer = match arrow.into_mutable() {
            Ok(mutable) => ByteBufferMut::from_owner(MutableBufferOwner(mutable)).freeze(),
            Err(arrow) => ByteBuffer::from_owner(ArrowOwner(arrow)),
        };

        if !alignment.is_ptr_aligned(buffer.as_ptr()) {
            vortex_panic!(
                "Arrow buffer is not aligned to the requested alignment: {}",
                alignment
            );
        }
        buffer.ensure_aligned(alignment)
    }
}

/// A wrapper giving `arrow_buffer::Buffer` the `AsRef<[u8]>` impl that buffer adoption needs.
struct ArrowOwner(arrow_buffer::Buffer);

impl AsRef<[u8]> for ArrowOwner {
    fn as_ref(&self) -> &[u8] {
        self.0.as_slice()
    }
}

/// A wrapper giving `arrow_buffer::MutableBuffer` an `AsMut<[u8]>` impl.
struct MutableBufferOwner(MutableBuffer);

impl AsMut<[u8]> for MutableBufferOwner {
    fn as_mut(&mut self) -> &mut [u8] {
        self.0.as_slice_mut()
    }
}

#[cfg(test)]
mod tests {
    use arrow_buffer::Buffer as ArrowBuffer;
    use arrow_buffer::ScalarBuffer;

    use crate::Alignment;
    use crate::Buffer;
    use crate::ByteBuffer;
    use crate::buffer;

    #[test]
    fn into_arrow_buffer() {
        let buf = buffer![0u8, 1, 2];
        let arrow: ArrowBuffer = buf.clone().into_arrow_buffer();
        assert_eq!(arrow.as_ref(), buf.as_slice(), "Buffer values differ");
        assert_eq!(arrow.as_ptr(), buf.as_ptr(), "Conversion not zero-copy")
    }

    #[test]
    fn into_arrow_scalar_buffer() {
        let buf = buffer![0i32, 1, 2];
        let scalar: ScalarBuffer<i32> = buf.clone().into_arrow_scalar_buffer();
        assert_eq!(scalar.as_ref(), buf.as_slice(), "Buffer values differ");
        assert_eq!(scalar.as_ptr(), buf.as_ptr(), "Conversion not zero-copy")
    }

    #[test]
    fn from_arrow_buffer() {
        let arrow = ArrowBuffer::from_vec(vec![0i32, 1, 2]);
        let buf = Buffer::from_arrow_buffer(arrow.clone(), Alignment::of::<i32>());
        assert_eq!(arrow.as_ref(), buf.as_slice(), "Buffer values differ");
        assert_eq!(arrow.as_ptr(), buf.as_ptr(), "Conversion not zero-copy");
    }

    #[test]
    fn sole_owner_arrow_buffer_stays_mutable() {
        let arrow = ArrowBuffer::from_vec(vec![0u8, 1, 2, 3]);
        let ptr = arrow.as_ptr();
        let buf = ByteBuffer::from_arrow_buffer(arrow, Alignment::of::<u8>());
        assert_eq!(buf.as_ptr(), ptr);

        let mut buf = buf
            .try_into_mut()
            .expect("sole owner of the Arrow allocation");
        buf[0] = 10;
        assert_eq!(buf.as_slice(), &[10, 1, 2, 3]);
        assert_eq!(buf.as_ptr(), ptr, "still no copy");
    }
}
