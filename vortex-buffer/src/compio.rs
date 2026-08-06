// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::MaybeUninit;

use compio_buf::IoBuf;
use compio_buf::IoBufMut;
use compio_buf::SetLen;

use crate::ByteBuffer;
use crate::ByteBufferMut;

impl IoBuf for ByteBuffer {
    fn as_init(&self) -> &[u8] {
        self.as_slice()
    }
}

impl IoBuf for ByteBufferMut {
    fn as_init(&self) -> &[u8] {
        self.as_slice()
    }
}

impl IoBufMut for ByteBufferMut {
    fn as_uninit(&mut self) -> &mut [MaybeUninit<u8>] {
        let ptr = self.bytes.as_mut_ptr().cast::<MaybeUninit<u8>>();
        let capacity = self.capacity();

        // SAFETY: `BytesMut` guarantees its pointer is valid for its capacity. `BufferMut<u8>` has
        // the same byte capacity, and `MaybeUninit<u8>` permits both its initialized prefix and its
        // uninitialized spare capacity.
        unsafe { std::slice::from_raw_parts_mut(ptr, capacity) }
    }
}

impl SetLen for ByteBufferMut {
    unsafe fn set_len(&mut self, len: usize) {
        // SAFETY: `SetLen` has the same requirements as `BufferMut::set_len`: `len` is within the
        // allocation and the newly exposed byte range has been initialized by the I/O operation.
        unsafe { ByteBufferMut::set_len(self, len) }
    }
}

#[cfg(test)]
mod tests {
    use compio_buf::IoBuf;
    use compio_buf::IoBufMut;
    use compio_buf::SetLen;

    use crate::Alignment;
    use crate::ByteBufferMut;

    #[test]
    fn byte_buffer_mut_supports_completion_io() {
        let alignment = Alignment::new(4096);
        let mut buffer = ByteBufferMut::with_capacity_aligned(4, alignment);

        assert!(buffer.as_init().is_empty());
        assert!(buffer.buf_capacity() >= 4);

        buffer.as_uninit()[..4].copy_from_slice(&[
            std::mem::MaybeUninit::new(1),
            std::mem::MaybeUninit::new(2),
            std::mem::MaybeUninit::new(3),
            std::mem::MaybeUninit::new(4),
        ]);
        // SAFETY: the first four bytes were initialized immediately above.
        unsafe { SetLen::set_len(&mut buffer, 4) };

        assert_eq!(buffer.as_init(), &[1, 2, 3, 4]);
        assert!(alignment.is_ptr_aligned(buffer.as_init().as_ptr()));

        let sealed = buffer.freeze();
        assert_eq!(sealed.as_init(), &[1, 2, 3, 4]);
        assert_eq!(sealed.alignment(), alignment);
    }
}
