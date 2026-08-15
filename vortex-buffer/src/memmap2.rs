// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use memmap2::Mmap;
use memmap2::MmapMut;

use crate::ByteBuffer;
use crate::ByteBufferMut;

impl From<Mmap> for ByteBuffer {
    fn from(value: Mmap) -> Self {
        ByteBuffer::from_owner(value)
    }
}

impl From<MmapMut> for ByteBufferMut {
    fn from(value: MmapMut) -> Self {
        // A writable mapping is exclusively ours, so the buffer can write straight into it.
        ByteBufferMut::from_owner(value)
    }
}
