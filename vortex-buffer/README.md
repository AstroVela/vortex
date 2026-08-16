# Vortex Buffer

A Vortex buffer is a window into a reference-counted region of memory, in the same model as the
Tokio `bytes` crate: cloning is a refcount bump and slicing is pointer arithmetic. It used to be a
thin wrapper around `bytes` itself; the region is now managed directly through `std::alloc`
(see https://github.com/tokio-rs/bytes/issues/437), which buys three things `bytes` cannot give us:

- **Custom alignment.** A region remembers the `Layout` it was allocated with, so a buffer that
  needs 256-byte alignment is allocated that way rather than over-allocated and offset into.
  `BufferMut<T>` maintains its alignment across every operation that reallocates.
- **Mutable foreign buffers.** A region records whether it may be written through, so memory
  adopted from a `Vec<T>`, an Arrow buffer, a writable memory map, or an FFI allocation can be
  turned back into a `BufferMut<T>` without a copy whenever the buffer is its only handle.
  `bytes::Bytes::try_into_mut` only ever succeeds for bytes that came out of `BytesMut::freeze`.
- **`Vec<T>` round-trips.** A region allocated with exactly `Layout::array::<T>(cap)` is
  indistinguishable from a `Vec<T>`'s allocation, so `Buffer::into_vec` can hand it straight back
  out.

`bytes::Bytes` remains a zero-copy conversion in both directions, via `Buffer::into_bytes` and
`impl From<bytes::Bytes> for ByteBuffer`.

The regions themselves live in [`vortex-bytes`](../vortex-bytes), which owns memory and nothing
else. This crate adds the typed layer on top: the element type, the element count, and the
alignment a buffer declares. Keeping the two apart means all of the allocation `unsafe` is
non-generic - it compiles once rather than once per element type, and it can be audited and
Miri-tested as a closed unit.
