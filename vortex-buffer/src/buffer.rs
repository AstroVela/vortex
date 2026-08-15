// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::type_name;
use std::cmp::Ordering;
use std::collections::Bound;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::marker::PhantomData;
use std::ops::Deref;
use std::ops::RangeBounds;

use bytes::Buf;
use bytes::Bytes;
use vortex_error::VortexExpect;
use vortex_error::vortex_panic;

use crate::Alignment;
use crate::BufferMut;
use crate::ByteBuffer;
use crate::alloc::SharedBytes;
use crate::debug::TruncatedDebug;
use crate::trusted_len::TrustedLen;

/// An immutable buffer of items of `T`.
///
/// Elements are treated as plain data: the buffer never runs `T`'s destructor.
#[derive(Clone)]
pub struct Buffer<T> {
    pub(crate) bytes: SharedBytes,
    pub(crate) length: usize,
    pub(crate) alignment: Alignment,
    pub(crate) _marker: PhantomData<T>,
}

impl<T> Default for Buffer<T> {
    fn default() -> Self {
        Self {
            bytes: SharedBytes::empty(),
            length: 0,
            alignment: Alignment::of::<T>(),
            _marker: PhantomData,
        }
    }
}

impl<T> PartialEq for Buffer<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.bytes == other.bytes
    }
}

impl<T> Eq for Buffer<T> {}

impl<T> Ord for Buffer<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.bytes.cmp(&other.bytes)
    }
}

impl<T> PartialOrd for Buffer<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Hash for Buffer<T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.bytes.as_slice().hash(state)
    }
}

impl<T> Buffer<T> {
    /// Returns a new `Buffer<T>` copied from the provided `Vec<T>`, `&[T]`, etc.
    ///
    /// Prefer [`from_vec`](Self::from_vec) or [`from_owner`](Self::from_owner) when the caller
    /// owns the data, since both are zero-copy.
    pub fn copy_from(values: impl AsRef<[T]>) -> Self {
        BufferMut::copy_from(values).freeze()
    }

    /// Returns a new `Buffer<T>` copied from the provided slice and with the requested alignment.
    ///
    /// The allocation is over-aligned to [`Alignment::DEFAULT_ALIGNMENT`] when that is larger than
    /// `alignment`. Use [`copy_from_preferred_aligned`] to control the over-alignment.
    ///
    /// [`copy_from_preferred_aligned`]: Self::copy_from_preferred_aligned
    pub fn copy_from_aligned(values: impl AsRef<[T]>, alignment: Alignment) -> Self {
        Self::copy_from_preferred_aligned(values, alignment, Some(Alignment::DEFAULT_ALIGNMENT))
    }

    /// Returns a new `Buffer<T>` copied from the provided slice and with the requested alignment.
    ///
    /// The buffer reports `alignment`, but the underlying allocation is over-aligned to the larger
    /// of `alignment` and `preferred_alignment`.
    pub fn copy_from_preferred_aligned(
        values: impl AsRef<[T]>,
        alignment: Alignment,
        preferred_alignment: Option<Alignment>,
    ) -> Self {
        BufferMut::copy_from_preferred_aligned(values, alignment, preferred_alignment).freeze()
    }

    /// Take zero-copy ownership of a `Vec<T>`.
    ///
    /// See [`BufferMut::from_vec`]. The resulting buffer is the sole owner of the allocation, so
    /// [`try_into_mut`](Self::try_into_mut) and [`try_into_vec`](Self::try_into_vec) both succeed
    /// on it until it is cloned or sliced.
    ///
    /// ## Example
    ///
    /// ```
    /// use vortex_buffer::Buffer;
    ///
    /// let buffer = Buffer::from_vec(vec![1i32, 2, 3]);
    /// let mut buffer = buffer.try_into_mut().expect("sole owner");
    /// buffer[0] = 10;
    /// assert_eq!(buffer.into_vec(), vec![10, 2, 3]);
    /// ```
    pub fn from_vec(vec: Vec<T>) -> Self {
        BufferMut::from_vec(vec).freeze()
    }

    /// Take zero-copy ownership of memory kept alive by `owner`.
    ///
    /// The buffer's contents are whatever `owner` currently references, and `owner` is dropped
    /// once the last handle to the buffer goes away. This is how foreign allocations - an Arrow
    /// buffer, a memory map, a slab handed over the FFI boundary - enter Vortex without a copy.
    ///
    /// The memory is treated as read-only: [`try_into_mut`](Self::try_into_mut) will copy rather
    /// than write through a pointer we only ever had shared access to. Use
    /// [`BufferMut::from_owner`] when the owner can hand over exclusive, writable access.
    ///
    /// ## Panics
    ///
    /// Panics if the owner's memory is not aligned to `align_of::<T>()`.
    ///
    /// ## Example
    ///
    /// ```
    /// use std::sync::Arc;
    /// use vortex_buffer::Buffer;
    ///
    /// let shared: Arc<[i32]> = Arc::from(vec![1, 2, 3]);
    /// let buffer = Buffer::from_owner(shared.clone());
    /// assert_eq!(buffer.as_slice(), &[1, 2, 3]);
    /// assert_eq!(buffer.as_ptr(), shared.as_ptr(), "no copy was made");
    /// ```
    pub fn from_owner<O>(owner: O) -> Self
    where
        O: AsRef<[T]> + Send + 'static,
    {
        let bytes = SharedBytes::from_owner::<O, T>(owner);
        let alignment = Alignment::of::<T>();
        if !alignment.is_ptr_aligned(bytes.as_ptr()) {
            vortex_panic!("Foreign buffer is not aligned to {alignment}");
        }
        let length = bytes.len() / size_of::<T>();
        Self {
            bytes,
            length,
            alignment,
            _marker: PhantomData,
        }
    }

    /// Borrow a `'static` slice without copying it.
    ///
    /// ## Panics
    ///
    /// Panics if the slice is not aligned to `align_of::<T>()`, which cannot happen for a slice
    /// obtained safely.
    pub fn from_static(values: &'static [T]) -> Self {
        // SAFETY: any `[T]` is a valid `[u8]` of `size_of_val` bytes for the purposes of reading.
        let bytes = unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), size_of_val(values))
        };
        let bytes = SharedBytes::from_static(bytes);
        let alignment = Alignment::of::<T>();
        if !alignment.is_ptr_aligned(bytes.as_ptr()) {
            vortex_panic!("Static buffer is not aligned to {alignment}");
        }
        Self {
            bytes,
            length: values.len(),
            alignment,
            _marker: PhantomData,
        }
    }

    /// Create a new zeroed `Buffer` with the given value.
    pub fn zeroed(len: usize) -> Self {
        Self::zeroed_aligned(len, Alignment::of::<T>())
    }

    /// Create a new zeroed `Buffer` with the requested alignment.
    ///
    /// The allocation is over-aligned to [`Alignment::DEFAULT_ALIGNMENT`] when that is larger than
    /// `alignment`. Use [`zeroed_preferred_aligned`] to control the over-alignment.
    ///
    /// [`zeroed_preferred_aligned`]: Self::zeroed_preferred_aligned
    pub fn zeroed_aligned(len: usize, alignment: Alignment) -> Self {
        Self::zeroed_preferred_aligned(len, alignment, Some(Alignment::DEFAULT_ALIGNMENT))
    }

    /// Create a new zeroed `Buffer` with the requested alignment.
    ///
    /// The buffer reports `alignment`, but the underlying allocation is over-aligned to the larger
    /// of `alignment` and `preferred_alignment`.
    pub fn zeroed_preferred_aligned(
        len: usize,
        alignment: Alignment,
        preferred_alignment: Option<Alignment>,
    ) -> Self {
        BufferMut::zeroed_preferred_aligned(len, alignment, preferred_alignment).freeze()
    }

    /// Create a new empty `ByteBuffer` with the provided alignment.
    pub fn empty() -> Self {
        Self::empty_aligned(Alignment::of::<T>())
    }

    /// Create a new empty `ByteBuffer` with the provided alignment.
    ///
    /// This does not allocate: empty buffers point at a dangling address that is aligned to
    /// [`Alignment::MAX`].
    pub fn empty_aligned(alignment: Alignment) -> Self {
        if !alignment.is_aligned_to(Alignment::of::<T>()) {
            vortex_panic!(
                "Alignment {} must align to the scalar type's alignment {}",
                alignment,
                Alignment::of::<T>(),
            );
        }
        Self {
            bytes: SharedBytes::empty(),
            length: 0,
            alignment,
            _marker: PhantomData,
        }
    }

    /// Create a new full `ByteBuffer` with the given value.
    pub fn full(item: T, len: usize) -> Self
    where
        T: Copy,
    {
        BufferMut::full(item, len).freeze()
    }

    /// Create a `Buffer<T>` zero-copy from a `ByteBuffer`.
    ///
    /// ## Panics
    ///
    /// Panics if the buffer is not aligned to the size of `T`, or the length is not a multiple of
    /// the size of `T`.
    pub fn from_byte_buffer(buffer: ByteBuffer) -> Self {
        // TODO(ngates): should this preserve the current alignment of the buffer?
        Self::from_byte_buffer_aligned(buffer, Alignment::of::<T>())
    }

    /// Create a `Buffer<T>` zero-copy from a `ByteBuffer`.
    ///
    /// ## Panics
    ///
    /// Panics if the buffer is not aligned to the given alignment, if the length is not a multiple
    /// of the size of `T`, or if the given alignment is not aligned to that of `T`.
    pub fn from_byte_buffer_aligned(buffer: ByteBuffer, alignment: Alignment) -> Self {
        Self::from_shared_aligned(buffer.bytes, alignment)
    }

    /// Create a `Buffer<T>` zero-copy from a `Bytes`.
    ///
    /// ## Panics
    ///
    /// Panics if the buffer is not aligned to the size of `T`, or the length is not a multiple of
    /// the size of `T`.
    pub fn from_bytes_aligned(bytes: Bytes, alignment: Alignment) -> Self {
        Self::from_shared_aligned(ByteBuffer::from(bytes).bytes, alignment)
    }

    fn from_shared_aligned(bytes: SharedBytes, alignment: Alignment) -> Self {
        if !alignment.is_aligned_to(Alignment::of::<T>()) {
            vortex_panic!(
                "Alignment {} must be compatible with the scalar type's alignment {}",
                alignment,
                Alignment::of::<T>(),
            );
        }
        if !alignment.is_ptr_aligned(bytes.as_ptr()) {
            vortex_panic!(
                "Bytes alignment must align to the requested alignment {}",
                alignment,
            );
        }
        if !bytes.len().is_multiple_of(size_of::<T>()) {
            vortex_panic!(
                "Bytes length {} must be a multiple of the scalar type's size {}",
                bytes.len(),
                size_of::<T>()
            );
        }
        let length = bytes.len() / size_of::<T>();
        Self {
            bytes,
            length,
            alignment,
            _marker: Default::default(),
        }
    }

    /// Create a buffer with values from the TrustedLen iterator.
    /// Should be preferred over `from_iter` when the iterator is known to be `TrustedLen`.
    pub fn from_trusted_len_iter<I: TrustedLen<Item = T>>(iter: I) -> Self {
        let (_, upper_bound) = iter.size_hint();
        let mut buffer = BufferMut::with_capacity(
            upper_bound.vortex_expect("TrustedLen iterator has no upper bound"),
        );
        buffer.extend_trusted(iter);
        buffer.freeze()
    }

    /// Map each element of the buffer with a closure, reusing the allocation where possible.
    ///
    /// ## Panics
    ///
    /// Panics if `R` does not have the same size and alignment as `T`. See
    /// [`BufferMut::map_each_in_place`].
    pub fn map_each_in_place<R, F>(self, mut f: F) -> BufferMut<R>
    where
        T: Copy,
        R: Copy,
        F: FnMut(T) -> R,
    {
        match self.try_into_mut() {
            Ok(mut_buf) => mut_buf.map_each_in_place(f),
            Err(buf) => {
                let len = buf.len();
                let mut out_buf = BufferMut::with_capacity(len);
                out_buf
                    .spare_capacity_mut()
                    .iter_mut()
                    .zip(buf)
                    .for_each(|(out, in_)| {
                        out.write(f(in_));
                    });
                // Safety: just assigned to each value
                unsafe { out_buf.set_len(len) }
                out_buf
            }
        }
    }

    /// Clear the buffer, preserving existing capacity.
    pub fn clear(&mut self) {
        self.bytes.clear();
        self.length = 0;
    }

    /// Returns the length of the buffer in elements of type T.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.length
    }

    /// Returns whether the buffer is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Returns the alignment of the buffer.
    #[inline(always)]
    pub fn alignment(&self) -> Alignment {
        self.alignment
    }

    /// Returns a slice over the buffer of elements of type T.
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: alignment of Buffer is checked on construction
        unsafe { std::slice::from_raw_parts(self.bytes.as_ptr().cast(), self.length) }
    }

    /// Return a view over the buffer as an opaque byte slice.
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8] {
        self.bytes.as_slice()
    }

    /// Returns an iterator over the buffer of elements of type T.
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            inner: self.as_slice().iter(),
        }
    }

    /// Returns a slice of self for the provided range.
    ///
    /// # Panics
    ///
    /// Requires that `begin <= end` and `end <= self.len()`.
    /// Also requires that both `begin` and `end` are aligned to the buffer's required alignment.
    #[inline(always)]
    pub fn slice(&self, range: impl RangeBounds<usize>) -> Self {
        self.slice_with_alignment(range, self.alignment)
    }

    /// Returns a slice of self for the provided range, with no guarantees about the resulting
    /// alignment.
    ///
    /// # Panics
    ///
    /// Requires that `begin <= end` and `end <= self.len()`.
    #[inline(always)]
    pub fn slice_unaligned(&self, range: impl RangeBounds<usize>) -> Self {
        self.slice_with_alignment(range, Alignment::of::<u8>())
    }

    /// Returns a slice of self for the provided range, ensuring the resulting slice has the
    /// given alignment.
    ///
    /// # Panics
    ///
    /// Requires that `begin <= end` and `end <= self.len()`.
    /// Also requires that both `begin` and `end` are aligned to the given alignment.
    pub fn slice_with_alignment(
        &self,
        range: impl RangeBounds<usize>,
        alignment: Alignment,
    ) -> Self {
        let len = self.len();
        let begin = match range.start_bound() {
            Bound::Included(&n) => n,
            Bound::Excluded(&n) => n.checked_add(1).vortex_expect("out of range"),
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(&n) => n.checked_add(1).vortex_expect("out of range"),
            Bound::Excluded(&n) => n,
            Bound::Unbounded => len,
        };

        if begin > end {
            vortex_panic!(
                "range start must not be greater than end: {:?} <= {:?}",
                begin,
                end
            );
        }
        if end > len {
            vortex_panic!("range end out of bounds: {:?} > {:?}", end, len);
        }

        if end == begin {
            // We prefer to return a new empty buffer instead of sharing this one and creating a
            // strong reference just to hold an empty slice.
            return Self::empty_aligned(alignment);
        }

        let begin_byte = begin * size_of::<T>();
        let end_byte = end * size_of::<T>();

        if !alignment.is_offset_aligned(begin_byte) {
            vortex_panic!(
                "range start must be aligned to {alignment:?}, byte {}",
                begin_byte
            );
        }
        if !alignment.is_aligned_to(Alignment::of::<T>()) {
            vortex_panic!("Slice alignment must at least align to type T")
        }

        Self {
            bytes: self.bytes.slice(begin_byte, end_byte),
            length: end - begin,
            alignment,
            _marker: Default::default(),
        }
    }

    /// Returns a slice of self that is equivalent to the given subset.
    ///
    /// When processing the buffer you will often end up with `&[T]` that is a subset
    /// of the underlying buffer. This function turns the slice into a slice of the buffer
    /// it has been taken from.
    ///
    /// # Panics:
    /// Requires that the given sub slice is in fact contained within the buffer; otherwise this function will panic.
    #[inline(always)]
    pub fn slice_ref(&self, subset: &[T]) -> Self {
        self.slice_ref_with_alignment(subset, Alignment::of::<T>())
    }

    /// Returns a slice of self that is equivalent to the given subset.
    ///
    /// When processing the buffer you will often end up with `&[T]` that is a subset
    /// of the underlying buffer. This function turns the slice into a slice of the buffer
    /// it has been taken from.
    ///
    /// # Panics:
    /// Requires that the given sub slice is in fact contained within the buffer; otherwise this function will panic.
    /// Also requires that the given alignment aligns to the type of slice and is smaller or equal to the buffers alignment
    pub fn slice_ref_with_alignment(&self, subset: &[T], alignment: Alignment) -> Self {
        if !alignment.is_aligned_to(Alignment::of::<T>()) {
            vortex_panic!("slice_ref alignment must at least align to type T")
        }

        if !self.alignment.is_aligned_to(alignment) {
            vortex_panic!("slice_ref subset alignment must at least align to the buffer alignment")
        }

        if !alignment.is_ptr_aligned(subset.as_ptr()) {
            vortex_panic!("slice_ref subset must be aligned to {:?}", alignment);
        }

        // SAFETY: any `[T]` is a valid `[u8]` of `size_of_val` bytes for the purposes of reading.
        let subset_u8 =
            unsafe { std::slice::from_raw_parts(subset.as_ptr().cast(), size_of_val(subset)) };

        Self {
            bytes: self.bytes.slice_ref(subset_u8),
            length: subset.len(),
            alignment,
            _marker: Default::default(),
        }
    }

    /// Return the ByteBuffer for this `Buffer<T>`.
    pub fn into_byte_buffer(self) -> ByteBuffer {
        ByteBuffer {
            bytes: self.bytes,
            length: self.length * size_of::<T>(),
            alignment: self.alignment,
            _marker: Default::default(),
        }
    }

    /// Convert this buffer into a `bytes::Bytes`, without copying.
    ///
    /// The `Bytes` keeps this buffer's allocation alive, so it is safe to hand out even for
    /// buffers backed by foreign memory. Converting back with `ByteBuffer::from` is zero-copy
    /// too, but a `Bytes` carries no alignment, so the buffer that comes back reports an
    /// alignment of 1.
    pub fn into_bytes(self) -> Bytes {
        Bytes::from_owner(self.into_byte_buffer())
    }

    /// Try to convert self into `BufferMut<T>` if there is only a single strong reference.
    ///
    /// Unlike `bytes::Bytes`, this succeeds for buffers built over foreign memory - a `Vec<T>`
    /// adopted with [`from_vec`](Self::from_vec), or any owner handed to
    /// [`BufferMut::from_owner`] - as long as nothing else holds a reference to it.
    ///
    /// The recovered capacity runs from the start of this buffer to the end of its allocation, so
    /// a buffer that is a slice of a larger region regains the rest of it. Nothing else can see
    /// those bytes, but note that for adopted foreign memory they are the owner's: writing past
    /// the buffer's length writes into the `Vec`, mapping, or Arrow buffer it came from.
    pub fn try_into_mut(self) -> Result<BufferMut<T>, Self> {
        let length = self.length;
        let alignment = self.alignment;
        self.bytes
            .try_into_unique()
            .map(|bytes| BufferMut {
                bytes,
                length,
                alignment,
                _marker: Default::default(),
            })
            .map_err(|bytes| Self {
                bytes,
                length,
                alignment,
                _marker: Default::default(),
            })
    }

    /// Convert self into `BufferMut<T>`, cloning the data if there are multiple strong references.
    pub fn into_mut(self) -> BufferMut<T> {
        self.try_into_mut()
            .unwrap_or_else(|buffer| BufferMut::<T>::copy_from_aligned(&buffer, buffer.alignment))
    }

    /// Convert the buffer into a `Vec<T>`, without copying where possible.
    ///
    /// See [`BufferMut::into_vec`] for when this is zero-copy; in addition, this buffer must be
    /// the only handle to its allocation.
    pub fn into_vec(self) -> Vec<T>
    where
        T: Copy,
    {
        match self.try_into_mut() {
            Ok(buffer) => buffer.into_vec(),
            Err(buffer) => buffer.as_slice().to_vec(),
        }
    }

    /// Convert the buffer into a `Vec<T>` without copying, or give it back.
    ///
    /// See [`into_vec`](Self::into_vec).
    pub fn try_into_vec(self) -> Result<Vec<T>, Self> {
        self.try_into_mut()?
            .try_into_vec()
            .map_err(BufferMut::freeze)
    }

    /// Returns whether this is the only handle to the buffer's allocation.
    ///
    /// When this is true, [`try_into_mut`](Self::try_into_mut) succeeds for any buffer over
    /// writable memory, and [`try_into_vec`](Self::try_into_vec) succeeds for any buffer that
    /// starts at the front of a `T`-aligned allocation.
    pub fn is_unique(&self) -> bool {
        self.bytes.is_unique()
    }

    /// Returns whether a `Buffer<T>` is aligned to the given alignment.
    pub fn is_aligned(&self, alignment: Alignment) -> bool {
        alignment.is_ptr_aligned(self.bytes.as_ptr())
    }

    /// Return a `Buffer<T>` with the given alignment. Where possible, this will be zero-copy.
    pub fn aligned(mut self, alignment: Alignment) -> Self {
        if alignment.is_ptr_aligned(self.as_ptr()) {
            self.alignment = alignment;
            self
        } else {
            #[cfg(feature = "warn-copy")]
            {
                let bt = std::backtrace::Backtrace::capture();
                tracing::warn!(
                    "Buffer is not aligned to requested alignment {alignment}, copying: {bt}"
                )
            }
            Self::copy_from_aligned(self, alignment)
        }
    }

    /// Return a `Buffer<T>` with the given alignment. Panics if the buffer is not aligned.
    pub fn ensure_aligned(mut self, alignment: Alignment) -> Self {
        if alignment.is_ptr_aligned(self.as_ptr()) {
            self.alignment = alignment;
            self
        } else {
            vortex_panic!("Buffer is not aligned to requested alignment {}", alignment)
        }
    }
}

impl<T> Buffer<T> {
    /// Transmute a `Buffer<T>` into a `Buffer<U>`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that all possible bit representations of type `T` are valid when
    /// interpreted as type `U`.
    /// See [`std::mem::transmute`] for more details.
    ///
    /// # Panics
    ///
    /// Panics if the type `U` does not have the same size and alignment as `T`.
    pub unsafe fn transmute<U>(self) -> Buffer<U> {
        assert_eq!(size_of::<T>(), size_of::<U>(), "Buffer type size mismatch");
        assert_eq!(
            align_of::<T>(),
            align_of::<U>(),
            "Buffer type alignment mismatch"
        );

        Buffer {
            bytes: self.bytes,
            length: self.length,
            alignment: self.alignment,
            _marker: PhantomData,
        }
    }
}

/// An iterator over Buffer elements.
///
/// This is an analog to the `std::slice::Iter` type.
pub struct Iter<'a, T> {
    inner: std::slice::Iter<'a, T>,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.inner.count()
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        self.inner.last()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.inner.nth(n)
    }
}

impl<T> ExactSizeIterator for Iter<'_, T> {
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<T: Debug> Debug for Buffer<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct(&format!("Buffer<{}>", type_name::<T>()))
            .field("length", &self.length)
            .field("alignment", &self.alignment)
            .field("as_slice", &TruncatedDebug(self.as_slice()))
            .finish()
    }
}

impl<T> Deref for Buffer<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> AsRef<[T]> for Buffer<T> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> FromIterator<T> for Buffer<T> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        BufferMut::from_iter(iter).freeze()
    }
}

impl<T> From<Vec<T>> for Buffer<T>
where
    T: Send + 'static,
{
    fn from(value: Vec<T>) -> Self {
        if std::mem::needs_drop::<T>() {
            // The buffer itself never runs `T`'s destructor, so keep the `Vec` alive to do it.
            // This costs us the ability to hand the allocation back out, which is the right
            // trade for a type that is not plain data.
            Self::from_owner(value)
        } else {
            Self::from_vec(value)
        }
    }
}

impl From<Bytes> for ByteBuffer {
    fn from(bytes: Bytes) -> Self {
        // Recover exclusive ownership where we can, so the resulting buffer stays mutable.
        match bytes.try_into_mut() {
            Ok(bytes) => ByteBufferMutOwner(bytes).into_buffer_mut().freeze(),
            Err(bytes) => ByteBuffer::from_owner(bytes),
        }
    }
}

/// Newtype giving `bytes::BytesMut` an [`AsMut`] impl for [`BufferMut::from_owner`].
struct ByteBufferMutOwner(bytes::BytesMut);

impl ByteBufferMutOwner {
    fn into_buffer_mut(self) -> crate::ByteBufferMut {
        crate::ByteBufferMut::from_owner(self)
    }
}

impl AsMut<[u8]> for ByteBufferMutOwner {
    #[inline]
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

impl Buf for ByteBuffer {
    #[inline]
    fn remaining(&self) -> usize {
        self.len()
    }

    #[inline]
    fn chunk(&self) -> &[u8] {
        self.as_slice()
    }

    #[inline]
    fn advance(&mut self, cnt: usize) {
        if !self.alignment.is_offset_aligned(cnt) {
            vortex_panic!(
                "Cannot advance buffer by {} items, resulting alignment is not {}",
                cnt,
                self.alignment
            );
        }
        self.bytes.advance(cnt);
        self.length -= cnt;
    }
}

/// Owned iterator over a [`Buffer`].
pub struct BufferIterator<T: Copy> {
    // Keep the buffer alive for the duration of the iteration.
    _buffer: Buffer<T>,
    ptr: *const T,
    end: *const T,
}

// SAFETY: `BufferIterator` is a `Buffer<T>` plus two cursors into it, so it can safely be
// `Send`/`Sync` exactly when `Buffer<T>` is. Same bounds as `std::vec::IntoIter`.
unsafe impl<T: Copy + Send> Send for BufferIterator<T> {}
unsafe impl<T: Copy + Sync> Sync for BufferIterator<T> {}

impl<T: Copy> Iterator for BufferIterator<T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.ptr == self.end {
            None
        } else {
            // SAFETY: ptr is within the buffer and has not reached end.
            let value = unsafe { self.ptr.read() };
            self.ptr = unsafe { self.ptr.add(1) };
            Some(value)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = unsafe { self.end.offset_from(self.ptr) } as usize;
        (remaining, Some(remaining))
    }
}

impl<T: Copy> ExactSizeIterator for BufferIterator<T> {}

impl<T: Copy> IntoIterator for Buffer<T> {
    type Item = T;
    type IntoIter = BufferIterator<T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        let ptr = self.as_slice().as_ptr();
        let end = unsafe { ptr.add(self.len()) };
        BufferIterator {
            _buffer: self,
            ptr,
            end,
        }
    }
}

impl<T> From<BufferMut<T>> for Buffer<T> {
    #[inline]
    fn from(value: BufferMut<T>) -> Self {
        value.freeze()
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use bytes::Buf;
    use bytes::Bytes;

    use crate::Alignment;
    use crate::Buffer;
    use crate::ByteBuffer;
    use crate::buffer;

    #[test]
    fn align() {
        let buf = buffer![0u8, 1, 2];
        let aligned = buf.aligned(Alignment::new(32));
        assert_eq!(aligned.alignment(), Alignment::new(32));
        assert_eq!(aligned.as_slice(), &[0, 1, 2]);
    }

    /// The storage types are unconditionally `Send`/`Sync` via `unsafe impl`, so it is the
    /// `PhantomData<T>` that keeps a buffer's auto traits tied to its element type. `Cell` is
    /// `Send` but not `Sync`, so a buffer of it must compile as `Send` and, were the bound ever
    /// widened, the `Sync` assertion below would start compiling for it too.
    #[test]
    fn auto_traits_follow_the_element_type() {
        const fn assert_send_sync<T: Send + Sync>() {}
        const fn assert_send<T: Send>() {}

        assert_send_sync::<Buffer<u8>>();
        assert_send_sync::<Buffer<i64>>();
        assert_send_sync::<crate::BufferMut<i64>>();
        assert_send::<Buffer<Cell<u8>>>();
        assert_send::<crate::BufferMut<Cell<u8>>>();
    }

    #[test]
    fn buffer_iterator_send_sync() {
        fn assert_send_sync<T: Send + Sync>(_: &T) {}

        let mut iter = buffer![0i32, 1, 2, 3].into_iter();
        assert_send_sync(&iter);
        iter.next();
        let remaining: Vec<i32> = std::thread::spawn(move || iter.collect()).join().unwrap();
        assert_eq!(remaining, vec![1, 2, 3]);
    }

    #[test]
    fn slice() {
        let buf = buffer![0, 1, 2, 3, 4];
        assert_eq!(buf.slice(1..3).as_slice(), &[1, 2]);
        assert_eq!(buf.slice(1..=3).as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn slice_unaligned() {
        let buf = buffer![0i32, 1, 2, 3, 4].into_byte_buffer();
        // With a regular slice, this would panic. See [`slice_bad_alignment`].
        let sliced = buf.slice_unaligned(1..2);
        // Verify the slice has the expected length (1 byte from index 1 to 2).
        assert_eq!(sliced.len(), 1);
        // The original buffer has i32 values [0, 1, 2, 3, 4].
        // In little-endian bytes, 0i32 = [0, 0, 0, 0], so byte at index 1 is 0.
        assert_eq!(sliced.as_slice(), &[0]);
    }

    #[test]
    #[should_panic]
    fn slice_bad_alignment() {
        let buf = buffer![0i32, 1, 2, 3, 4].into_byte_buffer();
        // We should only be able to slice this buffer on 4-byte (i32) boundaries.
        buf.slice(1..2);
    }

    #[test]
    fn bytes_buf() {
        let mut buf = ByteBuffer::copy_from("helloworld".as_bytes());
        assert_eq!(buf.remaining(), 10);
        assert_eq!(buf.chunk(), b"helloworld");

        buf.advance(5);
        assert_eq!(buf.remaining(), 5);
        assert_eq!(buf.as_slice(), b"world");
        assert_eq!(buf.chunk(), b"world");
    }

    #[test]
    fn buffer_zeroed() {
        const LEN: usize = 17;

        let buf = Buffer::<u32>::zeroed(LEN);

        assert!(buf.is_aligned(Alignment::of::<u32>()));
        assert_eq!(buf.as_slice(), &[0; LEN]);
    }

    #[test]
    fn buffer_zeroed_aligned() {
        const LEN: usize = 17;
        let alignment = Alignment::new(64);

        let buf = Buffer::<u32>::zeroed_aligned(LEN, alignment);

        assert!(buf.is_aligned(alignment));
        assert_eq!(buf.as_slice(), &[0; LEN]);
    }

    #[test]
    fn copy_from_over_aligns_to_default() {
        let values = [1u32, 2, 3];
        let buf = Buffer::<u32>::copy_from(values);

        // The buffer reports the scalar type's alignment, ...
        assert_eq!(buf.alignment(), Alignment::of::<u32>());
        // ... but the underlying allocation is over-aligned to DEFAULT_ALIGNMENT.
        assert!(buf.is_aligned(Alignment::DEFAULT_ALIGNMENT));
        assert_eq!(buf.as_slice(), &values);
    }

    #[test]
    fn zeroed_over_aligns_to_default() {
        const LEN: usize = 17;

        let buf = Buffer::<u32>::zeroed(LEN);

        assert_eq!(buf.alignment(), Alignment::of::<u32>());
        assert!(buf.is_aligned(Alignment::DEFAULT_ALIGNMENT));
        assert_eq!(buf.as_slice(), &[0; LEN]);
    }

    #[test]
    fn from_vec() {
        let vec = vec![1, 2, 3, 4, 5];
        let buff = Buffer::from(vec.clone());
        assert!(buff.is_aligned(Alignment::of::<i32>()));
        assert_eq!(vec, buff.as_ref());
    }

    #[test]
    fn from_vec_is_zero_copy() {
        let vec = vec![1i32, 2, 3, 4, 5];
        let ptr = vec.as_ptr();
        let buf = Buffer::from(vec);
        assert_eq!(buf.as_ptr(), ptr);
    }

    #[test]
    fn vec_round_trip_is_zero_copy() {
        let vec = vec![1i32, 2, 3, 4, 5];
        let ptr = vec.as_ptr();
        let buf = Buffer::from(vec);
        let vec = buf.try_into_vec().expect("sole owner of a Vec allocation");
        assert_eq!(vec.as_ptr(), ptr);
        assert_eq!(vec, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn vec_round_trip_copies_when_shared() {
        let buf = Buffer::from(vec![1i32, 2, 3]);
        let _shared = buf.clone();
        assert!(buf.try_into_vec().is_err());
    }

    #[test]
    fn from_vec_of_droppable_still_drops() {
        // `Buffer<T>` never runs destructors itself, so a `Vec` of droppable elements has to be
        // kept alive wholesale.
        let buf = Buffer::from(vec![String::from("hello"), String::from("world")]);
        assert_eq!(buf.len(), 2);
        assert_eq!(buf[0], "hello");
        // Not writable: we only ever had shared access to the `Vec`'s contents.
        assert!(buf.try_into_mut().is_err());
    }

    #[test]
    fn foreign_buffer_is_mutable_when_unique() {
        let boxed: Box<[i32]> = vec![1i32, 2, 3].into_boxed_slice();
        let ptr = boxed.as_ptr();

        let buf = crate::BufferMut::from_owner(boxed).freeze();
        assert_eq!(buf.as_ptr(), ptr);

        let mut buf = buf.try_into_mut().expect("sole owner of foreign memory");
        buf[0] = 10;
        assert_eq!(buf.as_slice(), &[10, 2, 3]);
        assert_eq!(buf.as_ptr(), ptr, "still no copy");
    }

    #[test]
    fn foreign_buffer_is_immutable_when_shared() {
        let buf = crate::BufferMut::from_owner(vec![1i32, 2, 3].into_boxed_slice()).freeze();
        let _shared = buf.clone();
        assert!(buf.try_into_mut().is_err());
    }

    #[test]
    fn from_owner_is_read_only() {
        let shared: std::sync::Arc<[i32]> = std::sync::Arc::from(vec![1i32, 2, 3]);
        let buf = Buffer::from_owner(std::sync::Arc::clone(&shared));
        assert_eq!(buf.as_ptr(), shared.as_ptr());
        // We only ever had shared access, so this must copy rather than write through it.
        let mut mutable = buf.into_mut();
        mutable[0] = 10;
        assert_eq!(shared.as_ref(), &[1, 2, 3]);
    }

    #[test]
    fn small_buffers_keep_the_default_alignment() {
        // A buffer grown from empty must still land on `DEFAULT_ALIGNMENT`: that is what lets the
        // SIMD and CUDA paths take it without a re-aligning copy.
        let collected: Buffer<i32> = (0i32..4).collect();
        assert!(collected.is_aligned(Alignment::DEFAULT_ALIGNMENT));

        let mut built = crate::BufferMut::<i32>::empty();
        built.extend(0i32..4);
        assert!(built.freeze().is_aligned(Alignment::DEFAULT_ALIGNMENT));
    }

    #[test]
    fn cloned_static_buffer_is_not_unique() {
        // A `'static` buffer carries no refcount, so we cannot claim to be its only handle.
        static VALUES: [i32; 3] = [1, 2, 3];
        let buf = Buffer::from_static(&VALUES);
        let clone = buf.clone();
        assert!(!buf.is_unique());
        drop(clone);
        assert!(!buf.is_unique());
        // Empty buffers own nothing at all, so they are trivially exclusive.
        assert!(Buffer::<i32>::empty().is_unique());
    }

    #[test]
    fn from_static_is_zero_copy() {
        static VALUES: [i32; 3] = [1, 2, 3];
        let buf = Buffer::from_static(&VALUES);
        assert_eq!(buf.as_ptr(), VALUES.as_ptr());
        // Static memory is never writable.
        assert!(buf.try_into_mut().is_err());
    }

    #[test]
    fn bytes_round_trip_is_zero_copy() {
        let buf = Buffer::from(vec![1u8, 2, 3]);
        let ptr = buf.as_ptr();
        let bytes: Bytes = buf.into_bytes();
        assert_eq!(bytes.as_ptr(), ptr);

        let buf = ByteBuffer::from(bytes);
        assert_eq!(buf.as_ptr(), ptr);
    }

    #[test]
    fn empty_aligned_max_alignment() {
        // Empty buffers point at a dangling address that satisfies any valid alignment.
        let buf = Buffer::<u8>::empty_aligned(Alignment::MAX);
        assert!(buf.is_empty());
        assert!(buf.is_aligned(Alignment::MAX));
    }

    #[test]
    fn empty_slice_preserves_alignment() {
        let buf = Buffer::<u64>::zeroed_aligned(8, Alignment::new(64));
        let sliced = buf.slice(0..0);
        assert!(sliced.is_empty());
        assert_eq!(sliced.alignment(), Alignment::new(64));
        assert!(sliced.is_aligned(Alignment::new(64)));
    }

    #[test]
    fn empty_into_mut_preserves_alignment() {
        let buf = Buffer::<u8>::empty_aligned(Alignment::new(64));
        let buf_mut = buf.into_mut();
        assert_eq!(buf_mut.alignment(), Alignment::new(64));
        assert!(buf_mut.is_empty());
    }

    #[test]
    fn test_slice_unaligned_end_pos() {
        let data = vec![0u8; 2];
        // Overalign the u8 vector.
        let aligned_buffer = Buffer::copy_from_aligned(&data, Alignment::new(8));
        // Previously, `Buffer::slice` incorrectly asserted that the end position
        // must be aligned. That assertion has been removed such that the end
        // position can be arbitrary and only the beginning of the slice needs
        // to be aligned.
        aligned_buffer.slice(0..1);
    }

    #[test]
    fn test_empty_equality() {
        let a = Buffer::<u16>::empty();
        let b = Buffer::<u16>::empty();

        assert_eq!(a, b);
    }

    #[test]
    fn try_into_mut_recovers_spare_capacity() {
        let mut buf = crate::BufferMut::<u8>::with_capacity(128);
        buf.extend_from_slice(&[1, 2, 3]);
        let buf = buf.freeze();
        let buf = buf.try_into_mut().expect("sole owner");
        assert!(buf.capacity() >= 128);
        assert_eq!(buf.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn sliced_buffer_is_mutable_when_unique() {
        let buf = Buffer::from(vec![1i32, 2, 3, 4]);
        let sliced = buf.slice(1..3);
        drop(buf);
        let mut sliced = sliced.try_into_mut().expect("sole remaining owner");
        sliced[0] = 20;
        assert_eq!(sliced.as_slice(), &[20, 3]);
    }
}
