// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use core::mem::MaybeUninit;
use std::any::type_name;
use std::cmp::max;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::io::Write;
use std::ops::Deref;
use std::ops::DerefMut;

use bytes::Buf;
use bytes::BufMut;
use bytes::buf::UninitSlice;
use vortex_error::VortexExpect;
use vortex_error::vortex_panic;

use crate::Alignment;
use crate::Buffer;
use crate::ByteBufferMut;
use crate::alloc::UniqueBytes;
use crate::debug::TruncatedDebug;
use crate::trusted_len::TrustedLen;

/// A mutable buffer that maintains a runtime-defined alignment through resizing operations.
///
/// Elements are treated as plain data: the buffer never runs `T`'s destructor, and never will.
pub struct BufferMut<T> {
    pub(crate) bytes: UniqueBytes,
    pub(crate) length: usize,
    pub(crate) alignment: Alignment,
    pub(crate) _marker: std::marker::PhantomData<T>,
}

impl<T> BufferMut<T> {
    /// Create a new `BufferMut` with the requested alignment and capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_aligned(capacity, Alignment::of::<T>())
    }

    /// Create a new `BufferMut` with the requested alignment and capacity.
    ///
    /// The allocation is over-aligned to [`Alignment::DEFAULT_ALIGNMENT`] when that is larger than
    /// `alignment`. Use [`with_capacity_preferred_aligned`] to control the over-alignment.
    ///
    /// [`with_capacity_preferred_aligned`]: Self::with_capacity_preferred_aligned
    pub fn with_capacity_aligned(capacity: usize, alignment: Alignment) -> Self {
        Self::with_capacity_preferred_aligned(
            capacity,
            alignment,
            Some(Alignment::DEFAULT_ALIGNMENT),
        )
    }

    /// Create a new `BufferMut` with the requested alignment and capacity.
    ///
    /// The buffer reports `alignment`, but the underlying allocation is over-aligned to the larger
    /// of `alignment` and `preferred_alignment`.
    pub fn with_capacity_preferred_aligned(
        capacity: usize,
        alignment: Alignment,
        preferred_alignment: Option<Alignment>,
    ) -> Self {
        let actual = Self::check_alignment(alignment, preferred_alignment);
        // Round the first allocation up to the alignment, so that a buffer built by pushing onto
        // an empty one still lands on the preferred alignment. The buffer only carries its own
        // declared `alignment`, so once the region exists its layout is the only record of the
        // preferred one - an empty buffer that allocated nothing would have nothing to grow from.
        let initial = (capacity * size_of::<T>()).max(*actual.min(Alignment::DEFAULT_ALIGNMENT));
        Self {
            bytes: UniqueBytes::with_capacity(initial, actual),
            length: 0,
            alignment,
            _marker: Default::default(),
        }
    }

    /// Validate the requested alignment and return the alignment to allocate with.
    fn check_alignment(alignment: Alignment, preferred_alignment: Option<Alignment>) -> Alignment {
        if !alignment.is_aligned_to(Alignment::of::<T>()) {
            vortex_panic!(
                "Alignment {} must align to the scalar type's alignment {}",
                alignment,
                align_of::<T>()
            );
        }
        max(
            alignment,
            preferred_alignment.unwrap_or(Alignment::of::<u8>()),
        )
    }

    /// Create a new zeroed `BufferMut`.
    pub fn zeroed(len: usize) -> Self {
        Self::zeroed_aligned(len, Alignment::of::<T>())
    }

    /// Create a new zeroed `BufferMut` with the requested alignment.
    ///
    /// The allocation is over-aligned to [`Alignment::DEFAULT_ALIGNMENT`] when that is larger than
    /// `alignment`. Use [`zeroed_preferred_aligned`] to control the over-alignment.
    ///
    /// [`zeroed_preferred_aligned`]: Self::zeroed_preferred_aligned
    pub fn zeroed_aligned(len: usize, alignment: Alignment) -> Self {
        Self::zeroed_preferred_aligned(len, alignment, Some(Alignment::DEFAULT_ALIGNMENT))
    }

    /// Create a new zeroed `BufferMut` with the requested alignment.
    ///
    /// The buffer reports `alignment`, but the underlying allocation is over-aligned to the larger
    /// of `alignment` and `preferred_alignment`.
    pub fn zeroed_preferred_aligned(
        len: usize,
        alignment: Alignment,
        preferred_alignment: Option<Alignment>,
    ) -> Self {
        let actual_alignment = max(
            preferred_alignment.unwrap_or(Alignment::of::<u8>()),
            alignment,
        );
        let bytes = UniqueBytes::zeroed(len * size_of::<T>(), actual_alignment);
        let actual_len = bytes.len().checked_div(size_of::<T>()).unwrap_or(0);
        Self {
            bytes,
            length: actual_len,
            alignment,
            _marker: Default::default(),
        }
    }

    /// Create a new empty `BufferMut` with the provided alignment.
    pub fn empty() -> Self {
        Self::empty_aligned(Alignment::of::<T>())
    }

    /// Create a new empty `BufferMut` with the provided alignment.
    ///
    /// The allocation is over-aligned to [`Alignment::DEFAULT_ALIGNMENT`] when that is larger than
    /// `alignment`. Use [`empty_preferred_aligned`] to control the over-alignment.
    ///
    /// [`empty_preferred_aligned`]: Self::empty_preferred_aligned
    pub fn empty_aligned(alignment: Alignment) -> Self {
        Self::empty_preferred_aligned(alignment, Some(Alignment::DEFAULT_ALIGNMENT))
    }

    /// Create a new empty `BufferMut` with the provided alignment.
    ///
    /// The buffer reports `alignment`, but the underlying allocation is over-aligned to the larger
    /// of `alignment` and `preferred_alignment`.
    pub fn empty_preferred_aligned(
        alignment: Alignment,
        preferred_alignment: Option<Alignment>,
    ) -> Self {
        BufferMut::with_capacity_preferred_aligned(0, alignment, preferred_alignment)
    }

    /// Create a new full `BufferMut` with the given value.
    pub fn full(item: T, len: usize) -> Self
    where
        T: Copy,
    {
        let mut buffer = BufferMut::<T>::with_capacity(len);
        buffer.push_n(item, len);
        buffer
    }

    /// Create a mutable scalar buffer by copying the contents of the slice.
    pub fn copy_from(other: impl AsRef<[T]>) -> Self {
        Self::copy_from_aligned(other, Alignment::of::<T>())
    }

    /// Create a mutable scalar buffer with the alignment by copying the contents of the slice.
    ///
    /// The allocation is over-aligned to [`Alignment::DEFAULT_ALIGNMENT`] when that is larger than
    /// `alignment`. Use [`copy_from_preferred_aligned`] to control the over-alignment.
    ///
    /// [`copy_from_preferred_aligned`]: Self::copy_from_preferred_aligned
    ///
    /// ## Panics
    ///
    /// Panics when the requested alignment isn't itself aligned to type T.
    pub fn copy_from_aligned(other: impl AsRef<[T]>, alignment: Alignment) -> Self {
        Self::copy_from_preferred_aligned(other, alignment, Some(Alignment::DEFAULT_ALIGNMENT))
    }

    /// Create a mutable scalar buffer with the alignment by copying the contents of the slice.
    ///
    /// The buffer reports `alignment`, but the underlying allocation is over-aligned to the larger
    /// of `alignment` and `preferred_alignment`.
    ///
    /// ## Panics
    ///
    /// Panics when the requested alignment isn't itself aligned to type T.
    pub fn copy_from_preferred_aligned(
        other: impl AsRef<[T]>,
        alignment: Alignment,
        preferred_alignment: Option<Alignment>,
    ) -> Self {
        if !alignment.is_aligned_to(Alignment::of::<T>()) {
            vortex_panic!("Given alignment is not aligned to type T")
        }
        let other = other.as_ref();
        let mut buffer =
            Self::with_capacity_preferred_aligned(other.len(), alignment, preferred_alignment);
        buffer.extend_from_slice(other);
        debug_assert_eq!(buffer.alignment(), alignment);
        buffer
    }

    /// Take zero-copy ownership of a `Vec<T>`.
    ///
    /// A `Vec<T>`'s allocation is exactly a `T`-aligned global allocation, so it can be adopted
    /// without copying and, as long as it is not re-aligned or grown, handed back out again with
    /// [`into_vec`](Self::into_vec).
    ///
    /// The resulting buffer reports an alignment of `align_of::<T>()`; use
    /// [`aligned`](Self::aligned) to request more, which may copy.
    ///
    /// The buffer never runs `T`'s destructor, so `T` should be plain data.
    ///
    /// ## Example
    ///
    /// ```
    /// use vortex_buffer::BufferMut;
    ///
    /// let vec = vec![1i32, 2, 3];
    /// let ptr = vec.as_ptr();
    ///
    /// let buffer = BufferMut::from_vec(vec);
    /// assert_eq!(buffer.as_ptr(), ptr, "adoption is zero-copy");
    ///
    /// let vec = buffer.into_vec();
    /// assert_eq!(vec.as_ptr(), ptr, "and so is handing it back");
    /// ```
    pub fn from_vec(vec: Vec<T>) -> Self {
        let length = vec.len();
        Self {
            bytes: UniqueBytes::from_vec(vec),
            length,
            alignment: Alignment::of::<T>(),
            _marker: Default::default(),
        }
    }

    /// Take zero-copy ownership of a foreign, writable allocation.
    ///
    /// The buffer's contents are whatever `owner` currently references, and `owner` is dropped
    /// (releasing the memory) when the last handle to the buffer goes away. Unlike
    /// [`Buffer::from_owner`], the resulting buffer is mutable: taking `owner` by value and
    /// reaching the memory through [`AsMut`] proves that nothing else may be observing it.
    ///
    /// The whole of the owner's memory becomes the buffer's capacity, so writes past the buffer's
    /// length still land in it rather than in a fresh allocation. For a writable memory map that
    /// means `extend_from_slice` writes through to the mapped file.
    ///
    /// The buffer reports an alignment of `align_of::<T>()`, and panics if the owner's memory does
    /// not meet it.
    ///
    /// ## Example
    ///
    /// ```
    /// use vortex_buffer::BufferMut;
    ///
    /// let mut buffer = BufferMut::<i32>::from_owner(vec![1i32, 2, 3].into_boxed_slice());
    /// buffer[0] = 10;
    /// assert_eq!(buffer.as_slice(), &[10, 2, 3]);
    /// ```
    pub fn from_owner<O>(owner: O) -> Self
    where
        O: AsMut<[T]> + Send + 'static,
    {
        let bytes = UniqueBytes::from_owner::<O, T>(owner);
        let alignment = Alignment::of::<T>();
        if !alignment.is_ptr_aligned(bytes.as_ptr()) {
            vortex_panic!("Foreign buffer is not aligned to {alignment}");
        }
        let length = bytes.len() / size_of::<T>();
        Self {
            bytes,
            length,
            alignment,
            _marker: Default::default(),
        }
    }

    /// Get the alignment of the buffer.
    #[inline(always)]
    pub fn alignment(&self) -> Alignment {
        self.alignment
    }

    /// Returns the length of the buffer.
    #[inline(always)]
    pub fn len(&self) -> usize {
        debug_assert_eq!(self.length, self.bytes.len() / size_of::<T>());
        self.length
    }

    /// Returns whether the buffer is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Returns the capacity of the buffer.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.bytes.capacity() / size_of::<T>()
    }

    /// Returns a slice over the buffer of elements of type T.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: alignment of Buffer is checked on construction
        unsafe { std::slice::from_raw_parts(self.bytes.as_ptr().cast(), self.length) }
    }

    /// Returns a slice over the buffer of elements of type T.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        let raw_slice = self.bytes.as_mut_slice();
        // SAFETY: alignment of Buffer is checked on construction
        unsafe { std::slice::from_raw_parts_mut(raw_slice.as_mut_ptr().cast(), self.length) }
    }

    /// Clear the buffer, retaining any existing capacity.
    #[inline]
    pub fn clear(&mut self) {
        // SAFETY: shrinking the buffer cannot expose uninitialized bytes.
        unsafe { self.bytes.set_len(0) }
        self.length = 0;
    }

    /// Shortens the buffer, keeping the first `len` bytes and dropping the
    /// rest.
    ///
    /// If `len` is greater than the buffer's current length, this has no
    /// effect.
    ///
    /// Existing underlying capacity is preserved.
    #[inline]
    pub fn truncate(&mut self, len: usize) {
        if len <= self.len() {
            // SAFETY: Shrinking the buffer cannot expose uninitialized bytes.
            unsafe { self.set_len(len) };
        }
    }

    /// Reserves capacity for at least `additional` more elements to be inserted in the buffer.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.bytes
            .reserve(additional * size_of::<T>(), self.alignment);
    }

    /// Returns the spare capacity of the buffer as a slice of `MaybeUninit<T>`.
    /// Has identical semantics to [`Vec::spare_capacity_mut`].
    ///
    /// The returned slice can be used to fill the buffer with data (e.g. by
    /// reading from a file) before marking the data as initialized using the
    /// [`set_len`] method.
    ///
    /// Note that the returned slice may be larger than the capacity requested at
    /// construction, since the underlying allocation can be rounded up (e.g. to
    /// satisfy alignment requirements).
    ///
    /// [`set_len`]: BufferMut::set_len
    /// [`Vec::spare_capacity_mut`]: Vec::spare_capacity_mut
    ///
    /// # Examples
    ///
    /// ```
    /// use vortex_buffer::BufferMut;
    ///
    /// // Allocate vector big enough for 10 elements.
    /// let mut b = BufferMut::<u64>::with_capacity(10);
    ///
    /// // Fill in the first 3 elements.
    /// let uninit = b.spare_capacity_mut();
    /// uninit[0].write(0);
    /// uninit[1].write(1);
    /// uninit[2].write(2);
    ///
    /// // Mark the first 3 elements of the vector as being initialized.
    /// unsafe {
    ///     b.set_len(3);
    /// }
    ///
    /// assert_eq!(b.as_slice(), &[0u64, 1, 2]);
    /// ```
    #[inline]
    pub fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<T>] {
        let spare = self.capacity() - self.length;
        let dst = self.bytes.spare_capacity_mut().as_mut_ptr();
        // SAFETY: `dst` is the start of `spare` uninitialized elements of `T`, and the buffer's
        // alignment guarantees it is well aligned for `T`.
        unsafe { std::slice::from_raw_parts_mut(dst as *mut MaybeUninit<T>, spare) }
    }

    /// Sets the length of the buffer.
    ///
    /// # Safety
    ///
    /// - `new_len` must be less than or equal to [`capacity()`].
    /// - The elements at `old_len..new_len` must be initialized.
    ///
    /// [`capacity()`]: Self::capacity
    #[inline]
    pub unsafe fn set_len(&mut self, len: usize) {
        debug_assert!(len <= self.capacity());
        // SAFETY: the caller guarantees the elements up to `len` are initialized.
        unsafe { self.bytes.set_len(len * size_of::<T>()) };
        self.length = len;
    }

    /// Appends a scalar to the buffer.
    #[inline]
    pub fn push(&mut self, value: T) {
        self.reserve(1);
        // SAFETY: we just reserved capacity for one more element.
        unsafe { self.push_unchecked(value) }
    }

    /// Appends a scalar to the buffer without checking for sufficient capacity.
    ///
    /// ## Safety
    ///
    /// The caller must ensure there is sufficient capacity in the array.
    #[inline]
    pub unsafe fn push_unchecked(&mut self, item: T) {
        // SAFETY: the caller ensures we have sufficient capacity
        unsafe {
            let dst: *mut T = self.bytes.spare_capacity_mut().as_mut_ptr().cast();
            dst.write(item);
            self.bytes.set_len(self.bytes.len() + size_of::<T>())
        }
        self.length += 1;
    }

    /// Appends n scalars to the buffer.
    ///
    /// This function is slightly more optimized than `extend(iter::repeat_n(item, b))`.
    #[inline]
    pub fn push_n(&mut self, item: T, n: usize)
    where
        T: Copy,
    {
        self.reserve(n);
        // SAFETY: we just reserved capacity for `n` more elements.
        unsafe { self.push_n_unchecked(item, n) }
    }

    /// Appends n scalars to the buffer.
    ///
    /// ## Safety
    ///
    /// The caller must ensure there is sufficient capacity in the array.
    #[inline]
    pub unsafe fn push_n_unchecked(&mut self, item: T, n: usize)
    where
        T: Copy,
    {
        let mut dst: *mut T = self.bytes.spare_capacity_mut().as_mut_ptr().cast();
        // SAFETY: we checked the capacity in the reserve call
        unsafe {
            let end = dst.add(n);
            while dst < end {
                dst.write(item);
                dst = dst.add(1);
            }
            self.bytes.set_len(self.bytes.len() + (n * size_of::<T>()));
        }
        self.length += n;
    }

    /// Appends a slice of type `T`, growing the internal buffer as needed.
    ///
    /// # Example:
    ///
    /// ```
    /// # use vortex_buffer::BufferMut;
    ///
    /// let mut builder = BufferMut::<u16>::with_capacity(10);
    /// builder.extend_from_slice(&[42, 44, 46]);
    ///
    /// assert_eq!(builder.len(), 3);
    /// ```
    #[inline]
    pub fn extend_from_slice(&mut self, slice: &[T]) {
        // SAFETY: any `[T]` is a valid `[u8]` of `size_of_val` bytes for the purposes of copying.
        let raw_slice =
            unsafe { std::slice::from_raw_parts(slice.as_ptr().cast(), size_of_val(slice)) };
        self.bytes.extend_from_slice(raw_slice, self.alignment);
        self.length += slice.len();
    }

    /// Splits the buffer into two at the given index.
    ///
    /// Afterward, self contains elements `[0, at)`, and the returned buffer contains elements
    /// `[at, capacity)`. It’s guaranteed that the memory does not move, that is, the address of
    /// self does not change, and the address of the returned slice is at bytes after that.
    ///
    /// This is an O(1) operation that just increases the reference count and sets a few indices.
    ///
    /// Panics if either half would have a length that is not a multiple of the alignment.
    pub fn split_off(&mut self, at: usize) -> Self {
        if at > self.capacity() {
            vortex_panic!("Cannot split buffer of capacity {} at {}", self.len(), at);
        }

        let bytes_at = at * size_of::<T>();
        if !self.alignment.is_offset_aligned(bytes_at) {
            vortex_panic!(
                "Cannot split buffer at {}, resulting alignment is not {}",
                at,
                self.alignment
            );
        }

        let new_bytes = self.bytes.split_off(bytes_at);

        // Adjust the lengths, given that length may be < at
        let new_length = self.length.saturating_sub(at);
        self.length = self.length.min(at);

        BufferMut {
            bytes: new_bytes,
            length: new_length,
            alignment: self.alignment,
            _marker: Default::default(),
        }
    }

    /// Absorbs a mutable buffer that was previously split off.
    ///
    /// If the two buffers were previously contiguous and not mutated in a way that causes
    /// re-allocation i.e., if other was created by calling split_off on this buffer, then this is
    /// an O(1) operation that just decreases a reference count and sets a few indices.
    ///
    /// Otherwise, this method degenerates to self.extend_from_slice(other.as_ref()).
    pub fn unsplit(&mut self, other: Self) {
        if self.alignment != other.alignment {
            vortex_panic!(
                "Cannot unsplit buffers with different alignments: {} and {}",
                self.alignment,
                other.alignment
            );
        }
        let alignment = self.alignment;
        self.bytes.unsplit(other.bytes, alignment);
        self.length = self.bytes.len() / size_of::<T>();
    }

    /// Return the [`ByteBufferMut`] for this [`BufferMut`].
    pub fn into_byte_buffer(self) -> ByteBufferMut {
        ByteBufferMut {
            bytes: self.bytes,
            length: self.length * size_of::<T>(),
            alignment: self.alignment,
            _marker: Default::default(),
        }
    }

    /// Freeze the `BufferMut` into a `Buffer`.
    pub fn freeze(self) -> Buffer<T> {
        Buffer {
            bytes: self.bytes.freeze(),
            length: self.length,
            alignment: self.alignment,
            _marker: Default::default(),
        }
    }

    /// Convert the buffer into a `Vec<T>`, without copying where possible.
    ///
    /// This is zero-copy when the buffer owns a `T`-aligned allocation that starts at the front of
    /// its window, which is the case for buffers built by [`from_vec`](Self::from_vec) and for
    /// buffers allocated with exactly `align_of::<T>()`. An over-aligned buffer cannot be given
    /// away, because `Vec` would free it with the wrong layout, so this copies instead.
    ///
    /// Use [`try_into_vec`](Self::try_into_vec) when a copy is not acceptable.
    pub fn into_vec(self) -> Vec<T>
    where
        T: Copy,
    {
        self.try_into_vec()
            .unwrap_or_else(|buffer| buffer.as_slice().to_vec())
    }

    /// Convert the buffer into a `Vec<T>` without copying, or give it back.
    ///
    /// See [`into_vec`](Self::into_vec) for when this succeeds.
    pub fn try_into_vec(self) -> Result<Vec<T>, Self> {
        let length = self.length;
        let alignment = self.alignment;
        self.bytes.try_into_vec::<T>().map_err(|bytes| Self {
            bytes,
            length,
            alignment,
            _marker: Default::default(),
        })
    }

    /// Map each element of the buffer with a closure.
    pub fn map_each_in_place<R, F>(self, mut f: F) -> BufferMut<R>
    where
        T: Copy,
        F: FnMut(T) -> R,
    {
        assert_eq!(
            size_of::<T>(),
            size_of::<R>(),
            "Size of T and R do not match"
        );
        // SAFETY: we have checked that `size_of::<T>` == `size_of::<R>`.
        let mut buf: BufferMut<R> = unsafe { std::mem::transmute(self) };
        buf.iter_mut()
            .for_each(|item| *item = f(unsafe { std::mem::transmute_copy(item) }));
        buf
    }

    /// Return a `BufferMut<T>` with the same data as this one with the given alignment.
    ///
    /// If the data is already properly aligned, this is a metadata-only operation.
    ///
    /// If the data is not aligned, we copy it into a new allocation.
    pub fn aligned(self, alignment: Alignment) -> Self {
        if alignment.is_ptr_aligned(self.as_ptr()) {
            Self {
                bytes: self.bytes,
                length: self.length,
                alignment,
                _marker: std::marker::PhantomData,
            }
        } else {
            Self::copy_from_aligned(self, alignment)
        }
    }

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
    pub unsafe fn transmute<U>(self) -> BufferMut<U> {
        assert_eq!(size_of::<T>(), size_of::<U>(), "Buffer type size mismatch");
        assert_eq!(
            align_of::<T>(),
            align_of::<U>(),
            "Buffer type alignment mismatch"
        );

        BufferMut {
            bytes: self.bytes,
            length: self.length,
            alignment: self.alignment,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T> Clone for BufferMut<T> {
    fn clone(&self) -> Self {
        // NOTE(ngates): we cannot derive Clone since the buffer owns its allocation exclusively,
        //  and the alignment must be preserved by the copy.
        let mut buffer = BufferMut::<T>::with_capacity_aligned(self.capacity(), self.alignment);
        buffer.extend_from_slice(self.as_slice());
        buffer
    }
}

impl<T> PartialEq for BufferMut<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.bytes == other.bytes
            && self.length == other.length
            && self.alignment == other.alignment
    }
}

impl<T> Eq for BufferMut<T> {}

impl<T> From<Vec<T>> for BufferMut<T> {
    #[inline]
    fn from(value: Vec<T>) -> Self {
        Self::from_vec(value)
    }
}

impl<T: Debug> Debug for BufferMut<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct(&format!("BufferMut<{}>", type_name::<T>()))
            .field("length", &self.length)
            .field("alignment", &self.alignment)
            .field("as_slice", &TruncatedDebug(self.as_slice()))
            .finish()
    }
}

impl<T> Default for BufferMut<T> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<T> Deref for BufferMut<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> DerefMut for BufferMut<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T> AsRef<[T]> for BufferMut<T> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> AsMut<[T]> for BufferMut<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T> BufferMut<T> {
    /// A helper method for the two [`Extend`] implementations.
    ///
    /// We use the lower bound hint on the iterator to manually write data, and then we continue to
    /// push items normally past the lower bound.
    fn extend_iter(&mut self, mut iter: impl Iterator<Item = T>) {
        // Since we do not know the length of the iterator, we can only guess how much memory we
        // need to reserve. Note that these hints may be inaccurate.
        let (lower_bound, _) = iter.size_hint();

        // We choose not to use the optional upper bound size hint to match the standard library.

        self.reserve(lower_bound);

        let unwritten = self.capacity() - self.len();

        // We store `begin` in the case that the lower bound hint is incorrect.
        let begin: *const T = self.bytes.spare_capacity_mut().as_mut_ptr().cast();
        let mut dst: *mut T = begin.cast_mut();

        // As a first step, we manually iterate the iterator up to the known capacity.
        for _ in 0..unwritten {
            let Some(item) = iter.next() else {
                // The lower bound hint may be incorrect.
                break;
            };

            // SAFETY: We have reserved enough capacity to hold this item, and `dst` is a pointer
            // derived from a valid reference to byte data.
            unsafe { dst.write(item) };

            // Note: We used to have `dst.add(iteration).write(item)`, here. However this was much
            // slower than just incrementing `dst`.
            // SAFETY: The offsets fits in `isize`, and because we were able to reserve the memory
            // we know that `add` will not overflow.
            unsafe { dst = dst.add(1) };
        }

        // SAFETY: `dst` was derived from `begin`, which were both valid references to byte data,
        // and since the only operation that `dst` has is `add`, we know that `dst >= begin`.
        let items_written = unsafe { dst.offset_from_unsigned(begin) };
        let length = self.len() + items_written;

        // SAFETY: We have written valid items between the old length and the new length.
        unsafe { self.set_len(length) };

        // Finally, since the iterator will have arbitrarily more items to yield, we push the
        // remaining items normally.
        iter.for_each(|item| self.push(item));
    }

    /// Extends the `BufferMut` with an iterator with `TrustedLen`.
    ///
    /// The caller guarantees that the iterator will have a trusted upper bound, which allows the
    /// implementation to reserve all of the memory needed up front.
    pub fn extend_trusted<I: TrustedLen<Item = T>>(&mut self, iter: I) {
        // Since we know the exact upper bound (from `TrustedLen`), we can reserve all of the memory
        // for this operation up front.
        let (_, upper_bound) = iter.size_hint();
        self.reserve(
            upper_bound
                .vortex_expect("`TrustedLen` iterator somehow didn't have valid upper bound"),
        );

        // We store `begin` in the case that the upper bound hint is incorrect.
        let begin: *const T = self.bytes.spare_capacity_mut().as_mut_ptr().cast();
        let mut dst: *mut T = begin.cast_mut();

        iter.for_each(|item| {
            // SAFETY: We have reserved enough capacity to hold this item, and `dst` is a pointer
            // derived from a valid reference to byte data.
            unsafe { dst.write(item) };

            // Note: We used to have `dst.add(iteration).write(item)`, here. However this was much
            // slower than just incrementing `dst`.
            // SAFETY: The offsets fits in `isize`, and because we were able to reserve the memory
            // we know that `add` will not overflow.
            unsafe { dst = dst.add(1) };
        });

        // SAFETY: `dst` was derived from `begin`, which were both valid references to byte data,
        // and since the only operation that `dst` has is `add`, we know that `dst >= begin`.
        let items_written = unsafe { dst.offset_from_unsigned(begin) };
        let length = self.len() + items_written;

        // SAFETY: We have written valid items between the old length and the new length.
        unsafe { self.set_len(length) };
    }

    /// Creates a `BufferMut` from an iterator with a trusted length.
    ///
    /// Internally, this calls [`extend_trusted()`](Self::extend_trusted).
    pub fn from_trusted_len_iter<I>(iter: I) -> Self
    where
        I: TrustedLen<Item = T>,
    {
        let (_, upper_bound) = iter.size_hint();
        let mut buffer = Self::with_capacity(
            upper_bound
                .vortex_expect("`TrustedLen` iterator somehow didn't have valid upper bound"),
        );

        buffer.extend_trusted(iter);
        buffer
    }

    /// Like [`extend_trusted()`](Self::extend_trusted), but the iterator yields `Result<T, E>`
    /// and the extension short-circuits on the first `Err`.
    ///
    /// On error, items written before the failure remain in the buffer.
    pub fn try_extend_trusted<E, I>(&mut self, iter: I) -> Result<(), E>
    where
        I: TrustedLen<Item = Result<T, E>>,
    {
        let (_, upper_bound) = iter.size_hint();
        self.reserve(
            upper_bound
                .vortex_expect("`TrustedLen` iterator somehow didn't have valid upper bound"),
        );

        let begin: *const T = self.bytes.spare_capacity_mut().as_mut_ptr().cast();
        let mut dst: *mut T = begin.cast_mut();
        let mut result: Result<(), E> = Ok(());

        for item in iter {
            match item {
                Ok(value) => {
                    // SAFETY: We reserved enough capacity to hold this item, and `dst` is a
                    // pointer derived from a valid reference to byte data.
                    unsafe { dst.write(value) };
                    // SAFETY: The offset fits in `isize` because we reserved that much capacity.
                    unsafe { dst = dst.add(1) };
                }
                Err(e) => {
                    result = Err(e);
                    break;
                }
            }
        }

        // SAFETY: `dst` was derived from `begin`, both valid references to byte data, and
        // `dst >= begin` since the only operation on `dst` is `add`.
        let items_written = unsafe { dst.offset_from_unsigned(begin) };
        let length = self.len() + items_written;
        // SAFETY: We have written valid items between the old length and the new length.
        unsafe { self.set_len(length) };

        result
    }

    /// Like [`from_trusted_len_iter()`](Self::from_trusted_len_iter), but the iterator yields
    /// `Result<T, E>` and construction short-circuits on the first `Err`.
    pub fn try_from_trusted_len_iter<E, I>(iter: I) -> Result<Self, E>
    where
        I: TrustedLen<Item = Result<T, E>>,
    {
        let (_, upper_bound) = iter.size_hint();
        let mut buffer = Self::with_capacity(
            upper_bound
                .vortex_expect("`TrustedLen` iterator somehow didn't have valid upper bound"),
        );

        buffer.try_extend_trusted(iter)?;
        Ok(buffer)
    }
}

impl<T> Extend<T> for BufferMut<T> {
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.extend_iter(iter.into_iter())
    }
}

impl<'a, T> Extend<&'a T> for BufferMut<T>
where
    T: Copy + 'a,
{
    #[inline]
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        self.extend_iter(iter.into_iter().copied())
    }
}

impl<T> FromIterator<T> for BufferMut<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        // We don't infer the capacity here and just let the first call to `extend` do it for us.
        let mut buffer = Self::with_capacity(0);
        buffer.extend(iter);
        buffer
    }
}

impl Buf for ByteBufferMut {
    fn remaining(&self) -> usize {
        self.len()
    }

    fn chunk(&self) -> &[u8] {
        self.as_slice()
    }

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

/// As per the BufMut implementation, we must support internal resizing when
/// asked to extend the buffer.
/// See: <https://github.com/tokio-rs/bytes/issues/131>
unsafe impl BufMut for ByteBufferMut {
    #[inline]
    fn remaining_mut(&self) -> usize {
        usize::MAX - self.len()
    }

    #[inline]
    unsafe fn advance_mut(&mut self, cnt: usize) {
        let new_len = self.length + cnt;
        if new_len > self.capacity() {
            vortex_panic!(
                "Cannot advance buffer by {} bytes, only {} bytes of capacity remain",
                cnt,
                self.capacity() - self.length
            );
        }
        // SAFETY: the caller guarantees the bytes up to `new_len` have been initialized.
        unsafe { self.set_len(new_len) };
    }

    #[inline]
    fn chunk_mut(&mut self) -> &mut UninitSlice {
        if self.capacity() == self.len() {
            self.reserve(64);
        }
        UninitSlice::uninit(self.spare_capacity_mut())
    }

    fn put<T: Buf>(&mut self, mut src: T)
    where
        Self: Sized,
    {
        while src.has_remaining() {
            let chunk = src.chunk();
            self.extend_from_slice(chunk);
            src.advance(chunk.len());
        }
    }

    #[inline]
    fn put_slice(&mut self, src: &[u8]) {
        self.extend_from_slice(src);
    }

    #[inline]
    fn put_bytes(&mut self, val: u8, cnt: usize) {
        self.push_n(val, cnt)
    }
}

impl Write for ByteBufferMut {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use bytes::Buf;
    use bytes::BufMut;

    use crate::Alignment;
    use crate::BufferMut;
    use crate::ByteBufferMut;
    use crate::buffer_mut;

    #[test]
    fn capacity() {
        let mut n = 57;
        let mut buf = BufferMut::<i32>::with_capacity_aligned(n, Alignment::new(1024));
        assert!(buf.capacity() >= 57);

        while n > 0 {
            buf.push(0);
            assert!(buf.capacity() >= n);
            n -= 1
        }

        assert_eq!(buf.alignment(), Alignment::new(1024));
    }

    #[test]
    fn from_iter() {
        let buf = BufferMut::from_iter([0, 10, 20, 30]);
        assert_eq!(buf.as_slice(), &[0, 10, 20, 30]);
    }

    #[test]
    fn try_from_trusted_len_iter_ok() {
        let buf = BufferMut::<i32>::try_from_trusted_len_iter(
            [0, 10, 20, 30].iter().map(|&v| Ok::<_, ()>(v)),
        )
        .unwrap();
        assert_eq!(buf.as_slice(), &[0, 10, 20, 30]);
    }

    #[test]
    fn try_from_trusted_len_iter_err() {
        let result: Result<BufferMut<i32>, &'static str> = BufferMut::try_from_trusted_len_iter(
            [0, 10, 20, 30]
                .iter()
                .map(|&v| if v == 20 { Err("bad") } else { Ok(v) }),
        );
        assert_eq!(result.err(), Some("bad"));
    }

    #[test]
    fn extend() {
        let mut buf = BufferMut::empty();
        buf.extend([0i32, 10, 20, 30]);
        buf.extend([40, 50, 60]);
        assert_eq!(buf.as_slice(), &[0, 10, 20, 30, 40, 50, 60]);
    }

    #[test]
    fn push() {
        let mut buf = BufferMut::empty();
        buf.push(1);
        buf.push(2);
        buf.push(3);
        assert_eq!(buf.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn push_n() {
        let mut buf = BufferMut::empty();
        buf.push_n(0, 100);
        assert_eq!(buf.as_slice(), &[0; 100]);
    }

    #[test]
    fn as_mut() {
        let mut buf = buffer_mut![0, 1, 2];
        // Uses DerefMut
        buf[1] = 0;
        // Uses as_mut
        buf.as_mut()[2] = 0;
        assert_eq!(buf.as_slice(), &[0, 0, 0]);
    }

    #[test]
    fn map_each() {
        let buf = buffer_mut![0i32, 1, 2];
        // Add one, and cast to an unsigned u32 in the same closure
        let buf = buf.map_each_in_place(|i| (i + 1) as u32);
        assert_eq!(buf.as_slice(), &[1u32, 2, 3]);
    }

    #[test]
    fn bytes_buf() {
        let mut buf = ByteBufferMut::copy_from("helloworld".as_bytes());
        assert_eq!(buf.remaining(), 10);
        assert_eq!(buf.chunk(), b"helloworld");

        buf.advance(5);
        assert_eq!(buf.remaining(), 5);
        assert_eq!(buf.as_slice(), b"world");
        assert_eq!(buf.chunk(), b"world");
    }

    #[test]
    fn bytes_buf_mut() {
        let mut buf = ByteBufferMut::copy_from("hello".as_bytes());
        assert_eq!(BufMut::remaining_mut(&buf), usize::MAX - 5);

        buf.put_slice(b"world");
        assert_eq!(buf.as_slice(), b"helloworld");
    }

    #[test]
    fn bytes_buf_mut_chunk_advance() {
        // `chunk_mut` + `advance_mut` is the generic `BufMut` write path: it must extend the
        // buffer, not shrink it.
        let mut buf = ByteBufferMut::copy_from(b"hello".as_slice());
        let chunk = buf.chunk_mut();
        assert!(chunk.len() >= 5);
        chunk[..5].copy_from_slice(b"world");
        // SAFETY: we just initialized 5 bytes of the chunk.
        unsafe { buf.advance_mut(5) };
        assert_eq!(buf.as_slice(), b"helloworld");
        assert_eq!(buf.len(), 10);
    }

    #[test]
    fn buffer_mut_zeroed() {
        const LEN: usize = 17;

        let mut buf = BufferMut::<u32>::zeroed(LEN);

        assert_eq!(buf.as_ptr().align_offset(*Alignment::of::<u32>()), 0);
        assert_eq!(buf.as_slice(), &[0; LEN]);

        buf[3] = 7;
        assert_eq!(buf.as_slice()[3], 7);
    }

    #[test]
    fn buffer_mut_zeroed_aligned() {
        const LEN: usize = 17;
        let alignment = Alignment::new(64);

        let mut buf = BufferMut::<u32>::zeroed_aligned(LEN, alignment);

        assert_eq!(buf.as_ptr().align_offset(*alignment), 0);
        assert_eq!(buf.as_slice(), &[0; LEN]);

        buf[3] = 7;
        assert_eq!(buf.as_slice()[3], 7);
    }

    #[test]
    fn from_vec_is_zero_copy_and_mutable() {
        let vec = vec![1i32, 2, 3];
        let ptr = vec.as_ptr();

        let mut buf = BufferMut::from_vec(vec);
        assert_eq!(buf.as_ptr(), ptr);
        buf[0] = 10;

        let vec = buf.into_vec();
        assert_eq!(vec.as_ptr(), ptr);
        assert_eq!(vec, [10, 2, 3]);
    }

    #[test]
    fn into_vec_copies_when_over_aligned() {
        let buf = BufferMut::<i32>::copy_from_aligned([1, 2, 3], Alignment::new(64));
        let vec = buf.into_vec();
        assert_eq!(vec, [1, 2, 3]);
    }

    #[test]
    fn try_into_vec_rejects_over_aligned() {
        let buf = BufferMut::<i32>::copy_from_aligned([1, 2, 3], Alignment::new(64));
        assert!(buf.try_into_vec().is_err());
    }

    #[test]
    fn from_owner_is_mutable() {
        let boxed: Box<[u64]> = vec![1u64, 2, 3].into_boxed_slice();
        let ptr = boxed.as_ptr();

        let mut buf = BufferMut::from_owner(boxed);
        assert_eq!(buf.as_ptr(), ptr);
        buf[2] = 30;
        assert_eq!(buf.as_slice(), &[1, 2, 30]);
    }

    #[test]
    fn split_off_and_unsplit_is_in_place() {
        let mut buf = BufferMut::<u8>::with_capacity(64);
        buf.extend_from_slice(&[1, 2, 3, 4]);
        let ptr = buf.as_ptr();

        let tail = buf.split_off(2);
        assert_eq!(buf.as_slice(), &[1, 2]);
        assert_eq!(tail.as_slice(), &[3, 4]);

        buf.unsplit(tail);
        assert_eq!(buf.as_slice(), &[1, 2, 3, 4]);
        assert_eq!(buf.as_ptr(), ptr, "unsplit did not move the data");
    }

    #[test]
    fn unsplit_copies_when_not_adjacent() {
        let mut a = BufferMut::<u8>::copy_from([1u8, 2].as_slice());
        let b = BufferMut::<u8>::copy_from([3u8, 4].as_slice());
        a.unsplit(b);
        assert_eq!(a.as_slice(), &[1, 2, 3, 4]);
    }

    #[test]
    fn unsplit_into_an_empty_buffer_with_capacity() {
        // `self` owns an allocation but holds no elements: absorbing `other` must not lose it.
        let mut a = BufferMut::<u8>::with_capacity(64);
        let b = BufferMut::<u8>::copy_from([1u8, 2, 3].as_slice());
        a.unsplit(b);
        assert_eq!(a.as_slice(), &[1, 2, 3]);
        assert_eq!(a.len(), 3);
    }

    #[test]
    fn split_off_past_the_length_keeps_both_lengths() {
        let mut a = BufferMut::<u8>::with_capacity(64);
        a.extend_from_slice(&[1, 2, 3, 4]);

        // Splitting inside the spare capacity leaves `a` whole and `b` empty.
        let b = a.split_off(32);
        assert_eq!(a.len(), 4);
        assert_eq!(a.capacity(), 32);
        assert_eq!(b.len(), 0);

        // The halves are not adjacent (`a` has a gap), so this falls back to a copy of nothing.
        a.unsplit(b);
        assert_eq!(a.as_slice(), &[1, 2, 3, 4]);
    }

    #[test]
    fn reserve_preserves_over_alignment() {
        let alignment = Alignment::new(512);
        let mut buf = BufferMut::<u8>::with_capacity_aligned(4, alignment);
        buf.extend_from_slice(&[0u8; 4]);
        // Force a re-allocation.
        buf.extend_from_slice(&[1u8; 4096]);
        assert!(alignment.is_ptr_aligned(buf.as_ptr()));
        assert_eq!(buf.len(), 4100);
    }
}
