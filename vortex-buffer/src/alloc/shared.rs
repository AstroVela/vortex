// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Ordering;
use std::hash::Hash;
use std::hash::Hasher;
use std::ptr::NonNull;
use std::sync::Arc;

use vortex_error::vortex_panic;

use super::Allocation;
use super::UniqueBytes;
use super::dangling;

/// An immutable, reference-counted window into an [`Allocation`].
///
/// This is the storage behind [`Buffer`](crate::Buffer). Cloning and slicing are `O(1)`.
pub(crate) struct SharedBytes {
    /// The first byte of the window.
    ptr: NonNull<u8>,
    /// The length of the window in bytes.
    len: usize,
    /// The region the window points into, or `None` when the window borrows memory that outlives
    /// every handle (a `'static` slice, or a zero-length dangling window).
    alloc: Option<Arc<Allocation>>,
}

// SAFETY: `Allocation` is `Send`/`Sync`, and a `SharedBytes` only ever hands out `&[u8]` into a
// region that no live handle may write to (`try_into_unique` requires unique ownership).
unsafe impl Send for SharedBytes {}
// SAFETY: see above.
unsafe impl Sync for SharedBytes {}

impl SharedBytes {
    /// An empty window that owns nothing, aligned to [`Alignment::MAX`](crate::Alignment::MAX).
    #[inline]
    pub(crate) fn empty() -> Self {
        Self {
            ptr: dangling(),
            len: 0,
            alloc: None,
        }
    }

    /// Borrow a `'static` slice without copying it.
    pub(crate) fn from_static(slice: &'static [u8]) -> Self {
        if slice.is_empty() {
            return Self::empty();
        }
        Self {
            ptr: NonNull::from(slice).cast(),
            len: slice.len(),
            alloc: None,
        }
    }

    /// Adopt a region kept alive by `owner`, without copying it.
    ///
    /// The window covers exactly the bytes `owner` currently references. The region is recorded
    /// as read-only, so [`try_into_unique`](Self::try_into_unique) will refuse it: `slice` is
    /// derived from a shared reference, and writing through such a pointer is undefined
    /// behaviour even when the memory itself is writable.
    pub(crate) fn from_owner<O, T>(owner: O) -> Self
    where
        O: AsRef<[T]> + Send + 'static,
    {
        // Box first so the owner's final address is fixed before we read the slice out of it.
        let owner: Box<O> = Box::new(owner);
        let slice: &[T] = (*owner).as_ref();
        let size = size_of_val(slice);
        if size == 0 {
            return Self::empty();
        }
        let base = NonNull::from(slice).cast::<u8>();
        // SAFETY: `slice` points into the boxed owner, which we keep alive for exactly as long as
        // the allocation. We record the region as read-only.
        let alloc = unsafe { Allocation::owned(base, size, false, owner) };
        Self {
            ptr: base,
            len: size,
            alloc: Some(Arc::new(alloc)),
        }
    }

    /// Construct from a window into an allocation.
    ///
    /// ## Safety
    ///
    /// `ptr..ptr + len` must lie within `alloc`'s region.
    #[inline]
    pub(super) unsafe fn from_parts(
        ptr: NonNull<u8>,
        len: usize,
        alloc: Option<Arc<Allocation>>,
    ) -> Self {
        Self { ptr, len, alloc }
    }

    #[inline]
    pub(crate) fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub(crate) fn as_slice(&self) -> &[u8] {
        // SAFETY: the window is a valid, initialised, immutable region of `len` bytes. When `len`
        // is zero `ptr` may dangle, which `from_raw_parts` permits so long as it is aligned and
        // non-null.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Returns a new handle to `self[begin..end]`.
    ///
    /// ## Panics
    ///
    /// Panics if the range is out of bounds.
    pub(crate) fn slice(&self, begin: usize, end: usize) -> Self {
        if begin > end {
            vortex_panic!("range start must not be greater than end: {begin} <= {end}");
        }
        if end > self.len {
            vortex_panic!("range end out of bounds: {end} > {}", self.len);
        }
        if begin == end {
            return Self::empty();
        }
        Self {
            // SAFETY: `begin < len`, so the offset stays inside the window.
            ptr: unsafe { self.ptr.add(begin) },
            len: end - begin,
            alloc: self.alloc.clone(),
        }
    }

    /// Returns a new handle to `subset`, which must be contained within this window.
    ///
    /// ## Panics
    ///
    /// Panics if `subset` is not contained within this window.
    pub(crate) fn slice_ref(&self, subset: &[u8]) -> Self {
        // An empty subset carries no address we can meaningfully check against.
        if subset.is_empty() {
            return Self::empty();
        }

        let self_start = self.ptr.as_ptr().addr();
        let sub_start = subset.as_ptr().addr();

        if sub_start < self_start {
            vortex_panic!("subset pointer starts before the buffer");
        }
        if sub_start + subset.len() > self_start + self.len {
            vortex_panic!("subset pointer ends past the end of the buffer");
        }

        let offset = sub_start - self_start;
        self.slice(offset, offset + subset.len())
    }

    /// Advance the start of the window by `cnt` bytes.
    ///
    /// ## Panics
    ///
    /// Panics if `cnt > len`.
    pub(crate) fn advance(&mut self, cnt: usize) {
        if cnt > self.len {
            vortex_panic!(
                "cannot advance past the end of the buffer: {cnt} > {}",
                self.len
            );
        }
        // SAFETY: `cnt <= len`, so the new start stays inside (or exactly at the end of) the
        // window.
        self.ptr = unsafe { self.ptr.add(cnt) };
        self.len -= cnt;
    }

    /// Shorten the window to zero bytes, keeping the start.
    #[inline]
    pub(crate) fn clear(&mut self) {
        self.len = 0;
    }

    /// Whether this is the only handle to the underlying region.
    pub(crate) fn is_unique(&self) -> bool {
        match &self.alloc {
            None => true,
            Some(alloc) => Arc::strong_count(alloc) == 1 && Arc::weak_count(alloc) == 0,
        }
    }

    /// Take exclusive ownership of the region, if this is the only handle to writable memory.
    ///
    /// The returned buffer's capacity runs from the start of this window to the end of the
    /// region, so a handle onto the front of a partially filled allocation regains the whole of
    /// its spare capacity.
    pub(crate) fn try_into_unique(mut self) -> Result<UniqueBytes, Self> {
        let capacity = match self.alloc.as_mut() {
            // Nothing owns the memory: a zero-length window is trivially exclusive, but a
            // borrowed `'static` slice can never be written to.
            None => {
                if self.len != 0 {
                    return Err(self);
                }
                0
            }
            Some(alloc) => {
                if !alloc.writable || Arc::get_mut(alloc).is_none() {
                    return Err(self);
                }
                alloc.capacity_from(self.ptr)
            }
        };

        // SAFETY: we hold the only handle to the region, and the window lies within it.
        Ok(unsafe { UniqueBytes::from_parts(self.ptr, self.len, capacity, self.alloc) })
    }
}

impl std::fmt::Debug for SharedBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedBytes")
            .field("ptr", &self.ptr)
            .field("len", &self.len)
            .field("owned", &self.alloc.is_some())
            .finish()
    }
}

impl Clone for SharedBytes {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            len: self.len,
            alloc: self.alloc.clone(),
        }
    }
}

impl AsRef<[u8]> for SharedBytes {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl PartialEq for SharedBytes {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Eq for SharedBytes {}

impl PartialOrd for SharedBytes {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SharedBytes {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl Hash for SharedBytes {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state)
    }
}
