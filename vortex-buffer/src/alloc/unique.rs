// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::alloc::Layout;
use std::alloc::handle_alloc_error;
use std::alloc::realloc;
use std::cmp::max;
use std::mem::ManuallyDrop;
use std::mem::MaybeUninit;
use std::ptr::NonNull;
use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_error::vortex_panic;

use super::Allocation;
use super::Release;
use super::SharedBytes;
use super::dangling;
use super::same_allocation;
use crate::Alignment;

/// A uniquely owned, writable window into an [`Allocation`].
///
/// This is the storage behind [`BufferMut`](crate::BufferMut). The window `ptr..ptr + cap` is
/// exclusively ours: no other handle may read or write it, even when the underlying region is
/// shared with the other half of a [`split_off`](Self::split_off).
pub(crate) struct UniqueBytes {
    /// The first byte of the window.
    ptr: NonNull<u8>,
    /// The number of initialised bytes at the front of the window.
    len: usize,
    /// The size of the window in bytes.
    cap: usize,
    /// The region the window points into, or `None` when there is no region to release - an
    /// unallocated buffer. Note that the converse does not hold: an adopted owner whose slice is
    /// empty has a region (which must still be released) but no capacity.
    alloc: Option<Arc<Allocation>>,
}

// SAFETY: `Allocation` is `Send`/`Sync`, and the window is exclusively owned by this handle.
unsafe impl Send for UniqueBytes {}
// SAFETY: see above.
unsafe impl Sync for UniqueBytes {}

impl UniqueBytes {
    /// A window that owns nothing, aligned to [`Alignment::MAX`].
    #[inline]
    pub(crate) fn empty() -> Self {
        Self {
            ptr: dangling(),
            len: 0,
            cap: 0,
            alloc: None,
        }
    }

    /// Allocate an empty window with room for `capacity` bytes, aligned to `alignment`.
    pub(crate) fn with_capacity(capacity: usize, alignment: Alignment) -> Self {
        Self::allocate(capacity, alignment, false)
    }

    /// Allocate a window of `len` zeroed bytes, aligned to `alignment`.
    pub(crate) fn zeroed(len: usize, alignment: Alignment) -> Self {
        let mut this = Self::allocate(len, alignment, true);
        this.len = len;
        this
    }

    fn allocate(capacity: usize, alignment: Alignment, zeroed: bool) -> Self {
        if capacity == 0 {
            // Nothing to allocate, but the dangling pointer still satisfies `alignment`.
            return Self::empty();
        }
        let alloc = Allocation::global(capacity, alignment, zeroed);
        Self {
            ptr: alloc.base,
            len: 0,
            cap: capacity,
            alloc: Some(Arc::new(alloc)),
        }
    }

    /// Take ownership of a `Vec<T>`'s allocation without copying it.
    ///
    /// The buffer treats the elements as plain bytes and never runs `T`'s destructor. Callers
    /// that need destructors must keep the `Vec` alive themselves, e.g. through
    /// [`SharedBytes::from_owner`].
    pub(crate) fn from_vec<T>(vec: Vec<T>) -> Self {
        let mut vec = ManuallyDrop::new(vec);
        let capacity = vec.capacity();
        let len = vec.len();
        // A `Vec` with no capacity owns no allocation, so there is nothing to adopt. Zero-sized
        // elements have no byte representation at all.
        if capacity == 0 || size_of::<T>() == 0 {
            drop(ManuallyDrop::into_inner(vec));
            return Self::empty();
        }

        // SAFETY: `as_mut_ptr` is derived from a unique reference to the `Vec`'s buffer, giving
        // the pointer write provenance over the whole `capacity`.
        let base = unsafe { NonNull::new_unchecked(vec.as_mut_ptr().cast::<u8>()) };
        let size = capacity * size_of::<T>();

        // `Vec<T>` allocates its buffer through the global allocator with exactly this layout, so
        // recording it as one of our own allocations is enough to free it correctly - and lets
        // `try_into_vec` hand it straight back out again.
        let layout = Layout::array::<T>(capacity)
            .unwrap_or_else(|_| vortex_panic!("a live Vec's layout is always representable"));

        let alloc = Allocation {
            base,
            size,
            writable: true,
            release: Release::Global(layout),
        };

        Self {
            ptr: base,
            len: len * size_of::<T>(),
            cap: size,
            alloc: Some(Arc::new(alloc)),
        }
    }

    /// Adopt a writable region kept alive by `owner`, without copying it.
    ///
    /// Taking `owner` by value and going through [`AsMut`] is what makes this safe: it proves
    /// that nothing else can be observing the region while we hold it.
    pub(crate) fn from_owner<O, T>(owner: O) -> Self
    where
        O: AsMut<[T]> + Send + 'static,
    {
        // Leak the box before reading the slice out of it, so that the pointer we keep is derived
        // from a raw pointer that nothing reborrows again.
        let owner: *mut O = Box::into_raw(Box::new(owner));
        // SAFETY: we have just created `owner` and nothing else can free it or reach into it.
        let slice: &mut [T] = unsafe { &mut *owner }.as_mut();
        let size = size_of_val(slice);
        // Note that an empty owner is still kept alive: its `Drop` may release resources the
        // caller expects the buffer to hold on to.
        let base = NonNull::from(slice).cast::<u8>();
        // SAFETY: `slice` points into the leaked owner, which the allocation keeps alive for
        // exactly as long as the region, and `base` is derived from a unique reference so it may
        // be written through.
        let alloc = unsafe { Allocation::owned(base, size, true, owner) };
        Self {
            ptr: base,
            len: size,
            cap: size,
            alloc: Some(Arc::new(alloc)),
        }
    }

    /// Construct from a window into an allocation.
    ///
    /// ## Safety
    ///
    /// The caller must hold the only handle to `ptr..ptr + cap`, that range must lie within
    /// `alloc`'s region, the region must be writable, and the first `len` bytes must be
    /// initialised.
    #[inline]
    pub(super) unsafe fn from_parts(
        ptr: NonNull<u8>,
        len: usize,
        cap: usize,
        alloc: Option<Arc<Allocation>>,
    ) -> Self {
        debug_assert!(len <= cap);
        debug_assert!(alloc.as_ref().is_none_or(|a| a.writable));
        Self {
            ptr,
            len,
            cap,
            alloc,
        }
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
    pub(crate) fn capacity(&self) -> usize {
        self.cap
    }

    #[inline]
    pub(crate) fn as_slice(&self) -> &[u8] {
        // SAFETY: the first `len` bytes of the window are initialised and exclusively ours.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    #[inline]
    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: the first `len` bytes of the window are initialised and exclusively ours.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// The uninitialised tail of the window.
    #[inline]
    pub(crate) fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        // SAFETY: `len..cap` is within the window, which is exclusively ours.
        unsafe {
            std::slice::from_raw_parts_mut(
                self.ptr.as_ptr().add(self.len).cast::<MaybeUninit<u8>>(),
                self.cap - self.len,
            )
        }
    }

    /// Set the number of initialised bytes.
    ///
    /// ## Safety
    ///
    /// `len` must not exceed [`capacity`](Self::capacity), and the bytes up to `len` must be
    /// initialised.
    #[inline]
    pub(crate) unsafe fn set_len(&mut self, len: usize) {
        debug_assert!(len <= self.cap);
        self.len = len;
    }

    /// Advance the start of the window by `cnt` bytes, giving up the bytes skipped over.
    pub(crate) fn advance(&mut self, cnt: usize) {
        if cnt > self.len {
            vortex_panic!(
                "cannot advance past the end of the buffer: {cnt} > {}",
                self.len
            );
        }
        // SAFETY: `cnt <= len <= cap`, so the new start stays inside the window.
        self.ptr = unsafe { self.ptr.add(cnt) };
        self.len -= cnt;
        self.cap -= cnt;
    }

    /// Ensure the window has room for `additional` more bytes past its length.
    ///
    /// The resulting window is aligned to at least `alignment`, and never less well aligned than
    /// it already was.
    #[inline]
    pub(crate) fn reserve(&mut self, additional: usize, alignment: Alignment) {
        if additional <= self.cap - self.len {
            return;
        }
        self.reserve_slow(additional, alignment);
    }

    /// The slow path of [`reserve`](Self::reserve), kept out of line so the common case inlines.
    #[cold]
    fn reserve_slow(&mut self, additional: usize, alignment: Alignment) {
        let required = self
            .len
            .checked_add(additional)
            .vortex_expect("buffer capacity overflow");
        // Amortise the cost of growing by at least doubling each time.
        let target = max(required, self.cap.saturating_mul(2));

        if self.reclaim(required, alignment) {
            return;
        }
        if self.grow_in_place(target, alignment) {
            return;
        }

        // Fall back to a fresh allocation. Preserve any over-alignment the current region has:
        // re-aligning a buffer is expensive, and the allocator charges nothing for it here.
        let alignment = max(alignment, self.allocation_alignment());
        let mut grown = Self::with_capacity(target, alignment);
        grown.extend_from_slice(self.as_slice(), alignment);
        *self = grown;
    }

    /// The alignment the current region was allocated with, or 1 when we did not allocate it.
    fn allocation_alignment(&self) -> Alignment {
        self.alloc
            .as_ref()
            .and_then(|alloc| alloc.global_layout())
            .map(|layout| Alignment::new(layout.align()))
            .unwrap_or_else(Alignment::none)
    }

    /// Reclaim capacity in our own region that a sibling window has since released.
    ///
    /// After `a.split_off(n)` the two halves share one region. Once the other half is dropped we
    /// are free to grow back over it without touching the allocator.
    fn reclaim(&mut self, required: usize, alignment: Alignment) -> bool {
        if !alignment.is_ptr_aligned(self.ptr.as_ptr()) {
            return false;
        }
        let Some(arc) = self.alloc.as_mut() else {
            return false;
        };
        if Arc::get_mut(arc).is_none() {
            return false;
        }
        let available = arc.capacity_from(self.ptr);
        if available < required {
            return false;
        }
        self.cap = available;
        true
    }

    /// Ask the allocator to grow our region in place.
    ///
    /// Only possible when we hold the region exclusively, our window starts at the front of it,
    /// and we allocated it ourselves: `realloc` preserves the original layout's alignment, so it
    /// cannot satisfy a stronger request than the region already meets.
    fn grow_in_place(&mut self, target: usize, alignment: Alignment) -> bool {
        let Some(arc) = self.alloc.as_mut() else {
            return false;
        };
        let Some(alloc) = Arc::get_mut(arc) else {
            return false;
        };
        if self.ptr != alloc.base {
            return false;
        }
        let Release::Global(layout) = &alloc.release else {
            return false;
        };
        let layout = *layout;
        if layout.align() < *alignment {
            return false;
        }
        let new_size = max(target, layout.size());
        if new_size == layout.size() {
            return false;
        }
        let Ok(new_layout) = Layout::from_size_align(new_size, layout.align()) else {
            return false;
        };

        // SAFETY: `alloc.base` was allocated by us with `layout`, `new_size` is non-zero, and
        // `new_layout` is a valid layout for it.
        let ptr = unsafe { realloc(alloc.base.as_ptr(), layout, new_size) };
        let Some(base) = NonNull::new(ptr) else {
            handle_alloc_error(new_layout)
        };

        alloc.base = base;
        alloc.size = new_size;
        alloc.release = Release::Global(new_layout);
        self.ptr = base;
        self.cap = new_size;
        true
    }

    /// Append `slice` to the window, growing it if needed.
    pub(crate) fn extend_from_slice(&mut self, slice: &[u8], alignment: Alignment) {
        self.reserve(slice.len(), alignment);
        // SAFETY: we just reserved `slice.len()` bytes past `len`, and `slice` cannot overlap the
        // spare capacity of a window we own exclusively.
        unsafe {
            std::ptr::copy_nonoverlapping(
                slice.as_ptr(),
                self.ptr.as_ptr().add(self.len),
                slice.len(),
            );
        }
        self.len += slice.len();
    }

    /// Split the window in two at `at`, keeping `..at` and returning `at..`.
    ///
    /// Both halves keep pointing into the same region; neither moves.
    pub(crate) fn split_off(&mut self, at: usize) -> Self {
        if at > self.cap {
            vortex_panic!("cannot split buffer of capacity {} at {at}", self.cap);
        }
        let other = Self {
            // SAFETY: `at <= cap`, so the split point is inside (or at the end of) the window.
            ptr: unsafe { self.ptr.add(at) },
            len: self.len.saturating_sub(at),
            cap: self.cap - at,
            alloc: self.alloc.clone(),
        };
        self.cap = at;
        self.len = self.len.min(at);
        other
    }

    /// Absorb a window previously produced by [`split_off`](Self::split_off).
    ///
    /// `O(1)` when the two windows are still adjacent in the same region; otherwise this
    /// degenerates to a copy.
    pub(crate) fn unsplit(&mut self, other: Self, alignment: Alignment) {
        if self.cap == 0 {
            *self = other;
            return;
        }
        if other.cap == 0 {
            return;
        }
        if same_allocation(self.alloc.as_ref(), other.alloc.as_ref())
            && self.ptr.as_ptr().addr() + self.len == other.ptr.as_ptr().addr()
        {
            self.cap += other.cap;
            self.len += other.len;
            return;
        }
        self.extend_from_slice(other.as_slice(), alignment);
    }

    /// Freeze the window into an immutable, shareable one.
    #[inline]
    pub(crate) fn freeze(self) -> SharedBytes {
        // SAFETY: the window lies within the region.
        unsafe { SharedBytes::from_parts(self.ptr, self.len, self.alloc) }
    }

    /// Hand the region out as a `Vec<T>`, if it is exactly a `Vec<T>`'s allocation.
    ///
    /// This succeeds when the region was allocated with `Layout::array::<T>(capacity)` - either
    /// because it came from a `Vec<T>` in the first place, or because it was allocated with
    /// exactly `align_of::<T>()` - and our window starts at the front of it. An over-aligned
    /// buffer cannot be given away, because `Vec` would free it with the wrong layout.
    pub(crate) fn try_into_vec<T>(mut self) -> Result<Vec<T>, Self> {
        let elem = size_of::<T>();
        if elem == 0 || !self.len.is_multiple_of(elem) {
            return Err(self);
        }
        // Nothing to hand over, so an empty `Vec` is trivially a zero-copy answer. This also
        // covers windows that own no region at all.
        if self.len == 0 {
            return Ok(Vec::new());
        }

        let ptr = self.ptr;
        let capacity = self.alloc.as_mut().and_then(|arc| {
            // We must be the only handle before we can give the region away.
            Arc::get_mut(arc)?;
            let layout = arc.global_layout()?;
            (ptr == arc.base
                && layout.align() == align_of::<T>()
                && layout.size().is_multiple_of(elem))
            .then(|| layout.size() / elem)
        });

        let Some(capacity) = capacity else {
            return Err(self);
        };
        let length = self.len / elem;

        // The `Vec` takes the region over, so our allocation must not free it.
        let alloc = self.alloc.take().vortex_expect("checked above");
        let alloc = Arc::into_inner(alloc).vortex_expect("we hold the only handle");
        let _defused = ManuallyDrop::new(alloc);

        // SAFETY: the region was allocated by the global allocator with exactly
        // `Layout::array::<T>(capacity)`, its first `length` elements are initialised, and we
        // have just given up our own claim to it.
        Ok(unsafe { Vec::from_raw_parts(ptr.cast::<T>().as_ptr(), length, capacity) })
    }
}

impl std::fmt::Debug for UniqueBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UniqueBytes")
            .field("ptr", &self.ptr)
            .field("len", &self.len)
            .field("cap", &self.cap)
            .field("owned", &self.alloc.is_some())
            .finish()
    }
}

impl AsRef<[u8]> for UniqueBytes {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl PartialEq for UniqueBytes {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Eq for UniqueBytes {}
