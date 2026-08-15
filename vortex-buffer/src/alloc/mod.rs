// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The allocation primitives that back [`Buffer`](crate::Buffer) and
//! [`BufferMut`](crate::BufferMut).
//!
//! These types mirror the model of `bytes::Bytes` and `bytes::BytesMut`: a handle is a window
//! `(ptr, len)` into a reference-counted region, cloning is a refcount bump, and slicing is
//! pointer arithmetic. They differ in that the region is managed directly through
//! [`std::alloc`], which buys three things `bytes` cannot give us:
//!
//! * **Native alignment.** The region's [`Layout`] records the alignment it was allocated with,
//!   so an over-aligned buffer no longer has to over-allocate by `alignment` bytes and offset
//!   into the middle of a `Vec<u8>`.
//! * **Mutable foreign buffers.** A region adopted from foreign memory records whether it may be
//!   written through. `bytes::Bytes::try_into_mut` only succeeds for bytes that came out of
//!   `BytesMut::freeze`, so an adopted `Vec<T>`, Arrow buffer, or mmap could never be made
//!   mutable again without a copy.
//! * **`Vec<T>` round-trips.** A region allocated with exactly `Layout::array::<T>(cap)` is
//!   indistinguishable from a `Vec<T>`'s allocation, so it can be handed back out as one.

use std::alloc::Layout;
use std::alloc::alloc;
use std::alloc::alloc_zeroed;
use std::alloc::dealloc;
use std::alloc::handle_alloc_error;
use std::ptr::NonNull;

use vortex_error::vortex_panic;

use crate::Alignment;

mod shared;
#[cfg(test)]
mod tests;
mod unique;

pub(crate) use shared::SharedBytes;
pub(crate) use unique::UniqueBytes;

/// A dangling but maximally aligned address used by buffers that own no allocation.
///
/// A zero-length slice never dereferences its pointer, so any non-null, sufficiently aligned
/// address is valid for it. The greatest power of two representable in a `usize` satisfies every
/// alignment up to [`Alignment::MAX`].
const DANGLING_ADDR: usize = 1usize << (usize::BITS - 1);

const _: () = assert!(Alignment::MAX.is_offset_aligned(DANGLING_ADDR));

/// A non-null, [`Alignment::MAX`]-aligned pointer to zero readable bytes.
#[inline]
pub(crate) fn dangling() -> NonNull<u8> {
    // SAFETY: `DANGLING_ADDR` is non-zero.
    unsafe { NonNull::new_unchecked(std::ptr::without_provenance_mut(DANGLING_ADDR)) }
}

/// How the memory behind an [`Allocation`] is released.
enum Release {
    /// Allocated by us through the global allocator, and freed with this exact layout.
    Global(Layout),
    /// Kept alive by an owner value; dropping the owner releases the memory.
    ///
    /// The owner is held as a leaked `Box<O>` rather than a `Box<dyn Any>` so that no reborrow of
    /// it ever happens after we have derived the region's pointer from it: moving a `Box` asserts
    /// unique access to its contents, which would invalidate that pointer. This mirrors what
    /// `bytes::Bytes::from_owner` does.
    ///
    /// We never hand out a reference to the owner, we only drop it, so `Send` alone is enough for
    /// the region to be shared across threads.
    Owner {
        owner: *mut (),
        drop: unsafe fn(*mut ()),
    },
}

/// Drop a leaked `Box<O>` that was erased to a `*mut ()`.
///
/// ## Safety
///
/// `ptr` must be the result of `Box::into_raw(Box::<O>::new(..))`, and must not have been dropped.
unsafe fn drop_owner<O>(ptr: *mut ()) {
    // SAFETY: the caller guarantees `ptr` came from `Box::<O>::into_raw` and is still live.
    drop(unsafe { Box::from_raw(ptr.cast::<O>()) })
}

/// A region of memory shared by one or more buffer handles.
///
/// The region is released when the last handle is dropped. Handles hold this behind an [`Arc`],
/// so `Arc::get_mut` answers "is this the only handle?".
///
/// [`Arc`]: std::sync::Arc
pub(crate) struct Allocation {
    /// The first byte of the region.
    base: NonNull<u8>,
    /// The size of the region in bytes.
    size: usize,
    /// Whether the region may be written through.
    ///
    /// This is `false` for regions we only ever obtained a shared reference to. Writing through a
    /// pointer derived from a shared reference is undefined behaviour, and the memory itself may
    /// genuinely be read-only (a `PROT_READ` mapping, a `.rodata` static).
    writable: bool,
    release: Release,
}

// SAFETY: `Allocation` owns its region exclusively and hands out access only through the handles
// in this module, which enforce that at most one of them may write to any given byte. The bytes
// themselves have no interior mutability, and `Release::Owner` is `Send`, so moving the
// `Allocation` (and therefore the eventual deallocation) to another thread is sound.
unsafe impl Send for Allocation {}
// SAFETY: see above. `&Allocation` exposes nothing but the region's extent.
unsafe impl Sync for Allocation {}

impl Allocation {
    /// Allocate `size` bytes aligned to `alignment` through the global allocator.
    ///
    /// ## Panics
    ///
    /// Panics if `size` is zero, or if the requested layout is invalid.
    fn global(size: usize, alignment: Alignment, zeroed: bool) -> Self {
        let layout = layout_for(size, alignment);
        // SAFETY: `layout_for` rejects zero-sized layouts.
        let ptr = unsafe {
            if zeroed {
                alloc_zeroed(layout)
            } else {
                alloc(layout)
            }
        };
        let Some(base) = NonNull::new(ptr) else {
            handle_alloc_error(layout)
        };
        Self {
            base,
            size,
            writable: true,
            release: Release::Global(layout),
        }
    }

    /// Adopt a region kept alive by a leaked `Box<O>`.
    ///
    /// ## Safety
    ///
    /// `owner` must be the result of `Box::into_raw(Box::<O>::new(..))` and must not be dropped by
    /// anyone else. `base..base + size` must lie within memory that stays valid for reads for as
    /// long as that box is alive, and must not be aliased by anything reachable other than through
    /// it. When `writable` is true, the region must additionally be valid for writes, and `base`
    /// must carry write provenance.
    unsafe fn owned<O: Send + 'static>(
        base: NonNull<u8>,
        size: usize,
        writable: bool,
        owner: *mut O,
    ) -> Self {
        Self {
            base,
            size,
            writable,
            release: Release::Owner {
                owner: owner.cast::<()>(),
                drop: drop_owner::<O>,
            },
        }
    }

    /// The address one past the last byte of the region.
    #[inline]
    fn end_addr(&self) -> usize {
        self.base.as_ptr().addr() + self.size
    }

    /// The number of bytes between `ptr` and the end of the region.
    ///
    /// ## Panics
    ///
    /// Panics (in debug builds) if `ptr` does not point into the region.
    #[inline]
    fn capacity_from(&self, ptr: NonNull<u8>) -> usize {
        let addr = ptr.as_ptr().addr();
        debug_assert!(addr >= self.base.as_ptr().addr() && addr <= self.end_addr());
        self.end_addr() - addr
    }

    /// The layout this region was allocated with, if we allocated it ourselves.
    ///
    /// This answers two questions at once: what alignment the region already satisfies, and
    /// whether it can be handed out as a `Vec<T>`. A `Vec<T>` frees its buffer with
    /// `Layout::array::<T>(capacity)`, so the region can only be given away when it was allocated
    /// with precisely that layout.
    fn global_layout(&self) -> Option<Layout> {
        match &self.release {
            Release::Global(layout) => Some(*layout),
            Release::Owner { .. } => None,
        }
    }
}

impl Drop for Allocation {
    fn drop(&mut self) {
        match &self.release {
            Release::Global(layout) => {
                // SAFETY: `base` and `layout` are always kept in step with the allocator call
                // that produced them - `Allocation::global`, the `Vec`'s own
                // `Layout::array::<T>(capacity)` in `UniqueBytes::from_vec`, or the `realloc` in
                // `UniqueBytes::grow_in_place` - so this frees the region with exactly the layout
                // it was allocated with. No handle to it survives the last `Arc`.
                unsafe { dealloc(self.base.as_ptr(), *layout) }
            }
            Release::Owner { owner, drop } => {
                // SAFETY: `Allocation::owned` requires `owner` to be a live leaked box that only
                // we may drop, and this runs once, when the last `Arc` goes away.
                unsafe { drop(*owner) }
            }
        }
    }
}

/// Build the layout for a non-empty allocation, panicking on the (unrepresentable) edge cases.
fn layout_for(size: usize, alignment: Alignment) -> Layout {
    if size == 0 {
        vortex_panic!("Cannot allocate a zero-sized buffer region");
    }
    Layout::from_size_align(size, *alignment).unwrap_or_else(|_| {
        vortex_panic!("Buffer of {size} bytes aligned to {alignment} exceeds the maximum layout")
    })
}

/// Whether two optional allocations are the same allocation.
fn same_allocation(
    a: Option<&std::sync::Arc<Allocation>>,
    b: Option<&std::sync::Arc<Allocation>>,
) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => std::sync::Arc::ptr_eq(a, b),
        _ => false,
    }
}
