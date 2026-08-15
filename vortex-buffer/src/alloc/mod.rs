// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The allocation primitives that back [`Buffer`](crate::Buffer) and
//! [`BufferMut`](crate::BufferMut).
//!
//! These types mirror the model of `bytes::Bytes` and `bytes::BytesMut`: a handle is a window
//! `(ptr, len)` into a region, cloning is a refcount bump, and slicing is pointer arithmetic. They
//! differ in that the region is managed directly through [`std::alloc`], which buys three things
//! `bytes` cannot give us:
//!
//! * **Native alignment.** A region records the alignment it was allocated with, so an
//!   over-aligned buffer no longer has to over-allocate by `alignment` bytes and offset into the
//!   middle of a `Vec<u8>`.
//! * **Mutable foreign buffers.** A region adopted from foreign memory records whether it may be
//!   written through. `bytes::Bytes::try_into_mut` only succeeds for bytes that came out of
//!   `BytesMut::freeze`, so an adopted `Vec<T>`, Arrow buffer, or mmap could never be made
//!   mutable again without a copy.
//! * **`Vec<T>` round-trips.** A region allocated with exactly `Layout::array::<T>(cap)` is
//!   indistinguishable from a `Vec<T>`'s allocation, so it can be handed back out as one.
//!
//! # Deferred sharing
//!
//! A handle that has never been shared owns its region outright, and describes it inline: no
//! refcount is allocated until a second handle actually exists. This is the same trick `bytes`
//! plays with its "promotable" vtables, and it is what keeps the common
//! build-freeze-read-drop path down to a single allocation.
//!
//! The ownership state is one word, [`State`]:
//!
//! ```text
//! bit  63                              8   7  2   1 0
//!     ┌─────────────────────────────────┬───────┬─────┐
//!     │ size                            │ align │ 0 1 │  OWNED  - inline description
//!     └─────────────────────────────────┴───────┴─────┘
//!     ┌───────────────────────────────────────────┬───┐
//!     │ *mut Shared                               │ 0 │  SHARED - refcounted, 8-aligned
//!     └───────────────────────────────────────────┴───┘
//!                                               0b10    STATIC - owns nothing
//! ```
//!
//! `OWNED` keeps the region's start in the handle's own `base` field, so advancing or truncating a
//! handle never has to allocate. Promotion to `SHARED` only ever rewrites the state word, never
//! `base`, which is what lets it happen through a shared reference with a single compare-exchange.

use std::alloc::Layout;
use std::alloc::alloc;
use std::alloc::alloc_zeroed;
use std::alloc::dealloc;
use std::alloc::handle_alloc_error;
use std::ptr::NonNull;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::atomic::fence;

use vortex_error::vortex_panic;

use crate::Alignment;

#[cfg(test)]
mod property_tests;
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

// -------------------------------------------------------------------------------------------
// State word
// -------------------------------------------------------------------------------------------

const KIND_MASK: usize = 0b11;
/// The state word is a `*mut Shared`. A `Box` is at least 8-aligned, so its low bits are zero.
const KIND_SHARED: usize = 0b00;
/// The handle owns a global allocation outright, described inline by this word and `base`.
const KIND_OWNED: usize = 0b01;
/// The handle owns nothing: `'static` memory, or an empty window over no region at all.
const KIND_STATIC: usize = 0b10;

const ALIGN_SHIFT: u32 = 2;
const ALIGN_BITS: u32 = 6;
const SIZE_SHIFT: u32 = ALIGN_SHIFT + ALIGN_BITS;

/// The largest region an `OWNED` state word can describe. Anything larger is held through a
/// [`Shared`] instead, which stores the size in full.
const MAX_OWNED_SIZE: usize = usize::MAX >> SIZE_SHIFT;

/// The largest alignment exponent is that of [`Alignment::MAX`], and it has to fit in the field.
const _: () = assert!((1usize << ALIGN_BITS) > (usize::BITS - 1) as usize);

/// The ownership state of a buffer handle. See the module docs for the encoding.
///
/// This wraps a *pointer* rather than a `usize` so that the `SHARED` case keeps its provenance:
/// rebuilding the `Shared` pointer from an integer address would make it undereferenceable. The
/// `OWNED` and `STATIC` cases are pure bit patterns that are never dereferenced, so they carry no
/// provenance and do not need any.
#[derive(Clone, Copy)]
pub(super) struct State(*mut ());

impl State {
    /// The state of a handle that owns nothing.
    const STATIC: Self = Self(std::ptr::without_provenance_mut(KIND_STATIC));

    /// Describe a global allocation inline, if it is small enough to fit in the word.
    ///
    /// Callers must pass the size and alignment of a `Layout` that is known to be valid, so that
    /// [`owned_layout`](Self::owned_layout) can rebuild it without re-checking.
    #[inline]
    fn owned(size: usize, alignment: Alignment) -> Option<Self> {
        (size <= MAX_OWNED_SIZE).then(|| {
            Self(std::ptr::without_provenance_mut(
                KIND_OWNED
                    | (usize::from(alignment.exponent()) << ALIGN_SHIFT)
                    | (size << SIZE_SHIFT),
            ))
        })
    }

    /// Describe a region held through a [`Shared`].
    ///
    /// ## Safety
    ///
    /// `shared` must be a live pointer from [`Shared::into_raw`], and this state takes over one
    /// of its references.
    #[inline]
    unsafe fn shared(shared: *mut Shared) -> Self {
        debug_assert_eq!(shared.addr() & KIND_MASK, KIND_SHARED, "Shared is aligned");
        Self(shared.cast())
    }

    #[inline]
    fn addr(self) -> usize {
        self.0.addr()
    }

    #[inline]
    fn kind(self) -> usize {
        self.addr() & KIND_MASK
    }

    #[inline]
    fn is_owned(self) -> bool {
        self.kind() == KIND_OWNED
    }

    #[inline]
    fn is_static(self) -> bool {
        self.kind() == KIND_STATIC
    }

    /// Whether the region is held through a [`Shared`], and so has an identity two handles can be
    /// compared on. `OWNED` words describe a region rather than naming one: two handles that
    /// allocated the same layout independently carry the same word.
    #[inline]
    fn is_shared(self) -> bool {
        self.kind() == KIND_SHARED
    }

    /// The size of the inline-described region.
    #[inline]
    fn owned_size(self) -> usize {
        debug_assert!(self.is_owned());
        self.addr() >> SIZE_SHIFT
    }

    /// The alignment of the inline-described region.
    #[inline]
    fn owned_alignment(self) -> Alignment {
        debug_assert!(self.is_owned());
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the exponent occupies ALIGN_BITS bits, so it fits in a u8"
        )]
        Alignment::from_exponent(((self.addr() >> ALIGN_SHIFT) & ((1 << ALIGN_BITS) - 1)) as u8)
    }

    /// The layout the inline-described region was allocated with.
    #[inline]
    fn owned_layout(self) -> Layout {
        // SAFETY: every `State::owned` caller passes the parts of a valid `Layout`, and the size
        // round-trips exactly because `owned` rejects anything wider than `MAX_OWNED_SIZE`.
        unsafe { Layout::from_size_align_unchecked(self.owned_size(), *self.owned_alignment()) }
    }

    /// The [`Shared`] this state points at.
    ///
    /// ## Safety
    ///
    /// The state must be `SHARED`, and the pointer must still be live.
    #[inline]
    unsafe fn as_shared(self) -> *mut Shared {
        debug_assert_eq!(self.kind(), KIND_SHARED);
        self.0.cast::<Shared>()
    }
}

impl PartialEq for State {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.addr() == other.addr()
    }
}

impl Eq for State {}

// -------------------------------------------------------------------------------------------
// Shared state
// -------------------------------------------------------------------------------------------

/// How the memory behind a region is released.
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

/// The refcounted description of a region shared by more than one handle.
///
/// This is allocated lazily: a handle that has never been shared describes its region inline in
/// its [`State`] instead.
pub(super) struct Shared {
    /// Number of live handles.
    refcount: AtomicUsize,
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

// SAFETY: `Shared` owns its region exclusively and hands out access only through the handles in
// this module, which enforce that at most one of them may write to any given byte. The bytes
// themselves have no interior mutability, and `Release::Owner` is `Send`, so moving the
// deallocation to another thread is sound.
unsafe impl Send for Shared {}
// SAFETY: see above. `&Shared` exposes nothing but the region's extent and its refcount.
unsafe impl Sync for Shared {}

impl Shared {
    /// Move this description onto the heap, where handles can point at it.
    #[inline]
    fn into_raw(self) -> *mut Shared {
        Box::into_raw(Box::new(self))
    }

    /// Take another reference.
    ///
    /// ## Safety
    ///
    /// `shared` must be live, and the caller must already hold a reference to it.
    #[inline]
    unsafe fn retain(shared: *mut Shared) {
        // SAFETY: the caller guarantees the pointer is live.
        let old = unsafe { &*shared }.refcount.fetch_add(1, Ordering::Relaxed);
        // The count can only overflow if handles are leaked in a loop; abort rather than wrap
        // into a premature free. `bytes` and `Arc` take the same precaution.
        if old > usize::MAX / 2 {
            std::process::abort();
        }
    }

    /// Give up a reference, releasing the region if it was the last one.
    ///
    /// ## Safety
    ///
    /// `shared` must be live, and the caller must hold the reference being given up.
    #[inline]
    unsafe fn release(shared: *mut Shared) {
        // SAFETY: the caller guarantees the pointer is live.
        if unsafe { &*shared }.refcount.fetch_sub(1, Ordering::Release) != 1 {
            return;
        }
        // Synchronise with every other handle's release before running the destructor.
        fence(Ordering::Acquire);
        // SAFETY: the refcount reached zero, so we hold the only reference and may free both the
        // region and this box.
        unsafe { drop(Box::from_raw(shared)) }
    }

    /// Whether this is the only handle to the region.
    #[inline]
    fn is_unique(&self) -> bool {
        self.refcount.load(Ordering::Acquire) == 1
    }

    /// The layout this region was allocated with, if we allocated it ourselves.
    #[inline]
    fn global_layout(&self) -> Option<Layout> {
        match &self.release {
            Release::Global(layout) => Some(*layout),
            Release::Owner { .. } => None,
        }
    }

    /// The address one past the last byte of the region.
    #[inline]
    fn end_addr(&self) -> usize {
        self.base.as_ptr().addr() + self.size
    }
}

impl Drop for Shared {
    fn drop(&mut self) {
        match &self.release {
            Release::Global(layout) => {
                // SAFETY: `base` and `layout` are always kept in step with the allocator call that
                // produced them, so this frees the region with exactly the layout it was
                // allocated with. The refcount reached zero, so no handle survives.
                unsafe { dealloc(self.base.as_ptr(), *layout) }
            }
            Release::Owner { owner, drop } => {
                // SAFETY: the owner is a live leaked box that only we may drop, and the refcount
                // reached zero.
                unsafe { drop(*owner) }
            }
        }
    }
}

// -------------------------------------------------------------------------------------------
// Allocation helpers
// -------------------------------------------------------------------------------------------

/// Build the layout for a non-empty allocation, panicking on the (unrepresentable) edge cases.
fn layout_for(size: usize, alignment: Alignment) -> Layout {
    if size == 0 {
        vortex_panic!("Cannot allocate a zero-sized buffer region");
    }
    Layout::from_size_align(size, *alignment).unwrap_or_else(|_| {
        vortex_panic!("Buffer of {size} bytes aligned to {alignment} exceeds the maximum layout")
    })
}

/// Allocate `size` bytes aligned to `alignment` through the global allocator.
fn allocate(size: usize, alignment: Alignment, zeroed: bool) -> (NonNull<u8>, Layout) {
    let layout = layout_for(size, alignment);
    // SAFETY: `layout_for` rejects zero-sized layouts.
    let ptr = unsafe {
        if zeroed {
            alloc_zeroed(layout)
        } else {
            alloc(layout)
        }
    };
    match NonNull::new(ptr) {
        Some(base) => (base, layout),
        None => handle_alloc_error(layout),
    }
}

/// Build the `Shared` for a region we allocated ourselves.
fn shared_global(base: NonNull<u8>, layout: Layout, refcount: usize) -> Shared {
    Shared {
        refcount: AtomicUsize::new(refcount),
        base,
        size: layout.size(),
        writable: true,
        release: Release::Global(layout),
    }
}
