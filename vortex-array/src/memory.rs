// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session-scoped memory allocation for host-side buffers.

use std::any::Any;
use std::fmt::Debug;
use std::mem::size_of;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use bytes::Bytes;
use parking_lot::Mutex;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::SessionExt;
use vortex_session::SessionGuard;
use vortex_session::SessionVar;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;

/// Mutable host buffer contract used by [`WritableHostBuffer`].
pub trait HostBufferMut: Send + 'static {
    /// Returns the logical byte length of the buffer.
    fn len(&self) -> usize;

    /// Whether the buffer is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the alignment of the buffer.
    fn alignment(&self) -> Alignment;

    /// Returns mutable access to the writable byte range.
    fn as_mut_slice(&mut self) -> &mut [u8];

    /// Freeze the buffer into an immutable [`ByteBuffer`].
    fn freeze(self: Box<Self>) -> ByteBuffer;
}

/// Exact-size writable host buffer returned by a [`HostAllocator`].
pub struct WritableHostBuffer {
    inner: Box<dyn HostBufferMut>,
}

impl WritableHostBuffer {
    /// Create a writable host buffer from an implementation of [`HostBufferMut`].
    pub fn new(inner: Box<dyn HostBufferMut>) -> Self {
        Self { inner }
    }

    /// Returns the logical byte length of the buffer.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns true when the buffer has zero bytes.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the alignment of the buffer.
    pub fn alignment(&self) -> Alignment {
        self.inner.alignment()
    }

    /// Returns mutable access to the writable byte range.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.inner.as_mut_slice()
    }

    /// Returns mutable access to the buffer as a typed slice.
    pub fn as_mut_slice_typed<T>(&mut self) -> VortexResult<&mut [T]> {
        vortex_ensure!(
            size_of::<T>() != 0,
            InvalidArgument: "Cannot create typed mutable slice for zero-sized type {}",
            std::any::type_name::<T>()
        );
        vortex_ensure!(
            self.alignment().is_aligned_to(Alignment::of::<T>()),
            InvalidArgument: "Buffer is not sufficiently aligned for type {}",
            std::any::type_name::<T>()
        );

        let bytes = self.as_mut_slice();
        let byte_len = bytes.len();
        let ptr = bytes.as_mut_ptr();
        let type_size = size_of::<T>();

        vortex_ensure!(
            byte_len.is_multiple_of(type_size),
            InvalidArgument: "Buffer length {byte_len} is not a multiple of {} for {}",
            type_size,
            std::any::type_name::<T>()
        );

        // SAFETY: We checked size divisibility and pointer alignment for `T`,
        // and we have exclusive mutable access to the underlying bytes.
        Ok(unsafe { std::slice::from_raw_parts_mut(ptr.cast::<T>(), byte_len / type_size) })
    }

    /// Freeze the writable buffer into an immutable [`ByteBuffer`].
    pub fn freeze(self) -> ByteBuffer {
        self.inner.freeze()
    }

    /// Freeze the writable buffer into a typed immutable [`Buffer<T>`].
    pub fn freeze_typed<T>(self) -> VortexResult<Buffer<T>> {
        vortex_ensure!(
            size_of::<T>() != 0,
            InvalidArgument: "Cannot freeze typed buffer for zero-sized type {}",
            std::any::type_name::<T>()
        );

        let buffer = self.freeze();
        let byte_len = buffer.len();
        let type_size = size_of::<T>();
        let type_align = Alignment::of::<T>();

        vortex_ensure!(
            byte_len.is_multiple_of(type_size),
            InvalidArgument: "Buffer length {byte_len} is not a multiple of {} for {}",
            type_size,
            std::any::type_name::<T>()
        );
        vortex_ensure!(
            buffer.is_aligned(type_align),
            InvalidArgument: "Buffer pointer is not aligned to {} for {}",
            type_align,
            std::any::type_name::<T>()
        );

        Ok(Buffer::from_byte_buffer(buffer))
    }
}

impl Debug for WritableHostBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WritableHostBuffer")
            .field("len", &self.len())
            .field("alignment", &self.alignment())
            .finish()
    }
}

/// Allocator for exact-size writable host buffers.
pub trait HostAllocator: Debug + Send + Sync + 'static {
    /// Allocate a writable host buffer with the requested byte length and alignment.
    fn allocate(&self, len: usize, alignment: Alignment) -> VortexResult<WritableHostBuffer>;
}

/// Shared allocator reference used throughout session-scoped memory APIs.
pub type HostAllocatorRef = Arc<dyn HostAllocator>;

/// Extension methods for [`HostAllocator`]s.
pub trait HostAllocatorExt: HostAllocator {
    /// Allocate host memory for `len` elements of `T` using `Alignment::of::<T>()`.
    fn allocate_typed<T>(&self, len: usize) -> VortexResult<WritableHostBuffer> {
        let bytes = len.checked_mul(size_of::<T>()).ok_or_else(|| {
            vortex_err!(
                "Typed host allocation overflow for type {} and len {}",
                std::any::type_name::<T>(),
                len
            )
        })?;
        self.allocate(bytes, Alignment::of::<T>())
    }
}

impl<A: HostAllocator + ?Sized> HostAllocatorExt for A {}

/// Session-scoped memory configuration for Vortex arrays.
#[derive(Clone, Debug)]
pub struct MemorySession {
    allocator: HostAllocatorRef,
}

impl MemorySession {
    /// Creates a new session memory configuration using the provided allocator.
    pub fn new(allocator: HostAllocatorRef) -> Self {
        Self { allocator }
    }

    /// Returns the configured allocator.
    pub fn allocator(&self) -> HostAllocatorRef {
        Arc::clone(&self.allocator)
    }

    /// Updates the configured allocator.
    pub fn set_allocator(&mut self, allocator: HostAllocatorRef) {
        self.allocator = allocator;
    }
}

impl Default for MemorySession {
    fn default() -> Self {
        Self::new(Arc::new(DefaultHostAllocator))
    }
}

impl SessionVar for MemorySession {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Extension trait for accessing session-scoped memory configuration.
pub trait MemorySessionExt: SessionExt {
    /// Returns the memory session for this execution/session context.
    fn memory(&self) -> SessionGuard<'_, MemorySession> {
        self.get::<MemorySession>()
    }

    /// Returns the configured host allocator for this execution/session context.
    fn allocator(&self) -> HostAllocatorRef {
        self.memory().allocator()
    }

    /// Configures the session to use `allocator` as its host allocator, mutating it in place and
    /// returning it for chaining.
    fn with_allocator(self, allocator: HostAllocatorRef) -> VortexSession {
        let session = self.session();
        session.get_mut::<MemorySession>().set_allocator(allocator);
        session
    }
}

impl<S: SessionExt> MemorySessionExt for S {}

/// Default host allocator.
#[derive(Debug, Default)]
pub struct DefaultHostAllocator;

impl HostAllocator for DefaultHostAllocator {
    fn allocate(&self, len: usize, alignment: Alignment) -> VortexResult<WritableHostBuffer> {
        let mut buffer = ByteBufferMut::with_capacity_aligned(len, alignment);
        // SAFETY: We fully initialize this slice before freezing it.
        unsafe { buffer.set_len(len) };
        Ok(WritableHostBuffer::new(Box::new(
            DefaultWritableHostBuffer { buffer, alignment },
        )))
    }
}

#[derive(Debug)]
struct DefaultWritableHostBuffer {
    buffer: ByteBufferMut,
    alignment: Alignment,
}

#[derive(Debug)]
struct HostBufferOwner {
    buffer: ByteBufferMut,
}

impl AsRef<[u8]> for HostBufferOwner {
    fn as_ref(&self) -> &[u8] {
        self.buffer.as_slice()
    }
}

impl HostBufferMut for DefaultWritableHostBuffer {
    fn len(&self) -> usize {
        self.buffer.len()
    }

    fn alignment(&self) -> Alignment {
        self.alignment
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        self.buffer.as_mut_slice()
    }

    fn freeze(self: Box<Self>) -> ByteBuffer {
        let Self { buffer, alignment } = *self;
        let bytes = Bytes::from_owner(HostBufferOwner { buffer });
        ByteBuffer::from_bytes_aligned(bytes, alignment)
    }
}

/// A [`HostAllocator`] that recycles freed buffers instead of returning them to the system
/// allocator.
///
/// Coalesced reads request buffers spanning up to [`CoalesceConfig::max_size`] bytes (4MiB for
/// local files, 16MiB for object storage). Allocations that large bypass the system allocator's
/// free lists and are served by fresh `mmap` regions, so every read pays for page faults and every
/// free returns the pages to the kernel.
///
/// This allocator keeps freed buffers in per-(size class, alignment) free lists. Because buffers
/// are handed back through the [`ByteBuffer`] owner, a buffer only returns to the pool once the
/// last zero-copy slice referencing it has been dropped.
///
/// Pooling is bounded on three axes: allocations outside `[min_pooled_size, max_pooled_size]` are
/// never pooled, each class holds at most `max_buffers_per_class` buffers, and the pool as a whole
/// holds at most `max_pooled_bytes`.
///
/// This is **not** the default allocator. On Linux/glibc, recycling read buffers does not measure
/// faster than allocating them: a freshly mapped region arrives pre-zeroed and its page faults are
/// paid by the `pread` that has to fill the buffer anyway, so the pool trades those faults for an
/// equivalent write to cold memory. See `vortex/examples/alloc_profile.rs` for the measurement.
/// It is retained as an opt-in for allocators and platforms where large-allocation handling is
/// more expensive, and for workloads that want a bounded, reusable buffer budget.
///
/// [`CoalesceConfig::max_size`]: https://docs.rs/vortex-io/latest/vortex_io/struct.CoalesceConfig.html
#[derive(Debug)]
pub struct PoolingHostAllocator {
    inner: Arc<Pool>,
}

/// Tuning for a [`PoolingHostAllocator`].
#[derive(Clone, Copy, Debug)]
pub struct PoolConfig {
    /// Allocations smaller than this are served directly by the system allocator.
    pub min_pooled_size: usize,
    /// Allocations larger than this are served directly by the system allocator.
    pub max_pooled_size: usize,
    /// The maximum number of buffers retained per (size class, alignment) pair.
    pub max_buffers_per_class: usize,
    /// The maximum number of bytes retained across every class.
    pub max_pooled_bytes: usize,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            // Below this the system allocator's free lists are already effective.
            min_pooled_size: 64 * 1024,
            max_pooled_size: 32 * 1024 * 1024,
            max_buffers_per_class: 8,
            max_pooled_bytes: 256 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Default)]
struct PoolStats {
    hits: AtomicU64,
    misses: AtomicU64,
}

#[derive(Debug)]
struct Pool {
    config: PoolConfig,
    classes: Mutex<HashMap<(usize, usize), Vec<ByteBufferMut>>>,
    pooled_bytes: AtomicUsize,
    stats: PoolStats,
}

impl Pool {
    /// Requests are rounded up to a power of two so that near-identical reads share a class.
    fn size_class(&self, len: usize) -> Option<usize> {
        (len >= self.config.min_pooled_size && len <= self.config.max_pooled_size)
            .then(|| len.next_power_of_two())
            .filter(|class| *class <= self.config.max_pooled_size)
    }

    fn take(&self, class: usize, alignment: Alignment) -> ByteBufferMut {
        let mut classes = self.classes.lock();
        while let Some(buffer) = classes
            .get_mut(&(class, *alignment))
            .and_then(|slot| slot.pop())
        {
            self.pooled_bytes.fetch_sub(class, Ordering::Relaxed);
            // A recycled buffer that somehow lost capacity is dropped rather than trusted.
            if buffer.capacity() >= class {
                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                return buffer;
            }
        }
        drop(classes);

        self.stats.misses.fetch_add(1, Ordering::Relaxed);
        ByteBufferMut::with_capacity_aligned(class, alignment)
    }

    fn put(&self, class: usize, alignment: Alignment, mut buffer: ByteBufferMut) {
        if self.pooled_bytes.load(Ordering::Relaxed) + class > self.config.max_pooled_bytes {
            return;
        }
        buffer.clear();

        let mut classes = self.classes.lock();
        let slot = classes.entry((class, *alignment)).or_default();
        if slot.len() < self.config.max_buffers_per_class {
            slot.push(buffer);
            self.pooled_bytes.fetch_add(class, Ordering::Relaxed);
        }
    }
}

impl PoolingHostAllocator {
    /// Create a pooling allocator with the given configuration.
    pub fn new(config: PoolConfig) -> Self {
        Self {
            inner: Arc::new(Pool {
                config,
                classes: Mutex::new(HashMap::default()),
                pooled_bytes: AtomicUsize::new(0),
                stats: PoolStats::default(),
            }),
        }
    }

    /// The number of allocations served from the pool.
    pub fn hits(&self) -> u64 {
        self.inner.stats.hits.load(Ordering::Relaxed)
    }

    /// The number of pool-eligible allocations that had to be served by the system allocator.
    pub fn misses(&self) -> u64 {
        self.inner.stats.misses.load(Ordering::Relaxed)
    }

    /// The number of bytes currently retained by the pool.
    pub fn pooled_bytes(&self) -> usize {
        self.inner.pooled_bytes.load(Ordering::Relaxed)
    }
}

impl Default for PoolingHostAllocator {
    fn default() -> Self {
        Self::new(PoolConfig::default())
    }
}

impl HostAllocator for PoolingHostAllocator {
    fn allocate(&self, len: usize, alignment: Alignment) -> VortexResult<WritableHostBuffer> {
        let Some(class) = self.inner.size_class(len) else {
            return DefaultHostAllocator.allocate(len, alignment);
        };

        let mut buffer = self.inner.take(class, alignment);
        // SAFETY: `take` returns a buffer with at least `class >= len` bytes of capacity, and the
        // caller fully initializes the slice before freezing it.
        unsafe { buffer.set_len(len) };

        Ok(WritableHostBuffer::new(Box::new(PooledHostBuffer {
            buffer,
            alignment,
            class,
            pool: Arc::clone(&self.inner),
        })))
    }
}

#[derive(Debug)]
struct PooledHostBuffer {
    buffer: ByteBufferMut,
    alignment: Alignment,
    class: usize,
    pool: Arc<Pool>,
}

impl HostBufferMut for PooledHostBuffer {
    fn len(&self) -> usize {
        self.buffer.len()
    }

    fn alignment(&self) -> Alignment {
        self.alignment
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        self.buffer.as_mut_slice()
    }

    fn freeze(self: Box<Self>) -> ByteBuffer {
        let Self {
            buffer,
            alignment,
            class,
            pool,
        } = *self;
        let bytes = Bytes::from_owner(PooledHostBufferOwner {
            buffer: Some(buffer),
            class,
            alignment,
            pool,
        });
        ByteBuffer::from_bytes_aligned(bytes, alignment)
    }
}

/// Owns a frozen pooled allocation, returning it to the pool once every slice of it is dropped.
#[derive(Debug)]
struct PooledHostBufferOwner {
    buffer: Option<ByteBufferMut>,
    class: usize,
    alignment: Alignment,
    pool: Arc<Pool>,
}

impl AsRef<[u8]> for PooledHostBufferOwner {
    fn as_ref(&self) -> &[u8] {
        self.buffer
            .as_ref()
            .map(ByteBufferMut::as_slice)
            .unwrap_or_default()
    }
}

impl Drop for PooledHostBufferOwner {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool.put(self.class, self.alignment, buffer);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use rstest::rstest;

    use super::*;

    #[derive(Debug)]
    struct CountingAllocator {
        allocations: Arc<AtomicUsize>,
    }

    impl HostAllocator for CountingAllocator {
        fn allocate(&self, len: usize, alignment: Alignment) -> VortexResult<WritableHostBuffer> {
            self.allocations.fetch_add(1, Ordering::Relaxed);
            DefaultHostAllocator.allocate(len, alignment)
        }
    }

    #[test]
    fn writable_host_buffer_freeze_round_trip() {
        let allocator = DefaultHostAllocator;
        let mut writable = allocator.allocate(16, Alignment::new(8)).unwrap();
        for (idx, byte) in writable.as_mut_slice().iter_mut().enumerate() {
            *byte = u8::try_from(idx).unwrap();
        }

        let host = writable.freeze();
        assert_eq!(host.len(), 16);
        assert!(host.is_aligned(Alignment::new(8)));
        assert_eq!(host.as_slice(), (0u8..16).collect::<Vec<_>>().as_slice());
    }

    #[test]
    fn memory_session_replaces_allocator() {
        let allocations = Arc::new(AtomicUsize::new(0));
        let allocator = Arc::new(CountingAllocator {
            allocations: Arc::clone(&allocations),
        });
        let mut session = MemorySession::default();
        session.set_allocator(allocator);
        drop(session.allocator().allocate(4, Alignment::none()).unwrap());
        assert_eq!(allocations.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn typed_allocation_uses_type_alignment() {
        let allocator = DefaultHostAllocator;
        let writable = allocator.allocate_typed::<u64>(4).unwrap();
        assert_eq!(writable.len(), 4 * size_of::<u64>());
        assert_eq!(writable.alignment(), Alignment::of::<u64>());
    }

    #[test]
    fn typed_mut_slice_round_trip() {
        let allocator = DefaultHostAllocator;
        let mut writable = allocator.allocate_typed::<u64>(4).unwrap();
        writable
            .as_mut_slice_typed::<u64>()
            .unwrap()
            .copy_from_slice(&[10, 20, 30, 40]);

        let frozen = writable.freeze();
        let values = unsafe {
            std::slice::from_raw_parts(
                frozen.as_slice().as_ptr().cast::<u64>(),
                frozen.len() / size_of::<u64>(),
            )
        };
        assert_eq!(values, [10, 20, 30, 40]);
    }

    #[test]
    fn typed_mut_slice_rejects_length_mismatch() {
        let allocator = DefaultHostAllocator;
        let mut writable = allocator.allocate(7, Alignment::none()).unwrap();
        assert!(writable.as_mut_slice_typed::<u32>().is_err());
    }

    #[test]
    fn freeze_typed_round_trip() {
        let allocator = DefaultHostAllocator;
        let mut writable = allocator.allocate_typed::<u64>(4).unwrap();
        writable
            .as_mut_slice_typed::<u64>()
            .unwrap()
            .copy_from_slice(&[1, 3, 5, 7]);

        let frozen = writable.freeze_typed::<u64>().unwrap();
        assert_eq!(frozen.as_slice(), [1, 3, 5, 7]);
    }

    #[test]
    fn freeze_typed_rejects_length_mismatch() {
        let allocator = DefaultHostAllocator;
        let writable = allocator.allocate(7, Alignment::none()).unwrap();
        let err = writable.freeze_typed::<u32>().unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("not a multiple of"));
    }

    fn write_and_freeze(
        allocator: &PoolingHostAllocator,
        len: usize,
        fill: u8,
    ) -> VortexResult<ByteBuffer> {
        let mut writable = allocator.allocate(len, Alignment::new(64))?;
        writable.as_mut_slice().fill(fill);
        Ok(writable.freeze())
    }

    #[test]
    fn pool_recycles_freed_buffers() -> VortexResult<()> {
        let allocator = PoolingHostAllocator::default();

        let first = write_and_freeze(&allocator, 1 << 20, 1)?;
        assert_eq!((allocator.hits(), allocator.misses()), (0, 1));
        drop(first);

        let second = write_and_freeze(&allocator, 1 << 20, 2)?;
        assert_eq!((allocator.hits(), allocator.misses()), (1, 1));
        assert!(second.as_slice().iter().all(|byte| *byte == 2));

        Ok(())
    }

    #[test]
    fn pool_retains_buffer_until_last_slice_drops() -> VortexResult<()> {
        let allocator = PoolingHostAllocator::default();

        let buffer = write_and_freeze(&allocator, 1 << 20, 7)?;
        let slice = buffer.slice(0..1024);
        drop(buffer);
        // The slice still references the allocation, so it must not have been recycled.
        assert_eq!(allocator.pooled_bytes(), 0);
        assert!(slice.as_slice().iter().all(|byte| *byte == 7));

        drop(slice);
        assert_eq!(allocator.pooled_bytes(), 1 << 20);

        Ok(())
    }

    #[rstest]
    #[case::below_min(1024)]
    #[case::above_max(64 * 1024 * 1024)]
    fn pool_skips_out_of_range_sizes(#[case] len: usize) -> VortexResult<()> {
        let allocator = PoolingHostAllocator::default();
        drop(write_and_freeze(&allocator, len, 3)?);
        assert_eq!(allocator.pooled_bytes(), 0);
        assert_eq!((allocator.hits(), allocator.misses()), (0, 0));
        Ok(())
    }

    #[test]
    fn pool_bounds_retained_buffers_per_class() -> VortexResult<()> {
        let allocator = PoolingHostAllocator::new(PoolConfig {
            max_buffers_per_class: 2,
            ..PoolConfig::default()
        });

        let buffers = (0..4)
            .map(|_| write_and_freeze(&allocator, 1 << 20, 4))
            .collect::<VortexResult<Vec<_>>>()?;
        drop(buffers);

        assert_eq!(allocator.pooled_bytes(), 2 << 20);
        Ok(())
    }

    #[test]
    fn pool_bounds_total_retained_bytes() -> VortexResult<()> {
        let allocator = PoolingHostAllocator::new(PoolConfig {
            max_pooled_bytes: 2 << 20,
            ..PoolConfig::default()
        });

        let buffers = (0..4)
            .map(|_| write_and_freeze(&allocator, 1 << 20, 5))
            .collect::<VortexResult<Vec<_>>>()?;
        drop(buffers);

        assert!(allocator.pooled_bytes() <= 2 << 20);
        Ok(())
    }

    #[test]
    fn pool_rounds_request_up_to_size_class() -> VortexResult<()> {
        let allocator = PoolingHostAllocator::default();

        // A 1.5MiB request lands in the 2MiB class, so a 2MiB request can reuse it.
        drop(write_and_freeze(&allocator, 1536 * 1024, 6)?);
        assert_eq!(allocator.pooled_bytes(), 2 << 20);

        let reused = write_and_freeze(&allocator, 2 << 20, 8)?;
        assert_eq!(allocator.hits(), 1);
        assert_eq!(reused.len(), 2 << 20);

        Ok(())
    }
}
