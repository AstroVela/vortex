// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Instrumentation harness that attributes large (>= 256KiB) heap allocations made during a
//! file scan to the Rust call site that requested them.
//!
//! Run with:
//! ```text
//! cargo run --release -p vortex-file --example alloc_profile
//! ```

#![allow(clippy::print_stdout, clippy::cast_precision_loss, clippy::unwrap_used)]

use std::alloc::GlobalAlloc;
use std::alloc::Layout;
use std::alloc::System;
use std::backtrace::Backtrace;
use std::cell::Cell;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use bytes::Bytes;
use futures::StreamExt;
use parking_lot::Mutex;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::array::dtype::DType;
use vortex::array::dtype::Nullability;
use vortex::array::memory::HostAllocator;
use vortex::array::memory::HostBufferMut;
use vortex::array::memory::MemorySessionExt;
use vortex::array::memory::WritableHostBuffer;
use vortex::buffer::Alignment;
use vortex::buffer::Buffer;
use vortex::buffer::ByteBuffer;
use vortex::buffer::ByteBufferMut;
use vortex::error::VortexResult;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;

/// Only allocations at least this large are recorded.
const THRESHOLD: usize = 256 * 1024;

static ENABLED: AtomicBool = AtomicBool::new(false);
static TOTAL_BYTES: AtomicU64 = AtomicU64::new(0);
static TOTAL_COUNT: AtomicU64 = AtomicU64::new(0);
static BATCHES: AtomicU64 = AtomicU64::new(0);
static ROWS: AtomicU64 = AtomicU64::new(0);
static ALL_COUNT: AtomicU64 = AtomicU64::new(0);
static BIG4K: AtomicU64 = AtomicU64::new(0);

struct Site {
    count: u64,
    bytes: u64,
}

static SITES: LazyLock<Mutex<HashMap<String, Site>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

thread_local! {
    /// Guards against re-entering the recorder from the allocations the recorder itself makes.
    static RECORDING: Cell<bool> = const { Cell::new(false) };
}

struct Profiler;

impl Profiler {
    fn record(size: usize) {
        if ENABLED.load(Ordering::Relaxed) {
            ALL_COUNT.fetch_add(1, Ordering::Relaxed);
            if size >= 4096 {
                BIG4K.fetch_add(1, Ordering::Relaxed);
            }
        }
        if size < THRESHOLD || !ENABLED.load(Ordering::Relaxed) {
            return;
        }
        let reentrant = RECORDING.with(|r| r.replace(true));
        if reentrant {
            return;
        }

        TOTAL_BYTES.fetch_add(size as u64, Ordering::Relaxed);
        TOTAL_COUNT.fetch_add(1, Ordering::Relaxed);

        let bt = Backtrace::force_capture().to_string();
        let key = summarize(&bt);
        let mut sites = SITES.lock();
        let entry = sites.entry(key).or_insert(Site { count: 0, bytes: 0 });
        entry.count += 1;
        entry.bytes += size as u64;
        drop(sites);

        RECORDING.with(|r| r.set(false));
    }
}

/// Keep the first few `vortex` frames of the backtrace: enough to identify the call site without
/// splitting one logical site across many async-poll spellings.
fn summarize(bt: &str) -> String {
    bt.lines()
        .filter_map(|line| {
            let line = line.trim();
            let frame = line.strip_prefix("at ").unwrap_or(line);
            frame
                .split_once(": ")
                .map(|(_, name)| name.trim().to_string())
                .filter(|name| name.starts_with("vortex") || name.starts_with("<vortex"))
        })
        .filter(|name| !name.contains("alloc_profile"))
        .take(6)
        .collect::<Vec<_>>()
        .join("\n    ")
}

unsafe impl GlobalAlloc for Profiler {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        Self::record(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if new_size > layout.size() {
            Self::record(new_size);
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        Self::record(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
}

#[global_allocator]
static GLOBAL: Profiler = Profiler;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| VortexSession::default().with_tokio());

/// Prototype pooling host allocator: recycles aligned byte buffers by (size-class, alignment)
/// instead of returning them to the system allocator.
#[derive(Debug, Default)]
struct PoolingHostAllocator {
    classes: Mutex<HashMap<(usize, usize), Vec<ByteBufferMut>>>,
    hits: AtomicU64,
    misses: AtomicU64,
}

/// Round a request up to the next power of two so that similarly-sized reads share a class.
fn size_class(len: usize) -> usize {
    len.next_power_of_two().max(4096)
}

impl PoolingHostAllocator {
    fn take(&self, class: usize, alignment: Alignment) -> ByteBufferMut {
        let mut classes = self.classes.lock();
        if let Some(buf) = classes
            .get_mut(&(class, *alignment))
            .and_then(|slot| slot.pop())
        {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return buf;
        }
        drop(classes);
        self.misses.fetch_add(1, Ordering::Relaxed);
        ByteBufferMut::with_capacity_aligned(class, alignment)
    }

    fn put(&self, class: usize, alignment: Alignment, mut buffer: ByteBufferMut) {
        buffer.clear();
        let mut classes = self.classes.lock();
        let slot = classes.entry((class, *alignment)).or_default();
        // Bound the pool so a burst of wide reads cannot pin memory forever.
        if slot.len() < 32 {
            slot.push(buffer);
        }
    }
}

impl HostAllocator for PoolingHostAllocator {
    fn allocate(&self, len: usize, alignment: Alignment) -> VortexResult<WritableHostBuffer> {
        let class = size_class(len);
        let mut buffer = self.take(class, alignment);
        // SAFETY: `take` returns a buffer with at least `class >= len` bytes of capacity, and the
        // caller fully initializes the slice before freezing it.
        unsafe { buffer.set_len(len) };
        Ok(WritableHostBuffer::new(Box::new(PooledBuffer {
            buffer,
            alignment,
            class,
            pool: POOL.clone(),
        })))
    }
}

struct PooledBuffer {
    buffer: ByteBufferMut,
    alignment: Alignment,
    class: usize,
    pool: Arc<PoolingHostAllocator>,
}

impl HostBufferMut for PooledBuffer {
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
        let bytes = Bytes::from_owner(PooledOwner {
            buffer: Some(buffer),
            class,
            alignment,
            pool,
        });
        ByteBuffer::from_bytes_aligned(bytes, alignment)
    }
}

/// Owns the frozen allocation: the buffer returns to the pool when the last slice is dropped.
struct PooledOwner {
    buffer: Option<ByteBufferMut>,
    class: usize,
    alignment: Alignment,
    pool: Arc<PoolingHostAllocator>,
}

impl AsRef<[u8]> for PooledOwner {
    fn as_ref(&self) -> &[u8] {
        self.buffer
            .as_ref()
            .map(|b| b.as_slice())
            .unwrap_or_default()
    }
}

impl Drop for PooledOwner {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool.put(self.class, self.alignment, buffer);
        }
    }
}

static POOL: LazyLock<Arc<PoolingHostAllocator>> =
    LazyLock::new(|| Arc::new(PoolingHostAllocator::default()));

const CHUNK: usize = 65_536;
const CHUNKS: usize = 64;

fn sample_data() -> VortexResult<ArrayRef> {
    let ints = ChunkedArray::from_iter((0..CHUNKS).map(|c| {
        Buffer::<i64>::from_iter((0..CHUNK).map(|i| ((c * CHUNK + i) % 100_000) as i64))
            .into_array()
    }))
    .into_array();

    let strings = ChunkedArray::from_iter((0..CHUNKS).map(|c| {
        VarBinArray::from_iter(
            (0..CHUNK).map(|i| Some(format!("value-{}-{}", c, i % 5_000))),
            DType::Utf8(Nullability::Nullable),
        )
        .into_array()
    }))
    .into_array();

    Ok(StructArray::from_fields(&[("ints", ints), ("strings", strings)])?.into_array())
}

/// Fully canonicalize a scan batch so decode work (and its allocations) actually happens.
fn consume(session: &VortexSession, array: ArrayRef) -> VortexResult<()> {
    let mut ctx = session.create_execution_ctx();
    let st = array.execute::<StructArray>(&mut ctx)?;
    let ints = st
        .unmasked_field_by_name("ints")?
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?;
    let strings = st
        .unmasked_field_by_name("strings")?
        .clone()
        .execute::<VarBinViewArray>(&mut ctx)?;
    std::hint::black_box((ints, strings));
    Ok(())
}

async fn time_scans(
    session: &VortexSession,
    file: &vortex::file::VortexFile,
    iterations: usize,
) -> VortexResult<std::time::Duration> {
    // Warm up once so neither configuration pays page-cache or pool-fill costs in the measurement.
    for _ in 0..(iterations + 1) {
        let mut stream = Box::pin(file.scan()?.into_array_stream()?);
        while let Some(array) = stream.next().await {
            consume(session, array?)?;
        }
    }

    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let mut stream = Box::pin(file.scan()?.into_array_stream()?);
        while let Some(array) = stream.next().await {
            consume(session, array?)?;
        }
    }
    Ok(start.elapsed())
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> VortexResult<()> {
    let data = sample_data()?;

    let mut buf = ByteBufferMut::empty();
    SESSION
        .write_options()
        .write(&mut buf, data.to_array_stream())
        .await?;
    let buf = buf.freeze();
    println!("file size: {:.1} MiB", buf.len() as f64 / (1024.0 * 1024.0));

    let path = std::env::temp_dir().join("vortex_alloc_profile.vortex");
    std::fs::write(&path, buf.as_ref())?;
    let file = SESSION.open_options().open_path(&path).await?;

    // Warm up, then profile the scan only.
    for _ in 0..2 {
        let mut stream = Box::pin(file.scan()?.into_array_stream()?);
        while let Some(array) = stream.next().await {
            consume(&SESSION, array?)?;
        }
    }

    ENABLED.store(true, Ordering::SeqCst);
    for _ in 0..5 {
        let mut stream = Box::pin(file.scan()?.into_array_stream()?);
        while let Some(array) = stream.next().await {
            let array = array?;
            BATCHES.fetch_add(1, Ordering::Relaxed);
            ROWS.fetch_add(array.len() as u64, Ordering::Relaxed);
            consume(&SESSION, array)?;
        }
    }
    ENABLED.store(false, Ordering::SeqCst);
    println!(
        "batches: {}, rows: {}, allocs: {}, >=4K: {}",
        BATCHES.load(Ordering::Relaxed),
        ROWS.load(Ordering::Relaxed),
        ALL_COUNT.load(Ordering::Relaxed),
        BIG4K.load(Ordering::Relaxed)
    );

    // A/B: default system allocator vs. the pooled allocator, same workload.
    let pooled_session = VortexSession::default()
        .with_tokio()
        .with_allocator(POOL.clone());
    let pooled_file = pooled_session.open_options().open_path(&path).await?;

    let default_time = time_scans(&SESSION, &file, 5).await?;
    let pooled_time = time_scans(&pooled_session, &pooled_file, 5).await?;
    println!(
        "scan x5: default {:.2}s, pooled {:.2}s (pool hits {}, misses {})",
        default_time.as_secs_f64(),
        pooled_time.as_secs_f64(),
        POOL.hits.load(Ordering::Relaxed),
        POOL.misses.load(Ordering::Relaxed)
    );

    let sites = SITES.lock();
    let mut ranked: Vec<_> = sites.iter().collect();
    ranked.sort_by_key(|(_, s)| std::cmp::Reverse(s.bytes));

    println!(
        "\n{} allocations >= {}KiB, {:.1} MiB total\n",
        TOTAL_COUNT.load(Ordering::Relaxed),
        THRESHOLD / 1024,
        TOTAL_BYTES.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0)
    );

    for (site, stat) in ranked.iter().take(25) {
        println!(
            "{:>6} allocs {:>9.1} MiB (avg {:>6.0} KiB)\n    {}\n",
            stat.count,
            stat.bytes as f64 / (1024.0 * 1024.0),
            stat.bytes as f64 / stat.count as f64 / 1024.0,
            site
        );
    }

    Ok(())
}
