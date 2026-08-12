// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Allocation instrumentation for the file scan path.
//!
//! Installs a global allocator shim that records every allocation above a size threshold and
//! attributes it to the Vortex call site that requested it, then reports the largest sites for a
//! full file scan. Finally it A/B benchmarks the same scan under [`DefaultHostAllocator`] and
//! [`PoolingHostAllocator`], opening the file fresh on each iteration so the segment cache does
//! not hide the I/O read path.
//!
//! ```text
//! cargo run --release -p vortex --example alloc_profile
//! ```
//!
//! `VORTEX_ALLOC_PROFILE_CHUNKS` controls the size of the generated file.

#![allow(clippy::print_stdout, clippy::cast_precision_loss, clippy::unwrap_used)]

use std::alloc::GlobalAlloc;
use std::alloc::Layout;
use std::alloc::System;
use std::backtrace::Backtrace;
use std::cell::Cell;
use std::path::Path;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Duration;
use std::time::Instant;

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
use vortex::array::memory::DefaultHostAllocator;
use vortex::array::memory::HostAllocator;
use vortex::array::memory::MemorySessionExt;
use vortex::array::memory::PoolingHostAllocator;
use vortex::buffer::Alignment;
use vortex::buffer::Buffer;
use vortex::buffer::ByteBufferMut;
use vortex::error::VortexResult;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::VortexFile;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;

/// Only allocations at least this large are recorded.
const THRESHOLD: usize = 256 * 1024;

static ENABLED: AtomicBool = AtomicBool::new(false);
static TOTAL_BYTES: AtomicU64 = AtomicU64::new(0);
static TOTAL_COUNT: AtomicU64 = AtomicU64::new(0);
/// Nanoseconds spent inside the system allocator for blocks at or above [`THRESHOLD`]. This is the
/// exact ceiling on what a buffer pool for those blocks could save.
static ALLOC_NANOS: AtomicU64 = AtomicU64::new(0);
static FREE_NANOS: AtomicU64 = AtomicU64::new(0);

#[derive(Default)]
struct Site {
    count: u64,
    bytes: u64,
}

static SITES: LazyLock<Mutex<HashMap<String, Site>>> =
    LazyLock::new(|| Mutex::new(HashMap::default()));

thread_local! {
    /// Guards against re-entering the recorder from the allocations the recorder itself makes.
    static RECORDING: Cell<bool> = const { Cell::new(false) };
}

struct Profiler;

impl Profiler {
    fn timing(size: usize) -> bool {
        size >= THRESHOLD && ENABLED.load(Ordering::Relaxed)
    }

    fn record(size: usize) {
        if size < THRESHOLD || !ENABLED.load(Ordering::Relaxed) {
            return;
        }
        if RECORDING.with(|recording| recording.replace(true)) {
            return;
        }

        TOTAL_BYTES.fetch_add(size as u64, Ordering::Relaxed);
        TOTAL_COUNT.fetch_add(1, Ordering::Relaxed);

        let site = summarize(&Backtrace::force_capture().to_string());
        let mut sites = SITES.lock();
        let entry = sites.entry(site).or_default();
        entry.count += 1;
        entry.bytes += size as u64;
        drop(sites);

        RECORDING.with(|recording| recording.set(false));
    }
}

/// Keep the first few `vortex` frames of the backtrace: enough to identify the call site without
/// splitting one logical site across many async-poll spellings.
fn summarize(backtrace: &str) -> String {
    backtrace
        .lines()
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
        if !Self::timing(layout.size()) {
            return unsafe { System.alloc(layout) };
        }
        let start = Instant::now();
        let ptr = unsafe { System.alloc(layout) };
        ALLOC_NANOS.fetch_add(
            start.elapsed().as_nanos().try_into().unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if !Self::timing(layout.size()) {
            unsafe { System.dealloc(ptr, layout) };
            return;
        }
        let start = Instant::now();
        unsafe { System.dealloc(ptr, layout) };
        FREE_NANOS.fetch_add(
            start.elapsed().as_nanos().try_into().unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
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

const CHUNK: usize = 65_536;

fn chunks() -> usize {
    std::env::var("VORTEX_ALLOC_PROFILE_CHUNKS")
        .ok()
        .and_then(|chunks| chunks.parse().ok())
        .unwrap_or(256)
}

fn sample_data(chunks: usize) -> VortexResult<ArrayRef> {
    let ints = ChunkedArray::from_iter((0..chunks).map(|chunk| {
        Buffer::<i64>::from_iter((0..CHUNK).map(|i| ((chunk * CHUNK + i) % 100_000) as i64))
            .into_array()
    }))
    .into_array();

    let strings = ChunkedArray::from_iter((0..chunks).map(|chunk| {
        VarBinArray::from_iter(
            (0..CHUNK).map(|i| Some(format!("value-{}-{}", chunk, i % 5_000))),
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
    let batch = array.execute::<StructArray>(&mut ctx)?;
    let ints = batch
        .unmasked_field_by_name("ints")?
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?;
    let strings = batch
        .unmasked_field_by_name("strings")?
        .clone()
        .execute::<VarBinViewArray>(&mut ctx)?;
    std::hint::black_box((ints, strings));
    Ok(())
}

async fn scan_file(session: &VortexSession, file: &VortexFile) -> VortexResult<()> {
    let mut stream = Box::pin(file.scan()?.into_array_stream()?);
    while let Some(array) = stream.next().await {
        consume(session, array?)?;
    }
    Ok(())
}

/// Time one open-and-scan. The file is reopened per iteration so that the segment cache does not
/// serve the reads that the host allocator is responsible for.
async fn timed_cold_scan(session: &VortexSession, path: &Path) -> VortexResult<Duration> {
    let start = Instant::now();
    let file = session.open_options().open_path(path).await?;
    scan_file(session, &file).await?;
    Ok(start.elapsed())
}

/// Minor page faults taken by this process so far.
///
/// A fresh mapping serves its first write with a fault; a recycled buffer is already mapped and
/// faulted in. The fault delta across a scan is therefore an upper bound on what any buffer pool
/// can remove.
fn minor_faults() -> u64 {
    std::fs::read_to_string("/proc/self/stat")
        .ok()
        .and_then(|stat| {
            stat.split_whitespace()
                .nth(9)
                .and_then(|faults| faults.parse().ok())
        })
        .unwrap_or_default()
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> VortexResult<()> {
    let chunks = chunks();
    let session = VortexSession::default().with_tokio();

    let mut buf = ByteBufferMut::empty();
    session
        .write_options()
        .write(&mut buf, sample_data(chunks)?.to_array_stream())
        .await?;
    let buf = buf.freeze();
    let file_size = buf.len();
    println!(
        "file: {} rows, {:.1} MiB",
        chunks * CHUNK,
        file_size as f64 / (1024.0 * 1024.0)
    );

    let path = std::env::temp_dir().join("vortex_alloc_profile.vortex");
    std::fs::write(&path, buf.as_ref())?;
    drop(buf);

    // 1. Attribute the large allocations of a scan to their call sites.
    let file = session.open_options().open_path(&path).await?;
    scan_file(&session, &file).await?;

    ENABLED.store(true, Ordering::SeqCst);
    scan_file(&session, &file).await?;
    ENABLED.store(false, Ordering::SeqCst);
    drop(file);

    println!(
        "time inside the system allocator for those blocks: {:.2}ms alloc + {:.2}ms free",
        ALLOC_NANOS.load(Ordering::Relaxed) as f64 / 1e6,
        FREE_NANOS.load(Ordering::Relaxed) as f64 / 1e6,
    );
    println!(
        "\n{} allocations >= {}KiB, {:.1} MiB total\n",
        TOTAL_COUNT.load(Ordering::Relaxed),
        THRESHOLD / 1024,
        TOTAL_BYTES.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0)
    );

    let mut ranked: Vec<_> = SITES
        .lock()
        .iter()
        .map(|(name, site)| (name.clone(), site.count, site.bytes))
        .collect();
    ranked.sort_by_key(|(_, _, bytes)| std::cmp::Reverse(*bytes));
    for (name, count, bytes) in ranked.iter().take(10) {
        println!(
            "{:>6} allocs {:>9.1} MiB (avg {:>6.0} KiB)\n    {}\n",
            count,
            *bytes as f64 / (1024.0 * 1024.0),
            *bytes as f64 / *count as f64 / 1024.0,
            name
        );
    }

    // 2. Bound the prize: how many page faults does a scan actually take?
    let file = session.open_options().open_path(&path).await?;
    scan_file(&session, &file).await?;
    let before = minor_faults();
    let start = Instant::now();
    scan_file(&session, &file).await?;
    let elapsed = start.elapsed();
    let faults = minor_faults() - before;
    println!(
        "warm scan: {:.1}ms, {} minor faults ({:.1} MiB faulted, ~{:.2}ms at 0.5us/fault, {:.1}% of scan)",
        elapsed.as_secs_f64() * 1e3,
        faults,
        (faults * 4096) as f64 / (1024.0 * 1024.0),
        faults as f64 * 0.5 / 1e3,
        (faults as f64 * 0.5e-6) / elapsed.as_secs_f64() * 100.0,
    );
    drop(file);

    // 3. A/B the same cold-cache scan under both host allocators.
    let pool = Arc::new(PoolingHostAllocator::default());
    let pooled = VortexSession::default()
        .with_tokio()
        .with_allocator(Arc::clone(&pool) as _);
    let unpooled = VortexSession::default()
        .with_tokio()
        .with_allocator(Arc::new(DefaultHostAllocator));

    const ITERATIONS: usize = 20;
    let mut pooled_samples = Vec::with_capacity(ITERATIONS);
    let mut unpooled_samples = Vec::with_capacity(ITERATIONS);

    // Warm the page cache and fill the pool before measuring.
    for _ in 0..3 {
        timed_cold_scan(&pooled, &path).await?;
        timed_cold_scan(&unpooled, &path).await?;
    }

    // Alternate so that drift affects both configurations equally.
    for _ in 0..ITERATIONS {
        unpooled_samples.push(timed_cold_scan(&unpooled, &path).await?);
        pooled_samples.push(timed_cold_scan(&pooled, &path).await?);
    }

    let unpooled_median = median(unpooled_samples);
    let pooled_median = median(pooled_samples);
    let throughput =
        |elapsed: Duration| file_size as f64 / (1024.0 * 1024.0) / elapsed.as_secs_f64() / 1024.0;
    println!(
        "cold-cache scan (median of {ITERATIONS}):\n  \
         default {:>8.2}ms  {:>5.2} GiB/s\n  \
         pooled  {:>8.2}ms  {:>5.2} GiB/s  ({:+.1}%, pool hits {}, misses {}, retained {:.0} MiB)",
        unpooled_median.as_secs_f64() * 1e3,
        throughput(unpooled_median),
        pooled_median.as_secs_f64() * 1e3,
        throughput(pooled_median),
        (unpooled_median.as_secs_f64() / pooled_median.as_secs_f64() - 1.0) * 100.0,
        pool.hits(),
        pool.misses(),
        pool.pooled_bytes() as f64 / (1024.0 * 1024.0),
    );

    // 4. Isolate the raw per-allocation cost the pool removes.
    const ALLOCATIONS: usize = 2_000;
    const SIZE: usize = 4 << 20;
    let raw = |allocator: &dyn HostAllocator| -> VortexResult<Duration> {
        let start = Instant::now();
        for _ in 0..ALLOCATIONS {
            let mut writable = allocator.allocate(SIZE, Alignment::new(64))?;
            // Touch every page: an mmap-backed allocation faults here, a recycled one does not.
            writable.as_mut_slice().fill(1);
            drop(writable.freeze());
        }
        Ok(start.elapsed())
    };
    let raw_default = raw(&DefaultHostAllocator)?;
    let raw_pool = raw(&PoolingHostAllocator::default())?;
    println!(
        "raw {SIZE}B alloc+touch+free x{ALLOCATIONS}: default {:.2}ms ({:.1}us each), pooled {:.2}ms ({:.1}us each)",
        raw_default.as_secs_f64() * 1e3,
        raw_default.as_secs_f64() * 1e6 / ALLOCATIONS as f64,
        raw_pool.as_secs_f64() * 1e3,
        raw_pool.as_secs_f64() * 1e6 / ALLOCATIONS as f64,
    );

    std::fs::remove_file(&path)?;
    Ok(())
}
