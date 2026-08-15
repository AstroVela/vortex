// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Minimal DuckDB scan used to compare caller-driven and threaded local-file I/O.

use std::alloc::Layout;
use std::collections::VecDeque;
use std::ffi::CString;
use std::ffi::c_void;
use std::fs::File;
use std::fs::OpenOptions;
use std::io;
use std::os::fd::AsRawFd;
use std::os::unix::fs::FileExt;
use std::os::unix::fs::OpenOptionsExt;
use std::path::PathBuf;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::Instant;

use io_uring::IoUring;
use io_uring::opcode;
use io_uring::types;
use parking_lot::Mutex;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex_utils::aliases::hash_map::HashMap;

use crate::cpp;
use crate::duckdb::ConnectionRef;
use crate::duckdb::DataChunk;
use crate::duckdb::ExtractedValue;
use crate::duckdb::LogicalType;
use crate::duckdb::Value;

const DIRECT_IO_ALIGNMENT: usize = 4096;

type BenchResult<T> = Result<T, String>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Engine {
    Pread,
    PerWorker,
    Threaded,
}

impl Engine {
    fn parse(value: &str) -> BenchResult<Self> {
        match value {
            "pread" => Ok(Self::Pread),
            "per-worker" => Ok(Self::PerWorker),
            "threaded" => Ok(Self::Threaded),
            other => Err(format!(
                "unknown I/O engine {other:?}; expected pread, per-worker, or threaded"
            )),
        }
    }
}

#[derive(Clone)]
struct Config {
    path: PathBuf,
    engine: Engine,
    rows: u64,
    block_rows: usize,
    prefetch: usize,
    direct: bool,
    workers: usize,
}

impl Config {
    fn block_bytes(&self) -> usize {
        self.block_rows * 2 * size_of::<i64>()
    }

    fn blocks(&self) -> u64 {
        self.rows.div_ceil(self.block_rows as u64)
    }

    fn rows_in_block(&self, block: u64) -> usize {
        usize::try_from(
            self.rows
                .saturating_sub(block * self.block_rows as u64)
                .min(self.block_rows as u64),
        )
        .unwrap_or(self.block_rows)
    }
}

struct BindData {
    config: Config,
    query: Mutex<Option<Arc<QueryState>>>,
}

struct QueryState {
    config: Config,
    file: Arc<File>,
    next_block: AtomicU64,
    buffers: Mutex<Vec<AlignedBuffer>>,
    metrics: Arc<Metrics>,
    threaded: Mutex<Option<Arc<ThreadedQueue>>>,
}

impl QueryState {
    fn claim_block(&self) -> BenchResult<Option<Block>> {
        let index = self.next_block.fetch_add(1, Ordering::Relaxed);
        if index >= self.config.blocks() {
            return Ok(None);
        }
        Ok(Some(Block {
            index,
            rows: self.config.rows_in_block(index),
            buffer: self
                .buffers
                .lock()
                .pop()
                .map_or_else(|| AlignedBuffer::new(self.config.block_bytes()), Ok)?,
            handoff_at: None,
        }))
    }

    fn recycle_buffer(&self, buffer: AlignedBuffer) {
        self.buffers.lock().push(buffer);
    }
}

#[derive(Default)]
struct Metrics {
    local_readers: AtomicU64,
    callbacks: AtomicU64,
    reads: AtomicU64,
    bytes: AtomicU64,
    submission_batches: AtomicU64,
    completions: AtomicU64,
    waits: AtomicU64,
    ready_hits: AtomicU64,
    callback_gap_ns: AtomicU64,
    max_callback_gap_ns: AtomicU64,
    submit_gap_ns: AtomicU64,
    max_submit_gap_ns: AtomicU64,
    last_submit_ns: AtomicU64,
    handoff_fast_receives: AtomicU64,
    handoff_fast_receive_ns: AtomicU64,
    handoff_fast_receive_max_ns: AtomicU64,
    handoff_wait_ns: AtomicU64,
    handoff_queue_ns: AtomicU64,
    handoff_queue_max_ns: AtomicU64,
    producer_send_ns: AtomicU64,
    producer_send_max_ns: AtomicU64,
    started_at: Mutex<Option<Instant>>,
}

impl Metrics {
    fn start(&self) {
        *self.started_at.lock() = Some(Instant::now());
    }

    fn record_callback_gap(&self, last: &mut Option<Instant>) {
        self.callbacks.fetch_add(1, Ordering::Relaxed);
        let now = Instant::now();
        if let Some(previous) = last.replace(now) {
            let gap = duration_ns(now.duration_since(previous));
            self.callback_gap_ns.fetch_add(gap, Ordering::Relaxed);
            self.max_callback_gap_ns.fetch_max(gap, Ordering::Relaxed);
        }
    }

    fn record_submission(&self) {
        self.submission_batches.fetch_add(1, Ordering::Relaxed);
        let Some(started) = *self.started_at.lock() else {
            return;
        };
        let now = duration_ns(started.elapsed());
        let previous = self.last_submit_ns.swap(now, Ordering::Relaxed);
        if previous != 0 {
            let gap = now.saturating_sub(previous);
            self.submit_gap_ns.fetch_add(gap, Ordering::Relaxed);
            self.max_submit_gap_ns.fetch_max(gap, Ordering::Relaxed);
        }
    }

    fn snapshot(&self) -> IoBenchMetrics {
        IoBenchMetrics {
            local_readers: self.local_readers.load(Ordering::Relaxed),
            callbacks: self.callbacks.load(Ordering::Relaxed),
            reads: self.reads.load(Ordering::Relaxed),
            bytes: self.bytes.load(Ordering::Relaxed),
            submission_batches: self.submission_batches.load(Ordering::Relaxed),
            completions: self.completions.load(Ordering::Relaxed),
            waits: self.waits.load(Ordering::Relaxed),
            ready_hits: self.ready_hits.load(Ordering::Relaxed),
            callback_gap_ns: self.callback_gap_ns.load(Ordering::Relaxed),
            max_callback_gap_ns: self.max_callback_gap_ns.load(Ordering::Relaxed),
            submit_gap_ns: self.submit_gap_ns.load(Ordering::Relaxed),
            max_submit_gap_ns: self.max_submit_gap_ns.load(Ordering::Relaxed),
            handoff_fast_receives: self.handoff_fast_receives.load(Ordering::Relaxed),
            handoff_fast_receive_ns: self.handoff_fast_receive_ns.load(Ordering::Relaxed),
            handoff_fast_receive_max_ns: self.handoff_fast_receive_max_ns.load(Ordering::Relaxed),
            handoff_wait_ns: self.handoff_wait_ns.load(Ordering::Relaxed),
            handoff_queue_ns: self.handoff_queue_ns.load(Ordering::Relaxed),
            handoff_queue_max_ns: self.handoff_queue_max_ns.load(Ordering::Relaxed),
            producer_send_ns: self.producer_send_ns.load(Ordering::Relaxed),
            producer_send_max_ns: self.producer_send_max_ns.load(Ordering::Relaxed),
        }
    }
}

fn duration_ns(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

/// Metrics from the most recently bound `io_bench_scan` query.
#[derive(Clone, Copy, Debug, Default)]
pub struct IoBenchMetrics {
    pub local_readers: u64,
    pub callbacks: u64,
    pub reads: u64,
    pub bytes: u64,
    pub submission_batches: u64,
    pub completions: u64,
    pub waits: u64,
    pub ready_hits: u64,
    pub callback_gap_ns: u64,
    pub max_callback_gap_ns: u64,
    pub submit_gap_ns: u64,
    pub max_submit_gap_ns: u64,
    pub handoff_fast_receives: u64,
    pub handoff_fast_receive_ns: u64,
    pub handoff_fast_receive_max_ns: u64,
    pub handoff_wait_ns: u64,
    pub handoff_queue_ns: u64,
    pub handoff_queue_max_ns: u64,
    pub producer_send_ns: u64,
    pub producer_send_max_ns: u64,
}

static LAST_METRICS: LazyLock<Mutex<Option<Arc<Metrics>>>> = LazyLock::new(Mutex::default);

/// Return a snapshot of metrics for the most recently bound benchmark scan.
pub fn io_bench_metrics() -> IoBenchMetrics {
    LAST_METRICS
        .lock()
        .as_ref()
        .map_or_else(IoBenchMetrics::default, |metrics| metrics.snapshot())
}

struct AlignedBuffer {
    pointer: NonNull<u8>,
    layout: Layout,
}

unsafe impl Send for AlignedBuffer {}

impl AlignedBuffer {
    fn new(length: usize) -> BenchResult<Self> {
        let layout = Layout::from_size_align(length, DIRECT_IO_ALIGNMENT)
            .map_err(|error| format!("invalid aligned I/O buffer layout: {error}"))?;
        let pointer = NonNull::new(unsafe { std::alloc::alloc(layout) })
            .ok_or_else(|| format!("failed to allocate {length} byte aligned I/O buffer"))?;
        Ok(Self { pointer, layout })
    }

    fn len(&self) -> usize {
        self.layout.size()
    }

    fn as_ptr(&self) -> *const u8 {
        self.pointer.as_ptr()
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.pointer.as_ptr()
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.len()) }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.pointer.as_ptr(), self.layout) }
    }
}

struct Block {
    index: u64,
    rows: usize,
    buffer: AlignedBuffer,
    handoff_at: Option<Instant>,
}

impl Block {
    fn offset(&self, config: &Config) -> u64 {
        self.index * config.block_bytes() as u64
    }
}

struct BlockCursor {
    block: Block,
    row: usize,
}

impl BlockCursor {
    fn emit(&mut self, config: &Config, output: &mut crate::duckdb::DataChunkRef) -> bool {
        let length = (self.block.rows - self.row).min(crate::duckdb::duckdb_vector_size());
        let values = unsafe {
            std::slice::from_raw_parts(
                self.block.buffer.as_ptr().cast::<i64>().add(self.row),
                length,
            )
        };
        let payload = unsafe {
            std::slice::from_raw_parts(
                self.block
                    .buffer
                    .as_ptr()
                    .cast::<i64>()
                    .add(config.block_rows + self.row),
                length,
            )
        };
        unsafe {
            output
                .get_vector_mut(0)
                .as_slice_mut::<i64>(length)
                .copy_from_slice(values);
            output
                .get_vector_mut(1)
                .as_slice_mut::<i64>(length)
                .copy_from_slice(payload);
        }
        output.set_len(length);
        self.row += length;
        self.row == self.block.rows
    }
}

struct GlobalState {
    _query: Arc<QueryState>,
    _threaded: Option<ThreadedDriver>,
}

enum LocalReader {
    Pread,
    PerWorker(Box<LocalUring>),
    Threaded(Arc<ThreadedQueue>),
}

struct LocalState {
    query: Arc<QueryState>,
    reader: LocalReader,
    current: Option<BlockCursor>,
    last_callback: Option<Instant>,
}

impl LocalState {
    fn next_block(&mut self) -> BenchResult<Option<Block>> {
        match &mut self.reader {
            LocalReader::Pread => {
                let Some(mut block) = self.query.claim_block()? else {
                    return Ok(None);
                };
                self.query.metrics.reads.fetch_add(1, Ordering::Relaxed);
                self.query
                    .metrics
                    .bytes
                    .fetch_add(block.buffer.len() as u64, Ordering::Relaxed);
                let offset = block.offset(&self.query.config);
                read_exact_at(&self.query.file, block.buffer.as_mut_slice(), offset)?;
                self.query
                    .metrics
                    .completions
                    .fetch_add(1, Ordering::Relaxed);
                Ok(Some(block))
            }
            LocalReader::PerWorker(reader) => reader.next_block(&self.query),
            LocalReader::Threaded(queue) => queue.pop(&self.query.metrics),
        }
    }

    fn scan(&mut self, output: &mut crate::duckdb::DataChunkRef) -> BenchResult<()> {
        self.query
            .metrics
            .record_callback_gap(&mut self.last_callback);

        if let LocalReader::PerWorker(reader) = &mut self.reader {
            reader.drive_nonblocking(&self.query)?;
        }

        if self.current.is_none() {
            self.current = self
                .next_block()?
                .map(|block| BlockCursor { block, row: 0 });
        }
        let Some(current) = self.current.as_mut() else {
            output.set_len(0);
            return Ok(());
        };
        if current.emit(&self.query.config, output)
            && let Some(cursor) = self.current.take()
        {
            self.query.recycle_buffer(cursor.block.buffer);
        }
        Ok(())
    }
}

fn read_exact_at(file: &File, mut buffer: &mut [u8], mut offset: u64) -> BenchResult<()> {
    while !buffer.is_empty() {
        let read = file
            .read_at(buffer, offset)
            .map_err(|error| format!("pread at offset {offset} failed: {error}"))?;
        if read == 0 {
            return Err(format!("pread reached EOF at offset {offset}"));
        }
        buffer = &mut buffer[read..];
        offset += read as u64;
    }
    Ok(())
}

struct LocalUring {
    ring: Option<IoUring>,
    pending: HashMap<u64, Block>,
    ready: VecDeque<Block>,
    completions: Vec<(u64, i32)>,
    next_id: u64,
}

impl LocalUring {
    fn new(depth: usize, mode: RingMode) -> BenchResult<Self> {
        Ok(Self {
            ring: Some(new_ring(depth, mode)?),
            pending: HashMap::with_capacity(depth),
            ready: VecDeque::with_capacity(depth),
            completions: Vec::with_capacity(depth),
            next_id: 1,
        })
    }

    fn ring(&mut self) -> BenchResult<&mut IoUring> {
        self.ring
            .as_mut()
            .ok_or_else(|| "io_uring was already closed".to_string())
    }

    fn drive_nonblocking(&mut self, query: &QueryState) -> BenchResult<()> {
        self.reap(query)?;
        let buffered = self.pending.len() + self.ready.len();
        let low_water = query.config.prefetch / 2;
        // Refill in half-window batches. Refilling after every pop turns a sequential scan into
        // one io_uring_enter call per block, while waiting for an empty window creates avoidable
        // consumer stalls.
        if buffered <= low_water {
            self.fill_to(query, query.config.prefetch)?;
        }
        Ok(())
    }

    fn next_block(&mut self, query: &QueryState) -> BenchResult<Option<Block>> {
        self.drive_nonblocking(query)?;
        if let Some(block) = self.ready.pop_front() {
            query.metrics.ready_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(Some(block));
        }
        if self.pending.is_empty() {
            return Ok(None);
        }
        query.metrics.waits.fetch_add(1, Ordering::Relaxed);
        self.ring()
            .and_then(|ring| ring.submit_and_wait(1).map_err(io_error))?;
        self.reap(query)?;
        self.fill_to(query, query.config.prefetch)?;
        self.ready
            .pop_front()
            .map(Some)
            .ok_or_else(|| "io_uring wait returned without a completion".to_string())
    }

    fn fill_to(&mut self, query: &QueryState, target: usize) -> BenchResult<()> {
        let mut submitted = 0;
        while self.pending.len() + self.ready.len() < target {
            let Some(mut block) = query.claim_block()? else {
                break;
            };
            let id = self.next_id;
            self.next_id = self.next_id.wrapping_add(1);
            let entry = opcode::Read::new(
                types::Fd(query.file.as_raw_fd()),
                block.buffer.as_mut_ptr(),
                u32::try_from(block.buffer.len())
                    .map_err(|error| format!("I/O block is too large: {error}"))?,
            )
            .offset(block.offset(&query.config))
            .build()
            .user_data(id);
            unsafe {
                self.ring()?
                    .submission()
                    .push(&entry)
                    .map_err(|_| "per-worker io_uring submission queue is full".to_string())?;
            }
            query.metrics.reads.fetch_add(1, Ordering::Relaxed);
            query
                .metrics
                .bytes
                .fetch_add(block.buffer.len() as u64, Ordering::Relaxed);
            self.pending.insert(id, block);
            submitted += 1;
        }
        if submitted != 0 {
            self.ring()?.submit().map_err(io_error)?;
            query.metrics.record_submission();
        }
        Ok(())
    }

    fn reap(&mut self, query: &QueryState) -> BenchResult<()> {
        self.completions.clear();
        let ring = self
            .ring
            .as_mut()
            .ok_or_else(|| "io_uring was already closed".to_string())?;
        self.completions.extend(
            ring.completion()
                .map(|completion| (completion.user_data(), completion.result())),
        );
        for (id, result) in self.completions.drain(..) {
            let block = self
                .pending
                .remove(&id)
                .ok_or_else(|| format!("unknown per-worker io_uring completion {id}"))?;
            check_completion(result, block.buffer.len())?;
            query.metrics.completions.fetch_add(1, Ordering::Relaxed);
            self.ready.push_back(block);
        }
        Ok(())
    }
}

impl Drop for LocalUring {
    fn drop(&mut self) {
        drop(self.ring.take());
        self.pending.clear();
    }
}

#[derive(Clone, Copy)]
enum RingMode {
    CallerDriven,
    DedicatedThread,
}

fn new_ring(depth: usize, mode: RingMode) -> BenchResult<IoUring> {
    let entries = u32::try_from(depth.max(2).next_power_of_two())
        .map_err(|error| format!("invalid io_uring depth {depth}: {error}"))?;
    let mut builder = IoUring::builder();
    builder.setup_single_issuer().setup_submit_all();
    if matches!(mode, RingMode::DedicatedThread) {
        // DEFER_TASKRUN requires periodic io_uring_enter calls. The dedicated driver waits in
        // enter continuously; a caller-driven ring may spend time in DuckDB and must not defer
        // completion delivery on that assumption.
        builder.setup_defer_taskrun();
    }
    builder
        .build(entries)
        .or_else(|_| IoUring::new(entries))
        .map_err(io_error)
}

fn io_error(error: io::Error) -> String {
    error.to_string()
}

fn check_completion(result: i32, expected: usize) -> BenchResult<()> {
    if result < 0 {
        return Err(io::Error::from_raw_os_error(-result).to_string());
    }
    if result as usize != expected {
        return Err(format!(
            "short io_uring read: expected {expected} bytes, received {result}"
        ));
    }
    Ok(())
}

struct ThreadedQueue {
    sender: kanal::Sender<Option<Block>>,
    receiver: kanal::Receiver<Option<Block>>,
    stop: AtomicBool,
    error: Mutex<Option<String>>,
    capacity: usize,
}

impl ThreadedQueue {
    fn new(capacity: usize) -> Self {
        let (sender, receiver) = kanal::bounded(capacity);
        Self {
            sender,
            receiver,
            stop: AtomicBool::new(false),
            error: Mutex::new(None),
            capacity,
        }
    }

    fn pop(&self, metrics: &Metrics) -> BenchResult<Option<Block>> {
        let started = Instant::now();
        match self.receiver.try_recv() {
            Ok(Some(Some(mut block))) => {
                let elapsed = duration_ns(started.elapsed());
                metrics
                    .handoff_fast_receives
                    .fetch_add(1, Ordering::Relaxed);
                metrics
                    .handoff_fast_receive_ns
                    .fetch_add(elapsed, Ordering::Relaxed);
                metrics
                    .handoff_fast_receive_max_ns
                    .fetch_max(elapsed, Ordering::Relaxed);
                Self::record_queue_time(metrics, &mut block);
                metrics.ready_hits.fetch_add(1, Ordering::Relaxed);
                Ok(Some(block))
            }
            Ok(Some(None)) => Ok(None),
            Ok(None) => {
                metrics.waits.fetch_add(1, Ordering::Relaxed);
                let mut result = self.receiver.recv().or_else(|_| self.closed());
                metrics
                    .handoff_wait_ns
                    .fetch_add(duration_ns(started.elapsed()), Ordering::Relaxed);
                if let Ok(Some(block)) = &mut result {
                    Self::record_queue_time(metrics, block);
                }
                result
            }
            Err(_) => self.closed(),
        }
    }

    fn record_queue_time(metrics: &Metrics, block: &mut Block) {
        if let Some(handoff_at) = block.handoff_at.take() {
            let elapsed = duration_ns(handoff_at.elapsed());
            metrics
                .handoff_queue_ns
                .fetch_add(elapsed, Ordering::Relaxed);
            metrics
                .handoff_queue_max_ns
                .fetch_max(elapsed, Ordering::Relaxed);
        }
    }

    fn closed(&self) -> BenchResult<Option<Block>> {
        self.error.lock().clone().map_or(Ok(None), Err)
    }

    fn close(&self) {
        self.stop.store(true, Ordering::Relaxed);
        drop(self.sender.close());
    }

    fn fail(&self, error: String) {
        *self.error.lock() = Some(error);
        self.close();
    }
}

struct ThreadedDriver {
    queue: Arc<ThreadedQueue>,
    thread: Option<thread::JoinHandle<()>>,
}

impl ThreadedDriver {
    fn start(query: Arc<QueryState>) -> BenchResult<Self> {
        let capacity = query
            .config
            .prefetch
            .saturating_mul(query.config.workers)
            .max(1);
        let queue = Arc::new(ThreadedQueue::new(capacity));
        let worker_queue = Arc::clone(&queue);
        let thread = thread::Builder::new()
            .name("vortex-duckdb-io-bench".to_string())
            .spawn(move || {
                if let Err(error) = run_threaded_reader(&query, &worker_queue) {
                    worker_queue.fail(error);
                }
            })
            .map_err(io_error)?;
        Ok(Self {
            queue,
            thread: Some(thread),
        })
    }
}

impl Drop for ThreadedDriver {
    fn drop(&mut self) {
        self.queue.close();
        if let Some(thread) = self.thread.take()
            && thread.join().is_err()
        {
            tracing::warn!("threaded io_uring benchmark driver panicked");
        }
    }
}

fn run_threaded_reader(query: &QueryState, queue: &ThreadedQueue) -> BenchResult<()> {
    let mut ring = LocalUring::new(queue.capacity, RingMode::DedicatedThread)?;
    loop {
        if queue.stop.load(Ordering::Relaxed) {
            return Ok(());
        }
        ring.reap(query)?;

        while let Some(mut block) = ring.ready.pop_front() {
            block.handoff_at = Some(Instant::now());
            let started = Instant::now();
            let sent = queue.sender.send(Some(block));
            let elapsed = duration_ns(started.elapsed());
            query
                .metrics
                .producer_send_ns
                .fetch_add(elapsed, Ordering::Relaxed);
            query
                .metrics
                .producer_send_max_ns
                .fetch_max(elapsed, Ordering::Relaxed);
            if sent.is_err() {
                return Ok(());
            }
        }

        ring.fill_to(query, queue.capacity)?;
        if ring.pending.is_empty() {
            for _ in 0..query.config.workers {
                if queue.sender.send(None).is_err() {
                    return Ok(());
                }
            }
            return Ok(());
        }
        ring.ring()?.submit_and_wait(1).map_err(io_error)?;
    }
}

fn open_file(config: &Config) -> BenchResult<File> {
    let mut options = OpenOptions::new();
    options.read(true);
    if config.direct {
        options.custom_flags(libc::O_DIRECT);
    }
    let file = options
        .open(&config.path)
        .map_err(|error| format!("failed to open {}: {error}", config.path.display()))?;
    let required = config.blocks() * config.block_bytes() as u64;
    let length = file
        .metadata()
        .map_err(|error| format!("failed to stat {}: {error}", config.path.display()))?
        .len();
    if length < required {
        return Err(format!(
            "{} is {length} bytes but the scan requires {required}",
            config.path.display()
        ));
    }
    Ok(file)
}

unsafe extern "C-unwind" fn drop_box<T>(pointer: *mut c_void) {
    if !pointer.is_null() {
        drop(unsafe { Box::from_raw(pointer.cast::<T>()) });
    }
}

unsafe extern "C-unwind" fn bind(info: cpp::duckdb_bind_info) {
    if let Err(error) = bind_inner(info) {
        set_bind_error(info, &error);
    }
}

fn bind_inner(info: cpp::duckdb_bind_info) -> BenchResult<()> {
    let config = Config {
        path: PathBuf::from(parameter_string(info, 0)?),
        engine: Engine::parse(&parameter_string(info, 1)?)?,
        rows: parameter_u64(info, 2)?,
        block_rows: parameter_usize(info, 3)?,
        prefetch: parameter_usize(info, 4)?.max(1),
        direct: parameter_bool(info, 5)?,
        workers: parameter_usize(info, 6)?.max(1),
    };
    if config.rows == 0 {
        return Err("rows must be non-zero".to_string());
    }
    if config.block_rows == 0 {
        return Err("block_rows must be non-zero".to_string());
    }
    if !config.block_bytes().is_multiple_of(DIRECT_IO_ALIGNMENT) {
        return Err(format!(
            "block size {} must be a multiple of {DIRECT_IO_ALIGNMENT}",
            config.block_bytes()
        ));
    }

    let bigint = LogicalType::new(cpp::DUCKDB_TYPE::DUCKDB_TYPE_BIGINT);
    let value = CString::new("value").map_err(|error| error.to_string())?;
    let payload = CString::new("payload").map_err(|error| error.to_string())?;
    unsafe {
        cpp::duckdb_bind_add_result_column(info, value.as_ptr(), bigint.as_ptr());
        cpp::duckdb_bind_add_result_column(info, payload.as_ptr(), bigint.as_ptr());
        cpp::duckdb_bind_set_cardinality(info, config.rows as _, true);
        cpp::duckdb_bind_set_bind_data(
            info,
            Box::into_raw(Box::new(BindData {
                config,
                query: Mutex::new(None),
            }))
            .cast(),
            Some(drop_box::<BindData>),
        );
    }
    Ok(())
}

unsafe extern "C-unwind" fn init_global(info: cpp::duckdb_init_info) {
    if let Err(error) = init_global_inner(info) {
        set_init_error(info, &error);
    }
}

fn init_global_inner(info: cpp::duckdb_init_info) -> BenchResult<()> {
    let bind = bind_data_from_init(info)?;
    let metrics = Arc::new(Metrics::default());
    metrics.start();
    *LAST_METRICS.lock() = Some(Arc::clone(&metrics));
    let query = Arc::new(QueryState {
        config: bind.config.clone(),
        file: Arc::new(open_file(&bind.config)?),
        next_block: AtomicU64::new(0),
        buffers: Mutex::new(Vec::new()),
        metrics,
        threaded: Mutex::new(None),
    });
    let threaded = if bind.config.engine == Engine::Threaded {
        let driver = ThreadedDriver::start(Arc::clone(&query))?;
        *query.threaded.lock() = Some(Arc::clone(&driver.queue));
        Some(driver)
    } else {
        None
    };
    *bind.query.lock() = Some(Arc::clone(&query));
    unsafe {
        cpp::duckdb_init_set_max_threads(info, bind.config.workers as _);
        cpp::duckdb_init_set_init_data(
            info,
            Box::into_raw(Box::new(GlobalState {
                _query: query,
                _threaded: threaded,
            }))
            .cast(),
            Some(drop_box::<GlobalState>),
        );
    }
    Ok(())
}

unsafe extern "C-unwind" fn init_local(info: cpp::duckdb_init_info) {
    if let Err(error) = init_local_inner(info) {
        set_init_error(info, &error);
    }
}

fn init_local_inner(info: cpp::duckdb_init_info) -> BenchResult<()> {
    let bind = bind_data_from_init(info)?;
    let query = bind
        .query
        .lock()
        .clone()
        .ok_or_else(|| "global scan state was not initialized".to_string())?;
    query.metrics.local_readers.fetch_add(1, Ordering::Relaxed);
    let reader = match query.config.engine {
        Engine::Pread => LocalReader::Pread,
        Engine::PerWorker => LocalReader::PerWorker(Box::new(LocalUring::new(
            query.config.prefetch,
            RingMode::CallerDriven,
        )?)),
        Engine::Threaded => LocalReader::Threaded(
            query
                .threaded
                .lock()
                .clone()
                .ok_or_else(|| "threaded reader was not initialized".to_string())?,
        ),
    };
    unsafe {
        cpp::duckdb_init_set_init_data(
            info,
            Box::into_raw(Box::new(LocalState {
                query,
                reader,
                current: None,
                last_callback: None,
            }))
            .cast(),
            Some(drop_box::<LocalState>),
        );
    }
    Ok(())
}

unsafe extern "C-unwind" fn scan(info: cpp::duckdb_function_info, output: cpp::duckdb_data_chunk) {
    let result = (|| {
        let local = unsafe { cpp::duckdb_function_get_local_init_data(info) }.cast::<LocalState>();
        let local =
            unsafe { local.as_mut() }.ok_or_else(|| "local scan state is null".to_string())?;
        let output = unsafe { DataChunk::borrow_mut(output) };
        local.scan(output)
    })();
    if let Err(error) = result {
        set_function_error(info, &error);
    }
}

fn bind_data_from_init(info: cpp::duckdb_init_info) -> BenchResult<&'static BindData> {
    let pointer = unsafe { cpp::duckdb_init_get_bind_data(info) }.cast::<BindData>();
    unsafe { pointer.as_ref() }.ok_or_else(|| "bind data is null".to_string())
}

fn parameter(info: cpp::duckdb_bind_info, index: usize) -> BenchResult<Value> {
    let pointer = unsafe { cpp::duckdb_bind_get_parameter(info, index as _) };
    if pointer.is_null() {
        return Err(format!("parameter {index} is null"));
    }
    Ok(unsafe { Value::own(pointer) })
}

fn parameter_string(info: cpp::duckdb_bind_info, index: usize) -> BenchResult<String> {
    match parameter(info, index)?.extract() {
        ExtractedValue::Varchar(value) => Ok(value.as_str().to_string()),
        value => Err(format!("parameter {index} must be VARCHAR, got {value:?}")),
    }
}

fn parameter_u64(info: cpp::duckdb_bind_info, index: usize) -> BenchResult<u64> {
    match parameter(info, index)?.extract() {
        ExtractedValue::UBigInt(value) => Ok(value),
        value => Err(format!("parameter {index} must be UBIGINT, got {value:?}")),
    }
}

fn parameter_usize(info: cpp::duckdb_bind_info, index: usize) -> BenchResult<usize> {
    usize::try_from(parameter_u64(info, index)?)
        .map_err(|error| format!("parameter {index} does not fit usize: {error}"))
}

fn parameter_bool(info: cpp::duckdb_bind_info, index: usize) -> BenchResult<bool> {
    match parameter(info, index)?.extract() {
        ExtractedValue::Boolean(value) => Ok(value),
        value => Err(format!("parameter {index} must be BOOLEAN, got {value:?}")),
    }
}

fn c_error(error: &str) -> CString {
    CString::new(error).unwrap_or_default()
}

fn set_bind_error(info: cpp::duckdb_bind_info, error: &str) {
    let error = c_error(error);
    unsafe { cpp::duckdb_bind_set_error(info, error.as_ptr()) }
}

fn set_init_error(info: cpp::duckdb_init_info, error: &str) {
    let error = c_error(error);
    unsafe { cpp::duckdb_init_set_error(info, error.as_ptr()) }
}

fn set_function_error(info: cpp::duckdb_function_info, error: &str) {
    let error = c_error(error);
    unsafe { cpp::duckdb_function_set_error(info, error.as_ptr()) }
}

impl ConnectionRef {
    /// Register the Linux-only `io_bench_scan` table function.
    pub fn register_io_bench_scan(&self) -> VortexResult<()> {
        let function = unsafe { cpp::duckdb_create_table_function() };
        if function.is_null() {
            vortex_bail!("failed to create io_bench_scan table function");
        }
        let name = c"io_bench_scan";
        let varchar = LogicalType::new(cpp::DUCKDB_TYPE::DUCKDB_TYPE_VARCHAR);
        let ubigint = LogicalType::new(cpp::DUCKDB_TYPE::DUCKDB_TYPE_UBIGINT);
        let boolean = LogicalType::new(cpp::DUCKDB_TYPE::DUCKDB_TYPE_BOOLEAN);
        unsafe {
            cpp::duckdb_table_function_set_name(function, name.as_ptr());
            cpp::duckdb_table_function_add_parameter(function, varchar.as_ptr());
            cpp::duckdb_table_function_add_parameter(function, varchar.as_ptr());
            cpp::duckdb_table_function_add_parameter(function, ubigint.as_ptr());
            cpp::duckdb_table_function_add_parameter(function, ubigint.as_ptr());
            cpp::duckdb_table_function_add_parameter(function, ubigint.as_ptr());
            cpp::duckdb_table_function_add_parameter(function, boolean.as_ptr());
            cpp::duckdb_table_function_add_parameter(function, ubigint.as_ptr());
            cpp::duckdb_table_function_set_bind(function, Some(bind));
            cpp::duckdb_table_function_set_init(function, Some(init_global));
            cpp::duckdb_table_function_set_local_init(function, Some(init_local));
            cpp::duckdb_table_function_set_function(function, Some(scan));
            let status = cpp::duckdb_register_table_function(self.as_ptr(), function);
            let mut function = function;
            cpp::duckdb_destroy_table_function(&raw mut function);
            if status != cpp::duckdb_state::DuckDBSuccess {
                vortex_bail!("failed to register io_bench_scan table function");
            }
        }
        Ok(())
    }
}
