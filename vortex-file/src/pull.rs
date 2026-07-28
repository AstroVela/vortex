// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pull-based (inverted-IO) scanning: the caller performs all reads.
//!
//! The types in this module invert the IO model of [`VortexFile`]: instead of Vortex issuing
//! reads through a [`VortexReadAt`](vortex_io::VortexReadAt) source driven by an async runtime,
//! a [`PullScan`] tells the caller which byte ranges of the file it needs, and the caller reads
//! them with whatever IO machinery it owns (a scheduler thread pool, io_uring, an external
//! engine's file system). Decoding happens on the caller's thread inside
//! [`advance`](PullScan::advance); no work is spawned onto a runtime.
//!
//! The protocol is a resumable coroutine:
//!
//! 1. [`PullScan::advance`] returns [`PullEvent::Reads`]: byte ranges with pre-allocated,
//!    correctly aligned destination buffers. The caller fills each buffer with the bytes at
//!    `[offset, offset + len)` and hands it back via [`PullScan::complete`]. Completions may
//!    arrive in any order, so many reads can be in flight at once.
//! 2. `advance` returns [`PullEvent::Batch`] when a decoded array is ready.
//! 3. `advance` returns [`PullEvent::Done`] when the scan is exhausted.
//!
//! [`PullFooter`] is the same protocol for the file open path: it wraps the sans-IO
//! [`FooterDeserializer`] so a footer can be parsed without Vortex performing any reads.
//!
//! Segment requests are deduplicated with [`SharedSegmentSource`], so a segment shared by
//! multiple row splits is read exactly once per scan.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;

use futures::FutureExt;
use futures::future::BoxFuture;
use futures::task::ArcWake;
use futures::task::waker;
use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::buffer::BufferHandle;
use vortex_array::memory::HostAllocatorRef;
use vortex_array::memory::MemorySessionExt;
use vortex_array::memory::WritableHostBuffer;
use vortex_buffer::Alignment;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_layout::scan::scan_builder::ScanBuilder;
use vortex_layout::segments::SegmentFuture;
use vortex_layout::segments::SegmentId;
use vortex_layout::segments::SegmentSource;
use vortex_layout::segments::SharedSegmentSource;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;

use crate::DeserializeStep;
use crate::EOF_SIZE;
use crate::MAX_POSTSCRIPT_SIZE;
use crate::VortexFile;
use crate::footer::Footer;
use crate::footer::FooterDeserializer;
use crate::footer::SegmentSpec;

/// A single byte-range read the caller must perform.
///
/// The destination buffer is owned by this value and already has the alignment the decoder
/// requires. Fill it with the bytes at `[offset(), offset() + len())` via
/// [`data`](Self::data), then return it with [`PullScan::complete`] (or
/// [`PullFooter::complete`]).
pub struct PullRead {
    key: u32,
    offset: u64,
    buf: WritableHostBuffer,
}

impl PullRead {
    /// The absolute file offset to read from.
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// The number of bytes to read.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Whether this read is empty.
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// The destination buffer to fill with the bytes at `[offset, offset + len)`.
    pub fn data(&mut self) -> &mut [u8] {
        self.buf.as_mut_slice()
    }
}

/// The state returned by one step of a [`PullScan`].
pub enum PullEvent {
    /// Reads the caller must perform and return via [`PullScan::complete`].
    ///
    /// An empty vector means the in-flight window is full: the caller must complete an
    /// outstanding read before more work is available.
    Reads(Vec<PullRead>),
    /// A decoded batch. Ownership passes to the caller.
    Batch(ArrayRef),
    /// The scan is exhausted.
    Done,
}

/// A pending, not-yet-issued segment request registered by a layout reader.
struct PendingRequest {
    id: SegmentId,
    tx: oneshot::Sender<VortexResult<BufferHandle>>,
}

#[derive(Default)]
struct PullQueue {
    pending: VecDeque<PendingRequest>,
}

/// A [`SegmentSource`] that performs no IO: requests are queued for the caller to serve.
struct PullSegmentSource {
    specs: Arc<[SegmentSpec]>,
    queue: Arc<Mutex<PullQueue>>,
}

impl SegmentSource for PullSegmentSource {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        if self.specs.get(*id as usize).is_none() {
            return futures::future::ready(Err(vortex_err!("Missing segment: {}", id))).boxed();
        }
        let (tx, rx) = oneshot::channel();
        self.queue
            .lock()
            .pending
            .push_back(PendingRequest { id, tx });
        rx.into_future()
            .map(|r| r.unwrap_or_else(|_| Err(vortex_err!("PullScan dropped"))))
            .boxed()
    }
}

/// Wakes a split by queueing its index for re-polling on the next `advance`.
struct SplitWaker {
    idx: usize,
    queued: AtomicBool,
    woken: Arc<Mutex<VecDeque<usize>>>,
}

impl ArcWake for SplitWaker {
    fn wake_by_ref(arc_self: &Arc<Self>) {
        if !arc_self.queued.swap(true, Ordering::AcqRel) {
            arc_self.woken.lock().push_back(arc_self.idx);
        }
    }
}

type SplitFuture = BoxFuture<'static, VortexResult<Option<ArrayRef>>>;

/// A pull-based scan over a single Vortex file.
///
/// Construct with [`try_new`](Self::try_new) from a parsed [`Footer`], then drive the
/// [`advance`](Self::advance)/[`complete`](Self::complete) loop. All decoding happens on the
/// calling thread; nothing is spawned onto a runtime and no IO is performed by Vortex.
///
/// Not thread-safe: drive one `PullScan` from one thread. Parallelism comes from creating
/// multiple scans over disjoint chunk-aligned row ranges (see
/// [`ScanBuilder::with_row_range`]).
pub struct PullScan {
    specs: Arc<[SegmentSpec]>,
    queue: Arc<Mutex<PullQueue>>,
    allocator: HostAllocatorRef,
    splits: Vec<Option<SplitFuture>>,
    wakers: Vec<Arc<SplitWaker>>,
    woken: Arc<Mutex<VecDeque<usize>>>,
    issued: HashMap<u32, oneshot::Sender<VortexResult<BufferHandle>>>,
    max_inflight: usize,
    live: usize,
    started: bool,
    batches: VecDeque<ArrayRef>,
}

impl PullScan {
    /// Create a pull scan over a file described by `footer`.
    ///
    /// `configure` customizes the scan (projection, filter, row range, ...) on the same
    /// [`ScanBuilder`] used by ordinary scans. `max_inflight` bounds how many reads are issued
    /// but not yet completed, which also bounds destination-buffer memory; pass 0 for no bound.
    pub fn try_new(
        footer: Footer,
        session: &VortexSession,
        max_inflight: usize,
        configure: impl FnOnce(ScanBuilder<ArrayRef>) -> ScanBuilder<ArrayRef>,
    ) -> VortexResult<Self> {
        let specs = footer.segment_specs_with_metadata();
        let queue = Arc::new(Mutex::new(PullQueue::default()));
        let source: Arc<dyn SegmentSource> =
            Arc::new(SharedSegmentSource::new(PullSegmentSource {
                specs: Arc::clone(&specs),
                queue: Arc::clone(&queue),
            }));
        let file = VortexFile::new(footer, source, session.clone());
        let splits = configure(file.scan()?).build()?;

        let woken = Arc::new(Mutex::new(VecDeque::new()));
        let wakers = (0..splits.len())
            .map(|idx| {
                Arc::new(SplitWaker {
                    idx,
                    queued: AtomicBool::new(false),
                    woken: Arc::clone(&woken),
                })
            })
            .collect();

        let live = splits.len();
        Ok(Self {
            specs,
            queue,
            allocator: session.allocator(),
            splits: splits.into_iter().map(Some).collect(),
            wakers,
            woken,
            issued: HashMap::default(),
            max_inflight: if max_inflight == 0 {
                usize::MAX
            } else {
                max_inflight
            },
            live,
            started: false,
            batches: VecDeque::new(),
        })
    }

    /// Advance the scan: harvest decoded batches, then issue reads up to the in-flight window.
    ///
    /// Returns [`PullEvent::Batch`] before issuing new reads so callers drain decoded data
    /// promptly.
    pub fn advance(&mut self) -> VortexResult<PullEvent> {
        if !self.started {
            self.started = true;
            for idx in 0..self.splits.len() {
                self.poll_split(idx)?;
            }
        } else {
            loop {
                let Some(idx) = self.woken.lock().pop_front() else {
                    break;
                };
                self.poll_split(idx)?;
            }
        }

        if let Some(batch) = self.batches.pop_front() {
            return Ok(PullEvent::Batch(batch));
        }

        let mut reads = Vec::new();
        while self.issued.len() < self.max_inflight {
            let Some(req) = self.queue.lock().pending.pop_front() else {
                break;
            };
            let spec = self.specs[*req.id as usize];
            let buf = self
                .allocator
                .allocate(spec.length as usize, spec.alignment)?;
            self.issued.insert(*req.id, req.tx);
            reads.push(PullRead {
                key: *req.id,
                offset: spec.offset,
                buf,
            });
        }
        if !reads.is_empty() {
            return Ok(PullEvent::Reads(reads));
        }

        if self.live == 0 {
            return Ok(PullEvent::Done);
        }
        if self.issued.is_empty() {
            vortex_bail!("PullScan stalled: no outstanding reads, batches, or live splits");
        }
        Ok(PullEvent::Reads(Vec::new()))
    }

    /// Hand back a filled read. Completions may arrive in any order.
    pub fn complete(&mut self, read: PullRead) -> VortexResult<()> {
        let tx = self
            .issued
            .remove(&read.key)
            .ok_or_else(|| vortex_err!("Unknown or already completed read"))?;
        // The receiver may be gone if the requesting split was dropped; that is not an error.
        drop(tx.send(Ok(BufferHandle::new_host(read.buf.freeze()))));
        Ok(())
    }

    fn poll_split(&mut self, idx: usize) -> VortexResult<()> {
        let Some(fut) = self.splits[idx].as_mut() else {
            return Ok(());
        };
        self.wakers[idx].queued.store(false, Ordering::Release);
        let waker = waker(Arc::clone(&self.wakers[idx]));
        let mut cx = Context::from_waker(&waker);
        if let Poll::Ready(result) = fut.as_mut().poll(&mut cx) {
            self.splits[idx] = None;
            self.live -= 1;
            if let Some(array) = result? {
                self.batches.push_back(array);
            }
        }
        Ok(())
    }
}

/// The state returned by one step of [`PullFooter`].
pub enum FooterEvent {
    /// A read the caller must perform and return via [`PullFooter::complete`].
    Read(PullRead),
    /// The parsed footer.
    Done(Footer),
}

enum FooterState {
    /// Waiting to issue the initial tail read.
    Init,
    /// Waiting for the caller to complete the initial tail read.
    ReadingTail,
    /// Waiting for the caller to complete a prefix read requested by the deserializer.
    ReadingMore(FooterDeserializer),
    /// Ready to run the next deserialization step.
    Deserializing(FooterDeserializer),
    /// Finished or failed.
    Complete,
}

/// A pull-based footer parser: opens a file without Vortex performing any IO.
///
/// Drives the sans-IO [`FooterDeserializer`]. Footer reads are sequential (each depends on the
/// previous parse step), so exactly one read is outstanding at a time.
pub struct PullFooter {
    session: VortexSession,
    file_size: u64,
    state: FooterState,
}

impl PullFooter {
    /// Create a footer parser for a file of `file_size` bytes.
    pub fn new(session: VortexSession, file_size: u64) -> Self {
        Self {
            session,
            file_size,
            state: FooterState::Init,
        }
    }

    /// Advance the parser: returns the next read to perform, or the parsed footer.
    pub fn advance(&mut self) -> VortexResult<FooterEvent> {
        match std::mem::replace(&mut self.state, FooterState::Complete) {
            FooterState::Init => {
                let initial_read_size =
                    (MAX_POSTSCRIPT_SIZE as usize + EOF_SIZE).min(self.file_size as usize);
                let offset = self.file_size - initial_read_size as u64;
                let read = self.alloc_read(offset, initial_read_size)?;
                self.state = FooterState::ReadingTail;
                Ok(FooterEvent::Read(read))
            }
            state @ (FooterState::ReadingTail | FooterState::ReadingMore(_)) => {
                self.state = state;
                vortex_bail!("PullFooter: complete the outstanding read before advancing")
            }
            FooterState::Deserializing(mut deserializer) => match deserializer.deserialize()? {
                DeserializeStep::NeedMoreData { offset, len } => {
                    let read = self.alloc_read(offset, len)?;
                    self.state = FooterState::ReadingMore(deserializer);
                    Ok(FooterEvent::Read(read))
                }
                DeserializeStep::NeedFileSize => {
                    vortex_bail!("PullFooter: file size was provided up front")
                }
                DeserializeStep::Done(footer) => {
                    footer.validate_file_size(self.file_size)?;
                    Ok(FooterEvent::Done(footer))
                }
            },
            FooterState::Complete => vortex_bail!("PullFooter already complete"),
        }
    }

    /// Hand back the filled read issued by the previous [`advance`](Self::advance).
    pub fn complete(&mut self, read: PullRead) -> VortexResult<()> {
        match std::mem::replace(&mut self.state, FooterState::Complete) {
            FooterState::ReadingTail => {
                let deserializer = Footer::deserializer(read.buf.freeze(), self.session.clone())
                    .with_size(self.file_size);
                self.state = FooterState::Deserializing(deserializer);
                Ok(())
            }
            FooterState::ReadingMore(mut deserializer) => {
                deserializer.prefix_data(read.buf.freeze());
                self.state = FooterState::Deserializing(deserializer);
                Ok(())
            }
            state => {
                self.state = state;
                vortex_bail!("PullFooter: no outstanding read to complete")
            }
        }
    }

    fn alloc_read(&self, offset: u64, len: usize) -> VortexResult<PullRead> {
        Ok(PullRead {
            key: 0,
            offset,
            buf: self.session.allocator().allocate(len, Alignment::none())?,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::expr::get_item;
    use vortex_array::expr::gt_eq;
    use vortex_array::expr::lit;
    use vortex_array::expr::root;
    use vortex_buffer::Buffer;
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::ByteBufferMut;
    use vortex_io::session::RuntimeSession;
    use vortex_layout::session::LayoutSession;
    use vortex_utils::aliases::hash_set::HashSet;

    use super::*;
    use crate::WriteOptionsSessionExt;
    use crate::WriteStrategyBuilder;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = array_session()
            .with::<LayoutSession>()
            .with::<RuntimeSession>();
        crate::register_default_encodings(&session);
        session
    });

    async fn write_sample_file() -> (ByteBuffer, StructArray) {
        let chunks = (0..4u32)
            .map(|chunk| {
                let base = chunk * 1000;
                let numbers = Buffer::from((base..base + 1000).collect::<Vec<u32>>()).into_array();
                let strings = VarBinArray::from(
                    (base..base + 1000)
                        .map(|i| format!("row-{i}"))
                        .collect::<Vec<_>>(),
                )
                .into_array();
                StructArray::from_fields(&[("numbers", numbers), ("strings", strings)])
                    .unwrap()
                    .into_array()
            })
            .collect::<Vec<_>>();
        let st = ChunkedArray::from_iter(chunks.clone()).into_array();

        let mut buf = ByteBufferMut::empty();
        // Keep row blocks at chunk granularity so row-range tests can observe segment skipping.
        let strategy = WriteStrategyBuilder::default()
            .with_row_block_size(1000)
            .with_data_block_target_bytes(None)
            .build();
        SESSION
            .write_options()
            .with_strategy(strategy)
            .write(&mut buf, st.to_array_stream())
            .await
            .unwrap();

        let whole = (0..4u32)
            .map(|chunk| chunk * 1000)
            .flat_map(|base| (base..base + 1000).collect::<Vec<_>>())
            .collect::<Vec<u32>>();
        let expected = StructArray::from_fields(&[
            ("numbers", Buffer::from(whole.clone()).into_array()),
            (
                "strings",
                VarBinArray::from(whole.iter().map(|i| format!("row-{i}")).collect::<Vec<_>>())
                    .into_array(),
            ),
        ])
        .unwrap();
        (buf.freeze(), expected)
    }

    fn serve(bytes: &ByteBuffer, read: &mut PullRead) {
        let start = read.offset() as usize;
        let end = start + read.len();
        read.data().copy_from_slice(&bytes.as_slice()[start..end]);
    }

    fn pull_footer(bytes: &ByteBuffer) -> VortexResult<Footer> {
        let mut footer = PullFooter::new(SESSION.clone(), bytes.len() as u64);
        loop {
            match footer.advance()? {
                FooterEvent::Read(mut read) => {
                    serve(bytes, &mut read);
                    footer.complete(read)?;
                }
                FooterEvent::Done(footer) => return Ok(footer),
            }
        }
    }

    type ServedReads = Vec<(u64, usize)>;

    /// Drives a scan to completion, returning batches and the set of served reads.
    fn drive(
        scan: &mut PullScan,
        bytes: &ByteBuffer,
    ) -> VortexResult<(Vec<ArrayRef>, ServedReads)> {
        let mut batches = Vec::new();
        let mut served = Vec::new();
        loop {
            match scan.advance()? {
                PullEvent::Reads(reads) => {
                    assert!(
                        !reads.is_empty(),
                        "a synchronous driver must never observe a full window"
                    );
                    for mut read in reads {
                        served.push((read.offset(), read.len()));
                        serve(bytes, &mut read);
                        scan.complete(read)?;
                    }
                }
                PullEvent::Batch(batch) => batches.push(batch),
                PullEvent::Done => return Ok((batches, served)),
            }
        }
    }

    fn sort_and_concat(mut batches: Vec<ArrayRef>) -> VortexResult<ArrayRef> {
        let mut ctx = SESSION.create_execution_ctx();
        batches.sort_by_key(|batch| {
            batch
                .execute_scalar(0, &mut ctx)
                .ok()
                .and_then(|s| s.as_struct().field_by_idx(0))
                .and_then(|s| u32::try_from(&s).ok())
                .unwrap_or(u32::MAX)
        });
        Ok(ChunkedArray::from_iter(batches).into_array())
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn pull_scan_round_trip() -> VortexResult<()> {
        let (bytes, expected) = write_sample_file().await;
        let footer = pull_footer(&bytes)?;

        let mut scan = PullScan::try_new(footer, &SESSION, 0, |b| b)?;
        let (batches, served) = drive(&mut scan, &bytes)?;

        let mut seen: HashSet<(u64, usize)> = HashSet::default();
        for read in &served {
            assert!(seen.insert(*read), "duplicate read of {read:?}");
        }

        let result = sort_and_concat(batches)?;
        let mut ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn pull_scan_depth_one_window() -> VortexResult<()> {
        let (bytes, expected) = write_sample_file().await;
        let footer = pull_footer(&bytes)?;

        let mut scan = PullScan::try_new(footer, &SESSION, 1, |b| b)?;
        let (batches, _) = drive(&mut scan, &bytes)?;

        let result = sort_and_concat(batches)?;
        let mut ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn pull_scan_filter_and_row_range() -> VortexResult<()> {
        let (bytes, _) = write_sample_file().await;
        let footer = pull_footer(&bytes)?;

        let mut scan = PullScan::try_new(footer, &SESSION, 0, |b| {
            b.with_filter(gt_eq(get_item("numbers", root()), lit(3500u32)))
        })?;
        let (batches, _) = drive(&mut scan, &bytes)?;
        let filtered: usize = batches.iter().map(|b| b.len()).sum();
        assert_eq!(filtered, 500);

        let footer = pull_footer(&bytes)?;
        let mut scan = PullScan::try_new(footer, &SESSION, 0, |b| b.with_row_range(1000..3000))?;
        let (batches, served) = drive(&mut scan, &bytes)?;
        let rows: usize = batches.iter().map(|b| b.len()).sum();
        assert_eq!(rows, 2000);
        // A chunk-aligned row range must not read the excluded chunks' data segments.
        let all_footer = pull_footer(&bytes)?;
        let mut full = PullScan::try_new(all_footer, &SESSION, 0, |b| b)?;
        let (_, full_served) = drive(&mut full, &bytes)?;
        assert!(served.len() < full_served.len());
        Ok(())
    }
}
