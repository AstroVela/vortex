// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An `io_uring` submission driver for local file reads.
//!
//! Vortex supports several async runtimes through [`Handle`][crate::runtime::Handle], so this
//! driver deliberately does not adopt a uring-based runtime such as `tokio-uring`. Instead a
//! single dedicated thread owns the ring, accepts operations over a channel, and completes each
//! one through a oneshot. The resulting future is runtime-agnostic and composes with the existing
//! `buffer_unordered` read driver in `vortex-file` without any changes there.
//!
//! `io_uring` is only worth using here in combination with `O_DIRECT`. On a buffered file
//! descriptor a read that hits the page cache is serviced synchronously, and one that misses is
//! handed to a kernel worker thread, so the ring ends up doing what the existing blocking thread
//! pool already does. With `O_DIRECT` the read is issued to the device asynchronously and the
//! submission path actually pays for itself.
//!
//! Exactly one thread ever touches the ring, which the driver leans on in three places: the ring
//! is created with `SINGLE_ISSUER` and `DEFER_TASKRUN` so the kernel can skip its cross-CPU
//! wakeups, in-flight operations live in a slab indexed by `user_data` rather than a hash map, and
//! the loop folds submission and waiting into a single `io_uring_enter` per batch.

use std::io;
use std::os::fd::AsRawFd;
use std::os::fd::RawFd;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::mpsc::sync_channel;

use io_uring::IoUring;
use io_uring::opcode;
use io_uring::types;
use vortex_array::memory::WritableHostBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

/// Number of submission queue entries for the shared ring.
const QUEUE_DEPTH: u32 = 256;

/// How long to spin on the operation channel before parking the driver thread.
///
/// Parking costs a futex wake on the next submission, which at low queue depth is a large fraction
/// of a read's latency. Spinning briefly hides that, but a driver that spins for long would burn a
/// core and give up the thing that makes the ring attractive under CPU pressure: it needs one
/// runnable thread no matter how many reads are in flight. This bound keeps the spin in the tens of
/// microseconds and yields to the scheduler rather than busy-waiting outright.
const SPIN_YIELDS: u32 = 32;

/// A completed direct read: the buffer the kernel filled.
pub(crate) type UringRead = WritableHostBuffer;

struct UringOp {
    fd: RawFd,
    offset: u64,
    /// Bytes that must be transferred for the read to be considered complete. The buffer may be
    /// longer than this when a request was widened past the end of the file.
    required: usize,
    buffer: WritableHostBuffer,
    reply: oneshot::Sender<VortexResult<UringRead>>,
}

struct Pending {
    fd: RawFd,
    offset: u64,
    required: usize,
    done: usize,
    buffer: WritableHostBuffer,
    reply: oneshot::Sender<VortexResult<UringRead>>,
}

/// In-flight operations, indexed by the `user_data` carried on their submission.
///
/// A slab rather than a map: `user_data` is ours to choose, so it can be a slot index that reaps in
/// O(1) with no hashing and no per-operation allocation.
#[derive(Default)]
struct Inflight {
    slots: Vec<Option<Pending>>,
    free: Vec<usize>,
    live: usize,
}

impl Inflight {
    /// Reserve a slot id without filling it.
    ///
    /// The submitted SQE has to carry its own slot id as `user_data`, so the id must be known
    /// before the operation can be pushed. Fill the slot with [`Inflight::occupy`] once the push
    /// succeeds, or hand it back with [`Inflight::release`] if it does not.
    fn reserve(&mut self) -> usize {
        self.live += 1;
        match self.free.pop() {
            Some(id) => id,
            None => {
                self.slots.push(None);
                self.slots.len() - 1
            }
        }
    }

    fn occupy(&mut self, id: usize, pending: Pending) {
        self.slots[id] = Some(pending);
    }

    fn release(&mut self, id: usize) {
        self.free.push(id);
        self.live -= 1;
    }

    /// Take an operation out for completion, leaving its slot reusable.
    fn take(&mut self, id: usize) -> Option<Pending> {
        let pending = self.slots.get_mut(id)?.take()?;
        self.free.push(id);
        self.live -= 1;
        Some(pending)
    }

    /// Return an operation to the slot it came from, for a partial read that must be resumed.
    fn restore(&mut self, id: usize, pending: Pending) {
        debug_assert_eq!(self.free.last(), Some(&id), "slot {id} was not just taken");
        self.free.pop();
        self.live += 1;
        self.slots[id] = Some(pending);
    }

    fn is_empty(&self) -> bool {
        self.live == 0
    }
}

/// A handle to the shared `io_uring` submission thread.
pub(crate) struct UringDriver {
    ops: kanal::Sender<UringOp>,
}

/// The process-wide driver, started on first use.
///
/// One ring is enough: submission is cheap and the queue depth bounds concurrency across all open
/// files. Starting a ring per file would multiply kernel memory for no gain.
static DRIVER: OnceLock<Option<Arc<UringDriver>>> = OnceLock::new();

/// Build the ring, preferring the flags that suit a single-threaded driver.
///
/// `DEFER_TASKRUN` lets the kernel run completion work when this thread next waits on the ring
/// instead of interrupting whichever CPU the I/O completed on, and it requires `SINGLE_ISSUER`.
/// Both are only sound because this driver is the sole thread that ever submits. Older kernels
/// reject the flags, so fall back rather than give up the ring entirely.
fn build_ring() -> io::Result<IoUring> {
    IoUring::builder()
        .setup_single_issuer()
        .setup_defer_taskrun()
        .build(QUEUE_DEPTH)
        .or_else(|_| IoUring::builder().setup_coop_taskrun().build(QUEUE_DEPTH))
        .or_else(|_| IoUring::new(QUEUE_DEPTH))
}

impl UringDriver {
    /// Returns the shared driver, or `None` if this kernel cannot provide an `io_uring`.
    pub(crate) fn get() -> Option<&'static Arc<Self>> {
        DRIVER.get_or_init(Self::start).as_ref()
    }

    fn start() -> Option<Arc<Self>> {
        let (ops, rx) = kanal::unbounded::<UringOp>();
        let (ready, started) = sync_channel::<bool>(1);

        // The ring is built on the driver thread rather than here, because `SINGLE_ISSUER` binds
        // it to the task that owns it. Success is reported back so that a kernel without io_uring,
        // or a seccomp policy that blocks `io_uring_setup`, still surfaces as a `None` here and
        // falls back to blocking reads.
        std::thread::Builder::new()
            .name("vortex-uring".into())
            .spawn(move || match build_ring() {
                Ok(ring) => {
                    ready.send(true).ok();
                    drive(ring, rx);
                }
                Err(err) => {
                    tracing::debug!("io_uring unavailable, falling back to blocking reads: {err}");
                    ready.send(false).ok();
                }
            })
            .inspect_err(|err| tracing::warn!("could not start io_uring driver thread: {err}"))
            .ok()?;

        // Blocks only until the driver thread reports whether its ring was created; the thread
        // dropping the sender (a panic during setup) resolves this as an error rather than a hang.
        started.recv().ok()?.then(|| Arc::new(Self { ops }))
    }

    /// Submit a positional read, returning a future that resolves once the ring completes it.
    pub(crate) fn read_at(
        &self,
        file: &std::fs::File,
        offset: u64,
        required: usize,
        buffer: WritableHostBuffer,
    ) -> VortexResult<oneshot::Receiver<VortexResult<UringRead>>> {
        let (reply, receiver) = oneshot::channel();
        self.ops
            .send(UringOp {
                fd: file.as_raw_fd(),
                offset,
                required,
                buffer,
                reply,
            })
            .map_err(|_| vortex_err!("io_uring driver thread has stopped"))?;
        Ok(receiver)
    }
}

/// Build and push a read SQE for `pending`, resuming from however much it has already read.
///
/// # Safety
///
/// The caller must keep `pending` alive in the in-flight slab until its completion is reaped, so
/// that the buffer the kernel writes into outlives the operation.
unsafe fn push(ring: &mut IoUring, id: usize, pending: &mut Pending) -> io::Result<()> {
    let slice = pending.buffer.as_mut_slice();
    let remaining = &mut slice[pending.done..];
    let entry = opcode::Read::new(
        types::Fd(pending.fd),
        remaining.as_mut_ptr(),
        u32::try_from(remaining.len()).unwrap_or(u32::MAX),
    )
    .offset(pending.offset + pending.done as u64)
    .build()
    .user_data(id as u64);

    // SAFETY: `remaining` points into `pending.buffer`, which the caller keeps alive in the
    // in-flight slab until this operation's completion is reaped.
    while unsafe { ring.submission().push(&entry) }.is_err() {
        // The submission queue is full; flush it to the kernel to make room.
        ring.submit()?;
        ring.submission().sync();
    }
    Ok(())
}

/// Deliver a result, ignoring a receiver whose caller has already gone away.
fn reply(sender: oneshot::Sender<VortexResult<UringRead>>, result: VortexResult<UringRead>) {
    drop(sender.send(result));
}

/// Take the next operation, spinning briefly before parking the thread.
fn next_op(rx: &kanal::Receiver<UringOp>) -> Option<UringOp> {
    for _ in 0..SPIN_YIELDS {
        match rx.try_recv() {
            Ok(Some(op)) => return Some(op),
            Ok(None) => std::thread::yield_now(),
            Err(_) => return None,
        }
    }
    rx.recv().ok()
}

fn drive(mut ring: IoUring, rx: kanal::Receiver<UringOp>) {
    let mut inflight = Inflight::default();

    // Accept an operation onto the ring. Returns false once the channel is gone.
    macro_rules! accept {
        ($op:expr) => {{
            let op: UringOp = $op;
            let mut pending = Pending {
                fd: op.fd,
                offset: op.offset,
                required: op.required,
                done: 0,
                buffer: op.buffer,
                reply: op.reply,
            };
            let id = inflight.reserve();
            // SAFETY: `pending` is stored in the slab immediately below and only removed once its
            // completion has been reaped, so the buffer outlives the kernel's write to it.
            match unsafe { push(&mut ring, id, &mut pending) } {
                Ok(()) => inflight.occupy(id, pending),
                Err(err) => {
                    inflight.release(id);
                    reply(pending.reply, Err(err.into()));
                }
            }
        }};
    }

    loop {
        // Take whatever work is already queued without blocking.
        let mut queued = 0usize;
        while inflight.live < QUEUE_DEPTH as usize {
            match rx.try_recv() {
                Ok(Some(op)) => {
                    accept!(op);
                    queued += 1;
                }
                Ok(None) => break,
                Err(_) => return,
            }
        }

        if inflight.is_empty() {
            match next_op(&rx) {
                Some(op) => accept!(op),
                None => return,
            }
            continue;
        }

        // Reap anything the kernel has already posted. This costs no syscall, so it is worth
        // trying before deciding to wait.
        let mut reaped = 0usize;
        loop {
            ring.completion().sync();
            let batch: Vec<(usize, i32)> = ring
                .completion()
                .filter_map(|cqe| Some((usize::try_from(cqe.user_data()).ok()?, cqe.result())))
                .collect();
            if batch.is_empty() {
                break;
            }
            reaped += batch.len();
            for (id, result) in batch {
                complete(&mut ring, &mut inflight, id, result);
            }
        }

        if reaped > 0 {
            // Completions freed queue space and may have unblocked callers who are about to submit
            // more work, so loop round rather than entering the kernel.
            if queued > 0 && ring.submit().is_err() {
                fail_all(&mut inflight, "io_uring submit failed");
                return;
            }
            continue;
        }

        // Nothing was ready. One `io_uring_enter` both flushes the queued submissions and waits,
        // so at low depth a read costs a single syscall rather than one to submit and one to wait.
        match ring.submit_and_wait(1) {
            Ok(_) => {}
            Err(err) if err.kind() == io::ErrorKind::Interrupted => {}
            Err(err) => {
                fail_all(&mut inflight, &format!("io_uring submit failed: {err}"));
                return;
            }
        }
    }
}

fn complete(ring: &mut IoUring, inflight: &mut Inflight, id: usize, result: i32) {
    let Some(mut pending) = inflight.take(id) else {
        return;
    };

    if result < 0 {
        let err = io::Error::from_raw_os_error(-result);
        if err.kind() == io::ErrorKind::Interrupted {
            resubmit(ring, inflight, id, pending);
            return;
        }
        reply(
            pending.reply,
            Err(vortex_err!(
                "io_uring read failed at offset {} ({} bytes): {err}",
                pending.offset,
                pending.buffer.len()
            )),
        );
        return;
    }

    pending.done += result as usize;

    if result == 0 || pending.done >= pending.required {
        let outcome = if pending.done < pending.required {
            Err(vortex_err!(
                "io_uring read hit end of file after {} of {} required bytes at offset {}",
                pending.done,
                pending.required,
                pending.offset
            ))
        } else {
            Ok(pending.buffer)
        };
        reply(pending.reply, outcome);
        return;
    }

    // Short read: resume from where the kernel stopped.
    resubmit(ring, inflight, id, pending);
}

fn resubmit(ring: &mut IoUring, inflight: &mut Inflight, id: usize, mut pending: Pending) {
    // SAFETY: `pending` is restored to the slab immediately below and only removed once its
    // completion has been reaped.
    match unsafe { push(ring, id, &mut pending) } {
        Ok(()) => inflight.restore(id, pending),
        Err(err) => reply(pending.reply, Err(err.into())),
    }
}

/// Fail every in-flight operation, so an unusable ring surfaces as errors rather than a hang.
fn fail_all(inflight: &mut Inflight, message: &str) {
    for slot in &mut inflight.slots {
        if let Some(pending) = slot.take() {
            reply(pending.reply, Err(vortex_err!("{message}")));
        }
    }
    inflight.live = 0;
}

/// Await a submitted read, translating a dropped driver thread into an error.
pub(crate) async fn await_read(
    receiver: oneshot::Receiver<VortexResult<UringRead>>,
) -> VortexResult<UringRead> {
    match receiver.await {
        Ok(result) => result,
        Err(_) => vortex_bail!("io_uring driver dropped a pending read"),
    }
}
