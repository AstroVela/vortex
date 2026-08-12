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

use std::io;
use std::os::fd::AsRawFd;
use std::os::fd::RawFd;
use std::sync::Arc;
use std::sync::OnceLock;

use io_uring::IoUring;
use io_uring::opcode;
use io_uring::types;
use vortex_array::memory::WritableHostBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_utils::aliases::hash_map::HashMap;

/// Number of submission queue entries for the shared ring.
const QUEUE_DEPTH: u32 = 256;

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

/// A handle to the shared `io_uring` submission thread.
pub(crate) struct UringDriver {
    ops: kanal::Sender<UringOp>,
}

/// The process-wide driver, started on first use.
///
/// One ring is enough: submission is cheap and the queue depth bounds concurrency across all open
/// files. Starting a ring per file would multiply kernel memory for no gain.
static DRIVER: OnceLock<Option<Arc<UringDriver>>> = OnceLock::new();

impl UringDriver {
    /// Returns the shared driver, or `None` if this kernel cannot provide an `io_uring`.
    pub(crate) fn get() -> Option<&'static Arc<Self>> {
        DRIVER.get_or_init(Self::start).as_ref()
    }

    fn start() -> Option<Arc<Self>> {
        // Build the ring on this thread so that a kernel without io_uring (or a seccomp policy
        // that blocks io_uring_setup) is detected here rather than inside the driver thread.
        let ring = match IoUring::new(QUEUE_DEPTH) {
            Ok(ring) => ring,
            Err(err) => {
                tracing::debug!("io_uring unavailable, falling back to blocking reads: {err}");
                return None;
            }
        };

        let (ops, rx) = kanal::unbounded::<UringOp>();
        std::thread::Builder::new()
            .name("vortex-uring".into())
            .spawn(move || drive(ring, rx))
            .inspect_err(|err| tracing::warn!("could not start io_uring driver thread: {err}"))
            .ok()?;

        Some(Arc::new(Self { ops }))
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
/// The caller must keep `pending` alive in `inflight` until its completion is reaped, so that the
/// buffer the kernel writes into outlives the operation.
unsafe fn push(ring: &mut IoUring, id: u64, pending: &mut Pending) -> io::Result<()> {
    let slice = pending.buffer.as_mut_slice();
    let remaining = &mut slice[pending.done..];
    let entry = opcode::Read::new(
        types::Fd(pending.fd),
        remaining.as_mut_ptr(),
        u32::try_from(remaining.len()).unwrap_or(u32::MAX),
    )
    .offset(pending.offset + pending.done as u64)
    .build()
    .user_data(id);

    // SAFETY: `remaining` points into `pending.buffer`, which the caller keeps alive in the
    // in-flight map until this operation's completion is reaped.
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

fn drive(mut ring: IoUring, rx: kanal::Receiver<UringOp>) {
    let mut inflight: HashMap<u64, Pending> = HashMap::new();
    let mut next_id = 0u64;

    let mut accept = |inflight: &mut HashMap<u64, Pending>, ring: &mut IoUring, op: UringOp| {
        let id = next_id;
        next_id = next_id.wrapping_add(1);
        let mut pending = Pending {
            fd: op.fd,
            offset: op.offset,
            required: op.required,
            done: 0,
            buffer: op.buffer,
            reply: op.reply,
        };
        // SAFETY: `pending` is moved into `inflight` immediately below and is only removed once
        // its completion has been reaped, so the buffer outlives the kernel's write to it.
        if let Err(err) = unsafe { push(ring, id, &mut pending) } {
            reply(pending.reply, Err(err.into()));
            return;
        }
        inflight.insert(id, pending);
    };

    loop {
        // Take whatever work is queued without blocking.
        while inflight.len() < QUEUE_DEPTH as usize {
            match rx.try_recv() {
                Ok(Some(op)) => accept(&mut inflight, &mut ring, op),
                Ok(None) => break,
                Err(_) => return,
            }
        }

        if inflight.is_empty() {
            // Nothing in flight, so park until there is work rather than spinning.
            match rx.recv() {
                Ok(op) => accept(&mut inflight, &mut ring, op),
                Err(_) => return,
            }
            continue;
        }

        // Note that this parks until a completion arrives, so an operation submitted while we are
        // blocked here waits for the next completion before reaching the kernel. With a queue this
        // deep completions are frequent, and the alternative (polling an eventfd registered with
        // the ring) costs more than the latency it saves.
        if let Err(err) = ring.submit_and_wait(1) {
            if err.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            // The ring is unusable; fail everything rather than hanging its callers.
            for (_, pending) in inflight.drain() {
                reply(
                    pending.reply,
                    Err(vortex_err!("io_uring submit failed: {err}")),
                );
            }
            return;
        }

        let completed: Vec<(u64, i32)> = ring
            .completion()
            .map(|cqe| (cqe.user_data(), cqe.result()))
            .collect();

        for (id, result) in completed {
            let Some(mut pending) = inflight.remove(&id) else {
                continue;
            };

            if result < 0 {
                let err = io::Error::from_raw_os_error(-result);
                if err.kind() == io::ErrorKind::Interrupted {
                    // SAFETY: re-inserted into `inflight` immediately below.
                    match unsafe { push(&mut ring, id, &mut pending) } {
                        Ok(()) => {
                            inflight.insert(id, pending);
                        }
                        Err(err) => {
                            reply(pending.reply, Err(err.into()));
                        }
                    }
                    continue;
                }
                reply(
                    pending.reply,
                    Err(vortex_err!(
                        "io_uring read failed at offset {} ({} bytes): {err}",
                        pending.offset,
                        pending.buffer.len()
                    )),
                );
                continue;
            }

            pending.done += result as usize;

            if result == 0 || pending.done >= pending.required {
                let result = if pending.done < pending.required {
                    Err(vortex_err!(
                        "io_uring read hit end of file after {} of {} required bytes at offset {}",
                        pending.done,
                        pending.required,
                        pending.offset
                    ))
                } else {
                    Ok(pending.buffer)
                };
                reply(pending.reply, result);
                continue;
            }

            // Short read: resume from where the kernel stopped.
            // SAFETY: re-inserted into `inflight` immediately below.
            match unsafe { push(&mut ring, id, &mut pending) } {
                Ok(()) => {
                    inflight.insert(id, pending);
                }
                Err(err) => {
                    reply(pending.reply, Err(err.into()));
                }
            }
        }
    }
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
