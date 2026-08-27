// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The scheduler-visible IO plane.
//!
//! Nodes never read. They *name* reads: [`PlanCx::register`](crate::PlanCx::register) takes an
//! [`IoBatch`] of [`IoUse`]s, each keyed to a whole stored unit, and hands back an [`IoTicket`].
//! Execution may only wait on tickets its own planning stream emitted.
//!
//! In P1 the plane is per-thread and every use is issued immediately, so the plane's job is
//! deduplication (two morsels naming the same segment share one cell) and prefetch (the whole
//! morsel's reads are issued during planning, before the first decode blocks). The
//! `source_range`, `producer` and demand-verdict fields are carried but not yet consulted; P2
//! makes them the scheduler's admission input.

use std::cell::RefCell;
use std::future::Future;
use std::ops::Range;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;
use std::task::Wake;
use std::task::Waker;

use futures::FutureExt;
use vortex_array::buffer::BufferHandle;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_layout::segments::SegmentFuture;
use vortex_layout::segments::SegmentId;
use vortex_layout::segments::SegmentSource;
use vortex_utils::aliases::hash_map::HashMap;

use crate::node::Wait;
use crate::stats::ScanStats;

/// The key of a shared cell: a whole stored unit. Morsels straddling a unit join the same cell.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum IoKey {
    /// A layout segment.
    Segment(SegmentId),
}

/// A ticket handed back by registration, naming the cell the read will land in.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct IoTicket(IoKey);

impl IoTicket {
    /// The cell this ticket names.
    pub fn key(&self) -> IoKey {
        self.0
    }
}

/// Identifies the node that emitted a use, so the scheduler can attribute and cancel it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProducerId(pub u32);

/// One named read.
#[derive(Clone, Debug)]
pub struct IoUse {
    /// The whole stored unit this use covers.
    pub key: IoKey,
    /// The rows of the stored unit, frozen at emission.
    pub extent: Range<u64>,
    /// The inverse image of `extent` in root coordinates, stamped at emission. The scheduler
    /// reads demand verdicts over this range without ever seeing an offset map.
    pub source_range: Range<u64>,
    /// The node that emitted this use.
    pub producer: ProducerId,
    /// The estimated size of the read, for admission accounting.
    pub estimated_bytes: usize,
}

/// A batch of uses emitted by one planning step.
#[derive(Clone, Debug, Default)]
pub struct IoBatch {
    uses: Vec<IoUse>,
}

impl IoBatch {
    /// An empty batch.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a use to the batch.
    pub fn push(&mut self, r#use: IoUse) {
        self.uses.push(r#use);
    }

    /// The uses in this batch.
    pub fn uses(&self) -> &[IoUse] {
        &self.uses
    }

    /// Whether the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.uses.is_empty()
    }
}

impl FromIterator<IoUse> for IoBatch {
    fn from_iter<T: IntoIterator<Item = IoUse>>(iter: T) -> Self {
        Self {
            uses: iter.into_iter().collect(),
        }
    }
}

enum Cell {
    Pending(SegmentFuture),
    Ready(BufferHandle),
}

/// The per-thread IO plane.
///
/// Interior mutability rather than locks: one plane per driving thread, so cells are never
/// contended. Cross-thread sharing of cells is P2 work.
pub struct IoPlane {
    source: Arc<dyn SegmentSource>,
    cells: RefCell<HashMap<IoKey, Cell>>,
    /// Reads at or below this size skip registration and are read inline at first use.
    inline_floor_bytes: usize,
}

impl IoPlane {
    /// Create a plane over a segment source.
    pub fn new(source: Arc<dyn SegmentSource>) -> Self {
        Self {
            source,
            cells: RefCell::new(HashMap::default()),
            inline_floor_bytes: 0,
        }
    }

    /// Set the size at or below which a read bypasses registration and is issued inline.
    pub fn with_inline_floor_bytes(mut self, bytes: usize) -> Self {
        self.inline_floor_bytes = bytes;
        self
    }

    /// Register a batch of uses, issuing any cell that does not already exist.
    pub fn register(&self, batch: IoBatch, stats: &mut ScanStats) -> VortexResult<Vec<IoTicket>> {
        let mut cells = self.cells.borrow_mut();
        let mut tickets = Vec::with_capacity(batch.uses().len());
        for r#use in batch.uses() {
            if !cells.contains_key(&r#use.key) {
                if r#use.estimated_bytes <= self.inline_floor_bytes {
                    stats.io_bypassed += 1;
                } else {
                    stats.io_registered += 1;
                }
                cells.insert(r#use.key, Cell::Pending(self.issue(r#use.key)));
                stats.io_requests += 1;
            } else {
                stats.io_cell_hits += 1;
            }
            tickets.push(IoTicket(r#use.key));
        }
        Ok(tickets)
    }

    fn issue(&self, key: IoKey) -> SegmentFuture {
        match key {
            IoKey::Segment(id) => self.source.request(id),
        }
    }

    /// Resolve the cells behind the given waits.
    pub fn wait(&self, waits: &[Wait], stats: &mut ScanStats) -> VortexResult<()> {
        for wait in waits {
            match wait {
                Wait::Io(ticket) => {
                    self.resolve(ticket.key(), stats)?;
                }
            }
        }
        Ok(())
    }

    /// Consume the bytes behind a ticket, resolving the cell if it is still pending.
    ///
    /// The cell is retained: a straddling morsel that names the same unit reuses it.
    pub fn consume(&self, ticket: IoTicket, stats: &mut ScanStats) -> VortexResult<BufferHandle> {
        self.resolve(ticket.key(), stats)
    }

    fn resolve(&self, key: IoKey, stats: &mut ScanStats) -> VortexResult<BufferHandle> {
        // A pending cell is taken out of the map before it is driven, so the borrow is released
        // before any waker work runs.
        let pending = {
            let mut cells = self.cells.borrow_mut();
            match cells.get(&key) {
                Some(Cell::Ready(handle)) => return Ok(handle.clone()),
                Some(Cell::Pending(_)) => match cells.remove(&key) {
                    Some(Cell::Pending(fut)) => fut,
                    _ => unreachable!("cell was pending"),
                },
                None => {
                    // A cell that was never registered: read it inline (the floor bypass).
                    stats.io_requests += 1;
                    stats.io_bypassed += 1;
                    self.issue(key)
                }
            }
        };

        let handle = block_on(pending)?;
        stats.io_bytes += handle.len() as u64;
        self.cells
            .borrow_mut()
            .insert(key, Cell::Ready(handle.clone()));
        Ok(handle)
    }

    /// Drop every cell. Called between morsel batches to bound retained bytes.
    pub fn clear(&self) {
        self.cells.borrow_mut().clear();
    }

    /// Drop the cell behind a key, if it is resolved.
    pub fn release(&self, key: IoKey) {
        self.cells.borrow_mut().remove(&key);
    }
}

struct ThreadWaker(std::thread::Thread);

impl Wake for ThreadWaker {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}

/// Block the current thread on a future.
///
/// The prototype drives morsels synchronously, so a leaf that must wait parks the OS thread
/// rather than allocating a task. With an in-memory segment source this never actually parks.
fn block_on<T>(mut fut: impl Future<Output = VortexResult<T>> + Unpin) -> VortexResult<T> {
    let waker = Waker::from(Arc::new(ThreadWaker(std::thread::current())));
    let mut cx = Context::from_waker(&waker);
    loop {
        match fut.poll_unpin(&mut cx) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::park(),
        }
    }
}

/// Error helper for a ticket consumed without ever having been planned.
pub fn unplanned_ticket(producer: ProducerId) -> vortex_error::VortexError {
    vortex_err!(
        "node {} waited on a ticket its planning stream never emitted",
        producer.0
    )
}
