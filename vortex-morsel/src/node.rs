// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The [`ExecNode`] contract and the arena that drives it.

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::buffer::BufferHandle;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_layout::segments::SegmentId;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::cache::DecodeCache;
use crate::io::IoBatch;
use crate::io::IoPlane;
use crate::io::IoTicket;
use crate::stats::ScanStats;

/// Index of a node within an [`Arena`].
pub type NodeId = u32;

/// A value produced by a node for its parent.
#[derive(Clone)]
pub enum Value {
    /// Dense rows: length equals the true count of the demand mask the node was executed under.
    Array(ArrayRef),
    /// A refinement of the demand mask the node was executed under; same length as that mask.
    Mask(Mask),
}

impl Value {
    /// Unwrap an array value, or fail if this is a mask.
    pub fn into_array(self) -> VortexResult<ArrayRef> {
        match self {
            Value::Array(array) => Ok(array),
            Value::Mask(_) => Err(vortex_err!("expected an array value, got a mask")),
        }
    }

    /// Unwrap a mask value, or fail if this is an array.
    pub fn into_mask(self) -> VortexResult<Mask> {
        match self {
            Value::Mask(mask) => Ok(mask),
            Value::Array(_) => Err(vortex_err!("expected a mask value, got an array")),
        }
    }
}

/// A value plus the dense range of *input* rows it accounts for.
pub struct ValueBatch {
    /// The root-coordinate row range this batch accounts for.
    pub coverage: Range<u64>,
    /// The value itself.
    pub value: Value,
}

/// What a node's planning stream produced.
pub enum PlanItem {
    /// A batch of named IO uses, already registered with the IO plane.
    Io(IoBatch),
    /// The node yielded before refining further; call `next_plan` again to resume.
    Plan,
}

/// The result of polling a node's planning stream.
pub enum PlanPoll {
    /// An item was produced.
    Item(PlanItem),
    /// Planning is parked on the given waits.
    Blocked(WaitSet),
    /// Planning has finished. This forfeits any further refinement of this node's IO.
    Complete,
}

/// The result of polling a node's execution.
pub enum ExecPoll {
    /// A value covering a dense input row range.
    Value(ValueBatch),
    /// Execution is parked on the given waits.
    Blocked(WaitSet),
    /// The node made progress but has not produced a value yet.
    Yield(Progress),
    /// The node has produced everything it will produce.
    Done,
}

/// A coarse progress marker returned with [`ExecPoll::Yield`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Progress {
    /// Rows of input consumed since the last poll.
    pub rows: u64,
}

/// Something a node can park on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Wait {
    /// An IO ticket the node's own planning stream emitted.
    Io(IoTicket),
}

/// A set of [`Wait`]s. Small by construction — a node parks on the handful of cells it named.
#[derive(Clone, Debug, Default)]
pub struct WaitSet(Vec<Wait>);

impl WaitSet {
    /// An empty wait set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Park on one more thing.
    pub fn push(&mut self, wait: Wait) {
        self.0.push(wait);
    }

    /// The waits in this set.
    pub fn waits(&self) -> &[Wait] {
        &self.0
    }

    /// Whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl FromIterator<Wait> for WaitSet {
    fn from_iter<T: IntoIterator<Item = Wait>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

/// A stateful, per-morsel execution node.
///
/// Nodes are arena-allocated once per driving thread and reset per morsel, so `&mut self` state
/// survives a suspension without allocating a task.
pub trait ExecNode: Send {
    /// Reset this node for a new morsel covering `range` (in this node's local coordinates).
    fn reset(&mut self, range: Range<u64>);

    /// Advance this node's planning stream.
    ///
    /// Planning only names IO; it never reads. A node that has more planning to do than its
    /// budget allows returns [`PlanItem::Plan`] and resumes from its own cursor.
    fn next_plan(&mut self, cx: &mut PlanCx<'_>) -> VortexResult<PlanPoll>;

    /// Advance this node's execution, producing values under the demand in `cx`.
    fn execute(&mut self, cx: &mut ExecCx<'_>) -> VortexResult<ExecPoll>;

    /// Release anything this node holds for the finished morsel.
    fn retire(&mut self, cx: &mut RetireCx<'_>);

    /// This node's children, in edge order.
    fn children(&self) -> &[NodeId];
}

/// An arena of nodes, owned by one driving thread and reused across morsels.
pub struct Arena {
    nodes: Vec<Option<Box<dyn ExecNode>>>,
}

impl Arena {
    /// Build an arena from a list of nodes.
    pub fn new(nodes: Vec<Box<dyn ExecNode>>) -> Self {
        Self {
            nodes: nodes.into_iter().map(Some).collect(),
        }
    }

    /// The number of nodes in the arena.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the arena is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Take a node out of the arena so its children can be driven through the remaining slots.
    ///
    /// The node must be put back with [`Arena::put`]. The take/put pair is what lets a node hold
    /// `&mut self` while recursively driving its children: the tree shape guarantees a node is
    /// never reachable from its own subtree, so a taken slot is never observed as empty.
    fn take(&mut self, id: NodeId) -> Box<dyn ExecNode> {
        self.nodes[id as usize].take().unwrap_or_else(|| {
            vortex_panic!("node {id} is already being driven: the exec graph is not a tree")
        })
    }

    fn put(&mut self, id: NodeId, node: Box<dyn ExecNode>) {
        self.nodes[id as usize] = Some(node);
    }

    /// Reset the subtree rooted at `id` for a morsel covering `range`.
    pub fn reset_subtree(&mut self, id: NodeId, range: Range<u64>) {
        let mut node = self.take(id);
        node.reset(range);
        self.put(id, node);
    }
}

/// Context handed to [`ExecNode::next_plan`].
pub struct PlanCx<'a> {
    arena: &'a mut Arena,
    io: &'a IoPlane,
    stats: &'a mut ScanStats,
    /// Remaining IO uses this planning quantum may emit before the node should yield.
    budget: u32,
}

impl<'a> PlanCx<'a> {
    /// The remaining planning budget, in IO uses.
    pub fn budget(&self) -> u32 {
        self.budget
    }

    /// Whether the planning quantum is exhausted.
    pub fn out_of_budget(&self) -> bool {
        self.budget == 0
    }

    /// Register a batch of IO uses, spending budget and returning tickets.
    pub fn register(&mut self, batch: IoBatch) -> VortexResult<Vec<IoTicket>> {
        self.budget = self
            .budget
            .saturating_sub(u32::try_from(batch.uses().len()).unwrap_or(u32::MAX));
        self.stats.io_uses += batch.uses().len() as u64;
        self.io.register(batch, self.stats)
    }

    /// Drive a child's planning stream to completion, cutting it to `range` first.
    ///
    /// Returns `true` when the child completed, `false` when the shared budget ran out and the
    /// caller should yield and resume at this child.
    pub fn plan_child(&mut self, id: NodeId, range: Range<u64>, fresh: bool) -> VortexResult<bool> {
        let mut node = self.arena.take(id);
        let result = (|| {
            if fresh {
                node.reset(range);
            }
            loop {
                match node.next_plan(self)? {
                    PlanPoll::Item(PlanItem::Io(_)) => continue,
                    PlanPoll::Item(PlanItem::Plan) => return Ok(false),
                    PlanPoll::Blocked(_) => {
                        // P1 has no gated planning: nothing can park a planning stream.
                        return Ok(false);
                    }
                    PlanPoll::Complete => return Ok(true),
                }
            }
        })();
        self.arena.put(id, node);
        result
    }
}

/// Context handed to [`ExecNode::execute`].
pub struct ExecCx<'a> {
    arena: &'a mut Arena,
    io: &'a IoPlane,
    cache: &'a DecodeCache,
    session: &'a VortexSession,
    stats: &'a mut ScanStats,
    demand: Mask,
}

impl<'a> ExecCx<'a> {
    /// The demand mask this node is executing under.
    ///
    /// Its length equals the number of rows in the node's local range; the node must produce
    /// exactly `demand().true_count()` rows.
    pub fn demand(&self) -> &Mask {
        &self.demand
    }

    /// The IO plane, for consuming cells behind tickets this node's planning stream emitted.
    pub fn io(&self) -> &IoPlane {
        self.io
    }

    /// The per-thread decoded-chunk cache.
    pub fn cache(&self) -> &DecodeCache {
        self.cache
    }

    /// The session, for creating expression execution contexts.
    pub fn session(&self) -> &VortexSession {
        self.session
    }

    /// Look up a decoded segment in the per-thread cache.
    pub fn cache_get(&mut self, id: SegmentId) -> Option<ArrayRef> {
        self.cache.get(id, self.stats)
    }

    /// Insert a decoded segment into the per-thread cache.
    pub fn cache_insert(&mut self, id: SegmentId, array: ArrayRef, bytes: usize) {
        self.cache.insert(id, array, bytes, self.stats)
    }

    /// Consume the bytes behind a ticket this node's planning stream emitted.
    pub fn consume(&mut self, ticket: IoTicket) -> VortexResult<BufferHandle> {
        self.io.consume(ticket, self.stats)
    }

    /// Mutable access to the run's counters.
    pub fn stats(&mut self) -> &mut ScanStats {
        self.stats
    }

    /// Drive a child to a value under `demand`.
    ///
    /// The child is polled until it yields a value or reports `Done`; `Blocked` is impossible in
    /// P1 because every ticket a node parks on is already resolvable inline.
    pub fn child_value(&mut self, id: NodeId, demand: Mask) -> VortexResult<Option<ValueBatch>> {
        let mut node = self.arena.take(id);
        let saved = std::mem::replace(&mut self.demand, demand);
        let result = (|| {
            loop {
                match node.execute(self)? {
                    ExecPoll::Value(batch) => return Ok(Some(batch)),
                    ExecPoll::Yield(_) => continue,
                    ExecPoll::Blocked(waits) => {
                        self.io.wait(waits.waits(), self.stats)?;
                    }
                    ExecPoll::Done => return Ok(None),
                }
            }
        })();
        self.demand = saved;
        self.arena.put(id, node);
        result
    }

    /// Drive a child to an array value, failing if it produced nothing.
    pub fn child_array(&mut self, id: NodeId, demand: Mask) -> VortexResult<ArrayRef> {
        self.child_value(id, demand)?
            .ok_or_else(|| vortex_err!("child node {id} produced no value"))?
            .value
            .into_array()
    }

    /// Drive a child to a mask value, failing if it produced nothing.
    pub fn child_mask(&mut self, id: NodeId, demand: Mask) -> VortexResult<Mask> {
        self.child_value(id, demand)?
            .ok_or_else(|| vortex_err!("child node {id} produced no value"))?
            .value
            .into_mask()
    }
}

/// Context handed to [`ExecNode::retire`].
pub struct RetireCx<'a> {
    arena: &'a mut Arena,
    stats: &'a mut ScanStats,
}

impl<'a> RetireCx<'a> {
    /// Retire a child subtree.
    pub fn retire_child(&mut self, id: NodeId) {
        let mut node = self.arena.take(id);
        node.retire(self);
        self.arena.put(id, node);
    }

    /// Mutable access to the run's counters.
    pub fn stats(&mut self) -> &mut ScanStats {
        self.stats
    }
}

/// The number of IO uses one planning quantum may emit before a node should yield.
pub const PLAN_BUDGET: u32 = 64;

/// Drive one morsel through the arena: plan, execute, retire.
pub fn drive_morsel(
    arena: &mut Arena,
    root: NodeId,
    range: Range<u64>,
    io: &IoPlane,
    cache: &DecodeCache,
    session: &VortexSession,
    stats: &mut ScanStats,
) -> VortexResult<Option<ArrayRef>> {
    arena.reset_subtree(root, range.clone());

    // Planning: name every read this morsel will make, in budget-bounded quanta.
    loop {
        let mut cx = PlanCx {
            arena,
            io,
            stats,
            budget: PLAN_BUDGET,
        };
        let mut node = cx.arena.take(root);
        let poll = node.next_plan(&mut cx);
        cx.arena.put(root, node);
        match poll? {
            PlanPoll::Item(PlanItem::Io(_)) | PlanPoll::Item(PlanItem::Plan) => continue,
            PlanPoll::Blocked(_) => break,
            PlanPoll::Complete => break,
        }
    }
    stats.morsels += 1;

    // Execution.
    let rows = usize::try_from(range.end - range.start)
        .map_err(|_| vortex_err!("morsel row count exceeds usize"))?;
    let value = {
        let mut cx = ExecCx {
            arena,
            io,
            cache,
            session,
            stats,
            demand: Mask::new_true(rows),
        };
        let mut node = cx.arena.take(root);
        let out = loop {
            match node.execute(&mut cx) {
                Ok(ExecPoll::Value(batch)) => break Ok(Some(batch)),
                Ok(ExecPoll::Yield(_)) => continue,
                Ok(ExecPoll::Blocked(waits)) => {
                    if let Err(err) = cx.io.wait(waits.waits(), cx.stats) {
                        break Err(err);
                    }
                }
                Ok(ExecPoll::Done) => break Ok(None),
                Err(err) => break Err(err),
            }
        };
        cx.arena.put(root, node);
        out?
    };

    // Retirement.
    {
        let mut cx = RetireCx { arena, stats };
        cx.retire_child(root);
    }

    let array = value.map(|batch| batch.value.into_array()).transpose()?;
    Ok(array.and_then(|a| (!a.is_empty()).then_some(a)))
}
