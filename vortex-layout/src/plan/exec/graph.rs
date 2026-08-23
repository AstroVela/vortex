// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use smallvec::SmallVec;

use super::ArraySlotId;
use super::FieldId;
use super::FlatEncoding;
use super::MorselId;
use super::ReadPhase;
use super::ResourceId;
use super::SegmentSlotId;
use super::TaskId;
use crate::segments::SegmentId;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourceLifetime {
    Pinned,
    Reusable,
    Dead,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetentionPolicy {
    RetainUntilDead,
    EvictWhenUnpinned,
}

#[derive(Clone, Debug)]
pub(crate) struct ResourceNode {
    pub id: ResourceId,
    pub field: FieldId,
    pub segment: SegmentId,
    pub root_coverage: Range<u64>,
    pub row_count: usize,
    pub estimated_bytes: Option<usize>,
    pub read_phase: ReadPhase,
    pub encoding: FlatEncoding,
    pub segment_slot: SegmentSlotId,
    pub array_slot: ArraySlotId,
    pub unresolved_users: usize,
    pub joined_users: usize,
    pub leases: usize,
    pub read_task: Option<TaskId>,
    pub decode_task: Option<TaskId>,
    pub completed_bytes: Option<usize>,
    pub predicate_consumed: bool,
    pub projection_consumed: bool,
    pub shared_reuse_recorded: bool,
    pub fused_predicate_waiters: SmallVec<[(MorselId, usize, usize); 2]>,
    pub fused_predicate_task_waiters: SmallVec<[(MorselId, usize, usize); 2]>,
}

impl ResourceNode {
    pub fn lifetime(&self) -> ResourceLifetime {
        if self.joined_users != 0 || self.leases != 0 {
            ResourceLifetime::Pinned
        } else if self.unresolved_users != 0 {
            ResourceLifetime::Reusable
        } else {
            ResourceLifetime::Dead
        }
    }
}
