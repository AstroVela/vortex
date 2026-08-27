// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The per-thread decoded-chunk cache.
//!
//! A chunk that straddles several morsels is decoded once per thread rather than once per morsel
//! that touches it. Bounded duplication across threads is deliberate: a shared cache would put a
//! lock on the hottest path in the executor, and re-decoding on another thread is always sound.

use std::cell::RefCell;
use std::collections::VecDeque;

use vortex_array::ArrayRef;
use vortex_layout::segments::SegmentId;
use vortex_utils::aliases::hash_map::HashMap;

use crate::stats::ScanStats;

/// A byte-budgeted, insertion-ordered cache of decoded segments.
pub struct DecodeCache {
    entries: RefCell<HashMap<SegmentId, ArrayRef>>,
    order: RefCell<VecDeque<SegmentId>>,
    bytes: RefCell<usize>,
    budget_bytes: usize,
}

impl DecodeCache {
    /// Create a cache with the given byte budget.
    pub fn new(budget_bytes: usize) -> Self {
        Self {
            entries: RefCell::new(HashMap::default()),
            order: RefCell::new(VecDeque::new()),
            bytes: RefCell::new(0),
            budget_bytes,
        }
    }

    /// Look up a decoded segment.
    pub fn get(&self, id: SegmentId, stats: &mut ScanStats) -> Option<ArrayRef> {
        let hit = self.entries.borrow().get(&id).cloned();
        if hit.is_some() {
            stats.decode_hits += 1;
        }
        hit
    }

    /// Insert a decoded segment, evicting oldest-first until the budget is met.
    pub fn insert(&self, id: SegmentId, array: ArrayRef, bytes: usize, stats: &mut ScanStats) {
        if self.budget_bytes == 0 {
            return;
        }
        {
            let mut entries = self.entries.borrow_mut();
            if entries.insert(id, array).is_none() {
                self.order.borrow_mut().push_back(id);
                *self.bytes.borrow_mut() += bytes;
            }
        }
        self.evict_to_budget(stats);
    }

    fn evict_to_budget(&self, stats: &mut ScanStats) {
        while *self.bytes.borrow() > self.budget_bytes {
            let Some(oldest) = self.order.borrow_mut().pop_front() else {
                break;
            };
            if let Some(array) = self.entries.borrow_mut().remove(&oldest) {
                let freed = array_bytes(&array);
                *self.bytes.borrow_mut() = self.bytes.borrow().saturating_sub(freed);
                stats.decode_evictions += freed as u64;
            }
        }
    }

    /// Drop every entry.
    pub fn clear(&self) {
        self.entries.borrow_mut().clear();
        self.order.borrow_mut().clear();
        *self.bytes.borrow_mut() = 0;
    }
}

/// A cheap size estimate for a decoded array, used only for cache accounting.
pub fn array_bytes(array: &ArrayRef) -> usize {
    usize::try_from(array.nbytes()).unwrap_or(usize::MAX)
}
