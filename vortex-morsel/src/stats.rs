// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Per-run counters. The eval matrix in the prototype plan records these per row.

use std::time::Duration;

/// Counters accumulated by one driving thread, summed across threads at the end of a run.
#[derive(Clone, Debug, Default)]
pub struct ScanStats {
    /// Morsels driven.
    pub morsels: u64,
    /// IO uses named by planning streams.
    pub io_uses: u64,
    /// Reads actually issued to the segment source.
    pub io_requests: u64,
    /// Uses that found an existing cell (a straddling morsel joining a unit already named).
    pub io_cell_hits: u64,
    /// Uses that went through registration.
    pub io_registered: u64,
    /// Uses that skipped registration and read inline (the floor bypass).
    pub io_bypassed: u64,
    /// Bytes returned by the segment source.
    pub io_bytes: u64,
    /// Segment decodes performed.
    pub decodes: u64,
    /// Decodes served from a shared cell published by another morsel.
    pub decode_reuses: u64,
    /// Conjuncts skipped because the mask was already all-false.
    pub conjuncts_short_circuited: u64,
    /// Morsels whose filter selected no rows.
    pub morsels_empty: u64,
    /// Time to the first batch emitted by this thread.
    pub time_to_first_batch: Option<Duration>,
}

impl ScanStats {
    /// Fold another thread's counters into this one.
    pub fn merge(&mut self, other: &ScanStats) {
        self.morsels += other.morsels;
        self.io_uses += other.io_uses;
        self.io_requests += other.io_requests;
        self.io_cell_hits += other.io_cell_hits;
        self.io_registered += other.io_registered;
        self.io_bypassed += other.io_bypassed;
        self.io_bytes += other.io_bytes;
        self.decodes += other.decodes;
        self.decode_reuses += other.decode_reuses;
        self.conjuncts_short_circuited += other.conjuncts_short_circuited;
        self.morsels_empty += other.morsels_empty;
        self.time_to_first_batch = match (self.time_to_first_batch, other.time_to_first_batch) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
    }
}
