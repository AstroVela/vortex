// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The morsel driver: threads self-schedule morsels off one atomic cursor.
//!
//! There is no coordinator. Each thread takes the next morsel index with a single
//! `fetch_add`, drives it to completion inline on its own arena, and writes the result into its
//! own slot. Emission order is restored by index at the end, so ordering costs one sort rather
//! than a synchronisation point per batch.

use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Instant;

use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_layout::segments::SegmentSource;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;

use crate::build::ExecPlan;
use crate::build::cut_morsels;
use crate::cells::SharedCells;
use crate::io::IoPlane;
use crate::node::drive_morsel;
use crate::stats::ScanStats;

/// The default morsel size in rows.
///
/// Chosen from the SF=1 TPC-H sweep: at 128k rows a morsel is roughly two of the file's ~65k-row
/// natural splits, which is large enough that per-morsel fixed cost is negligible and small
/// enough that a morsel's working set still fits cache. It beats the per-split cut on six of the
/// eight measured queries and is within noise on a seventh.
///
/// The deliberate cost is small scans: a 128k-row morsel means a scan of fewer than
/// `128k * threads` rows cannot fill every core, so short queries lose parallelism. That is
/// accepted — the alternative, sizing morsels to the file's splits, caps the morsel count at a
/// property of the file and starves a many-core host on exactly the large scans that matter.
pub const DEFAULT_MORSEL_ROWS: u64 = 131_072;

/// The morsel row ranges for a plan.
///
/// `target_rows` is a maximum morsel size: zero means one morsel per natural split (the V1 split
/// set, and the fair-comparison default), while a non-zero value both coalesces consecutive
/// splits up to it and subdivides any span longer than it. Subdivision is what decouples the
/// morsel count from the file's split count, which matters as soon as there are more cores than
/// a file has natural splits.
pub fn morsels(plan: &ExecPlan, target_rows: u64) -> Vec<Range<u64>> {
    cut_morsels(plan.natural_splits(), target_rows)
}

/// One configured run of the morsel executor.
pub struct MorselScan {
    plan: Arc<ExecPlan>,
    segments: Arc<dyn SegmentSource>,
    session: VortexSession,
    morsels: Arc<[Range<u64>]>,
    threads: usize,
    inline_floor_bytes: usize,
    share_decodes: bool,
}

impl MorselScan {
    /// Configure a scan over a built plan.
    pub fn new(
        plan: Arc<ExecPlan>,
        segments: Arc<dyn SegmentSource>,
        session: VortexSession,
    ) -> Self {
        let morsels = Arc::from(morsels(&plan, DEFAULT_MORSEL_ROWS));
        Self {
            plan,
            segments,
            session,
            morsels,
            threads: 1,
            inline_floor_bytes: 0,
            share_decodes: true,
        }
    }

    /// Set the number of driving threads.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads.max(1);
        self
    }

    /// Override the morsel cut.
    pub fn with_morsels(mut self, morsels: Vec<Range<u64>>) -> Self {
        self.morsels = Arc::from(morsels);
        self
    }

    /// Set the size at or below which a read bypasses registration.
    pub fn with_inline_floor_bytes(mut self, bytes: usize) -> Self {
        self.inline_floor_bytes = bytes;
        self
    }

    /// Enable or disable the leased shared decoded cells.
    ///
    /// Disabled, the executor holds no state across morsels — the configuration whose retained
    /// state exactly matches the V1 `LayoutReader`, kept for fair comparison and as the chaos
    /// check that sharing changes no output.
    pub fn with_share_decodes(mut self, share: bool) -> Self {
        self.share_decodes = share;
        self
    }

    /// Lease counts for the shared cells: for each stored unit, the number of (node, morsel)
    /// pairs whose ranges overlap. This is the same arithmetic planning runs per morsel, summed
    /// over the cut up front, which is what makes release-at-retire drain every count to zero.
    fn lease_counts(&self) -> HashMap<crate::io::IoKey, usize> {
        let mut counts: HashMap<crate::io::IoKey, usize> = HashMap::default();
        for (key, range) in self.plan.flat_uses() {
            let overlapping = self
                .morsels
                .iter()
                .filter(|morsel| morsel.start < range.end && range.start < morsel.end)
                .count();
            if overlapping > 0 {
                *counts.entry(key).or_default() += overlapping;
            }
        }
        counts
    }

    /// The morsels this scan will drive.
    pub fn morsel_ranges(&self) -> &[Range<u64>] {
        &self.morsels
    }

    /// Run the scan, returning batches in row order plus the run's counters.
    pub fn run(&self) -> VortexResult<(Vec<ArrayRef>, ScanStats)> {
        let cells = if self.share_decodes {
            SharedCells::with_leases(self.lease_counts())
        } else {
            SharedCells::disabled()
        };
        let cursor = AtomicUsize::new(0);
        let results: Mutex<Vec<(usize, ArrayRef)>> = Mutex::new(Vec::new());
        let stats = Mutex::new(ScanStats::default());
        let start = Instant::now();

        let run_thread = || -> VortexResult<()> {
            let mut arena = self.plan.instantiate();
            let io = IoPlane::new(Arc::clone(&self.segments))
                .with_inline_floor_bytes(self.inline_floor_bytes);
            let mut local = ScanStats::default();
            let mut local_results: Vec<(usize, ArrayRef)> = Vec::new();

            loop {
                let idx = cursor.fetch_add(1, Ordering::Relaxed);
                let Some(range) = self.morsels.get(idx) else {
                    break;
                };
                let batch = drive_morsel(
                    &mut arena,
                    self.plan.root(),
                    range.clone(),
                    &io,
                    &cells,
                    &self.session,
                    &mut local,
                )?;
                if let Some(batch) = batch {
                    if local.time_to_first_batch.is_none() {
                        local.time_to_first_batch = Some(start.elapsed());
                    }
                    local_results.push((idx, batch));
                }
            }

            results.lock().extend(local_results);
            stats.lock().merge(&local);
            Ok(())
        };

        if self.threads == 1 {
            run_thread()?;
        } else {
            std::thread::scope(|scope| -> VortexResult<()> {
                let handles: Vec<_> = (0..self.threads).map(|_| scope.spawn(run_thread)).collect();
                for handle in handles {
                    handle
                        .join()
                        .map_err(|_| vortex_err!("morsel driver thread panicked"))??;
                }
                Ok(())
            })?;
        }

        debug_assert_eq!(
            cells.live(),
            0,
            "every lease must be released by the end of the scan"
        );

        // Ordering is restored by index, not maintained during execution.
        let mut ordered = results.into_inner();
        ordered.sort_unstable_by_key(|(idx, _)| *idx);
        let batches = ordered.into_iter().map(|(_, array)| array).collect();
        let stats = stats.into_inner();
        Ok((batches, stats))
    }
}
