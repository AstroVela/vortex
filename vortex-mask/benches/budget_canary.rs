// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TEMPORARY: a deliberately over-budget benchmark, used to prove that the `bench-budget`
//! CI job actually detects and reports a violation.
//!
//! This exists only to exercise the check on the pull request that introduces it, and
//! **must be deleted before merge**. It intentionally violates the "keep per-iteration
//! execution time under ~1 ms" rule in `docs/developer-guide/benchmarking.md`.
//!
//! The work is real rather than a `sleep`, because CodSpeed's simulation instrument
//! excludes system calls -- a sleeping benchmark would look free there while still
//! burning CI wall-clock time.

#![allow(clippy::unwrap_used, clippy::cast_possible_truncation)]

use divan::Bencher;
use vortex_buffer::BitBuffer;
use vortex_mask::Mask;

fn main() {
    divan::main();
}

/// 256Mi bits is a 32 MiB bit buffer. Counting the set bits in a prefix of it is bound by
/// memory bandwidth, which puts a single iteration well past the 1 ms budget on any runner.
const CANARY_LEN: usize = 256 * 1024 * 1024;

#[divan::bench]
fn over_budget_prefix_count(bencher: Bencher) {
    let mask = Mask::from_buffer(BitBuffer::from_iter(
        (0..CANARY_LEN).map(|i| (i * 7 + 13) % 1000 < 900),
    ));
    let indices = [CANARY_LEN / 4, CANARY_LEN - CANARY_LEN / 8];
    bencher
        .with_inputs(|| (&mask, indices))
        .bench_refs(|(mask, indices)| mask.valid_counts_for_indices(indices));
}
