// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for the per-split zone-mask expansion in `ZonedReader::pruning_evaluation`.
//!
//! For every split, the reader turns the cached zone-level pruning mask into a row-aligned
//! stats mask and intersects it with the incoming mask. When every zone covering the split
//! agrees - none pruned, or all pruned - that expansion collapses to a constant, because
//! `Mask::from_buffer` canonicalises an all-ones buffer to `AllTrue` and an all-zeros buffer
//! to `AllFalse`, and owned-left `bitand` short circuits on both. The expanded buffer is
//! built, counted, and thrown away.
//!
//! `expand` is the old behaviour and `uniform` is the fast path that reads the covering zone
//! bits directly. `mixed` covers the case where the zones genuinely disagree and the
//! expansion still has to run, bounding what the added `count_range` costs when it cannot
//! short circuit.

#![allow(clippy::cast_possible_truncation)]

use std::hint::black_box;
use std::ops::BitAnd;

use divan::Bencher;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_mask::AllOr;
use vortex_mask::Mask;

fn main() {
    divan::main();
}

/// `(split_len, zones_per_split)`. The default zone and block length are both 8192 rows, so
/// one zone per split is the common shape; the wider cases cover multi-zone splits.
const SPLITS: &[(usize, usize)] = &[(8_192, 1), (8_192, 4), (65_536, 8)];

/// Splits with at least two covering zones, so the zones can actually disagree. A one-zone
/// split is uniform by construction and would silently measure a fast path instead.
const MIXED_SPLITS: &[(usize, usize)] = &[(8_192, 4), (65_536, 8)];

/// A realistic incoming scan mask: `Values`, not `AllTrue`, so `bitand` takes the
/// `(AllOr::Some, AllOr::All) => self` branch rather than the all-true short circuit.
fn incoming(len: usize) -> Mask {
    Mask::from_buffer(BitBuffer::from_iter((0..len).map(|i| i % 3 != 0)))
}

fn zone_lengths(split_len: usize, zones: usize) -> Vec<usize> {
    let per_zone = split_len / zones;
    (0..zones)
        .map(|z| {
            if z == zones - 1 {
                split_len - per_zone * (zones - 1)
            } else {
                per_zone
            }
        })
        .collect()
}

/// The zone-level pruning mask, restricted to the zones covering one split.
fn pruning_mask(zones: usize, pruned: impl Fn(usize) -> bool) -> Mask {
    Mask::from_buffer(BitBuffer::from_iter((0..zones).map(pruned)))
}

/// Expand the covering zones into a row-aligned buffer, then intersect. This is what ran for
/// every split before the uniform fast path.
fn expand(mask: Mask, pruning_mask: &Mask, zone_lengths: &[usize]) -> Mask {
    let mut builder = BitBufferMut::with_capacity(mask.len());
    for (zone_idx, &zone_length) in zone_lengths.iter().enumerate() {
        builder.append_n(!pruning_mask.value(zone_idx), zone_length);
    }
    let stats_mask = Mask::from(builder.freeze());
    mask.bitand(&stats_mask)
}

/// Count the pruned bits among the covering zones first, and skip the expansion when they
/// all agree.
fn uniform(mask: Mask, pruning_mask: &Mask, zone_lengths: &[usize]) -> Mask {
    let covered = zone_lengths.len();
    let pruned = match pruning_mask.bit_buffer() {
        AllOr::All => covered,
        AllOr::None => 0,
        AllOr::Some(buffer) => buffer.count_range(0, covered),
    };

    if pruned == 0 {
        mask
    } else if pruned == covered {
        Mask::new_false(mask.len())
    } else {
        expand(mask, pruning_mask, zone_lengths)
    }
}

/// No covering zone is pruned: the stats mask is all-true and the intersection is a no-op.
#[divan::bench(args = SPLITS)]
fn no_zone_pruned(bencher: Bencher, (split_len, zones): (usize, usize)) {
    let lengths = zone_lengths(split_len, zones);
    let pruning = pruning_mask(zones, |_| false);
    let mask = incoming(split_len);

    bencher
        .with_inputs(|| mask.clone())
        .bench_values(|mask| black_box(uniform(mask, &pruning, &lengths)));
}

/// The same split through the old expansion, for comparison.
#[divan::bench(args = SPLITS)]
fn no_zone_pruned_expanded(bencher: Bencher, (split_len, zones): (usize, usize)) {
    let lengths = zone_lengths(split_len, zones);
    let pruning = pruning_mask(zones, |_| false);
    let mask = incoming(split_len);

    bencher
        .with_inputs(|| mask.clone())
        .bench_values(|mask| black_box(expand(mask, &pruning, &lengths)));
}

/// Every covering zone is pruned: the stats mask is all-false and the split drops out.
#[divan::bench(args = SPLITS)]
fn all_zones_pruned(bencher: Bencher, (split_len, zones): (usize, usize)) {
    let lengths = zone_lengths(split_len, zones);
    let pruning = pruning_mask(zones, |_| true);
    let mask = incoming(split_len);

    bencher
        .with_inputs(|| mask.clone())
        .bench_values(|mask| black_box(uniform(mask, &pruning, &lengths)));
}

/// The same split through the old expansion, for comparison.
#[divan::bench(args = SPLITS)]
fn all_zones_pruned_expanded(bencher: Bencher, (split_len, zones): (usize, usize)) {
    let lengths = zone_lengths(split_len, zones);
    let pruning = pruning_mask(zones, |_| true);
    let mask = incoming(split_len);

    bencher
        .with_inputs(|| mask.clone())
        .bench_values(|mask| black_box(expand(mask, &pruning, &lengths)));
}

/// The covering zones disagree, so the expansion still runs and the added `count_range` is
/// the whole overhead.
#[divan::bench(args = MIXED_SPLITS)]
fn mixed_zones(bencher: Bencher, (split_len, zones): (usize, usize)) {
    let lengths = zone_lengths(split_len, zones);
    let pruning = pruning_mask(zones, |z| z % 2 == 0);
    let mask = incoming(split_len);

    bencher
        .with_inputs(|| mask.clone())
        .bench_values(|mask| black_box(uniform(mask, &pruning, &lengths)));
}

/// The same split through the old expansion, for comparison.
#[divan::bench(args = MIXED_SPLITS)]
fn mixed_zones_expanded(bencher: Bencher, (split_len, zones): (usize, usize)) {
    let lengths = zone_lengths(split_len, zones);
    let pruning = pruning_mask(zones, |z| z % 2 == 0);
    let mask = incoming(split_len);

    bencher
        .with_inputs(|| mask.clone())
        .bench_values(|mask| black_box(expand(mask, &pruning, &lengths)));
}
