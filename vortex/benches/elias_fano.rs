// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Evaluate Elias-Fano coding of monotone integer sequences against Vortex's
//! Delta + FoR + BitPacking cascade.
//!
//! Three Elias-Fano variants are implemented here, self-contained:
//!
//! - **EF**: plain Elias-Fano over the whole sequence (relative to its first value),
//!   `~ 2 + ceil(log2(universe / n))` bits per element.
//! - **PEF uniform**: partitioned Elias-Fano ([Ottaviano & Venturini, SIGIR'14]) with
//!   fixed-size partitions. Each partition picks the cheapest of three representations:
//!   implicit all-ones run, plain bitvector, or Elias-Fano, all relative to the partition's
//!   bounds. Partition upper bounds are themselves Elias-Fano coded.
//! - **PEF opt**: the same, but partition boundaries are chosen by a shortest-path
//!   dynamic program over candidate boundaries (multiples of 64, partitions capped at
//!   4096 elements), approximating the paper's optimal partitioning.
//!
//! The Vortex baseline is the real cascade the BtrBlocks-style compressor builds for
//! smooth integers: `Delta(bases: FoR+BitPacked, deltas: FoR+BitPacked)`, sized via
//! `Array::nbytes`. The default `BtrBlocksCompressor` pick is also reported for context.
//!
//! Running the bench first prints a compressed-size table (bits per element) for several
//! monotone integer distributions and verifies all schemes round-trip, then runs divan
//! throughput benchmarks for encode and full decode.
//!
//! Run with `cargo bench -p vortex --bench elias_fano`.
//! For the size table only: `cargo bench -p vortex --bench elias_fano -- --list`.
//!
//! [Ottaviano & Venturini, SIGIR'14]: https://dl.acm.org/doi/10.1145/2600428.2609615

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

use std::collections::BTreeMap;
use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::validity::Validity;
use vortex::buffer::Buffer;
use vortex::compressor::BtrBlocksCompressor;
use vortex::encodings::fastlanes::Delta;
use vortex::encodings::fastlanes::FoR;
use vortex::encodings::fastlanes::FoRArrayExt;
use vortex::encodings::fastlanes::FoRArraySlotsExt;
use vortex::encodings::fastlanes::FoRData;
use vortex::encodings::fastlanes::bitpack_compress::bitpack_to_best_bit_width;
use vortex::encodings::fastlanes::delta_compress;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::default);

const N: usize = 1 << 20;

// ---------------------------------------------------------------------------
// Bit utilities
// ---------------------------------------------------------------------------

/// Fixed-width bit vector with `width`-bit unaligned writes and reads.
fn bits_to_words(bits: usize) -> usize {
    // One padding word so unaligned reads never index past the end.
    bits.div_ceil(64) + 1
}

fn set_bits(words: &mut [u64], bit_pos: usize, value: u64, width: u8) {
    if width == 0 {
        return;
    }
    let word = bit_pos >> 6;
    let offset = bit_pos & 63;
    words[word] |= value << offset;
    if offset + width as usize > 64 {
        words[word + 1] |= value >> (64 - offset);
    }
}

fn get_bits(words: &[u64], bit_pos: usize, width: u8) -> u64 {
    if width == 0 {
        return 0;
    }
    let word = bit_pos >> 6;
    let offset = bit_pos & 63;
    let mask = u64::MAX >> (64 - width as u32);
    if offset + width as usize <= 64 {
        (words[word] >> offset) & mask
    } else {
        ((words[word] >> offset) | (words[word + 1] << (64 - offset))) & mask
    }
}

// ---------------------------------------------------------------------------
// Plain Elias-Fano
// ---------------------------------------------------------------------------

/// Elias-Fano coding of a non-decreasing sequence of `n` integers in `[0, universe)`.
///
/// The low `l = floor(log2(universe / n))` bits of each element are stored verbatim; the
/// high bits are stored as a unary-coded bitvector of `n + (universe >> l) + 1` bits.
struct EliasFano {
    n: usize,
    low_width: u8,
    /// Logical length of the unary high-bits bitvector: `n + (universe >> l) + 1`.
    high_bits: usize,
    lows: Vec<u64>,
    highs: Vec<u64>,
}

fn ef_low_width(n: usize, universe: u64) -> u8 {
    if n == 0 || universe <= n as u64 {
        0
    } else {
        (universe / n as u64).ilog2() as u8
    }
}

/// Exact size in bits of an Elias-Fano coding of `n` elements in `[0, universe)`.
fn ef_cost_bits(n: usize, universe: u64) -> u64 {
    let l = ef_low_width(n, universe);
    n as u64 * l as u64 + (n as u64 + (universe >> l) + 1)
}

impl EliasFano {
    /// `values` must be non-decreasing with every element `< universe`.
    fn encode(values: &[u64], universe: u64) -> Self {
        let n = values.len();
        let l = ef_low_width(n, universe);
        let high_bits = n + (universe >> l) as usize + 1;
        let mut lows = vec![0u64; bits_to_words(n * l as usize)];
        let mut highs = vec![0u64; bits_to_words(high_bits)];
        for (i, &x) in values.iter().enumerate() {
            if l > 0 {
                set_bits(
                    &mut lows,
                    i * l as usize,
                    x & (u64::MAX >> (64 - l as u32)),
                    l,
                );
            }
            let high_pos = (x >> l) as usize + i;
            highs[high_pos >> 6] |= 1u64 << (high_pos & 63);
        }
        Self {
            n,
            low_width: l,
            high_bits,
            lows,
            highs,
        }
    }

    /// Logical size in bits (n*l low bits + unary high bitvector), excluding headers.
    fn size_bits(&self) -> u64 {
        self.n as u64 * self.low_width as u64 + self.high_bits as u64
    }

    /// Append all decoded values, each offset by `add`, to `out`.
    fn decode_append(&self, out: &mut Vec<u64>, add: u64) {
        let low_width = self.low_width;
        let mut idx = 0usize;
        for (word_idx, &word) in self.highs.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let pos = (word_idx << 6) + bits.trailing_zeros() as usize;
                let high = (pos - idx) as u64;
                let low = get_bits(&self.lows, idx * low_width as usize, low_width);
                out.push(add + ((high << low_width) | low));
                idx += 1;
                bits &= bits - 1;
                if idx == self.n {
                    return;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Partitioned Elias-Fano
// ---------------------------------------------------------------------------

/// Per-partition representation, chosen by exact bit cost.
enum PefPartition {
    /// The partition is a contiguous run `lo..=ub`: zero bits.
    AllOnes,
    /// Bitvector over the partition's sub-universe.
    BitVec(Vec<u64>),
    /// Elias-Fano over values relative to the partition's lower bound.
    Ef(EliasFano),
}

struct PefEntry {
    lo: u64,
    m: usize,
    enc: PefPartition,
}

struct PartitionedEliasFano {
    n: usize,
    /// Elias-Fano over partition upper bounds (the "first level" of the index).
    upper_bounds: EliasFano,
    /// Elias-Fano over cumulative partition sizes; None for uniform partitions where
    /// the size is a constant stored in the header.
    boundaries: Option<EliasFano>,
    partitions: Vec<PefEntry>,
}

/// Bits for a bitvector partition representation.
fn bv_cost_bits(sub_universe: u64) -> u64 {
    sub_universe
}

/// Exact storage bits for the cheapest representation of a partition with `m` elements
/// and sub-universe `u` (values relative to the partition lower bound are in `[0, u)`).
fn partition_cost_bits(m: usize, sub_universe: u64) -> u64 {
    if m as u64 == sub_universe {
        return 0; // all-ones
    }
    bv_cost_bits(sub_universe).min(ef_cost_bits(m, sub_universe))
}

/// Per-partition fixed overhead used by the partitioning DP: a 2-bit representation tag
/// plus the amortized cost of one entry in each first-level Elias-Fano sequence.
const PARTITION_FIXED_BITS: u64 = 64;

impl PartitionedEliasFano {
    /// Encode a strictly increasing sequence using the given partition boundaries
    /// (`bounds[k]..bounds[k+1]` index ranges, with implicit 0 and n endpoints).
    fn encode_with_boundaries(values: &[u64], bounds: &[usize], uniform: bool) -> Self {
        let n = values.len();
        let mut partitions = Vec::with_capacity(bounds.len());
        let mut ubs = Vec::with_capacity(bounds.len());
        let mut start = 0usize;
        for &end in bounds {
            let part = &values[start..end];
            let m = part.len();
            let lo = if start == 0 {
                part[0]
            } else {
                values[start - 1] + 1
            };
            let ub = part[m - 1];
            let sub_universe = ub - lo + 1;
            let rel: Vec<u64> = part.iter().map(|&x| x - lo).collect();
            let enc = if m as u64 == sub_universe {
                PefPartition::AllOnes
            } else if bv_cost_bits(sub_universe) <= ef_cost_bits(m, sub_universe) {
                let mut words = vec![0u64; bits_to_words(sub_universe as usize)];
                for &r in &rel {
                    words[(r >> 6) as usize] |= 1u64 << (r & 63);
                }
                PefPartition::BitVec(words)
            } else {
                PefPartition::Ef(EliasFano::encode(&rel, sub_universe))
            };
            partitions.push(PefEntry { lo, m, enc });
            ubs.push(ub);
            start = end;
        }
        let upper_bounds = EliasFano::encode(&ubs, ubs.last().unwrap() + 1);
        let boundaries = (!uniform).then(|| {
            let cumulative: Vec<u64> = bounds.iter().map(|&b| b as u64).collect();
            EliasFano::encode(&cumulative, n as u64 + 1)
        });
        Self {
            n,
            upper_bounds,
            boundaries,
            partitions,
        }
    }

    fn encode_uniform(values: &[u64], partition_size: usize) -> Self {
        let bounds: Vec<usize> = (1..=values.len().div_ceil(partition_size))
            .map(|k| (k * partition_size).min(values.len()))
            .collect();
        Self::encode_with_boundaries(values, &bounds, true)
    }

    /// Choose partition boundaries with a shortest-path DP over candidate boundaries at
    /// multiples of `QUANTUM`, with partitions capped at `MAX_PARTITION` elements.
    fn encode_opt(values: &[u64]) -> Self {
        const QUANTUM: usize = 64;
        const MAX_PARTITION: usize = 4096;
        let len = values.len();
        let num_nodes = len.div_ceil(QUANTUM);
        let pos = |node: usize| (node * QUANTUM).min(len);
        // dp[node] = (cost of encoding values[..pos(node)], predecessor node)
        let mut dp: Vec<(u64, usize)> = vec![(u64::MAX, 0); num_nodes + 1];
        dp[0] = (0, 0);
        for node in 1..=num_nodes {
            let first_pred = node.saturating_sub(MAX_PARTITION / QUANTUM);
            for pred in first_pred..node {
                let (start, end) = (pos(pred), pos(node));
                let lo = if start == 0 {
                    values[0]
                } else {
                    values[start - 1] + 1
                };
                let sub_universe = values[end - 1] - lo + 1;
                let cost = dp[pred].0
                    + partition_cost_bits(end - start, sub_universe)
                    + PARTITION_FIXED_BITS;
                if cost < dp[node].0 {
                    dp[node] = (cost, pred);
                }
            }
        }
        let mut bounds = Vec::new();
        let mut node = num_nodes;
        while node > 0 {
            bounds.push(pos(node));
            node = dp[node].1;
        }
        bounds.reverse();
        Self::encode_with_boundaries(values, &bounds, false)
    }

    /// Logical compressed size in bits: first-level sequences plus each partition's
    /// representation and a 2-bit representation tag.
    fn size_bits(&self) -> u64 {
        let mut bits = 128; // n, universe, flags
        bits += self.upper_bounds.size_bits();
        bits += self.boundaries.as_ref().map_or(64, EliasFano::size_bits);
        for p in &self.partitions {
            bits += 2;
            bits += match &p.enc {
                PefPartition::AllOnes => 0,
                PefPartition::BitVec(_) => {
                    let last = self.partition_ub(p);
                    last - p.lo + 1
                }
                PefPartition::Ef(ef) => ef.size_bits(),
            };
        }
        bits
    }

    fn partition_ub(&self, p: &PefEntry) -> u64 {
        // Upper bound is recoverable from the encodings; recompute for size accounting.
        match &p.enc {
            PefPartition::AllOnes => p.lo + p.m as u64 - 1,
            PefPartition::BitVec(words) => {
                let w_idx = words.iter().rposition(|&w| w != 0).unwrap();
                p.lo + (w_idx as u64 * 64 + 63 - words[w_idx].leading_zeros() as u64)
            }
            PefPartition::Ef(ef) => {
                let mut tmp = Vec::with_capacity(ef.n);
                ef.decode_append(&mut tmp, p.lo);
                *tmp.last().unwrap()
            }
        }
    }

    fn decode(&self) -> Vec<u64> {
        let mut out = Vec::with_capacity(self.n);
        for p in &self.partitions {
            match &p.enc {
                PefPartition::AllOnes => out.extend(p.lo..p.lo + p.m as u64),
                PefPartition::BitVec(words) => {
                    for (w_idx, &word) in words.iter().enumerate() {
                        let mut w = word;
                        while w != 0 {
                            out.push(p.lo + ((w_idx as u64) << 6) + w.trailing_zeros() as u64);
                            w &= w - 1;
                        }
                    }
                }
                PefPartition::Ef(ef) => ef.decode_append(&mut out, p.lo),
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Vortex Delta + FoR + BitPacking baseline
// ---------------------------------------------------------------------------

/// FoR-then-bitpack, the tail of the compressor's integer cascade. Falls back to the
/// plain FoR (or raw primitive) child when bitpacking cannot shrink the array.
fn for_bitpack(array: PrimitiveArray, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let for_array = FoRData::encode(array, ctx)?;
    let reference = for_array.reference_scalar().clone();
    let child = for_array.encoded().clone().execute::<PrimitiveArray>(ctx)?;
    match bitpack_to_best_bit_width(&child, ctx) {
        Ok(packed) => Ok(FoR::try_new(packed.into_array(), reference)?.into_array()),
        Err(_) => Ok(for_array.into_array()),
    }
}

/// The cascade the Vortex compressor builds for smooth integer sequences:
/// `Delta(bases: FoR+BitPacked, deltas: FoR+BitPacked)`.
fn vortex_delta_bitpack(array: &PrimitiveArray, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let len = array.len();
    let (bases, deltas) = delta_compress(array, ctx)?;
    let bases = for_bitpack(bases, ctx)?;
    let deltas = for_bitpack(deltas, ctx)?;
    Ok(Delta::try_new(bases, deltas, 0, len)?.into_array())
}

// ---------------------------------------------------------------------------
// Datasets
// ---------------------------------------------------------------------------

/// Strictly increasing sequences from per-dataset gap distributions.
fn gen_dataset(name: &str) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(0x5EED);
    let mut geometric = |mean_gap: f64| -> Vec<u64> {
        // Inverse-CDF geometric gaps (>= 1), the classic random-postings model.
        let p = 1.0 / mean_gap;
        let mut acc = 0u64;
        (0..N)
            .map(|_| {
                let r: f64 = rng.random();
                let gap = if p >= 1.0 {
                    1
                } else {
                    1 + (r.ln() / (1.0 - p).ln()).floor() as u64
                };
                acc += gap;
                acc
            })
            .collect()
    };
    match name {
        // Dense postings: 80% of the universe is present.
        "dense_x1.25" => geometric(1.25),
        // Mid-density postings.
        "uniform_x32" => geometric(32.0),
        // Sparse postings.
        "sparse_x1024" => geometric(1024.0),
        // Bursty postings: dense runs of consecutive ids separated by large jumps.
        "clustered" => {
            let mut out = Vec::with_capacity(N + 512);
            let mut acc = 0u64;
            while out.len() < N {
                acc += rng.random_range(10_000u64..100_000);
                let run = rng.random_range(64usize..=512);
                for _ in 0..run {
                    acc += 1;
                    out.push(acc);
                }
            }
            out.truncate(N);
            out
        }
        // Near-regular event timestamps (microseconds with jitter) from a large epoch.
        "timestamps" => {
            let mut acc = 1_700_000_000_000_000u64;
            (0..N)
                .map(|_| {
                    acc += rng.random_range(968u64..=1032);
                    acc
                })
                .collect()
        }
        // Heavy-tailed (Pareto-ish) gaps: mostly small with rare huge jumps.
        "zipf_gaps" => {
            let mut acc = 0u64;
            (0..N)
                .map(|_| {
                    let r: f64 = rng.random();
                    let gap = (1.0 / (1.0 - r)).powf(1.0 / 1.2);
                    acc += (gap as u64).clamp(1, 100_000_000);
                    acc
                })
                .collect()
        }
        _ => unreachable!("unknown dataset {name}"),
    }
}

const DATASETS: &[&str] = &[
    "dense_x1.25",
    "uniform_x32",
    "sparse_x1024",
    "clustered",
    "timestamps",
    "zipf_gaps",
];

static DATA: LazyLock<BTreeMap<&'static str, Vec<u64>>> = LazyLock::new(|| {
    DATASETS
        .iter()
        .map(|&name| (name, gen_dataset(name)))
        .collect()
});

fn to_primitive(values: &[u64]) -> PrimitiveArray {
    PrimitiveArray::new(Buffer::copy_from(values), Validity::NonNullable)
}

// ---------------------------------------------------------------------------
// Size evaluation (printed once before the divan benchmarks run)
// ---------------------------------------------------------------------------

fn bpe(bits: u64) -> f64 {
    bits as f64 / N as f64
}

fn size_report() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let compressor = BtrBlocksCompressor::default();

    println!("Elias-Fano vs Vortex Delta+FoR+BitPacking, n = {N} u64 values");
    println!("(bits per element; PEF partitions: uniform 128 / uniform 1024 / DP-optimized)");
    println!(
        "{:<14} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>12}",
        "dataset", "delta+bp", "btrblocks", "EF", "PEF u128", "PEF u1024", "PEF opt", "best"
    );

    for (&name, values) in DATA.iter() {
        let base = values[0];
        let universe = values[values.len() - 1] - base + 1;

        // Vortex delta cascade.
        let array = to_primitive(values);
        let delta = vortex_delta_bitpack(&array, &mut ctx)?;
        let delta_bits = delta.nbytes() * 8;
        let roundtrip = delta.clone().execute::<PrimitiveArray>(&mut ctx)?;
        assert_eq!(
            roundtrip.as_slice::<u64>(),
            values.as_slice(),
            "{name}: delta roundtrip"
        );

        // Default compressor pick, for context.
        let btr = compressor.compress(&array.into_array(), &mut ctx)?;
        let btr_bits = btr.nbytes() * 8;

        // Elias-Fano variants (encoded relative to the first value, like FoR).
        let rel: Vec<u64> = values.iter().map(|&x| x - base).collect();
        let ef = EliasFano::encode(&rel, universe);
        let mut decoded = Vec::with_capacity(N);
        ef.decode_append(&mut decoded, base);
        assert_eq!(decoded, *values, "{name}: EF roundtrip");
        let ef_bits = ef.size_bits() + 128;

        let pef128 = PartitionedEliasFano::encode_uniform(values, 128);
        assert_eq!(pef128.decode(), *values, "{name}: PEF-128 roundtrip");
        let pef1024 = PartitionedEliasFano::encode_uniform(values, 1024);
        assert_eq!(pef1024.decode(), *values, "{name}: PEF-1024 roundtrip");
        let pef_opt = PartitionedEliasFano::encode_opt(values);
        assert_eq!(pef_opt.decode(), *values, "{name}: PEF-opt roundtrip");

        let rows = [
            ("delta+bp", delta_bits),
            ("btrblocks", btr_bits),
            ("EF", ef_bits),
            ("PEF u128", pef128.size_bits()),
            ("PEF u1024", pef1024.size_bits()),
            ("PEF opt", pef_opt.size_bits()),
        ];
        let best = rows.iter().min_by_key(|(_, bits)| *bits).unwrap();
        println!(
            "{:<14} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>12}",
            name,
            bpe(delta_bits),
            bpe(btr_bits),
            bpe(ef_bits),
            bpe(pef128.size_bits()),
            bpe(pef1024.size_bits()),
            bpe(pef_opt.size_bits()),
            best.0,
        );
    }

    println!();
    println!("btrblocks encoding trees:");
    for (&name, values) in DATA.iter() {
        let btr = compressor.compress(&to_primitive(values).into_array(), &mut ctx)?;
        println!("--- {name}\n{}", btr.tree_display());
    }
    Ok(())
}

fn main() {
    LazyLock::force(&SESSION);
    size_report().unwrap();
    divan::main();
}

// ---------------------------------------------------------------------------
// Throughput benchmarks
// ---------------------------------------------------------------------------

#[divan::bench(args = DATASETS)]
fn encode_vortex_delta_bitpack(bencher: Bencher, name: &str) {
    let array = to_primitive(&DATA[name]);
    let mut ctx = SESSION.create_execution_ctx();
    bencher
        .counter(ItemsCount::new(N))
        .bench_local(|| vortex_delta_bitpack(&array, &mut ctx).unwrap());
}

#[divan::bench(args = DATASETS)]
fn encode_elias_fano(bencher: Bencher, name: &str) {
    let values = &DATA[name];
    let base = values[0];
    let universe = values[values.len() - 1] - base + 1;
    let rel: Vec<u64> = values.iter().map(|&x| x - base).collect();
    bencher
        .counter(ItemsCount::new(N))
        .bench_local(|| EliasFano::encode(&rel, universe));
}

#[divan::bench(args = DATASETS)]
fn encode_pef_uniform_1024(bencher: Bencher, name: &str) {
    let values = &DATA[name];
    bencher
        .counter(ItemsCount::new(N))
        .bench_local(|| PartitionedEliasFano::encode_uniform(values, 1024));
}

#[divan::bench(args = DATASETS)]
fn decode_vortex_delta_bitpack(bencher: Bencher, name: &str) {
    let mut ctx = SESSION.create_execution_ctx();
    let array = vortex_delta_bitpack(&to_primitive(&DATA[name]), &mut ctx).unwrap();
    bencher
        .counter(ItemsCount::new(N))
        .bench_local(|| array.clone().execute::<PrimitiveArray>(&mut ctx).unwrap());
}

#[divan::bench(args = DATASETS)]
fn decode_elias_fano(bencher: Bencher, name: &str) {
    let values = &DATA[name];
    let base = values[0];
    let universe = values[values.len() - 1] - base + 1;
    let rel: Vec<u64> = values.iter().map(|&x| x - base).collect();
    let ef = EliasFano::encode(&rel, universe);
    bencher.counter(ItemsCount::new(N)).bench_local(|| {
        let mut out = Vec::with_capacity(N);
        ef.decode_append(&mut out, base);
        out
    });
}

#[divan::bench(args = DATASETS)]
fn decode_pef_uniform_1024(bencher: Bencher, name: &str) {
    let pef = PartitionedEliasFano::encode_uniform(&DATA[name], 1024);
    bencher
        .counter(ItemsCount::new(N))
        .bench_local(|| pef.decode());
}

#[divan::bench(args = DATASETS)]
fn decode_pef_opt(bencher: Bencher, name: &str) {
    let pef = PartitionedEliasFano::encode_opt(&DATA[name]);
    bencher
        .counter(ItemsCount::new(N))
        .bench_local(|| pef.decode());
}
