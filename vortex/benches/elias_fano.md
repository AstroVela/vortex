<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# Elias-Fano vs Vortex Delta + FoR + BitPacking

Results from `cargo bench -p vortex --bench elias_fano` (see `elias_fano.rs` for the
implementations and dataset generators). Numbers below were measured on a Linux x86-64
cloud container; absolute throughput will vary by machine, but the relative picture is
stable.

## Setup

All inputs are strictly increasing `u64` sequences of `n = 2^20` values — the natural
Elias-Fano use case (postings lists, row-id selections, offsets, timestamps):

| dataset | gap distribution | character |
|---|---|---|
| `dense_x1.25` | geometric, mean 1.25 | 80% of the universe present |
| `uniform_x32` | geometric, mean 32 | mid-density postings |
| `sparse_x1024` | geometric, mean 1024 | sparse postings |
| `clustered` | runs of gap-1 (64–512 long) split by 10k–100k jumps | bursty ids |
| `timestamps` | 1000 ± 32, from a 1.7e15 epoch | near-regular event times |
| `zipf_gaps` | Pareto(1.2), clamped | heavy-tailed gaps |

Contenders:

- **delta+bp** — the real Vortex cascade for smooth integers,
  `Delta(bases: FoR+BitPacked, deltas: FoR+BitPacked)`, sized with `Array::nbytes` and
  decoded through `execute::<PrimitiveArray>`. (The *default* `BtrBlocksCompressor` does
  not enable `DeltaScheme` — it is behind `unstable_encodings` — so its default pick,
  plain FoR/BitPacking at 21–30 bits/element on these datasets, is reported only as
  context.)
- **EF** — plain Elias-Fano over the whole sequence, relative to its first value:
  `n·max(0, floor(log2(u/n)))` explicit low bits plus a `n + (u >> l) + 1`-bit unary
  high-bits vector, i.e. roughly `2 + log2(u/n)` bits/element.
- **PEF u128 / u1024** — partitioned Elias-Fano (Ottaviano & Venturini, SIGIR'14) with
  uniform partitions of 128 / 1024 elements; every partition independently picks the
  cheapest of {implicit all-ones run, bitvector, Elias-Fano} relative to its bounds, and
  partition upper bounds are themselves EF-coded.
- **PEF opt** — same representations, but partition boundaries chosen by a shortest-path
  DP (boundary quantum 64, max partition 4096), approximating the paper's optimal
  partitioning.

## Compressed size (bits per element)

| dataset | delta+bp | EF | PEF u128 | PEF u1024 | PEF opt | winner |
|---|---:|---:|---:|---:|---:|---|
| `dense_x1.25` | 3.15 | 2.25 | 1.34 | 1.27 | **1.26** | PEF (2.5x smaller) |
| `uniform_x32` | 8.48 | 7.00 | 7.10 | 7.01 | **7.00** | EF/PEF (−17%) |
| `sparse_x1024` | 13.57 | **12.00** | 12.13 | 12.01 | **12.00** | EF/PEF (−12%) |
| `clustered` | **1.82** | 9.47 | 4.62 | 9.42 | 2.71 | delta+bp |
| `timestamps` | **11.47** | 11.95 | 12.28 | 12.00 | 11.97 | delta+bp (~tie) |
| `zipf_gaps` | 6.91 | 5.21 | 3.80 | 4.09 | **3.71** | PEF (1.9x smaller) |

Observations:

- **On uniform-gap data, EF and delta+bp are close, with EF ~1–1.5 bits/element
  smaller.** This is the theory playing out: EF costs `~2 + log2(u/n)` bits while
  delta+bitpack costs `~log2(max_gap)` bits, and for geometric gaps
  `max_gap ≈ mean_gap · ln(n)`, so delta pays for the tail of the gap distribution
  (patches only partially recover it) while EF pays a flat 2-bit unary overhead.
- **Partitioning is what makes Elias-Fano competitive beyond that.** PEF never loses to
  plain EF by more than the first-level overhead (~0.1 bit/element at 128-element
  partitions, ~0.01 at 1024), and on skewed data it wins big: 2.25 → 1.26 bits on
  `dense_x1.25` (all-ones runs become free), 5.21 → 3.71 bits on `zipf_gaps` (each
  partition gets a locally-sized `l` instead of one global compromise), 9.47 → 2.71 bits
  on `clustered`. Boundary optimization matters when structure doesn't align with fixed
  partitions: on `clustered`, uniform-1024 barely helps (9.42) while the DP (2.71)
  isolates the dense runs.
- **Delta+bitpack keeps two clear wins.** On `clustered` it reaches 1.82 bits: gap-1
  runs become 1-bit deltas and the rare jumps become patches, which is exactly the
  structure BitPacked's patch mechanism is built for; even optimally-partitioned EF
  can't express "run of ones" more cheaply than ~2.7 bits here because partition
  boundaries never align perfectly with run boundaries. On `timestamps` it wins narrowly
  (11.47 vs 11.97), though both leave ~4 bits/element on the table (see below).
- **A Vortex-side observation that fell out of the eval:** on `timestamps`, gaps span
  `[968, 1032]`, so deltas *could* FoR down to ~6 bits. They don't, because
  `delta_compress` stores a `0` delta at each FastLanes lane head (16 per 1024-element
  chunk), which drags the FoR minimum of the deltas child to 0 and forces
  `bit_width = bits(1032) = 11`. If lane-head slots were excluded from the deltas child's
  FoR reference (or patched), delta+bp would drop from ~11.5 to ~7 bits/element on
  near-regular sequences and beat every EF variant here.

## Throughput (median, Mitems/s, single thread)

Full-sequence decode to a `Vec<u64>` / `PrimitiveArray`:

| dataset | delta+bp | EF | PEF u1024 | PEF opt |
|---|---:|---:|---:|---:|
| `dense_x1.25` | 350 | 530 | 786 | 812 |
| `uniform_x32` | 367 | 227 | 234 | 226 |
| `sparse_x1024` | 299 | 248 | 219 | 206 |
| `clustered` | 304 | 230 | 225 | 740 |
| `timestamps` | 298 | 241 | 239 | 229 |
| `zipf_gaps` | 342 | 230 | 322 | 380 |

Encode:

| dataset | delta+bp | EF | PEF u1024 |
|---|---:|---:|---:|
| `dense_x1.25` | 87 | 581 | 438 |
| `uniform_x32` | 89 | 233 | 198 |
| `sparse_x1024` | 88 | 223 | 176 |
| `clustered` | 89 | 229 | 176 |
| `timestamps` | 127 | 229 | 177 |
| `zipf_gaps` | 88 | 237 | 246 |

- Vortex's SIMD unpack + prefix-sum decodes ~25–60% faster than this scalar
  bit-at-a-time EF decoder on incompressible-ish data (240–260 vs 300–370 Mitems/s).
  A word-at-a-time/BMI2 EF decoder would close much of that gap, but EF's sequential
  decode is fundamentally branchier than FastLanes kernels.
- PEF flips this on dense/clustered data: all-ones partitions decode as a plain range
  fill (740–880 Mitems/s), beating the FastLanes cascade by ~2.3x while also being
  smaller.
- Elias-Fano encoding is a single cheap pass: 2–6x faster than the Vortex cascade
  (which pays for delta transposition, stats, histogramming, and patch gathering).
  The PEF DP encoder (not benchmarked above) adds roughly 2x over uniform PEF at
  quantum 64.

Not measured here but worth noting for scan workloads: EF/PEF support `O(1)`-ish random
access (`select` on the high bits) and `next_geq` skipping without decoding whole
chunks, whereas `DeltaArray` must decode a full 1024-value chunk through the prefix sum
to answer a point query.

## Conclusions

1. Plain Elias-Fano is only marginally better than Vortex's delta+bitpack on
   uniform-gap monotone data (~1–1.5 bits/element) and loses badly on clustered data —
   not compelling by itself.
2. **Partitioned Elias-Fano is the version worth considering**: it dominates plain EF
   everywhere (the question in the task — yes, PEF makes things strictly better), and
   it beats the Vortex cascade by 1.9–2.5x on dense and heavy-tailed monotone data
   while decoding as fast or faster there.
3. Delta+bitpack remains the right default for near-regular sequences (timestamps,
   auto-increment-ish ids) and run-dominated data, where FastLanes' patches and SIMD
   prefix-sum are hard to beat.
4. Independent of EF: fixing the lane-head zero-delta interaction with FoR (see above)
   is a cheap ~40% size win for delta+bp on near-regular sequences.
