# 2. Prior art: how others pick encodings for speed

*Read this for evidence that the problem is tractable, and for the four strategies every
system uses (there are only four). Provenance: claims below were extracted from primary
sources (papers, official artifact repos, production source code) with verbatim quotes by a
fan-out research pass in July 2026; quotes were captured from the sources directly, but a
final independent verification pass was cut short — treat exact numbers as "as reported by
the cited source".*

**TL;DR:** Nobody ships a true per-candidate execution-time model *and* nobody ignores
execution speed. Every system lands on one of four strategies: **(A) restrict the pool** to
fast-decoding encodings and then optimize ratio (BtrBlocks, FastLanes, DuckDB's format, Data
Blocks, ClickHouse's RFC, CodecDB); **(B) size × hand-tuned speed penalty** per encoding
(DuckDB's selector, Nimble's default policy — and Vortex's own delta tax); **(C) calibrated
or learned time models** per objective (Damme et al./MorphStore, its EDBT'23 learned
successor); **(D) speed as a hard constraint or user-chosen objective** (Data Blocks'
byte-addressability, MorphStore's random-access bypass rule, Procella's user-weighted
size-vs-speed objective — the closest published precedent for what doc 4 proposes). Vortex
today is pure (A) with a pinch of undeclared (B).

- [Strategy A: curate the pool, then maximize ratio](#strategy-a-curate-the-pool-then-maximize-ratio)
- [Strategy B: size × penalty multipliers](#strategy-b-size--penalty-multipliers)
- [Strategy C: calibrated & learned time models](#strategy-c-calibrated--learned-time-models)
- [Strategy D: constraints and explicit objectives](#strategy-d-constraints-and-explicit-objectives)
- [How much does encoding choice matter? (numbers)](#how-much-does-encoding-choice-matter-numbers)
- [What transfers to Vortex](#what-transfers-to-vortex)

## Strategy A: curate the pool, then maximize ratio

*"Speed by construction": admit only fast-decoding schemes, then the ratio-argmax can't pick
anything slow.*

- **BtrBlocks (SIGMOD '23)** — the direct ancestor. Selection is a greedy argmax of sampled
  compression ratio (`chooseScheme` in `SchemePicker.hpp` keeps the max
  `expectedCompressionRatio`); **no speed term exists anywhere in the cost function**. Decode
  speed is enforced offline by pool construction (lightweight encodings only, no
  zstd/lz4/snappy) plus a hard `max_cascade_depth = 3`. Config exposes per-type scheme sets,
  forced overrides, sample size/count, and a `TRY_ALL` exhaustive mode — but the *objective*
  is not pluggable. Reported results: scans 2.2× faster and 1.8× cheaper end-to-end than
  Parquet(+codecs) on AWS; in-memory decompression 2.6–4.2× Parquet variants.
- **FastLanes (VLDB '23 + file format '25)** — the purist version: fix decode speed *in the
  layout* (unified transposed layout, virtual 1024-bit ISA) so DELTA/RLE/DICT/FOR all decode
  data-parallel (>40 values/cycle), then select expressions by ratio on a tiny sample (3
  vectors per rowgroup, "three-way" sampling, >99% ratio accuracy vs exhaustive). The pool
  itself was curated *offline with a leave-one-out ablation* measuring each scheme's marginal
  ratio **and** decode-speed contribution (e.g. dict: +42% ratio, +44% speed; patches: +2.5%
  ratio, −7% speed) — i.e., the speed/ratio tradeoff was paid once, by the format authors, so
  per-column selection doesn't have to think about it.
- **DuckDB's native format** — lightweight-only pool with the explicit rationale that
  general-purpose (de)compression bottlenecks RAM-resident data; goal stated as "compression
  on par with Parquet+Snappy, using only lightweight techniques that are very fast".
- **ClickHouse adaptive-codec RFC (#105404)** — per-block objective is smallest output, but a
  codec is only *admitted to the pool* if it beats LZ4 on both ratio and decode speed;
  cascades rejected as not worth the decode overhead.
- **CodecDB (SIGMOD '21)** — learned selection (NN over data-stats features, ~20k real
  columns, ~90% accuracy, up to +40% ratio vs rule-based selection) — but the label is
  **file size only**; scan-time feature extractors exist in the repo and are commented out.
  Their order-of-magnitude TPC-H wins come from encoding-aware *operators* (in-situ execution
  on encoded data), with lightweight-only encodings justified as "negligible decode cost".
  Lesson: even the learned-selection paper treated speed as a pool property, and located the
  speed win in the execution engine.

**Verdict on (A):** it works — and it is exactly what Vortex does (`ALL_SCHEMES` is
lightweight; `with_compact()` deliberately breaches the pool discipline for ratio). Its limit
is that one pool can't serve divergent workloads: hence Vortex's hand-rolled
`only_cuda_compatible()` second pool, and hence this research.

## Strategy B: size × penalty multipliers

*"Speed as a fudge factor": keep the size objective, inflate the size estimate of slow
encodings.*

- **DuckDB's selector** — two-phase analyze (full scan of each ~120k-row segment, no
  sampling) → minimum `FinalAnalyze` score across methods, where score = estimated size ×
  per-method penalty. The penalties are hardcoded and hand-calibrated: **Roaring ×2.0** (the
  commit trail shows the author measuring "1.5–2× slower than uncompressed" in a
  microbenchmark and rounding it into a constant), **FSST ×1.2**, **ZSTD ×1000 below a
  string-length threshold** (a soft ban that still allows forcing). This is production-proven
  and cheap — and it is precisely Vortex's `DELTA_PENALTY = 0.95` pattern, applied more
  systematically.
- **Nimble (Meta)** — the most on-point production precedent. Selection policy is a
  **pluggable interface** (`EncodingSelectionPolicy`) with three shipped implementations:
  the default cost-based `ManualEncodingSelectionPolicy` computing
  `cost = estimatedSize × readFactor` over *analytically estimated* sizes (no trial
  encoding), with read factors **Trivial = 0.7, FixedBitWidth = 0.9, Dictionary/RLE/etc. =
  1.0** — the source comments state these "represent mostly the CPU cost to decode values" —
  and factors are **runtime-configurable via a config string**; a
  `LearnedEncodingSelectionPolicy` (currently an embryonic 3-feature linear model); and a
  `ReplayedEncodingSelectionPolicy` that reproduces a captured encoding layout (testing,
  migration). Cascading is top-down greedy with irrevocable commitment and parent-type
  exclusion — structurally the same as Vortex. General-purpose compression is a separate,
  also-pluggable `CompressionPolicy` (accept if ratio ≥ 1/0.98).

**Verdict on (B):** multipliers are the minimum viable speed model: unitless, uncalibrated
(a Nimble dict candidate must be ~30% smaller than raw to win — why 30%? because 0.7), and
unable to express IO-vs-CPU or workload dependence. But two production systems ship them,
they're better than nothing, and Nimble's *pluggable-policy architecture around them* is
independently validated design precedent for doc 4.

## Strategy C: calibrated & learned time models

*"Actually model time": the academic line that validates doc 3's claim that decode cost is
predictable.*

- **Damme et al. (TODS '19) / LC-BaSe / MorphStore (VLDB '20)** — the most complete
  realization. A **grey-box** cost model: the white-box part analytically transforms a
  column's *bit-width histogram* through each logical technique (RLE/Delta/FoR/Dict "change
  functions"); the black-box part is **per-(algorithm, bit-width) runtime profiles measured
  by a ~1-hour calibration microbenchmark on the target machine**. Costs are per *objective*
  (compressed size, compression time, decompression time, aggregate-on-compressed time), with
  a `select()` that supports "fastest algorithm whose ratio is still acceptable". Cascades
  are costed **compositionally without compressing samples** (stage runtimes add; stage
  ratios multiply), with *separate calibration profiles for stand-alone vs in-cascade* use
  (cache-residency matters). Columns needing random access **bypass the cost model entirely**
  and get a rule-based fixed-width format. Outcome on SSB: cost-model-driven compression of
  base data + intermediates cut average query runtime ~54% (vs 19% for AVX-512 vectorization
  of uncompressed data alone) while halving memory footprint.
- **EDBT '23 learned successor** (same TU Dresden group) — replaces the analytical part with
  one gradient-boosted regressor per (algorithm × objective), trained on local-machine
  microbenchmarks, over 10 cheap data-statistics features. Reported per-objective selection
  accuracy 59–85% (mean 73%) — but crucially, **regret on the decompression-runtime objective
  is only 1.3–1.6%** on mis-selections (mis-picks are near-ties), versus up to ~21–25% regret
  on size objectives. That asymmetry is encouraging for Vortex: *speed-objective selection
  tolerates model error better than size-objective selection does.*
- Caveats for transfer: this line models flat/depth-2 integer codecs with SIMD kernels — far
  narrower than Vortex's scheme space (strings, floats, patches, structural schemes) — and
  its per-machine calibration assumes writer ≈ reader hardware, which object-storage formats
  can't assume.

## Strategy D: constraints and explicit objectives

- **Data Blocks / HyPer (SIGMOD '16)** — decode/point-access speed as a **hard constraint on
  the candidate set**: only byte-addressable encodings (dict/truncation/single-value; bit
  -packing explicitly rejected), then size-optimal per attribute per frozen chunk over the
  *full* data (no sampling). Deliberately concedes ratio (vs BitWeaving) for access speed.
  Also documents the *engine-side* cost of fine-grained adaptivity: the combinatorial variety
  of per-block representations forces an interpreted vectorized scan layer feeding JIT
  pipelines.
- **Procella / Artus (Google, VLDB '19)** — the closest published statement of doc 4's
  design: *"Each encoding has estimation methods for how small and **fast** it will be on the
  data supplied. Once the user has specified their objective function (relative value of size
  versus speed), Artus will use these methods to automatically choose the optimal
  encodings."* Plus: multi-pass stats-driven adaptive encoding, O(1) seeks for
  directly-indexable encodings vs O(B) skip-blocks for RLE (B = 32/128, an explicit
  size-vs-lookup knob), and pushdown by exposing dict indices/RLE runs to the engine (their
  vectorized engine gained ~2× on the old format but ~5× on Artus — encoding choice changed
  what the engine could exploit).
- **MorphStore's random-access rule** (also C) — worth repeating as a design pattern: when
  the access pattern is known to be random, *don't cost-model it* — apply a rule (fixed-width
  formats only). Discrete workload facts justify discrete handling.
- **Parquet/ORC (per Zeng et al., VLDB '23)** — the null precedent: encoding choice is
  hardcoded heuristics (Parquet: dict-first with 1 MB fallback; ORC: NDV-ratio ≤ 0.8 gate;
  RLE run threshold 8 hardcoded in every implementation), nothing pluggable anywhere in the
  ecosystem, and the paper's recommendations read like a to-do list for Vortex: favor decode
  speed for integer encodings, make block compression optional, keep raw as an option.

## How much does encoding choice matter? (numbers)

Evidence that the objective is worth optimizing — and that effects are step-function-sized,
supporting doc 3's "coarse models capture most of the value":

| Finding | Source |
|---|---|
| Block compression (zstd) adds up to **4.2× scan overhead**; wins only on slow storage tiers; on NVMe it's a net loss. Even dictionary encoding's compute cost can exceed its IO savings on NVMe — "trade-off, not Pareto improvement" anymore | Zeng et al., VLDB '23 format evaluation |
| Scan queries on Chimp-encoded floats are **59× slower** than on ALP; zstd 33×; scanning *uncompressed* is itself 2.2× slower than ALP (fused decode beats memcpy) | ALP artifact (SIGMOD '24), regenerated tables |
| Single-value random access: BtrBlocks is **800× slower** than FastLanes because a cascade forces whole-rowgroup decompression *per cascade layer* — direct measurement of "cascade depth × block size" as the random-access cost | FastLanes file-format paper |
| Forcing one codec (T64) on TPC-H decimal columns: +4.9% hot / +26.6% cold aggregate query speedup (up to +53% on single queries) | ClickHouse RFC #105404 measurements |
| Cost-model-driven compression of base + intermediate data: **−54% average query runtime** on SSB (vs −19% for SIMD alone) | MorphStore, VLDB '20 |
| ORC's finer-grained adaptive integer encoding (4 schemes per subsequence) → 4× more subsequences, **3× more branch mispredictions** at decode than Parquet's 2-scheme design — adaptivity itself has a decode price | Zeng et al. — and a caution for per-chunk flapping in Vortex |
| Vortex's own headline: TPC-H SF10 files 38% smaller and **10–25× faster to decompress** than Parquet+zstd with no general-purpose compression | Vortex README |
| FSST vs block compression for strings: FSST retains per-value random access; LZ4-block forfeits it | BtrBlocks/FSST line of work |
| Correlation-aware (multi-column) encodings: 30–85% size savings at a bounded, selectivity-dependent 1.4–2× decode penalty; RLE/Delta need checkpoints for random access, FOR/Dict+bitpack don't | Corra (2024) |

## What transfers to Vortex

1. **The architecture to copy is Nimble's** (pluggable selection policy, cost = mechanical
   estimate × policy data, replay policy for reproducibility), **the cost function to aspire
   to is Artus's** (per-encoding size *and* speed estimators under a user-weighted
   objective), **the calibration methodology to borrow is LC-BaSe's** (per-bit-width profiles
   from a one-shot microbenchmark; compositional cascade costing; separate in-cascade
   profiles), and **the discipline to keep is BtrBlocks/FastLanes'** (a curated lightweight
   pool remains the first line of defense — a cost model refines within it).
2. **Nobody solved workload-unknown-at-write-time.** Systems either assume (A), let the user
   say (Artus, Nimble's configurable factors), or apply rules for known access patterns
   (MorphStore). This confirms doc 3's conclusion: workload must be an input; pluggability is
   the mechanism.
3. **Every production "speed model" is coarse** (0.7/0.9/1.0; ×2.0; pool bans) and still
   pays off. The calibrated-time literature (C) shows finer models are *possible* and that
   speed objectives are forgiving of estimation error (1–2% regret) — but no file format has
   needed that fidelity yet. Phase the investment accordingly (doc 5).
4. **Two under-appreciated costs Vortex should watch** because the literature measured them:
   cascade depth × block size dominating random access (FastLanes' 800×), and
   encoding-flapping across chunks degrading branch prediction (Zeng's ORC finding).
5. **Replay/pinning is a feature, not a hack** — Nimble ships a policy whose only job is
   reproducing a previous layout; Vortex's determinism requirement points the same direction
   (pin the cost table with the config, doc 4).

### Primary sources

BtrBlocks: paper (SIGMOD '23, doi 10.1145/3589263) + reference impl `maxi-k/btrblocks`
(`SchemePicker.hpp`, `btrblocks.hpp`) · Nimble: `facebookincubator/nimble`
(`dwio/nimble/encodings/selection/EncodingSelectionPolicy.h`,
`dwio/nimble/compression/CompressionPolicy.h`) · DuckDB: "Lightweight compression in DuckDB"
blog + `src/storage/compression/{roaring/common.cpp,fsst.cpp,zstd.cpp}`,
`column_data_checkpointer.cpp` · Damme et al.: TODS 44(3) 2019 (doi 10.1145/3323991) +
`MorphStore/LC-BaSe` (`lcbase_py/costmodel.py`, `whitebox.py`) + MorphStore VLDB '20
artifacts · EDBT '23 learned selection: `lucaswo/learned-selection-strategy` (doi
10.48786/edbt.2023.47) · CodecDB: SIGMOD '21 + artifact (`Features.scala`,
`NNPredictor.scala`) · Data Blocks: SIGMOD '16 · Procella/Artus: VLDB '19 · White-box
compression: CIDR '20 (bi-objective cost model *defined*, size-only *implemented*) · ALP:
SIGMOD '24 + `cwida/ALP` (`encoder.hpp`) · FastLanes: VLDB '23 + `cwida/FastLanes` ·
Zeng et al.: "An Empirical Evaluation of Columnar Storage Formats", VLDB '23
(arXiv:2304.05028) · ClickHouse RFC #105404 · Corra (2024).
