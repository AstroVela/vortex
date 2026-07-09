# 1. How the compressor decides today

*Read this if you want to know exactly where the current "cost model" lives in the code, and
what implicit judgments are already baked in.*

**TL;DR:** The cost model is a single scalar — estimated compression ratio (`f64`, higher wins,
must beat 1.0) — produced per scheme, compared with a strict total order, and accepted only if
the output is byte-wise smaller. Execution-speed concerns *do* exist in the codebase today, but
they are expressed as fudge factors, registration-order comments, skip-gates, and scheme-set
presets, because ratio is the only currency the selection loop understands.

- [The decision pipeline](#the-decision-pipeline)
- [The currency: `EstimateScore`](#the-currency-estimatescore)
- [Catalogue: every judgment the compressor makes](#catalogue-every-judgment-the-compressor-makes)
- [Speed concerns already smuggled into the ratio scale](#speed-concerns-already-smuggled-into-the-ratio-scale)
- [What the compressor knows (and doesn't)](#what-the-compressor-knows-and-doesnt)
- [Structural properties that constrain any cost-model change](#structural-properties-that-constrain-any-cost-model-change)

## The decision pipeline

`vortex-compressor` is the framework (post-#7216); `vortex-btrblocks` registers the schemes.
(There is a rendered decision-tree diagram in `vortex-btrblocks/decision-tree.svg`, linked
from that crate's README.)

1. **Entry** — `CascadingCompressor::compress` (`vortex-compressor/src/compressor.rs:120`)
   canonicalizes + compacts, then recurses structurally (struct fields, list offsets/elements,
   extension storage, variant slots). Leaves go to `choose_and_compress`
   (`vortex-compressor/src/compressor.rs:297`).
2. **Eligibility** — schemes are filtered by `Scheme::matches` plus exclusion rules
   (self-exclusion, push/pull rules; `is_excluded`, `compressor.rs:488`). Empty, all-null, and
   constant arrays are handled by the compressor itself before any scheme runs.
3. **Selection** — `choose_best_scheme` (`compressor.rs:412`) runs two passes:
   - *Pass 1:* every scheme's `expected_compression_ratio` returns a `CompressionEstimate`:
     an immediate verdict (`Skip` / `AlwaysUse` / `Ratio(f64)`) or a deferred request.
     `AlwaysUse` short-circuits everything.
   - *Pass 2:* deferred work runs. `DeferredEstimate::Sample` compresses a ~1% stratified
     sample (64-value runs × ≥16 runs, deterministic seed; `sample.rs:15-26`) *through the
     scheme's full `compress`, cascades included*, and scores measured
     `before_bytes / after_bytes`. `DeferredEstimate::Callback` gets the best-so-far ratio as
     an early-exit threshold.
   - Winner = argmax ratio; ties break by registration order in `ALL_SCHEMES`.
4. **Acceptance** — the winner compresses the full array; the result is kept only if
   `after_nbytes < before_nbytes` (`compressor.rs:381`).
5. **Cascade** — the winner's `compress` calls `compress_child`, which re-enters the whole
   pipeline with a decremented budget. Budget is a hardcoded `MAX_CASCADE = 3`
   (`ctx.rs:16`, with a `TODO(connor): Why is this 3???`).

## Where this sits in a file write

Context that matters for any cost-model change (all defaults, from
`vortex-file/src/strategy.rs:236`):

- Columns are repartitioned to 8192-row blocks, zoned stats are computed per block, and
  **dictionary encoding is decided upstream of the compressor** by the layout-level
  `DictStrategy`; the per-chunk data compressor actually *excludes* `IntDictScheme`
  (`strategy.rs:254-260`). A second full compressor instance separately compresses stats
  tables and dictionary values (`strategy.rs:283-305`).
- Chunks are coalesced to ~1 MiB (8192-row multiples) before compression
  (`strategy.rs:266-280`), so scheme selection runs per column × ~1 MiB chunk.
- **Every chunk decides independently** — fresh `CompressorContext`, stats recomputed, no
  memory of the previous chunk's choice — and chunks/columns compress in parallel on the CPU
  pool (`vortex-layout/src/layouts/compressed.rs:90-118`).
- Pluggability today ends at the compressor object: `WriteStrategyBuilder::with_compressor`
  accepts any `CompressorPlugin` (even a closure), and `with_btrblocks_builder` swaps scheme
  sets. Inside the compressor, nothing about the *objective* is configurable. Python exposes
  exactly two presets (`default`, `compact`); DataFusion/DuckDB expose none.

## The currency: `EstimateScore`

Everything a scheme can say about itself must be encoded into one of
(`vortex-compressor/src/estimate.rs`):

| Signal | Meaning | Notes |
|---|---|---|
| `EstimateVerdict::Skip` | "not viable here" | also used for *policy* ("not worth it"), see catalogue |
| `EstimateVerdict::AlwaysUse` | "definitively best, stop evaluating" | decimal/temporal decomposition |
| `EstimateVerdict::Ratio(f64)` | estimated compression ratio | must be `> 1.0` to be valid (`estimate.rs:142`) |
| `EstimateScore::ZeroBytes` | sample compressed to 0 bytes | ineligible (issue #7268 artifact) |

Two hard-wired assumptions follow from this type:

1. **The objective is size.** `beats()` (`estimate.rs:152`) compares ratios; the acceptance
   check compares bytes. There is nowhere to say "smaller but slower to decode".
2. **The baseline is canonical, at zero cost.** `ratio > 1.0` treats the uncompressed array as
   cost 0 — which is exactly right for size, and exactly wrong for anything else (e.g. an
   encoded form that is both smaller *and faster to scan* than canonical, which real encodings
   like dictionary can be for some operations).

## Catalogue: every judgment the compressor makes

How each built-in scheme answers `expected_compression_ratio`:

| Scheme | Gate(s) (→ `Skip`) | Estimate | Where |
|---|---|---|---|
| FoR | min==0; FoR width ≥ BitPacking width; leaf ctx | analytic: `full_bits / for_bits` | `schemes/integer/for_.rs:66` |
| BitPacking | negative min | **sample** | `schemes/integer/bitpacking.rs:41` |
| ZigZag | no negatives; leaf ctx | **sample** | `schemes/integer/zigzag.rs:94` |
| IntDict | distinct > 50% of values | analytic: values + min(BP, RLE+BP) codes model | `builtins/dict/integer.rs:60` |
| RunEnd | avg run length < 4 | **sample** | `schemes/integer/runend.rs:100` |
| RLE (int/float) | avg run length < 4; leaf ctx | **sample** | `schemes/integer/rle.rs:172` |
| Sequence | is-sample; nulls; distinct≠len | callback: trial-encode, max ratio `len/2` | `schemes/integer/sequence.rs:67` |
| Delta (unstable) | leaf ctx; len < 1024 | callback: measure residual span, ratio × **0.95 penalty**, require **≥ 1.25** | `schemes/integer/delta.rs:117` |
| Sparse | null% < 90% and top-value% < 90% | analytic: `len / value_count` | `schemes/integer/sparse.rs:80` |
| Pco | u8/i8 | **sample** | `schemes/integer/pco.rs:33` |
| ALP / ALP-RD | f16; leaf ctx | **sample** | `schemes/float/alp.rs:48` |
| FSST | — | **sample** (trains a symbol table per sample — the expensive case flagged in #7697) | `schemes/string/fsst.rs:51` |
| StringDict / BinaryDict / FloatDict | distinct > 50% of values | **sample** | `vortex-compressor/src/builtins/dict/*.rs` |
| Zstd / OnPair (opt-in) | — | **sample** | `schemes/{string,binary}/zstd.rs` |
| Decimal / Temporal | — | `AlwaysUse` | `schemes/decimal.rs`, `schemes/temporal.rs` |

Note the three *kinds* of judgment mixed into one method:

- **Feasibility** ("BitPacking needs non-negative values") — a property of the encoding.
- **Size estimation** ("FoR saves `full_bits - for_bits` per value") — the mechanical signal.
- **Policy** ("don't dict if >50% distinct", "require delta to win by 25%") — *cost-model
  decisions*, currently hardcoded per scheme, invisible to the selection loop, and impossible
  to vary per workload.

## Speed concerns already smuggled into the ratio scale

These are the strongest evidence that a second objective already exists and has no legitimate
place to live:

1. **The "delta tax"** — `DELTA_PENALTY = 0.95` (`schemes/integer/delta.rs:61-67`):
   > "Unlike FoR/BitPacking, Delta breaks random access and adds a prefix-sum decode pass …
   > We therefore require Delta to be meaningfully (~5%) smaller than the best alternative
   > before it wins … This factor encodes that 'delta tax'."

   An execution-cost judgment expressed as a unitless, uncalibrated multiplier on a size
   estimate. Plus `min_ratio = 1.25` on top (`builder.rs:40`).
2. **Registration order as a speed prior** — `ALL_SCHEMES` (`vortex-btrblocks/src/builder.rs:24`):
   > "Prefer all other schemes above delta, for now (since its slower to decompress)."

   Order only matters on *exact ties*, so this is a nearly-inert lever that reads like a
   policy knob.
3. **`only_cuda_compatible()`** (`builder.rs:163`) — an entire workload profile ("will be
   decoded by GPU kernels") expressed as a hand-maintained scheme exclusion list, with the
   explicit caveat "may choose a larger encoded representation than the default". This is a
   cost model with exactly two values: possible and impossible.
4. **`with_compact()`** (`builder.rs:142`) — the opposite direction: opt into Zstd/Pco for
   ratio at the cost of decode speed. Again a scheme-set toggle, not a tunable objective.
5. **Run-length gates (`RUN_LENGTH_THRESHOLD = 4`, `RUN_END_THRESHOLD = 4`)** and the dict 50%
   distinct gate — part compression-quality heuristic, part "don't waste estimation time",
   part implicit speed prior; none of it labeled as such.

## What the compressor knows (and doesn't)

Stats available to estimators (`vortex-compressor/src/stats/`), computed lazily per leaf, with
expensive ones (distinct values + frequencies) gated by merged `GenerateStatsOptions`:

- **Integer** (`stats/integer.rs:243`): min/max (→ bit widths), null/value counts, average run
  length, optional distinct values with frequencies, most-frequent value.
- **Float** (`stats/float.rs:95`): min/max, counts, run length, optional distinct set.
- **String** (`stats/varbinview.rs:17`): value/null counts, estimated distinct count.
- **Bool** (`stats/bool.rs:14`): counts, true count, constancy.

What it has **no representation for**:

| Missing input | Why a speed-aware model needs it |
|---|---|
| Per-encoding decode cost (ns/value, by dtype/params/arch) | the core of "how fast is this choice" |
| Cascade decode cost (cost of the *tree*, not the node) | cascades pay decode per level; ratio composes, speed accumulates |
| Compute-kernel coverage (which ops run natively on which encoding) | encoded compute can beat canonical; fallback = canonicalize first |
| Random-access class (O(1) vs search vs full-block decode) | point-lookup workloads vs scan workloads pick differently |
| Target hardware (SIMD, GPU, cache) | decode throughput varies ~an order of magnitude across targets |
| IO/storage bandwidth | determines how much a byte saved is worth in seconds |
| Workload (scan vs filter vs point access mix, per column) | the weights of the objective function |

The selection loop also never sees the *shape* of a candidate — only its byte count. A sample
compression that produced `Dict(FoR(BitPacked(codes)), values)` and one that produced
`Zstd(bytes)` are compared as two integers.

## Structural properties that constrain any cost-model change

Things about the current architecture that a new cost model must respect or explicitly break:

1. **Scalar total order.** Selection, tie-breaking, and the deferred-callback early-exit
   threshold (`EstimateFn` contract, `estimate.rs:24-45`) all assume one comparable number.
   A multi-objective cost must collapse to a comparable scalar *at selection time* (which is
   fine — that's what a cost function is for) or the threshold plumbing changes too.
2. **Judgment is distributed.** Every scheme self-scores, and cross-scheme knowledge is
   embedded inside schemes (FoR consults BitPacking's applicability, `for_.rs:91-104`; IntDict
   *assumes* its codes will be BitPacked/RLE'd, `builtins/dict/integer.rs:86-96`). Changing the
   objective today means editing every scheme, including third-party ones.
3. **Greedy, top-down, per-node.** The winner at each level is chosen before children are
   compressed; sampling bakes the (greedy) child choices into the measured ratio. There is no
   search over trees — #7216 notes the exclusion system bounds the space, and #7697 lists
   adaptive cascading as a stretch goal.
4. **Acceptance is size-only.** Even with a perfect speed model, `after < before`
   (`compressor.rs:381`) would veto any "larger but faster" choice. (Almost always the right
   veto for a *compression* framework — worth stating as an axiom rather than an accident.)
5. **Determinism.** Fixed sample seed, deterministic tie-break order, reproducible output —
   any measured-time input to selection would break this (see doc 3).
6. **Per-chunk independence.** Each leaf (per column, per chunk) decides alone; nothing makes
   choices consistent across chunks of the same column (see doc 3 for why this matters more
   for speed than for size).
