# Implementation plan: pluggable compressor cost model

*The PR ladder that carries the research in this directory into the codebase, ending at a
pluggable `CostModel` with an execution-time objective. Written per
[`NEXT-STEPS.md`](./NEXT-STEPS.md); design vocabulary from
[doc 4](./04-design-pluggable-cost-model.md), phasing and open questions from
[doc 5](./05-roadmap-and-open-questions.md).*

- [Code verification](#code-verification-what-was-re-anchored)
- [Decisions on the open questions](#decisions-on-the-open-questions)
- [The PR ladder](#the-pr-ladder)
  - [Track A — refactor](#track-a--refactor-toward-the-pluggable-model)
  - [Track B — measurement](#track-b--measurement)
  - [Stop/go checkpoint](#stopgo-checkpoint-after-m1m3)
  - [Track C — the payoff](#track-c--the-payoff-gated-on-the-checkpoint)
  - [Adjacent small wins](#adjacent-small-wins)
- [Dependency sketch](#dependency-sketch)
- [Out of scope](#out-of-scope-for-this-ladder)

## Code verification (what was re-anchored)

The research docs cite `develop` at `11f15a9`, which is still the current `develop` tip, so
line references remain live. Every symbol this plan builds on was re-verified:

| Anchor | Verified location | Status |
|---|---|---|
| `choose_best_scheme`, two-pass selection | `vortex-compressor/src/compressor.rs:412-483` | as documented |
| `EstimateScore::is_valid` (`ratio > 1.0`, finite, non-subnormal), `beats` (strict `>`) | `vortex-compressor/src/estimate.rs:142-160` | as documented |
| `EstimateFn` callback contract (`best_so_far` early-exit hint) | `vortex-compressor/src/estimate.rs:24-45` | as documented |
| Sampling keeps only `nbytes` of the compressed sample | `vortex-compressor/src/estimate.rs:218-235` | as documented |
| Byte-acceptance gate | `vortex-compressor/src/compressor.rs:381` | **drift**: now `after_nbytes < before_nbytes \|\| compressed.is::<AnyScalarFn>()` — an L2-denormalization carve-out. The acceptance axiom below accounts for it. |
| `DELTA_PENALTY = 0.95`, `min_ratio` | `vortex-btrblocks/src/schemes/integer/delta.rs:67`; `min_ratio` is instance state, `DeltaScheme::new(1.25)` registered in `ALL_SCHEMES` (`builder.rs:40`), feature-gated behind `unstable_encodings` | as documented, R4 must handle the public constructor |
| `ALL_SCHEMES` order comments, `with_compact`, `only_cuda_compatible` | `vortex-btrblocks/src/builder.rs:24-196` | as documented |
| `MAX_CASCADE = 3` | `vortex-compressor/src/ctx.rs:16` | as documented |
| `CompressorPlugin`, `WriteStrategyBuilder`, `with_field_writer`, `IntDictScheme` exclusion, dual data/stats compressors | `vortex-layout/src/layouts/compressed.rs:30`, `vortex-file/src/strategy.rs:153-305` | as documented; note only `IntDictScheme` is excluded from the data compressor — Float/String/BinaryDict decisions still happen inside the compressor |
| Determinism requirement in `Scheme` docs | `vortex-compressor/src/scheme.rs:142-144` | as documented |
| Session-scoped kernel registry keyed `(outer_id, child_id)` | `vortex-array/src/optimizer/kernels.rs` | as documented |
| Deterministic sampler (fixed seed 1234567890, 64×16 runs) | `vortex-compressor/src/sample.rs` | as documented |
| Compression micro-bench gated out of CI tracking | `vortex-btrblocks/benches/compress.rs:6` (`#[cfg(not(codspeed))]`) | as documented |
| Sequence trial-encodes then discards | `vortex-btrblocks/src/schemes/integer/sequence.rs:94-112` (existing `TODO(connor)`) | as documented |
| Python surface | `VortexWriteOptions.default()/.compact()` only (`vortex-python/python/vortex/_lib/io.pyi:25-29`) | as documented |
| `insta` snapshot crate available in workspace | root `Cargo.toml:176` | usable for R0 |

## Decisions on the open questions

Doc 5's seven open questions, each resolved here. These are the positions the ladder is built
on; changing one later means revisiting the rungs that cite it.

### 1. Cost units and the canonical baseline (doc 5 Q1)

**Decision:** `Cost` is a newtype over `f64` that is an *opaque ordered scalar* — lower is
better, must be finite; `None` from `CostModel::cost` rejects the candidate. Units are defined
per model, not by the framework: `TimeCost` documents its unit as **estimated nanoseconds per
value**; `SizeCost` prices directly off the compression-ratio signal it receives today.

**Rationale:** costs are only ever compared *within* one model per compressor instance, so
cross-model comparability buys nothing. Forcing `SizeCost` into fake time units (a fictitious
bandwidth) would obscure the one property R2 must prove — bit-exact reproduction of today's
ordering. Concretely: today's currency is a ratio `f64` on both paths (analytic verdicts
return a ratio; sampling computes `before as f64 / after as f64` in
`EstimateScore::from_sample_sizes`). If `SizeCost` re-derived bytes as `input/ratio`, IEEE
division could collapse strict ratio inequalities into cost ties and flip winners via the
registration-order tie-break. So `SizeCost::cost` is defined as `Cost(-ratio)` over the exact
ratio value computed today, with `canonical_cost = Cost(-1.0)` reproducing the `ratio > 1.0`
validity gate — identical ordering by construction, enforced by R0.

**Canonical baseline for the selective-scan term** (lands in C3, axiom written down there):
every objective term prices *end-to-end operation time*, canonical included. For selective
scan: `io(bytes) + decode_to_operable_form + op_cost(form the op actually runs against)`,
where canonical has `decode = 0` and pays the canonical kernel's op cost. An encoding with a
specialized kernel (registry hit) pays its encoded-op cost instead and can therefore price
*below* canonical — which is the point of the term. An encoding without a kernel pays full
decode plus the canonical op cost. This makes canonical "just another candidate" under every
term instead of a special zero, and it is the same shape as the full-scan term (where the op
is "materialize"), so the two compose without a special case.

### 2. The byte-acceptance invariant (doc 5 Q2)

**Decision:** keep `after_nbytes < before_nbytes` as a hard invariant for **all** models —
compression never grows bytes — and state it as an axiom in the `CostModel` docs. A model
that prefers canonical for speed expresses that by pricing every candidate at or above
`canonical_cost`, so selection returns `None` and the array stays canonical; the gate never
forces a *bad encoding*, only "no encoding". The existing `AnyScalarFn` carve-out at
`compressor.rs:381` is orthogonal (semantic denormalization, like `AlwaysUse`) and stays
untouched, but gets documented as part of the axiom rather than remaining an unexplained hack.

**Rationale:** the gate is the file writer's size-safety guarantee and the cheapest possible
sanity check against estimator error. "Faster but larger than canonical" remains expressible
(keep canonical); "faster but larger *encoded* form beats smaller encoded form" also remains
expressible (cost ordering decides among candidates; the gate only compares winner vs input).
The only thing sacrificed is a hypothetical model that wants an output *larger than
canonical*, which contradicts being a compressor.

### 3. Who owns capability annotations (doc 5 Q3)

**Decision:** a side table in `vortex-compressor`'s new `cost` module, not encoding vtable
metadata. `CapabilityMatrix` is built explicitly from a `VortexSession`'s kernel registry
(kernel coverage is derived — a lookup keyed by `(operation, encoding)`), merged with a small
static table for random-access class and GPU-decodability seeded from the appendix. Unknown
(third-party) encodings get a conservative default (no kernels, block-decode random access,
not GPU-decodable) plus a one-time `tracing` warning. The matrix is snapshotted at compressor
build time, so "same config ⇒ same file" holds.

**Rationale:** vtable metadata would touch every encoding crate and change a public trait
third parties implement — a wide, breaking diff for one consumer's benefit. The side table
keeps the annotation next to its only consumer, needs conservative defaults for third-party
encodings *anyway* (so vtable metadata wouldn't remove that code path), and can migrate into
the vtable later if a second consumer appears. Migration is additive at that point.

### 4. Where dictionary policy lives (doc 5 Q4)

**Decision:** the layout-level `DictStrategy` keeps ownership of *integer* dictionary
encoding for this entire ladder. The cost model's dict opinions still reach real decisions —
`FloatDictScheme`/`StringDictScheme`/`BinaryDictScheme` remain inside the data compressor
(only `IntDictScheme` is excluded, `strategy.rs:254-260`), and the stats/values compressor
uses the model too. Making `DictStrategy` consult the cost model is flagged as the first
follow-up *after* C5, not a rung here.

**Rationale:** moving the dict decision is a layering change (layout ↔ compressor) that
multiplies the review surface of every rung it touches, and nothing in this ladder is blocked
by deferring it. Doing it after C3 exists means the integration can consult a model that
actually knows what a dictionary is worth per workload, instead of moving the decision first
and inventing the policy second.

### 5. Default bandwidth / default model (doc 5 Q5)

**Decision:** no default flip anywhere in this plan. `SizeCost` (i.e. `max_ratio`, today's
behavior) remains the default for `CascadingCompressor`, `BtrBlocksCompressorBuilder`,
`WriteStrategyBuilder`, Python, and every embedded use, indefinitely. `TimeCost` presets
require an explicit bandwidth (named constants: `Bandwidth::S3`, `::NVME`, `::MEMORY`), so
there is no "default bandwidth" to choose.

**Rationale:** `max_ratio` is the only objective that is safe for unknown readers, and the
default's stability is what makes every refactor rung provably safe (R0). Whether most Vortex
reads are NVMe-local is an empirical product question; M1's regret data is the input to that
future decision, and flipping a default is a one-line PR once someone makes it.

### 6. Python / DuckDB / DataFusion surface (doc 5 Q6)

**Decision:** presets only, everywhere outside Rust. Python grows preset constructors on
`VortexWriteOptions` (C1: `scan_optimized(bandwidth=...)`; C3: `point_lookup()`), mirroring
the existing `default()`/`compact()` shape. The `CostModel` trait, `DecodeCostTable`, and
`WorkloadProfile` weights stay Rust-only extension points. DuckDB/DataFusion option strings
are deferred to C5 and are preset names only.

**Rationale:** presets are testable, documentable, and forward-compatible (a preset's
internals can be retuned without breaking callers); exposing the trait or raw weights across
the FFI boundary freezes implementation details into a cross-language API before the model
has proven itself. This matches how `compact` is surfaced today.

### 7. Cross-chunk consistency (doc 5 Q7)

**Decision:** punt, with instrumentation. The leaf cost model stays leaf-scoped; no rung
introduces cross-chunk state. M1's harness additionally *records per-column choice-flapping*
(distinct winning schemes per column across chunks) so the punt is a measured decision rather
than a blind one. If flapping regret turns out material, the follow-up is a sticky-choice
layer in `CompressingStrategy` (file-writer scope), designed separately.

**Rationale:** per-chunk independence is load-bearing for parallel compression
(`vortex-layout/src/layouts/compressed.rs`), and any consistency mechanism belongs to the
layer that sees all chunks of a column — the writer strategy, not the per-leaf model. Folding
it in here would couple the ladder to the file writer's concurrency design.

## The PR ladder

Sizes: **S** ≈ < 200 changed lines, **M** ≈ 200–600, **L** > 600 (tests included). Every rung
follows repo `CLAUDE.md` (narrow tests first, `cargo +nightly fmt --all` + clippy for Rust
changes, sign-off trailer). "Golden test" always means the R0 suite.

### Track A — refactor toward the pluggable model

Strictly sequential: R0 → R1 → R2 → R3 → R4 → R5. Default output stays byte-identical
through the whole track; R0 is the proof mechanism every later rung cites.

#### R0 — Golden-corpus determinism test (S–M, test-only)

- **Scope:** a snapshot test suite that compresses a fixed, seed-generated corpus with the
  default compressor and snapshots, per corpus entry, the full encoding tree and
  `nbytes` (via `insta`, already a workspace dep). Corpus: deterministic synthetic arrays
  covering every scheme's habitat — int (monotone, low-cardinality, runs, sparse-null,
  negatives), float (ALP-friendly, dict-friendly), string (FSST-friendly, dict-friendly),
  binary, decimal, temporal, nested struct/list — sized above the sampling threshold
  (> 1024 values) so the sampled path is exercised. Two variants: default features, and
  `unstable_encodings` (+`zstd,pco` via `with_compact`) so Delta/Sequence/OnPair/Pco/Zstd
  paths are pinned too.
- **Crates/files:** `vortex-btrblocks/tests/golden.rs` (+ `tests/snapshots/`). No src changes.
- **Public API:** none.
- **Tests/acceptance:** the suite itself; green twice in a row locally (determinism), green in
  CI. Snapshot churn in any later rung is the reviewable signal of a behavior change.
- **Depends on:** nothing.
- **Note:** tree+size snapshots rather than whole-file hashes: array-level compression is the
  unit under refactor, is single-threaded and deterministic (fixed sample seed), and produces
  reviewable diffs when something *does* change. File-level byte identity is additionally
  covered for free by the existing `vortex-test/compat-gen` fixtures.

#### R1 — Internal `Candidate` plumbing; keep the sampled array (S–M)

- **Scope:** pure refactor inside `choose_best_scheme` and
  `estimate_compression_ratio_with_sampling`: introduce a crate-private `Candidate` carrying
  `{scheme, estimate (ratio or measured before/after bytes), input_nbytes, n_values,
  sampled: Option<ArrayRef>, cascade context}`; sampling returns the compressed sample
  array alongside its score instead of dropping it (`estimate.rs:218-235`). Selection
  bookkeeping switches from `(scheme, EstimateScore)` tuples to `Candidate`, still compared
  by ratio. The sampled array is dropped at end of selection (no lifetime extension yet).
- **Crates/files:** `vortex-compressor/src/{compressor.rs, estimate.rs}` (+ a new
  `candidate.rs`).
- **Public API:** none (all `pub(crate)`).
- **Tests/acceptance:** golden unchanged (no snapshot churn); existing `choose_best_scheme`
  unit tests pass unmodified; `cargo nextest run -p vortex-compressor -p vortex-btrblocks`.
- **Depends on:** R0.

#### R2 — `CostModel` trait + `SizeCost`; selection becomes argmin cost (M)

- **Scope:** new `vortex_compressor::cost` module: `Cost` (opaque `f64` newtype, `Option`
  = reject), `CostModel { cost(&Candidate) -> Option<Cost>, canonical_cost(..) -> Cost }`
  (no `lower_bound` yet — that's R3), `SizeCost` implementing decision 1 exactly
  (`cost = Cost(-ratio)` over today's exact ratio values, `canonical_cost = Cost(-1.0)`;
  ZeroBytes and non-finite/subnormal ratios → `None`, reproducing `EstimateScore::is_valid`).
  Selection: winner = argmin cost with strict `<` (preserving registration-order
  tie-breaking); valid iff `cost < canonical_cost`. `AlwaysUse` and the byte-acceptance gate
  (incl. the `AnyScalarFn` carve-out) are untouched and documented as outside the model
  (decision 2). `CascadingCompressor::with_cost_model(Arc<dyn CostModel>)`, default
  `SizeCost`. `Candidate` becomes `pub` (it is the trait's argument).
- **Crates/files:** `vortex-compressor/src/cost/{mod.rs, size.rs}`, `compressor.rs`,
  `lib.rs` re-exports.
- **Public API:** `Cost`, `CostModel`, `Candidate`, `SizeCost`,
  `CascadingCompressor::with_cost_model`. Additive; `Scheme` untouched; third-party schemes
  compile unmodified.
- **Tests/acceptance:** golden unchanged; a property-style unit test that `SizeCost` ordering
  equals ratio ordering (incl. tie and invalid-ratio cases); winning-compression trace span
  gains `estimated_cost` alongside `estimated_ratio` (observability parity, doc 4).
- **Depends on:** R1.

#### R3 — Generalize the deferred early-exit to the cost model (M)

- **Scope:** the trickiest rung; two mechanisms:
  1. **Callback threshold.** `EstimateFn`'s `Option<EstimateScore>` parameter becomes a
     threshold handle (working name `SkipThreshold`) owned by the compressor, wrapping
     (best cost so far, the model, the selection-site facts). Schemes keep their side of the
     bargain — knowing their own best case — and ask
     `threshold.best_case_ratio_cannot_win(max_ratio)`; the handle builds the best-case
     `Candidate`, prices it, and compares. Under `SizeCost` this reduces to exactly today's
     `max_ratio <= best_ratio` skip.
  2. **Model-side pruning.** `CostModel::lower_bound(scheme, data) -> Cost` (default:
     never prunes) lets the *compressor* skip a deferred `Sample`/`Callback` outright when
     `lower_bound >= best cost so far`. `SizeCost` keeps the default, so behavior is
     unchanged; this is the hook that later lets decode-heavy profiles skip FSST sampling.
- **Crates/files:** `vortex-compressor/src/{estimate.rs, compressor.rs, cost/mod.rs}`;
  `vortex-btrblocks/src/schemes/integer/{delta.rs, sequence.rs}` (the two callback users).
- **Public API:** **breaking** — the `EstimateFn` type alias signature changes. Call out
  with `changelog/break`; migration is mechanical (the handle exposes the old ratio view).
- **Tests/acceptance:** golden unchanged; targeted unit tests that Delta's and Sequence's
  skip decisions fire on exactly the same inputs as before (arrays constructed on both sides
  of each threshold); `cargo nextest run -p vortex-btrblocks --features unstable_encodings`.
- **Depends on:** R2.

#### R4 — Relocate delta policy into `SizeCost` priors (S–M)

- **Scope:** `SizeCost` gains a per-scheme prior table: `{multiplier, min_ratio}` keyed by
  `SchemeId`, with exactly two non-default entries — Delta's `0.95` multiplier and `1.25`
  min-ratio — documented as *policy, not measurement*. `DeltaScheme` stops applying
  `DELTA_PENALTY` and `min_ratio` itself and reports its raw estimated ratio
  (`full_width / delta_bits`); the model applies the prior in both the cost and the R3
  threshold path (same arithmetic, same order of operations ⇒ bit-identical scores).
  `DeltaScheme::new(min_ratio)` is deprecated in favor of configuring the prior on the model;
  the field keeps working during deprecation (scheme-level override wins over the table, so
  no silent config loss). The `ALL_SCHEMES` "prefer above delta" comment moves next to the
  prior table where the policy now lives.
- **Crates/files:** `vortex-compressor/src/cost/size.rs`,
  `vortex-btrblocks/src/schemes/integer/delta.rs`, `builder.rs`.
- **Public API:** `SizeCost::with_scheme_prior(..)` (additive); deprecation of
  `DeltaScheme::new` (removal later, with `changelog/break`).
- **Tests/acceptance:** golden unchanged under `unstable_encodings` (delta is feature-gated —
  this is why R0 pins that variant); unit test that prior application reproduces the old
  post-penalty ratios bit-for-bit. Run-length/dict gates deliberately stay in schemes
  (they double as estimation-cost savers; doc 4 migration note).
- **Depends on:** R3 (threshold path must already be model-mediated).

#### R5 — Plumb the model through the builders (S)

- **Scope:** `BtrBlocksCompressorBuilder::with_cost_model(Arc<dyn CostModel>)` (default
  `SizeCost`), threaded into `CascadingCompressor` at `build()`.
  `WriteStrategyBuilder`'s `CompressorConfig::BtrBlocks` carries it to both the data and
  stats compressors. No Python change yet — deviating from the NEXT-STEPS strawman ("Python
  preset stubs") deliberately: a Python surface with only the default model behind it is
  confusable dead API; it lands with the first real preset in C1.
- **Crates/files:** `vortex-btrblocks/src/builder.rs`, `vortex-file/src/strategy.rs`.
- **Public API:** additive builder methods.
- **Tests/acceptance:** golden unchanged; a unit test that a custom (test-only, e.g.
  inverted) model injected via each builder actually changes selection — proving the
  plumbing is live, not inert.
- **Depends on:** R2 (not R3/R4 — can land in parallel with them if sequencing pressure
  appears, but the default order keeps review context linear).

### Track B — measurement

Independent of Track A (no compressor changes); can start immediately and run in parallel.
M1 is the highest-value artifact in the whole plan.

#### M1 — Decision-regret harness (L, standalone)

- **Scope:** new standalone binary `benchmarks/regret-bench` (same workspace conventions as
  `compress-bench`). For each column chunk of a fixed corpus (the existing vortex-bench
  taxi/PBI/TPCH subsets, plus a string-heavy and float-heavy selection): force-compress with
  each viable scheme via `BtrBlocksCompressorBuilder::empty().with_new_scheme(..)` /
  `exclude_schemes(..)` (existing public API — no compressor changes needed), then measure
  actual `{bytes, full-decode ns, filtered-scan ns, take ns}` per result. Output: JSONL of
  per-chunk measurements + a report of **regret distributions** (chosen vs per-objective
  oracle, per objective) and **per-column choice-flapping counts** (decision 7). One command,
  regenerable.
- **Crates/files:** `benchmarks/regret-bench/**` only.
- **Public API:** none.
- **Tests/acceptance:** runs end-to-end on a small checked-in corpus in CI (smoke); full
  corpus run documented in its README. Exit criterion of phase 0 (doc 5): the regret report
  exists and is reproducible.
- **Depends on:** nothing.

#### M2 — Missing per-encoding decode benches (S × ~4, bench-only)

- **Scope:** criterion decode benches that the appendix identifies as missing: ALP decode,
  dict gather, delta decode, pco/zstd decode. Ship as a few tiny PRs (one per
  encoding crate), mirroring the existing bench layout (e.g.
  `encodings/runend/benches/run_end_decode.rs`).
- **Crates/files:** `encodings/{alp,dict,fastlanes,pco?}/benches/`,
  `vortex-btrblocks`-adjacent zstd bench where the scheme lives.
- **Public API:** none.
- **Tests/acceptance:** `cargo bench -p <crate> --no-run`; benches produce stable numbers
  locally.
- **Depends on:** nothing.

#### M3 — Calibration generator → `DecodeCostTable` artifact (M)

- **Scope:** an `xtask` subcommand (`cargo xtask calibrate`) that runs the per-encoding
  decode benches (M2 + existing suites) and emits a machine-readable cost table —
  `ns/value` coefficients keyed by `(encoding id, dtype class, shape parameter)` plus a
  per-array fixed overhead — as a versioned JSON artifact. Check in generated defaults for
  one arch class (`x86-simd-server`) under `vortex-compressor/data/` (data file only; the
  Rust consumer type lands in C1, keeping this rung compressor-code-free). Schema documented
  in the artifact itself; regeneration alarms on drift rather than silently overwriting
  (a `--check` mode for CI use later).
- **Crates/files:** `xtask/src/**`, checked-in JSON.
- **Public API:** none (artifact schema documented as unstable until C1).
- **Tests/acceptance:** `cargo xtask calibrate --check` round-trips; generated table
  hand-checked against known orders of magnitude (bitpacking ≪ zstd).
- **Depends on:** M2 (needs the benches to exist).

#### M4 — Un-gate compression micro-benches from codspeed (S, CI-only)

- **Scope:** remove/narrow the `#[cfg(not(codspeed))]` gate at
  `vortex-btrblocks/benches/compress.rs:6` so compressor throughput is tracked per commit —
  this is also the guardrail later rungs cite for "selection overhead unchanged".
  Follow `.github/AGENTS.md` for any workflow file changes.
- **Crates/files:** `vortex-btrblocks/benches/compress.rs`, possibly
  `.github/workflows/codspeed.yml`.
- **Tests/acceptance:** codspeed run in CI shows the new benches reporting.
- **Depends on:** nothing. Opportunistic — schedule anytime.

### Stop/go checkpoint (after M1+M3)

Before any Track C rung, run the M1 harness on the full corpus and hold it against these
criteria. Regret is defined per objective as
`(metric(compressor's choice) − metric(oracle's choice)) / metric(oracle's choice)` per
column chunk; "table-replay" means re-running selection offline with the M3 table as a
simulated `TimeCost` and scoring what it *would have* picked (a script over M1's JSONL —
no compressor changes needed to evaluate this).

**GO for C1 (sequential-scan `TimeCost`) requires all of:**

1. **Headroom:** median full-decode regret ≥ 10% *or* P90 ≥ 40% on at least two of the
   corpus datasets — i.e. today's ratio-argmax leaves real scan time on the table.
2. **Capturability:** table-replay closes ≥ half of that regret gap (validates that a
   coarse calibrated table is accurate enough — doc 3's "coarse is enough" claim, verified
   on Vortex's encodings before we build the production version).
3. **Bounded size cost:** the table-replay's choices cost ≤ 10% total size regression on
   the corpus at NVMe bandwidth settings.

**NO-GO:** if P90 full-decode regret < 10% everywhere, bank Tracks A+B (the refactor has
standalone value: policy is visible, testable, and third-party-pluggable; the harness keeps
future claims honest) and do not build `TimeCost`. Record the numbers in this directory.

**Partial outcomes:**

- C3 (`point_lookup`, selective-scan) is gated separately on the *random-access/take* regret
  numbers from the same report, same thresholds.
- C2 and C4 are **not** regret-gated: the capability matrix and `gpu_decode` replace a
  hand-maintained exclusion list with a derived property; their justification is
  correctness-by-construction and maintenance cost, and C4's acceptance is parity with the
  current preset, not a regret delta. They do require C1's `TimeCost` chassis, so a NO-GO on
  C1 converts C4 into "keep `only_cuda_compatible` as-is".

### Track C — the payoff (gated on the checkpoint)

#### C1 — `TimeCost`, sequential-scan objective only (two M-sized PRs)

- **C1a (Rust core, M):** `DecodeCostTable` type in `vortex-compressor::cost` (serde over
  M3's artifact; compiled-in `x86-simd-server` defaults via `include_str!`), `Bandwidth`
  constants, and `TimeCost { bandwidth, table }` with
  `cost = bytes/bandwidth + Σ_node decode_ns(node)` walking `Candidate::sampled`'s tree;
  analytic (unsampled) candidates priced from a static per-scheme output-shape table;
  unknown encodings priced conservatively + `tracing` warn. `canonical_cost` =
  `bytes(canonical)/bandwidth`. Implements `lower_bound` (best case: minimum plausible
  bytes, zero decode) so expensive sampling (FSST) is skipped when it can't win.
- **C1b (presets + Python, S–M):** `BtrBlocksCompressorBuilder::scan_speed(Bandwidth)`
  preset (scheme set unchanged, model swapped); `with_compact()` re-derived as
  scheme-set + explicit `SizeCost` (pure re-expression, golden-checked);
  Python `VortexWriteOptions.scan_optimized(bandwidth="nvme"|"s3"|"memory")` with
  `.pyi` stubs, basedpyright/ruff clean, pytest coverage.
- **Public API:** all additive. Trait unchanged.
- **Tests/acceptance:** golden unchanged for defaults; **the checkpoint's acceptance
  numbers re-measured through the real compressor** — `scan_speed(NVMe)` improves corpus
  full-decode time ≥ 15% at ≤ 10% size regression (strawman from doc 5, to be replaced by
  the checkpoint's measured capturability); selection overhead < 1% of compress time via
  the M4-tracked bench; determinism: same config ⇒ same file (table is config, compiled-in,
  never measured at runtime).
- **Depends on:** R5, M3, checkpoint GO.

#### C2 — Capability matrix (M)

- **Scope:** `CapabilityMatrix` in `vortex-compressor::cost` per decision 3: kernel
  coverage derived from a `VortexSession`'s kernel registry
  (`vortex-array/src/optimizer/kernels.rs`) at build time; static side-table (seeded from
  the appendix) for random-access class {O(1), O(log n), block-decode} and
  GPU-decodability; conservative defaults for unknown encodings. Standalone API — no
  consumer until C3/C4.
- **Public API:** `CapabilityMatrix`, `RandomAccessClass` (additive).
- **Tests/acceptance:** unit tests assert the derived matrix matches the appendix's
  verified table for built-in encodings (which doubles as drift detection when kernels
  land); unknown-encoding default path covered.
- **Depends on:** R2 only (parallelizable with C1).

#### C3 — Selective-scan + random-access terms; `point_lookup`/`query` presets (M–L)

- **Scope:** extend `TimeCost` with the two remaining terms per the decision-1 baseline
  axiom (written into the module docs as the normative statement): `pushdown_aware_scan_ns`
  (kernel hit ⇒ encoded-op cost; miss ⇒ full decode + canonical op) and `ra_class_ns`
  (priced from C2's classes). `WorkloadProfile { bandwidth, weights {full_scan,
  selective_scan, random_access} }`; presets `point_lookup()`, `query(Bandwidth)`. Python
  `point_lookup()`.
- **Public API:** `WorkloadProfile`, new presets (additive).
- **Tests/acceptance:** gated on the checkpoint's random-access regret; acceptance =
  `benchmarks/random-access-bench` improves under `point_lookup` on the corpus without
  pathological size loss (strawman ≤ 15% size regression); worked-example table from doc 4
  (RunEnd vs Dict, Delta vs FoR+BP) encoded as unit tests.
- **Depends on:** C1, C2.

#### C4 — `gpu_decode` preset; retire the hand list (M)

- **Scope:** `gpu_decode()` = `TimeCost` + `cost = None` (reject) for candidates whose tree
  contains a non-GPU-decodable node per C2. `only_cuda_compatible()` reimplemented on top,
  marked deprecated with a migration note (kept one release; removal is `changelog/break`).
- **Tests/acceptance:** on the golden corpus + vortex-bench datasets, `gpu_decode` chooses
  only GPU-decodable trees and reproduces (or beats, by ratio) the current preset's
  choices; `vortex-test/e2e-cuda` remains green (files written under the preset decode on
  GPU).
- **Depends on:** C1, C2.

#### C5 — Per-column profiles + external surface (M)

- **Scope:** per-column `WorkloadProfile` overrides riding the existing per-field-path
  mechanism (`WriteStrategyBuilder::with_field_writer`, `strategy.rs:189`) with a
  convenience `with_field_cost_model(path, model)`; Python per-column preset dict;
  DuckDB `COPY ... (COMPRESSION_PROFILE 'scan')`-style option and a DataFusion writer
  option, both preset-strings-only per decision 6.
- **Tests/acceptance:** a file written with mixed per-column profiles shows the expected
  per-column encoding differences (integration test); Python/DuckDB option round-trips.
- **Depends on:** C3 (needs profiles to exist).

### Adjacent small wins

On-path, PR-sized, schedule where convenient (all independent of each other):

| # | PR | Size | Constraint |
|---|---|---|---|
| A1 | Zero-byte/all-null sampling fix (#7268): distinguish real zero-byte wins from all-null sample artifacts (`estimate.rs:108-118` TODO) | S–M | **behavior-changing** — must land after R0 (golden snapshots churn and get reviewed as the fix's evidence); hardens every estimate the models consume |
| A2 | Cache Sequence's trial encoding (`sequence.rs:94` TODO) so `compress` doesn't re-encode | S | perf-only, golden-neutral; anytime |
| A3 | Estimator-accuracy mining script: corpus run with `RUST_LOG=vortex_compressor::encode=debug`, `jq` over `estimated_ratio` vs `achieved_ratio` (spans already exist, `vortex-compressor/src/lib.rs:45-63`) | S | lives next to M1; feeds calibration of size estimates |

Out of this plan entirely (they *consume* the cost model): adaptive cascade search
(`MAX_CASCADE` replacement — needs C1 + R3's `lower_bound` as its pruning rule), feedback
loops/learned models, `DictStrategy` integration (decision 4), sticky cross-chunk choices
(decision 7), finer-grained lazy stats.

## Dependency sketch

```
Track A (refactor)      R0 ──► R1 ──► R2 ──► R3 ──► R4 ──► R5 ─────────┐
                                       │                               │
                                       └────────► C2 (capability) ────┤
Track B (measurement)   M2 ──► M3 ─────────────────────────────────┐  │
                        M1 ────────────────► [STOP/GO checkpoint] ─┤  │
                        M4 (anytime, CI-only)                      │  │
                                                                   ▼  ▼
Track C (payoff)                                            C1a ► C1b ──► C3 ──► C5
                                                                    │      ▲
                                                                    └► C4 ◄┘ (C4 needs C1+C2 only)

Adjacent: A1 after R0; A2/A3 anytime.
```

Parallelism that matters: **M1/M2/M4 can start the same day as R0** (different crates, no
shared files); M3 follows M2; the checkpoint needs only M1+M3, not Track A; C2 needs only
R2 and can proceed while R3–R5 land. The critical path to the first user-visible payoff is
R0→R2→R3→R4→R5 + checkpoint → C1.

## Out of scope for this ladder

Restating the boundary so scope creep is visible: no default flip (decision 5), no dict-layer
move (decision 4), no cross-chunk state (decision 7), no adaptive cascading, no learned
models, no measured timing on the write path — ever (determinism is an invariant; cost models
are pure functions of the candidate and configured tables).
