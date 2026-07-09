# NEXT STEPS — briefing for the implementation-planning session

*This file is the prompt. If you are an agent reading this on a fresh conversation: your task
is to produce an **implementation plan** that carries the research in this directory into the
codebase as a sequence of **small, independently reviewable PRs**, ending at a pluggable
`CostModel` with an execution-time objective. You are planning (and possibly starting) the
work — the research phase is done; do not redo it.*

## Read order

1. [`README.md`](./README.md) — conclusions and vocabulary (5 minutes).
2. [`04-design-pluggable-cost-model.md`](./04-design-pluggable-cost-model.md) — the target
   design: `Candidate`, `CostModel`, `SizeCost`/`TimeCost`, presets, hook points.
3. [`05-roadmap-and-open-questions.md`](./05-roadmap-and-open-questions.md) — coarse phases,
   acceptance criteria, open questions your plan must take positions on.
4. Docs 1–3 and the appendix as reference while planning (current-state map with file refs,
   prior art, analysis).

## Ground rules

- **Verify before planning.** The docs cite `file:line` against `develop` at `11f15a9`
  (July 2026). Rebase this branch (or your working branch) onto current `develop` first, then
  re-anchor on symbol names (`choose_best_scheme`, `EstimateScore`, `DELTA_PENALTY`,
  `ALL_SCHEMES`, `CompressorPlugin`, …) — treat line numbers as approximate and re-verify any
  claim you build on.
- **Small PRs, each independently valuable.** Target the size of a normal reviewable Vortex
  PR (roughly: one idea, one crate focus, tests included). If a rung of the ladder below
  can't be described in two sentences, split it.
- **Default behavior does not change until an explicit flip.** Every refactor rung must keep
  the default compressor's output byte-identical (prove it — see rung R0). New behavior
  ships opt-in behind presets/builders.
- **Determinism is an invariant.** No timing measurements on the write path; cost models are
  pure functions of (candidate, configured tables).
- **Don't break third-party `Scheme` implementors** without calling it out; prefer additive
  trait changes with defaults. Note `changelog/break` labels where unavoidable.
- **Each PR states its own verification** (which tests, which benches) and follows the repo
  `CLAUDE.md` (narrow tests first, fmt/clippy for Rust changes, sign-off trailer).
- Where doc 5's open questions force a decision (cost units, canonical baseline for the
  selective-scan term, byte-acceptance axiom, where capability annotations live), **your plan
  must pick an answer and record the rationale** — don't leave them open.

## Deliverable

A plan document (add it to this directory as `IMPLEMENTATION-PLAN.md` on this branch, unless
told otherwise) containing:

- The PR ladder: for each PR — scope, crates/files touched, new/changed public API, tests and
  acceptance check, estimated size (S/M/L), and dependencies on earlier rungs.
- A dependency sketch showing which rungs can proceed in parallel (refactor track vs
  measurement track).
- The decisions taken on doc 5's open questions, each with a one-paragraph rationale.
- A "stop/go checkpoint" after the measurement rungs: what the regret numbers would have to
  show to justify (or cancel) the `TimeCost` rungs.

Only file GitHub sub-issues under #7697 if the user asks; the plan document comes first.

## Candidate PR ladder (starting point — refine, reorder, and challenge it)

Two parallel tracks after R0. "Safe" = default output provably unchanged.

**Track A — refactor toward the pluggable model** (vortex-compressor, vortex-btrblocks)

| # | PR | Why it's small & safe |
|---|---|---|
| R0 | Golden-corpus determinism test: compress a fixed corpus, snapshot encoding trees + sizes | test-only; the safety net every later rung cites |
| R1 | Internal `Candidate` plumbing in `choose_best_scheme`; keep the sampled array alongside its `nbytes` | pure refactor inside one function; no trait yet |
| R2 | `CostModel` trait + `SizeCost` default; selection becomes argmin cost; `CascadingCompressor::with_cost_model` | golden test proves identical output; trait is `pub` but only default impl exists |
| R3 | Generalize the deferred-callback `best_so_far` threshold to `CostModel::lower_bound` | touches the `EstimateFn` contract; must reproduce ratio-threshold semantics under `SizeCost` (Sequence/Delta callbacks are the test cases) |
| R4 | Relocate policy constants (delta tax, `min_ratio`) into `SizeCost` priors; document them as policy | behavior-preserving by construction; deletes policy from schemes |
| R5 | Plumb the model through `BtrBlocksCompressorBuilder` and `WriteStrategyBuilder` (+ Python preset stubs) | inert surface area until a non-default model exists |

**Track B — measurement (benchmarks/, xtask; no compressor changes)**

| # | PR | Why it's small & safe |
|---|---|---|
| M1 | Decision-regret harness: force-scheme compression over a corpus, measure {bytes, decode ns, filtered-scan ns, take ns}, report regret vs per-objective oracle | standalone binary under `benchmarks/`; produces the go/no-go numbers |
| M2 | Missing per-encoding decode benches (ALP decode, dict gather, pco/zstd/delta decode) | a few tiny bench-only PRs |
| M3 | Calibration generator: run decode benches, emit a machine-readable `DecodeCostTable`; check in defaults for one arch class | artifact + generator; no consumer yet |
| M4 | (Opportunistic) un-gate compression micro-benches from codspeed exclusion so they're tracked | CI-only |

**Track C — the payoff (needs R5 + M3; gated on M1's numbers)**

| # | PR | Notes |
|---|---|---|
| C1 | `TimeCost` with the sequential-scan objective only (`bytes/bandwidth + Σ decode_ns` over the sampled tree); `scan_speed(bandwidth)` preset | opt-in; acceptance numbers from M1 harness |
| C2 | Capability matrix derived from the session kernel registry + random-access/GPU annotations | standalone API; consumed later |
| C3 | Selective-scan + random-access cost terms; `point_lookup` preset | needs C2 |
| C4 | `gpu_decode` preset; deprecate the `only_cuda_compatible` hand list | needs C2; compare choices against the current preset |
| C5 | Per-column workload profiles via the existing per-field-path mechanism; Python/DuckDB preset surface | UX layer |

**Adjacent small wins** (from #7697, on-path and PR-sized — schedule where convenient):
the zero-byte/all-null sampling fix (#7268) hardens the estimates every model consumes;
"Sequence trial-encodes then throws away the array" (cache the encoding or add a cheap
`is_sequence` probe); finer-grained lazy stats. The adaptive-cascade search and any
feedback/learned work stay **out of scope** for this plan (they consume the cost model;
plan them only after C1 exists).

## Ultimate goal (restated)

A user writing Vortex files can say — per write or per column — "optimize for scan speed on
NVMe", "optimize for point lookups", "optimize for GPU decode", or say nothing and get
today's behavior byte-for-byte; the compressor's policy lives in one visible, testable,
swappable place; and every step that got there was a PR a reviewer could hold in their head.
