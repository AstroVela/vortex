# 5. Roadmap, risks, and open questions

*Read this for the staged plan (measurement first), the acceptance criteria per phase, and the
questions that genuinely don't have answers yet.*

**TL;DR:** Don't start with the trait — start with the ability to *score a decision after the
fact*. Vortex already tracks compression ratio, encode/decode throughput, random access, and
end-to-end query time across commits; what's missing is (a) a per-encoding calibration table
and (b) a "decision regret" harness that says how far the compressor's choices are from oracle
under a given objective. With those, the refactor (phase 1) is mechanical and every later
phase has a number to justify itself with.

- [Phase 0 — measure (prerequisite for everything)](#phase-0--measure)
- [Phase 1 — extract the cost model, change nothing](#phase-1--extract-the-cost-model-change-nothing)
- [Phase 2 — the first real time model (sequential scan)](#phase-2--the-first-real-time-model)
- [Phase 3 — capability-aware and workload-aware](#phase-3--capability-aware-and-workload-aware)
- [Phase 4 — adaptive cascading (uses the model, stretch)](#phase-4--adaptive-cascading)
- [Phase 5 — feedback and learning (research)](#phase-5--feedback-and-learning)
- [Risks](#risks)
- [Open questions](#open-questions)

## Phase 0 — measure

Infrastructure that must exist before any model can be trusted. Good news: much is in place —
`benchmarks/compress-bench` (ratio + encode/decode throughput over taxi/PBI/TPCH subsets),
`benchmarks/random-access-bench`, DataFusion/DuckDB query benches, v3 JSONL records shipped to
bench.vortex.dev per commit. Gaps:

1. **Calibration harness → `DecodeCostTable`.** A binary (xtask or bench) that runs
   per-encoding decode / take / compare microbenches (many already exist as criterion suites,
   see appendix; add the missing ones: ALP decode, dict gather, delta decode, pco/zstd decode)
   and emits the coefficient table as a machine-readable artifact. Re-run per arch class;
   check the defaults into the repo; alarm on drift in CI rather than regenerating silently.
   Note: the compression micro-benches are currently `#[cfg(not(codspeed))]`-gated out of CI
   tracking entirely — worth fixing independently.
2. **Decision-regret harness.** For each column chunk in a fixed corpus (PBI + taxi + a
   string-heavy and a float-heavy set): force-compress with each viable scheme (bounded
   cascade), measure actual `{bytes, full-decode ns, filtered-scan ns, take ns}`, and compare
   what the compressor chose against the per-objective oracle. Output: *regret distributions*.
   This answers, with numbers, three questions we can only hand-wave today:
   - How often is today's ratio-argmax also the speed-optimal choice? (If "usually", phases
     2–3 shrink; if "rarely", they grow.)
   - Which mistakes dominate — wrong scheme family, or wrong cascade depth?
   - How accurate does a `DecodeCostTable` need to be before model-picked ≈ oracle? (The
     integer-codec literature suggests coarse is enough — verify on Vortex's encodings.)
3. **Estimator-accuracy mining.** The tracing spans already emit `estimated_ratio` vs
   `achieved_ratio` per winner (`vortex-compressor/src/lib.rs:45-63`); a small script over a
   corpus run turns that into calibration data for the size estimates the cost model will
   consume. Cheap and immediately useful for #7697's other bullets too.

**Exit criterion:** a report (regenerable, one command) showing per-objective regret of the
current compressor on the corpus. This is the baseline every later phase must beat.

## Phase 1 — extract the cost model, change nothing

The refactor from doc 4 with `SizeCost` as the only model:

- `Candidate` + `CostModel` + argmin selection in `choose_best_scheme`; sampled arrays kept
  (not just their `nbytes`).
- Delta tax / `min_ratio` / registration-order preferences relocated into `SizeCost` as
  documented priors.
- **Acceptance:** golden-corpus test proving byte-identical files vs `develop`; compress-bench
  encode-throughput unchanged within noise. No public behavior change; `Scheme` implementors
  unaffected.

Deliberately boring. Its value is that policy becomes *visible and testable* — several other
#7697 bullets (scheme priorities, exclusion-rule reasoning, FSST estimation policy) get a
natural home.

## Phase 2 — the first real time model

Smallest model that means something: **sequential scan objective only** —
`cost = bytes/bandwidth + Σ decode_ns` over the candidate tree.

- `DecodeCostTable` from phase 0, compiled-in defaults for one arch class (x86-simd server).
- Presets: `max_ratio` (default, unchanged), `scan_speed(bandwidth)`, and re-derive
  `with_compact()` as `max_ratio` + extended scheme set.
- Surface: `BtrBlocksCompressorBuilder` + `WriteStrategyBuilder` + Python preset.
- **Acceptance (from the phase-0 harness):** `scan_speed(NVMe)` improves corpus full-decode
  time by a target margin (strawman: ≥15%) at bounded size regression (strawman: ≤10%);
  `max_ratio` output unchanged; selection overhead <1% of compress time.

Scope note: resist adding the selective-scan/pushdown term here — it needs the capability
matrix and multiplies the validation surface. Sequential scan alone already fixes cascade
pathologies and the zstd/pco/delta traps.

## Phase 3 — capability-aware and workload-aware

- `CapabilityMatrix` derived from the session kernel registry + static random-access/GPU
  annotations (new, small: an enum on the encoding vtable or a side registry).
- Add the `selective_scan` and `random_access` cost terms; presets `query(bandwidth)`,
  `point_lookup`, `gpu_decode` (retiring the hand-maintained `only_cuda_compatible` list).
- Per-column `WorkloadProfile` overrides through the existing per-field-path mechanism.
- **Acceptance:** regret harness under mixed objectives; `gpu_decode` reproduces (or beats)
  the current CUDA preset's choices; random-access benchmark improves under `point_lookup`
  without pathological size loss.

## Phase 4 — adaptive cascading

With costs additive along the tree and per-scheme lower bounds available, the exclusion-
bounded search tree from #7216 gets an objective and a pruning rule (branch-and-bound instead
of `MAX_CASCADE = 3`). This is #7697's stretch goal; it should *consume* the cost model, not
be designed together with it.

## Phase 5 — feedback and learning

Research-grade, sequenced last on purpose:

- **Feedback loop:** per-encoding decode/compute time attribution in scans (`vortex-metrics`
  currently records nothing about this) → aggregate → per-dataset profile suggestions or a
  re-encoding advisor. Raises real questions (metrics cardinality, privacy of workload data,
  cross-file attribution) — none block phases 0–4.
- **Learned model:** phase 0's regret harness produces exactly the (features → measured cost)
  ground truth a CodecDB-style learned selector needs. Only worth it if the analytical model's
  residual regret is demonstrably material.

## Risks

| Risk | Mitigation |
|---|---|
| Eval corpus unrepresentative → model tuned to benchmarks | reuse the existing bench datasets people already trust; keep the harness one-command so new datasets are cheap to add |
| Calibration table drifts from real kernels | CI job re-measures and *alarms* (doesn't silently regenerate); table versioned with the code |
| Hardware heterogeneity (write here, read there) | coarse arch classes + bandwidth as an explicit parameter; document that `max_ratio` remains the safe default for unknown readers |
| Cost-model complexity creeps into the write path | budget: pure table lookups, no I/O, no timing; compress-bench throughput gate in CI |
| Sampled tree ≠ full-array tree (estimation noise inherited from sampling) | same exposure as today's ratio estimate; phase-0 estimator-accuracy mining quantifies it; the zero-byte-sample issue (#7268) stays orthogonal but should be fixed first regardless |
| Third-party schemes mispriced | conservative unknown-scheme defaults + tracing warning; optional `cost hint` API later if demand exists |
| Per-chunk decision flapping becomes *more* visible under new models | out of scope for the leaf model; consider a sticky-choice layer in `CompressingStrategy` as separate work |

## Open questions

Genuinely unresolved; each is a decision, not a research project:

1. **Units and the canonical baseline.** ns/value is proposed (auditable, composable). But
   what is `canonical_cost` on the *selective-scan* term when canonical also skips decode?
   Needs one careful write-up so all models agree on the baseline.
2. **Does the byte-acceptance invariant ever bend?** A candidate can be faster *and larger*
   (e.g. keep canonical instead of pco under `scan_speed`). Keeping `after < before` as a hard
   gate is safe and recommended (it only forces "canonical", never a bad encoding), but it
   should be an explicit axiom.
3. **Who owns capability annotations?** Kernel coverage is derivable; random-access class and
   GPU-decodability need a declared home (encoding vtable metadata vs. a registry side-table
   in `vortex-compressor`). Third-party encodings need a conservative default.
4. **Where does dictionary policy live?** The writer's `DictStrategy` decides dict encoding
   above the compressor (doc 1). If the cost model should influence it — and for speed
   objectives it must, dict being the most workload-sensitive encoding — either the strategy
   consults the model, or dict selection moves back into the compressor (#7697's
   "hardcode DictScheme" bullet, inverted).
5. **What's the default bandwidth?** `max_ratio` (bandwidth→0) is the conservative default and
   preserves behavior; but if most Vortex reads are NVMe-local, a future default flip is a
   product decision that phase-0 data should inform.
6. **How much surface in Python/DuckDB/DataFusion?** Proposal: presets only (strings/enums),
   trait stays Rust. DuckDB `COPY ... (COMPRESSION_PROFILE 'scan')`-style options are cheap
   once presets exist.
7. **Cross-chunk consistency** — real effect on execution, no owner in this design. Punt with
   eyes open, or fold into the file-writer strategy layer later?
