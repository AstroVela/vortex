# Research: an execution-speed-aware cost model for the compressor

Investigation for [#7697 (Epic: Compressor Improvements)](https://github.com/vortex-data/vortex/issues/7697)
— specifically the "Different cost models (right now it is only based on compression ratio,
not compute)" item. Written July 2026 against the post-#7216 `vortex-compressor` /
`vortex-btrblocks` split.

## The documents

Read the TL;DR below, then dip into whichever doc you need — each opens with its own TL;DR
and a mini table of contents.

| Doc | What's in it | Read it when |
|---|---|---|
| [1. How the compressor decides today](./01-how-the-compressor-decides-today.md) | The decision pipeline, where every hardcoded judgment lives (with file:line), and the speed concerns already smuggled into the ratio scale | you want the current-state map |
| [2. Prior art](./02-prior-art.md) | How BtrBlocks, Nimble, DuckDB, Data Blocks, CodecDB, and the cost-model literature handle this; what transfers to Vortex | you want evidence this is tractable |
| [3. What a speed-aware cost model needs](./03-what-a-speed-aware-cost-model-needs.md) | The analysis: decomposing "execution speed", the five hard problems, what's predictable, what data is missing | you want the *why* behind the design |
| [4. Design: pluggable cost model](./04-design-pluggable-cost-model.md) | The concrete proposal: `Candidate` + `CostModel` trait, how selection changes, `SizeCost`/`TimeCost`, presets | you want the *how* |
| [5. Roadmap & open questions](./05-roadmap-and-open-questions.md) | Measurement-first phasing with acceptance criteria, risks, unresolved decisions | you want next steps |
| [Appendix: encoding capabilities](./appendix-encoding-capabilities.md) | Verified per-encoding kernel coverage, random-access classes, decode notes | reference material |

## TL;DR

**Is it even possible?** Yes — *if the objective is framed as estimated time, not as a rival
score to ratio*. `T ≈ bytes/bandwidth + Σ decode + compute`, where saved bytes have a
bandwidth-dependent price. Today's ratio-maximizing compressor is the `bandwidth → 0` point of
that family, which is why a time model can subsume the current behavior instead of replacing
it. What's *not* possible is a workload-free notion of "fast": the same column wants different
encodings for full scans vs point lookups. The unknowables (workload, reader hardware) must
become explicit parameters with safe defaults — which is precisely the argument for a
pluggable model.

**What do others do?** Every system surveyed lands on one of four strategies (doc 2): curate
a fast-decoding-only pool and then maximize ratio (BtrBlocks, FastLanes, DuckDB's format —
and Vortex today); size × hand-tuned per-encoding penalty (DuckDB's ×2.0 Roaring penalty,
Nimble's 0.7/0.9/1.0 read factors — and Vortex's delta tax, unofficially); calibrated or
learned per-objective time models (Damme et al./MorphStore — the evidence that decode cost is
predictable); or speed as a hard constraint / user-chosen objective (HyPer Data Blocks,
Google's Artus — whose "user specifies the relative value of size versus speed" is the
closest precedent for what doc 4 proposes). Meta's Nimble is the production proof that a
*pluggable selection policy* works. Nobody has solved workload-unknown-at-write-time except
by making the workload an input.

**What's the real gap?** Not data statistics (min/max/runs/distinct already cover what a cost
table needs) but the *cost side*: Vortex has no representation of decode cost, kernel
coverage, random-access class, or IO price anywhere the compressor can see. Meanwhile the
codebase is already full of implicit speed policy with nowhere to live: the delta "tax"
(`DELTA_PENALTY = 0.95` — a speed judgment expressed as a ratio fudge factor), scheme
registration-order comments, and the hand-maintained `only_cuda_compatible()` exclusion list
(a two-valued cost model: possible/impossible).

**Three findings that make this cheaper than it looks:**

1. **Sampled candidates already carry their full encoding tree** — the compressor compresses
   a ~1% sample through the real cascade and then throws away everything except `nbytes`.
   Walking that tree and pricing each node gives cascade-aware decode costs with zero new
   `Scheme` API.
2. **The capability bits are mechanically derivable.** Compute specialization lives in a
   session-scoped kernel registry keyed by (operation, encoding); whether an encoding
   filters/compares natively or degrades to canonicalize-then-compute is a lookup, not
   folklore. The big speed effects are these step functions (kernel exists / GPU-decodable /
   O(1) random access), not ±20% decode-throughput differences — coarse models capture most
   of the value.
3. **The measurement infrastructure mostly exists** (compress-bench decode throughput,
   random-access-bench, per-commit tracking). What's missing is a per-encoding calibration
   table and a *decision-regret harness* — and the regret harness is the single highest-value
   artifact: it tells us how far today's choices are from oracle under any objective, before
   we build anything.

**Should the cost model be pluggable?** Yes, in a specific shape: one small trait
(`CostModel`: price a `Candidate`, price the canonical baseline, provide a lower bound for
early-exit), with *policy as data* — a decode-cost table, a capability matrix, and a workload
profile. Users touch **presets** (`max_ratio` — today's behavior and the default,
`scan_speed(bandwidth)`, `point_lookup`, `gpu_decode`), not the trait. Dynamic per-workload
selection then falls out: pass a different model per write (or per column, via the existing
per-field-path mechanism in the file writer).

**Does the current compressor fit?** Better than expected. Selection has a single choke point
(`choose_best_scheme`), and the two-pass deferred-estimation machinery generalizes cleanly
(the ratio threshold becomes a cost lower bound — which can also skip expensive FSST sampling
when it can't win). The honest mismatches: policy is currently *distributed into the schemes*
(the refactor is extraction, not invention); the strict-total-order assumption must be
preserved (fine — cost is scalar); dictionary encoding is decided *above* the compressor by
the file writer's `DictStrategy`, so the model can't reach the most workload-sensitive
encoding decision without a follow-up integration; and per-chunk independence means no model
can see cross-chunk consistency effects.

**Recommended path** (detail in doc 5): ⓪ build the calibration table + regret harness and
put numbers on today's regret → ① extract `CostModel` with `SizeCost` as default,
byte-identical output, delta-tax and friends relocated into it → ② `TimeCost` with the
sequential-scan objective only, presets in Rust + Python → ③ capability-aware terms
(pushdown, random access, GPU) and per-column profiles → ④ use the cost model to replace
`MAX_CASCADE = 3` with bounded search → ⑤ feedback loop / learned models (research). Each
phase has an acceptance number from the phase-0 harness; if phase 0 shows today's regret is
small under realistic objectives, stop early and bank the refactor.

## Answers to the questions in the brief

| Question | Short answer |
|---|---|
| Is this possible? | Yes, as a parameterized *time* model with honest inputs; no, as a workload-free oracle. Expect "avoids pathological choices, enables meaningful presets", not "predicts query times". |
| What are the issues/constraints? | Workload & hardware unknown at write time; timing-on-samples is noisy *and* breaks determinism (use calibrated tables, not measurements, on the write path); errors compound along cascades; per-chunk independence; policy currently lives inside schemes; acceptance is size-gated. Docs 3 & 5. |
| Do we need information we don't track? | Yes, three things, in value order: a decode-cost table (calibration artifact), a capability matrix (derivable from the kernel registry + tiny annotations), a workload/environment profile (new input with a safe default). Notably *not* more data stats. |
| Pluggable & dynamic per workload? | Yes — trait + policy-as-data + presets; per-write and per-column. This also gives #7697's other items (scheme priorities, FSST estimation policy, adaptive cascading) a principled home. |
