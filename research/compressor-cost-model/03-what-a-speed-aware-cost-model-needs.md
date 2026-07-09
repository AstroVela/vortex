# 3. What a speed-aware cost model actually needs

*Read this for the analysis: what "optimize for execution speed" means precisely, why it's
hard, what's predictable anyway, and what information Vortex would have to start tracking.*

**TL;DR:** "Execution speed" is not a property of an encoding — it's a property of
*(encoding tree, operation, data shape, hardware)*. That sounds fatal, but it isn't: the right
move is to stop treating ratio and speed as competing objectives and model **time**, in which
saved bytes have a bandwidth-dependent price and decode work has a calibratable price. The
genuinely unknowable parts (workload, reader hardware) must become explicit *parameters*
instead of implicit assumptions. Most of the achievable win comes from coarse, discrete
signals — kernel coverage, random-access class, cascade depth, GPU-decodability — not from
nanosecond-accurate decode models.

- [Decomposing "execution speed"](#decomposing-execution-speed)
- [The five hard problems](#the-five-hard-problems)
- [What is actually predictable](#what-is-actually-predictable)
- [Information Vortex doesn't track today](#information-vortex-doesnt-track-today)
- [Verdict: is this possible?](#verdict-is-this-possible)

## Decomposing "execution speed"

For a leaf array of `n` values stored under encoding tree `E`, the wall-clock cost of a query
fragment decomposes roughly as:

```
T(E) ≈ T_io + T_decode + T_compute [+ T_random]

T_io      = bytes(E) / effective_bandwidth        ← where RATIO helps SPEED
T_decode  = Σ over tree nodes of decode_cost(node, n_node)   ← cascades pay per level
T_compute = cost of the operator on the representation it actually runs against
T_random  = per-row access cost × number of point accesses
```

Three consequences of writing it down this way:

1. **Ratio is not the enemy of speed — it's a term in the speed model.** On cold object
   storage (effective bandwidth per stream measured in tens–hundreds of MB/s), `T_io`
   dominates and today's ratio-maximizing behavior is *approximately correct*. On NVMe or
   memory-resident data, `T_decode`/`T_compute` dominate and it isn't. The "cost model
   question" is really: *what is a saved byte worth in nanoseconds?* That price
   (`1/effective_bandwidth`) is an environment fact the compressor currently has no way to
   receive.
2. **Cascades compose differently for size vs. time.** Size composes by replacement (the
   child's bytes *are* the parent's payload). Decode time composes by *accumulation* — every
   level adds a pass, and for point access a cascade can force whole-block decompression per
   layer (FastLanes measured BtrBlocks at 800× its single-value retrieval latency for exactly
   this reason — doc 2). Under a time model, `MAX_CASCADE = 3` stops being a magic constant
   and becomes an emergent property:
   deeper cascades must pay for their extra decode pass with IO savings. This directly serves
   #7697's "adaptive cascading" item — an explicit cost gives the bounded search tree from
   #7216 an objective function, and per-scheme cost lower bounds give it a pruning rule.
3. **`T_compute` is where the sign can flip.** For some (encoding, operation) pairs the
   encoded form is *faster than canonical* — filtering/grouping on dictionary codes, slicing
   run-ends, aggregating constants — because the operator has a specialized kernel and touches
   less data. For pairs with no kernel, execution degrades to canonicalize-then-compute, so
   the *entire* decode cost lands on the query's critical path. Whether a kernel exists is a
   discrete, statically-knowable bit that today lives only in the kernel registry, invisible
   to the compressor. (This also means "optimize for execution speed" is not always "compress
   less": dictionary-encoding a filter-heavy string column can beat canonical.)

## The five hard problems

Honest accounting of why this is an unsolved problem in general:

### 1. The workload is unknown at write time
Encoding binds at write; operations arrive at read. "Execution speed" without a workload is
ill-posed: delta might be optimal for full scans and catastrophic for point lookups *on the
same column*. Every system that has attacked this either (a) fixes an assumed workload
(BtrBlocks: "decompress everything fast"), (b) takes hints, or (c) closes a feedback loop from
observed queries. There is no fourth option — which is the strongest argument that the cost
model must be a *parameter* (pluggable, per-write, per-column), not a constant.

### 2. The reader's hardware is unknown too
A file written once is read from many machines: AVX-512 servers, laptops, GPUs (the
`only_cuda_compatible` preset exists precisely because one reader class couldn't decode some
encodings at all). Decode throughput varies by an order of magnitude across these. Calibration
tables are only valid per architecture class. Mitigation, not solution: coarse hardware
classes (x86-simd / arm / gpu) + user-supplied bandwidth, defaults chosen for the common
deployment.

### 3. Estimation error compounds along cascades — and timing samples is a trap
Sampled *size* is measured directly, whole-tree, cheaply. Time is not: 1024-value samples
decode in ~1µs, where allocator noise, cache-hot bias, and turbo swamp the signal; and
measured-time selection breaks the compressor's determinism guarantee (same input → same
file), which the codebase explicitly documents as a requirement (`Scheme` docs,
`vortex-compressor/src/scheme.rs:142-144`). Conclusion: per-node *model* costs (table
lookups), summed along the tree, are the workable path; direct timing belongs offline in the
calibration harness, not in the write path.

### 4. The objective is a vector even for one user
Full-scan decode, selective scan with pushdown, and random access rank encodings differently.
Any scalar cost is a weighted mix; someone chooses weights. This is fine (query optimizers
live like this) but it must be explicit — the weights *are* the workload profile.

### 5. Choices interact across chunks and with the layout
Per-chunk-independent decisions can flip encodings chunk-to-chunk (dict ↔ fsst), which
execution pays for in dispatch churn and lost dictionary reuse, and which no per-leaf cost
model can see. Layout-level decisions (chunk size, stats, pruning) also move query time more
than some encoding swaps. Scope discipline: the leaf cost model should stay leaf-scoped, and
cross-chunk consistency should be a separate, later mechanism (e.g. a sticky-choice layer in
the file writer).

## What is actually predictable

The good news, and why "possible" wins over "hopeless":

- **Decode cost per node is far more predictable than compression ratio.** Ratio depends on
  data values; decode throughput depends mostly on *(encoding, bit width / params, dtype,
  arch)* — bitpacking unpacks at near-memory speed regardless of the values; FSST decode is
  ~bytes-proportional; dict decode is a gather; run-end decode is ~runs + output. A table of
  `ns/value` coefficients keyed by encoding and one or two parameters, calibrated by
  microbenchmarks, is the approach the integer-compression literature validated (doc 2:
  Damme et al.'s grey-box model built exactly this from per-bit-width calibration profiles,
  and its learned successor reports that mis-selections under the *decompression-time*
  objective cost only 1–2% regret — speed objectives forgive estimation error). Vortex
  already has many of the needed criterion benches next to the encodings.
- **The discrete capability bits are free.** Kernel coverage (does `filter`/`compare`/`take`
  run natively on this encoding?), random-access class (O(1) vs binary-search vs
  block-decode), GPU-decodability — all statically knowable from the session's kernel
  registry and vtables at compressor-build time. No estimation involved; today they're
  hand-encoded as comments and preset exclusion lists.
- **Sampled candidates already carry their whole tree.** When the compressor samples, it gets
  back a real compressed array — `Dict(codes=BitPacked, values=FSST(...))` — and then throws
  everything but `nbytes` away (`estimate.rs:218-229`). Walking that tree and summing table
  costs gives a cascade-aware time estimate *with zero new scheme API*.
- **Where precision matters, it matters coarsely.** The practical failure modes a speed model
  must prevent are order-of-magnitude ones: a 4-level cascade on a hot column, FSST on a
  column the workload only ever filters, delta under point lookups, GPU-undecodable choices
  in a GPU pipeline. A model that is wrong by 2× on decode ns/value still gets all of these
  right. (Corollary: don't over-invest in model fidelity before the evaluation harness in
  doc 5 exists to prove it pays.)

## Information Vortex doesn't track today

Ordered by value-per-effort; (a) and (b) are prerequisites for any real version of this.

| # | New information | Form it should take | Used for |
|---|---|---|---|
| a | **Per-encoding decode/compute cost coefficients** | a calibration table (compiled-in defaults per arch class; regenerable via a bench harness, e.g. `cargo xtask calibrate`) | `T_decode`, `T_compute` terms |
| b | **Capability metadata per encoding** | queryable facts: kernel coverage from the registry; a small static annotation for random-access class & GPU-decode | step-function penalties; principled replacement for `only_cuda_compatible` |
| c | **Environment descriptor** | `effective_bandwidth` (+ hardware class) on the write config | the byte→time exchange rate |
| d | **Workload profile** | weights over {full-scan, selective-scan, random-access} + optional per-column overrides | objective weights |
| e | **Feedback from executed queries** | per-encoding time attribution in `vortex-metrics`, aggregated offline | closing the loop later (re-encode advisor, learned models) |

Notably, the *data-side* statistics are largely sufficient already (min/max/runs/distinct
cover the parameters the cost table is keyed on). The gap is entirely on the cost side — the
compressor can describe the data but has no vocabulary for time.

## Verdict: is this possible?

**Yes, with three provisos:**

1. **It's a time model, not a second score.** Frame the objective as estimated seconds
   (IO + decode + compute), with today's behavior recovered as the limit "bandwidth → 0"
   (bytes are infinitely expensive). This keeps one comparable scalar — which the selection
   machinery (strict ordering, early-exit thresholds) fundamentally relies on — while making
   the ratio-vs-speed tradeoff a *parameter* instead of a philosophy.
2. **The unknowables become inputs.** Workload and environment can't be inferred from the
   array bytes; they arrive as a profile (with a sane default) or the model silently reverts
   to guessing. This is the strongest argument for pluggability, and it matches precedent
   (Nimble made selection a pluggable policy; DuckDB hardcodes one opinionated mix; see
   doc 2).
3. **Expectations calibrated to "avoid pathologies + enable presets", not "oracle".** The
   compressor will not predict query times. It can reliably rank candidates under an explicit
   objective, kill the step-function mistakes, and let a `scan_speed` / `point_lookup` /
   `gpu` / `max_ratio` preset mean something principled instead of a hand-tuned scheme list.

The concrete design that follows from this analysis is in
[doc 4](./04-design-pluggable-cost-model.md); the staged plan (measurement first) is in
[doc 5](./05-roadmap-and-open-questions.md).
