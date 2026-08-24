# From 2.5x Slower to 2.5x Faster: The Self-Paced Executor, Step by Step

This is a tutorial-style walkthrough of one day of experiments on the restricted self-paced scan
executor, written to be read top to bottom. Each section introduces one idea, the measurement
that motivated it, and how the result changed the design. The
[findings report](self-paced-plan-exec-findings.md) holds the full tables; the
[handover](self-paced-plan-exec-handover.md) holds the current code map.

All numbers are `self-paced / V1` wall-time ratios on a 16-core, 30 GB host, pinned with
`taskset -c 0-15`, under the fair natural-split contract (same serialized bytes, same query
object, validated row counts and ordered output hashes, alternating iterations, medians).
Headline workload: FineWeb Q06, a three-conjunct selective scan over 14.9M rows.

## 0. Starting point: reproduce before touching anything

The handover left the executor at **2.19x slower** than V1 on Q06 (2.53x on this host). Before
changing code we reproduced the whole environment: the 15 FineWeb parquet shards, and — because
the required "physical split catalog" generator was never committed — a new audit tool
(`vortex-file/examples/fineweb_split_audit.rs`) that writes each dataset with the default Vortex
writer and records every field's natural chunk boundaries. The regenerated catalogs matched the
documented counts exactly (FineWeb 1,823/2,527 splits, TPC-H 458), which told us the environment
was faithful and the old numbers were real.

**Idea introduced:** never optimize against an unreproduced baseline.

## 1. Measure the coordinator before believing in it

The docs *claimed* the single coordinator thread was the bottleneck. We added phase timing
(`VORTEX_SELF_PACED_PHASE_TIMING=1`) attributing coordinator wall time to drain / advance /
schedule / dispatch / inline / wait, plus a queue-dwell timestamp on every worker completion.

Result: the coordinator was **89% busy** (advance 34%, completion handling 28%, dispatch 24%)
while finished worker results waited ~17us each to be adopted. The workers were starving behind
the coordinator, not the reverse.

**Idea introduced:** turn a hypothesis into a phase budget before acting on it.

## 2. Work reduction does not fix a serialized critical path

Guided by code audits, we applied the obvious micro-optimizations: allocation-free mask
adoption (a fused and-count), batching resource joins and fragment transitions, skipping a
useless scheduler pass, an all-true selection early exit. All correct, all measurable in
counters — and Q06 moved only **2.53x -> 2.32x**.

**Idea introduced:** shrinking the work on a serialized path barely moves wall time; you have to
parallelize or delete the path.

## 3. Shard the coordinator (2.32x -> 1.40x)

`VORTEX_SELF_PACED_SHARDS=N` splits the morsel list into N contiguous groups, each with its own
`Execution` and coordinator thread, sharing one worker pool. Because morsel boundaries align
with natural splits, no segment straddles shards: the sharded run read byte-identical I/O.
Four shards was the sweet spot (2 -> 1.64, 4 -> **1.40**, 8 -> 1.44).

**Idea introduced:** most coordination state is morsel-local; partitioning it is nearly free.

## 4. Delete the coordinator entirely: owned mode (1.40x -> 0.79x)

If sharded coordinators work, why have a coordinator/worker split at all? "Owned" mode runs 16
threads, each *both* coordinating and evaluating its own morsel group inline — no pool, no
completion channel, no dispatch, no queue dwell, and the thread count now exactly matches V1's.
Q06 flipped to a **win (0.79)** and 25 of 28 workloads won.

**Idea introduced:** morsel-driven self-coordination; cross-thread communication was the cost,
not the coordination logic itself.

## 5. Make "no caching" an enforced invariant, not a claim

Every timed iteration now must re-read at least its cold warmup's unique-segment byte floor
(`assert_cold_scan_io`), with byte-exactness required of self-paced under deterministic demand
policies, and per-iteration row counts checked against the warmup. This immediately caught a
real V1 behavior (dropped duplicate in-flight reads under-counting bytes ~0.01%) and later had
to be scoped when adaptive ordering legitimately changed which chunks are read.

Separately, 17 workloads were validated against **DuckDB over the original parquet** —
replicating every fixture derivation in SQL — and all 17 output row counts matched exactly.

**Idea introduced:** anti-cheat and correctness checks should run on every measurement, and at
least once against an oracle that shares no code with the thing being tested.

## 6. The pipeline executor: extensibility and speed from the same design (0.79x -> 0.41x)

Owned mode still carried the reactor's slot/offer/claim machinery. The pipeline rebuild kept
only two seams:

- **`MorselPipeline`** — the scheduler's entire knowledge of execution: morsel range in,
  `ExecBatch` out. Threads self-schedule morsels off one shared atomic cursor (work stealing,
  order restored by index). New nodes never touch the scheduler.
- **`DemandPolicy`** — the shared per-morsel demand mask that gates every struct child is
  computed by a pluggable policy (`cascade`, `eager`, and later `adaptive`).

Row-domain relationships became **executor-vtable methods**: a *down demand transform*
(`FieldDomain::push_demand`: cut a parent-range demand into priced child segments, so empty
children are never read) and *up result transforms* (`pull_mask`, `pull_array`). Each
relationship is modeled on the layout's native metadata — chunked concatenation on the
chunk-offset prefix sums, the struct identity as one refcounted demand handle shared by all
children, a future list node on its offsets buffer. Children with arbitrary, mutually unaligned
chunk boundaries just work (unit-tested).

Q06 hit **0.41 (11.4 ms)**. The attribution cornerstone is the wide select-all shape (Q09):
byte-identical physical I/O on both engines, nothing avoidable, and the pipeline is still ~2.5x
faster — the remaining advantage is scheduling-unit cost (tens of self-scheduled morsels versus
thousands of per-split futures) plus inline execution.

**Idea introduced:** the vtable seam costs ~0 because dispatch happens per chunk, never per row.

## 7. Planning does no compute

An eager "plan-time compilation" of the segment cutting was built — and measured *slower*
(0.34 -> 0.39 geomean): it serialized, onto one pre-thread path, arithmetic the threads do for
~100ns/segment in parallel, and per-scan planning amortizes nothing. It was removed. The rule
that survived, now in the module docs: planning wires topology and shares refcounted demand
handles; splits are computed once; all compute happens at execution on the owning threads.

**Idea introduced:** negative results get measured, documented, and deleted — not kept as
opt-in complexity.

## 8. Feed the wins back into V1

The I/O audit (per-request order/size dumps) showed V1 re-reading shared segments — up to
**2.7x the file size** on shared filter/projection scans — because `FlatReader::array_future`
rebuilt its "shared" future on every call. The pipeline's dedup idea, translated with V1's
memory discipline (a `WeakShared` memo: shared while any evaluation is live, freed after),
fixed it: 916 -> 307 requests on the shared-field scan, flat memory, unit-tested for both
dedup and release. Committed separately off `develop`
(branch `worktree-v1-flat-reader-dedup`) and also applied here so every comparison is against
honest V1. Even against fixed V1, all 42 workloads still won.

**Idea introduced:** experiment learnings should flow back to production, with production's
constraints (no scan-lifetime retention).

## 9. Widen the evidence: more shapes, more datasets

Coverage grew to 52+ workloads: FineWeb Q09-Q17 (wide select-all, shared field, deep chains,
empty result, ranges, project-all-fields), ClickBench Q43-Q51, and a new **statpopgen** suite
(gnomAD chr21 VCF via vortex-bench's data-gen; genomic region scans, quality thresholds,
population columns). Each dataset found something:

- ClickBench's few-morsel shapes exposed thread-tail imbalance -> fixed by work stealing
  (dashboard 1.06 -> 0.82, Q41 1.31 -> 0.64).
- Deep chains exposed a kernel defect: dense-but-partial demand used a per-row demand-checking
  `map_cmp`; two vectorized passes (full evaluate, then AND) are faster (Q45 1.06 -> 0.95).
- statpopgen exposed the fixed merge-16 roll-up: 1M compact rows -> 8 splits -> one morsel ->
  concurrency 1. The harness now targets ~2x the worker count (`clamp(splits/32, 1, 16)`).

**Idea introduced:** every new dataset class stresses a different assumption; add shapes until
new ones stop finding anything.

## 10. Adaptive demand: the policy seam pays off

Query-order conjunct evaluation read a wide column early on dashboard-style queries.
`AdaptiveDemand` (now default) orders conjuncts by observed survival, most selective first —
output-identical, validated by the hash gate on every suite — and later gained a
density switch: when current demand is >= 50% dense, gating costs more than it avoids
(measured: eager 1.31 vs cascade 2.38 on the dense statpopgen shape), so the conjunct is
evaluated in full and intersected.

**Idea introduced:** the pluggable-demand seam exists precisely so scheduling policy can evolve
without touching nodes or scheduler.

## 11. The tiny-scan regime and measurement discipline

Sub-millisecond scans (statpopgen) needed two corrections. Five-iteration medians swing
0.8-2.5x at this scale — 100-iteration medians are now standard there. And per-morsel constants
matter: per-run thread spawns were replaced with a reused pool, full-demand evaluation stopped
allocating an all-true mask, single-segment morsels return masks zero-copy, coverage checks use
already-priced counts, and the selection `Mask` is built once per morsel and shared by all
projected fields. Result: five of six statpopgen shapes at or better than parity.

**Idea introduced:** fixed cost per morsel is a distinct budget from cost per row; tiny scans
are where it shows.

## Where it stands

| Suite | Record vs (fixed) V1 | Geomean |
| --- | --- | ---: |
| FineWeb (18 shapes) | 18/18 wins | ~0.33 |
| TPC-H SF10 (3) | 3/3 wins | ~0.63 |
| ClickBench (25 shapes) | 25/25 wins | ~0.56 |
| statpopgen (6, sub-ms) | 3 wins, 2 ties, 1 open | — |

Q06 arc: 2.53 -> 2.32 (micro-opts) -> 1.40 (4 shards) -> 0.79 (owned) -> **0.41 (pipeline)**.

## How to refine from here

1. **The Q02 anomaly (open bug):** the dense statpopgen shape runs 1.31 under the eager policy
   but 2.62 under the logically equivalent in-policy dense switch. That delta should not exist.
   A samply profile is captured; this is the first thing to chase.
2. **TPC-H Q6 makespan:** 29 two-million-row morsels bound wall time at two serial morsels per
   thread. Intra-morsel parallelism, or a work-aware (byte/CPU) roll-up instead of split-count
   merging, is the fix.
3. **Real I/O:** everything here is an in-memory source. The recorded agenda: a ranged/multi-get
   `SegmentSource` for run coalescing (up to 86x fewer requests on wide scans), per-thread async
   read-ahead, writer-side chunk sizing for 50KB-segment datasets like FineWeb.
4. **Generalize the layout:** lift the aligned-chunks restriction for real files by adding the
   per-field root-range translation as `FieldDomain` impls (the misaligned-children test already
   proves the seam); then a list node over its offsets buffer.
5. **Ship the V1 fix:** `worktree-v1-flat-reader-dedup` is PR-ready against `develop` and
   independent of everything else.
6. **Validation depth:** extend the DuckDB oracle to value-level checksums and the remaining
   workloads; consider an opt-in per-iteration hash mode.
