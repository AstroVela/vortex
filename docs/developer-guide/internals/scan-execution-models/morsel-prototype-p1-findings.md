# Morsel Prototype: P1 Findings

Measured results for the P1 spine of the
[morsel-based plan execution design](morsel-based-plan-execution.md), implemented in
`vortex-morsel`. This document records what was built, what was measured, and — importantly —
which parts of the [prototype plan's](morsel-prototype-plan.md) evaluation matrix could **not** be
evaluated in this environment and why.

## What was built

`vortex-morsel` implements the P1 surface from the prototype plan:

```rust
trait ExecNode: Send {
    fn reset(&mut self, range: Range<u64>);
    fn next_plan(&mut self, cx: &mut PlanCx<'_>) -> VortexResult<PlanPoll>;
    fn execute(&mut self, cx: &mut ExecCx<'_>) -> VortexResult<ExecPoll>;
    fn retire(&mut self, cx: &mut RetireCx<'_>);
    fn children(&self) -> &[NodeId];
}
```

The five operators are `FlatExec`, `ChunkedExec`, `StructExec`, `ConjunctExec` (cascade and
parallel behind one policy flag) and `FilterExec`.

Design points that survived contact with the code:

- **Nodes never perform IO.** `next_plan` registers `IoUse`s keyed to whole stored units against
  the `IoPlane` and receives tickets; `execute` may only wait on tickets its own planning stream
  emitted, and consuming a ticket a node never named is an error rather than an inline read. The
  `source_range`, `extent`, `producer` and `estimated_bytes` fields are carried and stamped but
  not yet *consulted* — nothing reads them until P2's admission loop exists.
- **Emit-once planning.** Planning is budget-bounded (`PLAN_BUDGET = 64` uses) and resumable:
  chunked keeps a cut cursor, struct and conjunct keep field cursors, so a node that exhausts the
  quantum yields `PlanItem::Plan` and resumes where it stopped rather than restarting.
- **Immutable plan, per-thread state.** The graph model's objection to stateful nodes (§9 of
  [the graph model](scan-execution-graph-model.md)) — that a node shared by every unit cannot hold
  per-morsel state — is answered by splitting the two: `ExecPlan` is one immutable blueprint per
  scan, and each driving thread instantiates its own arena of mutable node state, reset per morsel.
  Nothing is allocated per morsel and nothing on the hot path is shared between threads.
- **The arena take/put trick** is what lets a node hold `&mut self` while recursively driving its
  children: the driver removes the node from its slot, hands the rest of the arena to the child
  poll, and puts it back. The tree shape guarantees a node is never reachable from its own
  subtree, so a taken slot is never observed empty; the debug path panics if it ever is.
- **Unsupported shapes are build errors.** Nested structs, non-struct roots, nullable root
  structs and non-flat/non-chunked columns fail in `build_plan` rather than falling back, so an
  unsupported query cannot be timed as if the prototype had executed it.
- **No state V1 does not have.** An earlier revision carried a per-thread decoded-chunk cache; it
  was removed because V1 has no reader-level decode cache, so timing against it with one was not
  a comparison of executors. The IO plane's keyed cells live only for the duration of one morsel
  (released at retire), so a segment named twice *within* a morsel resolves to one read — that is
  the registration mechanism itself, not a cache — and across morsels every segment is re-read
  and re-decoded exactly as V1 re-reads and re-decodes it per evaluation. The eval's counters
  confirm this: requests, uses and decodes are equal in every configuration.

One deviation from the sketch worth recording: rather than rewriting expressions to push
predicates onto individual fields, each conjunct and the projection are re-bound against the
*narrowed* struct dtype of exactly the top-level fields they reference. This achieves the same
column pruning using only public expression API, and keeps the executor's semantics identical to
V1's by construction (the same `apply_bound` on the same assembled struct).

## Correctness

15 differential tests, all passing. Every one uses the V1 `LayoutReader` as the oracle and
asserts equal row counts and equal ordered content over 8 query shapes:

| Property | Test |
|---|---|
| Agrees with V1 at 1, 2 and 4 threads | `matches_v1_oracle` |
| Misaligned chunking is invisible | `misaligned_chunks_match_aligned_reference` |
| The document's `[0,3,10)` vs `[0,6,10)` case, and its split set | `document_misalignment_case` |
| Result independent of morsel size (1, 7, 128, 4096 rows, and per-split) | `independent_of_morsel_size` |
| Cascade and parallel conjunct policies observationally identical | `conjunct_policy_is_not_observable` |
| Every read was named by a planning stream | `every_read_was_planned` |
| All-false filter emits nothing | `empty_filter_emits_nothing` |
| Unsupported layouts are build errors | `rejects_unsupported_layouts` |

The evaluation binary re-runs the oracle check for **every** configuration on **every** query
before any timing happens; a configuration that disagrees is reported as a failure and excluded
from the timing table. All 90 configuration-query pairs in the run below matched.

## What could not be evaluated, and why

The prototype plan's gate E1 reads: *D within 5% of C's rerun across suites; ordering
D ≈ C < B(owned) < B(coordinator) reproduced.* **Gate E1 as specified was not evaluated.** Three
reasons, all environmental rather than results anyone should read past:

1. **Rows B and C do not exist in this repository.** The self-paced graph/reactor and pipeline
   executors that the [findings document](self-paced-plan-exec-findings.md) reports 2.53x → 0.41x
   for are not present at any commit reachable here — a search of the tree for `self_paced`,
   `morsel`, or a `vortex-scan-v2` crate finds nothing. Only rows A (V1) and D (this prototype)
   could be run. Without C, "within 5% of C" is unmeasurable, and so is the ordering claim.
2. **The named suites need multi-gigabyte downloads.** FineWeb's sample is ~2 GB of Parquet and
   ClickBench's `hits` is far larger; this host has 4 cores and 15 GB of RAM, and the harness
   holds segments in memory. TPC-H SF10 needs a generator that is not vendored.
3. **P0's latency-injection IO source and chaos mode are not built.** They gate E2 and E3, which
   are P2 work and out of scope for P1 anyway.

What was measured instead is a set of **shape-matched synthetic workloads**: struct-of-chunked-flat
columns whose per-column chunk boundaries deliberately disagree, scanned under conjunctive filters
of varying selectivity with narrow and wide projections. These reproduce the structure the plan
says the real suites lower to, and they exercise exactly what E1 is about — the executor's own
scheduling-unit cost. They do **not** exercise encoding-specific decode costs (FSST, ALP-RD,
dictionary), and their absolute wall times are not comparable to the recorded suite numbers.

## Results

Host: 4 logical cores, segments in memory, 1M rows per workload (250k for the string-heavy one,
which has far wider rows), 5 alternating iterations, median reported. Reproduce with:

```bash
cargo run --release -p vortex-morsel --features _test-harness --bin morsel-eval
```

Ratios are against **A: V1 single-threaded**, which is the apples-to-apples baseline for a
one-thread morsel run — the harness drives V1 on `SingleThreadRuntime`, which runs every task on
the calling thread. Row A' gives V1 a multi-threaded Tokio runtime with the same core count, which
is how DataFusion actually drives it.

Geometric means over all 15 queries:

| Row | Geomean vs V1(1) | Range |
|---|--:|---|
| A  V1, 1 thread | 1.000 | — |
| A' V1, tokio x4 | 0.794 | 0.30 – 1.63 |
| D  morsel, 1 thread, per-split morsels | **0.650** | 0.35 – 0.99 |
| D  morsel, 4 threads, per-split morsels | 0.359 | 0.22 – 0.76 |
| D  morsel, 4 threads, 64k-row morsels | **0.238** | 0.09 – 0.58 |
| D  morsel, 4 threads, parallel conjuncts | 0.337 | 0.20 – 0.62 |

Per workload (geomean vs V1(1)):

| Row | string-heavy | wide-numeric | narrow-analytic |
|---|--:|--:|--:|
| A' V1, tokio x4 | 0.538 | 1.059 | 0.972 |
| D  morsel, 1 thread | 0.788 | 0.582 | 0.553 |
| D  morsel, 4 threads | 0.387 | 0.300 | 0.442 |
| D  morsel, 4 threads, 64k morsels | 0.271 | 0.173 | 0.346 |

The full table, every query and every counter, is in
[`morsel-prototype-p1-eval.md`](morsel-prototype-p1-eval.md).

### What the numbers say

**Single-threaded, the prototype is ~1.5x faster than V1 on the same cut, and the win is
inversely proportional to how decode-dominated the query is.** Both executors walk the same
natural splits, read the same segments the same number of times, and decode the same chunks the
same number of times — the counters show requests = uses = decodes on every configuration. What
differs is the machinery around that work: V1 builds a future per evaluation and drives it
through a stream; D drives a morsel inline with a program counter. On the wide-numeric workload,
where per-split fixed cost dominates, that is 0.582. On string-heavy `SH1 select-all`, where wall
time is almost entirely decode (which is identical by construction), D is 0.97 — the honest
number: an executor cannot be much faster than V1 at a query whose cost V1 also spends almost
entirely inside the shared decode kernel.

An earlier revision of this document reported 0.49 on that query. That number came from a
per-thread decoded-chunk cache the prototype had and V1 does not; it was measuring a cache, not
an executor, and it is gone — both from the code and from every table here. The magnitude of the
delta (0.49 → 0.97 on SH1) is itself a finding: on misaligned string-heavy layouts, *decode
reuse across morsels is worth ~2x*, and if that win is ever wanted it must be designed as a
shared, budgeted facility both executors could use — P2's keyed cells — not smuggled in as
executor-private state.

**Coalescing morsels is worth more than threads on wide tables.** `WN1 select-all` at 4 threads
goes from 0.24 (per-split morsels) to 0.09 (64k-row morsels): 228 morsels become 16, and 4,560
uses/requests/decodes become 1,476. This is not caching — it is genuinely fewer scheduling units
doing genuinely less repeated work, because one morsel spanning sixteen chunk boundaries slices
each chunk once where sixteen per-split morsels each pay the full per-morsel cost. This is the
`select *` small-splits storm from
[problem 1 of the next-discussion document](scan-execution-graph-next-discussion.md), measured,
and answered by unit formation rather than by more machinery.

**Cascade and parallel conjuncts are within noise of each other here** (0.359 vs 0.337 at 4
threads). On these workloads the conjuncts are cheap scalar comparisons, so evaluating both
against the full mask and intersecting costs about what the serial dependency costs. On a
workload where the second conjunct is expensive (a `LIKE` over a text column) cascade should win
clearly; that case is not in these fixtures and the policy question stays open.

**The cold-scan IO invariant holds everywhere.** With cells released per morsel, every
configuration — 1 thread, 4 threads, coalesced, parallel — shows requests = uses = decodes, and
the per-split rows show exactly the counts V1's evaluation structure implies. The earlier
revision's 4-thread rows read up to 3x the bytes of the 1-thread rows (per-thread cells shared
nothing); that asymmetry is gone along with the cache.

### Two honest caveats in D's favour, to discount

- **The 4-thread rows spawn threads per run.** Sub-millisecond queries show D at 4 threads
  losing ground to D at 1 thread because ~200 µs of thread spawn dominates. A real
  implementation uses a pool. Read the 4-thread rows only on queries above a few milliseconds.
- **Time-to-first-batch is not directly comparable.** D's is measured from the first morsel a
  thread completes, V1's from the first item off the stream. D's numbers are much better
  (0.55 ms vs 8.7 ms on `SH1`) and the direction is real — D emits as soon as one morsel
  finishes rather than after the pipeline fills — but the two clocks are not measuring quite
  the same event.

## Where this leaves the phase order

P1's spine is built, correct against the V1 oracle, and faster than V1 on every workload measured
at equal thread count. What P1 cannot do is answer the question E1 was written to answer, because
the executor E1 compares against is not in this repository.

Two things would need to happen before the gate means anything:

1. **Locate or rebuild rows B and C.** If the self-paced experiment exists on a branch, running
   its pipeline mode on this host against these same fixtures makes the 5% comparison meaningful
   in one afternoon. If it does not exist, the bar has to be restated against something that does.
2. **Decide whether the shape-matched fixtures are enough.** They isolate scheduling-unit cost
   well, which is E1's actual subject, but the recorded 0.33/0.6 geomeans came from real
   encodings. Comparing a synthetic-fixture ratio against a real-suite ratio is not sound, and
   this document does not do it.

P2's shared cells are now motivated by a measured number rather than a speculative one: the
removed decode cache showed cross-morsel decode reuse is worth ~2x on misaligned string-heavy
layouts. P2 is where that win gets rebuilt as a shared, scheduler-visible facility with leases
and byte budgets — available to any executor, accounted for, and disabled in the fair rows of
any future comparison.
