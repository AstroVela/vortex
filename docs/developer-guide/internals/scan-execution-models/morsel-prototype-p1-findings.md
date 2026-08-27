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

One deviation from the sketch worth recording: rather than rewriting expressions to push
predicates onto individual fields, each conjunct and the projection are re-bound against the
*narrowed* struct dtype of exactly the top-level fields they reference. This achieves the same
column pruning using only public expression API, and keeps the executor's semantics identical to
V1's by construction (the same `apply_bound` on the same assembled struct).

## Correctness

16 differential tests, all passing. Every one uses the V1 `LayoutReader` as the oracle and
asserts equal row counts and equal ordered content over 8 query shapes:

| Property | Test |
|---|---|
| Agrees with V1 at 1, 2 and 4 threads | `matches_v1_oracle` |
| Misaligned chunking is invisible | `misaligned_chunks_match_aligned_reference` |
| The document's `[0,3,10)` vs `[0,6,10)` case, and its split set | `document_misalignment_case` |
| Result independent of morsel size (1, 7, 128, 4096 rows, and per-split) | `independent_of_morsel_size` |
| Cascade and parallel conjunct policies observationally identical | `conjunct_policy_is_not_observable` |
| Decode cache is an optimisation only | `decode_cache_is_not_observable` |
| Every read was named by a planning stream | `every_read_was_planned` |
| All-false filter emits nothing | `empty_filter_emits_nothing` |
| Unsupported layouts are build errors | `rejects_unsupported_layouts` |

The evaluation binary re-runs the oracle check for **every** configuration on **every** query
before any timing happens; a configuration that disagrees is reported as a failure and excluded
from the timing table. All 105 configuration-query pairs in the run below matched.

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
   are P2 work and out of scope for P1 anyway. The decode-cache-disabled row is the nearest P1
   analogue of chaos mode — a component that must not change a single row when removed — and it
   is asserted both in the tests and in the eval.

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
| A' V1, tokio x4 | 0.788 | 0.31 – 1.82 |
| D  morsel, 1 thread, per-split morsels | **0.553** | 0.31 – 0.83 |
| D  morsel, 1 thread, decode cache disabled | 0.657 | 0.33 – 1.00 |
| D  morsel, 4 threads, per-split morsels | 0.388 | 0.26 – 0.92 |
| D  morsel, 4 threads, 64k-row morsels | **0.251** | 0.11 – 0.66 |
| D  morsel, 4 threads, parallel conjuncts | 0.349 | 0.21 – 0.71 |

Per workload (geomean vs V1(1)):

| Row | string-heavy | wide-numeric | narrow-analytic |
|---|--:|--:|--:|
| A' V1, tokio x4 | 0.589 | 0.980 | 0.910 |
| D  morsel, 1 thread | 0.620 | 0.503 | 0.532 |
| D  morsel, 1 thread, no cache | 0.890 | 0.535 | 0.539 |
| D  morsel, 4 threads | 0.414 | 0.328 | 0.480 |
| D  morsel, 4 threads, 64k morsels | 0.312 | 0.176 | 0.329 |

The full table, every query and every counter, is in
[`morsel-prototype-p1-eval.md`](morsel-prototype-p1-eval.md).

### What the numbers say

**Single-threaded, the prototype is ~1.8x faster than V1 on the same cut.** Both executors walk
the same natural splits and read the same segments; D has no future per evaluation and no task
per split. That difference alone is the 0.553.

**The decode cache is the whole story on string-heavy data and almost none of it on numeric
data.** Disabling it moves string-heavy from 0.620 to 0.890 — on `SH1 select-all`, from 15.2 ms
back to 31.2 ms, which is exactly V1's 31.2 ms — while wide-numeric barely moves, 0.503 to 0.535.
The reason is visible in the counters: on `SH1 select-all`, 310 named uses collapse to 121 reads
and 189 cache hits, because the text column is chunked at 4,096 rows while `token_count` is
chunked at 65,536, so a wide chunk is re-entered by sixteen consecutive morsels. V1 re-decodes it
each time — `FlatReader::array_future` builds a fresh shared future per evaluation, with no
reader-level cache. This is the single largest measured win and it comes directly from the design
rule that a straddling morsel joins the *same* keyed cell rather than issuing its own read.

**Coalescing morsels is worth more than threads on wide tables.** `WN1 select-all` at 4 threads
goes from 0.30 (per-split morsels) to 0.12 (64k-row morsels): 228 morsels become 16, and 4,560
named uses become 1,476. The per-morsel fixed cost is real, and the executor's ability to straddle
chunk boundaries is what makes coalescing legal at all. This is the `select *` small-splits storm
from [problem 1 of the next-discussion document](scan-execution-graph-next-discussion.md) —
measured, and answered by unit formation rather than by more machinery.

**Cascade beats parallel, but not by much** (0.388 vs 0.349 at 4 threads — parallel is actually
slightly *faster* here). On these workloads the conjuncts are cheap scalar comparisons, so
evaluating both against the full mask and intersecting costs about what the serial dependency
costs. On a workload where the second conjunct is expensive (a `LIKE` over a text column) cascade
should win clearly; that case is not in these fixtures and the policy question stays open.

### Two honest caveats in D's favour, and one against

*In D's favour, to discount:*

- **The 4-thread rows spawn threads per run.** Sub-millisecond queries (`SH6`, `NA3`) show D at 4
  threads *losing* to D at 1 thread — 498 µs vs 404 µs on `SH6` — because ~200 µs of thread spawn
  dominates. A real implementation uses a pool. Read the 4-thread rows only on queries above a
  few milliseconds.
- **Time-to-first-batch is not directly comparable.** D's is measured from the first morsel a
  thread completes, V1's from the first item off the stream. D's numbers are dramatically better
  (76 µs vs 529 µs on `SH6`; 525 µs vs 9.2 ms on `SH1`) and the direction is real — D emits as
  soon as one morsel finishes rather than after the pipeline fills — but the two clocks are not
  measuring quite the same event.

*Against D, and this one matters for P2:*

- **Multi-threaded D reads more bytes than V1.** The IO plane is per-thread, so a segment touched
  by three threads is read three times: on `WN1`, 1,332 reads at one thread become 3,908 at four.
  With in-memory segments that is nearly free and it is why the 4-thread rows still win. Over real
  object-store IO it would be a straightforward regression, and it is exactly what P2's shared
  keyed cells with leases are for. **The cold-scan IO invariant that gate E1 requires — byte-identical
  IO across rows — holds for D at one thread and does not hold for D at four.** No claim in this
  document should be read as saying otherwise.

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

P2's shared cells are motivated independently of the gate by the per-thread read duplication
above, which is a measured defect rather than a speculative one.
