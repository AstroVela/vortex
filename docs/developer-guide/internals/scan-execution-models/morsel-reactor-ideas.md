# Morsel Reactor Ideas and Decisions to Validate

## Status

This note is intentionally non-normative. It records design ideas raised while refining the
[morsel reactor architecture](morsel-reactor.md), along with the evidence needed to choose among
them. Correctness contracts belong in the architecture document; policies in this note may change
after simulation and benchmarking.

## 1. Fixed versus migratable morsel ownership

The baseline is one mutable owner at a time. Two policies are possible:

### Fixed owner

A morsel remains assigned to one worker until completion. The worker executes other stealable work
while the morsel waits.

Advantages:

- strongest cache locality;
- no reactor transfer protocol; and
- simple thread-local arenas.

Risks:

- an owner with several completions may become a planning bottleneck; and
- an imbalanced set of complex morsels may leave planning uneven even when task execution balances.

### Migratable owner

A parked reactor can be moved as one object to another worker. No task may hold a borrow into it.

Advantages:

- planning load can balance independently from task execution; and
- idle workers can adopt completion-heavy morsels.

Risks:

- weaker cache locality;
- more ownership-state transitions; and
- harder integration with thread-local allocation.

Start fixed. Add migration only if planning time becomes visible in profiles.

## 2. Planning replenishment policy

Owners should plan enough work to keep shared executors busy without expanding the complete morsel
or retaining excessive speculative state.

Candidate triggers include:

- local ready-work count below a watermark;
- global CPU deque depth below a watermark;
- I/O queue below its target concurrency;
- completion of a gate near the commit frontier;
- sealed frontier advancement; and
- an explicit steal request from an idle worker.

Possible policy:

```text
if required work is ready:
    advance until required queue reaches R
else if global queues are starved:
    expand candidate work until total queue reaches C
else:
    stop at local quiescence or the near planning horizon
```

Measure planning calls, transitions per call, queue starvation time, and speculative bytes.

## 3. Complete frontier versus incremental deltas

The semantic contract says a locally quiescent reactor has exposed every currently concrete
opportunity. The transport to the scheduler could be:

- a complete snapshot of outstanding work;
- only `Offer`, `Rescore`, `Promote`, and `Eliminate` deltas; or
- deltas normally, with a snapshot recovery path.

Deltas avoid repeatedly copying large work sets. Stable IDs and a snapshot recovery path make them
robust to scheduler reconstruction and debugging.

## 4. Predicate scheduling

Every remaining conjunction can expose its reads and, once inputs are ready, its CPU opportunity.
The scheduler chooses among three policies per block.

### Sequential

Run the best estimated predicate, refine demand, then offer later CPU against the smaller snapshot.
This minimizes work but may underuse workers and serialize high-latency reads.

### Concurrent

Run several predicates over the same immutable open snapshot. Their result masks commute under
intersection when expressions are deterministic and infallible. This lowers latency at the cost of
evaluating rows that another predicate may remove.

### Hybrid

Prefetch several predicate inputs, run the best predicate first, and admit later CPU only when
queues would otherwise drain. This is the initial policy to prototype.

A useful score may approximate:

```text
expected downstream cost removed / (predicate I/O cost + predicate CPU cost)
```

The score must also include distance from the commit frontier, cached inputs, uncertainty, and
global queue state.

## 5. Selectivity uncertainty

The exact candidate count is known for each open snapshot. Future survival is not. Possible
estimators include:

- historical selectivity per predicate;
- current-scan global selectivity;
- per-file or per-zone statistics;
- a conservative interval rather than one expected value; and
- no estimate until a predicate has executed.

Expected selectivity is scheduling evidence only. Zero demand requires exact proof. Begin with
exact candidate counts and a simple global historical rate, then measure whether more local models
change admission decisions.

## 6. Speculative projection I/O

Projection planning sees open demand. The scheduler must decide how much candidate I/O to admit.

Factors favoring early reads:

- remaining filters are expected to be non-selective;
- storage latency is high;
- compressed bytes are small;
- I/O capacity would otherwise idle;
- the same `ReadKey` is needed by a predicate; or
- the read is near the ordered commit frontier.

Factors favoring delay:

- remaining filters are expected to be highly selective;
- projection fields are wide;
- compressed or cache budgets are tight;
- cancellation or a limit is likely; or
- reads are far beyond the commit frontier.

Prototype at least conservative, balanced, and latency-biased policies against local NVMe and
object storage.

## 7. Speculative CPU for read discovery

Some projection reads cannot be named until CPU work resolves a gate:

- dictionary codes reveal value pages;
- list offsets reveal element ranges; and
- indexes reveal independently readable encoded pages.

Running safe discovery CPU under open demand may hide substantial I/O latency. Candidate classes:

```text
always safe:       metadata parsing, infallible offset/code decode
conditionally safe: deterministic decode with bounded retained output
sealed only:       fallible expressions and demand-sensitive value construction
```

The prototype should record speculative discovery CPU, bytes unlocked, eventual reuse, and waste.

## 8. Task fusion

Semantic nodes need not equal scheduler tasks. Candidate fusion cases include:

- several predicates over one decoded column;
- metadata checks over one footer;
- decode followed by a very small expression; and
- several small adjacent reads supported efficiently by the I/O backend.

Avoid fusion when it hides:

- a selective boundary;
- independent stealing parallelism;
- different priorities or credit classes;
- useful cancellation points; or
- fallible error order.

Measure task launch cost and establish a minimum useful CPU duration before adding adaptive fusion.

## 9. Demand update transport

Both predicate and projection planning observe open demand, but waking every projection node after
every mask revision would be expensive. Options include:

- direct subscription for nodes at the commit frontier;
- lazy generation-based rescoring for catalog entries;
- block-summary notifications for distant candidate work; and
- batching several predicate completions into one generation update.

The likely split is immediate notification for correctness-critical promotion or exact emptiness,
and lazy rescoring for speculative priority changes.

## 10. Dynamic-filter sealing semantics

An external dynamic filter may continue shrinking after local predicates finish. Possible
contracts are:

- wait for dynamic-filter completion before sealing affected blocks;
- freeze one dynamic-filter generation per block;
- apply new generations only to blocks not yet started; or
- restart an uncommitted suffix under a new epoch.

Freezing a generation per block is simple and pipeline-friendly but may miss late pruning. Waiting
maximizes filtering but can stall output. This decision requires integration-specific semantics and
latency measurements.

## 11. Work queues and stealing

Potential runtime organization:

```text
per-worker deque:
  CPU tasks, local-first and stealable

shared I/O queues:
  required reads
  candidate reads

per-morsel mailbox:
  completion facts

per-worker reactor queue:
  morsels needing advance
```

An alternative puts reactor continuations in the same work-stealing deque as CPU tasks. That makes
planning stealable but weakens fixed ownership. Start with a distinct owner-local reactor queue so
planning and expensive computation remain observable separately.

## 12. Graph representation

The initial graph should use compact IDs, arenas or slot maps, `SmallVec` subscribers, and queued
bits. Alternatives include:

- boxed trait-object nodes;
- an enum of built-in node states;
- a flat arena with vtable dispatch; and
- compile-time specialized graphs for common plans.

The graph-cost estimate in [scheduler-visible work](scheduler-visible-work.md) suggests bookkeeping
will be smaller than masks and data buffers. Do not optimize representation until measurements show
dispatch, allocation, or cache misses are material.

## 13. Fact retention

Completed task results may feed several consumers. Options include:

- explicit subscriber reference counts;
- frontier-based release;
- node-owned handles with scheduler-visible retained bytes; and
- a per-scan cache for reusable dictionary or metadata facts.

The releasing component must be the component charged for retention. Record high-water marks and
late-release causes in the deterministic simulator.

## 14. Planning budget

Possible transition budgets include:

- a fixed number of node transitions;
- elapsed coordination time;
- number of work offers produced;
- bytes of new candidate work exposed; or
- a combination with a hard time ceiling.

A fixed transition count is deterministic and testable. A time ceiling protects fairness in
production. Start with a transition count and collect elapsed-time metrics.

## 15. Prototype scenarios

The deterministic simulator should cover:

1. Three conjunctions scheduled sequentially, concurrently, and in hybrid mode.
2. Projection reads admitted before, during, and after filtering.
3. A running predicate completing against an older demand snapshot.
4. A block becoming empty while candidate projection I/O is in flight.
5. Dictionary codes unlocking several value-page reads.
6. A large list offset unlocking an oversized element range.
7. Blocks sealing out of order while ordered output waits on an earlier block.
8. One owner publishing CPU work stolen by several workers.
9. Planning budget exhaustion followed by immediate re-advance.
10. Dynamic-filter generations arriving before and after local predicate completion.
11. Shared filter and projection reads deduplicating to one physical request.
12. Cancellation and late task completion releasing all retained state.

For every scenario record:

- output and error equivalence to the current executor;
- work offers and lifecycle updates;
- admitted, completed, cancelled, and wasted work;
- bytes by credit class;
- dirty nodes and transitions per completion;
- queue starvation and steal success; and
- time to first and final output.

## Decisions currently recommended

The following defaults are plausible starting points, not settled architecture:

1. Fixed morsel ownership with globally stealable CPU tasks.
2. Hybrid predicate scheduling: broad I/O, selective CPU.
3. Open-demand projection I/O under byte and distance horizons.
4. Safe speculative CPU only for work that unlocks useful I/O.
5. Delta work updates with stable identities.
6. Immediate demand notification near commit; lazy rescoring farther ahead.
7. Fixed transition budgets in the simulator.
8. No adaptive task fusion until launch cost is measured.
