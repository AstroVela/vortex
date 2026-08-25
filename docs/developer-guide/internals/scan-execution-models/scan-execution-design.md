# Scan Execution Design: Morsel-Driven Demand Model

This is the consolidated design produced by the discussion series recorded in the
[graph model](scan-execution-graph-model.md) and
[demand, operators, and the filter law](scan-execution-demand-and-operators.md). Those documents
keep the derivations and rejected alternatives; this one states the resulting design in its own
terms, complete enough to build against. The
[one-pager](scan-execution-design-one-pager.md) is the compressed version.

## 1. Goals

- **Performance**: pipeline-grade hot paths (no coordinator on the row path), IO saturation via
  optimism, skipping that reaches IO, and no fixed cost proportional to split count.
- **Extensibility**: a layout author writes combining semantics and declarations, never
  scheduling, demand propagation, coordinate arithmetic, or buffering; ignoring demand is
  merely eager, never wrong.
- **Verifiability**: one eager oracle, a small set of testable laws, and a differential harness
  that makes the laws checks rather than conventions.

## 2. The three laws

Everything in the design is licensed by three statements.

1. **Value contract.** Every in-flight value is *positional over its row domain* (length n; rows
   outside the current need set are undefined and may hold anything). There is exactly one
   cardinality-changing primitive — **gather-by-map** — and it is an explicit planned node,
   never a runtime mode. "Compacted" data is ordinary positional data in a smaller (survivor)
   domain, reached by a domain-change edge whose map is a sealed mask: the same machinery as
   dict values and list elements.
2. **Commutation law.** For a row-local kernel `f`, total on its domain, and any superset
   snapshot of demand: `f(sel_R(x)) = sel_M(R)(f(x))`, transported through the edge's
   `DomainMap`. Running work early on stale, larger demand is therefore always correct; the
   selection applied later fixes it. Kernels must not trap on undefined rows (errors are
   values); a kernel that cannot be total is ineligible for speculation.
3. **Advisory demand.** Demand information is *never required and never blocks*. It is read at
   admission points; a lost or late update can only cost performance. The single
   synchronization point in the system is the gather's in-band mask input being final — an
   ordinary dataflow dependency.

## 3. Two planes

- **Data plane (in-band, exact, authoritative).** Batches and conjunct masks stream through
  operator pipelines. Masks are values; the AND happens where they meet; the gather consumes
  the final mask as its map. Correctness lives entirely here.
- **Control plane (out-of-band, advisory).** Demand cells, wired at bind time by composing the
  plan's `DomainMap`s into a producer-to-consumer routing table. Content is any *monotone
  shrinking superset*: exact bounds, zone verdicts, bloom filters or IN-lists from joins
  (sideways information passing is this plane, not a later feature), and the limit counter.
  Consumers are admission points only: morsel planning, the pre-issue look of an IO task, and
  scheduler pricing. Operators in a pipeline never see this plane.

Optimistic conjunct IO and optimistic projection IO are the same move: admit reads against
whatever the cell currently holds (top if nothing landed). Cascade versus parallel-eager is
"how stale was the snapshot at admission" — a continuum controlled by the scheduler, not two
code paths.

## 4. Architecture

Three parts.

### 4.1 Stateful exec plans

Per layout, per scan: operator pipelines in the DuckDB/Velox style — instances that hold
buffers, accumulators, and scratch, owned by the morsel (not the thread) and driven as plain
function calls. One pipeline per conjunct; a small number per projection. Struct is **not** a
pipeline breaker: fields are sequential stages of one instance, with elective field-parallel
fan-out only above the granularity floor. The only barrier-like point is the per-range mask
meet, implemented as a countdown. Statefulness is safe because demand is not the operators'
job (law 3).

### 4.2 The demand system

The control plane of §3: bind-time routing table; cells per (domain, region); producers are
pruning, conjunct seals, joins, and the limit; consumers are admission points. Non-identity
domain relationships (chunk shifts, dict codes-to-values, list offsets, the filter's survivor
crossing) are edge maps — static ones composed at bind, gated ones snapped in when their fact
seals.

### 4.3 The scheduler

Morsel-driven with work stealing.

- **Morsels are typed by row domain** and claimed from a per-domain queue. A domain change
  spawns **child-domain morsels** at gate seal — own demand (sealed at birth), own pipelines,
  own stash — whose results land in the parent's stash for the parent's combine.
- **Depth-first priority**: inner-domain morsels run before new outer splits are claimed
  (workers run their own newest work, steal the oldest/outermost). This bounds work-in-progress
  near workers × domain-depth and drains stashes before opening new ones.
- **No new morsels when**: (1) all rows claimed, in every domain; (2) the memory limit binds
  (deferred, §7); (3) the limit has sealed the remaining tail.
- **Work stealing**: the stolen unit is a task (a range's pipeline activation, an IO
  continuation); wakes prefer the owning worker's deque so continuations run cache-warm.

## 5. The morsel lifecycle

```text
scan open:  lower + bind once
            - routing table (composed maps), kernel table, pipelines per layout
            - file-level stats resolve; scan-wide cells allocated (dictionaries, stats)

per morsel (row domain R, rows [a, b)):

 1. PLAN (once, cheap, inline)
    - read OOB demand snapshot; a subtree with empty demand is never planned
    - cut against coverage tables; emit IO tasks, pipeline activations,
      and deferred notes ("plan X when fact F seals") for gated subtrees
    - first-wave morsels also emit the pruning-metadata reads (see below)
 2. IO
    - each read takes one late look at its demand cell just before issue
      (drop if dead, shrink if smaller); reads batch-coalesce within the
      morsel plan; dedup via scan-wide cells by SegmentId
 3. FILTER
    - conjunct pipelines run as CPU tasks; masks stream in-band and meet
      at the countdown; a sealed conjunct also writes its bound to the
      OOB plane for later morsels and later-admitted reads
 4. GATHER
    - the one cardinality change: sealed mask becomes the survivor-domain
      map; runs when its in-band mask input is final
 5. PROJECT + EMIT
    - projection pipelines reuse stash entries (decodes shared with the
      filter); pack; emit pull-driven and morsel-ordered
 6. RETIRE
    - stash dropped wholesale; child morsels must already be retired
```

**Pruning is warm-up work, not a phase.** Zone-map metadata is file-scoped and small. The first
morsels issue pruning-metadata reads and their first conjunct's IO optimistically in parallel;
the early wave simply does not benefit from pruning (accepted speculation cost, overlapped with
IO latency). When the stats fact seals, verdicts for every zone are computed in one bulk pass
and written to the OOB cells; every subsequent morsel is pruned for free, some subtrees never
planned. The more morsels a file has, the closer pruning is to free.

**One-off planning.** Planning never reruns on demand refinement; the late look at IO issue
captures the shrink (safe by law 2). Gated subtrees are deferred planning, not re-planning.

## 6. The per-morsel stash

Each morsel owns a scratch store keyed by (plan edge, range): decoded arrays shared between
filter and projection, partially computed arrays, masks awaiting the meet, child-morsel
results. Lifetime is the morsel's; the whole stash drops at retire. The morsel's live bytes are
the stash plus in-flight IO — the memory accounting unit. Cross-morsel sharing is a short
explicit list of scan-wide keyed cells: dictionary values, file stats, the pruning fact.
Chunks straddling a morsel boundary are decoded twice by default (bounded-duplicate
principle); promote straddlers to scan-wide cells only on measured need.

## 7. Memory (deferred, recorded)

Per-morsel live-byte approximation is the admission input. Unresolved and consciously
deferred: attribution of scan-wide cells (first-needer, split, or shared pool), and the
memory-times-ordering deadlock — complete-but-unemittable morsels holding bytes the oldest
morsel needs; candidate rule: *the oldest unemitted morsel is always admissible*. A single
oversized morsel needs a degenerate path (shrink or go sequential), not just "no new morsels."

## 8. Emission, limits, cancellation

Emission is pull-driven: the consumer stopping stops morsel claiming; in-flight morsels finish
or park. Limit is a first-k demand producer at the sink writing into projection's OOB cells —
per-morsel for ordered prefix consumption, a shared survivor counter for unordered — and
transitively bounds filter work. It is the one legitimately cross-morsel demand. Errors seal
cells with an error value and ride the ordinary wake path.

## 9. What a layout implements

- **Declarations** (bind time): edges with `DomainMap`s, coverage tables, kernels into the
  per-scan kernel table.
- **Planning contribution** (morsel plan time): cut demand against coverage, emit IO/CPU/
  deferred-planning tasks — derived generically from the declarations for most layouts;
  overridden only where planning is semantic (Zoned pruning, Dict, List).
- **Combine**: assemble pre-cut, pre-aligned children (zip, wrap, intersect, take). O(parts)
  if inline; a combine that touches rows must be priced. The opt-in `absorb` refinement
  (order-free folds, accumulator is a `Value`) exists for stragglers' memory and producer-
  thread cache warmth, tested against its buffer-then-combine blanket impl.

Explicitly not a layout's job: scheduling, demand, coordinates (driver slices by the recorded
cut, per edge map), buffering (stash), ordering, retention. Not expressible by design:
node-level mutable state outside the stash, order-dependent folds in the scan path.

## 10. Correctness and testing

- **Oracle**: the eager configuration (demand = top, no OOB plane, filters as all-true
  selections, gathers at emit) is `run_eager` and the permanent differential reference. Hashes
  compare *need-set rows*, never batches (batch boundaries legitimately vary with schedule).
- **Law suites**: `DomainMap` round-trip and composition-preserves-superset properties;
  kernel totality flags (`can_trap`) audited — v1's `CAST(a, u8) WHERE a < 256` comment is the
  live counterexample gating elective gathers and CPU speculation; OOB plane disabled and
  maximally delayed must produce identical results; `absorb` versus blanket impl.
- **Simulator**: deterministic scheduler tests — no deadlock under memory × ordering × limit ×
  cancellation, adversarial IO completion orders, steal-versus-wake races.
- **Performance gates**: Q01/Q06 (prefetch split), FineWeb `select *` (splits storm), selective
  string predicates (cascade + wide-value elision), dict page skipping, cold-scan IO parity
  (same bytes, comparable request count as V1 — the check on batch-scope coalescing),
  contention counters (entries-per-admission ≈ 1, queue-idle ≈ 0).

## 11. Migration

V1 (`LayoutReader`) is the semantic oracle throughout; its `FlatReader` already implements the
model in miniature (positional mask demand, the density switch, `intersect_by_rank` as unmap
through the survivor crossing, explicit filter at projection). Graft onto `PlanVTable` with
`edges()`/`bind()` beside `execute`; migrate SegmentScan, Concat, Eval, Pack first, then Zoned,
then the gated pair (Take/Dict, ListPack); unported subtrees fall back to V1 mid-tree. Every V1
layout (flat, chunked, struct, dict, list, zoned, row_idx, partitioned, table, compressed,
buffered, repartition, foreign, file_stats) needs its one-line story or the fallback. Degenerate
paths ship early: `select *` with no filter machinery, tiny single-morsel scans without
scheduler spin-up, repeated-scan fact reuse.

## 12. Deferred decisions

The memory model (§7); the `can_trap` audit mechanics; kernel-table payload representation;
pool scope (per-scan versus session); speculation floor; the ordered-emission window for
streaming consumers; whether any node-level trait remains once planning contributions and
combines are the whole layout surface.
