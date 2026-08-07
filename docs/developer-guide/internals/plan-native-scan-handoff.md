# Plan-native scan execution performance handoff

Status: work in progress on `ji/vx-plan-execution-perf-wip`.

This branch starts at `6f8023d653`, the top of the plan-native scan stack, and
contains the experimental execution changes described below. It is deliberately
not attached to a pull request yet.

## Goal and workload

The immediate goal was to explain and close the execution gap between the
existing `LayoutReader` scan and the plan-native v2 scan without adding branches
to the old scan implementation.

The comparison uses all 100 compressed ClickBench partitions in
`vortex-bench/data/clickbench_partitioned/vortex-file-compressed`, with:

- filter: `AdvEngineID != 0_i16`
- projection: `AdvEngineID`
- expected output: 630,500 rows
- one warm-up followed by the reported median execution samples

The branch includes
`vortex-scan-v2/examples/clickbench_plan_perf.rs` so both implementations run
through the same process, session setup, files, expression, projection, warm-up,
and concurrent execution strategy.

## Findings and changes

### Locate chunks by binary search instead of walking every chunk

`ChunkedPlan::execute` walked all `n` chunks on every split to accumulate row
offsets, instantiating each chunk's plan just to read its `row_count()`. A scan
performs one execution per split, so the work was `O(chunks x splits)` even
though a split usually intersects a single chunk.

`ChunkedLayout` now exposes its stored `chunk_offsets` prefix sum, and the plan
binary-searches it for the intersecting chunk range, matching `ChunkedReader`.
Chunk plans outside the range are never instantiated.

### Decode dictionary values once per plan, not once per split

`DictPlan::execute` executed its values child over the whole dictionary on every
split. `DictReader` instead caches the decoded values in a `OnceLock` and caches
expression evaluations over them, so the dictionary is read and any pushed
predicate is applied exactly once per scan.

`DictPlan` now caches the executed values array in a shared future, wrapped in a
`SharedArray` so canonicalization performed by one split is visible to the
others. Because the optimizer pushes safe boolean expressions into the values
child, this also memoizes the pushed predicate. The cache is reset by
`with_children`, since a rewritten child produces a different array.

On the dictionary workloads of the matched benchmark below, this moved v2 from
32-58% slower than v1 to 7-17% faster on projection, and from 33-38% slower to
roughly parity on filtering.

### Evaluate filter conjuncts one at a time

The v2 split executor evaluated the entire filter expression as one plan. The
`LayoutReader` scan splits the filter into its top-level conjuncts
(`FilterExpr`), evaluates them one at a time so each sees the mask narrowed by
its predecessors, stops as soon as no rows remain, and reorders them by the
selectivity measured on earlier splits.

`vortex-scan-v2` now plans one physical plan per conjunct and reuses
`FilterExpr` for ordering and selectivity feedback, so `vortex_layout::scan::filter`
is now public. `TaskContext::filter` holds a `FilterPlan` pairing the conjuncts
with their plans.

### Choose per-conjunct evaluation strategy by mask density

The previous revision of this branch always evaluated the filter with an
all-true mask and intersected afterwards, because applying a 48%-dense mask
below the comparison wrapped the compressed column in a filtered array that the
comparison then canonicalized. That is the right choice for a dense mask, but
the wrong one for a sparse mask: `FlatReader` picks between the two using an
`EXPR_EVAL_THRESHOLD` of 0.2.

`FilterPlan::execute` now applies the same threshold per conjunct. Above it the
conjunct is evaluated over the whole split and combined with `BitAnd`; below it
the mask is pushed into the conjunct's plan and the result combined with
`intersect_by_rank`. This keeps the dense-mask win from the earlier revision
while restoring v1's behaviour for sparse masks, which is the common case for
every conjunct after the first.

On the ClickBench workload, evaluating the predicate with an all-true mask and
intersecting afterwards reduced the representative v2 median from 46.460 ms to
37.861 ms. The density threshold preserves that result, because the incoming
mask on that workload is 48% dense.

### Cache zoned pruning evaluation across splits

`vortex-layout/src/plan/plans/zoned.rs` previously cached construction of the
zone map, but evaluated and rebound its statistics expression for every split.
The profile showed repeated `ZoneMap::applied_predicate` and expression binding
accounting for a significant part of the remaining v2 time.

`ZonedPruningState` now caches the evaluated pruning result and shares it across
split executions. It retains dynamic-expression correctness by tracking
`DynamicExprUpdates`: a newer dynamic-expression version invalidates and
recomputes the cached boolean pruning array. Static expressions return the
cached array directly.

After this change, the repeated zone-expression rewrite and bind stacks disappear
from the hot profile, and the representative v2 median falls to 28.899 ms.

## Results

These are representative medians from interleaved runs on the same checkout and
machine. Absolute timings are noisy; the profiles and repeated direction of the
change are the stronger evidence.

| implementation | state | execution median | output rows |
| --- | --- | ---: | ---: |
| v1 `LayoutReader` | unchanged baseline | 30.690 ms | 630,500 |
| v2 plan scan | before this branch | 46.460 ms | 630,500 |
| v2 plan scan | predicate before incoming mask | 37.861 ms | 630,500 |
| v2 plan scan | predicate ordering and zoned cache | 28.899 ms | 630,500 |

The final sampled profile used 826.835 ms of main-thread CPU over 25 executions,
down from 1,287.784 ms before the changes (35.8% less). The comparable v1 profile
used 862.435 ms. Planning remained approximately 2.6-3.0 ms for all 100 files.

### Matched workload sweep

The ClickBench partitions are not available in every development environment, so
`vortex-scan-v2/examples/synthetic_plan_perf.rs` generates its own dataset and
runs both implementations in one process over the same files, session,
expressions and concurrency, interleaving their samples.

The numbers below are `(v2 - v1) / v1` per workload, so negative is v2 winning.
Each column is the median of three rounds of eleven interleaved samples over
8 files of 1,000,000 rows in 65,536-row chunks. The absolute timings drift with
machine load; the paired deltas do not.

| workload | before | after |
| --- | ---: | ---: |
| `clustered-1pct` | -4.4% | -5.6% |
| `clustered-1pct-wide-projection` | -5.5% | -5.9% |
| `scattered-50pct` | -0.6% | -1.0% |
| `scattered-0p1pct` | -2.0% | -2.1% |
| `conjunction` | +0.9% | -0.0% |
| `string-equality` | +36.6% | +3.1% |
| `string-filter-only` | +21.4% | +1.9% |
| `string-projection-only` | +43.3% | -12.7% |
| `projection-only` | -13.5% | -5.9% |

The dictionary workloads carry the change. The remaining gap is the dictionary
filter path: `DictReader::filter_evaluation` evaluates `values.take(codes)`
directly, whereas `DictPlan` builds a dictionary array and optimizes it on every
split. Attempting the same shortcut in `DictPlan` did not produce a measurable
win above the noise on this machine, so it was not kept.

The `projection-only` column moved because v1 itself was slower in the rounds
that produced it, not because v2 regressed; the paired samples in each round
agree that v2 stays ahead.

## Reproduce

Build the matched benchmark once:

```bash
RUSTC_WRAPPER= cargo build --release -p vortex-scan-v2 \
  --example clickbench_plan_perf
```

Run either implementation from the repository root:

```bash
SCAN_VERSION=v1 PLAN_ITERS=5 EXEC_ITERS=11 \
  target/release/examples/clickbench_plan_perf

SCAN_VERSION=v2 PLAN_ITERS=5 EXEC_ITERS=11 \
  target/release/examples/clickbench_plan_perf
```

Set `PLAN_ONLY=1` to isolate planning. Set `RUST_LOG` to enable tracing from the
benchmark process.

The synthetic sweep needs no external dataset. It writes its files on first run
and reuses them afterwards:

```bash
RUSTC_WRAPPER= cargo run --release -p vortex-scan-v2 \
  --example synthetic_plan_perf
```

`VORTEX_SYNTH_DIR`, `SYNTH_FILES`, `SYNTH_ROWS` and `SYNTH_CHUNK` select the
dataset, `EXEC_ITERS` the sample count, and `WORKLOADS` a comma-separated subset.
Changing the chunk size is the way to vary the number of chunks per file, which
is what the chunk binary search responds to. Comparing revisions means building
the example on each and running the two binaries back to back, because each
binary reports its own paired v1/v2 medians.

The local profiles recorded during this investigation are:

- before v2 changes:
  `/private/tmp/plan-exec-repeated-before.profile.json.gz`
- v1 reader baseline:
  `/private/tmp/reader-exec-repeated-before.profile.json.gz`
- after direct compressed predicate evaluation:
  `/private/tmp/plan-exec-direct-compare-after.profile.json.gz`
- after both changes:
  `/private/tmp/plan-exec-zoned-cache-after.profile.json.gz`

For example:

```bash
samply load /private/tmp/plan-exec-zoned-cache-after.profile.json.gz
```

The profiles are local artifacts and are not committed to this branch.

## Validation completed

- `cargo +nightly fmt --all`
- `cargo clippy -p vortex-layout -p vortex-scan-v2 -p vortex-file -p vortex
  --all-targets --all-features`
- `cargo test -p vortex-layout -p vortex-scan-v2 -p vortex-file`
  (353 tests passed)
- matched v1/v2 scans agreed on output row counts for every synthetic workload,
  which the benchmark asserts before it measures anything
- `decodes_dictionary_values_once_across_splits` was confirmed to fail with the
  dictionary values cache disabled

A workspace-wide `cargo clippy` could not run in this environment: the
`lance-encoding` build dependency of `benchmarks/lance-bench` requires `protoc`,
which is not installed.

## Next steps

1. Add a focused zoned-plan test that changes a dynamic expression between split
   executions and proves the cached pruning result is invalidated exactly once.
2. Confirm the synthetic sweep's conclusions on the real ClickBench partitions
   with `clickbench_plan_perf`, which was not possible in this environment.
   The synthetic dataset covers low-, medium- and high-selectivity filters over
   struct, dictionary and chunked layouts, but it is a model of those shapes
   rather than a claim about every query.
3. Close the remaining dictionary-filter gap. `DictPlan::execute` optimizes a
   freshly built dictionary array on every split where `DictReader` takes the
   pre-evaluated boolean values directly. This needs a profiler to settle; the
   obvious shortcut was not distinguishable from noise here.
4. Decide the durable execution API. `FilterPlan` now makes the mask policy
   explicit for filters, but projections still rely on every expression plan
   interpreting a pushed mask optimally.
5. Profile the remaining `SerializedArray::decode` cost, which was 5.75% of the
   final sampled CPU. `FlatPlan` decodes its segment once per split, so a chunk
   wider than a split is decoded repeatedly; `FlatReader` has the same behaviour,
   so this is parity rather than a regression. If caching decoded arrays in
   `PlanExecutionContext`, use weak or execution-scoped ownership and include
   segment identity in the key so repeated scans do not retain the entire file or
   alias different sources.
6. `vortex_layout::scan::filter` became public so that v2 could reuse
   `FilterExpr`. When v1 is removed, move `FilterExpr` rather than duplicating it.
7. Once correctness coverage is complete, split the production changes onto the
   appropriate follow-up branches in the existing stack. Do not create another
   pull request solely for this WIP branch.
