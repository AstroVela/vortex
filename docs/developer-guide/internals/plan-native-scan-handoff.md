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

### Evaluate the predicate before applying the incoming mask

`vortex-scan-v2/src/tasks.rs` previously passed the zone-pruned input mask into
the filter expression plan. On this workload, zone pruning removes 51.831% of
the rows, leaving a 48%-dense mask. Applying that mask below the comparison
wrapped the compressed filter column in a filtered array, after which comparison
canonicalized it. The old flat reader instead compares the compressed input
directly and combines the predicate result with the incoming mask afterward.

The v2 split executor now evaluates the filter plan with an all-true mask and
then intersects the predicate and incoming masks with `BitAnd`. This reduced the
representative v2 median from 46.460 ms to 37.861 ms.

This is an execution-policy change, not just a local micro-optimization. Before
promoting it from the WIP branch, add focused coverage for sparse selections,
fallible expressions, and filters whose plans contain several physical layout
nodes. If selective execution is needed for other expressions, make that policy
explicit rather than implicitly pushing every incoming mask through every plan.

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
- `RUSTC_WRAPPER= cargo nextest run -p vortex-layout -p vortex-scan-v2`
  (215 tests passed)
- matched v1/v2 scans repeatedly produced 630,500 rows

## Next steps

1. Add a focused regression test proving that split-level pruning or selection
   masks are combined after predicate evaluation, including a sparse mask and a
   fallible predicate.
2. Add a focused zoned-plan test that changes a dynamic expression between split
   executions and proves the cached pruning result is invalidated exactly once.
3. Exercise representative low-, medium-, and high-selectivity filters, plus
   struct, dictionary, row-index, and multi-layout filters. The current evidence
   is intentionally narrow rather than a claim that all queries improve.
4. Decide the durable execution API. An explicit mask/expression execution policy
   would be clearer than relying on every expression plan to interpret a pushed
   mask optimally.
5. Profile the remaining `SerializedArray::decode` cost, which was 5.75% of the
   final sampled CPU. If caching decoded arrays in `PlanExecutionContext`, use
   weak or execution-scoped ownership and include segment identity in the key so
   repeated scans do not retain the entire file or alias different sources.
6. Once correctness coverage is complete, split the production changes onto the
   appropriate follow-up branches in the existing stack. Do not create another
   pull request solely for this WIP branch.
