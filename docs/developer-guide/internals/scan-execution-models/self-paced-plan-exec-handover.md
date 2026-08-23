# Self-Paced Plan Execution Handover

This document is the implementation handover for the restricted self-paced plan execution
experiment on branch `ji/self-paced-fair-natural-splits`. It describes the code as it exists at the
end of the segment-streamed demand experiment. It is not a proposal to merge this executor as a
production scan path.

The companion [findings report](self-paced-plan-exec-findings.md) contains the broader benchmark
record. The [learning ledger](self-paced-plan-exec-learnings.md) keeps provisional conclusions that
may turn out to be incomplete or wrong.

## Current question

The experiment asks whether an explicit plan execution graph can improve a scan by:

- scheduling reads and predicate work at segment granularity;
- publishing progressively smaller row demand between predicates;
- sharing a decoded segment when filter and projection use the same field; and
- choosing between dependency-driven and parallel predicate execution.

It deliberately supports only a highly restricted serialized layout. Do not generalize its
results to all Vortex layouts or SQL execution.

## Fair comparison contract

The current comparison has these non-negotiable properties:

- The input is serialized once and reopened by both paths. The measured FineWeb Q06 file is
  1,669,473,052 bytes with stable hash `0x886de969ce96c930`.
- The allowed layout is exactly `Struct(Chunked(Flat))`. Unsupported layouts and planning
  rewrites are disabled by the experiment's layout strategy.
- V1 runs normally over every real natural split. It is never given self-paced morsels and must
  not fall back to a fixed row split.
- Self-paced merges 16 consecutive natural splits into each outer morsel. A morsel can cross a
  chunk boundary and is never smaller than any natural split it contains.
- Both paths scan the same rows, query object, serialized segments, warm fixture, and output. A
  stable row count and ordered hash are checked before a timing is accepted.
- Both are capped at concurrency 16 and the process is pinned to CPUs 0-15. If fewer than 16
  self-paced morsels exist, self-paced concurrency is capped to its morsel count and the result
  must call that out.
- Fixture construction and Parquet ingestion are outside the timed region. Timed runs consume all
  output and alternate executor order.

For the final FineWeb Q06 run, all 15 local Parquet files produced 14,868,862 rows, 157 ingestion
chunks, 1,823 natural splits, and 116 merge-16 self-paced morsels. Both sides therefore had enough
independent work to occupy 16 workers.

## Implementation map

The experimental implementation is concentrated in these files:

- `vortex-layout/src/plan/exec/model.rs` defines operations, completions, cached predicate
  coverage, policies, metrics, and trace events.
- `vortex-layout/src/plan/exec/graph.rs` defines shared resource nodes and their predicate and
  projection consumers.
- `vortex-layout/src/plan/exec/reactor.rs` owns the mutable execution graph, fragment state,
  resource deduplication, task readiness, completion adoption, and metrics.
- `vortex-layout/src/plan/exec/evaluate.rs` performs reads, flat decoding, sparse predicate
  evaluation, selection, packing, and final fragment-mask concatenation.
- `vortex-layout/src/plan/exec/baseline.rs` is the concurrent driver. It owns one `Execution`,
  admits tasks, runs inline operations, and sends other operations to the worker pool.
- `vortex-layout/src/plan/exec/tests.rs` checks result parity, fragment streaming, reduced demand,
  empty demand, sharing, scheduling, and trace behavior.
- `vortex-file/benches/self_paced_vs_v1.rs` builds the serialized fixtures, enforces the comparison
  contract, runs the suites, and prints traces and metrics.

## Execution flow

Each outer self-paced morsel is split internally at serialized chunk boundaries. These fragments
are mask-progress units, not new output morsels.

1. A fragment begins with logical all-true demand and the first predicate conjunct.
2. The plan executor attaches that conjunct and the current demand coverage to the fragment's
   segment read/decode task.
3. A worker reads and decodes the segment, then evaluates that predicate only for demanded rows.
4. Completion returns the decoded array plus a cached predicate value, its evaluated-row bitmap,
   input true count, and evaluation time.
5. The coordinator adopts the result directly into every waiter whose captured coverage is valid.
   A later waiter may reuse it only if its current demand is a subset of the evaluated coverage.
6. Reduced fragment demand can unblock the next predicate or projection segment before sibling
   fragments finish. Fragments of one morsel may progress independently and in parallel.
7. Once all fragments seal, `MergeDemandFragments` concatenates their bit buffers in row order.
   Projection selection consumes that single outer-morsel mask and preserves the output contract.

`SegmentId` is the current physical resource identity. One resource node can have predicate and
projection consumers, and the decoded array is reused when both refer to that resource. This is an
accepted experiment restriction: a production identity will also need source/layout context.

## Scheduler ownership

Fragment masks, predicate dependencies, cache coverage, and readiness belong to plan execution.
The scheduler should only decide which ready work to admit under CPU, I/O, concurrency, and byte
budgets. It should not interpret or combine row masks.

The experiment currently has one orchestration thread because `run_self_paced_concurrent` owns one
mutable `Execution`. Its loop drains completions, advances morsels, discovers ready tasks, claims
them, performs inline transitions, and queues outputs. Only claimed evaluation work is parallel.
This avoided locks and made leases, deduplication, cancellation, and traces deterministic, but it
places thousands of small transitions and bitmap publications on one critical path.

A likely production direction is to shard plan execution by morsel or small morsel groups. Resource
completion events would be published to the owning shards; a shared scheduler would retain global
admission and byte accounting. Cross-shard segment deduplication needs an explicit resource owner
or concurrent registry rather than accidental coordinator serialization.

## Metrics added

The trace separates physical work from execution machinery:

| Group | Important metrics | Meaning |
| --- | --- | --- |
| I/O | `requests`, `unique_segments`, `bytes` | Physical segment requests and returned bytes |
| Sharing | `shared_resources`, `shared_read_bytes`, `shared_decode_reuse_hits` | Filter/projection resource overlap and reuse |
| Graph | `transitions`, `nodes_inspected`, `tasks_*` | Coordinator and scheduling work |
| Fragments | `demand_fragments`, `fragment_predicates_completed`, `fragment_demand_updates` | Progressive mask state |
| Unblocking | `fragment_projection_reads_unblocked` | Projection work exposed before the outer mask seals |
| Fused work | `segment_predicates_fused`, `fragment_cached_predicate_hits` | Predicate work completed with read/decode and then adopted |
| Reduced demand | `reduced_demand_predicates`, `reduced_demand_input_rows`, `reduced_demand_skipped_rows` | Sparse predicate applications and avoided row visits |
| CPU estimates | `segment_predicate_eval_ns`, `fragment_demand_adoption_ns`, `fragment_merge_elapsed_ns` | Aggregate measured operation time, not wall time |

Q06 uses disjoint filter and projection fields, so its sharing metrics correctly remain zero. Other
tests and query shapes demonstrate reuse when a field appears in both.

## Final measured state

The final five-iteration, non-traced FineWeb Q06 comparison was:

| Executor | Median |
| --- | ---: |
| V1 natural splits | 22.240 ms |
| Self-paced merge-16 | 48.695 ms |
| Self-paced/V1 | 2.190x |

The last detailed trace had nearly equal physical work: self-paced issued 10,918 unique requests
and returned 714,536,112 bytes, compared with about 10,931 V1 requests and 714.6 MB. Self-paced
performed 5,461 fused segment predicates over 1,823 fragments and 116 morsels. Of the later
predicate applications, 3,638 used reduced demand: they evaluated 24,957 requested rows and
skipped 29,689,351 row applications. Aggregate predicate CPU fell from about 18.8 ms in the
all-row fused version to about 10.9 ms.

That work reduction did not improve wall time. The demand-aware path still publishes and adopts
thousands of partial masks through the single coordinator. The outside mask merge was only about
0.57 ms, so optimizing final concatenation alone is unlikely to close the gap.

## Experiment history

The progression on full FineWeb Q06 is useful when deciding what not to repeat:

| Variant | Approximate self-paced/V1 | Result |
| --- | ---: | --- |
| Per-fragment CPU predicate tasks | 2.299x | Correct streaming, too many tiny tasks |
| All predicates fused into read/decode | 2.105x | Fewer tasks, but evaluates rows later masks reject |
| Completion-side adoption | 2.047x best sample | Removed a redundant transition class |
| Separate sparse CPU task after decode | 2.431x | Saved predicate work but restored task overhead |
| Fused sparse predicate with per-bit demand assembly | 2.922x | Coordinator bitmap construction dominated |
| Byte-copy demand assembly | 2.212x sample | Recovered most of the per-bit regression |
| Final coverage-safe demand-aware path | 2.190x | Less predicate CPU, orchestration still dominates |

Trace collection perturbs short timings. Use traces to explain task and byte counts, and use
non-traced alternating runs for performance comparisons.

## Priority work

### P0: establish the coordinator cost

- Add wall and CPU timing around completion draining, `advance`, readiness discovery, task claim,
  inline completion, mask adoption, and output delivery.
- Count queue operations and mask bytes copied by phase. Existing aggregate predicate timers do
  not measure how much of the coordinator's critical path is occupied.
- Prototype two or four independent execution shards without changing predicate semantics. A
  useful result is a measured reduction in coordinator wall time, not only more worker activity.

### P0: protect correctness

- Preserve exact output row count/hash checks for every benchmark.
- Keep explicit cache-coverage tests for a resource shared by fragments or morsels with different
  demand. Never treat a partially evaluated predicate as a full-segment cache.
- Extend fragment tests across empty masks, nullability, multiple chunks, and resources spanning
  more than one outer morsel before broadening supported layouts.

### P1: reduce mask machinery

- Represent unresolved demand as bit buffers and versions in execution state. Materialize a
  `BoolArray` only at an evaluator or public array boundary.
- Batch several completion adoptions and advance affected fragments once per batch.
- Keep no-op all-true and unchanged masks symbolic. Do not allocate a full all-true `BoolArray`
  for every morsel or fragment.
- Re-evaluate fragment size and natural-split rollup using estimated bytes and CPU work, not only a
  fixed split count.

### P1: make policy adaptive

- Estimate the CPU saved by waiting for the preceding predicate from observed input/output true
  counts and per-row predicate cost.
- Compare that saving with observed dependency wait and coordinator publication latency. Run
  independent predicates in parallel when waiting is expected to cost more than the avoided work.
- Preserve predicate-order feedback, but distinguish ordering from parallelism: the cheapest or
  most selective predicate can be launched first while another is admitted concurrently.

### P1: benchmark representative shapes

- Re-run all ClickBench scan shapes, TPC-H single-table scans, and all local FineWeb data after any
  scheduler change, always with the fair contract above.
- Report filter/projection overlap, selectivity, projected field count, bytes, natural split count,
  morsel count, and whether 16-way parallelism was available.
- Include broad/select-all scans, highly selective scans, expensive predicates, shared filter and
  projection fields, many/few columns, and both I/O-heavy and CPU-heavy layouts.

### P2: production coverage

Add compressed encodings, nullable arrays, general expressions, object-store latency, cancellation,
memory pressure, and source-aware segment identity only after the control path is competitive on
the restricted layout.

## Reproduction

The final focused command was:

```bash
taskset -c 0-15 env \
  VORTEX_FINEWEB_PARQUET=/mnt/vortex-ssd/data/fineweb/parquet \
  VORTEX_FINEWEB_SPLIT_CATALOG=/tmp/fineweb-natural-splits.json \
  VORTEX_SELF_PACED_COMPARE_ITERATIONS=5 \
  VORTEX_SELF_PACED_COMPARE_WORKLOAD=fineweb_q06 \
  /mnt/vortex-ssd/vortex/target/release/deps/self_paced_vs_v1-2351f4c1137e39da
```

The executable hash is build-specific; rebuild the `self_paced_vs_v1` benchmark and use the
resulting binary. For diagnosis, add `VORTEX_SELF_PACED_COMPARE_TRACE=1`, but do not compare that
timing with non-traced medians.

The final verification before handover was:

```text
cargo test -p vortex-layout plan::exec
cargo clippy -p vortex-layout -p vortex-file --all-targets --all-features -- -D warnings
cargo +nightly fmt --all
```

The targeted execution tests passed 30/30 and Clippy completed without warnings.
