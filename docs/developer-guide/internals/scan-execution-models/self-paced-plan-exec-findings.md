# Self-Paced Plan Execution Findings

This report records what was learned while implementing and optimizing the restricted
[self-paced plan execution experiment](self-paced-plan-exec-experiment.md). It includes the
original 100-iteration comparison on 2026-08-21 and a capacity-saturating FineWeb follow-up on
2026-08-22. It is evidence about this experiment, not a claim about a production executor.

## Original headline result

In the original comparison at 131,072 rows per self-paced morsel and concurrency 16, self-paced
execution won 15 of 28 scan workloads. The unweighted geometric mean of self-paced/V1 median time
ratios was `0.891`, or 10.9% faster overall.

| Suite | Workloads | Wins | Geometric mean self-paced/V1 | Interpretation |
| --- | ---: | ---: | ---: | --- |
| ClickBench scan shapes | 16 | 7 | 0.918 | Selective/reuse wins offset a 2.5-3.3% tax on broad scans |
| TPC-H scan shapes | 3 | 3 | 0.764 | Q6 benefits strongly from progressive filtering; the V1-friendly case is near parity |
| FineWeb scan analogues | 9 | 5 | 0.889 | Mixed; sub-millisecond cases expose fixed control costs |
| Combined | 28 | 15 | 0.891 | Promising for a restricted experiment, with workload-dependent wins |

The ratio is the geometric mean of per-workload median ratios. Summing all wall times gives a
different and less useful answer because long broad scans dominate that calculation.

### Complete-data follow-up

The original headline used ten of 100 ClickBench shards, a synthetic TPC-H fixture, and one of 15
FineWeb 10BT shards. Those inputs were useful while iterating but were not acceptable as final
benchmark evidence. A 20-iteration follow-up used every locally available benchmark row:

| Suite | Input | Rows | Self-paced wins | Ratio range |
| --- | --- | ---: | ---: | ---: |
| ClickBench | all 100 Parquet shards | 99,997,497 | 0/16 | 1.192-1.818 |
| TPC-H | full SF10 lineitem table | 59,986,052 | 0/3 | 1.154-1.501 |
| FineWeb | all 15 official 10BT sample shards | 14,868,862 | 0/9 | 1.140-3.135 |

This reverses the original headline: self-paced loses every complete-data scan shape. The result
is still about the restricted scan analogues, not full SQL query runtimes.

The final FineWeb measurements were:

| Workload | V1 ms | Self-paced ms | Ratio |
| --- | ---: | ---: | ---: |
| FineWeb Q00 analogue | 1.206 | 2.808 | 2.328 |
| FineWeb Q01 analogue | 5.312 | 6.058 | 1.140 |
| FineWeb Q02 analogue | 1.675 | 2.821 | 1.684 |
| FineWeb Q03 analogue | 2.607 | 7.220 | 2.770 |
| FineWeb Q04 analogue | 2.155 | 5.837 | 2.709 |
| FineWeb Q05 analogue | 2.147 | 5.846 | 2.723 |
| FineWeb Q06 analogue | 2.748 | 8.615 | 3.135 |
| FineWeb Q07 analogue | 2.151 | 5.849 | 2.720 |
| FineWeb Q08 analogue | 1.955 | 4.761 | 2.436 |

These are scan-only medians over 20 alternating iterations with speculative I/O disabled. Q06
gives both executors exactly 942 unique segments and 713,799,576 logical bytes and produces the
same 11,898 rows. Self-paced additionally executes 2,682 tasks, 1,632 reactor advances, 5,738
transitions, and 7,256 node inspections. Its scheduler considers only 1.03 entries per admission,
so scheduler rescanning and unequal I/O do not explain the `3.135x` result.

The full SF10 measurements were Q6 `16.366/23.580 ms` (`1.441x`), Q1 `8.460/12.698 ms`
(`1.501x`), and the V1-friendly case `3.863/4.459 ms` (`1.154x`). An earlier eager-copying Q1 path
took 173.7 ms. Returning lazy filtered projection views reduced it to 12.7 ms without changing
output, demonstrating that executor comparisons must align output materialization semantics.

## Comparison contract

The comparison was constructed as a scan comparison, not a like-for-like execution-model
comparison:

- V1 runs through `ScanBuilder` with its native splitting and no experiment morsels.
- Self-paced execution uses 131,072-row morsels, a transition budget of 32, and the adaptive
  predicate policy.
- Both paths use concurrency 16, the same serialized Vortex layout, in-memory `SegmentSource`,
  filter, projection, input rows, and warm fixture state.
- They do not use identical worker executors: V1 is driven by the 16-worker Tokio runtime, while
  self-paced non-inline tasks use a shared futures thread pool behind the same concurrency cap.
- Each path gets a warm-up. The 100 measured iterations alternate which executor runs first, and
  the reported time is the median.
- Every warm-up compares output row count and a stable ordered hash before timings are accepted.
- Timed runs consume every output. Fixture construction and Parquet ingestion are outside timing.

The original data sets were:

- the first ten real ClickBench Parquet shards, converted to the experiment's supported `i64`
  fields and totaling 10,000,000 rows;
- a deterministic 2,097,152-row TPC-H lineitem-shaped fixture; and
- one 1,046,615-row FineWeb Parquet sample converted to `i64` scan features.

FineWeb ingestion can now scale beyond the default sample. `VORTEX_FINEWEB_PARQUET` accepts either
a Parquet file or a directory; directory inputs use every `.parquet` file in sorted order.
`VORTEX_FINEWEB_MAX_FILES` optionally caps that list for repeatable size sweeps. The runner prints
the resulting file, chunk, and row counts before execution.

The complete-data runner also accepts `VORTEX_CLICKBENCH_MAX_FILES`; setting it to 100 consumes
every local ClickBench shard. `VORTEX_TPCH_LINEITEM_PARQUET` switches from the synthetic fixture
to a real lineitem Parquet table, converting decimal quantities and Arrow dates into the restricted
executor's `i64` domain. `clickbench_all`, `tpch_all`, and `fineweb_all` load only their selected
fixture and execute every scan shape in that suite.

All nine FineWeb analogues can be selected without loading the unrelated fixtures:

- `VORTEX_SELF_PACED_COMPARE_WORKLOAD=fineweb_q00` through `fineweb_q08`, or `fineweb_all`;
- `VORTEX_SELF_PACED_TRACE=fineweb-q00-128k` through `fineweb-q08-128k`; and
- `VORTEX_SELF_PACED_PROFILE=fineweb-q00-self-128k` or `fineweb-q00-v1-128k`, with any query ID.

The ClickBench and FineWeb cases are scan-input analogues. They preserve useful filter and
projection shapes, but exclude aggregation, grouping, ordering, strings, and disjunction because
those are outside the restricted evaluator. They must not be reported as full query runtimes.

## Final 128K results

Values below are median milliseconds over 100 alternating iterations. Ratios below one favor
self-paced execution.

| Workload | V1 ms | Self-paced ms | Ratio |
| --- | ---: | ---: | ---: |
| ClickBench selective | 0.864 | 0.743 | 0.860 |
| ClickBench dashboard | 1.708 | 1.651 | 0.967 |
| ClickBench Q00 | 11.512 | 11.823 | 1.027 |
| ClickBench Q01 | 2.014 | 1.051 | 0.522 |
| ClickBench Q02 | 22.572 | 23.141 | 1.025 |
| ClickBench Q03 | 11.516 | 11.896 | 1.033 |
| ClickBench Q04 | 11.516 | 11.886 | 1.032 |
| ClickBench Q05 | 11.512 | 11.893 | 1.033 |
| ClickBench Q06 | 11.512 | 11.893 | 1.033 |
| ClickBench Q07 | 2.015 | 1.050 | 0.521 |
| ClickBench Q08 | 22.558 | 23.184 | 1.028 |
| ClickBench Q09 | 44.555 | 45.651 | 1.025 |
| ClickBench Q39 | 5.810 | 5.318 | 0.915 |
| ClickBench Q40 | 1.259 | 1.299 | 1.032 |
| ClickBench Q41 | 1.838 | 1.776 | 0.966 |
| ClickBench Q42 | 2.331 | 2.259 | 0.969 |
| TPC-H Q6 scan | 1.193 | 0.622 | 0.522 |
| TPC-H Q1 scan | 11.325 | 9.897 | 0.874 |
| TPC-H V1-friendly | 2.480 | 2.421 | 0.976 |
| FineWeb Q00 analogue | 1.303 | 1.604 | 1.231 |
| FineWeb Q01 analogue | 1.000 | 0.677 | 0.677 |
| FineWeb Q02 analogue | 0.363 | 0.293 | 0.806 |
| FineWeb Q03 analogue | 0.355 | 0.397 | 1.119 |
| FineWeb Q04 analogue | 0.704 | 0.413 | 0.587 |
| FineWeb Q05 analogue | 0.313 | 0.313 | 1.001 |
| FineWeb Q06 analogue | 0.366 | 0.448 | 1.226 |
| FineWeb Q07 analogue | 0.324 | 0.316 | 0.976 |
| FineWeb Q08 analogue | 0.194 | 0.128 | 0.659 |

## Why self-paced can be faster

The main advantage is different work, not a universally cheaper executor.

Self-paced execution evaluates predicates against the current shrinking demand. Projection reads
and selection wait until demand seals, so empty or sparse morsels can avoid projection work. A
predicate result already contains false bits outside its input demand; when it was evaluated at the
current demand version, the executor adopts that result directly instead of intersecting the same
two masks again.

Shared resources are interned by `SegmentId` and retained across possible morsel users. This can
turn repeated V1 requests into one self-paced request. The clearest measured example was
ClickBench Q42:

| Metric | V1 | Self-paced | Change |
| --- | ---: | ---: | ---: |
| Output rows | 558,105 | 558,105 | identical |
| Stable output hash | `0xe0d6122ac6c3572e` | `0xe0d6122ac6c3572e` | identical |
| Segment requests | 255 | 42 | 83.5% fewer |
| Unique segments | 36 | 42 | self-paced touched more distinct segments |
| Bytes returned | 1,296,016,200 | 336,004,200 | 74.1% fewer |

This is why Q42 became competitive despite scheduler overhead: V1 repeatedly requested some
segments, while the scan-wide self-paced graph requested each of its 42 segments once. The result
also shows why unique-segment count alone is misleading; total requests and bytes explain the wall
time better.

TPC-H Q6 is the strongest predicate-pipelining result. Five conjuncts progressively reduce demand
before the two projected fields are materialized, producing a `0.522` ratio. Q1, with one broad
predicate and four projected fields, still reaches `0.874`, while the deliberately simple
V1-friendly scan reaches `0.976`. That near-parity case is useful: it bounds the fixed experimental
tax when there is little scheduling opportunity.

## Why self-paced can be slower

When nearly every row and projected value is needed, self-paced has little work to avoid. It still
pays for execution construction, slots, offers, claims, completion messages, `advance` calls,
demand masks, selection, and Struct packing. ClickBench Q00, Q02-Q06, Q08, and Q09 expose this
cost: the final ratios cluster from `1.025` to `1.033`.

The FineWeb traces separate I/O from control cost. Q03 read exactly 55 segments and 41,870,100
bytes in both executors, yet self-paced was 11.9% slower. Q06 read exactly 66 segments and
50,244,120 bytes in both, yet self-paced was 22.6% slower. With equal logical I/O and absolute
times below half a millisecond, task dispatch, mask handling, and dependency waits dominate.

The experiment does no statistics or metadata pruning. Both engines receive the same logical
filter, but their execution paths may request different segments due to late materialization,
native V1 splitting, and self-paced scan-wide retention. The counting source measures logical
requests and returned buffer bytes, not physical NVMe or object-store traffic.

## Graph and control overhead

The Q42 trace makes the size of the experimental control plane concrete:

```text
10,000,000 input rows
50 scan-wide resource nodes
616 morsel-local slots
569 advance calls
1,308 transitions
1,800 node inspections
410 offered, claimed, and completed tasks
243 direct demand adoptions
472 adaptive waits
162 predicate reorders
```

The graph is sized by resources, morsels, fields, and conjuncts rather than by logical rows.
`advance` inspected about 3.2 nodes per call in this trace and remained bounded by the transition
budget. The cost is nevertheless material on short scans because every task still crosses offer,
claim, completion, wake-up, and slot-state machinery.

The equal-I/O FineWeb traces show two different scheduler shapes:

| Metric | FineWeb Q03 | FineWeb Q06 |
| --- | ---: | ---: |
| Advance calls | 133 | 130 |
| Transitions | 288 | 341 |
| Nodes inspected | 413 | 463 |
| Tasks completed | 166 | 193 |
| Inline demand combinations | 8 | 5 |
| Direct demand adoptions | 8 | 13 |
| No-op demand adoptions | 0 | 6 |
| Adaptive launches | 8 | 16 |
| Adaptive waits | 0 | 16 |
| Predicate reorders | 0 | 3 |

Q06 has more predicate coordination without an I/O saving, matching its larger regression. This
is stronger evidence than attributing the result to mask intersection alone: only five explicit
combinations ran, and they ran inline.

Early Samply captures did not symbolicate the benchmark binary reliably, including one report
with zero of 370 raw addresses resolved. The conclusions above therefore rely on median timings,
event traces, and operation counters rather than unresolved sampled stacks.

## Morsel size

The 131,072-row setting improved the self-paced/V1 ratio over 65,536 rows on 27 of 28 workloads;
FineWeb Q00 was the exception. Larger morsels reduce the number of per-morsel graphs, masks, tasks,
queue operations, and output batches. They also give each predicate task enough rows to amortize
dispatch and array setup.

The complete-data fixtures expose ample parallel work: ClickBench has about 763 morsels, SF10
lineitem about 458, and FineWeb 10BT about 114. Their regressions therefore cannot be attributed to
having fewer outer morsels than the 16-worker concurrency setting.

Larger morsels are not intrinsically faster. They may reduce early output, increase mask and
temporary-array size, or leave fewer independent morsels when row counts are small. The experiment
therefore chose 128K as the final comparison point, not as a production constant.

Morsels partition the root row domain independently of storage chunks. The implemented layout is
`Struct<Chunked<Flat>>`; a morsel carries ordered Flat slices and may cross aligned field-chunk
boundaries. This avoids coupling scheduling granularity to physical chunking.

## Adaptive predicate scheduling

The adaptive policy supports both demand pipelining and parallel predicate execution. It records,
per conjunct, cumulative input rows, output rows, elapsed nanoseconds, and sample count from prior
completions. It then:

1. ranks predicates by expected rows eliminated per nanosecond;
2. uses observed survival, falling back to priors of 10% for equality and 50% for inequalities;
3. computes a per-morsel supply window from global concurrency and morsel count; and
4. launches another predicate only when estimated parallel latency, including a 3 microsecond
   launch cost, is lower than waiting and evaluating it on the expected survivors.

This is adaptive across completed morsels, not clairvoyant within the first morsel. Unseen
predicates use priors, and the policy waits when it lacks observations for either the outstanding
or next predicate. Reordering and launch/wait counts are explicit metrics.

Running predicates concurrently means they may capture different demand versions. Three cases
avoid unnecessary mask work:

- a result computed from the current demand is adopted directly;
- a stale result whose true count equals its captured input count eliminated nothing and is a
  no-op against any newer subset of that input; and
- only a stale result that eliminated rows needs an explicit `CombineDemand` intersection.

`CombineDemand` runs inline because it is small, dependency-critical work. `PackStruct` also runs
inline for the current adaptive policy. These choices avoid thread-pool round trips while keeping
reads, decodes, predicates, and selections parallel.

## Optimizations that mattered

The implementation converged on several small fast paths rather than one broad special case:

- reuse the materialized all-true initial demand by morsel length instead of allocating one per
  morsel;
- use direct and no-op demand adoption to remove redundant mask intersections;
- run dependency-critical mask combination and final Struct packing inline;
- wake only morsels recorded as waiting on a completed shared resource;
- retain and look up scan-wide resources directly by `SegmentId`;
- keep task inputs and leases in `SmallVec` storage for the common small arities; and
- retain the shared bit buffer in boolean summaries so resource-local range counts do not
  canonicalize the mask again;
- cache selected-row counts by morsel-relative range, sharing one count across aligned fields;
- omit projection reads and `SelectFlat` inputs for physical resource slices with zero selected
  rows, including when a morsel crosses chunk boundaries; and
- skip copying projected Flat values when demand is all true and one Flat slice covers the range;
  and
- stop scheduler selection when the available executor capacity is filled instead of constructing
  a full admissible frontier that the caller immediately truncates;
- traverse the adaptive ready frontier newest-first without allocating a reversed copy, allowing
  newly unblocked decode, predicate, and selection work to pipeline ahead of old reads; and
- return lazy filtered projection views for partial masks instead of copying selected values into
  eager compact buffers.

All-true selection returns slices of decoded arrays. Partial selection now wraps those same slices
in Vortex's compact logical `FilterArray`, matching V1's output materialization behavior.

The Q40-Q42 work showed that scheduler policy and fixed overhead interact. One optimization pass
reduced the 128K Q40, Q41, and Q42 times by 56.8%, 48.2%, and 18.2% respectively relative to the
preceding implementation. Subsequent fast paths brought the final ratios to `1.032`, `0.966`, and
`0.969`. The progression is evidence that the initial regression was mainly execution mechanics,
not an unavoidable cost of self-paced plans; the intermediate run does not isolate one causal
change.

FineWeb Q01 exposed the resource-local projection issue. With speculation disabled, the old
morsel-wide nonempty check completed 50 segments and 39.6 MB. Range-aware projection completed 37
segments and 29.2 MB, while V1 completed 30.8 MB. Ten-iteration self-paced time fell from about
2.46 ms to 2.21 ms. The remaining `2.11x` ratio is fixed scheduling cost on a roughly 1 ms V1
query, rather than excess projection I/O.

A subsequent control-plane pass made two costs directly visible. First, each projection field
walked the same partial demand mask. An intermediate implementation cached one immutable
selected-index buffer across fields, but complete-data Q1 showed that eagerly gathering values was
the wrong output contract regardless of index reuse. Lazy `FilterArray` views replaced that cache.
Second, the concurrent runner consumed one completion before returning to the reactor and
scheduler, even when many worker results were already queued. It now drains ready completions as a
batch before advancing morsels.

The first two changes reduced 128K FineWeb Q01 from 2.214 ms to 0.927 ms and Q06 from 0.949 ms to
0.626 ms in 20-iteration follow-ups. A trace then exposed a remaining scheduler issue: with
speculative I/O disabled, 143 candidate projection reads were retained in the external offered
queue. Q01's scheduler considered 2,807 entries to admit 118 tasks, a `23.79x` ratio. Candidate
tasks now remain dormant in reactor state unless their speculative class is enabled; promotion to
required work inserts them into the runnable queue. The same trace after the fix considered 197
entries for the same 118 admitted tasks, a `1.67x` ratio. Q01 reached 0.758 ms versus V1's 1.033 ms
(`0.734x`), and Q06 reached 0.576 ms versus V1's 0.373 ms (`1.545x`).

The executor metrics now report scheduler passes, tasks considered and admitted, completion
batches, completions drained, and maximum batch size. The repository-local
`summarize_self_paced_trace.py` tool combines those totals with per-operation task latency, wait
time, and reactor work from an execution trace. This is the routine diagnostic layer. Samply
spans remain the next step when the report points to CPU cost inside a particular operation.

The full SF10 trace exposed breadth-first launch waves despite full occupancy: the FIFO frontier
started with runs of 115 reads, 115 decodes, and 474 predicates, and emitted its first morsel at
25.9 ms of a 29.0 ms traced run. Adaptive newest-ready traversal reduced those initial runs to 16,
16, and 78 and emitted the first morsel at 2.0 ms. Every recorded wait still had 16 running tasks.
This removes the hidden wave behavior and dramatically improves time to first output, but total
throughput remains similar because all 4,584 tasks still execute.

### FineWeb Q00

Q00 applies `int_score > -1` and projects one field. On the available data, all 1,046,615 rows
survive. With 128K morsels, eight output morsels cross eleven physical chunks. The original
single-slice all-true fast path therefore missed most morsels and copied their values into compact
buffers. `SelectFlat` now returns a zero-copy `ChunkedArray` of sliced Flat inputs when all-true
ranges form a complete partition of a morsel. For a one-field projection, selection also produces
the final Struct directly, removing eight separate `PackStruct` tasks.

The comparison harness now hashes both outputs during its correctness warmup and excludes hashing
from timed iterations. This matters for zero-copy output: canonicalizing its chunked views for a
verification hash moved work outside the scan and previously obscured the executor improvement.
Over 100 alternating scan-only iterations, Q00 measured 0.164 ms in V1 and 0.207 ms self-paced, a
`1.260x` ratio and a 43 us absolute gap. The final trace has 60 tasks: 22 reads, 22 decodes, eight
predicates, and eight fused select/pack operations. It reports 60 scheduler considerations for 60
admissions and no control-plane warning. The remaining gap is fixed orchestration over only eight
morsels, not mask combination, projection copying, excess I/O, or scheduler rescanning.
An attempted all-match predicate pre-scan regressed self-paced time to 0.230 ms: the existing
bitmap collector uses multiversioned vector code, while the optimistic pre-scan was scalar. That
fast path was removed.

A 20-iteration scan-only sweep after the Q00 changes measured V1/self-paced ratios of `1.305`,
`0.838`, `1.452`, `1.517`, `1.428`, `1.338`, `1.688`, `1.365`, and `0.785` for Q00 through Q08.
The Q00 paths are gated to complete all-true partitions and single-field output, so they do not
add work to the multi-field filtered queries. Q02's trace completed only 56 tasks and spent a
95 us absolute gap mostly on fixed orchestration. Q06 completed 188 tasks: 66 reads, 66 decodes,
24 predicates, 24 selections, and eight packs. Its completion batches averaged 6.92 tasks and its
scheduler considered 1.34 tasks per admission; the remaining regression is task/reactor overhead,
not serialized completion handling or scheduler rescanning.

## Corrected natural-split baseline and projection fusion

The final comparison contract gives V1 only the 115 natural SF10 lineitem chunks and gives 128K
morsels only to self-paced. `SplitBy::Layout` is not a valid substitute because it silently
subdivides wide chunks. Under the corrected contract, the pre-optimization SF10 medians were
12.906/23.668 ms for Q6, 3.183/12.718 ms for Q1, and 2.203/4.443 ms for the V1-friendly shape
(V1/self-paced).

Q6 initially created five predicate tasks for every one of 458 morsels. Two strict bounds read
shipdate and two read discount, so self-paced traversed each of those decoded fields and produced a
mask twice. Planning now intersects compatible predicates on the same field into one strict range
predicate. Predicate tasks fell from 2,290 to 1,374 and aggregate traced predicate latency fell
from about 286 ms to 129 ms. A 21-iteration full SF10 run measured 13.218 ms V1 and 13.073 ms
self-paced.

Multi-field projection previously ran `SelectFlat` once per field and then packed the results. Q1
therefore created 1,832 selection tasks plus 458 packs and applied the same almost-all-true mask
four times. `SelectStruct` now gathers aligned decoded field slices, packs one morsel-local Struct,
and applies the shared selection once. Q1's total task count fell from 3,898 to 2,066 and its full
SF10 self-paced median fell from 12.357 to 9.064 ms. It remains 2.79x slower than V1 because its
nearly non-selective single predicate and 458 morsels still pay substantially more fixed control
cost than 115 natural V1 tasks.

The same projection fusion improved the complete 15-file FineWeb set without output mismatches.
Notable self-paced changes were Q03 7.220 to 6.401 ms, Q04/Q05/Q07 about 5.84 to 5.01 ms, Q06
8.615 to 7.821 ms, and Q08 4.761 to 4.215 ms. FineWeb Q01 measured 5.640 ms V1 and 5.673 ms
self-paced. All configured ClickBench queries were also validated over all 100 shards; they remain
2.19x to 3.26x slower than natural-split V1, showing that fixed per-morsel orchestration is now the
larger cost for their mostly narrow scan shapes.

Three experiments are worth retaining as evidence. Replacing resource-completion scans with
explicit waiter lists reduced completion wake candidates from roughly 421,000 to 7,048 on Q6 but
did not measurably change wall time; it remains as a bounded reverse-dependency lookup and exposed
metric. Moving worker tasks from the separate futures pool onto the Tokio session runtime regressed
Q6 by 7.1%, and sharing offered tasks through `Arc<Task>` regressed it by about 3%; both
runtime/task-representation experiments were reverted.

Q1 also tested an adaptive dense-output lane. After eight predicate observations showed at least
90% survival, it constructed the exact lazy `FilterArray` and Struct output inline in the reactor
instead of offering `SelectStruct` to the worker pool. This preserved masks, I/O, cache behavior,
and 128K morsels, but regressed the 31-iteration SF10 Q1 median from about 9.0 ms to 10.489 ms
(roughly 16%). Slicing fields, assembling cross-resource chunks, and building the exact selection
mask are cheap enough to make worker-task overhead visible, but expensive enough to stall the
single reactor. The experiment was reverted. A viable dense path must retain parallel execution,
for example by submitting adjacent sealed dense morsels as one worker batch and distributing its
results back to the original morsel slots.

A second experiment batched adjacent `SelectStruct` operations onto one worker submission after
the same dense-demand signal. It retained exact per-morsel masks and outputs, but did not improve
Q1 (about 9.588 ms versus 9.453 ms in the paired run, and 9.459 ms alone), so it was reverted.
Changing projection speculation from adaptive to eager was likewise neutral: Q1 measured about
9.005 ms adaptive and 9.107 ms eager. Neither worker submission count nor projection-read waiting
is therefore the dominant remaining Q1 cost.

All authoritative comparisons are process-pinned with `taskset -c 0-15`, in addition to setting
execution concurrency to 16. Earlier runs without CPU affinity are retained only as diagnostics.
The original shared futures executor created 96 worker threads on this host even though admission
was capped at 16. The executor is now reused per configured concurrency and creates exactly 16
workers for these comparisons. Under 16-core affinity this reduced SF10 Q1 from 9.454 to 9.288 ms
and improved the tested TPCH cases by roughly 1-3%, but the remaining Q1 gap to natural-split V1 is
still about 2.76x.

The timed self-paced path also cloned the complete immutable `SourcePlan` on every execution,
including every chunk, serialized flat encoding context, field name, and range. Execution now
borrows the plan and copies only the resource state it must own; source-specific byte estimates are
filled into that new execution state. This does not retain decoded arrays, masks, or scan results.
In a 51-iteration, 16-core-pinned SF10 Q1 comparison, the retained old binary measured 9.511 ms
self-paced and 3.311 ms V1; the concurrency-sized pool plus borrowed plan measured 9.196 ms
self-paced and 3.287 ms V1, improving self-paced by 3.3% and the ratio from 2.873x to 2.797x.

## Real-file split audit

The restricted benchmark's earlier "natural" boundaries were the chunks of its hand-built
`Struct(Chunked(Flat))` fixture. They are not the physical splits produced by the default Vortex
writer. A raw `LayoutReader::register_splits` audit, performed before `SplitBy::Layout` can insert
its own 100K-row subdivisions, measured the actual written files. Morsels were formed by greedily
combining adjacent whole natural splits up to 131,072 rows and were never allowed to cut a split.

- TPCH SF10 lineitem has 59,986,052 rows and 7,323 all-field physical splits. The Q1, Q6, and
  single-quantity query masks each expose 458 natural spans of 86,148 to 131,072 rows, so they
  produce 458 128K-target morsels.
- All 100 ClickBench files have 99,997,497 rows and 19,599 all-field physical splits. Most audited
  query masks produce 800 morsels, eight per file, ranging from 79,993 to 131,072 rows.
- ClickBench Q01 and Q07 are exceptions. Their single `AdvEngineID` input exposes only two natural
  spans per file, ranging from 473,209 to 524,288 rows. Preserving real splits produces 200 large
  morsels, not 800 128K morsels. Fixed 128K row slicing would cut 600 physical spans across the
  dataset and must not be described as natural-split rollup.
- Only one FineWeb Vortex file is currently written: `sample.vortex`, with 1,046,615 rows. Its nine
  audited query masks produce eight morsels of 129,111 to 131,072 rows. The complete 15-file,
  14,868,862-row FineWeb results elsewhere in this document use the restricted Parquet-derived
  fixture and are not evidence about the unwritten files' physical split distribution.

Consequently, 128K is a target rather than an invariant when morsels preserve physical leaves. A
natural span wider than the target must remain one larger morsel. The previous fixed-row benchmark
still measures executor overhead, but it is not a real-layout end-to-end comparison.

### Split-count rollup comparison

The follow-up replaced the row target with file-local split-count rollups. A self-paced morsel is
the complete row range covered by 16 or 32 adjacent query-visible natural splits; the final morsel
in each file takes the remainder. V1 receives every unmerged natural split. Both engines run with
`min(16, self_paced_morsel_count)` workers and the process is pinned with `taskset -c 0-15`.
Morsels may cross physical chunks within a file but never cross source files.

The data was TPCH SF10 lineitem (59,986,052 rows), all 100 ClickBench shards (99,997,497 rows), and
all 15 FineWeb shards (14,868,862 rows). The previously missing 14 FineWeb Vortex files were
written with the default `WriteStrategyBuilder` before collecting their raw boundaries. Timings
below are median milliseconds from 11 alternating iterations for TPCH and five for ClickBench and
FineWeb. Ratios are self-paced divided by V1.

| Workload | Natural splits | Morsels 16 / 32 | V1 ms 16 / 32 | Self-paced ms 16 / 32 | Ratio 16 / 32 |
|---|---:|---:|---:|---:|---:|
| TPCH Q6 | 458 | 29 / 15 | 15.167 / 15.316 | 10.902 / 9.122 | 0.719 / 0.596 |
| TPCH Q1 | 458 | 29 / 15 | 6.098 / 6.092 | 4.101 / 4.076 | 0.672 / 0.669 |
| TPCH friendly | 458 | 29 / 15 | 3.401 / 3.264 | 2.422 / 2.394 | 0.712 / 0.733 |
| Click selective | 740 | 100 / 100 | 6.745 / 6.543 | 5.661 / 5.591 | 0.839 / 0.854 |
| Click dashboard | 908 | 100 / 100 | 12.568 / 12.923 | 9.220 / 8.339 | 0.734 / 0.645 |
| Click Q00 | 800 | 100 / 100 | 5.469 / 5.514 | 4.103 / 4.122 | 0.750 / 0.748 |
| Click Q01 | 200 | 100 / 100 | 4.120 / 4.058 | 4.243 / 4.238 | 1.030 / 1.044 |
| Click Q02 | 800 | 100 / 100 | 6.916 / 6.867 | 4.733 / 4.711 | 0.684 / 0.686 |
| Click Q03 | 908 | 100 / 100 | 5.963 / 5.972 | 4.372 / 4.345 | 0.733 / 0.728 |
| Click Q04 | 908 | 100 / 100 | 5.854 / 5.866 | 4.390 / 4.321 | 0.750 / 0.737 |
| Click Q05 | 800 | 100 / 100 | 5.575 / 5.560 | 4.528 / 4.339 | 0.812 / 0.780 |
| Click Q06 | 800 | 100 / 100 | 5.545 / 5.572 | 4.387 / 4.391 | 0.791 / 0.788 |
| Click Q07 | 200 | 100 / 100 | 3.980 / 4.038 | 4.382 / 4.351 | 1.101 / 1.077 |
| Click Q08 | 908 | 100 / 100 | 7.388 / 7.514 | 4.789 / 4.707 | 0.648 / 0.626 |
| Click Q09 | 908 | 100 / 100 | 10.504 / 10.626 | 5.272 / 5.433 | 0.502 / 0.511 |
| Click Q39 | 1,316 | 110 / 100 | 14.359 / 14.170 | 8.219 / 9.040 | 0.572 / 0.638 |
| Click Q40 | 1,316 | 110 / 100 | 8.454 / 8.088 | 6.710 / 6.755 | 0.794 / 0.835 |
| Click Q41 | 1,048 | 100 / 100 | 7.103 / 7.176 | 9.105 / 9.170 | 1.282 / 1.278 |
| Click Q42 | 800 | 100 / 100 | 5.478 / 5.475 | 9.501 / 9.705 | 1.734 / 1.773 |
| FineWeb Q00 | 1,823 | 116 / 59 | 8.196 / 7.625 | 2.116 / 1.702 | 0.258 / 0.223 |
| FineWeb Q01 | 2,527 | 168 / 86 | 66.703 / 67.131 | 6.077 / 4.559 | 0.091 / 0.068 |
| FineWeb Q02 | 1,823 | 116 / 59 | 12.996 / 14.010 | 2.253 / 1.758 | 0.173 / 0.125 |
| FineWeb Q03 | 1,823 | 116 / 59 | 17.920 / 17.960 | 4.728 / 3.733 | 0.264 / 0.208 |
| FineWeb Q04 | 1,823 | 116 / 59 | 16.736 / 16.841 | 3.608 / 2.946 | 0.216 / 0.175 |
| FineWeb Q05 | 1,823 | 116 / 59 | 16.084 / 15.954 | 3.655 / 3.003 | 0.227 / 0.188 |
| FineWeb Q06 | 1,823 | 116 / 59 | 18.912 / 18.749 | 5.673 / 4.591 | 0.300 / 0.245 |
| FineWeb Q07 | 1,823 | 116 / 59 | 16.322 / 16.307 | 3.654 / 3.055 | 0.224 / 0.187 |
| FineWeb Q08 | 2,527 | 168 / 86 | 21.124 / 19.001 | 3.847 / 2.894 | 0.182 / 0.152 |

The only case with fewer morsels than the 16-worker cap was TPCH at merge 32: 15 morsels, so both
engines used 15 workers. ClickBench is usually one morsel per physical file after either rollup;
Q39 retains 110 morsels at merge 16. FineWeb retains at least 59 morsels. Self-paced wins every
TPCH and FineWeb case and 12 of 16 ClickBench shapes. ClickBench Q01, Q07, Q41, and Q42 remain
slower; Q42 is the largest regression at 1.73-1.77x.

These timings isolate execution-grain effects using real default-writer boundary distributions,
but both engines still execute the restricted in-memory `Struct(Chunked(Flat))` fixture. They are
not compressed-file end-to-end I/O timings.

## Architectural findings

The experiment supports these decisions:

- Keep the immutable source plan separate from mutable per-scan execution state.
- Keep reusable segment and decoded-array resources scan-wide, but demand and transformed arrays
  morsel-local.
- Carry boolean length, true count, and a shared bit-buffer view with every resolved mask.
  Scheduling and sealing can inspect whole-mask scalars and cache exact resource-range counts
  without canonicalizing arrays.
- Make offers descriptive and claim them into immutable input snapshots with leases. Revocation
  remains safe, and workers never access the mutable slot store.
- Transport promotion and revocation updates in addition to offers. An external scheduler can
  retain an offer after its necessity changes.
- Track possible users, joined users, and task leases separately. They answer different lifetime
  questions and allow retirement without a scan-wide graph walk.
- Bound `advance` by cheap transitions and expose work externally. Data-plane work does not belong
  in the reactor transition loop.

The experiment also revealed costs that should not automatically move into a production object:

- `BTreeMap` and `BTreeSet` favor determinism and inspection over hot-path efficiency;
- extensive trace strings and metrics enlarge the execution object and add branches;
- a full materialized boolean demand remains necessary for the evaluator, although sharing the
  all-true instance removes repeated initialization; and
- deduplicating only by `SegmentId` assumes a single segment source. A production key must include
  source identity.

These are acceptable at this highly restricted experiment boundary. They should be measured or
replaced before treating the module as production machinery.

## Speculative I/O admission

Unsealed reads are now visible to the scheduler as candidate work. Reads needed by the next
predicate, and projection reads after demand seals nonempty, are promoted to required work. The
scheduler independently configures predicate and projection candidates as disabled, eager, or
adaptive. Adaptive admission uses the current demand row count multiplied by observed or prior
survival rates for predicates that still have to run.

Admission has a byte budget. File and in-memory segment sources report exact segment sizes when
they know them; wrappers forward the estimate. A source that cannot estimate a segment returns
`None`, and the scheduler charges the configured conservative unknown-read size. Setting that
charge to zero rejects unknown-size speculative reads. Required reads never consume the
speculative budget.

The comparison benchmark accepts these controls:

- `VORTEX_SELF_PACED_SPECULATIVE_IO=off|predicate|projection|adaptive|predicate-eager|projection-eager|eager`
- `VORTEX_SELF_PACED_SPECULATIVE_IO_MAX_BYTES`, defaulting to 64 MiB
- `VORTEX_SELF_PACED_SPECULATIVE_IO_UNKNOWN_BYTES`, defaulting to 8 MiB
- `VORTEX_SELF_PACED_SPECULATIVE_IO_MIN_ROWS`, defaulting to 1 row

Metrics report candidate offers and admissions, known estimated bytes, unknown-size offers,
completed physical bytes, and the completed bytes later proved useful or wasted. Predicate and
projection offer counts are separate; a physical read used by both is counted once in byte
metrics. Trace output records each admitted read's phase, estimate, byte charge, current demand,
and expected surviving rows.

A five-iteration FineWeb follow-up showed why admission must consider projection width and
selectivity, not merely whether expected output is nonzero. On Q06, all 37.1 MB admitted early
were eventually required. On Q01, only 9.6 MB of 19.2 MB admitted early became required;
self-paced returned 49.2 MB from the source versus V1's 30.8 MB. With the default one-row
threshold, adaptive read-ahead improved a few latency-hiding cases but regressed most of the
sub-millisecond suite. This is evidence for a cost/benefit admission score, not for enabling the
current adaptive default broadly.

## Real natural-split rollup comparison

A later comparison replaced fixed 128K morsels with morsels formed by merging 16 or 32 consecutive
natural splits from the real benchmark Vortex files. The source catalogs contain 99,997,497
ClickBench rows in 100 files, 59,986,052 SF10 lineitem rows, and 14,868,862 FineWeb rows in 15
files. Split boundaries are query-specific unions over only the physical fields read by that query.
Merging restarts at every file boundary.

One initially collected result was invalid. It applied boundaries from the production Vortex files
to an unrelated coarse in-memory layout. V1 then evaluated several exact splits against the same
coarse segment and repeatedly decoded it, while self-paced retained the segment scan-wide. That
artifact produced implausible 7-14x FineWeb gains. Those measurements are rejected.

The corrected harness writes one complete Vortex byte buffer with a restricted
`Struct<Chunked<Flat<i64>>>` strategy, freezes it, and reopens it through `vortex-file`. The writer
edition permits only `vortex.primitive`; the strategy rejects nullable roots and every field type
other than non-nullable `i64`, so unsupported encodings and layout strategies cannot silently enter
the fixture. `SourcePlan::try_from_layout` independently validates the reopened footer, including
aligned field-chunk boundaries. A single-chunk field retains its `Chunked` wrapper.

Both executors scan the same reopened layout and `SegmentSource`, and the harness prints the exact
serialized byte length and a stable byte hash. Each comparison also materializes its query bundle
once and clones that same bundle into both execution paths. V1 receives every natural interval
unchanged. Self-paced alone receives unions of 16 or 32 intervals. Both are pinned with
`taskset -c 0-15`, use concurrency `min(16, morsel count)`, have speculative I/O disabled, validate
ordered output hashes before timing, and alternate execution order for 20 measured iterations.
Fixture construction, serialization, reopening, and rechunking are outside timing.

At merge factor 16, self-paced won 3 of 28 workloads. Its unweighted geometric-mean time ratio was
`1.463`, or 46.3% slower overall:

| Suite | Workloads | Self-paced wins | Geometric mean self-paced/V1 |
| --- | ---: | ---: | ---: |
| ClickBench | 16 | 1 | 1.498 |
| TPC-H | 3 | 2 | 1.093 |
| FineWeb | 9 | 0 | 1.545 |
| Combined | 28 | 3 | 1.463 |

At merge factor 32, self-paced won 2 of 28 and the combined geometric-mean ratio worsened to
`1.706`. The suite ratios were `1.504` for ClickBench, `1.328` for TPC-H, and `2.320` for FineWeb.

TPC-H illustrates the useful tradeoff. Its 458 query-relevant natural intervals become 29 morsels
at merge 16 and 15 at merge 32. Merge 16 measured Q6 at `15.940/14.886 ms` (`0.934x`), Q1 at
`6.864/10.491 ms` (`1.528x`), and the V1-friendly scan at `3.519/3.225 ms` (`0.916x`). Merge 32
caps both engines to 15-way concurrency and regresses Q6 to `1.155x` and Q1 to `2.174x`; reducing
control units did not compensate for lost parallelism and larger cross-chunk assembly work.

ClickBench usually has fewer than 16 relevant intervals per file, so both merge factors stop at one
morsel per file. Merge 16 is effectively tied on Q00 (`0.997x`) and slower on the other 15 shapes.
The largest ratios are Q39 `1.756x`, Q40 `2.315x`, Q41 `2.395x`, and Q42 `2.113x`. For Q39 and
Q40 only, merge 32 reduces 110 morsels to 100 and makes both slower (`1.872x` and `2.449x`).

FineWeb has 1,823 or 2,527 query-relevant natural intervals. Merge 16 creates 116 or 168 morsels
and ranges from `1.036x` on Q02 to `2.460x` on Q06. Merge 32 creates 59 or 86 morsels but is slower
on every query, ranging from `1.587x` to `4.294x`. In this restricted executor, fewer larger morsels
increase the number of physical resource slices assembled by each task and reduce opportunities to
schedule independent morsels. Natural-split rollup therefore needs a byte/work-aware target; a
fixed count of 32 is not a generally better aggregation policy.

## What remains unknown

The current results do not establish performance for compressed production encodings, unaligned
field chunks, nullable or non-`i64` arrays, arbitrary expressions, dynamic filters, object-store
latency, realistic byte-budget backpressure, stealing, or multi-source segment identity. They also
do not measure time to first batch or peak memory in the final real-data sweep.

The next useful experiment is a production-shaped scan prototype that preserves the proven
contracts while replacing deterministic maps, trace strings, and fixed experiment operations with
the real plan and scheduler interfaces. Its gate should include equal output, physical I/O,
first-batch latency, peak resident memory, and CPU occupancy, with 128K retained as one point in a
morsel-size sweep rather than a default.
