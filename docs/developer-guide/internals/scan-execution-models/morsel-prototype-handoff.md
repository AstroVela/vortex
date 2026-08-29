# Morsel Prototype: Handoff

Everything needed to rerun and interpret the morsel-executor evaluation. The original prototype is
on branch `claude/morsel-executor-prototype-vvrscx`; the current three-reader comparison and latest
optimizations are on `ji/morsel-layout27-comparison`.

The latest measurements used a 16-core/32-thread Intel Xeon 6975P. Compute-only results use CPUs
0–15 (one hardware thread per physical core). File-backed results use all 32 logical CPUs because
SMT hides file-driver latency on this machine.

## Latest comparison branch and optimizations (2026-08-28)

The comparison harness now runs the V1 `LayoutReader`, layout-v27, and the preferred morsel design
over the same generated segments and verifies identical dtype, row count, and ordered content before
reporting timings. Layout-v27 was ported only for study and benchmarking; the morsel implementation
does not depend on its executor.

The signed-off comparison checkpoint is commit `82bf78284` on
`ji/morsel-layout27-comparison`. A push was attempted from the measurement host, but that host has
neither HTTPS GitHub credentials nor the `gh` executable, so publication remains pending from an
authenticated checkout.

The accumulated morsel optimizations on this branch are:

- worker threads and reusable thread-local arenas are scan-owned, created before the first timed
  execution, and retained across warm-up and repeated runs;
- one affinity-owned morsel remains active per worker, with no migration of partial node state;
- plan-driven required and speculative reads share one scan-level background I/O service, while
  exact ticket completion wakes only the suspended execution that needs it;
- file-backed `execute` methods consume completed requests and perform no file syscalls; immutable
  in-memory segments retain the zero-wait inline lookup;
- scan-wide raw cells and lease-counted decoded cells deduplicate reads and decoding across morsels;
- read-time morsels are 131,072 rows by default without changing the normal 8192-row/1 MiB write
  pipeline;
- bounded 64 KiB/4 MiB local-file coalescing avoids both tiny reads and excessive gap amplification;
- top-level conjuncts referencing the same column now share one column input and one `PredicateExec`
  plan node. The column is decoded once and its predicates progressively refine one mask;
- conjunct order is adaptive across morsels owned by the same worker. Independently decoded groups
  learn active execution time per rejected row from their first complete morsel, while predicates
  sharing one decoded column reorder only when the winner is selective enough to cross the existing
  20% sparse-evaluation threshold. Parallel conjunct mode retains expression order;
- learned orders reuse worker-arena vectors and stop collecting observations after the first sample,
  so steady-state morsels allocate no ordering state and take no ordering clocks;
- sparse same-column ranges can use a normalized fused expression, while dense ranges retain the
  original encoding-aware comparisons. This recovers Q6's range win without the decimal `Between`
  regression observed on Q19;
- a single-column predicate binds directly to the column array, avoiding a redundant one-field
  `StructExec` plus `GetItem` evaluation.
- each worker creates one array `ExecutionCtx` per run and reuses it across every morsel, rather
  than snapshotting the session kernel registry once per execution quantum;
- sparse rank refinement composes an already-cached base index vector with a sufficiently sparse
  bitmap rank mask directly. The resulting mask keeps its indices cached for the projection.

The latest SF=1 in-memory run used all 32 logical CPUs, one warm-up, and five grouped samples. These
numbers are a current-branch smoke comparison, separate from the controlled x16 physical-core table
below.

| query | V1 Tokio x32 | layout-v27 x32 | morsel x32/128k | morsel vs V1 |
|---|--:|--:|--:|--:|
| Q6 | 2.160 ms | 5.294 ms | 1.767 ms | 1.22x |
| Q1 | 1.801 ms | 3.920 ms | 1.234 ms | 1.46x |
| Q14 | 1.353 ms | 3.897 ms | 1.013 ms | 1.34x |
| Q15 | 1.446 ms | 3.673 ms | 1.036 ms | 1.40x |
| Q12 | 2.734 ms | 6.148 ms | 2.016 ms | 1.36x |
| Q19 | 6.368 ms | 7.444 ms | 1.621 ms | 3.93x |
| scan-6col | 1.398 ms | 2.824 ms | 0.730 ms | 1.92x |
| selective | 1.619 ms | 5.052 ms | 1.208 ms | 1.34x |

For Q6 specifically, grouping reduced logical decode consumers from 322 to 230 while physical work
remained 161 reads and 34,559,892 segment bytes. Its single-worker median improved from 24.58 ms
before same-column range fusion to 21.08 ms. Q19 remained on the dense encoding-aware path at
13.09 ms on one worker and 1.62 ms at x32/128k instead of regressing to roughly 25 ms with
unconditional decimal range fusion.

### Follow-up profiling and rejected shortcuts

Two small follow-up cleanups remain useful even though they are not headline speedups. Lease counts
depend only on the immutable plan and morsel cut, so they are computed with the other scan preparation
before timing. Operators in one execution quantum share one array `ExecutionCtx` instead of
repeatedly snapshotting the same session kernel registry for each predicate. The latest follow-up
moves that context one level further out: each worker now creates one per run and reuses it across
all of its morsels.

An exact comparison against `82bf78284` measured Q12 at 19.364 ms before and 19.247 ms after on one
physical core. A bare six-column scan moved from 0.702 to 0.698 ms. Both differences are below 1%
and are not measurable wins; treat these changes as orchestration cleanup, not throughput claims.

The follow-up Q12/128k profile is `/tmp/q12-current-lazy-inline.profile.json.gz` on the measurement
host. It was recorded while testing the subsequently rejected lazy-inline scheduler variant, so it
is evidence for hotspot selection rather than a profile of the final branch byte-for-byte. Resolving
its hottest worker addresses identifies FastLanes FOR/bitpacking decompression, vectorized `i32`
comparisons, and mask-index construction as the remaining compute work. Open it with:

```bash
samply load /tmp/q12-current-lazy-inline.profile.json.gz
```

Two broader scheduler shortcuts were tested and rejected:

- avoiding all scheduler work for immediately available in-memory tickets changed Q12 by less than
  measurement noise and improved scan-6col by only 1.6%, while introducing a second fallback
  submission path and less clear I/O-batch accounting;
- moving Q12's grouped receipt-date range ahead of its two-column comparisons made no measurable
  difference;
- merging conjuncts merely because their field sets overlap reduced Q12 logical consumers from 460
  to 368 but regressed x16/128k from roughly 1.72 to 2.11 ms, because one eager three-column input
  destroyed the useful cascade boundary.

The rule therefore remains deliberately narrower: conjuncts share one `PredicateExec` only when
each conjunct's complete field set is the same single column.

### Current worker CPU overhead and retained hot-path changes

A symbol-rich Samply recording of 2,000 SF=1 Q12/128k runs is
`/tmp/q12-adaptive-retained.profile.json.gz`. Its build ID
`9566b6fffeb3f7b6e37e3228814483bd19293793` exactly matches the recorded binary. Because the old
benchmark recreated its prepared pool for every run, the profile contains 31,968 short-lived
`vortex-morsel-*` threads. Aggregating all of those workers, rather than selecting one thread ID,
accounts for 48.47 seconds of sampled worker CPU.

The approximate self-CPU breakdown, obtained by resolving the 2,500 hottest application addresses,
is:

| worker CPU category | share |
|---|--:|
| FastLanes decode | 25.1% |
| predicate comparison kernels | 21.9% |
| other array compute, led by rank-mask intersection | 11.9% |
| mask index/slice materialization | 8.4% |
| allocation and other memory copies | 6.5% |
| projection gather/filter copies | 6.4% |
| morsel scheduler and node dispatch | 3.3% |
| other resolved application code | 2.2% |
| unresolved/shared-library/long-tail frames | 14.3% |

This answers the main overhead question: decode and comparisons are almost half the worker CPU;
mask representation changes plus rank intersection are the largest avoidable handoff; dynamic
morsel scheduling itself is small. Q12 records zero pending I/O polls, zero blocked morsels and zero
asynchronous wait, so none of this hot-run profile is storage waiting.

Three changes/experiments followed from that evidence:

1. `ExecutionCtx` is now worker-local for a run. Four alternating SF=1 x16/8192-row comparisons
   moved the aggregate median from 1.798 to 1.779 ms (1.1%). At 128k rows the observed difference
   was only 1.696 to 1.691 ms. Retain this as low-risk setup removal, not a large throughput claim.
2. The benchmark now keeps a prepared `MorselScan`, worker threads, and affinity-owned arenas alive
   across warm-up and every sample. Runs over one scan are serialized because one pool drives one
   scheduler at a time. Cold file runs still call `POSIX_FADV_DONTNEED` and reset physical-read
   counters before every sample. Worker creation was already outside the reported wall interval,
   so this primarily fixes real end-to-end lifecycle cost and profile noise rather than rewriting
   old table medians.
3. Q12's final sparse predicate enters rank space with roughly 12% of rows and leaves roughly 2%
   overall. Its base indices have already been materialized to filter the predicate input. The new
   path enumerates the short bitmap rank mask and maps those ranks through the cached base indices,
   producing an index-backed result for projection. A pinned single-core microbenchmark at 131,072
   rows measured 6.415 us before versus 4.567 us after, a 1.40x speedup (28.8% less time) for that
   exact operation. The committed `cached_base/q12_shape` benchmark preserves this case.
4. Dense bitmap rank intersection now dispatches to an AVX-512 VPOPCNTDQ kernel when the host also
   has BMI2. It computes eight base-word popcounts together, then uses scalar BMI2 PDEP per lane;
   x86 has no vector bit-level PDEP. Two pinned release runs measured stable kernel speedups of
   1.49-1.50x at 65,536 rows, 1.50-1.51x at 131,072 rows, and 1.52-1.53x at 1,048,576 rows. The
   existing sparse cached-index path remains preferable for Q12's final refinement, so this is a
   dense-bitmap improvement rather than a claim of an end-to-end Q12 gain.

Lowering the dense-to-sparse expression threshold from 20% to 10% was rejected. It avoided the
rank-space handoff but forced Q12's final comparison over the full array and regressed three
alternating SF=1 x16 runs from about 1.69 ms to 1.80 ms, roughly 6.5%. The sparse evaluation choice
is correct; the representation transition was the part to optimize.

### Adaptive conjunct ordering

Ordering observations are worker-local and survive `reset` when that worker advances to its next
morsel. A morsel and its partial node state still never migrate. Cross-group timing covers only
active `execute` calls and accumulates across suspension, so background I/O wait time is not charged
as predicate CPU cost. A worker keeps expression order until every group has one observation; this
avoids speculative exploration changing short-circuit/decode accounting when the original first
group already empties the morsel.

Two broader versions were rejected while implementing this:

- scan-wide atomic observations caused cache-line contention between workers;
- allocating and cloning new order vectors for every morsel regressed Q6 even when no order changed.

The retained implementation uses persistent vectors in each worker arena and observes each
predicate once. On 16 physical cores with 128k-row morsels, 101-iteration SF=1 medians were neutral:
Q6 was 1.695 ms fixed versus 1.687 ms adaptive with no reorder, and Q12 was 1.702 versus 1.699 ms
with cross-group reordering in 30 of 46 morsels. SF=10 gives learning more runway: Q12 improved from
14.679 to 14.396 ms (1.9%) over 458 morsels, with identical 2,061 physical reads and 393,141,420
segment bytes; 442 morsels used a learned group order.

Q12 is currently the only TPC-H scan shape that changes order. Its groups are
`0 = commitdate < receiptdate`, `1 = shipdate < commitdate`, and
`2 = receiptdate in [1994-01-01, 1995-01-01)`. The learner changes the original `[0, 1, 2]` into
`[1, 0, 2]`. Forcing all six orders at SF=10, five iterations each, confirmed that the selected
order is the best measured permutation:

| Q12 group order | median | relative to best |
|---|--:|--:|
| `[1, 0, 2]` | 14.275 ms | — |
| `[0, 1, 2]` | 14.819 ms | 1.038x |
| `[2, 1, 0]` | 14.834 ms | 1.039x |
| `[2, 0, 1]` | 15.125 ms | 1.060x |
| `[1, 2, 0]` | 15.312 ms | 1.073x |
| `[0, 2, 1]` | 15.856 ms | 1.111x |

Every order issued the same 2,061 physical reads and 393,141,420 bytes and reproduced V1 exactly.
This isolates the difference to compute order and decode reuse, not I/O volume.

The current score nevertheless mixes populations and should not be treated as a general correlated-
predicate optimizer. On the learning morsel, group 0 was observed at 82,841/131,072 survivors,
group 1 at 15,519/82,841 survivors after group 0, and group 2 at 2,425/15,519 survivors after both.
Group 1's independently materialized dense mask actually had 63,750/131,072 survivors. Comparing
its conditional 18.7% pass rate directly with group 0's unconditional 63.2% rate is invalid.

For two candidates `A` and `B` after the same prefix mask `S`, compare
`cost(A | S) / (1 - P(A | S))` with `cost(B | S) / (1 - P(B | S))`. Both probabilities must use
the same `S`. A robust learner should evaluate every group once against the same learning-morsel
input, retain the individual masks and costs, derive survivor counts for each candidate prefix by
mask intersection, and enumerate small predicate sets (or use subset dynamic programming). The cost
model must also distinguish dense and sparse evaluation because the 20% transition, fixed decode
cost, and mask materialization make cost non-linear in row count. Q12's chosen swap is empirically
right, but the present heuristic reaches it with evidence that is not generally comparable.

## 1. Run it

```bash
git fetch origin ji/morsel-layout27-comparison
git checkout ji/morsel-layout27-comparison
cargo build --release -p vortex-morsel --features _test-harness --bins
cargo test -p vortex-morsel

# Compute only: immutable in-memory segments, one worker per physical core.
taskset -c 0-15 ./target/release/tpch-eval 1

# Real file reads, hot and advisory-cold, using SMT to hide I/O latency.
TPCH_DISK_PATH=target/tpch-morsel-sf1.segments TPCH_CACHE_MODE=hot \
  taskset -c 0-31 ./target/release/tpch-eval 1
TPCH_DISK_PATH=target/tpch-morsel-sf1.segments TPCH_CACHE_MODE=cold \
  taskset -c 0-31 ./target/release/tpch-eval 1

# Focus one query or report only the primary 128k morsel row.
TPCH_QUERY=Q12 TPCH_ITERATIONS=31 ./target/release/tpch-eval 1
TPCH_QUERY=Q12 TPCH_MORSEL_ONLY=1 TPCH_ITERATIONS=31 \
  ./target/release/tpch-eval 1

# Thread and morsel-size sweeps for the memory backend.
TPCH_SWEEP=1 ./target/release/tpch-eval 1
```

TPC-H is generated in-process; there are no downloads. SF=10 wants roughly 24 GB. Relevant knobs
are `TPCH_SCALE`, `TPCH_ROW_BLOCK` (default 8192), `TPCH_BLOCK_BYTES` (default 1 MiB and the first
write-side knob to vary for more natural splits), `TPCH_DISK_PATH`,
`TPCH_CACHE_MODE={hot,cold}`, `TPCH_QUERY`, `TPCH_ITERATIONS`, `TPCH_MORSEL_ONLY=1`,
`TPCH_COALESCE_DISTANCE`, `TPCH_COALESCE_MAX_BYTES`, and `MORSEL_EVAL_ROWS`.

The primary read-side morsel is 131,072 rows. This does not rewrite or repartition the file. The
file still uses the normal 8192-row write repartition and 1 MiB write coalescing target; 128k is
only a read-time row-range cut.

At SF=1 the segment pack contains 1,789 logical segments and 174,410,852 payload bytes. Segment
sizes are 3,780/102,396/393,492 bytes min/median/max. The aligned pack is a benchmark payload file,
not a complete Vortex file with a footer.

## 2. Fairness and correctness

Before timing, every configuration is compared with V1 on dtype, row count, and ordered content.
A mismatch aborts the run. The 33 `vortex-morsel` tests include differential coverage against the
V1 `LayoutReader` and scheduler/I/O regressions.

The executor rejects unsupported layouts at plan-build time. Each benchmark configuration prepares
one scan and retains its worker threads and reusable arenas across warm-up and every timed sample.
Memory configurations get an independent warm-up and are sampled as grouped steady-state runs.
File configurations alternate readers and report median plus min/max; advisory-cold runs still
evict before every sample.

Statistics pruning is disabled for V1 until morsel execution implements the same pruning. Zone maps
and dictionary layout are disabled for both paths, so the comparison measures execution rather
than giving only V1 a pruning capability.

## 3. Execution and I/O model

There is one affinity-owned active morsel per worker. Its arena and partial operator state never
migrate. Each worker reuses one arena across the scan. Planning registers keyed cells and divides
work into shared required and speculative queues. While its morsel is suspended, a worker polls
I/O work. Exact ticket completion wakes only the waiting continuation; stale generation/epoch
wakes are ignored.

File sources advertise background reads. Planning therefore creates and submits every file future
before execute can consume its ticket: the file-backed execute path makes no `open`, `pread`,
`preadv2`, or other system call. An unfiltered scan exposes its complete exact segment set before
workers start. A filtered scan exposes the initial active window—one morsel per worker—which is the
same speculative depth worker-local planning immediately names anyway. Later morsels remain
demand-driven. This improves coalescing visibility without migrating state or increasing lookahead.

In-memory sources retain the inline path. Their `request_nowait` is an immutable buffer lookup, not
a system call, and they never enter the background queue or report blocked morsels.

The local-file coalescing window is 64 KiB/4 MiB (distance/maximum range), changed from 1 MiB/4 MiB
after the larger gap repeatedly amplified bytes. The file request stream also defers dispatch for
one bounded cooperative turn after registration. On `scan-6col`, exact whole-plan lookahead reduces
169 physical reads to 16, versus V1's 18, while reading exactly 59,933,416 bytes.

## 4. In-memory results

These are SF=1 medians for V1 Tokio and the primary 128k morsel using CPUs 0–15, one warm-up, and
15 grouped samples. The immutable source performs no file opens, syscalls, page-cache probes,
background reads, or I/O waits.

| query | V1 x16 | morsel x16/128k | speedup |
|---|--:|--:|--:|
| Q6 | 2.756 ms | 1.888 ms | 1.46x |
| Q1 | 2.253 ms | 0.954 ms | 2.36x |
| Q14 | 1.567 ms | 0.895 ms | 1.75x |
| Q15 | 1.634 ms | 0.868 ms | 1.88x |
| Q12 | 2.830 ms | 1.802 ms | 1.57x |
| Q19 | 6.795 ms | 1.336 ms | 5.09x |
| scan-6col | 1.550 ms | 0.645 ms | 2.40x |
| selective | 1.742 ms | 1.047 ms | 1.66x |

The SMT question is settled for this compute-only workload. Focused Q12/128k medians were 1.784 ms
on 16 physical workers, 2.198 ms on 23 workers, and 2.166 ms on all 32 SMT threads. Use one worker
per physical core for memory scans on this host. File scans move the other way: x32 was consistently
faster than x16 because workers can poll background I/O while their morsels wait.

Samply showed roughly 9.3% of focused Q12 worker CPU in timed per-scan arena/node teardown. Reusing
one thread-local arena per worker moved that work outside the timer and improved Q12 by about 6–9%.
The profile is `/tmp/q12-memory-128k.profile.json.gz` on the measurement host and can be opened with:

```bash
samply load /tmp/q12-memory-128k.profile.json.gz
```

## 5. Local-file hot and cold results

This complete SF=1 matrix uses all 32 logical CPUs, 128k-row morsels, 15 alternating samples, and
the best V1 row (`LayoutReader` on Tokio x32). Bytes are successful reader bytes after coalescing,
shown as V1/morsel decimal MB.

| query | hot V1 / morsel | hot speedup | hot MB V1/morsel | cold V1 / morsel | cold speedup | cold MB V1/morsel |
|---|--:|--:|--:|--:|--:|--:|
| Q6 | 3.652 / 3.202 ms | 1.14x | 43.87 / 34.56 | 3.909 / 3.197 ms | 1.22x | 44.36 / 34.56 |
| Q1 | 3.326 / 2.990 ms | 1.11x | 39.91 / 39.91 | 3.295 / 3.036 ms | 1.09x | 39.91 / 39.91 |
| Q14 | 3.061 / 2.327 ms | 1.32x | 52.56 / 43.55 | 3.104 / 2.095 ms | 1.48x | 52.56 / 43.55 |
| Q15 | 2.958 / 2.244 ms | 1.32x | 49.56 / 40.55 | 2.827 / 2.155 ms | 1.31x | 49.56 / 40.55 |
| Q12 | 4.498 / 3.675 ms | 1.22x | 40.48 / 39.69 | 4.468 / 3.679 ms | 1.21x | 39.69 / 39.77 |
| Q19 | 9.949 / 4.001 ms | 2.49x | 43.04 / 43.04 | 9.312 / 3.513 ms | 2.65x | 43.04 / 43.04 |
| scan-6col | 3.073 / 1.902 ms | 1.62x | 59.93 / 59.93 | 2.951 / 1.877 ms | 1.57x | 59.93 / 59.93 |
| selective | 3.376 / 2.553 ms | 1.32x | 55.44 / 44.92 | 3.381 / 2.695 ms | 1.25x | 56.73 / 44.92 |

Every primary morsel row beats the best V1 row in both modes. Morsel physical bytes equal named
segment bytes except for 75,572 bytes (0.19%) of cold Q12 coalescing. Q6, Q14, Q15, and the
selective scan read materially fewer bytes than V1 because their filter/projection plan avoids
V1's wider physical ranges; this is not statistics pruning.

`cold` means `POSIX_FADV_DONTNEED` before every reader construction. It is advisory. The current
device is local non-rotational NVMe (`nvme0n1`), not the earlier 125 MiB/s EBS volume. A direct,
cache-bypassing read of the 174,410,852-byte pack completed in 22.9 ms (7.6 GB/s), so millisecond
cold rows and close hot/cold medians are plausible. For strict device-level attribution use an
aligned `O_DIRECT` fixture or block-I/O tracing; successful reader bytes are not a block-device
counter.

Time to first batch is internal readiness because output is collected and reordered at scan end;
it is not yet delivery time to a streaming consumer.

The accounting invariant remains: `decodes + reuses` with sharing equals decode work without
sharing. A dedicated test also proves that 15 straddled segments produce exactly 15 source requests
across four workers when decoded sharing is disabled.

## 6. Scope and remaining work

- Statistics pruning must be added to morsel execution before enabling it for either reader.
- Local file I/O is covered; object-store latency and cancellation are not.
- Output needs a bounded reorder buffer for real consumer-visible streaming.
- Completed raw buffers remain in the scan-wide service until scan teardown; add byte-bounded
  retention without breaking exact-ticket wakeups.
- Q12 is now the smallest 128k in-memory win in the latest all-logical-core smoke run. Profile its
  worker occupancy and mask/predicate work before changing scheduling.
- The one-turn coalescing delay is intentionally bounded. A production implementation should make
  coalescing/admission policy explicit rather than growing an unbounded timer-based window.
- Add real zone-map pruning and rerun V1 and morsel with identical statistics.
- The joins in Q12/Q14/Q15/Q19 are above the scan; only `lineitem` scan work is measured.
- ClickBench and FineWeb still require downloads and remain represented by `morsel-eval` synthetic
  workloads.

## 7. Code map

| path | responsibility |
|---|---|
| `vortex-morsel/src/node.rs` | `ExecNode`, exact wait sets, and retry propagation |
| `vortex-morsel/src/nodes/` | FLAT, CHUNKED, STRUCT, PREDICATE, CONJUNCT, and FILTER nodes |
| `vortex-morsel/src/io.rs` | Scan-wide raw cells and morsel-local ticket views |
| `vortex-morsel/src/cells.rs` | Lease-counted shared decoded cells |
| `vortex-morsel/src/build.rs` | Immutable `ExecPlan` and initial I/O lookahead set |
| `vortex-morsel/src/driver.rs` | Persistent worker affinity, I/O queues, wakeups, and output ordering |
| `vortex-morsel/src/harness.rs` | Reusable prepared scan and three-reader harness entry points |
| `vortex-morsel/src/bin/tpch-eval.rs` | Exactness, persistent benchmark runners, hot/cold backends, counters, and timing matrix |
| `vortex-mask/src/intersect_by_rank.rs` | Rank-mask representation dispatch, cached-index composition, and AVX-512 dense-bitmap intersection |
| `vortex-io/src/read_at.rs` | Reader contract and local/object-store coalescing defaults |
| `vortex-file/src/read/driver.rs` | Physical request batching and coalescing |
| `vortex-file/src/segments/source.rs` | File segment futures and background-read preference |

Design context: [morsel-based plan execution](morsel-based-plan-execution.md),
[graph model](scan-execution-graph-model.md). Related results:
[TPC-H findings](morsel-prototype-tpch-findings.md),
[P1 findings](morsel-prototype-p1-findings.md).
