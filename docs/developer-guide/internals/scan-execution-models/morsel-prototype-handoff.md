# Morsel Prototype: Handoff

Everything needed to rerun and interpret the morsel-executor evaluation. The code is on branch
`claude/morsel-executor-prototype-vvrscx`.

The latest measurements used a 16-core/32-thread Intel Xeon 6975P. Compute-only results use CPUs
0–15 (one hardware thread per physical core). File-backed results use all 32 logical CPUs because
SMT hides file-driver latency on this machine.

## 1. Run it

```bash
git fetch origin claude/morsel-executor-prototype-vvrscx
git checkout claude/morsel-executor-prototype-vvrscx
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
A mismatch aborts the run. The 27 `vortex-morsel` tests include differential coverage against the
V1 `LayoutReader` and scheduler/I/O regressions.

The executor rejects unsupported layouts at plan-build time. Timed runs prepare worker threads and
their reusable arenas before the timer. Memory configurations get an independent warm-up and are
sampled as grouped steady-state runs. File configurations alternate readers and report median plus
min/max.

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
- Q6 is now the smallest in-memory and file-backed win. Profile worker occupancy and predicate
  decode before changing scheduling.
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
| `vortex-morsel/src/nodes/` | FLAT, CHUNKED, STRUCT, CONJUNCT, and FILTER nodes |
| `vortex-morsel/src/io.rs` | Scan-wide raw cells and morsel-local ticket views |
| `vortex-morsel/src/cells.rs` | Lease-counted shared decoded cells |
| `vortex-morsel/src/build.rs` | Immutable `ExecPlan` and initial I/O lookahead set |
| `vortex-morsel/src/driver.rs` | Worker affinity, I/O queues, wakeups, and output ordering |
| `vortex-morsel/src/bin/tpch-eval.rs` | Exactness, hot/cold backends, counters, and timing matrix |
| `vortex-io/src/read_at.rs` | Reader contract and local/object-store coalescing defaults |
| `vortex-file/src/read/driver.rs` | Physical request batching and coalescing |
| `vortex-file/src/segments/source.rs` | File segment futures and background-read preference |

Design context: [morsel-based plan execution](morsel-based-plan-execution.md),
[graph model](scan-execution-graph-model.md). Related results:
[TPC-H findings](morsel-prototype-tpch-findings.md),
[P1 findings](morsel-prototype-p1-findings.md).
