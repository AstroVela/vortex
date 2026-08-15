# DuckDB `io_uring` scan experiment handover

## Goal

Determine whether Vortex's DuckDB scan can use one caller-driven `io_uring` per DuckDB worker
without a background I/O thread, and whether DuckDB CPU work starves I/O completion processing.

The experiment compares:

- synchronous `pread`;
- one caller-driven `io_uring` per DuckDB worker;
- one `io_uring` driven by a dedicated background thread;
- the current V1 Vortex layout reader.

## Implementation

The Linux-only table function `io_bench_scan` is in
`vortex-duckdb/src/io_bench.rs`. It emits two `BIGINT` columns from a deliberately simple raw
columnar file. Each physical block contains all values followed by all payloads. The scan does no
decoding and supports only a fixed row limit supplied at bind time.

The direct benchmark is `benchmarks/duckdb-bench/src/bin/io_uring_scan.rs`. It generates both raw
and uncompressed V1 inputs with identical data and runs:

- light SQL: two sums;
- heavy SQL: grouping plus nested hashes.

Aggregate pushdown is deliberately not registered, ensuring that the extra work remains in
DuckDB. The raw scan enforces the requested row count internally; do not restore an outer SQL
`LIMIT`, because that caused DuckDB to use only one local scan state and invalidated the
per-worker comparison.

Important implementation details:

- aligned I/O buffers are pooled and reused;
- per-worker rings refill in half-window batches;
- caller-driven rings do not enable `DEFER_TASKRUN`;
- the dedicated-thread ring does enable `DEFER_TASKRUN`;
- the threaded design uses a bounded Kanal MPMC queue;
- completion sentinels are emitted once per configured DuckDB worker.

## Matched 4 GiB benchmark

Inputs:

- 268,435,456 rows;
- two identical `i64` columns generated using the same SplitMix64 function;
- 65,536 rows per block/chunk;
- raw size: 4,294,967,296 bytes;
- uncompressed V1 size: 4,296,181,300 bytes (0.028% larger);
- eight DuckDB workers;
- prefetch depth four per worker;
- page-cache eviction before each measured query;
- three iterations with no warm-up.

Command:

```bash
/mnt/vortex-ssd/vortex/target/release_debug/io_uring_scan \
  --path /mnt/vortex-ssd/bench-data/duckdb-io-bench-4g.bin \
  --vortex-path /mnt/vortex-ssd/bench-data/duckdb-io-bench-4g-uncompressed.vortex \
  --file-rows 268435456 \
  --rows 268435456 \
  --block-rows 65536 \
  --prefetch 4 \
  --threads 8 \
  --warmup 0 \
  --iterations 3 \
  --direct=false \
  --evict-cache \
  --regenerate-vortex
```

Median results:

| Engine | Light SQL | Heavy SQL |
| --- | ---: | ---: |
| `pread` | 1.129 s | 1.566 s |
| per-worker `io_uring` | **0.743 s** | **1.212 s** |
| threaded `io_uring` | 0.891 s | 1.295 s |
| current V1 layout reader | 0.897 s | 1.288 s |

Per-worker rings were fastest in both workloads. Heavy SQL did not starve I/O: it produced only
eight completion waits, one initial wait per ring, while the remaining 4,088 blocks were ready
when consumed. Light SQL produced roughly 153-230 waits because it consumed blocks faster.

The per-worker depth sweep was:

| Depth per worker | Light SQL |
| ---: | ---: |
| 2 | 0.741 s |
| 4 | 0.742 s |
| 8 | 0.748 s |
| 16 | 0.759 s |

The threaded ring is therefore not slower because the queue is too short. Two to four outstanding
requests per worker were sufficient.

## Cross-thread handoff isolation

The payload buffer is not copied through the Kanal queue. `Block` owns an `AlignedBuffer`, which is
a pointer and allocation layout; moving a `Block` transfers this small descriptor. The dedicated
I/O thread reaps completions but does not inspect payload bytes before sending the descriptor to a
DuckDB worker.

Instrumentation records:

- fast, non-blocking descriptor receives and their total/maximum duration;
- consumer time blocked waiting for the producer;
- time completed blocks reside in the handoff queue;
- producer send time, including queue-full backpressure.

These quantities must be interpreted separately. Producer send time can overlap useful DuckDB
work and mostly indicates backpressure; it is not additive wall-clock overhead. Consumer blocking
time can include storage latency. Fast receive duration is the closest direct measurement of the
queue/descriptor-transfer overhead.

The instrumented binary was run on the same 4 GiB input with the same eight workers, depth four,
cold-cache eviction, and light SQL. Both engines were measured in the same process and binary,
with one warm-up and five measured iterations:

| Measurement | Per-worker ring | Threaded ring |
| --- | ---: | ---: |
| median query time | 743.239 ms | 922.754 ms |
| reads | 4,096 | 4,096 |
| bytes | 4,294,967,296 | 4,294,967,296 |
| callbacks | 131,080 | 131,080 |
| DuckDB local readers | 8 | 8 |
| median summed fast descriptor receive time | n/a | 0.226 ms |
| median summed producer send time | n/a | 2.882 ms |

The query-time gap was 179.515 ms. Fast descriptor receives accounted for 0.126% of that gap.
Even treating all producer send time as serial overhead, which overstates it because the producer
runs concurrently, accounts for only 1.6% of the gap. The queue moves the small `Block` descriptor,
not the 1 MiB allocation, so descriptor handoff is conclusively not the main cause of the threaded
ring's slowdown.

The summed queue-residence and consumer-wait measurements can exceed wall time because eight
workers wait concurrently and completed blocks can reside in the queue while DuckDB computes.
They describe scheduling/backpressure and must not be added to elapsed query time.

A separate five-iteration threaded-only light run had median query time 887.625 ms and median fast
receive time 0.207 ms, showing the instrumentation itself did not materially change the earlier
roughly 891 ms threaded result. Heavy runs were bimodal, but the median fast receive total remained
only 0.397 ms.

Remaining questions:

1. Isolate central-ring serialization separately. A useful next
   variant is one shared ring with worker-affine completion queues; that keeps a single submitter
   while removing MPMC consumer contention.
2. If cache placement is still suspected, run buffered and `O_DIRECT` variants separately. With
   buffered reads, kernel copying can establish cache residency on a different CPU even though the
   userspace I/O thread does not read the payload. With `O_DIRECT`, device DMA and first-touch
   behavior differ.

Suggested build and run:

```bash
CARGO_TARGET_DIR=/mnt/vortex-ssd/vortex/target \
  cargo build -p duckdb-bench --bin io_uring_scan --profile release_debug

/mnt/vortex-ssd/vortex/target/release_debug/io_uring_scan \
  --path /mnt/vortex-ssd/bench-data/duckdb-io-bench-4g.bin \
  --file-rows 268435456 \
  --rows 268435456 \
  --block-rows 65536 \
  --prefetch 4 \
  --threads 8 \
  --engines threaded \
  --workloads light,heavy \
  --warmup 1 \
  --iterations 5 \
  --direct=false \
  --evict-cache
```

## Validation state

Completed before the handover:

- `cargo +nightly fmt --all`;
- `cargo clippy -p vortex-duckdb --all-targets`;
- `cargo clippy -p duckdb-bench --bin io_uring_scan`;
- matched full-size benchmark and callback/read-count invariants before adding handoff metrics.

The two targeted clippy commands also passed after adding the handoff metrics. The instrumented
release binary was successfully rebuilt with
`CARGO_TARGET_DIR=/mnt/vortex-ssd/vortex/target` and used for the isolation runs above. An earlier
attempt selected `/home/ec2-user/vortex-5/target/release_debug` and filled the root filesystem;
that incomplete directory was cleaned externally.

Workspace-wide clippy was previously blocked by the host's Python 3.9 being older than the
workspace's `abi3-py311` requirement; this is unrelated to the experiment.
