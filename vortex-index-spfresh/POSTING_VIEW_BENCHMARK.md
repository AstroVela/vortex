# Read-Only Posting View Qualification

Local Linux qualification on 2026-10-04, based on Vortex
`c6f7e497f99205937a6f4e73b4ab9ac3b74593b3` (merged PR #16), with the posting-view
changes in `feat/vortex-index-posting-view-20261004`. The outer benchmark driver
is duckdb-vortex `094d36b943d11f726b31824ae5e6dd2a206be7d7`.
SPFresh remains pinned to `5893eb61ee3b18610b6b00f1939be7dae1af8904` with the
checked-in static-only patch; this is not unmodified upstream SPFresh.

## Method

- Same executable and frozen generation for each layer's A/B comparison.
  `VORTEX_SPFRESH_POSTING_VIEW=0` uses the original copying reader; `1` uses views.
  Each comparison starts new processes; runs are sequential, with no competing
  builds or benchmark measurements and no OS page-cache eviction.
- Intel Xeon E5-2686 v4, Linux x86_64, AVX2 dispatch. The whole driver and its
  children are restricted to CPU 8, 100% CPU quota, 2 GiB memory and no swap.
  `OMP_NUM_THREADS`, `RAYON_NUM_THREADS` and `TOKIO_WORKER_THREADS` are all `1`.
- Rust 1.97.1 release builds, 16 codegen units, LTO disabled and empty
  `RUSTFLAGS`. SQL shells reuse the same DuckDB v1.5.0 SDK/link objects
  (`d8a9d61d59`); the changed Vortex archive is rebuilt from this checkout.
- First 1,000 original queries, one complete warmup round and three measured
  rounds: 3,000 samples per full mode. Native/Provider processes handle at most
  256 queries each, with a complete warmup for every chunk. Open/close and output
  serialization are excluded from their per-query timers.
- `k=10`, `max_check=32768`, `search_pages=posting_page_limit=128`. Frozen
  `internal_results` is 64 for SIFT100K and 512 for SIFT1M. The original Float32
  vectors, positional IDs and supplied ground truth are preserved.
- SQL measures prepared query profile latency, including ranked original-row
  materialization, not only ANN search. PREPARE is outside the timer. Strict SQL
  remains the default; snapshot SQL is explicitly selected for its own runs.
- SIFT100K uses 4,096 heads and four replicas; SIFT1M uses 16,384 heads and one
  replica. Neither fixture is rebuilt between the copying and view runs.

SIFT1M's 534,710,361 artifact bytes exceed the production snapshot cache's
256 MiB retained-artifact budget. Its snapshot measurements therefore use a
**separately compiled, benchmark-only 1 GiB budget**. Both A/B modes verify that
the default 256 MiB shell rejects this snapshot. SIFT100K uses the production
budget. No production limits or strict validation behavior are changed.

## Latency

All times are milliseconds. A/B Native and Provider ranked IDs/distances are
exactly equal for every measured query and round. Each run also verifies SQL
cross-layer equality, repeated results and original-row retrieval. Recall@10
is unchanged: **0.9977 for SIFT100K**, **0.9940 for SIFT1M**.

| Dataset | Layer | Copy p50 | View p50 | Copy p99 | View p99 | p50 reduction |
|---|---|---:|---:|---:|---:|---:|
| SIFT100K | Native `cpp-reuse` | 2.439 | 1.982 | 2.894 | 2.290 | 18.7% |
| SIFT100K | Bridge | 2.456 | 2.016 | 2.859 | 2.281 | 17.9% |
| SIFT100K | Provider | 2.497 | 2.039 | 2.934 | 2.309 | 18.4% |
| SIFT100K | Prepared snapshot SQL | 6.674 | 6.358 | 7.346 | 6.863 | 4.7% |
| SIFT1M | Native `cpp-reuse` | 10.216 | 5.963 | 13.473 | 7.492 | 41.6% |
| SIFT1M | Bridge | 10.330 | 5.942 | 13.573 | 7.423 | 42.5% |
| SIFT1M | Provider | 10.321 | 5.973 | 13.767 | 7.467 | 42.1% |
| SIFT1M | Prepared snapshot SQL, experimental budget | 16.001 | 10.200 | 19.880 | 11.975 | 36.3% |

View-path p50 by measured round:

| Dataset | Native `cpp-reuse` | Provider | Snapshot SQL |
|---|---|---|---|
| SIFT100K | 1.981 / 1.979 / 1.985 | 2.051 / 2.022 / 2.046 | 6.365 / 6.352 / 6.362 |
| SIFT1M | 5.985 / 5.981 / 5.911 | 5.972 / 5.941 / 6.001 | 10.152 / 10.199 / 10.247 |

The workspace-reset control also improves: `cpp-reset` p50 is 15.444 -> 12.796 ms
on SIFT100K and 38.765 -> 24.649 ms on SIFT1M. `cpp-reuse` remains a direct
single-handle/thread lower bound, not a supported multi-handle API.

Strict SQL is a four-query control only (IDs 0, 333, 666, 999; 12 measured
samples), not the full-query distribution. Its p50 is 727.071 -> 743.094 ms for
SIFT100K and 6070.812 -> 6221.009 ms for SIFT1M. These small controls show no
strict-mode latency improvement; source validation/provider open still dominate.
Do not mix their percentiles with the 3,000-sample snapshot/ANN measurements.

## Memory Tradeoff

Peak process RSS in MiB, including startup; Native/Provider use the maximum
across chunks. These are not anonymous-memory or retained-artifact counters.

| Dataset | Layer | Copy | View |
|---|---|---:|---:|
| SIFT100K | Native `cpp-reuse` | 49.08 | 221.11 |
| SIFT100K | Provider | 52.12 | 223.77 |
| SIFT100K | Snapshot SQL | 143.06 | 313.87 |
| SIFT1M | Native `cpp-reuse` | 147.81 | 530.62 |
| SIFT1M | Provider | 164.26 | 546.42 |
| SIFT1M | Snapshot SQL, experimental budget | 389.51 | 750.46 |

Touched mapped file pages become file-backed process RSS. Avoiding posting
copies does not mean lower total RSS, and the existing workspace buffers remain
allocated for fallback paths. No total-RSS quota or memory-cache equivalence to
OpenData/SlateDB is claimed. Use the copying toggle when this tradeoff is unsuitable.

## Read/Copy Diagnostic

After the latency runs, `strace -c` on the same final native executable processes
the first 256 SIFT1M queries with `cpp-reuse`, no warmup and one round. Totals
include process startup/open/close. Trace overhead makes these runs unsuitable
for latency comparisons; their 2,560 ranked hits are exactly equal.

| System call | Copy | View |
|---|---:|---:|
| `read` | 132,234 | 1,195 |
| `lseek` | 131,039 | 0 |
| `fstat` | 8 | 131,048 |
| `newfstatat` | 5 | 5 |
| `pread64` | 2 | 2 |

The remaining `fstat` calls are intentional per-view file-size checks and are
included in all final query timings. They detect sequential truncation before
touching the map. `MAP_PRIVATE` does not freeze mutable files: concurrent
rewrite/truncation still violates the private immutable-generation contract.

The fast path reads bounded, aligned, uncompressed Float32 postings directly;
it does not call the mutable decoding path or copy the posting into a page
buffer. Delta decoding and truth diagnostics retain copying behavior. Mapping
ownership ends with the native reader, before its private lease is removed.

## Regression Checks

- Three CTest targets pass, including five consecutive repetitions and a final
  copying-toggle run. Posting tests exercise real multipage postings, counted
  view/copy calls, exact fallback parity, bounded extents/alignment, truncation,
  path replacement, nonblocking non-file rejection, cleanup and toggle capture.
- A control executable with the view-processing branch disabled fails the
  multipage zero-copy assertion. The existing Rust read-failure test initially
  reproduced a fault on sequential truncation; it passes with the file-size guard.
- `vortex-index-spfresh` release library tests: 18 with all features, two without
  default features, all passing. No new ASAN setup or ASAN runs.
- Final SQL shell qualification: 66 independent processes, 38 negative cases,
  including disabled-local-filesystem policy, NUL arguments, invalid vector
  types, first-execution/prepared reference pinning and cross-process reopen.
- Five outer benchmark-tool test modules: 186 passed using the final C API
  executable, including real C API/systemd opt-ins.
- Nightly Rust formatting, clang-format 23 checks and `git diff --check` pass.
  Scoped Clippy with `--all-targets --all-features --no-deps -- -D warnings`
  passes. Ordinary Clippy is blocked by unchanged baseline lints exposed by
  Rust 1.97 in `vortex-array` (`useless_conversion`, `manual_filter`) and, for
  release checks, `vortex-buffer` (`manual_map`). No unrelated fixes are included.

## Reproduction And Evidence

Local artifacts are under
`/home/kaka/vortex-worktrees/vane-index-posting-view-20261004/target/posting-view-qualification/`:

- `sift100k-{copy,view}/summary.json` and `sift1m-{copy,view}/summary.json` include
  inputs/artifact hashes, every mode's timings, worker RSS and method details;
  subdirectories retain samples and SQL profiles.
- `provenance.json` records the base revisions, source/native/Provider/SQL hashes,
  resource restrictions and A/B toggles. All recorded hashes are rechecked at
  the end of the full driver.
- `syscalls-final-{copy,view}.{txt,csv}` retain the final diagnostic evidence;
  `sql-native-tests-final/summary.json` retains SQL qualification results.

In duckdb-vortex, use `scripts/bench_index_sift.py` with qualified binaries built
from this checkout, not the unchanged outer Vortex dependency. Run sequentially
in the same restricted scope, with a new output directory for each toggle:

```sh
VORTEX_SPFRESH_POSTING_VIEW=0 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
TOKIO_WORKER_THREADS=1 uv run --with numpy python scripts/bench_index_sift.py \
  --dataset sift100k --input-manifest "$INPUT_MANIFEST" \
  --duckdb "$DEFAULT_SQL" --native "$NATIVE" --provider "$PROVIDER" \
  --fixture-builder "$FIXTURE_BUILDER" --fixture "$FIXTURE" \
  --output-dir "$OUTPUT_COPY" --queries 1000 --strict-queries 4 \
  --warmup-rounds 1 --rounds 3 --probes 64
```

Repeat with toggle `1` and a new output directory. For SIFT1M use its fixture,
`--probes 512`, `--snapshot-duckdb "$BENCHMARK_ONLY_SQL_1G"` and
`--snapshot-cache-budget-mib 1024`. That separately compiled shell must differ
only in the retained-artifact budget; retain the default shell for rejection
controls. Per-run cross-layer checks are performed by the driver; compare A/B
Native/Provider sample `(round, query, ranked_hits)` tuples as well.

These measurements qualify warm, serialized static search, not cold storage,
concurrent throughput, dynamic SPFresh updates, NVMe/SPDK, a released extension
wheel, or an equivalent cache configuration against another product.
