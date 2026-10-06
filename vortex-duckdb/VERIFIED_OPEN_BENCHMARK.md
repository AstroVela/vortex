<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Single Pass Verified Index Open Qualification

Local Linux qualification on 2026-10-05 compares Vortex
`67303ad0157f873eea54e9f333737a4ad6bdaa36` with the changes in
`feat/vortex-index-verified-open-20261005`. Combining artifact verification and
private copying removes one complete artifact read on a cache miss. First-use
latency falls by about 30%-34% in the three-process comparison; warm query
latency, result contents and recall remain effectively unchanged.

## Scope And Method

- The outer driver is duckdb-vortex `5c02fd34c59fc3b1f0ae2546a7d679200de37889`.
  Candidate builds replace only `vortex-index`, `vortex-index-spfresh` and
  `vortex-duckdb` with local source overlays. The generated lockfile differs
  only by removing the Git source from those three packages. Production
  manifests, lockfiles and the outer dependency pin are unchanged.
- Both versions reuse the same embedded Vane DuckDB v1.5.0 SDK, source ID
  `d8a9d61d59`, benchmark C++ program, native library and frozen SIFT fixtures.
  SPFresh remains pinned to `5893eb61ee3b18610b6b00f1939be7dae1af8904` with the
  existing static-only patch. No native algorithm or format checks changed.
- Intel Xeon E5-2686 v4, AVX2, Linux x86_64. Each worker is pinned to CPU 8
  with a one-core quota, 2 GiB memory limit and no swap. CPU 8 and its SMT
  sibling 26 are monitored. Rust 1.97.1, release optimization level 3,
  16 codegen units, no LTO and empty `RUSTFLAGS` match the baseline.
- Runs are sequential, after builds finish, with warm OS page cache and no
  cache eviction. This is **fresh-process first use**, not cold-storage latency.
  `VORTEX_SPFRESH_POSTING_VIEW=1`; OpenMP, Rayon and Tokio worker counts are 1.
- SQL explicitly selects `validation_mode := 'snapshot'`. Strict remains the
  default and continues revalidating canonical artifacts on every execution.
  Timers include C API prepared execution and materializing all ten original
  rows, including their 128-dimensional vectors. Prepare, parameter binding
  and CSV serialization are outside the per-query timer.
- `k=10`, `max_check=32768`, `search_pages=posting_page_limit=128`;
  `internal_results=64` for SIFT100K and 512 for SIFT1M. Source and artifact
  digests are checked before and after runs; no fixture is rebuilt.
- First-use comparison: first 16 original queries, three fresh processes per
  version and dataset, alternating baseline/candidate ordering, one warmup and
  one measured round. Separate full-query runs use the first 1,000 queries,
  one complete warmup and three measured rounds per version and dataset.

SIFT100K uses the production 256 MiB retained-artifact budget. SIFT1M's
534,710,361 artifact bytes require a separate **benchmark-only 1 GiB** build in
both versions. Both production-budget binaries still reject this snapshot.
No production cache limits or validation defaults were relaxed.

## Results

First-use times are milliseconds, with phase logging disabled. All three runs
are included; the main comparison is their median.

| Dataset | Baseline runs | Candidate runs | Median before | Median after | Reduction |
|---|---|---|---:|---:|---:|
| SIFT100K | 1926.195 / 1903.993 / 1929.622 | 1256.981 / 1384.086 / 1277.941 | 1926.195 | 1277.941 | 33.7% |
| SIFT1M | 5562.411 / 5563.733 / 5571.638 | 3858.671 / 3870.466 / 4040.494 | 5563.733 | 3870.466 | 30.4% |

The independent full-query runs contain 3,000 measured samples per cell.
These sub-percent warm changes are not claimed as a query-speed improvement.

| Dataset | Before p50 | After p50 | Before p99 | After p99 | Recall@10, both |
|---|---:|---:|---:|---:|---:|
| SIFT100K | 4.736 | 4.713 | 5.089 | 5.076 | 0.9977 |
| SIFT1M | 9.007 | 8.978 | 10.673 | 10.699 | 0.9940 |

Candidate round p50s are 4.685 / 4.719 / 4.738 ms for SIFT100K and
8.943 / 8.931 / 9.044 ms for SIFT1M. Full-query peak RSS is 310.05 -> 311.20 MiB
and 749.54 -> 748.61 MiB, respectively; no meaningful RSS reduction is claimed.
Their first queries independently measure 1914.586 -> 1287.773 ms and
5741.775 -> 3846.744 ms.

Across 24 successful workers, 166,400 complete returned records pass checks
against original vectors, squared L2 distances and frozen Provider results.
Additional ordered digests over IDs and Float32 distances/vectors are exactly
equal between versions for every series. These totals include warmup and
separate diagnostic runs, not just the measured samples.

## Read Evidence And Timing Limits

The isolated first-search resource interval reports these logical read bytes
(`rchar`), identically across repeated runs:

| Dataset | Before | After | Saved, exactly artifact size | Read calls before / after |
|---|---:|---:|---:|---:|
| SIFT100K | 628,798,000 | 430,056,404 | 198,741,596 | 32,772 / 29,726 |
| SIFT1M | 1,815,870,918 | 1,281,160,557 | 534,710,361 | 91,101 / 82,928 |

Artifact writes are unchanged: the private copy is still required. These are
logical reads, not physical disk savings; first-search kernel `read_bytes` is
zero in every qualified run. SIFT1M first-search CPU time has a median of
5528.286 -> 3821.191 ms in the three-process comparison.

Two separate phase-logging diagnostics retain SIFT1M candidate wall times of
5761.883 and 5069.898 ms, with CPU times of 4247.606 and 4112.185 ms. CPU 8's
sampled I/O wait is 24.8% and 18.1% over the surrounding intervals; sibling
CPU usage is below 0.4%. Both still read exactly one artifact pass less, pass
all result checks and show a first cache miss followed by hits. The exact
source of the I/O waiting was not traced. These noisy diagnostics are retained,
not substituted into or used to select the uninstrumented comparison.

Phase attribution changes: `source_validation_ms` now includes source and
manifest validation, while artifact verification moves into `provider_open_ms`.
Compare total time or their sum, not either phase alone. This change does not
eliminate source verification, private-copy writes, native format validation,
native opening or the first ANN search. First use still takes seconds.

## Regression Checks

- `vortex-index --all-features`: 133 unit tests and two doctests pass.
  New tests count streamed bytes and cover corruption, truncation, growth,
  missing/non-regular files, FIFO rejection, limits, inventory validation,
  private permissions, failed-copy cleanup, independent leases, concurrent
  handoff, directory replacement and lifetime ownership.
- `vortex-index-spfresh --all-features`: 19 tests pass with the real native
  library, including ordinary/materialized parity, recall, ranked original-row
  retrieval and opening after the canonical generation has been deleted.
- `vortex-duckdb --all-features index::`: 140 tests pass against its cached
  DuckDB v1.5.5 test library. Optimized fixture providers exercise prepared
  ownership, strict mutation checks, snapshot pinning, access policy, budgets
  and failure both before and after the private lease transfer.
- The actual embedded Vane v1.5.0 shell separately passes SQL qualification:
  66 processes, 38 negative cases, cross-process reopen, result parity,
  prepared/first-execution reference pinning, NUL rejection, disabled local
  filesystems and invalid vector types. Both default-budget C API binaries
  reject the oversized SIFT1M snapshot.
- Nightly formatting and `git diff --check` pass. Scoped Clippy for index and
  SPFresh passes with `--all-targets --all-features --no-deps -- -D warnings`.
  DuckDB passes with the same flags plus `-A clippy::question_mark` for the
  unchanged `src/table_function.rs:952`. Ordinary Clippy is blocked by baseline
  Rust 1.97 lints in `vortex-array` and `vortex-edition`; no unrelated lint fixes
  are included. No ASAN setup or runs were added.

## Reproduction And Local Evidence

Local outputs are under
`/home/kaka/vortex-worktrees/vane-index-verified-open-20261005/target/verified-open-qualification/`:

- `prepare.py`, `source.json`, `tracked-changes.patch` and `sources/` retain
  the build procedure, input source hashes and isolated three-crate overlay.
  `capi-{default,cache1g}/build.json` records binary/archive/source hashes,
  SDK links, budgets and exact build settings. Original qualified binaries in
  the outer checkout were not overwritten.
- `first/`, `full/`, `stages/` and `stages-repeat/` retain summaries, raw CSV,
  phase logs, cgroup resource snapshots and host observations. `analysis.json`
  includes exact A/B result digests and source/binary rechecks.
- `sql-regression/summary.json`, `budget-control/summary.json`,
  `index-tests.log` and `duckdb-tests.log` retain the regression results.
- `full-argument-order-rejected/` contains an excluded driver attempt whose
  warmup/round argument order was wrong. Result validation rejected it; its
  samples are not included in any successful-run counts or latency summaries.

From this checkout, the retained local helper runs the constrained comparisons:

```sh
uv run --offline --no-project --with numpy python \
  target/verified-open-qualification/compare.py first --label first-new
uv run --offline --no-project --with numpy python \
  target/verified-open-qualification/compare.py full --label full-new
```

The underlying C API invocation is `CAPI QUERY_SQL QUERIES_F32BIN OUTPUT_CSV 1 3 10`
for one warmup and three measured rounds. Use the helper's verified cgroup and
phase barriers for comparable CPU, memory and I/O accounting. This qualification
does not cover cold storage, concurrent throughput, dynamic updates, SPDK,
OpenData cache equivalence or a released extension wheel.
