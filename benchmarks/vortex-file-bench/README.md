# Vortex file benchmarks

File-level benchmarks for Vortex and other columnar formats. Both suites here measure a whole
file rather than a query — how big it is, how long it takes to write and read back, and how long
it takes to fetch individual rows out of it. Neither loads a query engine, which is what
separates them from the SQL benchmarks in `benchmarks/datafusion-bench` and
`benchmarks/duckdb-bench`.

```bash
cargo run -p vortex-file-bench --profile release_debug -- <suite> [options]
```

The two suites are `compress` and `random-access`. They share the output flags
(`-d/--display-format`, `-o/--output-path`, `--ingest-jsonl`, `-v/--verbose`, `--tracing`,
`--log-format`); everything else is per-suite.

Both report the **median** run, via `vortex_bench::measurements::median`, as does `string-bench`.
Every per-iteration time is still emitted in `all_runtimes_ns` for anything that wants to compute
a different statistic downstream. The suites differ only in how they decide when to stop:
`compress` runs a fixed `--iterations` count, while `random-access` runs until `--time-limit`
seconds have elapsed.

## `compress`

Measures compression and decompression throughput, plus resulting file sizes, for Vortex versus
Parquet (and optionally Lance) across a range of datasets: NYC taxi data, several
[Public BI](https://github.com/cwida/public_bi_benchmark) tables (Arade, Bimbo, CMSprovider,
Euro2016, Food, HashTags), TPC-H `l_comment` variants, and synthetic nested data. This is the
workload behind the `Compression` PR comment.

Alongside the raw timings and sizes it emits cross-format ratios (`vortex:parquet-zstd`,
`vortex:lance`) for size, compress time, and decompress time.

See [`src/compress/mod.rs`](./src/compress/mod.rs) for the dataset list and CLI flags
(`--formats`, `--datasets`, `--ops compress,decompress`).

```bash
cargo run -p vortex-file-bench --profile release_debug -- compress
```

Note that the `decompress` op is a full scan: it opens the file, applies the projection where
the benchmark defines one, and drains the record-batch stream.

GPU decompression is opt-in and runs only the benchmark names allow-listed in
`src/compress/mod.rs`:

```bash
cargo run -p vortex-file-bench --profile release_debug \
  --features cuda,unstable_encodings -- compress --gpu-decompress
```

On Linux, GPU files are read with direct IO (`O_DIRECT`) so repeated iterations measure storage
bandwidth rather than page-cache hits.

## `random-access`

Measures point-lookup latency: fetching individual rows by index from a file, rather than
scanning it. This is the workload behind the `Random Access` PR comment.

Two access patterns are generated with a fixed seed (see
[`src/random_access/mod.rs`](./src/random_access/mod.rs)):

- **correlated**: several clusters of consecutive indices scattered across the dataset,
  simulating lookups with spatial locality;
- **uniform**: indices drawn from a Poisson process spread uniformly across the dataset,
  simulating lookups with no locality.

Each pattern runs over four datasets (`taxi`, `feature-vectors`, `nested-lists`,
`nested-structs`) in Parquet, Lance, and Vortex, both with a cached open file handle and
reopening the file per lookup. CI drives the full matrix via
[`scripts/random-access-split.py`](../../scripts/random-access-split.py).

```bash
cargo run -p vortex-file-bench --profile release_debug --features lance -- random-access
```
