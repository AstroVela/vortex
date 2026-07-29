# Benchmarks

There are a number of benchmarks in this repository that can be run using the `cargo bench` command. These behave more
or less how you'd expect.

There are also some binaries that are not run by default, but produce some reporting artifacts that can be useful for
comparing vortex compression to parquet and debugging vortex compression performance. These are:

## The SQL benchmark pipeline

Every SQL benchmark — `duckdb-bench`, `datafusion-bench` and `lance-bench` — runs through the same three stages,
defined in [`src/pipeline.rs`](./src/pipeline.rs):

1. **Generate data.** `pipeline::generate_data` writes the canonical Parquet base data and derives every requested
   format from it. It is idempotent, so it is a no-op when the data is already on disk. CI runs it ahead of time as
   its own step (`vx-bench prepare-data`, the `data-gen` binary), but each benchmark binary calls it too, so running
   a benchmark directly still works.
2. **Register tables (or views).** Each engine implements `pipeline::TableRegistrar`, and `pipeline::register_tables`
   drives it once per format, before any query runs. Resolving a table to the files backing it — the format
   directory, the file glob, the pinned schema — is shared; only the registration itself is engine-specific. DuckDB
   creates SQL views (or, for the on-disk DuckDB format, tables loaded from Parquet); DataFusion registers a
   `ListingTable` or a `VortexTable`; Lance registers a `LanceTableProvider`.
3. **Run the queries.** `runner::SqlBenchmarkRunner` executes each query, collects timings and memory, and validates
   row counts.

To add an engine, implement `TableRegistrar` for it and pass the registration to `SqlBenchmarkRunner::run_all` (or
`run_all_async`) as its register stage. To add a benchmark suite, implement `Benchmark`: `table_specs`, `pattern` and
`format_path` are all stage 2 needs.

### `compress.rs`

This binary compresses a file using vortex compression and writes the compressed file to disk where it can be examined
or used for other operations.


### `query_bench`

This is the unified benchmark runner that supports multiple benchmark suites including TPC-H, ClickBench, and TPC-DS.

To run the TPC-H benchmarks you can use:

```bash
cargo run --bin query_bench -- tpch
```

To run the ClickBench benchmarks:

```bash
cargo run --bin query_bench -- clickbench
```

For profiling, you can open in Instruments using the following invocation:

```
cargo instruments -p vortex-bench --bin query_bench --template Time --profile bench -- tpch
```

### Data directory

There is a data directory at `vortex/vortex-bench/data` where parquet and vortex files used for the benchmark runs
can be found.

## Memory allocators

If you don't want to use the default system allocator, there are `"jemalloc"` and `"mimalloc"` features available that
configure a different allocators at compile time.

As of this writing, if both are enabled `mimalloc` will be used.

## Common Issues

If the benchmarks fail because of this error:

```
Failed to compress to parquet: No such file or directory (os error 2)
```

You likely do not have the required packages installed. On macOS, try this:

```
brew install duckdb cmake ninja pkg-config vcpkg
```
