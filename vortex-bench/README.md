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
2. **Register tables (or views).** Each benchmark's DDL is checked in as `sql/{benchmark}/create.sql`, the
   companion of its query files, and `pipeline::register_tables` runs it once per format before any query. See
   [Table registration](#table-registration) below.
3. **Run the queries.** `runner::SqlBenchmarkRunner` executes each query, collects timings and memory, and validates
   row counts.

### Table registration

`sql/{benchmark}/create.sql` holds the DDL registering that benchmark's tables, next to the queries that read
them. The data directory depends on the scale factor and the format under test, so the file is a template; the
harness substitutes these placeholders before executing it:

| Placeholder | Meaning |
| --- | --- |
| `{object}` | `TABLE` or `VIEW`, depending on whether the engine loads the data or reads it in place |
| `{dir}` | absolute path (or URL) of the directory holding the requested format's files |
| `{ext}` | file extension of that format: `parquet` or `vortex` |
| `{read}` | DuckDB reader for that format: `read_parquet` or `read_vortex` |
| `{format}` | DataFusion `STORED AS` name for that format: `PARQUET` or `VORTEX` |

DuckDB and DataFusion do not share a DDL dialect, so the file carries one section per engine, introduced by an
`-- @engine <name>` header:

```sql
-- @engine duckdb
CREATE {object} IF NOT EXISTS nation AS SELECT * FROM {read}('{dir}/nation_*.{ext}');

-- @engine datafusion
CREATE EXTERNAL TABLE IF NOT EXISTS nation STORED AS {format} LOCATION '{dir}/nation_*.{ext}';
```

Statements are split on semicolons, so a comment must never contain one — the same constraint the query files
carry. `STORED AS VORTEX` works because the benchmark session registers `VortexFormatFactory`; DataFusion infers
each table's schema from the files rather than using the pinned `TableSpec` schema.

Two paths do not go through `create.sql`, because no DDL expresses them: Lance's `LanceTableProvider`, and
DataFusion's `VortexTable` scan API under `VORTEX_USE_SCAN_API=1`. Both implement `TableRegistrar::register`
directly. Benchmarks whose table list is only known at runtime (Public BI, SpatialBench's optional `zone`) have no
`create.sql`; registration falls back to statements generated from `table_specs()`.

To add an engine, implement `TableRegistrar` — return a `SqlDialect` from `dialect()` to run `create.sql`, or
implement `register()` to build providers — and pass the registration to `SqlBenchmarkRunner::run_all` (or
`run_all_async`) as its register stage. To add a benchmark suite, implement `Benchmark` and add a `create.sql`;
`table_specs`, `pattern` and `format_path` are all the generated fallback needs.

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
