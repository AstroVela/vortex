# Benchmarks

There are a number of benchmarks in this repository that can be run using the `cargo bench` command. These behave more
or less how you'd expect.

There are also some binaries that are not run by default, but produce some reporting artifacts that can be useful for
comparing vortex compression to parquet and debugging vortex compression performance. These are:

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

Benchmark binaries use mimalloc by default. On Linux, the `jemalloc` feature switches the shared
`vortex-bench` allocator to jemalloc, enables sampled heap profiling, and serves pprof data from
`http://127.0.0.1:6060/debug/pprof/allocs`:

```bash
cargo run -p datafusion-bench --profile release_debug --features jemalloc -- \
  tpch --formats vortex --queries 1 --iterations 20

curl --fail --output /tmp/vortex-heap.pb.gz \
  http://127.0.0.1:6060/debug/pprof/allocs
```

Set `VORTEX_HEAP_PROFILE_ADDR` to override the listen address. The profiler samples one allocation
per 512 KiB on average. See
[`rust-jemalloc-pprof`](https://github.com/polarsignals/rust-jemalloc-pprof) for the pprof format
and allocator requirements. On Linux, the feature also overrides C and C++ `malloc` calls.
`duckdb-bench` additionally compiles DuckDB without its private jemalloc so DuckDB's internal and
direct allocations are captured by the same profile.

## Common Issues

If the benchmarks fail because of this error:

```
Failed to compress to parquet: No such file or directory (os error 2)
```

You likely do not have the required packages installed. On macOS, try this:

```
brew install duckdb cmake ninja pkg-config vcpkg
```
