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

## Hugging Face mirror (`scripts/hf_mirror.py`)

`scripts/hf_mirror.py` mirrors Hugging Face datasets and compares the mirrored Parquet against
Vortex. Every public dataset on the Hub is auto-converted to Parquet under the
`refs/convert/parquet` branch, so this works for CSV-, JSON-, and Arrow-backed datasets too, not
just the ones published as Parquet.

Browse the Hub rankings, then mirror whatever looks interesting:

```bash
# Hub rankings: trending / likes / downloads, optionally restricted to tabular Parquet.
./vortex-bench/scripts/hf_mirror.py list --sort trending --tabular

# The curated shortlist, chosen to cover distinct corners of the encoding space.
./vortex-bench/scripts/hf_mirror.py list --sort curated

# Mirror one shard each and report Parquet vs Vortex sizes, for both strategies.
./vortex-bench/scripts/hf_mirror.py mirror --max-shards 1 HuggingFaceFW/fineweb-edu mteb/results

# Mirror the whole curated set, then re-print the report without re-downloading.
./vortex-bench/scripts/hf_mirror.py mirror --max-shards 1 --max-bytes 400MiB
./vortex-bench/scripts/hf_mirror.py report
```

Datasets are named `dataset[:config[:split]]`. Pin the config where the alphabetically-first one
is unrepresentative — `wikimedia/wikipedia` would otherwise resolve to Abkhazian and `allenai/c4`
to Afrikaans. Downloads are idempotent and land in `vortex-bench/data/hf/`, so re-running only
converts. Set `HF_TOKEN` for gated datasets.

The script builds `vx` via `cargo run --release` unless `vx` is on `PATH` or `--vx` points at a
binary. Prefer passing a pre-built binary when mirroring more than a couple of datasets.
