# Compression benchmark

Measures compression and decompression throughput, plus resulting file sizes, for Vortex
versus Parquet (and optionally Lance) across a range of datasets: NYC taxi data, several
[Public BI](https://github.com/cwida/public_bi_benchmark) tables (Arade, Bimbo,
CMSprovider, Euro2016, Food, HashTags), TPC-H `l_comment` variants, and synthetic nested
data. This is the workload behind the `Compression` PR comment.

See [`src/main.rs`](./src/main.rs) for the dataset list and CLI flags (`--formats`,
`--datasets`, `--ops compress,decompress`).

## Running locally

```bash
cargo run -p compress-bench --profile release_debug
```

GPU decompression is opt-in. By default it runs only the existing benchmark names
allow-listed in `src/main.rs`:

```bash
cargo run -p compress-bench --profile release_debug \
  --features cuda,unstable_encodings -- --gpu-decompress
```

An explicit `--datasets` filter overrides the allow-list, which is how you run a
not-yet-promoted dataset on GPU to do the end-to-end verification that promotes it:

```bash
cargo run -p compress-bench --profile release_debug \
  --features cuda,unstable_encodings -- --gpu-decompress --datasets 'Arade'
```

On Linux, GPU files are read with direct IO (`O_DIRECT`) so repeated iterations measure
storage bandwidth rather than page-cache hits.
