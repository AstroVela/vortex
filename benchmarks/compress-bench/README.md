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

GPU decompression is opt-in and runs the full dataset suite:

```bash
cargo run -p compress-bench --profile release_debug \
  --features cuda,unstable_encodings -- --gpu-decompress
```

A dataset only decodes on GPU if every encoding `only_cuda_compatible()` picks for it has a
kernel registered in `initialize_cuda`. Where one does not, the run fails with `No CUDA kernel
for encoding ...` — the `wide table` datasets encode their columns as `vortex.list`, which has
no kernel. Use `--datasets` to narrow the run to the datasets that decode:

```bash
cargo run -p compress-bench --profile release_debug \
  --features cuda,unstable_encodings -- --gpu-decompress --datasets 'TPC-H l_comment canonical'
```

On Linux, GPU files are read with direct IO (`O_DIRECT`) so repeated iterations measure
storage bandwidth rather than page-cache hits.
