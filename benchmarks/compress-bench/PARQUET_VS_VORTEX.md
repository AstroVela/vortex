# `parquet-vs-vortex`

Re-encodes one or more Parquet files into a single Vortex file and compares the two on on-disk
size and full-scan decompression throughput.

`compress-bench` walks a fixed dataset list and round-trips each dataset through memory. This
binary instead takes arbitrary Parquet paths and streams both the conversion and the scans, so it
can be pointed at inputs far larger than RAM.

```bash
cargo build --release -p compress-bench --features unstable_encodings --bin parquet-vs-vortex

target/release/parquet-vs-vortex \
  --parquet shard-{1,2,3,4}.parquet \
  --vortex out.vortex \
  --verify --per-column --parquet-zstd 6,9 --iterations 3
```

Flags:

- `--compact` writes with `BtrBlocksCompressorBuilder::with_compact`, adding Zstd for strings and
  binary.
- `--verify` decodes both formats and compares a per-column value digest before timing anything.
  Digests are per column rather than per batch because the two readers pick different batch sizes.
  Utf8 and Utf8View hash identically, so matching values produce matching digests.
- `--per-column` additionally times a single-column projection for every top-level column.
- `--parquet-zstd <levels>` re-encodes the input as Zstd Parquet at each level and measures encode
  time, size and decode time.

Build the binary with `--features unstable_encodings` to write with the preview edition encodings
(OnPair strings, Zstd buffer compression, Delta integers); without it the file uses only the
frozen `core` edition.

## Ultra-FineWeb-L1 results

First 4 shards of [`openbmb/Ultra-FineWeb-L1`](https://huggingface.co/datasets/openbmb/Ultra-FineWeb-L1)
`data/CC-MAIN-2025-30` — 1,827,723,846 bytes, 788,876 rows, columns `uid`/`content`/`meta`
(string) and `dataset_index` (int64). `content` is 91% of the compressed bytes. The shards ship as
Snappy with roughly 1k-row row groups.

Measured on 4 cores / 15 GB RAM, warm page cache, fastest of 3 runs. Every Vortex file was
verified to decode to values identical to the Parquet input.

### Size and encode time

| File | Size | vs source | Encode |
| --- | --- | --- | --- |
| Source Parquet (Snappy) | 1,827,723,846 | — | — |
| Vortex default, stable | 1,706,465,052 | −6.6% | — |
| Vortex default, unstable | 1,308,972,508 | −28.4% | 34.7s |
| Vortex compact, stable | 1,128,387,488 | −38.3% | — |
| Vortex compact, unstable | 1,128,387,768 | −38.3% | 24.9s |
| Parquet zstd-6 | 1,058,590,622 | −42.1% | 99.5s |
| Parquet zstd-9 | 1,027,387,394 | −43.8% | 151.1s |

The unstable encodings account for a 23.3% size reduction on the default strategy
(1.71 GB to 1.31 GB) at identical decode speed. They make no difference under `--compact`, where
the Zstd string scheme is chosen either way.

Against a *well-configured* Parquet baseline the size ranking reverses: Parquet zstd-9 is 8.9%
smaller than the best Vortex file here. The headline "−28.4% versus Parquet" compares against
Snappy with small row groups, not against Parquet at its best.

### Decode, all columns

| File | Decode | vs source Parquet |
| --- | --- | --- |
| Source Parquet (Snappy) | 8.51s | 1.00x |
| Parquet zstd-6 | 8.61s | 0.99x |
| Parquet zstd-9 | 8.48s | 1.00x |
| Vortex default, stable | 0.63s | 13.5x |
| Vortex default, unstable | 0.59s | 14.4x |
| Vortex compact, stable | 1.76s | 4.8x |
| Vortex compact, unstable | 1.81s | 4.7x |

Parquet decode is flat across codecs, so the Parquet read is bound by Arrow string materialisation
rather than by decompression.

### Caveats

- **Thread count.** The Vortex scan uses the Tokio runtime across all 4 cores; the
  `ParquetRecordBatchStream` read is a single serial stream. Pinned to one core with `taskset -c 0`
  the same comparison is 7.04s versus 2.18s, a **3.2x** gap rather than 14x. The multi-core figures
  measure two readers as they ship, not decode efficiency per core.
- **Arrow representation.** Vortex yields `Utf8View`, Parquet yields `Utf8`, so the two report
  different `get_array_memory_size` totals (2.99 GiB versus 4.31 GiB) for identical values. Compare
  wall time and rows/s, not the byte throughput columns.
- **Row groups.** The zstd re-encode also replaces the source's ~1k-row row groups with the writer
  default, so it is a well-configured Parquet baseline rather than a codec swap alone.
