# FastLanes BitPacked V2 — chunk-local bit widths

Handoff notes for `fastlanes.bitpacked_v2`: what it is, what is implemented, and the tests and
benchmarks that still need to be run.

## What it is

`BitPacked` (v1) packs a whole array at a single bit width. One wide region therefore taxes every
1024-element FastLanes chunk in the column: either the width rises for everyone, or the wide values
become patches.

`BitPackedV2` gives every FastLanes chunk its own bit width:

- `packed` holds one variable-width FastLanes block per chunk, concatenated. Chunk `c` occupies
  `128 * bit_widths[c]` bytes, so a chunk of small values is genuinely smaller on disk.
- `bit_widths` is a second buffer holding one `u8` per chunk. It costs 1 byte per 1024 values.
- Chunk blocks are always a whole multiple of 128 bytes, so every chunk stays aligned for its
  packed primitive type no matter what the preceding chunks chose. This is what makes the
  concatenation legal.
- Byte offsets per chunk are a prefix sum over `bit_widths`, computed once at construction and
  cached in the array data, so chunk lookup stays O(1).
- Widths may be `0` (an all-zero chunk stores nothing) up to the full type width (an incompressible
  chunk needs no patches at all — v1 has to bail out in that case).

Per chunk, the width is chosen with the same cost model v1 uses globally
(`find_best_bit_width`): the width minimising packed bytes plus the cost of the exceptions left
behind. Exceptions still become `Patches`, shared array-wide and indexed per chunk through the
existing `patch_chunk_offsets` machinery. Null slots are counted as zero-width and are never
patched, since their values are undefined.

## Layout of the code

| Path | Contents |
| --- | --- |
| `src/bitpacking_v2/array/mod.rs` | slots, `BitPackedV2Data`, validation, chunk accessors |
| `src/bitpacking_v2/array/compress.rs` | per-chunk width selection, packing, patch gathering |
| `src/bitpacking_v2/array/decompress.rs` | chunk-wise decode, `scalar_at` single-value unpack |
| `src/bitpacking_v2/vtable/` | `VTable`, serde metadata, validity, slice rule, kernels |
| `src/bitpacking_v2/compute/slice.rs` | slice by dropping whole chunks, keeping their widths |
| `src/bitpacking_v2/tests.rs` | round trips, widths, patches, slices, `scalar_at`, consistency |
| `vortex-btrblocks/src/schemes/integer/bitpacking_v2.rs` | `BitPackingV2Scheme` |

The encoding joins the `preview` edition (`vortex/src/editions/preview/v2026_06.rs`), so only a
session that enables the preview edition — i.e. one built with the `unstable_encodings` feature —
can write it to a file.

## How to turn it on

Two independent switches, both required:

1. Build with `--features unstable_encodings` (registers the scheme, enables the preview edition
   for writing).
2. Set `VORTEX_BITPACKED_V2=1` (lets `BitPackingV2Scheme` enter the scheme race; without it the
   scheme returns `Skip`, which is why the btrblocks golden snapshots are untouched by this
   change).

The scheme sits immediately after `BitPackingScheme` in `ALL_SCHEMES`, so v1 wins ties and v2 only
displaces it when the sample says it is strictly smaller.

## What is *not* implemented yet

Deliberately scoped out of the first cut; all of these are correctness-neutral (the generic path
canonicalises first) but cost performance:

- Compute kernels: `filter`, `take`, `compare`, `compare_fused`, `between`, `cast`,
  `stream_predicate`, `is_constant`. v1 has all of these; v2 has only `slice`. **Expect scan-heavy
  TPC-H queries to regress on any column that picks v2 until these land.**
- No CUDA decode kernel.
- No `Patched`-array shim for `VORTEX_EXPERIMENTAL_PATCHED_ARRAY=1`.
- No fuzz target coverage.
- Width selection is per-chunk-greedy. It does not consider merging neighbouring chunks of equal
  width, nor does it trade a wider chunk against the array-wide patch buffer's own compression.

## Tests

Already run and passing (see the measured sections below for the TPC-H runs):

```bash
cargo nextest run -p vortex-fastlanes                          # 342 + 22 new tests
cargo nextest run -p vortex-btrblocks --features unstable_encodings
cargo test -p vortex --features unstable_encodings --lib editions
```

Still to run before this is merge-ready:

```bash
# Workspace-wide, both with and without the feature, to catch registry/edition fallout.
cargo nextest run --workspace
cargo nextest run --workspace --features unstable_encodings

# Lints and formatting.
cargo +nightly fmt --all
cargo clippy --all-targets --all-features

# Docs.
cargo test --doc -p vortex-fastlanes
```

Worth adding:

- A fuzz target case for `BitPackedV2` alongside the existing bit-packing ones in `fuzz/`.
- A `vortex-btrblocks` test that drives `BitPackingV2Scheme::compress` directly (the env-var gate
  makes a compressor-level test order-dependent, so call the scheme, not the compressor).
- Sliced-array serde: the current file round-trip test writes an unsliced array, so a non-zero
  `offset` has only been exercised in memory.

## Measured: TPC-H sf=1 compression

Run on this branch, `data-gen` release build with `--features unstable_encodings`, converting the
same generated Parquet twice — once with `VORTEX_BITPACKED_V2` unset, once with it set to `1`.

| table | v1 bytes | v2 bytes | delta | change |
| --- | ---: | ---: | ---: | ---: |
| customer | 8,681,588 | 8,681,588 | 0 | 0.00% |
| lineitem | 161,637,688 | 161,590,968 | -46,720 | -0.03% |
| nation | 8,508 | 8,812 | +304 | +3.57% |
| orders | 37,253,916 | 34,441,740 | -2,812,176 | -7.55% |
| part | 5,018,416 | 4,958,784 | -59,632 | -1.19% |
| partsupp | 24,646,920 | 24,645,200 | -1,720 | -0.01% |
| region | 5,216 | 5,216 | 0 | 0.00% |
| supplier | 582,832 | 582,832 | 0 | 0.00% |
| **total** | **237,835,084** | **234,915,140** | **-2,919,944** | **-1.23%** |

Reading these:

- **`orders` is the win** (-7.55%, and 96% of the total saving). It is the table whose integer
  columns drift in magnitude down the file, which is exactly the shape chunk-local widths exist
  for. Confirmed reproducible by regenerating `orders.vortex` alone, twice each way: 37,253,916
  bytes without the env var, 34,441,740 with it, both times.
- **`lineitem` barely moves** (-0.03%) despite being 68% of the dataset. Worth understanding
  before drawing conclusions: its integer columns mostly cascade through delta or dictionary
  encoding first, so bit-packing sees a residual that is already uniform. If v2 is meant to pay
  off here, the lever is probably the interaction with the *parent* scheme, not the packing.
- **`nation` regresses** (+304 bytes on a 184-row table). This is the widths-buffer overhead
  showing up where there is nothing to amortise it against, plus the sample-based estimator
  picking v2 on a sample too small to be representative. A guard — skip v2 below some chunk count,
  or require the estimate to beat v1 by a margin rather than a tie — is the obvious fix and is not
  implemented.
- Four tables are byte-identical, i.e. v2 correctly declined to displace v1 where it had nothing
  to offer.

These files were subsequently verified to read back correctly — see the next section.

## Measured: TPC-H sf=1 correctness and query time

`datafusion-bench tpch --formats vortex -i 10`, run against both directories on the same machine.

**Correctness gate: passed.** All 22 queries returned the expected row counts against the v2 files
(`runner.rs` asserts each query's count against `EXPECTED_ROW_COUNTS_SF1`, so a decode bug fails
the run). This exercises compressor-produced v2 arrays through real scans — filters, projections,
joins — including patches and slicing, which the unit tests do not cover.

**Query time: within noise, do not quote these numbers.** Totals came out at -3.5% (v2 faster) at
10 iterations, after +3.3% (v2 slower) at 3 iterations on the same files. Individual queries swung
by ±40% between the two runs and disagreed on sign for most of them. This was measured on a shared
4-core VM; it needs re-running on a quiet machine before anything is concluded.

What *is* worth following up: four queries regressed in both runs, which is the pattern you would
expect from v2's missing kernels rather than from noise — q16, q19, q20 (+21% at 10 iterations,
the largest), q21. If a real regression survives a clean re-run, those queries name the columns
whose filter/take kernels should be written first.

The absence of a large overall regression is itself mildly surprising given that every filter and
take over a v2 column currently canonicalises first. The likely explanation is that at sf=1 the
smaller reads offset the fallback, which would not hold at larger scale factors or on a
filter-heavy workload. Do not read it as evidence that the kernels are unnecessary.

## Benchmarks to run

### 1. TPC-H sf=1 compression ratio (the headline number)

Generate the data once, then convert to Vortex twice — once with the encoding off, once on — and
diff the file sizes. The conversion is idempotent on the output path, so the Vortex directory has
to be moved aside between runs.

```bash
cargo build --release -p vortex-bench --bin data-gen --features unstable_encodings

# Baseline (v1 bit-packing). Generates parquet into vortex-bench/data/tpch/1.0/ as a side effect.
./target/release/data-gen tpch --formats vortex
mv vortex-bench/data/tpch/1.0/vortex-file-compressed \
   vortex-bench/data/tpch/1.0/vortex-baseline

# Chunk-local widths.
VORTEX_BITPACKED_V2=1 ./target/release/data-gen tpch --formats vortex
mv vortex-bench/data/tpch/1.0/vortex-file-compressed \
   vortex-bench/data/tpch/1.0/vortex-v2

du -b vortex-bench/data/tpch/1.0/vortex-baseline/* | sort -k2
du -b vortex-bench/data/tpch/1.0/vortex-v2/* | sort -k2
```

The conversion is idempotent on the output path, so the directory really does have to be moved
between runs or the second one silently no-ops. Note the output directory is named
`vortex-file-compressed`, and the format flag is `--formats vortex` (not `on-disk-vortex`).

Report per-table sizes, not just the total: `lineitem` dominates sf=1, and the interesting columns
are the ones whose magnitude drifts across the file (`l_orderkey`, `o_orderkey`, the date columns
once frame-of-reference has been applied). A column that is uniformly distributed should come out
byte-identical to v1 apart from the widths buffer, so a *regression* on such a column is a signal
that the sample-based estimator is picking v2 when it should not.

Useful cross-check on which encoding each column actually got:

```bash
cargo run --release -p vortex-tui --features native -- <file.vortex>   # encoding tree
```

Note that the compressor's `RUST_LOG=debug` scheme trace does *not* list `vortex.int.bitpacking_v2`
among its `scheme_candidate` spans even when the scheme is active and winning, so do not use the
trace to decide whether v2 was selected — compare file sizes or inspect the encoding tree.

### 2. TPC-H sf=1 query time

Since v2 has no filter/take/compare kernels yet, this is expected to regress and the point is to
measure by how much, and on which queries:

```bash
cargo build --release -p datafusion-bench --features unstable_encodings
./target/release/datafusion-bench tpch --formats vortex -i 10 --hide-progress-bar
```

Run it against both directories produced above (rename the one under test back to
`vortex-file-compressed`). If the regression is confined to queries that push filters into a v2
column, that is the prioritised list for which kernels to write first.

This run doubles as the correctness gate described above: the benchmark compares each query's
result row count against `EXPECTED_ROW_COUNTS_SF1`, so a decode bug in v2 shows up as a wrong
count rather than as a silent corruption.

### 3. Encoding/decoding throughput

`encodings/fastlanes/benches/` has divan benches for v1 (`canonicalize_bench`, `bitpacking_take`,
`compute_between`, `bitpack_compare`). None of them cover v2 yet. The decode path worth measuring
is `decode_into`: full chunks unpack straight into the output buffer, but any chunk clipped by the
array's offset or length goes through a scratch buffer and a copy, so a sliced array pays for two
chunks of copying.

## Open questions for review

1. **Widths as a buffer vs. metadata.** They are a second `BufferHandle` today, so they land in the
   data section and are counted by `nbytes()` (and therefore by the compression estimator). The
   alternative — a `bytes` field in the prost metadata — keeps the buffer count at one but bloats
   the flatbuffer for long columns. Buffer felt right; worth a second opinion.
2. **Should v2 replace v1 rather than compete with it?** v2's packed payload is never larger than
   v1's for the same width policy, so once the kernels exist, keeping both is mostly cost. The
   env-var gate is a stepping stone, not a destination.
3. **Chunk-width smoothing.** Alternating narrow/wide chunks currently store a width byte each. If
   real data shows long runs of equal width, the widths buffer itself could be run-length encoded,
   or the encoder could be biased toward reusing the previous chunk's width when the saving is
   marginal.
