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
behind. Exceptions become `PatchesV2`: indices are chunk-local `u16` values and `u32` prefix
offsets delimit each chunk, avoiding the global-index v1 representation. Null slots are counted as
zero-width and are never patched, since their values are undefined.

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

Build with `--features unstable_encodings`. Preview builds register `BitPackingV2Scheme` instead of
`BitPackingScheme` and also use v2 for the encoded child of frame-of-reference arrays. Stable builds
retain v1 because the preview edition is required to serialize v2.

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

Already run and passing:

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

## Benchmarks to run

### 1. TPC-H sf=1 compression ratio (the headline number)

Generate the data once, then convert to Vortex twice — once with a stable build (v1) and once with
an `unstable_encodings` build (v2) — and diff the file sizes. The conversion is idempotent on the
output path, so the Vortex directory has to be moved aside between runs.

```bash
# Baseline (v1 bit-packing). Generates parquet into vortex-bench/data/tpch/1.0/ as a side effect.
cargo build --release -p vortex-bench --bin data-gen
./target/release/data-gen tpch --formats vortex
mv vortex-bench/data/tpch/1.0/vortex-file-compressed vortex-bench/data/tpch/1.0/vortex-baseline

# Chunk-local widths.
cargo build --release -p vortex-bench --bin data-gen --features unstable_encodings
./target/release/data-gen tpch --formats vortex
mv vortex-bench/data/tpch/1.0/vortex-file-compressed vortex-bench/data/tpch/1.0/vortex-v2

du -b vortex-bench/data/tpch/1.0/vortex-baseline/* | sort -k2
du -b vortex-bench/data/tpch/1.0/vortex-v2/* | sort -k2
```

Report per-table sizes, not just the total: `lineitem` dominates sf=1, and the interesting columns
are the ones whose magnitude drifts across the file (`l_orderkey`, `o_orderkey`, the date columns
once frame-of-reference has been applied). A column that is uniformly distributed should come out
byte-identical to v1 apart from the widths buffer, so a *regression* on such a column is a signal
that the sample-based estimator is picking v2 when it should not.

Useful cross-check on which encoding each column actually got:

```bash
cargo run --release -p vortex-tui -- <file.vortex>   # inspect the encoding tree
```

### 2. TPC-H sf=1 query time

Since v2 has no filter/take/compare kernels yet, this is expected to regress and the point is to
measure by how much, and on which queries:

```bash
cargo run --release -p datafusion-bench --features unstable_encodings -- \
    --benchmark tpch --formats on-disk-vortex
```

Run it against both directories produced above. If the regression is confined to queries that
push filters into a v2 column, that is the prioritised list for which kernels to write first.

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
