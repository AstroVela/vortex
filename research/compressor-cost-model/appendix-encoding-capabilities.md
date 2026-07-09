# Appendix: per-encoding execution capabilities (ground truth, July 2026)

*Reference material for docs 3–5: what execution actually does per encoding today. Everything
here was verified against the code at the cited paths; expect it to drift as kernels land.*

## How compute dispatch works (30-second version)

Vortex compute is **deferred**: `filter`/`take`/`compare`/... build lazy op-arrays
(`take` is literally a `Dict` array of codes over values; `compare`/`cast`/`between` are
`ScalarFnArray`s), and `execute` materializes them in layers
(`vortex-array/src/executor.rs`, `docs/developer-guide/internals/execution.md`):

1. `reduce` / `reduce_parent` — metadata-only rewrites.
2. `execute_parent` — **specialized kernels, keyed by (operation, child encoding), stored in a
   session-scoped registry** (`vortex-array/src/optimizer/kernels.rs`; snapshotted into each
   `ExecutionCtx`). This is "pushdown".
3. `execute` — the encoding's own decode-one-step. **No kernel ⇒ canonicalize-then-compute.**

Two consequences for a cost model:

- Whether an (op, encoding) pair is fast is a **registry lookup**, not folklore — the
  compressor could query the same session it runs in.
- The penalty for a missing kernel is not "slightly slower op", it's "pay the whole decode
  first" — a step function, which is why coarse capability bits carry most of the signal.

## Kernel coverage (specialized vs canonicalize-fallback)

✔ = buffer-level `execute_parent` kernel (or equivalent specialized path); blank =
canonicalize-then-compute. The aggregates column lists registered aggregate kernels / stats
short-circuits.

| Encoding | filter | take | compare | between | cast | mask | slice | aggregates | Notes |
|---|---|---|---|---|---|---|---|---|---|
| constant | ✔ | ✔ | reduce | ✔ | ✔ | | ✔ | via reduce | near-free everything |
| dict | reduce | ✔ | ✔ | | ✔ | ✔ | ✔ | min_max, is_constant, is_sorted | compare runs on codes/values |
| runend | ✔ | ✔ | ✔ | | ✔ | | ✔ (bin-search) | min_max, is_constant, is_sorted | |
| fastlanes bitpacking | ✔ | ✔ | ✔ (+fused predicate) | ✔ | ✔ | | ✔ | is_constant | SIMD unpack; richest coverage |
| FoR | | ✔ | ✔ | | ✔ | | | is_constant, is_sorted | |
| sequence | ✔ | ✔ | ✔ | | ✔ | | ✔ | is_sorted, min_max, list_contains | |
| ALP | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | nan_count | |
| ALP-RD | ✔ | ✔ | | | ✔ | ✔ | ✔ | | |
| sparse | ✔ | ✔ | ✔ | ✔ | ✔ | | ✔ | min_max, null_count, sum, … | |
| FSST | ✔ | ✔ | ✔ | | ✔ | | | like, byte_length | LIKE/compare via DFA, **no decompression** |
| **delta** | | | | | ✔ | | | | cast only — everything else decodes |
| **pco** | | | | | ✔ | | | | cast only |
| **zstd** | | | | | ✔ | | | | cast only |
| fastlanes RLE | | | | | ✔ | | ✔ | | |
| zigzag | | ✔ | | | ✔ | | | | |
| datetime-parts | ✔ | ✔ | ✔ | | ✔ | ✔ | ✔ | is_constant | |
| decimal-byte-parts | ✔ | ✔ | ✔ | | ✔ | ✔ | | is_constant | |

(Registration sites: per-encoding `kernel.rs`/`kernels.rs`/`rules.rs` files, e.g.
`encodings/fastlanes/src/bitpacking/vtable/kernels.rs:25-38`,
`encodings/runend/src/kernel.rs:31-35`, `encodings/alp/src/alp/rules.rs:27-31`,
`encodings/fsst/src/kernel.rs:25-30`, `vortex-array/src/arrays/dict/vtable/kernel.rs:19-22`.)

The pattern worth naming: **the encodings added by `with_compact()` (zstd, pco) are exactly
the ones with no compute coverage and block-decode random access** — today's "compact = slower"
folklore is visible in the registry.

## Random-access (`scalar_at`) classes

| Class | Encodings | Mechanism |
|---|---|---|
| O(1) | primitive, bool, decimal, varbin(view), constant; bitpacking (`unpack_single` + patch check); FoR (child + reference); zigzag; sequence (`start + step·i`); dict (two chained O(1) reads) | direct or arithmetic |
| O(log n) | runend (search run ends), sparse (search patches), chunked (search chunk bounds) | binary search |
| Block decode | **delta** (slices 1 row ⇒ decodes a whole cumulative chunk), **zstd**, **pco** (decompress a 1-row slice ⇒ block decode), FSST (per-value decompress — cheap-ish) | pay (part of) the decode |

## Decode-path speed notes

- **bitpacking**: SIMD unpack straight into the output buffer, 1024-value chunks
  (`bitpack_decompress.rs:28-66`) — near memory speed.
- **FSST**: bulk-decompresses the whole string heap in one pass (`fsst/src/canonical.rs:52-75`).
- **dict**: gather; Dict-over-RLE fused decompression is an explicitly implemented optimization
  (`execution.md:262-281`).
- **delta**: cumulative (prefix-sum-like) decode per 1024-lane chunk — inherently sequential
  (also called out in `only_cuda_compatible`, `vortex-btrblocks/src/builder.rs:180-183`).
- **zstd/pco**: general-purpose block decompressors.

## Existing benches usable for calibration

Per-encoding criterion suites already isolate many of the relevant costs:
`encodings/runend/benches/{run_end_decode,run_end_filter,run_end_take}.rs`,
`encodings/fastlanes/benches/{canonicalize_bench,bitpacking_take,bitpack_compare,compute_between}.rs`,
`encodings/fsst/benches/{fsst_like,fsst_url_compare}.rs`,
`encodings/sparse/benches/{sparse_canonical,sparse_pushdown}.rs`, plus op-level benches in
`vortex-array/benches/` (compare, take, filter, aggregates) and end-to-end
`benchmarks/compress-bench` (encode/decode throughput) and
`benchmarks/random-access-bench`. Gaps: no isolated decode benches for ALP/dict-gather/pco/
zstd/delta; nothing emits a machine-readable cost table yet (see doc 5, phase 0).
