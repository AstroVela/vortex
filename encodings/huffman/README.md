# vortex-huffman

Experimental block-based canonical Huffman codec for byte data — the first step towards
a Vortex encoding based on [PIVCO-Huffman](https://github.com/MarcinZukowski/pivco-huffman)
([paper](https://marcinzukowski.github.io/pivco-huffman/paper-1.0/ph.html)).

## Status

This crate is **not yet a Vortex array encoding**. It contains:

- a safe, dependency-free implementation of the *baseline* codec from the PIVCO work:
  order-0, length-limited (12-bit) canonical Huffman coding over independent 64 KiB
  blocks, with a 4-stream interleaved single-lookup-table decoder (the same structure
  as zstd's huf0 4-stream mode, minus the SIMD);
- benchmarks measuring compression ratio and decompression speed on **real datasets**:
  the ClickBench `hits` `URL` column and Wikipedia text (enwik8).

Per the PIVCO paper, PIVCO's encoded size is within 1–4% of traditional Huffman, so the
compression ratios measured here transfer directly to a future PIVCO implementation.
Decompression speed is where PIVCO differs: its SIMD tree-walk decoder reaches
4–24 GB/s on a single core (2–6× zstd's huf0), while this scalar baseline is expected
to land near huf0's scalar throughput. Porting the PIVCO wire format and its
NEON/AVX-512 kernels is the follow-up step once the ratio numbers justify it.

## Benchmarks

```bash
cargo bench -p vortex-huffman --features _test-harness --bench real_data
```

The bench downloads (~160 MB, once, cached in `~/.cache/vortex-huffman-bench` or
`$VORTEX_HUFFMAN_DATA_DIR`):

- `hits_0.parquet` — first partition of ClickBench hits; the `URL` column is extracted
  and newline-joined;
- `enwik8` — first 10^8 bytes of an English Wikipedia XML dump.

It prints a compression-ratio table (Huffman vs zstd-3 vs FSST, plus the order-0
entropy bound) and then runs divan timing benches for decompression and compression
throughput, counted in uncompressed bytes.

## Results (2026-07, x86-64 cloud VM, single thread)

Compression ratio (raw / compressed), 32 MiB per corpus:

| dataset | order-0 entropy | huffman (this crate) | zstd-3 | fsst | fsst+huffman | onpair | onpair+huffman |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clickbench-urls | 5.516 bpb (bound 1.450x) | 1.443x | 11.311x | 1.950x | 2.246x | 3.029x | 3.632x |
| wikipedia | 5.083 bpb (bound 1.574x) | 1.569x | 2.821x | 1.691x | 1.818x | 1.631x | 1.930x |

Median single-thread decompression throughput (counted in uncompressed bytes), plus
huffman compression for context. `onpair+huffman` is the full stacked decode:
huffman-decode the code stream and dictionary, rebuild + widen the dictionary, then
onpair-decode the rows:

| dataset | huffman | zstd-3 | fsst | onpair | onpair+huffman | huffman compress |
| --- | --- | --- | --- | --- | --- | --- |
| clickbench-urls | 1.01 GB/s | 2.06 GB/s | 1.23 GB/s | 8.76 GB/s | 1.67 GB/s | 148 MB/s |
| wikipedia | 1.01 GB/s | 893 MB/s | 1.17 GB/s | 4.88 GB/s | 976 MB/s | 128 MB/s |

The stacked `onpair+huffman` array tree the bench models (sizes for the 32 MiB URL
corpus; the bench prints both datasets' trees):

```text
OnPair(utf8, 433421 rows, 31.59 MiB of row bytes)
├─ codes:        Huffman(9.20 MB) over PrimitiveArray<u16> x5520224 (11.04 MB raw)
├─ dict_bytes:   Huffman(15.8 KB) over token bytes (21.4 KB raw, 4096 tokens)
├─ dict_offsets: PrimitiveArray<u32> x4097 (16.4 KB)
├─ row_offsets:  PrimitiveArray<u32> x433422 (excluded from ratio accounting,
│                like FSST's; FastLanes-bitpacked in practice)
╰─ validity:     all-valid
```

Observations:

- Huffman lands within 0.5% of the order-0 entropy bound on both corpora, confirming
  the codec is ratio-optimal for what order-0 entropy coding can do — but that bound
  itself (~1.45–1.57x) is far below zstd, which exploits LZ matches these corpora are
  full of.
- Stacked on string-codec output, Huffman removes the order-0 redundancy those codecs
  leave behind: FSST 1.95x → 2.25x (+15%) and OnPair 3.03x → 3.63x (+20%) on URLs;
  FSST 1.69x → 1.82x and OnPair 1.63x → 1.93x (+18%) on Wikipedia. That stacking, not
  standalone Huffman, is the promising integration path for Vortex string columns — it
  keeps the string codec's random-access-friendly per-row structure while shrinking
  storage.
- Stacking costs decode speed with the scalar Huffman: OnPair alone bulk-decodes at
  8.8 / 4.9 GB/s, stacked it drops to 1.67 GB/s / 976 MB/s — the Huffman stage takes
  ~80% of the stacked decode time. Notably, on URLs the stack still *dominates FSST
  alone* (1.9x smaller and ~1.4x faster to decode).
- Scalar 4-stream Huffman decode reaches ~1 GB/s; the PIVCO paper's SIMD decoder
  reports ~4.3–4.9 GB/s single-core on real-text and high-entropy inputs (Apple M4).
  Substituting that for the measured Huffman-stage time projects the stacked decode
  at roughly 5 GB/s on URLs and ~2.9 GB/s on Wikipedia — i.e. with the PIVCO port the
  entropy stage would no longer dominate, which is the case for doing it.

## Format

```text
container := u64-LE raw_len, block*
block     := tag u8 (0=raw, 1=rle, 2=huffman), u32-LE raw_block_len, body
raw body  := raw_block_len bytes
rle body  := u8 symbol
huff body := 128 B nibble-packed code lengths (256 symbols, canonical
             reconstruction), 4 × u32-LE stream byte-lengths, 4 concatenated
             LSB-first bitstreams (block split into 4 equal segments)
```

The decoder is memory-safe on arbitrary input, but corrupt input may decode to
garbage rather than an error.
