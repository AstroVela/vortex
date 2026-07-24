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

| dataset | order-0 entropy | huffman (this crate) | zstd-3 | fsst | fsst+huffman |
| --- | --- | --- | --- | --- | --- |
| clickbench-urls | 5.516 bpb (bound 1.450x) | 1.443x | 11.311x | 1.950x | 2.246x |
| wikipedia | 5.083 bpb (bound 1.574x) | 1.569x | 2.821x | 1.691x | 1.818x |

Median single-thread throughput (counted in uncompressed bytes):

| dataset | huffman decompress | zstd-3 decompress | fsst decompress | huffman compress |
| --- | --- | --- | --- | --- |
| clickbench-urls | 969 MB/s | 1.67 GB/s | 930 MB/s | 137 MB/s |
| wikipedia | 976 MB/s | 802 MB/s | 877 MB/s | 117 MB/s |

Observations:

- Huffman lands within 0.5% of the order-0 entropy bound on both corpora, confirming
  the codec is ratio-optimal for what order-0 entropy coding can do — but that bound
  itself (~1.45–1.57x) is far below zstd, which exploits LZ matches these corpora are
  full of.
- Stacked on FSST output (`fsst+huffman`), Huffman removes the order-0 redundancy FSST
  leaves behind: 1.95x → 2.25x on URLs (+15%) and 1.69x → 1.82x on Wikipedia. That
  stacking, not standalone Huffman, is the promising integration path for Vortex string
  columns — it keeps FSST's random-access-friendly structure per block while shrinking
  storage.
- Scalar 4-stream decode reaches ~1 GB/s. The PIVCO paper's SIMD decoder reports
  4.3–4.8 GB/s on real text on Apple M4, which is the motivation for the follow-up port.

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
