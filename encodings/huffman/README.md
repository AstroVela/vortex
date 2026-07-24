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

### The codes child in isolation: Huffman vs primitive encodings

The stacked tree is dominated by the `codes` child (a u16 token stream, dict-12 so
values < 4096), and the honest baseline for that child in Vortex is not raw u16 but a
FastLanes-style 12-bit bit-packed primitive, which decodes at >10 GB/s:

clickbench-urls (5,520,224 tokens):

| representation | size | bits/token | vs raw u16 |
| --- | --- | --- | --- |
| raw u16 primitive | 11040448 B | 16.0 | 1.000x |
| bitpack-12 (FastLanes-style primitive) | 8280336 B | 12.0 | 1.333x |
| huffman, interleaved u16-LE bytes | 9201966 B | 13.34 | 1.200x |
| huffman, split byte planes | 8032979 B | 11.64 | 1.374x |
| token-alphabet entropy bound | 7201596 B | 10.44 | 1.533x |

wikipedia (10,270,440 tokens):

| representation | size | bits/token | vs raw u16 |
| --- | --- | --- | --- |
| raw u16 primitive | 20540880 B | 16.0 | 1.000x |
| bitpack-12 (FastLanes-style primitive) | 15405660 B | 12.0 | 1.333x |
| huffman, interleaved u16-LE bytes | 17355542 B | 13.52 | 1.184x |
| huffman, split byte planes | 15288030 B | 11.91 | 1.344x |
| token-alphabet entropy bound | 13992341 B | 10.90 | 1.468x |

Isolated decode of the codes child runs at ~860 MB/s (interleaved) and ~820 MB/s
(both planes + u16 reassembly), in codes bytes.

Observations:

- Standalone Huffman lands within 0.5% of the order-0 entropy bound on both corpora,
  confirming the codec is ratio-optimal for what order-0 entropy coding can do — but
  that bound itself (~1.45–1.57x) is far below zstd, which exploits LZ matches these
  corpora are full of.
- **The isolation corrects the stacking story.** Byte-oriented Huffman over the
  interleaved u16 code stream (13.3–13.5 bits/token) *loses* to plain 12-bit
  bit-packing; most of the apparent "onpair+huffman" gain over the raw-u16 tree was
  just recovering the packing a primitive encoding gives for free. Splitting the u16
  into two huffman-coded byte planes fixes the mixed-distribution problem but still
  only beats bit-packing by ~3% (URLs) / ~1% (Wikipedia) — at ~820 MB/s versus
  >10 GB/s for bit-unpacking. Recomputing the full tree with bit-packed codes:
  onpair+bitpack reaches 4.03x / 2.17x, versus 4.16x / 2.19x for onpair+plane-huffman.
- What *would* move the needle is a token-alphabet entropy coder: the per-token
  entropy bound is 10.44 / 10.90 bits/token, i.e. +13% / +9% beyond bit-packing
  (full-tree ~4.6x / ~2.4x). That points at Huffman/FSE over 12-bit symbols — note
  PIVCO-Huffman ships a u16-symbol encoder (`pivco_huffman_u16enc.h`) — rather than
  byte-oriented Huffman, for code-stream-shaped children.
- On the raw text itself the stack still holds: on URLs, onpair+huffman decodes at
  1.7 GB/s even with the scalar Huffman stage (~80% of stacked decode time) and beats
  FSST alone on both ratio and speed. The PIVCO paper's SIMD decoder reports
  ~4.3–4.9 GB/s single-core (Apple M4); substituting that for the measured
  Huffman-stage time projects stacked decode at roughly 5 GB/s on URLs and ~2.9 GB/s
  on Wikipedia. A port would target the u16-symbol mode to also capture the
  token-alphabet ratio headroom above.

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
