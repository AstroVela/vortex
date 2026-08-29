# Vortex OnPair

A Vortex Encoding for Binary and Utf8 data that uses the
[OnPair][onpair] short-string compression algorithm. OnPair is a
dictionary-based encoder with fast per-row random access.

The trainer / encoder lives in the standalone [`onpair`][onpair-crate]
crate; this crate wraps the resulting column as a Vortex array with
cascading-compressor support on every integer child.

## Compute

Like the FSST encoding, this crate pushes down common operations over the
encoded representation. It supports `cast`, `filter`, byte length, and
constant equality / inequality. Unsupported operators fall back to ordinary
decompression.

## Default Configuration

The default training configuration uses OnPair's default dictionary budget and
a fixed seed. Vortex stores token codes as an integer child array; downstream
integer compression may narrow or bit-pack that child independently.

## Layout

- Buffer 0 — `dict_bytes`: dictionary blob built by the OnPair trainer,
  including the read padding required by the decoder.
- Slot 0 — `dict_offsets`: integer child, len `dict_size + 1`.
- Slot 1 — `codes`: integer child, length `total_tokens`.
- Slot 2 — `codes_offsets`: integer child, length `num_rows + 1`.
- Slot 3 — `uncompressed_lengths`: integer child, length `num_rows`.
- Slot 4 — optional validity child.

All four integer slot children flow through the standard cascading
compressor pipeline (FoR / BitPacking / RunEnd / etc.).

## Compress Path

`onpair_compress` gathers the array's valid rows into the contiguous
`(bytes, offsets)` pair `onpair::compress` accepts, then wraps what comes back
as the slot children above. Byte offsets are gathered at the narrowest width
that spans the corpus — `u32` for any chunk up to 4 GiB, which is every chunk
in practice. That width is also the width the library reports its row layer in,
so `codes_offsets` adopts that buffer directly instead of re-collecting it.
The dictionary blob and the code stream are likewise adopted, not copied.

Wall-clock on a 1 M-row, 44 MiB URL/log column (`benches/compress.rs`,
`(UrlLog, 1000000)`) splits roughly:

| Phase                             |     Time | Share |
| --------------------------------- | -------: | ----: |
| `onpair::Parser::parse` (encode)  | ~200 ms  |   66% |
| `onpair::Parser::train`           |  ~88 ms  |   29% |
| gather (flatten + offsets)        |  ~11 ms  |  3.7% |
| canonicalize input                | ~1.7 ms  |  0.6% |
| wrap as slot children             |  ~23 µs  | <0.1% |

So the Vortex side of compression is already close to free, and the headroom
is upstream. What follows is what the library would have to expose for the
remaining redundancy to go away.

## Upstream API Opportunities

Ordered by the size of the win, largest first.

1. **Reusable trained encoder across the estimate and the real compress.**
   `BtrBlocksCompressor` picks OnPair via `DeferredEstimate::Sample`, which
   compresses a sample of the column and then compresses the whole column
   again — training two dictionaries and keeping the second. `Parser` is
   already public, `Clone`, and separable from encoding, so this needs no new
   upstream surface, only a Vortex-side channel that carries the sample's
   `Parser` into `Scheme::compress`. OnPair itself trains on a 15 % byte
   sample, so a dictionary trained on the compressor's sample should be
   comparable; the ratio needs measuring before this is taken. Worth ~29 % of
   compress time.

2. **A range-scoped encode, so the encode pass can be split across threads.**
   Encoding is a pure function of the trained matcher and one row range, but
   `Parser::parse` only encodes a whole corpus into freshly allocated vectors.
   An `encode_range_into(rows, range, &mut codes, &mut row_offsets)` would let
   Vortex fan the 66 % phase out over rayon and stitch the row layer with a
   prefix sum. Nothing about the algorithm is sequential across rows.

3. **A row source instead of a contiguous `(bytes, offsets)` pair.** Neither
   `train` (which walks rows in shuffled index order) nor `encode_strings`
   (sequential) ever reads across a row boundary, so the contiguity
   requirement is incidental. An index-addressable source —
   `trait RowSource { fn len(&self) -> usize; fn row(&self, i: usize) -> &[u8];
   fn total_bytes(&self) -> usize; }` — would let a `VarBinViewArray` be fed
   in place and delete the gather entirely: a corpus-sized allocation plus a
   copy of every byte, per chunk. Note that `&[&[u8]]` (the shape
   `fsst::Compressor::train` takes) would not do: 16 bytes per row is worse
   than the 4 bytes of offsets it replaces.

4. **Caller-owned output buffers for the code stream.**
   `encoding::parser::encode_strings` sizes its code vector for the worst case
   of one token per corpus byte, i.e. `2 * corpus_bytes`. On the URL column
   above that is an 89 MiB allocation holding 12.7 MiB of codes — a 7×
   over-allocation that would otherwise stay resident for as long as the
   encoded array. `onpair_compress` reclaims it with a gated `shrink_to_fit`,
   but an `encode_into(&mut Vec<Token>, ..)` taking a caller-owned (and
   reusable across chunks) buffer would avoid the round trip, and would let
   Vortex hand down a `BufferMut<u16>` directly.

5. **Move the dictionary into the column instead of cloning it.**
   `Parser::parse` clones its `CompactDictionary` into every `Column` it
   produces. For the one-shot `onpair::compress` path the parser is dropped
   immediately afterwards, so a consuming `Parser::into_column` (or an
   `Arc`-shared dictionary) would drop a copy of the whole dictionary blob.
   Small in absolute terms — the blob is at most `2^max_dict_bits *
   MAX_TOKEN_SIZE` bytes — but it is pure waste.

6. **An alignment guarantee on the dictionary blob.** Vortex publishes
   `dict_bytes` 8-aligned so the segment holding it deserializes cleanly.
   The allocator satisfies that for a dictionary-sized block in practice, so
   `Buffer::aligned` normally just claims it, but the fallback is a copy.
   Either a documented alignment guarantee, or letting the caller supply the
   blob's allocation, would remove the fallback.

[onpair]: https://arxiv.org/abs/2508.02280
[onpair-crate]: https://github.com/spiraldb/onpair
