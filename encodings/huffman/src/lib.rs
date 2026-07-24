// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Block-based canonical Huffman codec for byte data.
//!
//! This crate is the first step towards a Vortex encoding based on
//! [PIVCO-Huffman](https://github.com/MarcinZukowski/pivco-huffman). It implements the
//! *baseline* codec from that work — order-0, length-limited canonical Huffman coding —
//! in safe, dependency-free Rust, so that compression ratio and decompression speed can
//! be measured on real datasets before committing to the full PIVCO wire format and its
//! SIMD tree-walk decoder. Per the PIVCO paper, PIVCO's encoded size is within 1-4% of
//! traditional Huffman, so the ratios measured here transfer directly.
//!
//! Design:
//!
//! - The input is split into independent blocks (default 64 KiB), each with its own
//!   Huffman table, mirroring how a columnar file format would compress chunk-by-chunk.
//! - Code lengths are computed with the package-merge algorithm, limited to
//!   [`MAX_CODE_LEN`] bits so the decoder can use a single-lookup table
//!   (`2^12` entries, 8 KiB, L1-resident).
//! - Each block's payload is encoded into [`NUM_STREAMS`] independent bitstreams that
//!   the decoder consumes in an interleaved loop, giving the CPU instruction-level
//!   parallelism across streams (the same trick as zstd's huf0 4-stream mode).
//! - Blocks that would not shrink are stored raw; single-symbol blocks are run-length
//!   encoded.
//!
//! The decoder is memory-safe on arbitrary input but does not validate that corrupt
//! input round-trips to meaningful data.

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

#[cfg(test)]
mod tests;

/// Maximum Huffman code length in bits. Chosen so the decode table has `2^12`
/// entries and stays L1-resident.
pub const MAX_CODE_LEN: usize = 12;

/// Number of independent interleaved bitstreams per block.
pub const NUM_STREAMS: usize = 4;

/// Default block length used by [`compress`].
pub const DEFAULT_BLOCK_LEN: usize = 64 * 1024;

const TABLE_SIZE: usize = 1 << MAX_CODE_LEN;

const BLOCK_RAW: u8 = 0;
const BLOCK_RLE: u8 = 1;
const BLOCK_HUFF: u8 = 2;

/// Per-block header past the tag byte and raw-length u32: 128 bytes of nibble-packed
/// code lengths plus four u32 stream lengths.
const HUFF_HEADER_LEN: usize = 128 + NUM_STREAMS * 4;

/// Compress `input` with the default block length of [`DEFAULT_BLOCK_LEN`].
pub fn compress(input: &[u8]) -> Vec<u8> {
    compress_with_block_len(input, DEFAULT_BLOCK_LEN)
}

/// Compress `input`, splitting it into independently-coded blocks of `block_len` bytes.
///
/// Larger blocks amortize the per-block table header (149 bytes) better; smaller blocks
/// adapt faster to local distribution shifts.
pub fn compress_with_block_len(input: &[u8], block_len: usize) -> Vec<u8> {
    // Blocks must fit the u32 length fields; 2^30 is far above any sensible block size.
    let block_len = block_len.clamp(1, 1 << 30);
    let mut out = Vec::with_capacity(input.len() / 2 + 16);
    out.extend_from_slice(&(input.len() as u64).to_le_bytes());
    for block in input.chunks(block_len) {
        compress_block(block, &mut out);
    }
    out
}

/// Decompressed length of a buffer produced by [`compress`].
pub fn decompressed_len(data: &[u8]) -> VortexResult<usize> {
    let header = data
        .get(..8)
        .ok_or_else(|| vortex_err!("huffman: truncated container header"))?;
    let raw_len = u64::from_le_bytes(header.try_into().map_err(|_| vortex_err!("unreachable"))?);
    usize::try_from(raw_len).map_err(|_| vortex_err!("huffman: length overflows usize"))
}

/// Decompress a buffer produced by [`compress`].
pub fn decompress(data: &[u8]) -> VortexResult<Vec<u8>> {
    let mut out = vec![0u8; decompressed_len(data)?];
    decompress_into(data, &mut out)?;
    Ok(out)
}

/// Decompress a buffer produced by [`compress`] into a caller-provided buffer whose
/// length must equal [`decompressed_len`].
pub fn decompress_into(data: &[u8], out: &mut [u8]) -> VortexResult<()> {
    let raw_len = decompressed_len(data)?;
    if out.len() != raw_len {
        vortex_bail!(
            "huffman: output buffer length {} != decompressed length {}",
            out.len(),
            raw_len
        );
    }
    let mut rest = &data[8..];
    let mut filled = 0usize;
    while filled < raw_len {
        let consumed = decompress_block(rest, &mut out[filled..])?;
        rest = &rest[consumed.0..];
        filled += consumed.1;
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Block encoding
////////////////////////////////////////////////////////////////////////////////////////////////////

fn compress_block(block: &[u8], out: &mut Vec<u8>) {
    debug_assert!(!block.is_empty());
    let mut hist = [0u64; 256];
    for &byte in block {
        hist[usize::from(byte)] += 1;
    }
    let distinct = hist.iter().filter(|&&count| count > 0).count();

    // `block.len() as u32` is safe: blocks are clamped to 2^30 bytes.
    #[allow(clippy::cast_possible_truncation)]
    let block_len_u32 = block.len() as u32;

    if distinct == 1 {
        out.push(BLOCK_RLE);
        out.extend_from_slice(&block_len_u32.to_le_bytes());
        out.push(block[0]);
        return;
    }

    let lens = build_code_lengths(&hist);
    let payload_bits: u64 = hist
        .iter()
        .zip(lens.iter())
        .map(|(&count, &len)| count * u64::from(len))
        .sum();
    // Each stream may waste up to 7 bits of padding.
    let estimate = payload_bits.div_ceil(8) + (HUFF_HEADER_LEN + NUM_STREAMS) as u64;
    if estimate >= block.len() as u64 {
        out.push(BLOCK_RAW);
        out.extend_from_slice(&block_len_u32.to_le_bytes());
        out.extend_from_slice(block);
        return;
    }

    let enc = EncTable::new(&lens);
    out.push(BLOCK_HUFF);
    out.extend_from_slice(&block_len_u32.to_le_bytes());
    for pair in lens.chunks_exact(2) {
        out.push(pair[0] | (pair[1] << 4));
    }
    let streams: [Vec<u8>; NUM_STREAMS] = encode_streams(block, &enc);
    for stream in &streams {
        // Stream lengths are bounded by the block length, which fits u32.
        #[allow(clippy::cast_possible_truncation)]
        let stream_len = stream.len() as u32;
        out.extend_from_slice(&stream_len.to_le_bytes());
    }
    for stream in &streams {
        out.extend_from_slice(stream);
    }
}

/// Byte ranges of the [`NUM_STREAMS`] segments a block of `len` bytes is split into.
fn segment_lens(len: usize) -> [usize; NUM_STREAMS] {
    let seg = len.div_ceil(NUM_STREAMS);
    std::array::from_fn(|i| len.min(i * seg + seg) - len.min(i * seg))
}

fn encode_streams(block: &[u8], enc: &EncTable) -> [Vec<u8>; NUM_STREAMS] {
    let seg = block.len().div_ceil(NUM_STREAMS);
    std::array::from_fn(|i| {
        let start = block.len().min(i * seg);
        let end = block.len().min(start + seg);
        let mut writer = BitWriter::default();
        for &byte in &block[start..end] {
            writer.push(enc.codes[usize::from(byte)], enc.lens[usize::from(byte)]);
        }
        writer.finish()
    })
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Code construction
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Optimal length-limited code lengths (≤ [`MAX_CODE_LEN`]) via package-merge.
///
/// Returns a length per symbol, 0 for absent symbols. Requires at least two distinct
/// symbols present.
fn build_code_lengths(hist: &[u64; 256]) -> [u8; 256] {
    #[derive(Clone)]
    struct Item {
        weight: u64,
        syms: Vec<u8>,
    }

    let originals: Vec<Item> = (0u8..=255)
        .filter(|&sym| hist[usize::from(sym)] > 0)
        .map(|sym| Item {
            weight: hist[usize::from(sym)],
            syms: vec![sym],
        })
        .collect();
    let num_symbols = originals.len();
    debug_assert!(num_symbols >= 2);

    let mut originals = originals;
    originals.sort_by_key(|item| item.weight);

    // `list` holds the current level's coins, sorted by weight, starting at the deepest
    // level. Each of the MAX_CODE_LEN-1 iterations packages pairs and merges in a fresh
    // copy of the per-symbol coins for the next level up.
    let mut list = originals.clone();
    for _ in 1..MAX_CODE_LEN {
        let mut packaged: Vec<Item> = list
            .chunks_exact(2)
            .map(|pair| Item {
                weight: pair[0].weight + pair[1].weight,
                syms: [pair[0].syms.as_slice(), pair[1].syms.as_slice()].concat(),
            })
            .collect();
        packaged.extend(originals.iter().cloned());
        packaged.sort_by_key(|item| item.weight);
        list = packaged;
    }

    let mut lens = [0u8; 256];
    for item in list.iter().take(2 * num_symbols - 2) {
        for &sym in &item.syms {
            lens[usize::from(sym)] += 1;
        }
    }
    debug_assert!(lens.iter().all(|&len| usize::from(len) <= MAX_CODE_LEN));
    debug_assert!(
        lens.iter()
            .filter(|&&len| len > 0)
            .map(|&len| 1u64 << (MAX_CODE_LEN - usize::from(len)))
            .sum::<u64>()
            <= 1 << MAX_CODE_LEN
    );
    lens
}

/// Per-symbol canonical codes, bit-reversed for LSB-first bitstream emission.
struct EncTable {
    codes: [u16; 256],
    lens: [u8; 256],
}

impl EncTable {
    fn new(lens: &[u8; 256]) -> Self {
        let mut codes = [0u16; 256];
        let mut next_code = canonical_first_codes(lens);
        for sym in 0..256 {
            let len = lens[sym];
            if len > 0 {
                let code = next_code[usize::from(len)];
                next_code[usize::from(len)] += 1;
                codes[sym] = reverse_code(code, len);
            }
        }
        Self { codes, lens: *lens }
    }
}

/// First canonical (MSB-first) code for each length, RFC1951-style.
fn canonical_first_codes(lens: &[u8; 256]) -> [u16; MAX_CODE_LEN + 1] {
    let mut bl_count = [0u16; MAX_CODE_LEN + 1];
    for &len in lens {
        if len > 0 {
            bl_count[usize::from(len)] += 1;
        }
    }
    let mut next_code = [0u16; MAX_CODE_LEN + 1];
    let mut code = 0u16;
    for bits in 1..=MAX_CODE_LEN {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }
    next_code
}

fn reverse_code(code: u16, len: u8) -> u16 {
    code.reverse_bits() >> (16 - u32::from(len))
}

/// Single-lookup decode table: the low [`MAX_CODE_LEN`] bits of the bit-buffer index
/// into `(symbol, code_len)` pairs.
struct DecTable(Vec<[u8; 2]>);

impl DecTable {
    fn new(lens: &[u8; 256]) -> Self {
        let mut table = vec![[0u8, 0u8]; TABLE_SIZE];
        let mut next_code = canonical_first_codes(lens);
        for sym in 0u8..=255 {
            let len = lens[usize::from(sym)];
            if len == 0 {
                continue;
            }
            let code = next_code[usize::from(len)];
            next_code[usize::from(len)] += 1;
            let reversed = usize::from(reverse_code(code, len));
            let step = 1usize << len;
            let mut idx = reversed;
            while idx < TABLE_SIZE {
                table[idx] = [sym, len];
                idx += step;
            }
        }
        Self(table)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Bitstreams
////////////////////////////////////////////////////////////////////////////////////////////////////

/// LSB-first bit writer.
#[derive(Default)]
struct BitWriter {
    buf: Vec<u8>,
    bitbuf: u64,
    nbits: usize,
}

impl BitWriter {
    #[inline]
    fn push(&mut self, code: u16, len: u8) {
        self.bitbuf |= u64::from(code) << self.nbits;
        self.nbits += usize::from(len);
        while self.nbits >= 8 {
            // Truncation to the low byte is intentional.
            #[allow(clippy::cast_possible_truncation)]
            self.buf.push(self.bitbuf as u8);
            self.bitbuf >>= 8;
            self.nbits -= 8;
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            #[allow(clippy::cast_possible_truncation)]
            self.buf.push(self.bitbuf as u8);
        }
        self.buf
    }
}

/// LSB-first bit reader over one stream, with a branch-light 56-bit refill.
struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    bitbuf: u64,
    nbits: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            bitbuf: 0,
            nbits: 0,
        }
    }

    /// Top the bit-buffer up to at least 56 valid bits (fewer only near end-of-stream,
    /// where the missing bits read as zeros).
    #[inline]
    fn refill(&mut self) {
        if self.pos + 8 <= self.data.len() {
            let mut word = [0u8; 8];
            word.copy_from_slice(&self.data[self.pos..self.pos + 8]);
            self.bitbuf |= u64::from_le_bytes(word) << self.nbits;
            self.pos += (63 - self.nbits) >> 3;
            self.nbits |= 56;
        } else {
            while self.nbits < 56 && self.pos < self.data.len() {
                self.bitbuf |= u64::from(self.data[self.pos]) << self.nbits;
                self.pos += 1;
                self.nbits += 8;
            }
        }
    }

    /// Decode one symbol. Requires `refill` to have been called recently enough that
    /// the buffered bits cover the next code (up to [`MAX_CODE_LEN`] bits).
    #[inline]
    // The table index is masked to MAX_CODE_LEN bits, so the cast cannot truncate.
    #[allow(clippy::cast_possible_truncation)]
    fn decode_one(&mut self, table: &DecTable) -> u8 {
        let [sym, len] = table.0[(self.bitbuf & (TABLE_SIZE as u64 - 1)) as usize];
        self.bitbuf >>= usize::from(len);
        self.nbits = self.nbits.saturating_sub(usize::from(len));
        sym
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Block decoding
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Decode one block from the front of `data` into the front of `out`.
/// Returns `(bytes consumed from data, bytes written to out)`.
fn decompress_block(data: &[u8], out: &mut [u8]) -> VortexResult<(usize, usize)> {
    let (&tag, rest) = data
        .split_first()
        .ok_or_else(|| vortex_err!("huffman: truncated block tag"))?;
    let (len_bytes, rest) = rest
        .split_at_checked(4)
        .ok_or_else(|| vortex_err!("huffman: truncated block length"))?;
    let block_len = u32::from_le_bytes(
        len_bytes
            .try_into()
            .map_err(|_| vortex_err!("unreachable"))?,
    );
    let block_len =
        usize::try_from(block_len).map_err(|_| vortex_err!("huffman: block length overflow"))?;
    if block_len == 0 || block_len > out.len() {
        vortex_bail!(
            "huffman: block length {} exceeds remaining output",
            block_len
        );
    }
    let out = &mut out[..block_len];

    match tag {
        BLOCK_RAW => {
            let payload = rest
                .get(..block_len)
                .ok_or_else(|| vortex_err!("huffman: truncated raw block"))?;
            out.copy_from_slice(payload);
            Ok((1 + 4 + block_len, block_len))
        }
        BLOCK_RLE => {
            let &sym = rest
                .first()
                .ok_or_else(|| vortex_err!("huffman: truncated rle block"))?;
            out.fill(sym);
            Ok((1 + 4 + 1, block_len))
        }
        BLOCK_HUFF => {
            let consumed = decompress_huff_block(rest, out)?;
            Ok((1 + 4 + consumed, block_len))
        }
        other => vortex_bail!("huffman: unknown block tag {}", other),
    }
}

fn decompress_huff_block(data: &[u8], out: &mut [u8]) -> VortexResult<usize> {
    let (header, payload) = data
        .split_at_checked(HUFF_HEADER_LEN)
        .ok_or_else(|| vortex_err!("huffman: truncated huffman block header"))?;

    let mut lens = [0u8; 256];
    for (pair_idx, &packed) in header[..128].iter().enumerate() {
        lens[pair_idx * 2] = packed & 0x0F;
        lens[pair_idx * 2 + 1] = packed >> 4;
    }
    let table = DecTable::new(&lens);

    let mut stream_lens = [0usize; NUM_STREAMS];
    for (stream_idx, len_bytes) in header[128..].chunks_exact(4).enumerate() {
        let stream_len = u32::from_le_bytes(
            len_bytes
                .try_into()
                .map_err(|_| vortex_err!("unreachable"))?,
        );
        stream_lens[stream_idx] =
            usize::try_from(stream_len).map_err(|_| vortex_err!("huffman: stream len overflow"))?;
    }
    let total_stream_len: usize = stream_lens.iter().sum();
    if payload.len() < total_stream_len {
        vortex_bail!("huffman: truncated huffman block payload");
    }

    let seg_lens = segment_lens(out.len());
    decode_streams(payload, &stream_lens, &seg_lens, &table, out);
    Ok(HUFF_HEADER_LEN + total_stream_len)
}

fn decode_streams(
    payload: &[u8],
    stream_lens: &[usize; NUM_STREAMS],
    seg_lens: &[usize; NUM_STREAMS],
    table: &DecTable,
    out: &mut [u8],
) {
    let readers: [BitReader<'_>; NUM_STREAMS] = {
        let mut offset = 0usize;
        std::array::from_fn(|i| {
            let reader = BitReader::new(&payload[offset..offset + stream_lens[i]]);
            offset += stream_lens[i];
            reader
        })
    };

    let mut outs: [&mut [u8]; NUM_STREAMS] = {
        let (seg0, rest) = out.split_at_mut(seg_lens[0]);
        let (seg1, rest) = rest.split_at_mut(seg_lens[1]);
        let (seg2, seg3) = rest.split_at_mut(seg_lens[2]);
        [seg0, seg1, seg2, seg3]
    };

    // Interleaved main loop: per round, each stream refills once (>= 56 bits) and
    // decodes 4 symbols (<= 48 bits), so the four streams' loads and table lookups
    // overlap in the pipeline. Streams are unrolled into locals so the readers stay
    // in registers.
    let [mut reader0, mut reader1, mut reader2, mut reader3] = readers;
    let [out0, out1, out2, out3] = &mut outs;
    let rounds = seg_lens[NUM_STREAMS - 1] / 4;
    let mut base = 0usize;
    for _ in 0..rounds {
        reader0.refill();
        reader1.refill();
        reader2.refill();
        reader3.refill();
        let dst0 = &mut out0[base..base + 4];
        dst0[0] = reader0.decode_one(table);
        dst0[1] = reader0.decode_one(table);
        dst0[2] = reader0.decode_one(table);
        dst0[3] = reader0.decode_one(table);
        let dst1 = &mut out1[base..base + 4];
        dst1[0] = reader1.decode_one(table);
        dst1[1] = reader1.decode_one(table);
        dst1[2] = reader1.decode_one(table);
        dst1[3] = reader1.decode_one(table);
        let dst2 = &mut out2[base..base + 4];
        dst2[0] = reader2.decode_one(table);
        dst2[1] = reader2.decode_one(table);
        dst2[2] = reader2.decode_one(table);
        dst2[3] = reader2.decode_one(table);
        let dst3 = &mut out3[base..base + 4];
        dst3[0] = reader3.decode_one(table);
        dst3[1] = reader3.decode_one(table);
        dst3[2] = reader3.decode_one(table);
        dst3[3] = reader3.decode_one(table);
        base += 4;
    }
    let mut readers = [reader0, reader1, reader2, reader3];
    for stream in 0..NUM_STREAMS {
        let reader = &mut readers[stream];
        for idx in (rounds * 4)..seg_lens[stream] {
            reader.refill();
            outs[stream][idx] = reader.decode_one(table);
        }
    }
}
