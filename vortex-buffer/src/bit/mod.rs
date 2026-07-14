// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Packed bitmaps that can be used to store boolean values.
//!
//! This module provides a wrapper on top of the `Buffer` type to store mutable and immutable
//! bitsets. The bitsets are stored in little-endian order, meaning that the least significant bit
//! of the first byte is the first bit in the bitset.
#[cfg(feature = "arrow")]
mod arrow;
mod buf;
mod buf_mut;
mod count_ones;
mod macros;
mod meta;
mod ops;
mod select;
mod view;

pub use arrow_buffer::bit_chunk_iterator::BitChunkIterator;
pub use arrow_buffer::bit_chunk_iterator::BitChunks;
pub use arrow_buffer::bit_chunk_iterator::UnalignedBitChunk;
pub use arrow_buffer::bit_chunk_iterator::UnalignedBitChunkIterator;
pub use arrow_buffer::bit_iterator::BitIndexIterator;
pub use arrow_buffer::bit_iterator::BitIterator;
pub use arrow_buffer::bit_iterator::BitSliceIterator;
pub use buf::*;
pub use buf_mut::*;
pub use meta::*;
pub use view::*;

/// Packs up to 64 boolean values into a little-endian `u64` word.
#[inline]
pub fn collect_bool_word<F>(len: usize, mut f: F) -> u64
where
    F: FnMut(usize) -> bool,
{
    assert!(len <= 64, "cannot pack {len} bits into a u64 word");

    if len == 64 {
        return collect_bool_word_full(f);
    }
    let mut packed = 0;
    for bit_idx in 0..len {
        packed |= (f(bit_idx) as u64) << bit_idx;
    }
    packed
}

/// Packs exactly 64 boolean values into a `u64` word, LSB-first.
///
/// Stages the values as one byte each and then packs the whole chunk, instead of
/// or-shifting bit by bit: LLVM never turns the or-shift reduction into a movemask, while
/// the staging loop auto-vectorizes cleanly and the byte pack lowers to `pmovmskb` on
/// x86_64 (5-7x faster end-to-end than the or-shift form).
#[inline]
fn collect_bool_word_full<F>(mut f: F) -> u64
where
    F: FnMut(usize) -> bool,
{
    let mut bytes = [0u8; 64];
    for (bit_idx, byte) in bytes.iter_mut().enumerate() {
        *byte = f(bit_idx) as u8;
    }
    pack_word_from_bytes(&bytes)
}

/// Packs 64 bytes that are each 0 or 1 into a `u64` bitmask, LSB-first.
#[inline]
fn pack_word_from_bytes(bytes: &[u8; 64]) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: sse2 is a baseline feature of x86_64, so it is always available.
        unsafe { pack_word_from_bytes_sse2(bytes) }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        pack_word_from_bytes_swar(bytes)
    }
}

/// `pmovmskb`-based pack: 4x (load, shift the 0/1 byte into the sign bit, movemask).
///
/// Building the vectors via `u64::from_le_bytes` + `_mm_set_epi64x` keeps this free of
/// pointer intrinsics; LLVM folds each pair into a single 16-byte load.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
fn pack_word_from_bytes_sse2(bytes: &[u8; 64]) -> u64 {
    use core::arch::x86_64::_mm_movemask_epi8;
    use core::arch::x86_64::_mm_set_epi64x;
    use core::arch::x86_64::_mm_slli_epi16;

    let mut packed = 0u64;
    for (chunk_idx, chunk) in bytes.chunks_exact(16).enumerate() {
        let lo = read_u64_le(&chunk[..8]);
        let hi = read_u64_le(&chunk[8..]);
        let v = _mm_set_epi64x(hi as i64, lo as i64);
        // The movemask of a 128-bit vector only populates the low 16 bits.
        let mask = (_mm_movemask_epi8(_mm_slli_epi16::<7>(v)) as u64) & 0xFFFF;
        packed |= mask << (chunk_idx * 16);
    }
    packed
}

/// Portable SWAR pack: each aligned group of 8 bytes collapses to 8 bits with one
/// multiply (every byte's bit 0 is shifted into the top byte of the product).
#[cfg(any(test, not(target_arch = "x86_64")))]
#[inline]
fn pack_word_from_bytes_swar(bytes: &[u8; 64]) -> u64 {
    let mut packed = 0u64;
    for (chunk_idx, chunk) in bytes.chunks_exact(8).enumerate() {
        let x = read_u64_le(chunk);
        packed |= (x.wrapping_mul(0x0102_0408_1020_4080) >> 56) << (chunk_idx * 8);
    }
    packed
}

/// Pack `len` boolean values returned by `f` into the prefix of `words`, LSB-first,
/// 64 bits per `u64`. `words` must have capacity for at least `len.div_ceil(64)` entries.
///
/// Writes via `=` (not `|=`), so the destination need not be zero-initialised.
///
/// For best performance the closure should be branch-free: hoist bounds checks with
/// `get_unchecked` (safe here since `f` only ever sees indices below `len`), as the
/// compare/between kernels do. A closure that can panic blocks vectorization of the
/// internal byte-staging loop and loses most of the SIMD packing benefit.
#[inline]
pub fn collect_bool_words<F>(words: &mut [u64], len: usize, mut f: F)
where
    F: FnMut(usize) -> bool,
{
    let num_words = len.div_ceil(64);
    assert!(
        words.len() >= num_words,
        "words slice has {} entries, need at least {num_words}",
        words.len(),
    );

    let full = len / 64;
    let remainder = len % 64;

    collect_full_bool_words(&mut words[..full], &mut f);

    if remainder != 0 {
        let offset = full * 64;
        words[full] = collect_bool_word(remainder, |bit_idx| f(offset + bit_idx));
    }
}

/// Pack `words.len() * 64` boolean values returned by `f` into `words`, LSB-first.
///
/// Dispatches once per call (not per word) so that the AVX-512 body — which cannot be
/// inlined into non-AVX-512 callers — amortizes its call overhead across the whole slice.
#[inline]
fn collect_full_bool_words<F>(words: &mut [u64], f: &mut F)
where
    F: FnMut(usize) -> bool,
{
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512bw") {
        // SAFETY: avx512bw support (which implies avx512f) was just detected at runtime.
        unsafe { collect_full_bool_words_avx512(words, f) };
        return;
    }

    for (word_idx, word) in words.iter_mut().enumerate() {
        let offset = word_idx * 64;
        *word = collect_bool_word_full(|bit_idx| f(offset + bit_idx));
    }
}

/// AVX-512 variant of [`collect_full_bool_words`]: the staging loop vectorizes at 512-bit
/// width and each 64-byte chunk packs with a single `vptestmb` into a mask register.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
fn collect_full_bool_words_avx512<F>(words: &mut [u64], f: &mut F)
where
    F: FnMut(usize) -> bool,
{
    use core::arch::x86_64::_mm512_loadu_si512;
    use core::arch::x86_64::_mm512_test_epi8_mask;

    for (word_idx, word) in words.iter_mut().enumerate() {
        let offset = word_idx * 64;
        let mut bytes = [0u8; 64];
        for (bit_idx, byte) in bytes.iter_mut().enumerate() {
            *byte = f(offset + bit_idx) as u8;
        }
        // SAFETY: `bytes` is a valid 64-byte read and the load has no alignment requirement.
        let v = unsafe { _mm512_loadu_si512(bytes.as_ptr().cast()) };
        *word = _mm512_test_epi8_mask(v, v);
    }
}

/// Read up to 8 bytes as a little-endian `u64`, zero-padding the high bytes when fewer than 8
/// bytes are supplied.
///
/// This preserves Vortex's least-significant-bit-first bitmap numbering on little- and big-endian
/// targets. For a full 8-byte slice it lowers to a single word load.
#[inline]
pub fn read_u64_le(bytes: &[u8]) -> u64 {
    debug_assert!(bytes.len() <= 8);
    let mut buf = [0u8; 8];
    buf[..bytes.len()].copy_from_slice(bytes);
    u64::from_le_bytes(buf)
}

/// Splice a packed word `w` (whose bits above the highest valid bit are zero) into
/// `words` at the given bit position.
///
/// The destination word at `bit_offset / 64` is OR'd, preserving any bits below
/// `bit_offset % 64`. When `w` has high bits that spill into the next word, those
/// bits are *assigned* (not OR'd) — so callers must ensure that next slot is zero
/// (e.g. via `BufferMut::zeroed`).
///
/// `words.len()` need only cover the slots `w` actually writes to: skipping the
/// spillover when its bits are all zero means a tail that fits entirely in the
/// leading word never touches `words[dest_word + 1]`.
#[inline]
pub fn splice_word_at_bit(words: &mut [u64], bit_offset: usize, word: u64) {
    let dest_word = bit_offset / 64;
    let bit_in_word = bit_offset % 64;
    words[dest_word] |= word << bit_in_word;
    if bit_in_word != 0 {
        let high = word >> (64 - bit_in_word);
        if high != 0 {
            words[dest_word + 1] = high;
        }
    }
}

/// Pack `len` boolean values returned by `f` into `words` starting at bit position
/// `bit_offset`, LSB-first.
///
/// Composes [`collect_bool_word`] (pack up to 64 bools into a u64) with
/// [`splice_word_at_bit`] (merge the packed word into the destination via shift-OR).
///
/// `words` must have at least `(bit_offset + len).div_ceil(64)` entries; see
/// [`splice_word_at_bit`] for zero-init requirements on words above the cursor.
#[inline]
pub fn pack_bools_into_words<F>(words: &mut [u64], bit_offset: usize, len: usize, mut f: F)
where
    F: FnMut(usize) -> bool,
{
    if len == 0 {
        return;
    }
    let num_words = (bit_offset + len).div_ceil(64);
    assert!(
        words.len() >= num_words,
        "words slice has {} entries, need at least {num_words}",
        words.len(),
    );

    let mut done = 0;
    while len - done >= 64 {
        let word = collect_bool_word(64, |bit| f(done + bit));
        splice_word_at_bit(words, bit_offset + done, word);
        done += 64;
    }
    let tail = len - done;
    if tail > 0 {
        let word = collect_bool_word(tail, |bit| f(done + bit));
        splice_word_at_bit(words, bit_offset + done, word);
    }
}

/// Get the bit value at `index` out of `buf`.
///
/// # Panics
///
/// Panics if `index` is not between 0 and length of `buf * 8`.
#[inline(always)]
pub fn get_bit(buf: &[u8], index: usize) -> bool {
    buf[index / 8] & (1 << (index % 8)) != 0
}

/// Get the bit value at `index` out of `buf` without bounds checking.
///
/// # Safety
///
/// `index` must be between 0 and length of `buf * 8`.
#[inline(always)]
pub unsafe fn get_bit_unchecked(buf: *const u8, index: usize) -> bool {
    (unsafe { *buf.add(index / 8) } & (1 << (index % 8))) != 0
}

/// Set the bit value at `index` in `buf` without bounds checking.
///
/// # Safety
///
/// `index` must be between 0 and length of `buf * 8`.
#[inline(always)]
pub unsafe fn set_bit_unchecked(buf: *mut u8, index: usize) {
    unsafe { *buf.add(index / 8) |= 1 << (index % 8) };
}

/// Unset the bit value at `index` in `buf` without bounds checking.
///
/// # Safety
///
/// `index` must be between 0 and length of `buf * 8`.
#[inline(always)]
pub unsafe fn unset_bit_unchecked(buf: *mut u8, index: usize) {
    unsafe { *buf.add(index / 8) &= !(1 << (index % 8)) };
}

#[cfg(test)]
mod tests {
    use super::collect_bool_word;
    use super::collect_bool_words;
    use super::pack_bools_into_words;
    use super::pack_word_from_bytes;
    use super::pack_word_from_bytes_swar;
    use super::read_u64_le;

    fn test_pattern(i: usize) -> bool {
        (i.wrapping_mul(2654435761)) % 7 < 3
    }

    #[test]
    fn collect_bool_word_packs_lsb_first() {
        let word = collect_bool_word(5, |idx| idx.is_multiple_of(2));
        assert_eq!(word, 0b10101);
    }

    #[test]
    fn collect_bool_word_full_matches_scalar_reference() {
        let mut reference = 0u64;
        for bit_idx in 0..64 {
            reference |= (test_pattern(bit_idx) as u64) << bit_idx;
        }
        assert_eq!(collect_bool_word(64, test_pattern), reference);
    }

    #[test]
    fn collect_bool_words_matches_pattern_across_lengths() {
        for len in [0usize, 1, 7, 63, 64, 65, 127, 128, 130, 1000] {
            let mut words = vec![0u64; len.div_ceil(64)];
            collect_bool_words(&mut words, len, test_pattern);
            for i in 0..len {
                assert_eq!(
                    (words[i / 64] >> (i % 64)) & 1 == 1,
                    test_pattern(i),
                    "bit {i} for len {len}"
                );
            }
        }
    }

    #[test]
    fn swar_pack_matches_dispatched_pack() {
        let mut bytes = [0u8; 64];
        for (i, byte) in bytes.iter_mut().enumerate() {
            *byte = test_pattern(i) as u8;
        }
        assert_eq!(
            pack_word_from_bytes_swar(&bytes),
            pack_word_from_bytes(&bytes)
        );
    }

    #[test]
    fn collect_bool_word_empty() {
        assert_eq!(collect_bool_word(0, |_| true), 0);
    }

    #[test]
    fn read_u64_le_zero_pads_tail() {
        assert_eq!(read_u64_le(&[0x34, 0x12]), 0x1234);
        assert_eq!(read_u64_le(&[0xff; 8]), u64::MAX);
    }

    #[test]
    #[should_panic(expected = "cannot pack 65 bits into a u64 word")]
    fn collect_bool_word_rejects_too_many_bits() {
        let _ = collect_bool_word(65, |_| true);
    }

    fn pack(bit_offset: usize, len: usize, f: impl Fn(usize) -> bool) -> Vec<bool> {
        let num_words = (bit_offset + len).div_ceil(64);
        let mut words = vec![0u64; num_words];
        pack_bools_into_words(&mut words, bit_offset, len, &f);
        (0..bit_offset + len)
            .map(|i| (words[i / 64] >> (i % 64)) & 1 == 1)
            .collect()
    }

    #[test]
    fn pack_bools_aligned_multi_word_with_tail() {
        let bits = pack(0, 130, |i| i.is_multiple_of(3));
        for i in 0..130 {
            assert_eq!(bits[i], i.is_multiple_of(3), "bit {i}");
        }
    }

    #[test]
    fn pack_bools_unaligned_crossing_words() {
        let bits = pack(40, 200, |i| i.is_multiple_of(7));
        assert!(bits[..40].iter().all(|&b| !b));
        for i in 0..200 {
            assert_eq!(bits[40 + i], i.is_multiple_of(7), "bit {}", 40 + i);
        }
    }

    #[test]
    fn pack_bools_preserves_low_bits_of_leading_word() {
        let mut words = vec![0u64; 2];
        words[0] = 0b11111;
        pack_bools_into_words(&mut words, 5, 70, |_| true);
        for i in 0..5 {
            assert_eq!((words[0] >> i) & 1, 1, "preserved bit {i}");
        }
        for i in 5..75 {
            assert_eq!((words[i / 64] >> (i % 64)) & 1, 1, "extended bit {i}");
        }
    }
}
