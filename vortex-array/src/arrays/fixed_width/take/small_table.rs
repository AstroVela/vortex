// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Byte-table take for `u8` codes and a small table of one-byte values.
//!
//! The values are loaded into vector registers once and every output byte is produced by an
//! in-register permute, so the inner loop touches memory only to stream codes in and results out.
//! What the host implements decides how many table entries the loop can address, and that ceiling
//! is what matters: past it the caller falls back to the scalar loop, which measures roughly a
//! sixth of the vector rate because AVX2 has no gather narrower than 32 bits.
//!
//! | Kernel                          | Table entries | Instructions per vector |
//! |---------------------------------|---------------|-------------------------|
//! | AVX2 `vpshufb`                  |            16 | 1                       |
//! | AVX-512VL+BW `vpshufb` + blends |            64 | up to 7                 |
//! | AVX-512VBMI `vpermb`            |            64 | 1                       |
//! | NEON `vqtbl4q`                  |            64 | 1                       |
//!
//! `vpshufb` permutes within each 128-bit lane, so it addresses 16 entries however wide the
//! register is. Reaching 64 entries without a cross-lane permute means shuffling four sub-tables
//! and selecting between the results on bits 4 and 5 of each code — cheap only because AVX-512
//! mask registers blend in a single unmicrocoded uop. That is the useful thing AVX-512 brings
//! here: `k` masks, not 512-bit vectors. Widening this loop to 512 bits measured as a wash when
//! streaming and a regression in cache on a Cascade Lake host, so the 512-bit register file is
//! used only for `vpermb`, which earns it by collapsing the whole lookup back to one instruction
//! and which exists only on cores that run 512-bit code at full width and frequency.
//!
//! Every kernel pads its table with zeros and defers bounds checking to a single test after the
//! loop: an out-of-range code selects padding instead of reading out of bounds, so no iteration
//! has to branch.

use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;

use super::FixedWidthTakeValue;

/// Minimum number of indices before a vector kernel repays loading the table.
const MIN_INDICES: usize = 64;

/// Widens one-byte `values` into a zero-padded table of `N` entries.
fn byte_table<const N: usize, T: FixedWidthTakeValue>(values: &[T]) -> [u8; N] {
    debug_assert_eq!(
        size_of::<T>(),
        1,
        "byte table entries must be one byte wide"
    );
    debug_assert!(values.len() <= N, "value count exceeds the table size");

    let mut table = [0u8; N];
    // SAFETY: one-byte values have the same representation as bytes, and the dispatch in `take`
    // guarantees `values.len() <= N`.
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr().cast::<u8>(),
            table.as_mut_ptr(),
            values.len(),
        );
    }
    table
}

/// Bounds-checks every code, writes the positions the vector loop left over, and seals the buffer.
///
/// `vector_max_code` is the largest code the vector loop saw. Folding the remainder into it lets
/// one assertion cover the whole input, and it runs before any value is read, so an out-of-range
/// code panics rather than indexing past `values`.
///
/// # Safety
///
/// The caller's vector loop must have initialized every position of `output` below `offset`.
unsafe fn finish<T: FixedWidthTakeValue>(
    mut output: BufferMut<T>,
    values: &[T],
    indices: &[u8],
    offset: usize,
    vector_max_code: u8,
) -> Buffer<T> {
    let max_code = indices[offset..]
        .iter()
        .copied()
        .fold(vector_max_code, u8::max);
    assert!(
        usize::from(max_code) < values.len(),
        "take index {max_code} out of bounds for length {}",
        values.len()
    );

    let spare = output.spare_capacity_mut();
    for offset in offset..indices.len() {
        spare[offset].write(values[usize::from(indices[offset])]);
    }

    // SAFETY: the caller initialized every position below `offset` and the loop above initialized
    // the rest.
    unsafe { output.set_len(indices.len()) };
    // Do not expose the temporary vector over-alignment on the returned buffer.
    output.aligned(Alignment::of::<T>()).freeze()
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
mod arch {
    use std::arch::aarch64::uint8x16_t;
    use std::arch::aarch64::uint8x16x4_t;
    use std::arch::aarch64::vdupq_n_u8;
    use std::arch::aarch64::vld1q_u8;
    use std::arch::aarch64::vmaxq_u8;
    use std::arch::aarch64::vmaxvq_u8;
    use std::arch::aarch64::vqtbl4q_u8;
    use std::arch::aarch64::vst1q_u8;

    use vortex_buffer::Alignment;
    use vortex_buffer::Buffer;
    use vortex_buffer::BufferMut;

    use super::super::FixedWidthTakeValue;
    use super::MIN_INDICES;
    use super::byte_table;
    use super::finish;

    /// Entries addressable by a four-register `vqtbl4q` table lookup.
    const TABLE_ENTRIES: usize = 64;

    pub(crate) fn take<T: FixedWidthTakeValue>(values: &[T], indices: &[u8]) -> Option<Buffer<T>> {
        if size_of::<T>() != 1
            || values.is_empty()
            || values.len() > TABLE_ENTRIES
            || indices.len() < MIN_INDICES
        {
            return None;
        }

        // SAFETY: AArch64 always implements NEON, `T` is one byte wide, and the values fit the
        // table.
        Some(unsafe { take_neon(values, indices) })
    }

    /// # Safety
    ///
    /// `T` must be one byte wide and `values` must hold at most [`TABLE_ENTRIES`] entries.
    unsafe fn take_neon<T: FixedWidthTakeValue>(values: &[T], indices: &[u8]) -> Buffer<T> {
        let table = byte_table::<TABLE_ENTRIES, T>(values);
        // SAFETY: `table` is exactly four vectors wide. `vqtbl4q_u8` writes zero for any index
        // past the fourth register, which is what defers the bounds check out of the loop.
        let table = unsafe {
            uint8x16x4_t(
                vld1q_u8(table.as_ptr()),
                vld1q_u8(table.as_ptr().add(16)),
                vld1q_u8(table.as_ptr().add(32)),
                vld1q_u8(table.as_ptr().add(48)),
            )
        };

        let mut output =
            BufferMut::<T>::with_capacity_aligned(indices.len(), Alignment::of::<uint8x16_t>());
        let output_ptr = output.spare_capacity_mut().as_mut_ptr().cast::<u8>();
        // SAFETY: NEON is unconditionally available.
        let mut max_codes = unsafe { vdupq_n_u8(0) };

        let mut offset = 0;
        while offset + 16 <= indices.len() {
            // SAFETY: 16 codes remain to be read and 16 output bytes remain reserved.
            unsafe {
                let codes = vld1q_u8(indices.as_ptr().add(offset));
                max_codes = vmaxq_u8(max_codes, codes);
                vst1q_u8(output_ptr.add(offset), vqtbl4q_u8(table, codes));
            }
            offset += 16;
        }

        // SAFETY: the loop above initialized every output position below `offset`.
        unsafe { finish(output, values, indices, offset, vmaxvq_u8(max_codes)) }
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod arch {
    use std::arch::x86_64::__m256i;
    use std::arch::x86_64::__m512i;
    use std::arch::x86_64::_mm_loadu_si128;
    use std::arch::x86_64::_mm256_broadcastsi128_si256;
    use std::arch::x86_64::_mm256_loadu_si256;
    use std::arch::x86_64::_mm256_mask_blend_epi8;
    use std::arch::x86_64::_mm256_max_epu8;
    use std::arch::x86_64::_mm256_set1_epi8;
    use std::arch::x86_64::_mm256_setzero_si256;
    use std::arch::x86_64::_mm256_shuffle_epi8;
    use std::arch::x86_64::_mm256_storeu_si256;
    use std::arch::x86_64::_mm256_test_epi8_mask;
    use std::arch::x86_64::_mm512_loadu_si512;
    use std::arch::x86_64::_mm512_max_epu8;
    use std::arch::x86_64::_mm512_permutexvar_epi8;
    use std::arch::x86_64::_mm512_setzero_si512;
    use std::arch::x86_64::_mm512_storeu_si512;
    use std::sync::LazyLock;

    use vortex_buffer::Alignment;
    use vortex_buffer::Buffer;
    use vortex_buffer::BufferMut;

    use super::super::FixedWidthTakeValue;
    use super::MIN_INDICES;
    use super::byte_table;
    use super::finish;

    /// Entries addressable by one in-lane `vpshufb` sub-table.
    const SHUFFLE_ENTRIES: usize = 16;

    /// Entries addressable by `vpermb`, or by blending four `vpshufb` sub-tables.
    const MAX_ENTRIES: usize = 64;

    /// The byte-table kernel this host can run.
    #[derive(Clone, Copy)]
    enum Kernel {
        /// One `vpshufb`, so 16 entries and no way to select between sub-tables cheaply.
        Avx2,
        /// `vpshufb` sub-tables selected by `k`-mask blends, up to 64 entries.
        Avx512Vl,
        /// A single cross-lane `vpermb` over a 64-entry table.
        Avx512Vbmi,
    }

    static KERNEL: LazyLock<Option<Kernel>> = LazyLock::new(|| {
        if !is_x86_feature_detected!("avx2") {
            return None;
        }
        let masked_bytes =
            is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vl");
        if masked_bytes
            && is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512vbmi")
        {
            Some(Kernel::Avx512Vbmi)
        } else if masked_bytes {
            Some(Kernel::Avx512Vl)
        } else {
            Some(Kernel::Avx2)
        }
    });

    pub(crate) fn take<T: FixedWidthTakeValue>(values: &[T], indices: &[u8]) -> Option<Buffer<T>> {
        if size_of::<T>() != 1 || values.is_empty() || indices.len() < MIN_INDICES {
            return None;
        }

        // SAFETY: `KERNEL` names a kernel only once its features have been detected, `T` is one
        // byte wide, and each arm checks that the values fit that kernel's table.
        match (*KERNEL)? {
            Kernel::Avx512Vbmi if values.len() <= MAX_ENTRIES => {
                Some(unsafe { take_permute512(values, indices) })
            }
            // A 16-entry table needs no blend, so it runs the same single-shuffle loop as AVX2.
            Kernel::Avx2 | Kernel::Avx512Vl if values.len() <= SHUFFLE_ENTRIES => {
                Some(unsafe { take_shuffle256(values, indices) })
            }
            Kernel::Avx512Vl if values.len() <= 2 * SHUFFLE_ENTRIES => {
                Some(unsafe { take_shuffle256_pair(values, indices) })
            }
            Kernel::Avx512Vl if values.len() <= MAX_ENTRIES => {
                Some(unsafe { take_shuffle256_quad(values, indices) })
            }
            _ => None,
        }
    }

    /// Reduces a 256-bit accumulator of unsigned bytes to its largest lane.
    ///
    /// # Safety
    ///
    /// The caller must ensure the `avx2` feature is enabled.
    #[target_feature(enable = "avx2")]
    unsafe fn reduce_max_epu8(codes: __m256i) -> u8 {
        let mut lanes = [0u8; 32];
        // SAFETY: `lanes` is exactly one 256-bit vector wide.
        unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), codes) };
        lanes.into_iter().fold(0, u8::max)
    }

    /// Broadcasts the 16 bytes of `table` at `offset` into both 128-bit lanes.
    ///
    /// # Safety
    ///
    /// The caller must ensure the `avx2` feature is enabled and that `table` holds at least 16
    /// bytes past `offset`.
    #[target_feature(enable = "avx2")]
    unsafe fn sub_table(table: &[u8], offset: usize) -> __m256i {
        // SAFETY: the caller guarantees 16 readable bytes at `offset`.
        _mm256_broadcastsi128_si256(unsafe { _mm_loadu_si128(table.as_ptr().add(offset).cast()) })
    }

    /// Takes through a 64-entry table with one `vpermb` per vector.
    ///
    /// This is the only kernel here that uses 512-bit registers. It is worth it because `vpermb`
    /// replaces the whole shuffle-and-blend tree with one instruction, and because AVX-512VBMI
    /// exists only on cores that run 512-bit code at full width and frequency.
    ///
    /// # Safety
    ///
    /// The caller must ensure the `avx512f`, `avx512bw` and `avx512vbmi` features are enabled,
    /// that `T` is one byte wide, and that `values` holds at most [`MAX_ENTRIES`] entries.
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
    unsafe fn take_permute512<T: FixedWidthTakeValue>(values: &[T], indices: &[u8]) -> Buffer<T> {
        let table = byte_table::<MAX_ENTRIES, T>(values);
        // SAFETY: `table` is exactly one 512-bit vector wide.
        let table = unsafe { _mm512_loadu_si512(table.as_ptr().cast()) };

        let mut output =
            BufferMut::<T>::with_capacity_aligned(indices.len(), Alignment::of::<__m512i>());
        let output_ptr = output.spare_capacity_mut().as_mut_ptr().cast::<u8>();
        let mut max_codes = _mm512_setzero_si512();

        let mut offset = 0;
        while offset + 64 <= indices.len() {
            // SAFETY: 64 codes remain to be read and 64 output bytes remain reserved. `vpermb`
            // reads only the low six bits of each code, so a larger code wraps into the table
            // rather than reading out of bounds; `max_codes` catches it after the loop.
            unsafe {
                let codes = _mm512_loadu_si512(indices.as_ptr().add(offset).cast());
                max_codes = _mm512_max_epu8(max_codes, codes);
                _mm512_storeu_si512(
                    output_ptr.add(offset).cast(),
                    _mm512_permutexvar_epi8(codes, table),
                );
            }
            offset += 64;
        }

        let mut lanes = [0u8; 64];
        // SAFETY: `lanes` is exactly one 512-bit vector wide.
        unsafe { _mm512_storeu_si512(lanes.as_mut_ptr().cast(), max_codes) };
        // SAFETY: the loop above initialized every output position below `offset`.
        unsafe {
            finish(
                output,
                values,
                indices,
                offset,
                lanes.into_iter().fold(0, u8::max),
            )
        }
    }

    /// Takes through a 16-entry table with one `vpshufb` per vector.
    ///
    /// # Safety
    ///
    /// The caller must ensure the `avx2` feature is enabled, that `T` is one byte wide, and that
    /// `values` holds at most [`SHUFFLE_ENTRIES`] entries.
    #[target_feature(enable = "avx2")]
    unsafe fn take_shuffle256<T: FixedWidthTakeValue>(values: &[T], indices: &[u8]) -> Buffer<T> {
        let table = byte_table::<SHUFFLE_ENTRIES, T>(values);
        // SAFETY: `avx2` is enabled and `table` holds 16 bytes.
        let table = unsafe { sub_table(&table, 0) };

        let mut output =
            BufferMut::<T>::with_capacity_aligned(indices.len(), Alignment::of::<__m256i>());
        let output_ptr = output.spare_capacity_mut().as_mut_ptr().cast::<u8>();
        let mut max_codes = _mm256_setzero_si256();

        let mut offset = 0;
        while offset + 32 <= indices.len() {
            // SAFETY: 32 codes remain to be read and 32 output bytes remain reserved.
            unsafe {
                let codes = _mm256_loadu_si256(indices.as_ptr().add(offset).cast());
                max_codes = _mm256_max_epu8(max_codes, codes);
                _mm256_storeu_si256(
                    output_ptr.add(offset).cast(),
                    _mm256_shuffle_epi8(table, codes),
                );
            }
            offset += 32;
        }

        // SAFETY: the loop above initialized every output position below `offset`.
        unsafe { finish(output, values, indices, offset, reduce_max_epu8(max_codes)) }
    }

    /// Takes through a 32-entry table as two `vpshufb` sub-tables blended on bit 4 of each code.
    ///
    /// # Safety
    ///
    /// The caller must ensure the `avx2`, `avx512bw` and `avx512vl` features are enabled, that
    /// `T` is one byte wide, and that `values` holds at most `2 * SHUFFLE_ENTRIES` entries.
    #[target_feature(enable = "avx2,avx512bw,avx512vl")]
    unsafe fn take_shuffle256_pair<T: FixedWidthTakeValue>(
        values: &[T],
        indices: &[u8],
    ) -> Buffer<T> {
        let table = byte_table::<{ 2 * SHUFFLE_ENTRIES }, T>(values);
        // SAFETY: `avx2` is enabled and `table` holds 32 bytes.
        let (low, high) = unsafe { (sub_table(&table, 0), sub_table(&table, 16)) };
        let bit4 = _mm256_set1_epi8(0x10);

        let mut output =
            BufferMut::<T>::with_capacity_aligned(indices.len(), Alignment::of::<__m256i>());
        let output_ptr = output.spare_capacity_mut().as_mut_ptr().cast::<u8>();
        let mut max_codes = _mm256_setzero_si256();

        let mut offset = 0;
        while offset + 32 <= indices.len() {
            // SAFETY: 32 codes remain to be read and 32 output bytes remain reserved. Codes past
            // the table select the zero padding, so nothing reads out of bounds; `max_codes`
            // catches them after the loop.
            unsafe {
                let codes = _mm256_loadu_si256(indices.as_ptr().add(offset).cast());
                max_codes = _mm256_max_epu8(max_codes, codes);
                let select_high = _mm256_test_epi8_mask(codes, bit4);
                _mm256_storeu_si256(
                    output_ptr.add(offset).cast(),
                    _mm256_mask_blend_epi8(
                        select_high,
                        _mm256_shuffle_epi8(low, codes),
                        _mm256_shuffle_epi8(high, codes),
                    ),
                );
            }
            offset += 32;
        }

        // SAFETY: the loop above initialized every output position below `offset`.
        unsafe { finish(output, values, indices, offset, reduce_max_epu8(max_codes)) }
    }

    /// Takes through a 64-entry table as four `vpshufb` sub-tables blended on bits 4 and 5.
    ///
    /// # Safety
    ///
    /// The caller must ensure the `avx2`, `avx512bw` and `avx512vl` features are enabled, that
    /// `T` is one byte wide, and that `values` holds at most [`MAX_ENTRIES`] entries.
    #[target_feature(enable = "avx2,avx512bw,avx512vl")]
    unsafe fn take_shuffle256_quad<T: FixedWidthTakeValue>(
        values: &[T],
        indices: &[u8],
    ) -> Buffer<T> {
        let table = byte_table::<MAX_ENTRIES, T>(values);
        // SAFETY: `avx2` is enabled and `table` holds 64 bytes.
        let (t0, t1, t2, t3) = unsafe {
            (
                sub_table(&table, 0),
                sub_table(&table, 16),
                sub_table(&table, 32),
                sub_table(&table, 48),
            )
        };
        let bit4 = _mm256_set1_epi8(0x10);
        let bit5 = _mm256_set1_epi8(0x20);

        let mut output =
            BufferMut::<T>::with_capacity_aligned(indices.len(), Alignment::of::<__m256i>());
        let output_ptr = output.spare_capacity_mut().as_mut_ptr().cast::<u8>();
        let mut max_codes = _mm256_setzero_si256();

        let mut offset = 0;
        while offset + 32 <= indices.len() {
            // SAFETY: 32 codes remain to be read and 32 output bytes remain reserved. Codes past
            // the table select the zero padding, so nothing reads out of bounds; `max_codes`
            // catches them after the loop.
            unsafe {
                let codes = _mm256_loadu_si256(indices.as_ptr().add(offset).cast());
                max_codes = _mm256_max_epu8(max_codes, codes);
                let select_odd = _mm256_test_epi8_mask(codes, bit4);
                let select_high = _mm256_test_epi8_mask(codes, bit5);
                let low = _mm256_mask_blend_epi8(
                    select_odd,
                    _mm256_shuffle_epi8(t0, codes),
                    _mm256_shuffle_epi8(t1, codes),
                );
                let high = _mm256_mask_blend_epi8(
                    select_odd,
                    _mm256_shuffle_epi8(t2, codes),
                    _mm256_shuffle_epi8(t3, codes),
                );
                _mm256_storeu_si256(
                    output_ptr.add(offset).cast(),
                    _mm256_mask_blend_epi8(select_high, low, high),
                );
            }
            offset += 32;
        }

        // SAFETY: the loop above initialized every output position below `offset`.
        unsafe { finish(output, values, indices, offset, reduce_max_epu8(max_codes)) }
    }
}

pub(super) use arch::take;
