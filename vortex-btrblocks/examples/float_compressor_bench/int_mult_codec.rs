// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::size_of;

use fastlanes::BitPacking;
use vortex_error::VortexResult;

const CHUNK_LEN: usize = 1_024;
const BITMAP_WORDS: usize = CHUNK_LEN / u64::BITS as usize;
const STREAM_PADDING: usize = 7;
const SERIALIZED_BLOCK_METADATA_BYTES: usize = 24;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntMultCodec32 {
    len: usize,
    base: u32,
    blocks: Vec<IntMultBlock32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntMultDenseCodec64 {
    len: usize,
    base: u64,
    blocks: Vec<IntMultDenseBlock64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct IntMultDenseBlock64 {
    len: u16,
    quotient_base: u64,
    quotient_width: u8,
    quotient_high_width: u8,
    quotient_lows: Vec<u64>,
    quotient_patch_bitmap: Vec<u64>,
    quotient_highs: Vec<u8>,
    remainders: Vec<u64>,
    quotient_patch_count: u16,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct IntMultBlock32 {
    len: u16,
    quotient_base: u32,
    quotient_width: u8,
    quotient_high_width: u8,
    quotient_lows: Vec<u32>,
    quotient_patch_bitmap: Vec<u64>,
    quotient_patch_gaps: Vec<u8>,
    quotient_highs: Vec<u8>,
    remainder_mode: u32,
    remainder_width: u8,
    remainder_exception_bitmap: Vec<u64>,
    remainder_exception_gaps: Vec<u8>,
    remainder_pair_bitmap: Vec<u64>,
    remainder_pair_values: Vec<u8>,
    remainder_exceptions: Vec<u8>,
    quotient_patch_count: u16,
    remainder_exception_count: u16,
}

impl IntMultCodec32 {
    pub fn encode(values: &[u32], base: u32) -> VortexResult<Self> {
        vortex_error::vortex_ensure!(base > 1, "IntMult base must exceed one");
        let blocks = values
            .chunks(CHUNK_LEN)
            .map(|block| encode_block(block, base))
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(Self {
            len: values.len(),
            base,
            blocks,
        })
    }

    pub fn decode(&self) -> Vec<u32> {
        self.decode_variant::<true, true, false, false>()
    }

    pub fn decode_gaps(&self) -> Vec<u32> {
        self.decode_variant::<true, true, true, false>()
    }

    pub fn decode_pairs(&self) -> Vec<u32> {
        self.decode_variant::<true, true, false, true>()
    }

    pub fn decode_gaps_without_quotient_patches(&self) -> Vec<u32> {
        self.decode_variant::<false, true, true, false>()
    }

    pub fn decode_gaps_without_remainder_exceptions(&self) -> Vec<u32> {
        self.decode_variant::<true, false, true, false>()
    }

    pub fn decode_gaps_without_exceptions(&self) -> Vec<u32> {
        self.decode_variant::<false, false, true, false>()
    }

    pub fn decode_without_quotient_patches(&self) -> Vec<u32> {
        self.decode_variant::<false, true, false, false>()
    }

    pub fn decode_without_remainder_exceptions(&self) -> Vec<u32> {
        self.decode_variant::<true, false, false, false>()
    }

    pub fn decode_without_exceptions(&self) -> Vec<u32> {
        self.decode_variant::<false, false, false, false>()
    }

    fn decode_variant<
        const QUOTIENT_PATCHES: bool,
        const REMAINDER_EXCEPTIONS: bool,
        const GAPS: bool,
        const PAIRS: bool,
    >(
        &self,
    ) -> Vec<u32> {
        let mut values = Vec::with_capacity(self.len);
        let mut quotients = [0_u32; CHUNK_LEN];
        let mut remainder_exceptions = [0_u8; CHUNK_LEN];
        for block in &self.blocks {
            if block.quotient_width == 0 {
                quotients.fill(0);
            } else {
                // SAFETY: The encoder creates one complete FastLanes chunk.
                unsafe {
                    u32::unchecked_unpack(
                        usize::from(block.quotient_width),
                        &block.quotient_lows,
                        &mut quotients,
                    );
                }
            }

            if QUOTIENT_PATCHES {
                let mut high_index = 0usize;
                let mut apply = |position| {
                    let high = read_bits(
                        &block.quotient_highs,
                        high_index * usize::from(block.quotient_high_width),
                        block.quotient_high_width,
                    );
                    quotients[position] |= high << block.quotient_width;
                    high_index += 1;
                };
                if GAPS {
                    for_each_gap(&block.quotient_patch_gaps, &mut apply);
                } else {
                    for_each_set_bit(&block.quotient_patch_bitmap, &mut apply);
                }
            }
            let block_len = usize::from(block.len);
            for quotient in &mut quotients[..block_len] {
                *quotient = quotient
                    .wrapping_add(block.quotient_base)
                    .wrapping_mul(self.base)
                    .wrapping_add(block.remainder_mode);
            }
            if REMAINDER_EXCEPTIONS {
                let mut exception_index = 0usize;
                if PAIRS {
                    let mut pair_value_index = 0usize;
                    for_each_set_bit(&block.remainder_pair_bitmap, |pair_index| {
                        let position = pair_index * 2;
                        // SAFETY: The encoder stores one byte for every set pair bit.
                        let pair =
                            unsafe { *block.remainder_pair_values.get_unchecked(pair_value_index) };
                        quotients[position] = quotients[position]
                            .wrapping_sub(block.remainder_mode)
                            .wrapping_add(u32::from(pair & 0x0f));
                        if position + 1 < block_len {
                            quotients[position + 1] = quotients[position + 1]
                                .wrapping_sub(block.remainder_mode)
                                .wrapping_add(u32::from(pair >> 4));
                        }
                        pair_value_index += 1;
                    });
                } else if GAPS {
                    for_each_gap(&block.remainder_exception_gaps, |position| {
                        let remainder = read_bits(
                            &block.remainder_exceptions,
                            exception_index * usize::from(block.remainder_width),
                            block.remainder_width,
                        );
                        quotients[position] = quotients[position]
                            .wrapping_sub(block.remainder_mode)
                            .wrapping_add(remainder);
                        exception_index += 1;
                    });
                } else {
                    if block.remainder_width == 4 {
                        let exception_count = usize::from(block.remainder_exception_count);
                        for index in 0..exception_count.div_ceil(2) {
                            // SAFETY: The encoder stores two exceptions in each payload byte.
                            let byte = unsafe { *block.remainder_exceptions.get_unchecked(index) };
                            remainder_exceptions[index * 2] = byte & 0x0f;
                            remainder_exceptions[index * 2 + 1] = byte >> 4;
                        }
                        for_each_set_bit(&block.remainder_exception_bitmap, |position| {
                            let remainder = u32::from(remainder_exceptions[exception_index]);
                            quotients[position] = quotients[position]
                                .wrapping_sub(block.remainder_mode)
                                .wrapping_add(remainder);
                            exception_index += 1;
                        });
                    } else {
                        for_each_set_bit(&block.remainder_exception_bitmap, |position| {
                            let remainder = read_bits(
                                &block.remainder_exceptions,
                                exception_index * usize::from(block.remainder_width),
                                block.remainder_width,
                            );
                            quotients[position] = quotients[position]
                                .wrapping_sub(block.remainder_mode)
                                .wrapping_add(remainder);
                            exception_index += 1;
                        });
                    }
                }
            }
            values.extend_from_slice(&quotients[..block_len]);
        }
        values
    }

    pub fn scalar_at(&self, index: usize) -> VortexResult<u32> {
        self.scalar_at_variant::<false>(index)
    }

    pub fn scalar_at_gaps(&self, index: usize) -> VortexResult<u32> {
        self.scalar_at_variant::<true>(index)
    }

    fn scalar_at_variant<const GAPS: bool>(&self, index: usize) -> VortexResult<u32> {
        vortex_error::vortex_ensure!(
            index < self.len,
            "index {index} is out of bounds for length {}",
            self.len
        );
        let block = &self.blocks[index / CHUNK_LEN];
        let index_in_block = index % CHUNK_LEN;
        let mut quotient_residual = if block.quotient_width == 0 {
            0
        } else {
            // SAFETY: The encoder creates one complete FastLanes chunk.
            unsafe {
                u32::unchecked_unpack_single(
                    usize::from(block.quotient_width),
                    &block.quotient_lows,
                    index_in_block,
                )
            }
        };
        let quotient_rank = if GAPS {
            gap_rank(&block.quotient_patch_gaps, index_in_block)
        } else if bitmap_value(&block.quotient_patch_bitmap, index_in_block) {
            Some(bitmap_rank(&block.quotient_patch_bitmap, index_in_block))
        } else {
            None
        };
        if let Some(rank) = quotient_rank {
            let high = read_bits(
                &block.quotient_highs,
                rank * usize::from(block.quotient_high_width),
                block.quotient_high_width,
            );
            quotient_residual |= high << block.quotient_width;
        }
        let quotient = block.quotient_base.wrapping_add(quotient_residual);

        let remainder_rank = if GAPS {
            gap_rank(&block.remainder_exception_gaps, index_in_block)
        } else if bitmap_value(&block.remainder_exception_bitmap, index_in_block) {
            Some(bitmap_rank(
                &block.remainder_exception_bitmap,
                index_in_block,
            ))
        } else {
            None
        };
        let remainder = if let Some(rank) = remainder_rank {
            read_bits(
                &block.remainder_exceptions,
                rank * usize::from(block.remainder_width),
                block.remainder_width,
            )
        } else {
            block.remainder_mode
        };
        Ok(quotient.wrapping_mul(self.base).wrapping_add(remainder))
    }

    pub fn encoded_size(&self) -> usize {
        size_of::<u32>()
            + self
                .blocks
                .iter()
                .map(|block| {
                    SERIALIZED_BLOCK_METADATA_BYTES
                        + block.quotient_lows.len() * size_of::<u32>()
                        + block.quotient_patch_bitmap.len() * size_of::<u64>()
                        + block.quotient_highs.len()
                        + block.remainder_exception_bitmap.len() * size_of::<u64>()
                        + block.remainder_exceptions.len()
                })
                .sum::<usize>()
    }

    pub fn encoded_size_gaps(&self) -> usize {
        size_of::<u32>()
            + self
                .blocks
                .iter()
                .map(|block| {
                    SERIALIZED_BLOCK_METADATA_BYTES
                        + block.quotient_lows.len() * size_of::<u32>()
                        + block.quotient_patch_gaps.len()
                        + block.quotient_highs.len()
                        + block.remainder_exception_gaps.len()
                        + block.remainder_exceptions.len()
                })
                .sum::<usize>()
    }

    pub fn encoded_size_pairs(&self) -> usize {
        size_of::<u32>()
            + self
                .blocks
                .iter()
                .map(|block| {
                    SERIALIZED_BLOCK_METADATA_BYTES
                        + block.quotient_lows.len() * size_of::<u32>()
                        + block.quotient_patch_bitmap.len() * size_of::<u64>()
                        + block.quotient_highs.len()
                        + block.remainder_pair_bitmap.len() * size_of::<u64>()
                        + block.remainder_pair_values.len()
                })
                .sum::<usize>()
    }

    pub fn quotient_patch_count(&self) -> usize {
        self.blocks
            .iter()
            .map(|block| usize::from(block.quotient_patch_count))
            .sum()
    }

    pub fn remainder_exception_count(&self) -> usize {
        self.blocks
            .iter()
            .map(|block| usize::from(block.remainder_exception_count))
            .sum()
    }

    pub fn gap_bytes(&self) -> (usize, usize) {
        self.blocks
            .iter()
            .fold((0, 0), |(quotient, remainder), block| {
                (
                    quotient + block.quotient_patch_gaps.len(),
                    remainder + block.remainder_exception_gaps.len(),
                )
            })
    }

    pub fn bitmap_bytes(&self) -> (usize, usize) {
        self.blocks
            .iter()
            .fold((0, 0), |(quotient, remainder), block| {
                (
                    quotient + block.quotient_patch_bitmap.len() * size_of::<u64>(),
                    remainder + block.remainder_exception_bitmap.len() * size_of::<u64>(),
                )
            })
    }
}

impl IntMultDenseCodec64 {
    pub fn encode(values: &[u64], base: u64) -> VortexResult<Self> {
        vortex_error::vortex_ensure!(base > 1, "IntMult base must exceed one");
        let blocks = values
            .chunks(CHUNK_LEN)
            .map(|block| encode_dense_block64(block, base))
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(Self {
            len: values.len(),
            base,
            blocks,
        })
    }

    pub fn decode(&self) -> Vec<u64> {
        self.decode_with_quotient_patches(true)
    }

    pub fn decode_without_quotient_patches(&self) -> Vec<u64> {
        self.decode_with_quotient_patches(false)
    }

    fn decode_with_quotient_patches(&self, apply_patches: bool) -> Vec<u64> {
        let mut values = Vec::with_capacity(self.len);
        let mut quotients = [0_u64; CHUNK_LEN];
        let mut remainders = [0_u64; CHUNK_LEN];
        for block in &self.blocks {
            if block.quotient_width == 0 {
                quotients.fill(0);
            } else {
                // SAFETY: The encoder creates one complete FastLanes chunk.
                unsafe {
                    u64::unchecked_unpack(
                        usize::from(block.quotient_width),
                        &block.quotient_lows,
                        &mut quotients,
                    );
                }
            }
            if apply_patches {
                let mut high_index = 0usize;
                for_each_set_bit(&block.quotient_patch_bitmap, |position| {
                    let high = read_bits64(
                        &block.quotient_highs,
                        high_index * usize::from(block.quotient_high_width),
                        block.quotient_high_width,
                    );
                    quotients[position] |= high << block.quotient_width;
                    high_index += 1;
                });
            }

            let remainder_width = bit_width64(self.base - 1);
            if remainder_width == 0 {
                remainders.fill(0);
            } else {
                // SAFETY: The encoder creates one complete FastLanes chunk.
                unsafe {
                    u64::unchecked_unpack(
                        usize::from(remainder_width),
                        &block.remainders,
                        &mut remainders,
                    );
                }
            }

            let block_len = usize::from(block.len);
            for index in 0..block_len {
                quotients[index] = quotients[index]
                    .wrapping_add(block.quotient_base)
                    .wrapping_mul(self.base)
                    .wrapping_add(remainders[index]);
            }
            values.extend_from_slice(&quotients[..block_len]);
        }
        values
    }

    pub fn scalar_at(&self, index: usize) -> VortexResult<u64> {
        vortex_error::vortex_ensure!(
            index < self.len,
            "index {index} is out of bounds for length {}",
            self.len
        );
        let block = &self.blocks[index / CHUNK_LEN];
        let index_in_block = index % CHUNK_LEN;
        let mut quotient_residual = if block.quotient_width == 0 {
            0
        } else {
            // SAFETY: The encoder creates one complete FastLanes chunk.
            unsafe {
                u64::unchecked_unpack_single(
                    usize::from(block.quotient_width),
                    &block.quotient_lows,
                    index_in_block,
                )
            }
        };
        if bitmap_value(&block.quotient_patch_bitmap, index_in_block) {
            let rank = bitmap_rank(&block.quotient_patch_bitmap, index_in_block);
            let high = read_bits64(
                &block.quotient_highs,
                rank * usize::from(block.quotient_high_width),
                block.quotient_high_width,
            );
            quotient_residual |= high << block.quotient_width;
        }
        let remainder_width = bit_width64(self.base - 1);
        let remainder = if remainder_width == 0 {
            0
        } else {
            // SAFETY: The encoder creates one complete FastLanes chunk.
            unsafe {
                u64::unchecked_unpack_single(
                    usize::from(remainder_width),
                    &block.remainders,
                    index_in_block,
                )
            }
        };
        Ok(quotient_residual
            .wrapping_add(block.quotient_base)
            .wrapping_mul(self.base)
            .wrapping_add(remainder))
    }

    pub fn encoded_size(&self) -> usize {
        size_of::<u64>()
            + self
                .blocks
                .iter()
                .map(|block| {
                    SERIALIZED_BLOCK_METADATA_BYTES
                        + block.quotient_lows.len() * size_of::<u64>()
                        + block.quotient_patch_bitmap.len() * size_of::<u64>()
                        + block.quotient_highs.len()
                        + block.remainders.len() * size_of::<u64>()
                })
                .sum::<usize>()
    }

    pub fn quotient_patch_count(&self) -> usize {
        self.blocks
            .iter()
            .map(|block| usize::from(block.quotient_patch_count))
            .sum()
    }
}

fn encode_block(values: &[u32], base: u32) -> VortexResult<IntMultBlock32> {
    let mut quotients = Vec::with_capacity(CHUNK_LEN);
    let mut remainders = Vec::with_capacity(CHUNK_LEN);
    for &value in values {
        quotients.push(value / base);
        remainders.push(value % base);
    }

    let quotient_base = quotients.iter().copied().min().unwrap_or_default();
    let mut quotient_residuals = quotients
        .iter()
        .map(|&quotient| quotient - quotient_base)
        .collect::<Vec<_>>();
    let (quotient_width, quotient_high_width, quotient_patch_count) =
        choose_quotient_width(&quotient_residuals, values.len());
    quotient_residuals.resize(CHUNK_LEN, 0);
    let quotient_mask = low_mask(quotient_width);
    let quotient_lows = quotient_residuals
        .iter()
        .map(|&residual| residual & quotient_mask)
        .collect::<Vec<_>>();

    let mut quotient_patch_bitmap = vec![0_u64; BITMAP_WORDS];
    let mut quotient_patch_positions = Vec::with_capacity(quotient_patch_count);
    let mut quotient_highs =
        BitWriter::with_capacity(quotient_patch_count * usize::from(quotient_high_width));
    if quotient_high_width > 0 {
        for (position, &residual) in quotient_residuals[..values.len()].iter().enumerate() {
            let high = residual >> quotient_width;
            if high != 0 {
                set_bitmap(&mut quotient_patch_bitmap, position);
                quotient_patch_positions.push(u16::try_from(position)?);
                quotient_highs.write(high, quotient_high_width);
            }
        }
    }
    if quotient_patch_count == 0 {
        quotient_patch_bitmap.clear();
    }

    let remainder_mode = mode(&remainders, base)?;
    let remainder_width = bit_width(base - 1);
    let mut remainder_exception_bitmap = vec![0_u64; BITMAP_WORDS];
    let remainder_exception_count = remainders
        .iter()
        .filter(|&&remainder| remainder != remainder_mode)
        .count();
    let mut remainder_exceptions =
        BitWriter::with_capacity(remainder_exception_count * usize::from(remainder_width));
    let mut remainder_exception_positions = Vec::with_capacity(remainder_exception_count);
    for (position, &remainder) in remainders.iter().enumerate() {
        if remainder != remainder_mode {
            set_bitmap(&mut remainder_exception_bitmap, position);
            remainder_exception_positions.push(u16::try_from(position)?);
            remainder_exceptions.write(remainder, remainder_width);
        }
    }
    if remainder_exception_count == 0 {
        remainder_exception_bitmap.clear();
    }

    let mut remainder_pair_bitmap = vec![0_u64; BITMAP_WORDS / 2];
    let mut remainder_pair_values = Vec::with_capacity(values.len().div_ceil(2));
    let mut remainder_pair_count = 0usize;
    if base <= 16 {
        for (pair_index, pair) in remainders.chunks(2).enumerate() {
            let left_exception = pair[0] != remainder_mode;
            let right_exception = pair
                .get(1)
                .is_some_and(|&remainder| remainder != remainder_mode);
            if !left_exception && !right_exception {
                continue;
            }
            set_bitmap(&mut remainder_pair_bitmap, pair_index);
            let right = pair.get(1).copied().unwrap_or(remainder_mode);
            remainder_pair_values.push(u8::try_from(pair[0] | (right << 4))?);
            remainder_pair_count += 1;
        }
    }
    if remainder_pair_count == 0 {
        remainder_pair_bitmap.clear();
    }

    Ok(IntMultBlock32 {
        len: u16::try_from(values.len())?,
        quotient_base,
        quotient_width,
        quotient_high_width,
        quotient_lows: fast_pack(&quotient_lows, quotient_width),
        quotient_patch_bitmap,
        quotient_patch_gaps: encode_gaps(&quotient_patch_positions),
        quotient_highs: quotient_highs.finish(),
        remainder_mode,
        remainder_width,
        remainder_exception_bitmap,
        remainder_exception_gaps: encode_gaps(&remainder_exception_positions),
        remainder_pair_bitmap,
        remainder_pair_values,
        remainder_exceptions: remainder_exceptions.finish(),
        quotient_patch_count: u16::try_from(quotient_patch_count)?,
        remainder_exception_count: u16::try_from(remainder_exception_count)?,
    })
}

fn encode_dense_block64(values: &[u64], base: u64) -> VortexResult<IntMultDenseBlock64> {
    let mut quotient_residuals = Vec::with_capacity(CHUNK_LEN);
    let mut remainders = Vec::with_capacity(CHUNK_LEN);
    let quotient_base = values
        .iter()
        .map(|value| value / base)
        .min()
        .unwrap_or_default();
    for &value in values {
        quotient_residuals.push(value / base - quotient_base);
        remainders.push(value % base);
    }
    let (quotient_width, quotient_high_width, quotient_patch_count) =
        choose_quotient_width64(&quotient_residuals, values.len());
    quotient_residuals.resize(CHUNK_LEN, 0);
    remainders.resize(CHUNK_LEN, 0);
    let quotient_mask = low_mask64(quotient_width);
    let quotient_lows = quotient_residuals
        .iter()
        .map(|&residual| residual & quotient_mask)
        .collect::<Vec<_>>();

    let mut quotient_patch_bitmap = vec![0_u64; BITMAP_WORDS];
    let mut quotient_highs =
        BitWriter64::with_capacity(quotient_patch_count * usize::from(quotient_high_width));
    if quotient_high_width > 0 {
        for (position, &residual) in quotient_residuals[..values.len()].iter().enumerate() {
            let high = residual >> quotient_width;
            if high != 0 {
                set_bitmap(&mut quotient_patch_bitmap, position);
                quotient_highs.write(high, quotient_high_width);
            }
        }
    }
    if quotient_patch_count == 0 {
        quotient_patch_bitmap.clear();
    }

    Ok(IntMultDenseBlock64 {
        len: u16::try_from(values.len())?,
        quotient_base,
        quotient_width,
        quotient_high_width,
        quotient_lows: fast_pack64(&quotient_lows, quotient_width),
        quotient_patch_bitmap,
        quotient_highs: quotient_highs.finish(),
        remainders: fast_pack64(&remainders, bit_width64(base - 1)),
        quotient_patch_count: u16::try_from(quotient_patch_count)?,
    })
}

fn choose_quotient_width(residuals: &[u32], value_count: usize) -> (u8, u8, usize) {
    let mut width_counts = [0_usize; 33];
    let mut maximum_width = 0u8;
    for &residual in residuals {
        let width = bit_width(residual);
        width_counts[usize::from(width)] += 1;
        maximum_width = maximum_width.max(width);
    }

    let mut patch_count = value_count;
    let mut best = (maximum_width, 0_u8, 0_usize);
    let mut best_bits = CHUNK_LEN * usize::from(maximum_width);
    for residual_width in 0..=maximum_width {
        patch_count -= width_counts[usize::from(residual_width)];
        let high_width = if patch_count == 0 {
            0
        } else {
            maximum_width - residual_width
        };
        let bits = CHUNK_LEN * usize::from(residual_width)
            + usize::from(patch_count > 0) * CHUNK_LEN
            + patch_count * usize::from(high_width);
        if bits < best_bits {
            best = (residual_width, high_width, patch_count);
            best_bits = bits;
        }
    }
    best
}

fn choose_quotient_width64(residuals: &[u64], value_count: usize) -> (u8, u8, usize) {
    let mut width_counts = [0_usize; 65];
    let mut maximum_width = 0u8;
    for &residual in residuals {
        let width = bit_width64(residual);
        width_counts[usize::from(width)] += 1;
        maximum_width = maximum_width.max(width);
    }

    let mut patch_count = value_count;
    let mut best = (maximum_width, 0_u8, 0_usize);
    let mut best_bits = CHUNK_LEN * usize::from(maximum_width);
    for residual_width in 0..=maximum_width {
        patch_count -= width_counts[usize::from(residual_width)];
        let high_width = if patch_count == 0 {
            0
        } else {
            maximum_width - residual_width
        };
        let bits = CHUNK_LEN * usize::from(residual_width)
            + usize::from(patch_count > 0) * CHUNK_LEN
            + patch_count * usize::from(high_width);
        if bits < best_bits {
            best = (residual_width, high_width, patch_count);
            best_bits = bits;
        }
    }
    best
}

fn mode(values: &[u32], base: u32) -> VortexResult<u32> {
    let mut counts = vec![0_u16; usize::try_from(base)?];
    for &value in values {
        counts[usize::try_from(value)?] += 1;
    }
    Ok(counts
        .iter()
        .enumerate()
        .max_by_key(|(_, count)| **count)
        .map(|(value, _)| u32::try_from(value))
        .transpose()?
        .unwrap_or_default())
}

fn fast_pack(values: &[u32], width: u8) -> Vec<u32> {
    if width == 0 {
        return Vec::new();
    }
    let mut packed = vec![0_u32; CHUNK_LEN * usize::from(width) / u32::BITS as usize];
    // SAFETY: Both slices have the exact lengths required for one FastLanes chunk.
    unsafe { u32::unchecked_pack(usize::from(width), values, &mut packed) };
    packed
}

fn fast_pack64(values: &[u64], width: u8) -> Vec<u64> {
    if width == 0 {
        return Vec::new();
    }
    let mut packed = vec![0_u64; CHUNK_LEN * usize::from(width) / u64::BITS as usize];
    // SAFETY: Both slices have the exact lengths required for one FastLanes chunk.
    unsafe { u64::unchecked_pack(usize::from(width), values, &mut packed) };
    packed
}

fn bit_width(value: u32) -> u8 {
    u8::try_from(u32::BITS - value.leading_zeros()).unwrap_or(u8::MAX)
}

fn low_mask(width: u8) -> u32 {
    if width == 32 {
        u32::MAX
    } else if width == 0 {
        0
    } else {
        (1_u32 << width) - 1
    }
}

fn bit_width64(value: u64) -> u8 {
    u8::try_from(u64::BITS - value.leading_zeros()).unwrap_or(u8::MAX)
}

fn low_mask64(width: u8) -> u64 {
    if width == 64 {
        u64::MAX
    } else if width == 0 {
        0
    } else {
        (1_u64 << width) - 1
    }
}

fn set_bitmap(bitmap: &mut [u64], index: usize) {
    bitmap[index / 64] |= 1_u64 << (index % 64);
}

fn bitmap_value(bitmap: &[u64], index: usize) -> bool {
    bitmap
        .get(index / 64)
        .is_some_and(|word| word & (1_u64 << (index % 64)) != 0)
}

fn bitmap_rank(bitmap: &[u64], index: usize) -> usize {
    let full_words = index / 64;
    let prior = bitmap[..full_words]
        .iter()
        .map(|word| word.count_ones() as usize)
        .sum::<usize>();
    let bit_index = index % 64;
    let current = bitmap
        .get(full_words)
        .map(|word| (word & ((1_u64 << bit_index).wrapping_sub(1))).count_ones() as usize)
        .unwrap_or_default();
    prior + current
}

fn encode_gaps(positions: &[u16]) -> Vec<u8> {
    let mut gaps = Vec::with_capacity(positions.len());
    let mut next_position = 0usize;
    for &position in positions {
        let mut gap = usize::from(position) - next_position;
        while gap >= 255 {
            gaps.push(255);
            gap -= 255;
        }
        gaps.push(u8::try_from(gap).unwrap_or_else(|_| unreachable!("gap is below 255")));
        next_position = usize::from(position) + 1;
    }
    gaps
}

#[inline(always)]
fn for_each_gap(gaps: &[u8], mut function: impl FnMut(usize)) {
    let mut next_position = 0usize;
    let mut gap = 0usize;
    for &part in gaps {
        gap += usize::from(part);
        if part < 255 {
            let position = next_position + gap;
            function(position);
            next_position = position + 1;
            gap = 0;
        }
    }
}

fn gap_rank(gaps: &[u8], target: usize) -> Option<usize> {
    let mut next_position = 0usize;
    let mut gap = 0usize;
    let mut rank = 0usize;
    for &part in gaps {
        gap += usize::from(part);
        if part < 255 {
            let position = next_position + gap;
            if position >= target {
                return (position == target).then_some(rank);
            }
            next_position = position + 1;
            gap = 0;
            rank += 1;
        }
    }
    None
}

#[inline(always)]
fn for_each_set_bit(bitmap: &[u64], mut function: impl FnMut(usize)) {
    for (word_index, &word) in bitmap.iter().enumerate() {
        let mut remaining = word;
        while remaining != 0 {
            let bit = remaining.trailing_zeros() as usize;
            function(word_index * 64 + bit);
            remaining &= remaining - 1;
        }
    }
}

#[inline(always)]
fn read_bits(bytes: &[u8], bit_offset: usize, width: u8) -> u32 {
    if width == 0 {
        return 0;
    }
    let byte_offset = bit_offset / 8;
    let shift = bit_offset % 8;
    // SAFETY: Each nonempty stream contains seven readable padding bytes.
    let word = unsafe {
        bytes
            .as_ptr()
            .add(byte_offset)
            .cast::<u64>()
            .read_unaligned()
    };
    u32::try_from((u64::from_le(word) >> shift) & u64::from(low_mask(width)))
        .unwrap_or_else(|_| unreachable!("masked bit stream value fits u32"))
}

#[inline(always)]
fn read_bits64(bytes: &[u8], bit_offset: usize, width: u8) -> u64 {
    if width == 0 {
        return 0;
    }
    let byte_offset = bit_offset / 8;
    let shift = bit_offset % 8;
    // SAFETY: Each nonempty stream contains fifteen readable padding bytes.
    let low = unsafe {
        bytes
            .as_ptr()
            .add(byte_offset)
            .cast::<u64>()
            .read_unaligned()
    };
    let mut value = u64::from_le(low) >> shift;
    if shift + usize::from(width) > 64 {
        // SAFETY: Each nonempty stream contains fifteen readable padding bytes.
        let high = unsafe {
            bytes
                .as_ptr()
                .add(byte_offset + size_of::<u64>())
                .cast::<u64>()
                .read_unaligned()
        };
        value |= u64::from_le(high) << (64 - shift);
    }
    value & low_mask64(width)
}

struct BitWriter {
    bytes: Vec<u8>,
    bit_len: usize,
}

struct BitWriter64 {
    bytes: Vec<u8>,
    bit_len: usize,
}

impl BitWriter64 {
    fn with_capacity(bits: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(bits.div_ceil(8) + 15),
            bit_len: 0,
        }
    }

    fn write(&mut self, value: u64, width: u8) {
        for bit in 0..width {
            let position = self.bit_len + usize::from(bit);
            let byte_index = position / 8;
            if byte_index == self.bytes.len() {
                self.bytes.push(0);
            }
            self.bytes[byte_index] |= (((value >> bit) & 1) as u8) << (position % 8);
        }
        self.bit_len += usize::from(width);
    }

    fn finish(mut self) -> Vec<u8> {
        if self.bit_len == 0 {
            return Vec::new();
        }
        self.bytes.extend_from_slice(&[0; 15]);
        self.bytes
    }
}

impl BitWriter {
    fn with_capacity(bits: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(bits.div_ceil(8) + STREAM_PADDING),
            bit_len: 0,
        }
    }

    fn write(&mut self, value: u32, width: u8) {
        for bit in 0..width {
            let position = self.bit_len + usize::from(bit);
            let byte_index = position / 8;
            if byte_index == self.bytes.len() {
                self.bytes.push(0);
            }
            self.bytes[byte_index] |= (((value >> bit) & 1) as u8) << (position % 8);
        }
        self.bit_len += usize::from(width);
    }

    fn finish(mut self) -> Vec<u8> {
        if self.bit_len == 0 {
            return Vec::new();
        }
        self.bytes.extend_from_slice(&[0; STREAM_PADDING]);
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use super::IntMultCodec32;
    use super::IntMultDenseCodec64;

    #[test]
    fn roundtrip_and_scalar_access() -> VortexResult<()> {
        let values = (0_u32..20_000)
            .map(|index| match index % 5 {
                0 => u32::MAX - index,
                1 => 2_000_000_000 + index % 10,
                _ => 1_000_000 + (index % 31) * 10,
            })
            .collect::<Vec<_>>();
        for base in [2, 10, 100, 1_000] {
            let codec = IntMultCodec32::encode(&values, base)?;
            assert_eq!(codec.decode(), values);
            assert_eq!(codec.decode_gaps(), values);
            if base <= 16 {
                assert_eq!(codec.decode_pairs(), values);
            }
            for index in [0, 1, 1_023, 1_024, values.len() - 1] {
                assert_eq!(codec.scalar_at(index)?, values[index]);
                assert_eq!(codec.scalar_at_gaps(index)?, values[index]);
            }
        }
        Ok(())
    }

    #[test]
    fn dense_u64_roundtrip_and_scalar_access() -> VortexResult<()> {
        let values = (0_u64..20_000)
            .map(|index| match index % 4 {
                0 => u64::MAX - index,
                1 => (1_u64 << 63) + index % 10,
                _ => 1_000_000 + (index % 31) * 10,
            })
            .collect::<Vec<_>>();
        for base in [2, 10, 100, 1_000] {
            let codec = IntMultDenseCodec64::encode(&values, base)?;
            assert_eq!(codec.decode(), values);
            for index in [0, 1, 1_023, 1_024, values.len() - 1] {
                assert_eq!(codec.scalar_at(index)?, values[index]);
            }
        }
        Ok(())
    }
}
