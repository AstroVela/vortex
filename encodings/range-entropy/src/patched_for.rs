// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fastlanes::BitPacking;
use vortex_error::VortexResult;

const CHUNK_LEN: usize = 1024;
const BASE_SAMPLE_LEN: usize = 64;
const HIGH_PADDING: usize = 15;
const SERIALIZED_BLOCK_METADATA_BYTES: usize = 12;

/// A prototype patched frame-of-reference codec for ordered unsigned latents.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PatchedFoRCodec {
    len: usize,
    blocks: Vec<PatchedFoRBlock>,
}

/// Serialized children for the one-reference block residual codec.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockResidualParts {
    pub len: usize,
    pub bases: Vec<u64>,
    pub residual_widths: Vec<u8>,
    pub high_widths: Vec<u8>,
    pub residual_starts: Vec<u32>,
    pub patch_starts: Vec<u32>,
    pub high_starts: Vec<u32>,
    pub residual_words: Vec<u64>,
    pub patch_positions: Vec<u16>,
    pub patch_highs: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PatchedFoRBlock {
    len: u16,
    bases: Vec<u64>,
    cluster_width: u8,
    residual_width: u8,
    high_width: u8,
    clusters: Vec<u64>,
    residuals: Vec<u64>,
    patch_positions: Vec<u16>,
    patch_highs: Vec<u8>,
}

impl PatchedFoRCodec {
    /// Encode ordered unsigned latents in independent 1024-value blocks.
    pub fn encode(values: &[u64]) -> VortexResult<Self> {
        let blocks = values
            .chunks(CHUNK_LEN)
            .map(encode_block)
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(Self {
            len: values.len(),
            blocks,
        })
    }

    /// Encode with one base and a residual-width histogram per block.
    pub fn encode_single_base(values: &[u64]) -> VortexResult<Self> {
        let blocks = values
            .chunks(CHUNK_LEN)
            .map(encode_single_base_block)
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(Self {
            len: values.len(),
            blocks,
        })
    }

    /// Convert a one-reference codec into serialized array children.
    pub fn into_single_base_parts(self) -> VortexResult<BlockResidualParts> {
        let mut parts = BlockResidualParts {
            len: self.len,
            bases: Vec::with_capacity(self.blocks.len()),
            residual_widths: Vec::with_capacity(self.blocks.len()),
            high_widths: Vec::with_capacity(self.blocks.len()),
            residual_starts: Vec::with_capacity(self.blocks.len() + 1),
            patch_starts: Vec::with_capacity(self.blocks.len() + 1),
            high_starts: Vec::with_capacity(self.blocks.len() + 1),
            residual_words: Vec::new(),
            patch_positions: Vec::new(),
            patch_highs: Vec::new(),
        };
        parts.residual_starts.push(0);
        parts.patch_starts.push(0);
        parts.high_starts.push(0);
        for block in self.blocks {
            vortex_error::vortex_ensure!(
                block.bases.len() == 1 && block.cluster_width == 0,
                "block residual parts require one reference per block"
            );
            parts.bases.push(block.bases[0]);
            parts.residual_widths.push(block.residual_width);
            parts.high_widths.push(block.high_width);
            parts.residual_words.extend(block.residuals);
            parts.patch_positions.extend(block.patch_positions);
            parts.patch_highs.extend(block.patch_highs);
            parts
                .residual_starts
                .push(u32::try_from(parts.residual_words.len())?);
            parts
                .patch_starts
                .push(u32::try_from(parts.patch_positions.len())?);
            parts
                .high_starts
                .push(u32::try_from(parts.patch_highs.len())?);
        }
        Ok(parts)
    }

    /// Reconstruct a one-reference codec from serialized array children.
    pub fn try_from_single_base_parts(parts: BlockResidualParts) -> VortexResult<Self> {
        let block_count = parts.len.div_ceil(CHUNK_LEN);
        vortex_error::vortex_ensure!(
            parts.bases.len() == block_count
                && parts.residual_widths.len() == block_count
                && parts.high_widths.len() == block_count,
            "block residual metadata child lengths are invalid"
        );
        validate_starts(
            &parts.residual_starts,
            block_count,
            parts.residual_words.len(),
            "residual",
        )?;
        validate_starts(
            &parts.patch_starts,
            block_count,
            parts.patch_positions.len(),
            "patch",
        )?;
        validate_starts(
            &parts.high_starts,
            block_count,
            parts.patch_highs.len(),
            "patch high",
        )?;

        let mut blocks = Vec::with_capacity(block_count);
        for block_index in 0..block_count {
            let residual_start = usize::try_from(parts.residual_starts[block_index])?;
            let residual_stop = usize::try_from(parts.residual_starts[block_index + 1])?;
            let patch_start = usize::try_from(parts.patch_starts[block_index])?;
            let patch_stop = usize::try_from(parts.patch_starts[block_index + 1])?;
            let high_start = usize::try_from(parts.high_starts[block_index])?;
            let high_stop = usize::try_from(parts.high_starts[block_index + 1])?;
            let block_start = block_index * CHUNK_LEN;
            let block_len = (parts.len - block_start).min(CHUNK_LEN);
            let residual_width = parts.residual_widths[block_index];
            let high_width = parts.high_widths[block_index];
            vortex_error::vortex_ensure!(
                residual_width <= 64 && high_width <= 64,
                "block residual bit width exceeds 64"
            );
            vortex_error::vortex_ensure!(
                residual_stop - residual_start
                    == CHUNK_LEN * usize::from(residual_width) / u64::BITS as usize,
                "block residual packed word count is invalid"
            );
            let patch_positions = &parts.patch_positions[patch_start..patch_stop];
            vortex_error::vortex_ensure!(
                patch_positions
                    .iter()
                    .all(|&position| usize::from(position) < block_len)
                    && patch_positions
                        .windows(2)
                        .all(|window| window[0] < window[1]),
                "block residual patch positions are invalid"
            );
            let expected_high_len = if patch_positions.is_empty() {
                0
            } else {
                (patch_positions.len() * usize::from(high_width)).div_ceil(8) + HIGH_PADDING
            };
            vortex_error::vortex_ensure!(
                high_stop - high_start == expected_high_len,
                "block residual patch high payload length is invalid"
            );
            blocks.push(PatchedFoRBlock {
                len: u16::try_from(block_len)?,
                bases: vec![parts.bases[block_index]],
                cluster_width: 0,
                residual_width,
                high_width,
                clusters: Vec::new(),
                residuals: parts.residual_words[residual_start..residual_stop].to_vec(),
                patch_positions: parts.patch_positions[patch_start..patch_stop].to_vec(),
                patch_highs: parts.patch_highs[high_start..high_stop].to_vec(),
            });
        }
        Ok(Self {
            len: parts.len,
            blocks,
        })
    }

    /// Decode all values.
    pub fn decode(&self) -> VortexResult<Vec<u64>> {
        let mut values = Vec::with_capacity(self.len);
        let mut clusters = [0u64; CHUNK_LEN];
        let mut residuals = [0u64; CHUNK_LEN];
        for block in &self.blocks {
            clusters.fill(0);
            residuals.fill(0);
            if block.cluster_width > 0 {
                // SAFETY: The encoder creates one complete FastLanes chunk.
                unsafe {
                    u64::unchecked_unpack(
                        usize::from(block.cluster_width),
                        &block.clusters,
                        &mut clusters,
                    );
                }
            }
            if block.residual_width > 0 {
                // SAFETY: The encoder creates one complete FastLanes chunk.
                unsafe {
                    u64::unchecked_unpack(
                        usize::from(block.residual_width),
                        &block.residuals,
                        &mut residuals,
                    );
                }
            }

            let mut high_bit_position = 0usize;
            for &position in &block.patch_positions {
                // SAFETY: The encoder appends fifteen readable padding bytes.
                let high = unsafe {
                    read_wide_bits(&block.patch_highs, high_bit_position, block.high_width)
                };
                residuals[usize::from(position)] |= high << block.residual_width;
                high_bit_position += usize::from(block.high_width);
            }

            let block_len = usize::from(block.len);
            match block.bases.as_slice() {
                [base] => {
                    for residual in &mut residuals[..block_len] {
                        *residual = residual.wrapping_add(*base);
                    }
                }
                bases => {
                    for (index, residual) in residuals[..block_len].iter_mut().enumerate() {
                        let cluster = usize::try_from(clusters[index])?;
                        *residual = residual.wrapping_add(bases[cluster]);
                    }
                }
            }
            values.extend_from_slice(&residuals[..block_len]);
        }
        Ok(values)
    }

    /// Decode ordered `f64` latents with a fused inverse transform.
    pub fn decode_ordered_f64(&self) -> VortexResult<Vec<f64>> {
        let mut values = Vec::with_capacity(self.len);
        let mut clusters = [0_u64; CHUNK_LEN];
        let mut residuals = [0_u64; CHUNK_LEN];
        for block in &self.blocks {
            clusters.fill(0);
            residuals.fill(0);
            if block.cluster_width > 0 {
                // SAFETY: The encoder creates one complete FastLanes chunk.
                unsafe {
                    u64::unchecked_unpack(
                        usize::from(block.cluster_width),
                        &block.clusters,
                        &mut clusters,
                    );
                }
            }
            if block.residual_width > 0 {
                // SAFETY: The encoder creates one complete FastLanes chunk.
                unsafe {
                    u64::unchecked_unpack(
                        usize::from(block.residual_width),
                        &block.residuals,
                        &mut residuals,
                    );
                }
            }
            let mut high_bit_position = 0_usize;
            for &position in &block.patch_positions {
                // SAFETY: The encoder appends fifteen readable padding bytes.
                let high = unsafe {
                    read_wide_bits(&block.patch_highs, high_bit_position, block.high_width)
                };
                residuals[usize::from(position)] |= high << block.residual_width;
                high_bit_position += usize::from(block.high_width);
            }
            let block_len = usize::from(block.len);
            match block.bases.as_slice() {
                [base] => {
                    for residual in &mut residuals[..block_len] {
                        *residual = residual.wrapping_add(*base);
                    }
                }
                bases => {
                    for (index, residual) in residuals[..block_len].iter_mut().enumerate() {
                        let cluster = usize::try_from(clusters[index])?;
                        *residual = residual.wrapping_add(bases[cluster]);
                    }
                }
            }
            values.extend(residuals[..block_len].iter().map(|&ordered| {
                let bits = if ordered & (1_u64 << 63) == 0 {
                    !ordered
                } else {
                    ordered ^ (1_u64 << 63)
                };
                f64::from_bits(bits)
            }));
        }
        Ok(values)
    }

    /// Decode one value with direct packed access and a binary patch search.
    pub fn scalar_at(&self, index: usize) -> VortexResult<u64> {
        vortex_error::vortex_ensure!(
            index < self.len,
            "index {index} is out of bounds for length {}",
            self.len
        );
        let block = &self.blocks[index / CHUNK_LEN];
        let index_in_block = index % CHUNK_LEN;
        let cluster = if block.cluster_width == 0 {
            0
        } else {
            // SAFETY: The encoder creates one complete FastLanes chunk.
            unsafe {
                usize::try_from(u64::unchecked_unpack_single(
                    usize::from(block.cluster_width),
                    &block.clusters,
                    index_in_block,
                ))?
            }
        };
        let mut residual = if block.residual_width == 0 {
            0
        } else {
            // SAFETY: The encoder creates one complete FastLanes chunk.
            unsafe {
                u64::unchecked_unpack_single(
                    usize::from(block.residual_width),
                    &block.residuals,
                    index_in_block,
                )
            }
        };
        if let Ok(patch_index) = block
            .patch_positions
            .binary_search(&u16::try_from(index_in_block)?)
        {
            // SAFETY: The encoder appends fifteen readable padding bytes.
            let high = unsafe {
                read_wide_bits(
                    &block.patch_highs,
                    patch_index * usize::from(block.high_width),
                    block.high_width,
                )
            };
            residual |= high << block.residual_width;
        }
        Ok(block.bases[cluster].wrapping_add(residual))
    }

    /// Return the logical value count.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Return true when the codec contains no values.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Return the encoded bytes, including estimated serialized block metadata.
    pub fn encoded_size(&self) -> usize {
        self.blocks
            .iter()
            .map(|block| {
                SERIALIZED_BLOCK_METADATA_BYTES
                    + block.bases.len() * size_of::<u64>()
                    + block.clusters.len() * size_of::<u64>()
                    + block.residuals.len() * size_of::<u64>()
                    + block.patch_positions.len() * size_of::<u16>()
                    + block.patch_highs.len()
            })
            .sum()
    }

    /// Return the logical block count.
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Return the total patch count.
    pub fn patch_count(&self) -> usize {
        self.blocks
            .iter()
            .map(|block| block.patch_positions.len())
            .sum()
    }

    /// Return the sum of base counts across all blocks.
    pub fn total_base_count(&self) -> usize {
        self.blocks.iter().map(|block| block.bases.len()).sum()
    }

    /// Return the sum of main residual widths across all blocks.
    pub fn total_residual_width(&self) -> usize {
        self.blocks
            .iter()
            .map(|block| usize::from(block.residual_width))
            .sum()
    }
}

fn encode_single_base_block(values: &[u64]) -> VortexResult<PatchedFoRBlock> {
    let base = values.iter().copied().min().unwrap_or(0);
    let mut residuals = Vec::with_capacity(CHUNK_LEN);
    let mut width_counts = [0usize; 65];
    let mut maximum_width = 0u8;
    for &value in values {
        let residual = value - base;
        let width = bit_width(residual);
        residuals.push(residual);
        width_counts[usize::from(width)] += 1;
        maximum_width = maximum_width.max(width);
    }
    residuals.resize(CHUNK_LEN, 0);

    let mut patch_count = values.len();
    let mut best = (usize::MAX, maximum_width, 0u8, 0usize);
    for residual_width in 0..=maximum_width {
        patch_count -= width_counts[usize::from(residual_width)];
        let high_width = if patch_count == 0 {
            0
        } else {
            maximum_width - residual_width
        };
        let cost_bits = usize::from(residual_width) * CHUNK_LEN
            + patch_count * (u16::BITS as usize + usize::from(high_width))
            + u64::BITS as usize
            + SERIALIZED_BLOCK_METADATA_BYTES * 8
            + usize::from(patch_count > 0) * HIGH_PADDING * 8;
        if cost_bits < best.0 {
            best = (cost_bits, residual_width, high_width, patch_count);
        }
    }

    materialize_block(
        values,
        BlockPlan {
            bases: vec![base],
            cluster_width: 0,
            residual_width: best.1,
            high_width: best.2,
            clusters: vec![0; CHUNK_LEN],
            residuals,
            patch_count: best.3,
            cost_bits: best.0,
        },
    )
}

fn validate_starts(
    starts: &[u32],
    block_count: usize,
    payload_len: usize,
    name: &str,
) -> VortexResult<()> {
    vortex_error::vortex_ensure!(
        starts.len() == block_count + 1,
        "block residual {name} offsets have an invalid length"
    );
    vortex_error::vortex_ensure!(
        starts.first() == Some(&0) && usize::try_from(*starts.last().unwrap_or(&0))? == payload_len,
        "block residual {name} offsets do not cover the payload"
    );
    vortex_error::vortex_ensure!(
        starts.windows(2).all(|window| window[0] <= window[1]),
        "block residual {name} offsets are not ordered"
    );
    Ok(())
}

fn encode_block(values: &[u64]) -> VortexResult<PatchedFoRBlock> {
    let sampled = sampled_values(values);
    let mut best: Option<BlockPlan> = None;
    for requested_base_count in [1, 2, 4] {
        let bases = quantile_bases(&sampled, requested_base_count);
        let plan = plan_block(values, bases)?;
        if best
            .as_ref()
            .is_none_or(|current| plan.cost_bits < current.cost_bits)
        {
            best = Some(plan);
        }
    }
    materialize_block(
        values,
        best.ok_or_else(|| vortex_error::vortex_err!("patched FoR block has no plan"))?,
    )
}

fn sampled_values(values: &[u64]) -> Vec<u64> {
    if values.len() <= BASE_SAMPLE_LEN {
        let mut sampled = values.to_vec();
        sampled.sort_unstable();
        return sampled;
    }

    let mut sampled = (0..BASE_SAMPLE_LEN)
        .map(|index| values[index * values.len() / BASE_SAMPLE_LEN])
        .collect::<Vec<_>>();
    sampled.push(values.iter().copied().min().unwrap_or(0));
    sampled.sort_unstable();
    sampled
}

struct BlockPlan {
    bases: Vec<u64>,
    cluster_width: u8,
    residual_width: u8,
    high_width: u8,
    clusters: Vec<u64>,
    residuals: Vec<u64>,
    patch_count: usize,
    cost_bits: usize,
}

fn quantile_bases(sorted: &[u64], requested_count: usize) -> Vec<u64> {
    let mut bases = (0..requested_count)
        .map(|index| sorted[index * sorted.len() / requested_count])
        .collect::<Vec<_>>();
    bases.dedup();
    bases
}

fn plan_block(values: &[u64], bases: Vec<u64>) -> VortexResult<BlockPlan> {
    let cluster_width = bit_width(u64::try_from(bases.len() - 1)?);
    let mut clusters = Vec::with_capacity(CHUNK_LEN);
    let mut residuals = Vec::with_capacity(CHUNK_LEN);
    let mut width_counts = [0usize; 65];
    let mut maximum_width = 0u8;
    for &value in values {
        let cluster = bases
            .partition_point(|&base| base <= value)
            .saturating_sub(1);
        let residual = value - bases[cluster];
        clusters.push(u64::try_from(cluster)?);
        residuals.push(residual);
        let width = bit_width(residual);
        width_counts[usize::from(width)] += 1;
        maximum_width = maximum_width.max(width);
    }
    clusters.resize(CHUNK_LEN, 0);
    residuals.resize(CHUNK_LEN, 0);

    let mut patch_count = values.len();
    let mut best = (usize::MAX, maximum_width, 0u8, patch_count);
    for residual_width in 0..=maximum_width {
        patch_count -= width_counts[usize::from(residual_width)];
        let high_width = if patch_count == 0 {
            0
        } else {
            maximum_width - residual_width
        };
        let cost_bits = usize::from(cluster_width) * CHUNK_LEN
            + usize::from(residual_width) * CHUNK_LEN
            + patch_count * (u16::BITS as usize + usize::from(high_width))
            + bases.len() * u64::BITS as usize
            + SERIALIZED_BLOCK_METADATA_BYTES * 8
            + usize::from(patch_count > 0) * HIGH_PADDING * 8;
        if cost_bits < best.0 {
            best = (cost_bits, residual_width, high_width, patch_count);
        }
    }

    Ok(BlockPlan {
        bases,
        cluster_width,
        residual_width: best.1,
        high_width: best.2,
        clusters,
        residuals,
        patch_count: best.3,
        cost_bits: best.0,
    })
}

fn materialize_block(values: &[u64], plan: BlockPlan) -> VortexResult<PatchedFoRBlock> {
    let clusters = fast_pack(&plan.clusters, plan.cluster_width);
    let residual_mask = low_mask(plan.residual_width);
    let low_residuals = plan
        .residuals
        .iter()
        .map(|&residual| residual & residual_mask)
        .collect::<Vec<_>>();
    let residuals = fast_pack(&low_residuals, plan.residual_width);
    let mut patch_positions = Vec::with_capacity(plan.patch_count);
    let mut patch_highs = BitWriter::with_capacity(plan.patch_count * 8);
    if plan.high_width > 0 {
        for (position, &residual) in plan.residuals[..values.len()].iter().enumerate() {
            let high = residual >> plan.residual_width;
            if high != 0 {
                patch_positions.push(u16::try_from(position)?);
                patch_highs.write(high, plan.high_width);
            }
        }
    }
    let patch_highs = if patch_positions.is_empty() {
        Vec::new()
    } else {
        let mut encoded = patch_highs.finish();
        encoded.extend_from_slice(&[0; HIGH_PADDING]);
        encoded
    };

    Ok(PatchedFoRBlock {
        len: u16::try_from(values.len())?,
        bases: plan.bases,
        cluster_width: plan.cluster_width,
        residual_width: plan.residual_width,
        high_width: plan.high_width,
        clusters,
        residuals,
        patch_positions,
        patch_highs,
    })
}

fn fast_pack(values: &[u64], width: u8) -> Vec<u64> {
    if width == 0 {
        return Vec::new();
    }
    let mut packed = vec![0u64; CHUNK_LEN * usize::from(width) / u64::BITS as usize];
    // SAFETY: Both slices have the exact lengths required for one FastLanes chunk.
    unsafe { u64::unchecked_pack(usize::from(width), values, &mut packed) };
    packed
}

fn bit_width(value: u64) -> u8 {
    u8::try_from(u64::BITS - value.leading_zeros()).unwrap_or(64)
}

fn low_mask(bits: u8) -> u64 {
    match bits {
        0 => 0,
        64 => u64::MAX,
        _ => (1_u64 << bits) - 1,
    }
}

pub(crate) unsafe fn read_wide_bits(bytes: &[u8], bit_position: usize, width: u8) -> u64 {
    let byte_position = bit_position / 8;
    let bits_past_byte = bit_position % 8;
    // SAFETY: The caller provides fifteen readable padding bytes.
    let first = unsafe { read_u64_unaligned(bytes.as_ptr().add(byte_position)) };
    if width <= 57 {
        (first >> bits_past_byte) & low_mask(width)
    } else {
        // SAFETY: The caller provides fifteen readable padding bytes.
        let second = unsafe { read_u64_unaligned(bytes.as_ptr().add(byte_position + 7)) };
        let processed = 56 - bits_past_byte;
        ((first >> bits_past_byte) | (second << processed)) & low_mask(width)
    }
}

unsafe fn read_u64_unaligned(pointer: *const u8) -> u64 {
    // SAFETY: The caller provides eight readable bytes at the pointer.
    u64::from_le(unsafe { pointer.cast::<u64>().read_unaligned() })
}

struct BitWriter {
    bytes: Vec<u8>,
    pending: u64,
    pending_bits: u8,
}

impl BitWriter {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(capacity),
            pending: 0,
            pending_bits: 0,
        }
    }

    fn write(&mut self, value: u64, width: u8) {
        if width == 0 {
            return;
        }
        let available = 64 - self.pending_bits;
        self.pending |= value << self.pending_bits;
        if width < available {
            self.pending_bits += width;
            return;
        }
        self.bytes.extend_from_slice(&self.pending.to_le_bytes());
        let remaining = width - available;
        self.pending = if remaining == 0 {
            0
        } else {
            value >> available
        };
        self.pending_bits = remaining;
    }

    fn finish(mut self) -> Vec<u8> {
        if self.pending_bits > 0 {
            let byte_count = usize::from(self.pending_bits).div_ceil(8);
            self.bytes
                .extend_from_slice(&self.pending.to_le_bytes()[..byte_count]);
        }
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use super::PatchedFoRCodec;

    #[test]
    fn roundtrip_lengths_and_domains() -> VortexResult<()> {
        for len in [0, 1, 1_023, 1_024, 1_025, 20_000] {
            let values = (0..len)
                .map(|index| match index % 4 {
                    0 => index as u64,
                    1 => 1_000_000 + index as u64 % 31,
                    2 => u64::MAX - index as u64 % 13,
                    _ => 42,
                })
                .collect::<Vec<_>>();
            let codec = PatchedFoRCodec::encode(&values)?;
            assert_eq!(codec.decode()?, values);
            for (index, &value) in values.iter().enumerate() {
                assert_eq!(codec.scalar_at(index)?, value);
            }
        }
        Ok(())
    }

    #[test]
    fn constant_roundtrip() -> VortexResult<()> {
        let values = vec![42; 10_000];
        let codec = PatchedFoRCodec::encode(&values)?;
        assert_eq!(codec.decode()?, values);
        assert_eq!(codec.scalar_at(4_321)?, 42);
        Ok(())
    }

    #[test]
    fn single_base_parts_roundtrip() -> VortexResult<()> {
        let values = (0..4_099)
            .map(|index| {
                let value = u64::try_from(index)?;
                Ok(1_000_000_u64.wrapping_add(value * value))
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let parts = PatchedFoRCodec::encode_single_base(&values)?.into_single_base_parts()?;
        let codec = PatchedFoRCodec::try_from_single_base_parts(parts)?;
        assert_eq!(codec.decode()?, values);
        Ok(())
    }
}
