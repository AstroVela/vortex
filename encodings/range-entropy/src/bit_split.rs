// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fastlanes::BitPacking;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

const CHUNK_LEN: usize = 1024;
const SERIALIZED_BLOCK_METADATA_BYTES: usize = 8;

/// A prototype block-local prefix dictionary with fixed-width suffixes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitSplitCodec {
    len: usize,
    blocks: Vec<BitSplitBlock>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct BitSplitBlock {
    len: u16,
    suffix_width: u8,
    code_width: u8,
    prefixes: Vec<u64>,
    codes: Vec<u64>,
    suffixes: Vec<u64>,
}

impl BitSplitCodec {
    /// Encode ordered unsigned latents in independent 1,024-value blocks.
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

    /// Decode all values.
    pub fn decode(&self) -> VortexResult<Vec<u64>> {
        let mut values = Vec::with_capacity(self.len);
        let mut codes = [0_u64; CHUNK_LEN];
        let mut suffixes = [0_u64; CHUNK_LEN];
        for block in &self.blocks {
            codes.fill(0);
            suffixes.fill(0);
            if block.code_width > 0 {
                // SAFETY: The encoder creates one complete FastLanes chunk.
                unsafe {
                    u64::unchecked_unpack(usize::from(block.code_width), &block.codes, &mut codes);
                }
            }
            if block.suffix_width > 0 {
                // SAFETY: The encoder creates one complete FastLanes chunk.
                unsafe {
                    u64::unchecked_unpack(
                        usize::from(block.suffix_width),
                        &block.suffixes,
                        &mut suffixes,
                    );
                }
            }

            for index in 0..usize::from(block.len) {
                let prefix = block.prefixes[usize::try_from(codes[index])?];
                values.push(join(prefix, suffixes[index], block.suffix_width));
            }
        }
        Ok(values)
    }

    /// Decode one value with two direct packed reads and one dictionary lookup.
    pub fn scalar_at(&self, index: usize) -> VortexResult<u64> {
        vortex_ensure!(
            index < self.len,
            "index {index} is out of bounds for length {}",
            self.len
        );
        let block = &self.blocks[index / CHUNK_LEN];
        let index_in_block = index % CHUNK_LEN;
        let code = if block.code_width == 0 {
            0
        } else {
            // SAFETY: The encoder creates one complete FastLanes chunk.
            unsafe {
                usize::try_from(u64::unchecked_unpack_single(
                    usize::from(block.code_width),
                    &block.codes,
                    index_in_block,
                ))?
            }
        };
        let suffix = if block.suffix_width == 0 {
            0
        } else {
            // SAFETY: The encoder creates one complete FastLanes chunk.
            unsafe {
                u64::unchecked_unpack_single(
                    usize::from(block.suffix_width),
                    &block.suffixes,
                    index_in_block,
                )
            }
        };
        Ok(join(block.prefixes[code], suffix, block.suffix_width))
    }

    /// Return the encoded bytes, including estimated serialized metadata.
    pub fn encoded_size(&self) -> usize {
        self.blocks
            .iter()
            .map(|block| {
                SERIALIZED_BLOCK_METADATA_BYTES
                    + block.prefixes.len() * size_of::<u64>()
                    + block.codes.len() * size_of::<u64>()
                    + block.suffixes.len() * size_of::<u64>()
            })
            .sum()
    }

    /// Return the average suffix width across all blocks.
    pub fn average_suffix_width(&self) -> f64 {
        self.blocks
            .iter()
            .map(|block| f64::from(block.suffix_width))
            .sum::<f64>()
            / self.blocks.len().max(1) as f64
    }

    /// Return the average prefix count across all blocks.
    pub fn average_prefix_count(&self) -> f64 {
        self.blocks
            .iter()
            .map(|block| block.prefixes.len() as f64)
            .sum::<f64>()
            / self.blocks.len().max(1) as f64
    }
}

fn encode_block(values: &[u64]) -> VortexResult<BitSplitBlock> {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted.dedup();

    let mut divergence_counts = [0_usize; 65];
    for adjacent in sorted.windows(2) {
        divergence_counts[usize::from(bit_width(adjacent[0] ^ adjacent[1]))] += 1;
    }
    let mut prefix_counts = [0_usize; 65];
    let mut active_divergences = 0_usize;
    for suffix_width in (0..=64_usize).rev() {
        prefix_counts[suffix_width] = usize::from(!sorted.is_empty()) + active_divergences;
        active_divergences += divergence_counts[suffix_width];
    }

    let mut best_suffix_width = 64_u8;
    let mut best_prefix_count = 1_usize;
    let mut best_size = CHUNK_LEN * size_of::<u64>();
    for suffix_width in 0..=64_u8 {
        let prefix_count = prefix_counts[usize::from(suffix_width)];
        let code_width = bit_width(u64::try_from(prefix_count.saturating_sub(1))?);
        let size = prefix_count * size_of::<u64>()
            + (CHUNK_LEN * usize::from(code_width)).div_ceil(8)
            + (CHUNK_LEN * usize::from(suffix_width)).div_ceil(8);
        if size < best_size {
            best_size = size;
            best_suffix_width = suffix_width;
            best_prefix_count = prefix_count;
        }
    }

    let mut prefixes = Vec::with_capacity(best_prefix_count);
    for value in sorted {
        let prefix = prefix(value, best_suffix_width);
        if prefixes.last() != Some(&prefix) {
            prefixes.push(prefix);
        }
    }
    let code_width = bit_width(u64::try_from(prefixes.len().saturating_sub(1))?);
    let suffix_mask = low_mask(best_suffix_width);
    let mut code_values = [0_u64; CHUNK_LEN];
    let mut suffix_values = [0_u64; CHUNK_LEN];
    for (index, &value) in values.iter().enumerate() {
        let prefix = prefix(value, best_suffix_width);
        let code = prefixes
            .binary_search(&prefix)
            .map_err(|_| vortex_err!("prefix dictionary does not contain {prefix}"))?;
        code_values[index] = u64::try_from(code)?;
        suffix_values[index] = value & suffix_mask;
    }

    Ok(BitSplitBlock {
        len: u16::try_from(values.len())?,
        suffix_width: best_suffix_width,
        code_width,
        prefixes,
        codes: fast_pack(&code_values, code_width),
        suffixes: fast_pack(&suffix_values, best_suffix_width),
    })
}

fn prefix(value: u64, suffix_width: u8) -> u64 {
    if suffix_width == 64 {
        0
    } else {
        value >> suffix_width
    }
}

fn join(prefix: u64, suffix: u64, suffix_width: u8) -> u64 {
    if suffix_width == 64 {
        suffix
    } else {
        (prefix << suffix_width) | suffix
    }
}

fn fast_pack(values: &[u64; CHUNK_LEN], width: u8) -> Vec<u64> {
    if width == 0 {
        return Vec::new();
    }
    let mut packed = vec![0_u64; CHUNK_LEN * usize::from(width) / u64::BITS as usize];
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

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use super::BitSplitCodec;

    #[test]
    fn roundtrip_and_scalar_access() -> VortexResult<()> {
        let values = (0..2_051)
            .map(|index| {
                let prefix = [11_u64, 1_000, 5_000, 90_000][index % 4];
                Ok((prefix << 19) | u64::try_from(index * 17)?)
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let codec = BitSplitCodec::encode(&values)?;
        assert_eq!(codec.decode()?, values);
        for (index, &value) in values.iter().enumerate() {
            assert_eq!(codec.scalar_at(index)?, value);
        }
        Ok(())
    }
}
