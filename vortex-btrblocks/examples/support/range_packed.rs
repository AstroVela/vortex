// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Ordering;
use std::mem::MaybeUninit;
use std::ops::Range;

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

const MAX_BINS: usize = 64;
const SPLIT_CANDIDATES: usize = 64;
const TRAINING_SAMPLE_SIZE: usize = 8_192;
const BIN_TABLE_COST_BITS: f64 = 128.0;
const SYMBOL_PADDING: usize = 8;
const OFFSET_PADDING: usize = 15;
const CHECKPOINT_INTERVAL: usize = 32;
const MAX_BLOCK_LEN: usize = 1_024;

/// Fixed-width range identifiers with variable-width offsets and scalar checkpoints.
#[derive(Clone, Debug)]
pub struct RangePackedCodec {
    len: usize,
    block_len: usize,
    symbol_width: u8,
    bins: Vec<Bin>,
    block_offsets: Vec<u32>,
    payload: Vec<u8>,
}

#[derive(Clone, Copy, Debug)]
struct Bin {
    lower: u64,
    offset_bits: u8,
}

impl RangePackedCodec {
    pub fn encode(values: &[u64], block_len: usize) -> VortexResult<Self> {
        vortex_ensure!(
            block_len > 0 && block_len <= MAX_BLOCK_LEN,
            "range packed block length must be in 1..={MAX_BLOCK_LEN}"
        );
        if values.is_empty() {
            return Ok(Self {
                len: 0,
                block_len,
                symbol_width: 0,
                bins: Vec::new(),
                block_offsets: vec![0],
                payload: Vec::new(),
            });
        }

        let (sample, minimum, maximum) = training_sample(values);
        let bins = cover_domain(optimize_bins(&sample), minimum, maximum);
        vortex_ensure!(
            bins.len().is_power_of_two(),
            "range packed bin count must be a power of two"
        );
        let symbols = assign_bins(values, &bins)?;
        let symbol_width = bit_width(u64::try_from(bins.len() - 1)?);
        let block_count = values.len().div_ceil(block_len);
        let mut block_offsets = Vec::with_capacity(block_count + 1);
        let mut payload = Vec::new();
        block_offsets.push(0);
        for block_index in 0..block_count {
            let start = block_index * block_len;
            let stop = (start + block_len).min(values.len());
            encode_block(
                &values[start..stop],
                &symbols[start..stop],
                &bins,
                symbol_width,
                &mut payload,
            )?;
            block_offsets.push(u32::try_from(payload.len())?);
        }

        Ok(Self {
            len: values.len(),
            block_len,
            symbol_width,
            bins,
            block_offsets,
            payload,
        })
    }

    pub fn decode(&self) -> VortexResult<Vec<u64>> {
        if self.bins.iter().all(|bin| bin.offset_bits <= 57) {
            self.decode_with_width::<false>()
        } else {
            self.decode_with_width::<true>()
        }
    }

    fn decode_with_width<const WIDE: bool>(&self) -> VortexResult<Vec<u64>> {
        let mut values = Vec::with_capacity(self.len);
        let table = DecodeTable::new(&self.bins);
        for block_index in 0..self.block_count() {
            let payload = self.block_payload(block_index)?;
            let value_count = self.block_value_count(block_index);
            let output_start = values.len();
            decode_block::<WIDE>(
                payload,
                value_count,
                self.symbol_width,
                &table,
                &mut values.spare_capacity_mut()[..value_count],
            )?;
            // SAFETY: decode_block initialized each output slot.
            unsafe { values.set_len(output_start + value_count) };
        }
        Ok(values)
    }

    pub fn scalar_at(&self, index: usize) -> VortexResult<u64> {
        vortex_ensure!(index < self.len, "range packed index exceeds array length");
        let block_index = index / self.block_len;
        let index_in_block = index % self.block_len;
        let value_count = self.block_value_count(block_index);
        let payload = self.block_payload(block_index)?;
        let layout = BlockLayout::new(value_count, self.symbol_width, payload.len())?;
        let symbol = read_symbol(payload, index_in_block, self.symbol_width)?;
        let node = *self
            .bins
            .get(symbol)
            .ok_or_else(|| vortex_error::vortex_err!("range packed symbol exceeds bin table"))?;
        let checkpoint_index = index_in_block / CHECKPOINT_INTERVAL;
        let checkpoint_start = layout.checkpoints_start + checkpoint_index * size_of::<u16>();
        let checkpoint =
            u16::from_le_bytes([payload[checkpoint_start], payload[checkpoint_start + 1]]);
        let mut offset_position = usize::from(checkpoint);
        let scan_start = checkpoint_index * CHECKPOINT_INTERVAL;
        for preceding in scan_start..index_in_block {
            let preceding_symbol = read_symbol(payload, preceding, self.symbol_width)?;
            offset_position += usize::from(
                self.bins
                    .get(preceding_symbol)
                    .ok_or_else(|| {
                        vortex_error::vortex_err!("range packed symbol exceeds bin table")
                    })?
                    .offset_bits,
            );
        }
        let offset = read_bits_padded(
            &payload[layout.offsets_start..],
            offset_position,
            node.offset_bits,
        );
        Ok(node.lower.wrapping_add(offset))
    }

    pub fn encoded_size(&self) -> usize {
        self.bins.len() * (size_of::<u64>() + size_of::<u8>())
            + self.block_offsets.len() * size_of::<u32>()
            + self.payload.len()
    }

    pub fn bin_count(&self) -> usize {
        self.bins.len()
    }

    fn block_count(&self) -> usize {
        self.len.div_ceil(self.block_len)
    }

    fn block_value_count(&self, block_index: usize) -> usize {
        (self.len - block_index * self.block_len).min(self.block_len)
    }

    fn block_payload(&self, block_index: usize) -> VortexResult<&[u8]> {
        let start = usize::try_from(self.block_offsets[block_index])?;
        let stop = usize::try_from(self.block_offsets[block_index + 1])?;
        Ok(&self.payload[start..stop])
    }
}

struct DecodeTable {
    lowers: [u64; MAX_BINS],
    widths: [u8; MAX_BINS],
    bin_count: usize,
}

impl DecodeTable {
    fn new(bins: &[Bin]) -> Self {
        let mut lowers = [0_u64; MAX_BINS];
        let mut widths = [0_u8; MAX_BINS];
        for (index, bin) in bins.iter().enumerate() {
            lowers[index] = bin.lower;
            widths[index] = bin.offset_bits;
        }
        Self {
            lowers,
            widths,
            bin_count: bins.len(),
        }
    }
}

struct BlockLayout {
    checkpoints_start: usize,
    offsets_start: usize,
}

impl BlockLayout {
    fn new(value_count: usize, symbol_width: u8, payload_len: usize) -> VortexResult<Self> {
        let symbol_bytes = (value_count * usize::from(symbol_width)).div_ceil(8);
        let checkpoints_start = symbol_bytes + SYMBOL_PADDING;
        let checkpoint_bytes = value_count.div_ceil(CHECKPOINT_INTERVAL) * size_of::<u16>();
        let offsets_start = checkpoints_start + checkpoint_bytes;
        vortex_ensure!(
            payload_len >= offsets_start + OFFSET_PADDING,
            "range packed block is too short"
        );
        Ok(Self {
            checkpoints_start,
            offsets_start,
        })
    }
}

fn encode_block(
    values: &[u64],
    symbols: &[u8],
    bins: &[Bin],
    symbol_width: u8,
    payload: &mut Vec<u8>,
) -> VortexResult<()> {
    let mut symbol_writer = BitWriter::with_capacity(values.len());
    let mut offset_writer = BitWriter::with_capacity(size_of_val(values));
    let mut checkpoints = Vec::with_capacity(values.len().div_ceil(CHECKPOINT_INTERVAL));
    for (index, (&value, &symbol)) in values.iter().zip(symbols).enumerate() {
        if index.is_multiple_of(CHECKPOINT_INTERVAL) {
            checkpoints.push(u16::try_from(offset_writer.bit_len())?);
        }
        symbol_writer.write(u64::from(symbol), symbol_width);
        let bin = bins[usize::from(symbol)];
        offset_writer.write(value - bin.lower, bin.offset_bits);
    }
    payload.extend_from_slice(&symbol_writer.finish());
    payload.extend_from_slice(&[0; SYMBOL_PADDING]);
    for checkpoint in checkpoints {
        payload.extend_from_slice(&checkpoint.to_le_bytes());
    }
    payload.extend_from_slice(&offset_writer.finish());
    payload.extend_from_slice(&[0; OFFSET_PADDING]);
    Ok(())
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the symbol mask never exceeds 63"
)]
fn decode_block<const WIDE: bool>(
    payload: &[u8],
    value_count: usize,
    symbol_width: u8,
    table: &DecodeTable,
    values: &mut [MaybeUninit<u64>],
) -> VortexResult<()> {
    let layout = BlockLayout::new(value_count, symbol_width, payload.len())?;
    let offsets = &payload[layout.offsets_start..];
    let mut offset_position = 0usize;
    let full_len = value_count / 8 * 8;
    let symbol_mask = low_mask(symbol_width);
    for base in (0..full_len).step_by(8) {
        let symbol_position = base * usize::from(symbol_width);
        let packed = read_bits_padded(payload, symbol_position, symbol_width * 8);
        for lane in 0..8 {
            let symbol = ((packed >> (lane * usize::from(symbol_width))) & symbol_mask) as usize;
            debug_assert!(symbol < table.bin_count);
            // SAFETY: A power-of-two bin count makes every symbol code valid.
            let width = unsafe { *table.widths.get_unchecked(symbol) };
            let offset = read_offset::<WIDE>(offsets, offset_position, width);
            offset_position += usize::from(width);
            // SAFETY: A power-of-two bin count makes every symbol code valid.
            let lower = unsafe { *table.lowers.get_unchecked(symbol) };
            // SAFETY: This lane is within a complete eight-value output batch.
            unsafe {
                values
                    .get_unchecked_mut(base + lane)
                    .write(lower.wrapping_add(offset))
            };
        }
    }
    for index in full_len..value_count {
        let symbol = read_symbol(payload, index, symbol_width)?;
        debug_assert!(symbol < table.bin_count);
        // SAFETY: A power-of-two bin count makes every symbol code valid.
        let width = unsafe { *table.widths.get_unchecked(symbol) };
        let offset = read_offset::<WIDE>(offsets, offset_position, width);
        offset_position += usize::from(width);
        // SAFETY: A power-of-two bin count makes every symbol code valid.
        let lower = unsafe { *table.lowers.get_unchecked(symbol) };
        // SAFETY: The trailing index is less than value_count.
        unsafe {
            values
                .get_unchecked_mut(index)
                .write(lower.wrapping_add(offset))
        };
    }
    let offset_bytes = offset_position.div_ceil(8);
    vortex_ensure!(
        offsets.len() == offset_bytes + OFFSET_PADDING,
        "range packed offset stream has unused bytes"
    );
    Ok(())
}

fn read_symbol(payload: &[u8], index: usize, symbol_width: u8) -> VortexResult<usize> {
    usize::try_from(read_bits_padded(
        payload,
        index * usize::from(symbol_width),
        symbol_width,
    ))
    .map_err(Into::into)
}

#[inline(always)]
fn read_offset<const WIDE: bool>(offsets: &[u8], bit_position: usize, width: u8) -> u64 {
    if WIDE {
        read_bits_padded(offsets, bit_position, width)
    } else if width == 0 {
        0
    } else {
        let byte_position = bit_position / 8;
        let bits_past_byte = bit_position % 8;
        // SAFETY: Offset streams include fifteen readable padding bytes.
        let packed = unsafe { read_u64_unaligned(offsets.as_ptr().add(byte_position)) };
        (packed >> bits_past_byte) & low_mask(width)
    }
}

fn read_bits_padded(bytes: &[u8], bit_position: usize, width: u8) -> u64 {
    if width == 0 {
        return 0;
    }
    let byte_position = bit_position / 8;
    let bits_past_byte = bit_position % 8;
    // SAFETY: Every encoded stream includes at least eight readable padding bytes.
    let first = unsafe { read_u64_unaligned(bytes.as_ptr().add(byte_position)) };
    if usize::from(width) <= 64 - bits_past_byte {
        (first >> bits_past_byte) & low_mask(width)
    } else {
        // SAFETY: Offset streams include fifteen readable padding bytes.
        let second = unsafe { read_u64_unaligned(bytes.as_ptr().add(byte_position + 7)) };
        let processed = 56 - bits_past_byte;
        ((first >> bits_past_byte) | (second << processed)) & low_mask(width)
    }
}

unsafe fn read_u64_unaligned(pointer: *const u8) -> u64 {
    // SAFETY: The caller provides eight readable bytes at the pointer.
    u64::from_le(unsafe { pointer.cast::<u64>().read_unaligned() })
}

fn assign_bins(values: &[u64], bins: &[Bin]) -> VortexResult<Vec<u8>> {
    values
        .iter()
        .map(|&value| {
            let symbol = bins
                .partition_point(|bin| bin.lower <= value)
                .saturating_sub(1);
            let bin = bins[symbol];
            vortex_ensure!(
                value - bin.lower <= low_mask(bin.offset_bits),
                "range packed value exceeds its bin"
            );
            Ok(u8::try_from(symbol)?)
        })
        .collect()
}

fn optimize_bins(sorted: &[u64]) -> Vec<Bin> {
    let mut segments = Vec::with_capacity(MAX_BINS);
    segments.push(0..sorted.len());
    let mut best_segments = segments.clone();
    let mut best_cost = partition_cost(sorted, &segments);

    while segments.len() < MAX_BINS {
        let best = segments
            .iter()
            .enumerate()
            .filter_map(|(segment_index, segment)| {
                best_split(sorted, segment).map(|split| (segment_index, split))
            })
            .max_by(|(_, left), (_, right)| {
                left.gain
                    .partial_cmp(&right.gain)
                    .unwrap_or(Ordering::Equal)
            });
        let Some((segment_index, split)) = best else {
            break;
        };
        if split.gain <= 0.0 {
            break;
        }
        let segment = segments.remove(segment_index);
        segments.push(segment.start..split.at);
        segments.push(split.at..segment.end);
        segments.sort_unstable_by_key(|segment| segment.start);
        let cost = partition_cost(sorted, &segments);
        if segments.len().is_power_of_two() && cost < best_cost {
            best_cost = cost;
            best_segments.clone_from(&segments);
        }
    }

    best_segments
        .into_iter()
        .map(|segment| Bin::from_values(sorted[segment.start], sorted[segment.end - 1]))
        .collect()
}

fn partition_cost(sorted: &[u64], segments: &[Range<usize>]) -> f64 {
    let residual_cost = segments
        .iter()
        .map(|segment| {
            segment.len() as f64
                * f64::from(bit_width(sorted[segment.end - 1] - sorted[segment.start]))
        })
        .sum::<f64>();
    let symbol_cost = sorted.len() as f64 * f64::from(bit_width((segments.len() - 1) as u64));
    residual_cost + symbol_cost + segments.len() as f64 * BIN_TABLE_COST_BITS
}

fn training_sample(values: &[u64]) -> (Vec<u64>, u64, u64) {
    let mut minimum = u64::MAX;
    let mut maximum = 0;
    for &value in values {
        minimum = minimum.min(value);
        maximum = maximum.max(value);
    }
    let mut sample = if values.len() <= TRAINING_SAMPLE_SIZE {
        values.to_vec()
    } else {
        (0..TRAINING_SAMPLE_SIZE)
            .map(|index| values[index * values.len() / TRAINING_SAMPLE_SIZE])
            .collect()
    };
    sample.push(minimum);
    sample.push(maximum);
    sample.sort_unstable();
    (sample, minimum, maximum)
}

fn cover_domain(mut bins: Vec<Bin>, minimum: u64, maximum: u64) -> Vec<Bin> {
    bins[0].lower = minimum;
    for index in 0..bins.len() - 1 {
        let upper = bins[index + 1].lower - 1;
        bins[index].offset_bits = bit_width(upper - bins[index].lower);
    }
    let last = bins.len() - 1;
    bins[last].offset_bits = bit_width(maximum - bins[last].lower);
    bins
}

#[derive(Clone, Copy)]
struct Split {
    at: usize,
    gain: f64,
}

fn best_split(sorted: &[u64], segment: &Range<usize>) -> Option<Split> {
    if segment.len() < 2 || sorted[segment.start] == sorted[segment.end - 1] {
        return None;
    }
    let whole_cost = segment.len() as f64
        * f64::from(bit_width(sorted[segment.end - 1] - sorted[segment.start]));
    let mut best = None;
    for candidate in 1..SPLIT_CANDIDATES {
        let mut at = segment.start + segment.len() * candidate / SPLIT_CANDIDATES;
        if at <= segment.start || at >= segment.end {
            continue;
        }
        while at < segment.end && sorted[at - 1] == sorted[at] {
            at += 1;
        }
        if at >= segment.end {
            continue;
        }
        let left_cost = (at - segment.start) as f64
            * f64::from(bit_width(sorted[at - 1] - sorted[segment.start]));
        let right_cost =
            (segment.end - at) as f64 * f64::from(bit_width(sorted[segment.end - 1] - sorted[at]));
        let split = Split {
            at,
            gain: whole_cost - left_cost - right_cost,
        };
        if best.is_none_or(|current: Split| split.gain > current.gain) {
            best = Some(split);
        }
    }
    best
}

impl Bin {
    fn from_values(lower: u64, upper: u64) -> Self {
        Self {
            lower,
            offset_bits: bit_width(upper - lower),
        }
    }
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

struct BitWriter {
    bytes: Vec<u8>,
    pending: u64,
    pending_bits: u8,
    bit_len: usize,
}

impl BitWriter {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(capacity),
            pending: 0,
            pending_bits: 0,
            bit_len: 0,
        }
    }

    fn bit_len(&self) -> usize {
        self.bit_len
    }

    fn write(&mut self, value: u64, width: u8) {
        self.bit_len += usize::from(width);
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

    use super::RangePackedCodec;

    #[test]
    fn clustered_roundtrip_and_scalar_access() -> VortexResult<()> {
        let values = (0_u64..20_000)
            .map(|index| match index % 4 {
                0 => index % 17,
                1 => 1_000_000 + index % 31,
                2 => u64::MAX - index % 13,
                _ => 1_000 + index % 7,
            })
            .collect::<Vec<_>>();
        let encoded = RangePackedCodec::encode(&values, 1_024)?;
        assert_eq!(encoded.decode()?, values);
        for index in [0, 1, 31, 32, 33, 1_023, 1_024, 19_999] {
            assert_eq!(encoded.scalar_at(index)?, values[index]);
        }
        Ok(())
    }

    #[test]
    fn full_width_roundtrip() -> VortexResult<()> {
        let values = [0, u64::MAX, 1, u64::MAX - 1];
        let encoded = RangePackedCodec::encode(&values, 4)?;
        assert_eq!(encoded.decode()?, values);
        for (index, value) in values.into_iter().enumerate() {
            assert_eq!(encoded.scalar_at(index)?, value);
        }
        Ok(())
    }
}
