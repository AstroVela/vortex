// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fixed-bin range packing with bounded random access.

mod array;
mod kernel;

use std::cmp::Ordering;
use std::hash::Hash;
use std::mem::MaybeUninit;
use std::ops::Range;

pub use array::*;
use vortex_array::IntoArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::session::ArraySessionExt;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_int_mult::IntMult;
use vortex_int_mult::IntMultArray;
use vortex_session::VortexSession;

const MAX_BINS: usize = 64;
const SPLIT_CANDIDATES: usize = 64;
const TRAINING_SAMPLE_SIZE: usize = 8_192;
const BIN_TABLE_COST_BITS: f64 = 128.0;
const SYMBOL_PADDING: usize = 8;
const OFFSET_PADDING: usize = 15;
const CHECKPOINT_INTERVAL: usize = 32;
const VALIDITY_CHECKPOINT_INTERVAL: usize = 256;
const MAX_BLOCK_LEN: usize = 1_024;

/// Fixed-width range identifiers with variable-width offsets and scalar checkpoints.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RangePackedCodec {
    pub(crate) len: usize,
    pub(crate) block_len: usize,
    pub(crate) symbol_width: u8,
    pub(crate) bins: Vec<Bin>,
    pub(crate) block_offsets: Vec<u32>,
    pub(crate) payload: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Bin {
    pub(crate) lower: u64,
    pub(crate) offset_bits: u8,
}

/// A fixed-bin split before compression of its component arrays.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RangeDecomposition {
    bin_starts: Vec<u64>,
    codes: Vec<u8>,
    offsets: Vec<u64>,
}

impl RangeDecomposition {
    /// Fit at most 64 bins and split values into bin codes and offsets.
    pub fn encode(values: &[u64]) -> VortexResult<Self> {
        if values.is_empty() {
            return Ok(Self {
                bin_starts: Vec::new(),
                codes: Vec::new(),
                offsets: Vec::new(),
            });
        }

        let (sample, minimum, maximum) = training_sample(values);
        let bins = cover_domain(optimize_bins_arbitrary(&sample), minimum, maximum);
        let codes = assign_bins(values, &bins)?;
        let offsets = values
            .iter()
            .zip(&codes)
            .map(|(&value, &code)| value - bins[usize::from(code)].lower)
            .collect();
        Ok(Self {
            bin_starts: bins.iter().map(|bin| bin.lower).collect(),
            codes,
            offsets,
        })
    }

    /// Return the selected bin starts.
    pub fn bin_starts(&self) -> &[u64] {
        &self.bin_starts
    }

    /// Return the per-value bin codes.
    pub fn codes(&self) -> &[u8] {
        &self.codes
    }

    /// Return each value's offset from its selected bin start.
    pub fn offsets(&self) -> &[u64] {
        &self.offsets
    }

    /// Return the number of bits required for each fixed-width code.
    pub fn code_width(&self) -> u8 {
        self.bin_starts
            .len()
            .checked_sub(1)
            .map_or(0, |maximum| bit_width(maximum as u64))
    }

    /// Compose the raw components from a dictionary and an IntMult array.
    pub fn into_array(self, validity: Validity) -> VortexResult<IntMultArray> {
        let codes = PrimitiveArray::new(self.codes, validity).into_array();
        let starts = PrimitiveArray::from_iter(self.bin_starts).into_array();
        let references = DictArray::try_new(codes, starts)?.into_array();
        let offsets = PrimitiveArray::from_iter(self.offsets).into_array();
        IntMult::try_new(references, offsets, 1)
    }
}

/// Register the RangePacked encoding in one session.
pub fn initialize(session: &VortexSession) {
    session.arrays().register(RangePacked);
    kernel::initialize(session);
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
        self.decoder()?.decode_mapped(|value| value)
    }

    pub fn decode_mapped<T, F>(&self, map: F) -> VortexResult<Vec<T>>
    where
        F: FnMut(u64) -> T,
    {
        self.decoder()?.decode_mapped(map)
    }

    pub fn scalar_at(&self, index: usize) -> VortexResult<u64> {
        self.decoder()?.scalar_at(index)
    }

    pub fn encoded_size(&self) -> usize {
        self.bins.len() * (size_of::<u64>() + size_of::<u8>())
            + self.block_offsets.len() * size_of::<u32>()
            + self.payload.len()
    }

    pub fn bin_count(&self) -> usize {
        self.bins.len()
    }

    pub fn max_offset_bits(&self) -> u8 {
        self.bins
            .iter()
            .map(|bin| bin.offset_bits)
            .max()
            .unwrap_or_default()
    }

    pub fn offset_widths(&self) -> String {
        self.bins
            .iter()
            .map(|bin| bin.offset_bits.to_string())
            .collect::<Vec<_>>()
            .join(",")
    }

    fn decoder(&self) -> VortexResult<RangePackedDecoder<'_>> {
        RangePackedDecoder::try_new(
            self.len,
            self.block_len,
            self.symbol_width,
            BinTableView::Interleaved(&self.bins),
            &self.block_offsets,
            &self.payload,
        )
    }
}

pub(crate) fn estimate_codec_nbytes(values: &[u64], block_len: usize) -> VortexResult<usize> {
    vortex_ensure!(
        block_len > 0 && block_len <= MAX_BLOCK_LEN,
        "range packed block length must be in 1..={MAX_BLOCK_LEN}"
    );
    if values.is_empty() {
        return Ok(size_of::<u32>());
    }

    let (sample, minimum, maximum) = training_sample(values);
    let bins = cover_domain(optimize_bins(&sample), minimum, maximum);
    let symbol_width = bit_width(u64::try_from(bins.len() - 1)?);
    let block_count = values.len().div_ceil(block_len);
    let mut payload_len = 0usize;
    for block_index in 0..block_count {
        let start = block_index * block_len;
        let stop = (start + block_len).min(values.len());
        let values = &values[start..stop];
        let symbol_bytes = (values.len() * usize::from(symbol_width)).div_ceil(8);
        let checkpoint_bytes = values.len().div_ceil(CHECKPOINT_INTERVAL) * size_of::<u16>();
        let offset_bits = values.iter().try_fold(0usize, |total, &value| {
            let symbol = bins
                .partition_point(|bin| bin.lower <= value)
                .saturating_sub(1);
            let bin = bins[symbol];
            vortex_ensure!(
                value - bin.lower <= low_mask(bin.offset_bits),
                "range packed value exceeds its bin"
            );
            Ok::<usize, vortex_error::VortexError>(total + usize::from(bin.offset_bits))
        })?;
        payload_len += symbol_bytes
            + SYMBOL_PADDING
            + checkpoint_bytes
            + offset_bits.div_ceil(8)
            + OFFSET_PADDING;
    }

    Ok(bins.len() * (size_of::<u64>() + size_of::<u8>())
        + (block_count + 1) * size_of::<u32>()
        + payload_len)
}

#[derive(Clone, Copy)]
pub(crate) enum BinTableView<'a> {
    Interleaved(&'a [Bin]),
    Split { lowers: &'a [u64], widths: &'a [u8] },
}

impl BinTableView<'_> {
    fn len(self) -> usize {
        match self {
            Self::Interleaved(bins) => bins.len(),
            Self::Split { lowers, .. } => lowers.len(),
        }
    }

    fn lower(self, index: usize) -> Option<u64> {
        match self {
            Self::Interleaved(bins) => bins.get(index).map(|bin| bin.lower),
            Self::Split { lowers, .. } => lowers.get(index).copied(),
        }
    }

    fn width(self, index: usize) -> Option<u8> {
        match self {
            Self::Interleaved(bins) => bins.get(index).map(|bin| bin.offset_bits),
            Self::Split { widths, .. } => widths.get(index).copied(),
        }
    }

    fn max_width(self) -> u8 {
        (0..self.len())
            .map(|index| self.width(index).unwrap_or_default())
            .max()
            .unwrap_or_default()
    }
}

pub(crate) struct RangePackedDecoder<'a> {
    len: usize,
    block_len: usize,
    symbol_width: u8,
    bins: BinTableView<'a>,
    block_offsets: &'a [u32],
    payload: &'a [u8],
}

impl<'a> RangePackedDecoder<'a> {
    pub(crate) fn try_new(
        len: usize,
        block_len: usize,
        symbol_width: u8,
        bins: BinTableView<'a>,
        block_offsets: &'a [u32],
        payload: &'a [u8],
    ) -> VortexResult<Self> {
        vortex_ensure!(
            block_len > 0 && block_len <= MAX_BLOCK_LEN,
            "range packed block length is invalid"
        );
        let expected_bins = if len == 0 {
            0
        } else {
            vortex_ensure!(
                symbol_width <= 6,
                "range packed symbol width exceeds six bits"
            );
            1_usize << symbol_width
        };
        vortex_ensure!(
            bins.len() == expected_bins,
            "range packed bin count is invalid"
        );
        if let BinTableView::Split { lowers, widths } = bins {
            vortex_ensure!(
                lowers.len() == widths.len(),
                "range packed split bin table lengths differ"
            );
        }
        vortex_ensure!(
            block_offsets.len() == len.div_ceil(block_len) + 1,
            "range packed block offset count is invalid"
        );
        vortex_ensure!(
            block_offsets.first() == Some(&0)
                && block_offsets
                    .last()
                    .copied()
                    .map(usize::try_from)
                    .transpose()?
                    == Some(payload.len()),
            "range packed payload boundaries are invalid"
        );
        Ok(Self {
            len,
            block_len,
            symbol_width,
            bins,
            block_offsets,
            payload,
        })
    }

    fn decode_mapped<T, F>(&self, map: F) -> VortexResult<Vec<T>>
    where
        F: FnMut(u64) -> T,
    {
        self.decode_mapped_range(0..self.len, map)
    }

    pub(crate) fn decode_mapped_range<T, F>(
        &self,
        range: Range<usize>,
        map: F,
    ) -> VortexResult<Vec<T>>
    where
        F: FnMut(u64) -> T,
    {
        vortex_ensure!(
            range.start <= range.end && range.end <= self.len,
            "range packed decode range exceeds array length"
        );
        let max_offset_bits = self.bins.max_width();
        if max_offset_bits <= 40 {
            self.decode_mapped_range_with_width::<false, false, T, F>(range, map)
        } else if max_offset_bits <= 57 {
            self.decode_mapped_range_with_width::<false, true, T, F>(range, map)
        } else {
            self.decode_mapped_range_with_width::<true, true, T, F>(range, map)
        }
    }

    fn decode_mapped_range_with_width<const WIDE: bool, const PARALLEL: bool, T, F>(
        &self,
        range: Range<usize>,
        mut map: F,
    ) -> VortexResult<Vec<T>>
    where
        F: FnMut(u64) -> T,
    {
        let mut values = Vec::with_capacity(range.len());
        if range.is_empty() {
            return Ok(values);
        }
        let table = DecodeTable::new(self.bins);
        let first_block = range.start / self.block_len;
        let last_block = range.end.div_ceil(self.block_len);
        for block_index in first_block..last_block {
            let block_start = block_index * self.block_len;
            let value_count = self.block_value_count(block_index);
            let local_start = range.start.saturating_sub(block_start);
            let local_stop = (range.end - block_start).min(value_count);
            let payload = self.block_payload(block_index)?;
            if local_start == 0 && local_stop == value_count {
                let output_start = values.len();
                decode_block::<WIDE, PARALLEL, T, F>(
                    payload,
                    value_count,
                    self.symbol_width,
                    &table,
                    &mut values.spare_capacity_mut()[..value_count],
                    &mut map,
                )?;
                // SAFETY: decode_block initialized each output slot.
                unsafe { values.set_len(output_start + value_count) };
                continue;
            }

            let mut block_values = Vec::with_capacity(value_count);
            decode_block::<WIDE, PARALLEL, T, F>(
                payload,
                value_count,
                self.symbol_width,
                &table,
                &mut block_values.spare_capacity_mut()[..value_count],
                &mut map,
            )?;
            // SAFETY: decode_block initialized each output slot.
            unsafe { block_values.set_len(value_count) };
            values.extend(block_values.drain(local_start..local_stop));
        }
        Ok(values)
    }

    pub(crate) fn scalar_at(&self, index: usize) -> VortexResult<u64> {
        vortex_ensure!(index < self.len, "range packed index exceeds array length");
        let block_index = index / self.block_len;
        let index_in_block = index % self.block_len;
        let value_count = self.block_value_count(block_index);
        let payload = self.block_payload(block_index)?;
        let layout = BlockLayout::new(value_count, self.symbol_width, payload.len())?;
        let symbol = read_symbol(payload, index_in_block, self.symbol_width)?;
        let lower = self
            .bins
            .lower(symbol)
            .ok_or_else(|| vortex_error::vortex_err!("range packed symbol exceeds bin table"))?;
        let width = self
            .bins
            .width(symbol)
            .ok_or_else(|| vortex_error::vortex_err!("range packed symbol exceeds bin table"))?;
        let checkpoint_index = index_in_block / CHECKPOINT_INTERVAL;
        let checkpoint_start = layout.checkpoints_start + checkpoint_index * size_of::<u16>();
        let checkpoint_bytes = payload
            .get(checkpoint_start..checkpoint_start + size_of::<u16>())
            .ok_or_else(|| vortex_error::vortex_err!("range packed checkpoint is missing"))?;
        let checkpoint = u16::from_le_bytes([checkpoint_bytes[0], checkpoint_bytes[1]]);
        let mut offset_position = usize::from(checkpoint);
        let scan_start = checkpoint_index * CHECKPOINT_INTERVAL;
        for preceding in scan_start..index_in_block {
            let preceding_symbol = read_symbol(payload, preceding, self.symbol_width)?;
            offset_position += usize::from(self.bins.width(preceding_symbol).ok_or_else(|| {
                vortex_error::vortex_err!("range packed symbol exceeds bin table")
            })?);
        }
        let offset = read_bits_padded(&payload[layout.offsets_start..], offset_position, width);
        Ok(lower.wrapping_add(offset))
    }

    fn block_value_count(&self, block_index: usize) -> usize {
        (self.len - block_index * self.block_len).min(self.block_len)
    }

    fn block_payload(&self, block_index: usize) -> VortexResult<&'a [u8]> {
        let start =
            usize::try_from(*self.block_offsets.get(block_index).ok_or_else(|| {
                vortex_error::vortex_err!("range packed block offset is missing")
            })?)?;
        let stop =
            usize::try_from(*self.block_offsets.get(block_index + 1).ok_or_else(|| {
                vortex_error::vortex_err!("range packed block offset is missing")
            })?)?;
        vortex_ensure!(start <= stop, "range packed block offsets are not ordered");
        self.payload
            .get(start..stop)
            .ok_or_else(|| vortex_error::vortex_err!("range packed block exceeds its payload"))
    }
}

struct DecodeTable {
    lowers: [u64; MAX_BINS],
    widths: [u8; MAX_BINS],
    masks: [u64; MAX_BINS],
    bin_count: usize,
}

impl DecodeTable {
    fn new(bins: BinTableView<'_>) -> Self {
        let mut lowers = [0_u64; MAX_BINS];
        let mut widths = [0_u8; MAX_BINS];
        let mut masks = [0_u64; MAX_BINS];
        for index in 0..bins.len() {
            lowers[index] = bins
                .lower(index)
                .unwrap_or_else(|| unreachable!("validated range packed bin table"));
            widths[index] = bins
                .width(index)
                .unwrap_or_else(|| unreachable!("validated range packed bin table"));
            masks[index] = low_mask(widths[index]);
        }
        Self {
            lowers,
            widths,
            masks,
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
fn decode_block<const WIDE: bool, const PARALLEL: bool, T, F>(
    payload: &[u8],
    value_count: usize,
    symbol_width: u8,
    table: &DecodeTable,
    values: &mut [MaybeUninit<T>],
    map: &mut F,
) -> VortexResult<()>
where
    F: FnMut(u64) -> T,
{
    let layout = BlockLayout::new(value_count, symbol_width, payload.len())?;
    let offsets = &payload[layout.offsets_start..];
    let mut offset_position = 0usize;
    let full_len = value_count / 8 * 8;
    let symbol_mask = low_mask(symbol_width);
    for base in (0..full_len).step_by(8) {
        let symbol_position = base * usize::from(symbol_width);
        let packed = read_bits_padded(payload, symbol_position, symbol_width * 8);
        if PARALLEL {
            let mut widths = [0_u8; 8];
            let mut lowers = [0_u64; 8];
            let mut masks = [0_u64; 8];
            let mut positions = [0_usize; 8];
            for lane in 0..8 {
                let symbol =
                    ((packed >> (lane * usize::from(symbol_width))) & symbol_mask) as usize;
                debug_assert!(symbol < table.bin_count);
                // SAFETY: A power-of-two bin count makes every symbol code valid.
                widths[lane] = unsafe { *table.widths.get_unchecked(symbol) };
                // SAFETY: A power-of-two bin count makes every symbol code valid.
                lowers[lane] = unsafe { *table.lowers.get_unchecked(symbol) };
                // SAFETY: A power-of-two bin count makes every symbol code valid.
                masks[lane] = unsafe { *table.masks.get_unchecked(symbol) };
                positions[lane] = offset_position;
                offset_position += usize::from(widths[lane]);
            }
            for lane in 0..8 {
                let offset = if WIDE {
                    read_offset::<true>(offsets, positions[lane], widths[lane])
                } else {
                    read_offset_masked(offsets, positions[lane], masks[lane])
                };
                // SAFETY: This lane is within a complete eight-value output batch.
                unsafe {
                    values
                        .get_unchecked_mut(base + lane)
                        .write(map(lowers[lane].wrapping_add(offset)))
                };
            }
        } else {
            for lane in 0..8 {
                let symbol =
                    ((packed >> (lane * usize::from(symbol_width))) & symbol_mask) as usize;
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
                        .write(map(lower.wrapping_add(offset)))
                };
            }
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
                .write(map(lower.wrapping_add(offset)))
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
fn read_offset_masked(offsets: &[u8], bit_position: usize, mask: u64) -> u64 {
    let byte_position = bit_position / 8;
    let bits_past_byte = bit_position % 8;
    // SAFETY: Offset streams include fifteen readable padding bytes.
    let packed = unsafe { read_u64_unaligned(offsets.as_ptr().add(byte_position)) };
    (packed >> bits_past_byte) & mask
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
    optimize_bins_with_count_policy(sorted, usize::is_power_of_two)
}

fn optimize_bins_arbitrary(sorted: &[u64]) -> Vec<Bin> {
    optimize_bins_with_count_policy(sorted, |_| true)
}

fn optimize_bins_with_count_policy(
    sorted: &[u64],
    allowed_count: impl Fn(usize) -> bool,
) -> Vec<Bin> {
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
        if allowed_count(segments.len()) && cost < best_cost {
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
    use rstest::rstest;
    use vortex_alp::ALP;
    use vortex_alp::ALPArrayExt;
    use vortex_alp::ALPArraySlotsExt;
    use vortex_alp::alp_encode;
    use vortex_array::ArrayContext;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::Primitive;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::half::f16;
    use vortex_array::serde::SerializeOptions;
    use vortex_array::serde::SerializedArray;
    use vortex_block_residual::OrderedFloat;
    use vortex_block_residual::OrderedFloatArraySlotsExt;
    use vortex_buffer::ByteBufferMut;
    use vortex_error::VortexResult;
    use vortex_session::registry::ReadContext;

    use super::RangeDecomposition;
    use super::RangePacked;
    use super::RangePackedCodec;
    use super::initialize;

    #[test]
    fn decomposition_uses_arbitrary_bin_count() -> VortexResult<()> {
        let values = (0_u64..10)
            .flat_map(|cluster| (0_u64..1_000).map(move |_| cluster * 1_000_000_000))
            .collect::<Vec<_>>();
        let decomposition = RangeDecomposition::encode(&values)?;
        assert_eq!(decomposition.bin_starts().len(), 10);
        assert_eq!(decomposition.code_width(), 4);
        assert_arrays_eq!(
            decomposition.into_array(vortex_array::validity::Validity::NonNullable)?,
            PrimitiveArray::from_iter(values),
            &mut array_session().create_execution_ctx()
        );
        Ok(())
    }

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

    #[rstest]
    #[case::f16(PrimitiveArray::from_option_iter([
        Some(f16::from_f32(-7.5)),
        None,
        Some(f16::from_f32(0.25)),
        Some(f16::INFINITY),
    ]))]
    #[case::f32(PrimitiveArray::from_option_iter([
        Some(-7.5_f32),
        None,
        Some(0.25),
        Some(f32::INFINITY),
    ]))]
    #[case::f64(PrimitiveArray::from_option_iter([
        Some(-7.5_f64),
        None,
        Some(0.25),
        Some(f64::INFINITY),
    ]))]
    fn ordered_float_parent_kernel(#[case] expected: PrimitiveArray) -> VortexResult<()> {
        let session = array_session();
        initialize(&session);
        let mut ctx = session.create_execution_ctx();
        let ordered = OrderedFloat::from_primitive(expected.as_view())?;
        let packed = RangePacked::from_primitive_with_null_positions(
            ordered.encoded().as_::<Primitive>(),
            &mut ctx,
        )?;
        let encoded = OrderedFloat::try_new(packed.into_array(), expected.ptype())?;
        assert_arrays_eq!(encoded, expected, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case::f32(PrimitiveArray::from_option_iter([
        Some(1.25_f32),
        None,
        Some(17.5),
        Some(f32::MAX),
    ]))]
    #[case::f64(PrimitiveArray::from_option_iter([
        Some(1.25_f64),
        None,
        Some(17.5),
        Some(f64::MAX),
    ]))]
    fn alp_parent_kernel(#[case] expected: PrimitiveArray) -> VortexResult<()> {
        let session = array_session();
        initialize(&session);
        let mut ctx = session.create_execution_ctx();
        let alp = alp_encode(expected.as_view(), None, &mut ctx)?;
        let packed = RangePacked::from_primitive_with_null_positions(
            alp.encoded().as_::<Primitive>(),
            &mut ctx,
        )?;
        let encoded = ALP::try_new(packed.into_array(), alp.exponents(), alp.patches())?;
        assert_arrays_eq!(encoded, expected, &mut ctx);
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

    #[rstest]
    #[case::dense(false)]
    #[case::full(true)]
    fn nullable_roundtrip_and_rank_boundaries(
        #[case] preserve_null_positions: bool,
    ) -> VortexResult<()> {
        let expected = PrimitiveArray::from_option_iter((0_i64..600).map(|value| {
            (!matches!(value, 0 | 1 | 63 | 64 | 255 | 256 | 257 | 511)).then_some(value - 300)
        }));
        let session = array_session();
        let mut ctx = session.create_execution_ctx();
        let encoded = if preserve_null_positions {
            RangePacked::from_primitive_with_null_positions(expected.as_view(), &mut ctx)?
        } else {
            RangePacked::from_primitive(expected.as_view(), &mut ctx)?
        };
        assert_arrays_eq!(encoded, expected, &mut ctx);
        for index in [0, 1, 2, 63, 64, 65, 255, 256, 257, 258, 511, 512, 599] {
            let actual = encoded.execute_scalar(index, &mut ctx)?;
            let expected_scalar = expected
                .clone()
                .into_array()
                .execute_scalar(index, &mut ctx)?;
            assert_eq!(actual, expected_scalar);
        }
        Ok(())
    }

    #[rstest]
    #[case::nonnullable(PrimitiveArray::from_iter((0_u64..4_096).map(|value| {
        (value % 8) * 1_000_000 + value.wrapping_mul(7_919) % 1_024
    })))]
    #[case::nullable(PrimitiveArray::from_option_iter((0_i64..4_096).map(|value| {
        (value % 17 != 0).then_some((value % 8) * 1_000_000 + value * 7_919 % 1_024)
    })))]
    fn full_position_estimate_matches_encoded_size(
        #[case] expected: PrimitiveArray,
    ) -> VortexResult<()> {
        let session = array_session();
        let mut ctx = session.create_execution_ctx();
        let estimate =
            RangePacked::estimate_primitive_with_null_positions(expected.as_view(), &mut ctx)?;
        let encoded =
            RangePacked::from_primitive_with_null_positions(expected.as_view(), &mut ctx)?;

        assert_eq!(estimate, encoded.nbytes());
        Ok(())
    }

    #[rstest]
    #[case::u8(PrimitiveArray::from_iter([0_u8, 1, u8::MAX]).into_array())]
    #[case::u16(PrimitiveArray::from_iter([0_u16, 1, u16::MAX]).into_array())]
    #[case::u32(PrimitiveArray::from_iter([0_u32, 1, u32::MAX]).into_array())]
    #[case::u64(PrimitiveArray::from_iter([0_u64, 1, u64::MAX]).into_array())]
    #[case::i8(PrimitiveArray::from_iter([i8::MIN, -1, 0, i8::MAX]).into_array())]
    #[case::i16(PrimitiveArray::from_iter([i16::MIN, -1, 0, i16::MAX]).into_array())]
    #[case::i32(PrimitiveArray::from_iter([i32::MIN, -1, 0, i32::MAX]).into_array())]
    #[case::i64(PrimitiveArray::from_iter([i64::MIN, -1, 0, i64::MAX]).into_array())]
    fn integer_ptype_roundtrip(#[case] expected: ArrayRef) -> VortexResult<()> {
        let session = array_session();
        let mut ctx = session.create_execution_ctx();
        let encoded = RangePacked::from_primitive(expected.as_::<Primitive>(), &mut ctx)?;
        assert_arrays_eq!(encoded, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn nullable_slice_crosses_rank_and_value_blocks() -> VortexResult<()> {
        let expected = PrimitiveArray::from_option_iter((0_i32..2_500).map(|value| {
            (!matches!(value, 255 | 256 | 257 | 1_023 | 1_024 | 1_025)).then_some(value - 1_250)
        }));
        let session = array_session();
        let mut ctx = session.create_execution_ctx();
        let encoded = RangePacked::from_primitive(expected.as_view(), &mut ctx)?
            .into_array()
            .slice(253..1_030)?;
        let expected = expected.into_array().slice(253..1_030)?;
        assert_arrays_eq!(encoded, expected, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case::dense(false)]
    #[case::full(true)]
    fn serialized_nullable_slice_roundtrip(
        #[case] preserve_null_positions: bool,
    ) -> VortexResult<()> {
        let expected = PrimitiveArray::from_option_iter(
            (0_i64..1_500).map(|value| (value % 17 != 0).then_some(value - 750)),
        );
        let session = array_session();
        initialize(&session);
        let mut ctx = session.create_execution_ctx();
        let encoded = if preserve_null_positions {
            RangePacked::from_primitive_with_null_positions(expected.as_view(), &mut ctx)?
        } else {
            RangePacked::from_primitive(expected.as_view(), &mut ctx)?
        }
        .into_array()
        .slice(251..1_101)?;
        let expected = expected.into_array().slice(251..1_101)?;
        let dtype = encoded.dtype().clone();
        let len = encoded.len();
        let array_ctx = ArrayContext::empty();
        let serialized = encoded.serialize(&array_ctx, &session, &SerializeOptions::default())?;
        let mut bytes = ByteBufferMut::empty();
        for buffer in serialized {
            bytes.extend_from_slice(buffer.as_ref());
        }
        let parts = SerializedArray::try_from(bytes.freeze())?;
        let decoded = parts.decode(&dtype, len, &ReadContext::new(array_ctx.to_ids()), &session)?;
        assert_arrays_eq!(decoded, expected, &mut ctx);
        Ok(())
    }
}
