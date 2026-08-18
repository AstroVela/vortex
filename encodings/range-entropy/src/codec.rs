// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Ordering;
use std::cmp::Reverse;
use std::ops::Deref;
use std::ops::DerefMut;
use std::ops::Range;

use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

const MAX_SCALE_BITS: u8 = 10;
const ANS_INTERLEAVING: usize = 4;
const ANS_STATE_BYTES: usize = ANS_INTERLEAVING * size_of::<u16>();
const ANS_PADDING: usize = size_of::<u64>();
const DECODE_BATCH_SIZE: usize = 256;
const OFFSET_PADDING: usize = 15;
const MAX_BINS: usize = 64;
const SPLIT_CANDIDATES: usize = 64;
const TRAINING_SAMPLE_SIZE: usize = 8192;
const BIN_TABLE_COST_BITS: f64 = 128.0;

#[repr(align(64))]
struct DecodeScratch<T>([T; DECODE_BATCH_SIZE]);

impl<T> Deref for DecodeScratch<T> {
    type Target = [T; DECODE_BATCH_SIZE];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for DecodeScratch<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// A restart-block range entropy stream for ordered unsigned latents.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RangeEntropyCodec {
    len: usize,
    block_len: usize,
    scale_bits: u8,
    bin_lowers: Vec<u64>,
    offset_widths: Vec<u8>,
    weights: Vec<u16>,
    block_offsets: Vec<u32>,
    payload: ByteBuffer,
}

/// Owned components of a range entropy stream.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RangeEntropyParts {
    /// Logical value count.
    pub len: usize,
    /// Logical values per restart block.
    pub block_len: usize,
    /// Base-two logarithm of the ANS table size.
    pub scale_bits: u8,
    /// Lower bound for each range bin.
    pub bin_lowers: Vec<u64>,
    /// Fixed offset width for each range bin.
    pub offset_widths: Vec<u8>,
    /// Quantized ANS weight for each range bin.
    pub weights: Vec<u16>,
    /// Byte offsets for restart blocks.
    pub block_offsets: Vec<u32>,
    /// Encoded ANS and offset streams.
    pub payload: ByteBuffer,
}

/// A restart-block range codec with fixed-width bin identifiers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RangePackedCodec {
    len: usize,
    block_len: usize,
    symbol_width: u8,
    bin_lowers: Vec<u64>,
    offset_widths: Vec<u8>,
    block_offsets: Vec<u32>,
    payload: ByteBuffer,
}

/// A restart-block range codec with hot-bin tags and a cold-bin escape stream.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RangeTwoLevelCodec {
    len: usize,
    block_len: usize,
    tag_width: u8,
    cold_width: u8,
    bin_lowers: Vec<u64>,
    offset_widths: Vec<u8>,
    block_offsets: Vec<u32>,
    payload: ByteBuffer,
}

/// A restart-block range codec with one fixed-width residual stream per bin.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RangeGroupedCodec {
    len: usize,
    block_len: usize,
    symbol_width: u8,
    bin_lowers: Vec<u64>,
    offset_widths: Vec<u8>,
    block_offsets: Vec<u32>,
    payload: ByteBuffer,
}

impl RangeEntropyCodec {
    /// Encode ordered unsigned latents into independent restart blocks.
    pub fn encode(values: &[u64], block_len: usize) -> VortexResult<Self> {
        vortex_ensure!(block_len > 0, "block length must be greater than zero");

        if values.is_empty() {
            return Ok(Self {
                len: 0,
                block_len,
                scale_bits: 0,
                bin_lowers: Vec::new(),
                offset_widths: Vec::new(),
                weights: Vec::new(),
                block_offsets: vec![0],
                payload: ByteBuffer::empty(),
            });
        }

        let (training_sample, minimum, maximum) = training_sample(values);
        let bins = cover_domain(optimize_bins(&training_sample), minimum, maximum);
        let (bins, symbols, counts) = assign_bins(values, bins)?;
        let bin_lowers: Vec<_> = bins.iter().map(|bin| bin.lower).collect();
        let offset_widths: Vec<_> = bins.iter().map(|bin| bin.offset_bits).collect();

        let scale_bits = choose_scale_bits(values.len(), bins.len());
        let weights = quantize_weights(&counts, scale_bits)?;
        let table = AnsEncoderTable::new(&weights, scale_bits)?;

        let n_blocks = values.len().div_ceil(block_len);
        let mut block_offsets = Vec::with_capacity(n_blocks + 1);
        let mut payload = Vec::new();
        block_offsets.push(0);

        for block_idx in 0..n_blocks {
            let start = block_idx * block_len;
            let stop = (start + block_len).min(values.len());
            encode_block(
                &values[start..stop],
                &symbols[start..stop],
                &bins,
                &table,
                &mut payload,
            )?;
            block_offsets.push(u32::try_from(payload.len())?);
        }

        let codec = Self {
            len: values.len(),
            block_len,
            scale_bits,
            bin_lowers,
            offset_widths,
            weights,
            block_offsets,
            payload: ByteBuffer::from(payload),
        };
        codec.validate()?;
        Ok(codec)
    }

    /// Construct a codec from stored components.
    pub fn try_from_parts(parts: RangeEntropyParts) -> VortexResult<Self> {
        let codec = Self {
            len: parts.len,
            block_len: parts.block_len,
            scale_bits: parts.scale_bits,
            bin_lowers: parts.bin_lowers,
            offset_widths: parts.offset_widths,
            weights: parts.weights,
            block_offsets: parts.block_offsets,
            payload: parts.payload,
        };
        codec.validate()?;
        Ok(codec)
    }

    /// Consume the codec and return its stored components.
    pub fn into_parts(self) -> RangeEntropyParts {
        RangeEntropyParts {
            len: self.len,
            block_len: self.block_len,
            scale_bits: self.scale_bits,
            bin_lowers: self.bin_lowers,
            offset_widths: self.offset_widths,
            weights: self.weights,
            block_offsets: self.block_offsets,
            payload: self.payload,
        }
    }

    /// Decode all values.
    pub fn decode(&self) -> VortexResult<Vec<u64>> {
        self.decode_range(0..self.len)
    }

    /// Decode one logical range through the intersecting restart blocks.
    pub fn decode_range(&self, range: Range<usize>) -> VortexResult<Vec<u64>> {
        self.validate()?;
        vortex_ensure!(
            range.start <= range.end && range.end <= self.len,
            "decode range {:?} exceeds length {}",
            range,
            self.len
        );
        if range.is_empty() {
            return Ok(Vec::new());
        }

        let first_block = range.start / self.block_len;
        let last_block = (range.end - 1) / self.block_len;
        let bins = self.bins()?;
        let table = AnsDecoderTable::new(&self.weights, self.scale_bits, &bins)?;
        let mut values = Vec::with_capacity(range.len());
        for block_idx in first_block..=last_block {
            let block_start = block_idx * self.block_len;
            let start = range.start.saturating_sub(block_start);
            let stop = (range.end - block_start).min(self.block_value_count(block_idx));
            self.decode_block_into(block_idx, start..stop, &table, &mut values)?;
        }
        Ok(values)
    }

    /// Decode one restart block.
    pub fn decode_block(&self, block_idx: usize) -> VortexResult<Vec<u64>> {
        self.validate()?;
        vortex_ensure!(
            block_idx < self.n_blocks(),
            "block index {block_idx} is out of bounds for {} blocks",
            self.n_blocks()
        );

        let bins = self.bins()?;
        let table = AnsDecoderTable::new(&self.weights, self.scale_bits, &bins)?;
        let n_values = self.block_value_count(block_idx);
        let mut values = Vec::with_capacity(n_values);
        self.decode_block_into(block_idx, 0..n_values, &table, &mut values)?;
        Ok(values)
    }

    fn decode_block_into(
        &self,
        block_idx: usize,
        output_range: Range<usize>,
        table: &AnsDecoderTable,
        values: &mut Vec<u64>,
    ) -> VortexResult<()> {
        let start = usize::try_from(self.block_offsets[block_idx])?;
        let stop = usize::try_from(self.block_offsets[block_idx + 1])?;
        let n_values = self.block_value_count(block_idx);
        decode_block_into(
            &self.payload.as_slice()[start..stop],
            n_values,
            output_range,
            table,
            values,
        )
    }

    /// Return one value after one restart-block decode.
    pub fn value(&self, index: usize) -> VortexResult<u64> {
        vortex_ensure!(
            index < self.len,
            "value index {index} is out of bounds for length {}",
            self.len
        );
        let block_idx = index / self.block_len;
        let within_block = index % self.block_len;
        Ok(self.decode_block(block_idx)?[within_block])
    }

    /// Return the logical value count.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Return true when the stream has no values.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Return the restart block length.
    pub fn block_len(&self) -> usize {
        self.block_len
    }

    /// Return the ANS table size log.
    pub fn scale_bits(&self) -> u8 {
        self.scale_bits
    }

    /// Return the bin lower bounds.
    pub fn bin_lowers(&self) -> &[u64] {
        &self.bin_lowers
    }

    /// Return the fixed offset width for each bin.
    pub fn offset_widths(&self) -> &[u8] {
        &self.offset_widths
    }

    /// Return the quantized ANS weight for each bin.
    pub fn weights(&self) -> &[u16] {
        &self.weights
    }

    /// Return the byte offsets for restart blocks.
    pub fn block_offsets(&self) -> &[u32] {
        &self.block_offsets
    }

    /// Return the encoded payload.
    pub fn payload(&self) -> &[u8] {
        self.payload.as_slice()
    }

    /// Return the total bytes in the variable tables and payload.
    pub fn encoded_size(&self) -> usize {
        self.bin_lowers.len() * size_of::<u64>()
            + self.offset_widths.len() * size_of::<u8>()
            + self.weights.len() * size_of::<u16>()
            + self.block_offsets.len() * size_of::<u32>()
            + self.payload.len()
    }

    fn validate(&self) -> VortexResult<()> {
        vortex_ensure!(self.block_len > 0, "block length must be greater than zero");
        vortex_ensure!(
            self.scale_bits <= MAX_SCALE_BITS,
            "ANS table log {} exceeds {MAX_SCALE_BITS}",
            self.scale_bits
        );
        vortex_ensure!(
            self.bin_lowers.len() == self.offset_widths.len()
                && self.bin_lowers.len() == self.weights.len(),
            "range entropy bin tables have different lengths"
        );

        if self.len == 0 {
            vortex_ensure!(self.bin_lowers.is_empty(), "empty streams cannot have bins");
            vortex_ensure!(
                self.payload.is_empty(),
                "empty streams cannot have payload bytes"
            );
            vortex_ensure!(
                self.block_offsets == [0],
                "empty streams require one zero block offset"
            );
            return Ok(());
        }

        vortex_ensure!(
            !self.bin_lowers.is_empty(),
            "non-empty streams require bins"
        );
        vortex_ensure!(
            self.bin_lowers.len() <= MAX_BINS,
            "bin count {} exceeds maximum {MAX_BINS}",
            self.bin_lowers.len()
        );
        vortex_ensure!(
            self.offset_widths.iter().all(|&width| width <= 64),
            "offset width exceeds 64 bits"
        );
        vortex_ensure!(
            self.weights.iter().all(|&weight| weight > 0),
            "ANS weights must be positive"
        );
        vortex_ensure!(
            self.weights
                .iter()
                .map(|&weight| u32::from(weight))
                .sum::<u32>()
                == 1_u32 << self.scale_bits,
            "ANS weights do not sum to the table size"
        );
        vortex_ensure!(
            self.block_offsets.len() == self.n_blocks() + 1,
            "expected {} block offsets, got {}",
            self.n_blocks() + 1,
            self.block_offsets.len()
        );
        vortex_ensure!(
            self.block_offsets[0] == 0,
            "first block offset must be zero"
        );
        vortex_ensure!(
            self.block_offsets.windows(2).all(|pair| pair[0] <= pair[1]),
            "block offsets must be monotonic"
        );
        vortex_ensure!(
            usize::try_from(*self.block_offsets.last().unwrap_or(&0))? == self.payload.len(),
            "last block offset does not match payload length"
        );
        Ok(())
    }

    fn bins(&self) -> VortexResult<Vec<Bin>> {
        self.bin_lowers
            .iter()
            .copied()
            .zip(self.offset_widths.iter().copied())
            .map(|(lower, offset_bits)| Bin::try_new(lower, offset_bits))
            .collect()
    }

    fn n_blocks(&self) -> usize {
        self.len.div_ceil(self.block_len)
    }

    fn block_value_count(&self, block_idx: usize) -> usize {
        let start = block_idx * self.block_len;
        (self.len - start).min(self.block_len)
    }
}

impl RangePackedCodec {
    /// Encode ordered unsigned latents with fixed-width bin identifiers.
    pub fn encode(values: &[u64], block_len: usize) -> VortexResult<Self> {
        vortex_ensure!(block_len > 0, "block length must be greater than zero");

        if values.is_empty() {
            return Ok(Self {
                len: 0,
                block_len,
                symbol_width: 0,
                bin_lowers: Vec::new(),
                offset_widths: Vec::new(),
                block_offsets: vec![0],
                payload: ByteBuffer::empty(),
            });
        }

        let (training_sample, minimum, maximum) = training_sample(values);
        let bins = cover_domain(optimize_bins(&training_sample), minimum, maximum);
        let (bins, symbols, _) = assign_bins(values, bins)?;
        let symbol_width = bit_width(u64::try_from(bins.len() - 1)?);
        let bin_lowers = bins.iter().map(|bin| bin.lower).collect();
        let offset_widths = bins.iter().map(|bin| bin.offset_bits).collect();

        let n_blocks = values.len().div_ceil(block_len);
        let mut block_offsets = Vec::with_capacity(n_blocks + 1);
        let mut payload = Vec::new();
        block_offsets.push(0);
        for block_idx in 0..n_blocks {
            let start = block_idx * block_len;
            let stop = (start + block_len).min(values.len());
            encode_packed_block(
                &values[start..stop],
                &symbols[start..stop],
                &bins,
                symbol_width,
                &mut payload,
            );
            block_offsets.push(u32::try_from(payload.len())?);
        }

        let codec = Self {
            len: values.len(),
            block_len,
            symbol_width,
            bin_lowers,
            offset_widths,
            block_offsets,
            payload: ByteBuffer::from(payload),
        };
        codec.validate()?;
        Ok(codec)
    }

    /// Decode all values.
    pub fn decode(&self) -> VortexResult<Vec<u64>> {
        self.validate()?;
        let bins = self.bins()?;
        let mut values = Vec::with_capacity(self.len);
        for block_idx in 0..self.n_blocks() {
            let start = usize::try_from(self.block_offsets[block_idx])?;
            let stop = usize::try_from(self.block_offsets[block_idx + 1])?;
            decode_packed_block_into(
                &self.payload.as_slice()[start..stop],
                self.block_value_count(block_idx),
                self.symbol_width,
                &bins,
                &mut values,
            )?;
        }
        Ok(values)
    }

    /// Return the logical value count.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Return true when the stream has no values.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Return the fixed bin identifier width.
    pub fn symbol_width(&self) -> u8 {
        self.symbol_width
    }

    /// Return the bin lower bounds.
    pub fn bin_lowers(&self) -> &[u64] {
        &self.bin_lowers
    }

    /// Return the total bytes in the variable tables and payload.
    pub fn encoded_size(&self) -> usize {
        self.bin_lowers.len() * size_of::<u64>()
            + self.offset_widths.len() * size_of::<u8>()
            + self.block_offsets.len() * size_of::<u32>()
            + self.payload.len()
    }

    fn validate(&self) -> VortexResult<()> {
        vortex_ensure!(self.block_len > 0, "block length must be greater than zero");
        vortex_ensure!(
            self.bin_lowers.len() == self.offset_widths.len(),
            "range packed bin tables have different lengths"
        );

        if self.len == 0 {
            vortex_ensure!(self.bin_lowers.is_empty(), "empty streams cannot have bins");
            vortex_ensure!(
                self.payload.is_empty(),
                "empty streams cannot have payload bytes"
            );
            vortex_ensure!(
                self.block_offsets == [0],
                "empty streams require one zero block offset"
            );
            return Ok(());
        }

        vortex_ensure!(
            !self.bin_lowers.is_empty() && self.bin_lowers.len() <= MAX_BINS,
            "range packed bin count is invalid"
        );
        vortex_ensure!(
            self.symbol_width == bit_width(u64::try_from(self.bin_lowers.len() - 1)?),
            "range packed symbol width does not match its bin count"
        );
        vortex_ensure!(
            self.offset_widths.iter().all(|&width| width <= 64),
            "offset width exceeds 64 bits"
        );
        vortex_ensure!(
            self.block_offsets.len() == self.n_blocks() + 1,
            "range packed block offset count is invalid"
        );
        vortex_ensure!(
            self.block_offsets.first() == Some(&0)
                && self.block_offsets.windows(2).all(|pair| pair[0] <= pair[1]),
            "range packed block offsets are invalid"
        );
        vortex_ensure!(
            usize::try_from(*self.block_offsets.last().unwrap_or(&0))? == self.payload.len(),
            "last block offset does not match payload length"
        );
        Ok(())
    }

    fn bins(&self) -> VortexResult<Vec<Bin>> {
        self.bin_lowers
            .iter()
            .copied()
            .zip(self.offset_widths.iter().copied())
            .map(|(lower, offset_bits)| Bin::try_new(lower, offset_bits))
            .collect()
    }

    fn n_blocks(&self) -> usize {
        self.len.div_ceil(self.block_len)
    }

    fn block_value_count(&self, block_idx: usize) -> usize {
        let start = block_idx * self.block_len;
        (self.len - start).min(self.block_len)
    }
}

impl RangeTwoLevelCodec {
    /// Encode ordered unsigned latents with hot-bin tags and cold-bin escapes.
    pub fn encode(values: &[u64], block_len: usize) -> VortexResult<Self> {
        vortex_ensure!(block_len > 0, "block length must be greater than zero");
        if values.is_empty() {
            return Ok(Self {
                len: 0,
                block_len,
                tag_width: 0,
                cold_width: 0,
                bin_lowers: Vec::new(),
                offset_widths: Vec::new(),
                block_offsets: vec![0],
                payload: ByteBuffer::empty(),
            });
        }

        let (training_sample, minimum, maximum) = training_sample(values);
        let bins = cover_domain(optimize_bins(&training_sample), minimum, maximum);
        let (bins, symbols, counts) = assign_bins(values, bins)?;
        let (tag_width, direct_count, order) = choose_two_level_layout(&counts);
        let cold_count = bins.len() - direct_count;
        let cold_width = bit_width(u64::try_from(cold_count.saturating_sub(1))?);
        let mut rank_by_symbol = vec![0usize; bins.len()];
        for (rank, &symbol) in order.iter().enumerate() {
            rank_by_symbol[symbol] = rank;
        }
        let reordered_bins = order.iter().map(|&symbol| bins[symbol]).collect::<Vec<_>>();
        let ranks = symbols
            .into_iter()
            .map(|symbol| rank_by_symbol[symbol])
            .collect::<Vec<_>>();

        let n_blocks = values.len().div_ceil(block_len);
        let mut block_offsets = Vec::with_capacity(n_blocks + 1);
        let mut payload = Vec::new();
        block_offsets.push(0);
        for block_idx in 0..n_blocks {
            let start = block_idx * block_len;
            let stop = (start + block_len).min(values.len());
            encode_two_level_block(
                &values[start..stop],
                &ranks[start..stop],
                &reordered_bins,
                tag_width,
                direct_count,
                cold_width,
                &mut payload,
            )?;
            block_offsets.push(u32::try_from(payload.len())?);
        }

        Ok(Self {
            len: values.len(),
            block_len,
            tag_width,
            cold_width,
            bin_lowers: reordered_bins.iter().map(|bin| bin.lower).collect(),
            offset_widths: reordered_bins.iter().map(|bin| bin.offset_bits).collect(),
            block_offsets,
            payload: ByteBuffer::from(payload),
        })
    }

    /// Decode all values.
    pub fn decode(&self) -> VortexResult<Vec<u64>> {
        if self.len == 0 {
            return Ok(Vec::new());
        }
        let bins = self
            .bin_lowers
            .iter()
            .copied()
            .zip(self.offset_widths.iter().copied())
            .map(|(lower, offset_bits)| Bin::try_new(lower, offset_bits))
            .collect::<VortexResult<Vec<_>>>()?;
        let direct_count = ((1usize << self.tag_width) - 1).min(bins.len());
        let mut values = Vec::with_capacity(self.len);
        for block_idx in 0..self.len.div_ceil(self.block_len) {
            let start = usize::try_from(self.block_offsets[block_idx])?;
            let stop = usize::try_from(self.block_offsets[block_idx + 1])?;
            let block_start = block_idx * self.block_len;
            let block_value_count = (self.len - block_start).min(self.block_len);
            decode_two_level_block_into(
                &self.payload.as_slice()[start..stop],
                block_value_count,
                self.tag_width,
                direct_count,
                self.cold_width,
                &bins,
                &mut values,
            )?;
        }
        Ok(values)
    }

    /// Return the fixed hot-bin tag width.
    pub fn tag_width(&self) -> u8 {
        self.tag_width
    }

    /// Return the fixed cold-bin identifier width.
    pub fn cold_width(&self) -> u8 {
        self.cold_width
    }

    /// Return the total bytes in the variable tables and payload.
    pub fn encoded_size(&self) -> usize {
        self.bin_lowers.len() * size_of::<u64>()
            + self.offset_widths.len() * size_of::<u8>()
            + self.block_offsets.len() * size_of::<u32>()
            + self.payload.len()
    }
}

impl RangeGroupedCodec {
    /// Encode ordered unsigned latents with one residual stream per bin.
    pub fn encode(values: &[u64], block_len: usize) -> VortexResult<Self> {
        vortex_ensure!(block_len > 0, "block length must be greater than zero");
        if values.is_empty() {
            return Ok(Self {
                len: 0,
                block_len,
                symbol_width: 0,
                bin_lowers: Vec::new(),
                offset_widths: Vec::new(),
                block_offsets: vec![0],
                payload: ByteBuffer::empty(),
            });
        }

        let (training_sample, minimum, maximum) = training_sample(values);
        let bins = cover_domain(optimize_bins(&training_sample), minimum, maximum);
        let (bins, symbols, _) = assign_bins(values, bins)?;
        let symbol_width = bit_width(u64::try_from(bins.len() - 1)?);
        let n_blocks = values.len().div_ceil(block_len);
        let mut block_offsets = Vec::with_capacity(n_blocks + 1);
        let mut payload = Vec::new();
        block_offsets.push(0);
        for block_idx in 0..n_blocks {
            let start = block_idx * block_len;
            let stop = (start + block_len).min(values.len());
            encode_grouped_block(
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
            bin_lowers: bins.iter().map(|bin| bin.lower).collect(),
            offset_widths: bins.iter().map(|bin| bin.offset_bits).collect(),
            block_offsets,
            payload: ByteBuffer::from(payload),
        })
    }

    /// Decode all values.
    pub fn decode(&self) -> VortexResult<Vec<u64>> {
        if self.len == 0 {
            return Ok(Vec::new());
        }
        let bins = self
            .bin_lowers
            .iter()
            .copied()
            .zip(self.offset_widths.iter().copied())
            .map(|(lower, offset_bits)| Bin::try_new(lower, offset_bits))
            .collect::<VortexResult<Vec<_>>>()?;
        let mut values = Vec::with_capacity(self.len);
        for block_idx in 0..self.len.div_ceil(self.block_len) {
            let start = usize::try_from(self.block_offsets[block_idx])?;
            let stop = usize::try_from(self.block_offsets[block_idx + 1])?;
            let block_start = block_idx * self.block_len;
            let block_value_count = (self.len - block_start).min(self.block_len);
            decode_grouped_block_into(
                &self.payload.as_slice()[start..stop],
                block_value_count,
                self.symbol_width,
                &bins,
                &mut values,
            )?;
        }
        Ok(values)
    }

    /// Return the total bytes in the variable tables and payload.
    pub fn encoded_size(&self) -> usize {
        self.bin_lowers.len() * size_of::<u64>()
            + self.offset_widths.len() * size_of::<u8>()
            + self.block_offsets.len() * size_of::<u32>()
            + self.payload.len()
    }
}

#[derive(Clone, Copy, Debug)]
struct Bin {
    lower: u64,
    offset_bits: u8,
}

impl Bin {
    fn try_new(lower: u64, offset_bits: u8) -> VortexResult<Self> {
        vortex_ensure!(offset_bits <= 64, "offset width exceeds 64 bits");
        Ok(Self { lower, offset_bits })
    }

    fn from_values(lower: u64, upper: u64) -> Self {
        let offset_bits = bit_width(upper - lower);
        Self { lower, offset_bits }
    }

    fn contains(&self, value: u64) -> bool {
        if value < self.lower {
            return false;
        }
        self.offset_bits == 64 || value - self.lower <= low_mask(self.offset_bits)
    }
}

fn bit_width(value: u64) -> u8 {
    u8::try_from(u64::BITS - value.leading_zeros()).unwrap_or(64)
}

fn low_mask(bits: u8) -> u64 {
    if bits == 64 {
        u64::MAX
    } else if bits == 0 {
        0
    } else {
        (1_u64 << bits) - 1
    }
}

fn find_bin(value: u64, bins: &[Bin], search_order: &[usize]) -> VortexResult<usize> {
    search_order
        .iter()
        .copied()
        .find(|&index| bins[index].contains(value))
        .ok_or_else(|| vortex_error::vortex_err!("no range entropy bin contains value {value}"))
}

fn assign_bins(
    values: &[u64],
    mut bins: Vec<Bin>,
) -> VortexResult<(Vec<Bin>, Vec<usize>, Vec<usize>)> {
    loop {
        let mut search_order: Vec<_> = (0..bins.len()).collect();
        search_order.sort_by_key(|&index| bins[index].offset_bits);
        let mut counts = vec![0usize; bins.len()];
        let mut symbols = Vec::with_capacity(values.len());
        for &value in values {
            let symbol = find_bin(value, &bins, &search_order)?;
            counts[symbol] += 1;
            symbols.push(symbol);
        }
        if counts.iter().all(|&count| count > 0) {
            return Ok((bins, symbols, counts));
        }
        bins = bins
            .into_iter()
            .zip(counts)
            .filter_map(|(bin, count)| (count > 0).then_some(bin))
            .collect();
    }
}

fn choose_two_level_layout(counts: &[usize]) -> (u8, usize, Vec<usize>) {
    let mut order = (0..counts.len()).collect::<Vec<_>>();
    order.sort_unstable_by_key(|&symbol| Reverse(counts[symbol]));
    if counts.len() <= 1 {
        return (0, counts.len(), order);
    }

    let flat_width = bit_width((counts.len() - 1) as u64);
    let mut best = (usize::MAX, flat_width, counts.len());
    let minimum_tag_width = flat_width.min(2);
    for tag_width in minimum_tag_width..=flat_width {
        let direct_count = ((1usize << tag_width) - 1).min(counts.len());
        let cold_count = counts.len() - direct_count;
        let cold_width = bit_width(cold_count.saturating_sub(1) as u64);
        let cold_values = order[direct_count..]
            .iter()
            .map(|&symbol| counts[symbol])
            .sum::<usize>();
        let cost = counts.iter().sum::<usize>() * usize::from(tag_width)
            + cold_values * usize::from(cold_width);
        if cost < best.0 {
            best = (cost, tag_width, direct_count);
        }
    }
    (best.1, best.2, order)
}

fn optimize_bins(sorted: &[u64]) -> Vec<Bin> {
    debug_assert!(!sorted.is_empty());
    let mut segments = Vec::with_capacity(MAX_BINS);
    segments.push(0..sorted.len());

    while segments.len() < MAX_BINS {
        let best = segments
            .iter()
            .enumerate()
            .filter_map(|(segment_idx, segment)| {
                best_split(sorted, segment).map(|split| (segment_idx, split))
            })
            .max_by(|(_, left), (_, right)| {
                left.gain
                    .partial_cmp(&right.gain)
                    .unwrap_or(Ordering::Equal)
            });

        let Some((segment_idx, split)) = best else {
            break;
        };
        if split.gain <= BIN_TABLE_COST_BITS {
            break;
        }

        let segment = segments.remove(segment_idx);
        segments.push(segment.start..split.at);
        segments.push(split.at..segment.end);
        segments.sort_unstable_by_key(|range| range.start);
    }

    segments
        .into_iter()
        .map(|range| Bin::from_values(sorted[range.start], sorted[range.end - 1]))
        .collect()
}

fn training_sample(values: &[u64]) -> (Vec<u64>, u64, u64) {
    let mut minimum = u64::MAX;
    let mut maximum = u64::MIN;
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
    if let Some(first) = bins.first_mut() {
        first.lower = minimum;
    }
    for index in 0..bins.len().saturating_sub(1) {
        let upper = bins[index + 1].lower - 1;
        bins[index].offset_bits = bit_width(upper - bins[index].lower);
    }
    if let Some(last) = bins.last_mut() {
        last.offset_bits = bit_width(maximum - last.lower);
    }
    bins
}

#[derive(Clone, Copy)]
struct Split {
    at: usize,
    gain: f64,
}

fn best_split(sorted: &[u64], segment: &Range<usize>) -> Option<Split> {
    let len = segment.len();
    if len < 2 || sorted[segment.start] == sorted[segment.end - 1] {
        return None;
    }

    let whole_width = f64::from(bit_width(sorted[segment.end - 1] - sorted[segment.start]));
    let whole_cost = len as f64 * whole_width;
    let mut best: Option<Split> = None;

    for candidate in 1..SPLIT_CANDIDATES {
        let mut at = segment.start + len * candidate / SPLIT_CANDIDATES;
        if at <= segment.start || at >= segment.end {
            continue;
        }
        while at < segment.end && sorted[at - 1] == sorted[at] {
            at += 1;
        }
        if at >= segment.end {
            continue;
        }

        let left_len = at - segment.start;
        let right_len = segment.end - at;
        let left_width = f64::from(bit_width(sorted[at - 1] - sorted[segment.start]));
        let right_width = f64::from(bit_width(sorted[segment.end - 1] - sorted[at]));
        let symbol_cost = entropy_split_cost(left_len, right_len);
        let split_cost =
            left_len as f64 * left_width + right_len as f64 * right_width + symbol_cost;
        let gain = whole_cost - split_cost;

        if best.is_none_or(|current| gain > current.gain) {
            best = Some(Split { at, gain });
        }
    }
    best
}

fn entropy_split_cost(left: usize, right: usize) -> f64 {
    let total = (left + right) as f64;
    let left = left as f64;
    let right = right as f64;
    left * (total / left).log2() + right * (total / right).log2()
}

fn choose_scale_bits(value_count: usize, bin_count: usize) -> u8 {
    if bin_count <= 1 {
        return 0;
    }

    let required_bits = usize::BITS - (bin_count - 1).leading_zeros();
    let useful_bits = usize::BITS - (value_count - 1).leading_zeros();
    u8::try_from(required_bits.max(useful_bits.min(u32::from(MAX_SCALE_BITS))))
        .unwrap_or(MAX_SCALE_BITS)
}

fn quantize_weights(counts: &[usize], scale_bits: u8) -> VortexResult<Vec<u16>> {
    let total_weight = 1_u32 << scale_bits;
    let used: Vec<_> = counts
        .iter()
        .enumerate()
        .filter(|(_, count)| **count > 0)
        .map(|(index, &count)| (index, count))
        .collect();
    vortex_ensure!(!used.is_empty(), "cannot quantize an empty histogram");
    vortex_ensure!(
        used.len() <= total_weight as usize,
        "symbol count exceeds the ANS table size"
    );

    let total_count: usize = used.iter().map(|(_, count)| count).sum();
    let distributable = usize::try_from(total_weight)? - used.len();
    let mut weights = vec![0u16; counts.len()];
    let mut remainders = Vec::with_capacity(used.len());
    let mut allocated = 0usize;

    for &(index, count) in &used {
        let scaled = count * distributable;
        let extra = scaled / total_count;
        let remainder = scaled % total_count;
        weights[index] = u16::try_from(1 + extra)?;
        allocated += 1 + extra;
        remainders.push((index, remainder));
    }

    remainders.sort_unstable_by_key(|entry| Reverse(entry.1));
    for &(index, _) in remainders
        .iter()
        .take(usize::try_from(total_weight)? - allocated)
    {
        weights[index] += 1;
    }
    Ok(weights)
}

struct AnsEncoderSymbol {
    renorm_bit_cutoff: u32,
    min_renorm_bits: u8,
    next_states: Vec<u32>,
}

#[derive(Clone, Copy)]
struct AnsDecoderNode {
    next_state_idx_base: u16,
    offset_bits: u8,
    bits_to_read: u8,
}

struct AnsEncoderTable {
    table_size: u32,
    encoder_symbols: Vec<AnsEncoderSymbol>,
}

struct AnsDecoderTable {
    table_size: u32,
    decoder_nodes: Vec<AnsDecoderNode>,
    lowers: Vec<u64>,
}

fn spread_symbols(weights: &[u16], scale_bits: u8) -> VortexResult<Vec<usize>> {
    let table_size = 1_u32 << scale_bits;
    let mut state_symbols = vec![usize::MAX; usize::try_from(table_size)?];
    let mut stride = 3 * table_size / 5;
    if stride.is_multiple_of(2) {
        stride += 1;
    }
    let mask = table_size - 1;
    let mut step = 0u32;
    for (symbol, &weight) in weights.iter().enumerate() {
        for _ in 0..weight {
            let state_idx = (stride * step) & mask;
            state_symbols[usize::try_from(state_idx)?] = symbol;
            step += 1;
        }
    }
    vortex_ensure!(
        state_symbols.iter().all(|&symbol| symbol != usize::MAX),
        "ANS table has unassigned states"
    );
    Ok(state_symbols)
}

fn validate_weights(weights: &[u16], scale_bits: u8) -> VortexResult<u32> {
    let table_size = 1_u32 << scale_bits;
    vortex_ensure!(
        weights.iter().map(|&weight| u32::from(weight)).sum::<u32>() == table_size,
        "ANS weights do not sum to the table size"
    );
    vortex_ensure!(
        weights.iter().all(|&weight| weight > 0),
        "ANS weights must be positive"
    );
    Ok(table_size)
}

impl AnsEncoderTable {
    fn new(weights: &[u16], scale_bits: u8) -> VortexResult<Self> {
        let table_size = 1_u32 << scale_bits;
        validate_weights(weights, scale_bits)?;
        let state_symbols = spread_symbols(weights, scale_bits)?;

        let mut encoder_symbols = Vec::with_capacity(weights.len());
        for &weight in weights {
            let frequency = u32::from(weight);
            let max_x_s = 2 * frequency - 1;
            let min_renorm_bits = scale_bits - u8::try_from(max_x_s.ilog2())?;
            encoder_symbols.push(AnsEncoderSymbol {
                renorm_bit_cutoff: 2 * frequency * (1_u32 << min_renorm_bits),
                min_renorm_bits,
                next_states: Vec::with_capacity(usize::from(weight)),
            });
        }
        for (state_idx, &symbol) in state_symbols.iter().enumerate() {
            encoder_symbols[symbol]
                .next_states
                .push(table_size + u32::try_from(state_idx)?);
        }

        Ok(Self {
            table_size,
            encoder_symbols,
        })
    }
}

impl AnsDecoderTable {
    fn new(weights: &[u16], scale_bits: u8, bins: &[Bin]) -> VortexResult<Self> {
        let table_size = validate_weights(weights, scale_bits)?;
        vortex_ensure!(
            weights.len() == bins.len(),
            "ANS weights and range bins have different lengths"
        );
        let state_symbols = spread_symbols(weights, scale_bits)?;
        let mut symbol_x_s: Vec<_> = weights.iter().map(|&weight| u32::from(weight)).collect();
        let mut decoder_nodes = Vec::with_capacity(usize::try_from(table_size)?);
        let mut lowers = Vec::with_capacity(usize::try_from(table_size)?);
        for symbol in state_symbols {
            let next_state_base = symbol_x_s[symbol];
            let bits_to_read =
                u8::try_from(next_state_base.leading_zeros() - table_size.leading_zeros())?;
            let next_state_idx_base = (next_state_base << bits_to_read) - table_size;
            let bin = bins[symbol];
            decoder_nodes.push(AnsDecoderNode {
                next_state_idx_base: u16::try_from(next_state_idx_base)?,
                offset_bits: bin.offset_bits,
                bits_to_read,
            });
            lowers.push(bin.lower);
            symbol_x_s[symbol] += 1;
        }

        Ok(Self {
            table_size,
            decoder_nodes,
            lowers,
        })
    }
}

fn encode_block(
    values: &[u64],
    symbols: &[usize],
    bins: &[Bin],
    table: &AnsEncoderTable,
    payload: &mut Vec<u8>,
) -> VortexResult<()> {
    let ans = encode_symbols(symbols, table)?;
    payload.extend_from_slice(&u32::try_from(ans.len())?.to_le_bytes());
    payload.extend_from_slice(&ans);

    let mut offsets = BitWriter::with_capacity(size_of_val(values));
    for (&value, &symbol) in values.iter().zip(symbols) {
        let bin = &bins[symbol];
        offsets.write(value - bin.lower, bin.offset_bits);
    }
    payload.extend_from_slice(&offsets.finish());
    payload.extend_from_slice(&[0; OFFSET_PADDING]);
    Ok(())
}

fn encode_packed_block(
    values: &[u64],
    symbols: &[usize],
    bins: &[Bin],
    symbol_width: u8,
    payload: &mut Vec<u8>,
) {
    let mut encoded_symbols = BitWriter::with_capacity(symbols.len());
    for &symbol in symbols {
        encoded_symbols.write(symbol as u64, symbol_width);
    }
    payload.extend_from_slice(&encoded_symbols.finish());
    payload.extend_from_slice(&[0; ANS_PADDING]);

    let mut offsets = BitWriter::with_capacity(size_of_val(values));
    for (&value, &symbol) in values.iter().zip(symbols) {
        let bin = &bins[symbol];
        offsets.write(value - bin.lower, bin.offset_bits);
    }
    payload.extend_from_slice(&offsets.finish());
    payload.extend_from_slice(&[0; OFFSET_PADDING]);
}

fn encode_two_level_block(
    values: &[u64],
    ranks: &[usize],
    bins: &[Bin],
    tag_width: u8,
    direct_count: usize,
    cold_width: u8,
    payload: &mut Vec<u8>,
) -> VortexResult<()> {
    let escape_tag = direct_count;
    let cold_value_count = ranks.iter().filter(|&&rank| rank >= direct_count).count();
    payload.extend_from_slice(&u32::try_from(cold_value_count)?.to_le_bytes());

    let mut tags = BitWriter::with_capacity(ranks.len());
    let mut cold = BitWriter::with_capacity(cold_value_count);
    for &rank in ranks {
        if rank < direct_count {
            tags.write(rank as u64, tag_width);
        } else {
            tags.write(escape_tag as u64, tag_width);
            cold.write((rank - direct_count) as u64, cold_width);
        }
    }
    payload.extend_from_slice(&tags.finish());
    payload.extend_from_slice(&[0; ANS_PADDING]);
    payload.extend_from_slice(&cold.finish());
    payload.extend_from_slice(&[0; ANS_PADDING]);

    let mut offsets = BitWriter::with_capacity(size_of_val(values));
    for (&value, &rank) in values.iter().zip(ranks) {
        let bin = &bins[rank];
        offsets.write(value - bin.lower, bin.offset_bits);
    }
    payload.extend_from_slice(&offsets.finish());
    payload.extend_from_slice(&[0; OFFSET_PADDING]);
    Ok(())
}

fn encode_grouped_block(
    values: &[u64],
    symbols: &[usize],
    bins: &[Bin],
    symbol_width: u8,
    payload: &mut Vec<u8>,
) -> VortexResult<()> {
    let mut encoded_symbols = BitWriter::with_capacity(symbols.len());
    let mut streams = (0..bins.len())
        .map(|_| BitWriter::with_capacity(values.len() / bins.len().max(1)))
        .collect::<Vec<_>>();
    for (&value, &symbol) in values.iter().zip(symbols) {
        let bin = &bins[symbol];
        encoded_symbols.write(symbol as u64, symbol_width);
        streams[symbol].write(value - bin.lower, bin.offset_bits);
    }
    payload.extend_from_slice(&encoded_symbols.finish());
    payload.extend_from_slice(&[0; ANS_PADDING]);

    let mut stream_offsets = Vec::with_capacity(bins.len() + 1);
    let mut encoded_streams = Vec::new();
    stream_offsets.push(0u32);
    for stream in streams {
        encoded_streams.extend_from_slice(&stream.finish());
        encoded_streams.extend_from_slice(&[0; OFFSET_PADDING]);
        stream_offsets.push(u32::try_from(encoded_streams.len())?);
    }
    for offset in stream_offsets {
        payload.extend_from_slice(&offset.to_le_bytes());
    }
    payload.extend_from_slice(&encoded_streams);
    Ok(())
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "bin identifiers and stream positions fit their destination types"
)]
fn decode_grouped_block_into(
    payload: &[u8],
    n_values: usize,
    symbol_width: u8,
    bins: &[Bin],
    values: &mut Vec<u64>,
) -> VortexResult<()> {
    let symbol_bytes = (n_values * usize::from(symbol_width)).div_ceil(8);
    let offset_table_start = symbol_bytes + ANS_PADDING;
    let offset_table_bytes = (bins.len() + 1) * size_of::<u32>();
    let streams_start = offset_table_start + offset_table_bytes;
    vortex_ensure!(
        payload.len() >= streams_start,
        "grouped restart block is too short"
    );
    let mut stream_starts = [0usize; MAX_BINS];
    let mut stream_stops = [0usize; MAX_BINS];
    for index in 0..bins.len() {
        let read_offset = |entry: usize| {
            let start = offset_table_start + entry * size_of::<u32>();
            u32::from_le_bytes([
                payload[start],
                payload[start + 1],
                payload[start + 2],
                payload[start + 3],
            ])
        };
        stream_starts[index] = streams_start + usize::try_from(read_offset(index))?;
        stream_stops[index] = streams_start + usize::try_from(read_offset(index + 1))?;
        vortex_ensure!(
            stream_starts[index] <= stream_stops[index]
                && stream_stops[index] <= payload.len()
                && stream_stops[index] - stream_starts[index] >= OFFSET_PADDING,
            "grouped residual stream is invalid"
        );
    }

    let mut symbols = BitReader::new(payload, symbol_bytes)?;
    let mut table = [PackedDecodeNode::default(); MAX_BINS];
    for (index, bin) in bins.iter().enumerate() {
        table[index] = PackedDecodeNode {
            lower: bin.lower,
            offset_bits: u32::from(bin.offset_bits),
        };
    }
    let mut stream_positions = [0u32; MAX_BINS];
    let symbol_mask = low_mask(symbol_width);
    let mut seen_symbols = 0u64;

    for _ in (0..n_values / 8 * 8).step_by(8) {
        // SAFETY: The padded symbol stream provides eight readable bytes.
        let packed = unsafe { symbols.read_unchecked(symbol_width * 8) };
        for lane in 0..8 {
            let symbol = ((packed >> (lane * usize::from(symbol_width))) & symbol_mask) as usize;
            seen_symbols |= 1_u64 << symbol;
            let node = table[symbol];
            let bit_position = stream_positions[symbol];
            let byte_position = stream_starts[symbol] + bit_position as usize / 8;
            let bits_past_byte = bit_position % 8;
            // SAFETY: Each stream contains fifteen padding bytes.
            let first = unsafe { read_u64_unaligned(payload.as_ptr().add(byte_position)) };
            let offset = if node.offset_bits <= 57 {
                (first >> bits_past_byte) & ((1_u64 << node.offset_bits) - 1)
            } else {
                // SAFETY: Each stream contains fifteen padding bytes.
                let second = unsafe { read_u64_unaligned(payload.as_ptr().add(byte_position + 7)) };
                let processed = 56 - bits_past_byte;
                ((first >> bits_past_byte) | (second << processed))
                    & low_mask(node.offset_bits as u8)
            };
            stream_positions[symbol] += node.offset_bits;
            values.push(node.lower.wrapping_add(offset));
        }
    }
    for _ in n_values / 8 * 8..n_values {
        let symbol = symbols.read(symbol_width)? as usize;
        seen_symbols |= 1_u64 << symbol;
        let node = table[symbol];
        let bit_position = stream_positions[symbol];
        let byte_position = stream_starts[symbol] + bit_position as usize / 8;
        let bits_past_byte = bit_position % 8;
        // SAFETY: Each stream contains fifteen padding bytes.
        let first = unsafe { read_u64_unaligned(payload.as_ptr().add(byte_position)) };
        let offset = if node.offset_bits <= 57 {
            (first >> bits_past_byte) & ((1_u64 << node.offset_bits) - 1)
        } else {
            // SAFETY: Each stream contains fifteen padding bytes.
            let second = unsafe { read_u64_unaligned(payload.as_ptr().add(byte_position + 7)) };
            let processed = 56 - bits_past_byte;
            ((first >> bits_past_byte) | (second << processed)) & low_mask(node.offset_bits as u8)
        };
        stream_positions[symbol] += node.offset_bits;
        values.push(node.lower.wrapping_add(offset));
    }

    let valid_symbols = low_mask(u8::try_from(bins.len())?);
    vortex_ensure!(
        seen_symbols & !valid_symbols == 0,
        "grouped symbol exceeds bin table"
    );
    symbols.finish()?;
    for index in 0..bins.len() {
        let used_bytes = usize::try_from(stream_positions[index])?.div_ceil(8);
        vortex_ensure!(
            stream_starts[index] + used_bytes + OFFSET_PADDING == stream_stops[index],
            "grouped residual stream has unused bytes"
        );
    }
    Ok(())
}

#[derive(Clone, Copy, Default)]
struct PackedDecodeNode {
    lower: u64,
    offset_bits: u32,
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "bin identifiers and block offset positions fit their destination types"
)]
fn decode_two_level_block_into(
    payload: &[u8],
    n_values: usize,
    tag_width: u8,
    direct_count: usize,
    cold_width: u8,
    bins: &[Bin],
    values: &mut Vec<u64>,
) -> VortexResult<()> {
    vortex_ensure!(payload.len() >= 4, "two-level restart block is too short");
    let cold_value_count = usize::try_from(u32::from_le_bytes([
        payload[0], payload[1], payload[2], payload[3],
    ]))?;
    let tag_bytes = (n_values * usize::from(tag_width)).div_ceil(8);
    let cold_bytes = (cold_value_count * usize::from(cold_width)).div_ceil(8);
    let tags_start = 4;
    let cold_start = tags_start + tag_bytes + ANS_PADDING;
    let offsets_start = cold_start + cold_bytes + ANS_PADDING;
    vortex_ensure!(
        payload.len() >= offsets_start + OFFSET_PADDING,
        "two-level restart block is too short"
    );
    let mut tags = BitReader::new(&payload[tags_start..], tag_bytes)?;
    let mut cold = BitReader::new(&payload[cold_start..], cold_bytes)?;
    let offsets = &payload[offsets_start..];
    let mut table = [PackedDecodeNode::default(); MAX_BINS];
    for (index, bin) in bins.iter().enumerate() {
        table[index] = PackedDecodeNode {
            lower: bin.lower,
            offset_bits: u32::from(bin.offset_bits),
        };
    }

    let escape_tag = direct_count;
    let tag_mask = low_mask(tag_width);
    let mut offset_bit_position = 0u32;
    let full_len = n_values / 8 * 8;
    for _ in (0..full_len).step_by(8) {
        // SAFETY: The encoded codec invariant provides eight tag padding bytes.
        let packed = unsafe { tags.read_unchecked(tag_width * 8) };
        for lane in 0..8 {
            let tag = ((packed >> (lane * usize::from(tag_width))) & tag_mask) as usize;
            let rank = if tag == escape_tag {
                // SAFETY: The encoded codec invariant provides eight cold padding bytes.
                direct_count + unsafe { cold.read_unchecked(cold_width) } as usize
            } else {
                tag
            };
            let node = table[rank];
            // SAFETY: The private codec fields come from the validated encoder.
            let offset = unsafe { read_packed_offset(offsets, offset_bit_position, node) };
            offset_bit_position += node.offset_bits;
            values.push(node.lower.wrapping_add(offset));
        }
    }
    for _ in full_len..n_values {
        let tag = tags.read(tag_width)? as usize;
        let rank = if tag == escape_tag {
            direct_count + cold.read(cold_width)? as usize
        } else {
            tag
        };
        let node = table[rank];
        // SAFETY: The private codec fields come from the validated encoder.
        let offset = unsafe { read_packed_offset(offsets, offset_bit_position, node) };
        offset_bit_position += node.offset_bits;
        values.push(node.lower.wrapping_add(offset));
    }

    tags.finish()?;
    cold.finish()?;
    let offset_bytes = offset_bit_position.div_ceil(8) as usize;
    vortex_ensure!(
        offsets.len() == offset_bytes + OFFSET_PADDING,
        "offset stream has unused bytes"
    );
    Ok(())
}

#[inline(always)]
#[expect(
    clippy::cast_possible_truncation,
    reason = "validated offset widths do not exceed 64 bits"
)]
unsafe fn read_packed_offset(offsets: &[u8], bit_position: u32, node: PackedDecodeNode) -> u64 {
    let byte_position = bit_position as usize / 8;
    let bits_past_byte = bit_position % 8;
    // SAFETY: The encoded codec invariant provides fifteen readable padding bytes.
    let first = unsafe { read_u64_unaligned(offsets.as_ptr().add(byte_position)) };
    if node.offset_bits <= 57 {
        (first >> bits_past_byte) & ((1_u64 << node.offset_bits) - 1)
    } else {
        // SAFETY: The encoded codec invariant provides fifteen readable padding bytes.
        let second = unsafe { read_u64_unaligned(offsets.as_ptr().add(byte_position + 7)) };
        let processed = 56 - bits_past_byte;
        ((first >> bits_past_byte) | (second << processed)) & low_mask(node.offset_bits as u8)
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "bin identifiers and block offset positions fit their destination types"
)]
fn decode_packed_block_into(
    payload: &[u8],
    n_values: usize,
    symbol_width: u8,
    bins: &[Bin],
    values: &mut Vec<u64>,
) -> VortexResult<()> {
    let symbol_bits = n_values * usize::from(symbol_width);
    let symbol_bytes = symbol_bits.div_ceil(8);
    vortex_ensure!(
        payload.len() >= symbol_bytes + ANS_PADDING + OFFSET_PADDING,
        "range packed restart block is too short"
    );
    let offsets = &payload[symbol_bytes + ANS_PADDING..];
    let mut symbols = BitReader::new(payload, symbol_bytes)?;
    let mut table = [PackedDecodeNode::default(); MAX_BINS];
    for (index, bin) in bins.iter().enumerate() {
        table[index] = PackedDecodeNode {
            lower: bin.lower,
            offset_bits: u32::from(bin.offset_bits),
        };
    }

    let symbol_mask = low_mask(symbol_width);
    let mut offset_bit_position = 0u32;
    let full_len = n_values / 8 * 8;
    for _ in (0..full_len).step_by(8) {
        // SAFETY: The encoded codec invariant provides eight symbol padding bytes.
        let packed = unsafe { symbols.read_unchecked(symbol_width * 8) };
        for lane in 0..8 {
            let symbol = ((packed >> (lane * usize::from(symbol_width))) & symbol_mask) as usize;
            let node = table[symbol];
            // SAFETY: The private codec fields come from the validated encoder.
            let offset = unsafe { read_packed_offset(offsets, offset_bit_position, node) };
            offset_bit_position += node.offset_bits;
            values.push(node.lower.wrapping_add(offset));
        }
    }
    for _ in full_len..n_values {
        let symbol = symbols.read(symbol_width)? as usize;
        let node = table[symbol];
        // SAFETY: The private codec fields come from the validated encoder.
        let offset = unsafe { read_packed_offset(offsets, offset_bit_position, node) };
        offset_bit_position += node.offset_bits;
        values.push(node.lower.wrapping_add(offset));
    }

    symbols.finish()?;
    let offset_bytes = offset_bit_position.div_ceil(8) as usize;
    vortex_ensure!(
        offsets.len() == offset_bytes + OFFSET_PADDING,
        "offset stream has unused bytes"
    );
    vortex_ensure!(
        offsets[offset_bytes..].iter().all(|&byte| byte == 0),
        "offset stream has nonzero padding"
    );
    Ok(())
}

#[inline(never)]
#[expect(
    clippy::cast_possible_truncation,
    reason = "ANS transitions read at most MAX_SCALE_BITS bits"
)]
unsafe fn read_full_ans_batch(
    symbols: &mut BitReader<'_>,
    table: &AnsDecoderTable,
    states: &mut [u32; ANS_INTERLEAVING],
    lowers: &mut DecodeScratch<u64>,
    widths: &mut DecodeScratch<u32>,
    bit_positions: &mut DecodeScratch<u32>,
) -> u32 {
    let [mut state_0, mut state_1, mut state_2, mut state_3] = *states;
    let mut offset_bit_position = 0u32;

    for base_index in (0..DECODE_BATCH_SIZE).step_by(ANS_INTERLEAVING) {
        // SAFETY: Initial states are validated, and each ANS transition stays in the table.
        let node_0 = unsafe { table.decoder_nodes.get_unchecked(state_0 as usize) };
        // SAFETY: Initial states are validated, and each ANS transition stays in the table.
        let node_1 = unsafe { table.decoder_nodes.get_unchecked(state_1 as usize) };
        // SAFETY: Initial states are validated, and each ANS transition stays in the table.
        let node_2 = unsafe { table.decoder_nodes.get_unchecked(state_2 as usize) };
        // SAFETY: Initial states are validated, and each ANS transition stays in the table.
        let node_3 = unsafe { table.decoder_nodes.get_unchecked(state_3 as usize) };
        let width_0 = node_0.bits_to_read;
        let width_1 = node_1.bits_to_read;
        let width_2 = node_2.bits_to_read;
        let width_3 = node_3.bits_to_read;
        let packed_width = width_0 + width_1 + width_2 + width_3;
        // SAFETY: The caller verifies enough readable bytes for the worst-case batch.
        let packed = unsafe { symbols.read_unchecked(packed_width) };
        let shift_1 = width_0;
        let shift_2 = shift_1 + width_1;
        let shift_3 = shift_2 + width_2;

        macro_rules! write_symbol {
            ($index:expr, $state:ident, $node:ident, $shift:expr, $width:ident) => {{
                // SAFETY: The decoder node uses the same validated state index.
                let lower = unsafe { *table.lowers.get_unchecked($state as usize) };
                // SAFETY: Every scratch index is within this full batch.
                unsafe { *lowers.get_unchecked_mut($index) = lower };
                // SAFETY: Every scratch index is within this full batch.
                unsafe { *bit_positions.get_unchecked_mut($index) = offset_bit_position };
                // SAFETY: Every scratch index is within this full batch.
                unsafe { *widths.get_unchecked_mut($index) = u32::from($node.offset_bits) };
                offset_bit_position += u32::from($node.offset_bits);
                $state = u32::from($node.next_state_idx_base)
                    + ((packed >> $shift) & ((1_u64 << $width) - 1)) as u32;
            }};
        }

        write_symbol!(base_index, state_0, node_0, 0, width_0);
        write_symbol!(base_index + 1, state_1, node_1, shift_1, width_1);
        write_symbol!(base_index + 2, state_2, node_2, shift_2, width_2);
        write_symbol!(base_index + 3, state_3, node_3, shift_3, width_3);
    }

    *states = [state_0, state_1, state_2, state_3];
    offset_bit_position
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "validated ANS and offset widths fit their destination types"
)]
fn decode_block_into(
    payload: &[u8],
    n_values: usize,
    output_range: Range<usize>,
    table: &AnsDecoderTable,
    values: &mut Vec<u64>,
) -> VortexResult<()> {
    vortex_ensure!(
        output_range.start <= output_range.end && output_range.end <= n_values,
        "output range exceeds restart block"
    );
    vortex_ensure!(
        payload.len() >= 4 + ANS_STATE_BYTES,
        "restart block is too short"
    );
    let ans_len = usize::try_from(u32::from_le_bytes([
        payload[0], payload[1], payload[2], payload[3],
    ]))?;
    vortex_ensure!(
        ans_len >= ANS_STATE_BYTES + ANS_PADDING,
        "ANS stream is too short"
    );
    vortex_ensure!(
        4 + ans_len <= payload.len(),
        "ANS stream exceeds restart block"
    );

    let encoded_symbols = &payload[4..4 + ans_len];
    let mut state_idxs = [0u32; ANS_INTERLEAVING];
    for (index, state_idx) in state_idxs.iter_mut().enumerate() {
        let byte_idx = index * size_of::<u16>();
        *state_idx = u32::from(u16::from_le_bytes([
            encoded_symbols[byte_idx],
            encoded_symbols[byte_idx + 1],
        ]));
    }
    vortex_ensure!(
        state_idxs
            .iter()
            .all(|&state_idx| state_idx < table.table_size),
        "ANS state index exceeds the table"
    );
    let offsets = &payload[4 + ans_len..];
    let symbol_bytes = &payload[4 + ANS_STATE_BYTES..];
    let symbol_unpadded_len = ans_len - ANS_STATE_BYTES - ANS_PADDING;
    let mut symbols = BitReader::new(symbol_bytes, symbol_unpadded_len)?;
    let [mut state_0, mut state_1, mut state_2, mut state_3] = state_idxs;
    let mut offset_bit_position = 0u32;
    let mut lowers = DecodeScratch([0u64; DECODE_BATCH_SIZE]);
    let mut widths = DecodeScratch([0u32; DECODE_BATCH_SIZE]);
    let mut bit_positions = DecodeScratch([0u32; DECODE_BATCH_SIZE]);
    let mut decoded = DecodeScratch([0u64; DECODE_BATCH_SIZE]);

    for batch_start in (0..n_values).step_by(DECODE_BATCH_SIZE) {
        let batch_len = (n_values - batch_start).min(DECODE_BATCH_SIZE);
        let full_batch_len = batch_len / ANS_INTERLEAVING * ANS_INTERLEAVING;
        let batch_offset_start = offset_bit_position;
        let mut batch_offset_bits = 0u32;
        let fast_full_batch = batch_len == DECODE_BATCH_SIZE
            && symbols.can_read_unchecked(DECODE_BATCH_SIZE * usize::from(MAX_SCALE_BITS));
        if fast_full_batch {
            let mut states = [state_0, state_1, state_2, state_3];
            // SAFETY: The size check provides readable bytes for the worst-case batch.
            batch_offset_bits = unsafe {
                read_full_ans_batch(
                    &mut symbols,
                    table,
                    &mut states,
                    &mut lowers,
                    &mut widths,
                    &mut bit_positions,
                )
            };
            [state_0, state_1, state_2, state_3] = states;
            symbols.check_in_bounds()?;
        } else {
            for base_index in (0..full_batch_len).step_by(ANS_INTERLEAVING) {
                // SAFETY: Initial states are validated, and each ANS transition stays in the table.
                let node_0 = unsafe { table.decoder_nodes.get_unchecked(state_0 as usize) };
                // SAFETY: Initial states are validated, and each ANS transition stays in the table.
                let node_1 = unsafe { table.decoder_nodes.get_unchecked(state_1 as usize) };
                // SAFETY: Initial states are validated, and each ANS transition stays in the table.
                let node_2 = unsafe { table.decoder_nodes.get_unchecked(state_2 as usize) };
                // SAFETY: Initial states are validated, and each ANS transition stays in the table.
                let node_3 = unsafe { table.decoder_nodes.get_unchecked(state_3 as usize) };
                let width_0 = node_0.bits_to_read;
                let width_1 = node_1.bits_to_read;
                let width_2 = node_2.bits_to_read;
                let width_3 = node_3.bits_to_read;
                let packed_width = width_0 + width_1 + width_2 + width_3;
                let packed = symbols.read(packed_width)?;
                let shift_1 = width_0;
                let shift_2 = shift_1 + width_1;
                let shift_3 = shift_2 + width_2;
                // SAFETY: The corresponding decoder nodes use the same validated state indexes.
                lowers[base_index] = unsafe { *table.lowers.get_unchecked(state_0 as usize) };
                // SAFETY: The corresponding decoder nodes use the same validated state indexes.
                lowers[base_index + 1] = unsafe { *table.lowers.get_unchecked(state_1 as usize) };
                // SAFETY: The corresponding decoder nodes use the same validated state indexes.
                lowers[base_index + 2] = unsafe { *table.lowers.get_unchecked(state_2 as usize) };
                // SAFETY: The corresponding decoder nodes use the same validated state indexes.
                lowers[base_index + 3] = unsafe { *table.lowers.get_unchecked(state_3 as usize) };
                for (index, width) in [
                    node_0.offset_bits,
                    node_1.offset_bits,
                    node_2.offset_bits,
                    node_3.offset_bits,
                ]
                .into_iter()
                .enumerate()
                {
                    bit_positions[base_index + index] = batch_offset_bits;
                    widths[base_index + index] = u32::from(width);
                    batch_offset_bits += u32::from(width);
                }
                state_0 = u32::from(node_0.next_state_idx_base)
                    + (packed & ((1_u64 << width_0) - 1)) as u32;
                state_1 = u32::from(node_1.next_state_idx_base)
                    + ((packed >> shift_1) & ((1_u64 << width_1) - 1)) as u32;
                state_2 = u32::from(node_2.next_state_idx_base)
                    + ((packed >> shift_2) & ((1_u64 << width_2) - 1)) as u32;
                state_3 = u32::from(node_3.next_state_idx_base)
                    + ((packed >> shift_3) & ((1_u64 << width_3) - 1)) as u32;
            }
        }

        state_idxs = [state_0, state_1, state_2, state_3];
        for index in full_batch_len..batch_len {
            let lane = index - full_batch_len;
            let state_idx = &mut state_idxs[lane];
            let current_state = *state_idx;
            // SAFETY: Initial states are validated, and each ANS transition stays in the table.
            let node = unsafe { table.decoder_nodes.get_unchecked(current_state as usize) };
            *state_idx =
                u32::from(node.next_state_idx_base) + symbols.read(node.bits_to_read)? as u32;
            // SAFETY: The corresponding decoder node was read from the same state index.
            lowers[index] = unsafe { *table.lowers.get_unchecked(current_state as usize) };
            bit_positions[index] = batch_offset_bits;
            widths[index] = u32::from(node.offset_bits);
            batch_offset_bits += u32::from(node.offset_bits);
        }
        [state_0, state_1, state_2, state_3] = state_idxs;

        let last_bit_position = batch_offset_start + bit_positions[batch_len - 1];
        let last_byte_position = last_bit_position as usize / 8;
        vortex_ensure!(
            last_byte_position + OFFSET_PADDING <= offsets.len(),
            "offset stream is truncated"
        );
        decoded[..batch_len].copy_from_slice(&lowers[..batch_len]);
        if widths[..batch_len].iter().all(|&width| width <= 57) {
            // SAFETY: The caller verified eight readable bytes after each start position.
            unsafe {
                read_offsets::<8>(
                    offsets,
                    batch_offset_start,
                    &bit_positions.0,
                    &widths.0,
                    &mut decoded.0,
                    batch_len,
                )
            }
        } else {
            // SAFETY: The caller verified fifteen readable bytes after each start position.
            unsafe {
                read_offsets::<15>(
                    offsets,
                    batch_offset_start,
                    &bit_positions.0,
                    &widths.0,
                    &mut decoded.0,
                    batch_len,
                )
            }
        }
        offset_bit_position += batch_offset_bits;

        let output_start = output_range.start.max(batch_start);
        let output_stop = output_range.end.min(batch_start + batch_len);
        if output_start < output_stop {
            values
                .extend_from_slice(&decoded[output_start - batch_start..output_stop - batch_start]);
        }
    }
    symbols.finish()?;
    let offset_bytes = offset_bit_position.div_ceil(8) as usize;
    vortex_ensure!(
        offsets.len() == offset_bytes + OFFSET_PADDING,
        "offset stream has unused bytes"
    );
    vortex_ensure!(
        offsets[offset_bytes..].iter().all(|&byte| byte == 0),
        "offset stream has nonzero padding"
    );
    if !offset_bit_position.is_multiple_of(8) {
        let used_bits = offset_bit_position % 8;
        let trailing_mask = !low_mask(u8::try_from(used_bits)?).to_le_bytes()[0];
        vortex_ensure!(
            offsets[offset_bytes - 1] & trailing_mask == 0,
            "offset stream has nonzero trailing bits"
        );
    }
    state_idxs = [state_0, state_1, state_2, state_3];
    vortex_ensure!(
        state_idxs.iter().all(|&state_idx| state_idx == 0),
        "ANS stream ended in an invalid state"
    );
    Ok(())
}

#[inline(never)]
#[expect(
    clippy::cast_possible_truncation,
    reason = "validated offset widths do not exceed 64 bits"
)]
unsafe fn read_offsets<const READ_BYTES: usize>(
    bytes: &[u8],
    base_bit_position: u32,
    bit_positions: &[u32],
    widths: &[u32],
    values: &mut [u64],
    len: usize,
) {
    for index in 0..len {
        let bit_position = base_bit_position + bit_positions[index];
        let byte_position = bit_position as usize / 8;
        let bits_past_byte = bit_position % 8;
        let first = unsafe { read_u64_unaligned(bytes.as_ptr().add(byte_position)) };
        let offset = match READ_BYTES {
            8 => {
                debug_assert!(widths[index] <= 57);
                (first >> bits_past_byte) & ((1_u64 << widths[index]) - 1)
            }
            15 => {
                let second = unsafe { read_u64_unaligned(bytes.as_ptr().add(byte_position + 7)) };
                let processed = 56 - bits_past_byte;
                ((first >> bits_past_byte) | (second << processed)) & low_mask(widths[index] as u8)
            }
            _ => unreachable!("invalid offset read width {READ_BYTES}"),
        };
        values[index] = values[index].wrapping_add(offset);
    }
}

type ReadOffsetsFn = unsafe fn(&[u8], u32, &[u32], &[u32], &mut [u64], usize);

#[used]
static FORCE_EXPORT_READ_OFFSETS_8: ReadOffsetsFn = read_offsets::<8>;

#[used]
static FORCE_EXPORT_READ_OFFSETS_15: ReadOffsetsFn = read_offsets::<15>;

unsafe fn read_u64_unaligned(pointer: *const u8) -> u64 {
    // SAFETY: The caller provides eight readable bytes at the pointer.
    u64::from_le(unsafe { pointer.cast::<u64>().read_unaligned() })
}

fn encode_symbols(symbols: &[usize], table: &AnsEncoderTable) -> VortexResult<Vec<u8>> {
    let mut states = [table.table_size; ANS_INTERLEAVING];
    let mut writes = vec![(0u32, 0u8); symbols.len()];
    let full_len = symbols.len() / ANS_INTERLEAVING * ANS_INTERLEAVING;

    let mut encode_one = |index: usize, lane: usize| -> VortexResult<()> {
        let symbol = symbols[index];
        let state = states[lane];
        let symbol_info = table
            .encoder_symbols
            .get(symbol)
            .ok_or_else(|| vortex_error::vortex_err!("symbol {symbol} exceeds ANS table"))?;
        let renorm_bits = if state >= symbol_info.renorm_bit_cutoff {
            symbol_info.min_renorm_bits + 1
        } else {
            symbol_info.min_renorm_bits
        };
        let frequency = u32::try_from(symbol_info.next_states.len())?;
        let x_s = state >> renorm_bits;
        let next_state_idx = usize::try_from(x_s - frequency)?;
        let next_state = *symbol_info
            .next_states
            .get(next_state_idx)
            .ok_or_else(|| vortex_error::vortex_err!("ANS state index exceeds the symbol table"))?;
        writes[index] = (state, renorm_bits);
        states[lane] = next_state;
        Ok(())
    };

    for lane in (0..symbols.len() - full_len).rev() {
        encode_one(full_len + lane, lane)?;
    }
    for base_index in (0..full_len).step_by(ANS_INTERLEAVING).rev() {
        for lane in (0..ANS_INTERLEAVING).rev() {
            encode_one(base_index + lane, lane)?;
        }
    }

    let mut encoded = Vec::with_capacity(ANS_STATE_BYTES + symbols.len() / 2);
    for state in states {
        encoded.extend_from_slice(&u16::try_from(state - table.table_size)?.to_le_bytes());
    }
    let mut writer = BitWriter::with_capacity(symbols.len() * 2);
    for (value, width) in writes {
        writer.write(u64::from(value) & low_mask(width), width);
    }
    encoded.extend_from_slice(&writer.finish());
    encoded.extend_from_slice(&[0; ANS_PADDING]);
    Ok(encoded)
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
        debug_assert!(width <= 64);
        debug_assert!(width == 64 || value <= low_mask(width));
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

struct BitReader<'a> {
    bytes: &'a [u8],
    unpadded_len: usize,
    bit_position: usize,
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8], unpadded_len: usize) -> VortexResult<Self> {
        vortex_ensure!(
            bytes.len() >= unpadded_len + ANS_PADDING,
            "ANS stream is too short"
        );
        Ok(Self {
            bytes,
            unpadded_len,
            bit_position: 0,
        })
    }

    fn can_read_unchecked(&self, max_bits: usize) -> bool {
        let byte_position = self.bit_position / 8;
        let bits_past_byte = self.bit_position % 8;
        let max_bytes = (bits_past_byte + max_bits).div_ceil(8) + size_of::<u64>();
        byte_position + max_bytes <= self.bytes.len()
    }

    fn check_in_bounds(&self) -> VortexResult<()> {
        vortex_ensure!(
            self.bit_position <= self.unpadded_len * 8,
            "ANS stream is truncated"
        );
        Ok(())
    }

    #[inline(always)]
    unsafe fn read_unchecked(&mut self, width: u8) -> u64 {
        let byte_position = self.bit_position / 8;
        let bits_past_byte = self.bit_position % 8;
        // SAFETY: The caller verifies eight readable bytes after the current position.
        let packed = unsafe { read_u64_unaligned(self.bytes.as_ptr().add(byte_position)) };
        self.bit_position += usize::from(width);
        (packed >> bits_past_byte) & ((1_u64 << width) - 1)
    }

    #[inline(always)]
    fn read(&mut self, width: u8) -> VortexResult<u64> {
        vortex_ensure!(width <= 57, "ANS read width exceeds 57 bits");
        let required = self.bit_position + usize::from(width);
        vortex_ensure!(required <= self.unpadded_len * 8, "ANS stream is truncated");
        // SAFETY: The bounds check and stream padding provide eight readable bytes.
        Ok(unsafe { self.read_unchecked(width) })
    }

    fn finish(&self) -> VortexResult<()> {
        let used_bytes = self.bit_position.div_ceil(8);
        vortex_ensure!(
            used_bytes == self.unpadded_len,
            "ANS stream has unused bytes"
        );
        vortex_ensure!(
            self.bytes[self.unpadded_len..self.unpadded_len + ANS_PADDING]
                .iter()
                .all(|&byte| byte == 0),
            "ANS stream has nonzero padding"
        );
        if !self.bit_position.is_multiple_of(8) {
            let used_bits = self.bit_position % 8;
            let trailing_mask = !low_mask(u8::try_from(used_bits)?).to_le_bytes()[0];
            vortex_ensure!(
                self.bytes[used_bytes - 1] & trailing_mask == 0,
                "ANS stream has nonzero trailing bits"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::ByteBuffer;
    use vortex_error::VortexResult;

    use super::RangeEntropyCodec;
    use super::RangeEntropyParts;
    use super::RangeGroupedCodec;
    use super::RangePackedCodec;
    use super::RangeTwoLevelCodec;

    #[test]
    fn empty_roundtrip() -> VortexResult<()> {
        let encoded = RangeEntropyCodec::encode(&[], 256)?;
        assert!(encoded.is_empty());
        assert_eq!(encoded.decode()?, Vec::<u64>::new());
        Ok(())
    }

    #[test]
    fn constant_roundtrip() -> VortexResult<()> {
        let values = vec![42; 10_000];
        let encoded = RangeEntropyCodec::encode(&values, 256)?;
        assert_eq!(encoded.bin_lowers(), [42]);
        assert_eq!(encoded.offset_widths(), [0]);
        assert_eq!(encoded.decode()?, values);
        Ok(())
    }

    #[test]
    fn clustered_roundtrip_across_blocks() -> VortexResult<()> {
        let values: Vec<_> = (0..20_000)
            .map(|index| match index % 4 {
                0 => 1_000 + index % 17,
                1 => 1_000_000 + index % 31,
                2 => u64::MAX - index % 13,
                _ => index % 7,
            })
            .collect();
        let encoded = RangeEntropyCodec::encode(&values, 1_024)?;
        assert!(encoded.bin_lowers().len() > 1);
        assert_eq!(
            encoded.block_offsets().len(),
            values.len().div_ceil(1_024) + 1
        );
        assert_eq!(encoded.decode()?, values);
        assert_eq!(encoded.value(10_777)?, values[10_777]);
        Ok(())
    }

    #[test]
    fn full_width_roundtrip() -> VortexResult<()> {
        let values = [0, u64::MAX, 1, u64::MAX - 1];
        let encoded = RangeEntropyCodec::encode(&values, 3)?;
        assert_eq!(encoded.decode()?, values);
        Ok(())
    }

    #[test]
    fn rejects_zero_block_length() {
        assert!(RangeEntropyCodec::encode(&[1, 2, 3], 0).is_err());
    }

    #[test]
    fn restart_blocks_are_independent() -> VortexResult<()> {
        let values: Vec<_> = (0_u64..1_000).map(|value| value * value).collect();
        let encoded = RangeEntropyCodec::encode(&values, 100)?;
        for block_idx in 0..10 {
            assert_eq!(
                encoded.decode_block(block_idx)?,
                values[block_idx * 100..(block_idx + 1) * 100]
            );
        }
        Ok(())
    }

    #[test]
    fn range_decode_touches_boundary_blocks() -> VortexResult<()> {
        let values: Vec<_> = (0_u64..1_000).map(|value| value * value).collect();
        let encoded = RangeEntropyCodec::encode(&values, 100)?;
        assert_eq!(encoded.decode_range(95..205)?, values[95..205]);
        assert_eq!(encoded.decode_range(500..500)?, Vec::<u64>::new());
        assert!(encoded.decode_range(999..1_001).is_err());
        Ok(())
    }

    #[test]
    fn stored_parts_roundtrip() -> VortexResult<()> {
        let values: Vec<_> = (0_u64..4_000).map(|value| value % 97).collect();
        let parts = RangeEntropyCodec::encode(&values, 256)?.into_parts();
        let restored = RangeEntropyCodec::try_from_parts(parts)?;
        assert_eq!(restored.decode()?, values);
        Ok(())
    }

    #[test]
    fn corrupt_block_offset_is_rejected() -> VortexResult<()> {
        let values: Vec<_> = (0_u64..1_000).collect();
        let mut parts = RangeEntropyCodec::encode(&values, 100)?.into_parts();
        parts.block_offsets.pop();
        assert!(RangeEntropyCodec::try_from_parts(parts).is_err());
        Ok(())
    }

    #[test]
    fn corrupt_payload_is_rejected_during_decode() -> VortexResult<()> {
        let values: Vec<_> = (0_u64..1_000).map(|value| value % 31).collect();
        let mut parts = RangeEntropyCodec::encode(&values, 100)?.into_parts();
        let mut payload = parts.payload.to_vec();
        payload.pop();
        parts.payload = ByteBuffer::from(payload);
        if let Some(last) = parts.block_offsets.last_mut() {
            *last = u32::try_from(parts.payload.len())?;
        }
        let restored = RangeEntropyCodec::try_from_parts(parts)?;
        assert!(restored.decode().is_err());
        Ok(())
    }

    #[test]
    fn pseudo_random_roundtrip() -> VortexResult<()> {
        let mut state = 0x4d59_5df4_d0f3_3173_u64;
        let values: Vec<_> = (0..8_192)
            .map(|index| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                match index % 3 {
                    0 => state,
                    1 => state % 1_000,
                    _ => u64::MAX - state % 257,
                }
            })
            .collect();
        for block_len in [1, 7, 256, 1_024, 8_192] {
            assert_eq!(
                RangeEntropyCodec::encode(&values, block_len)?.decode()?,
                values
            );
        }
        Ok(())
    }

    #[test]
    fn range_packed_roundtrip() -> VortexResult<()> {
        let mut state = 0x4d59_5df4_d0f3_3173_u64;
        let values: Vec<_> = (0..20_000)
            .map(|index| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                match index % 4 {
                    0 => state,
                    1 => state % 1_000,
                    2 => u64::MAX - state % 257,
                    _ => 42,
                }
            })
            .collect();
        for block_len in [1, 7, 256, 1_024, 8_192] {
            assert_eq!(
                RangePackedCodec::encode(&values, block_len)?.decode()?,
                values
            );
        }
        Ok(())
    }

    #[test]
    fn range_packed_empty_and_constant_roundtrip() -> VortexResult<()> {
        let empty = RangePackedCodec::encode(&[], 256)?;
        assert!(empty.is_empty());
        assert_eq!(empty.decode()?, Vec::<u64>::new());

        let values = vec![42; 10_000];
        let constant = RangePackedCodec::encode(&values, 256)?;
        assert_eq!(constant.symbol_width(), 0);
        assert_eq!(constant.decode()?, values);
        Ok(())
    }

    #[test]
    fn range_two_level_roundtrip() -> VortexResult<()> {
        let mut state = 0x4d59_5df4_d0f3_3173_u64;
        let values: Vec<_> = (0..20_000)
            .map(|index| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                match index % 4 {
                    0 => state,
                    1 => state % 1_000,
                    2 => u64::MAX - state % 257,
                    _ => 42,
                }
            })
            .collect();
        for block_len in [1, 7, 256, 1_024, 8_192] {
            assert_eq!(
                RangeTwoLevelCodec::encode(&values, block_len)?.decode()?,
                values
            );
        }
        Ok(())
    }

    #[test]
    fn range_grouped_roundtrip() -> VortexResult<()> {
        let values = (0_u64..20_000)
            .map(|index| match index % 4 {
                0 => index,
                1 => 1_000_000 + index % 31,
                2 => u64::MAX - index % 13,
                _ => 42,
            })
            .collect::<Vec<_>>();
        for block_len in [1, 7, 256, 1_024, 8_192] {
            assert_eq!(
                RangeGroupedCodec::encode(&values, block_len)?.decode()?,
                values
            );
        }
        Ok(())
    }

    #[test]
    fn invalid_weight_sum_is_rejected() -> VortexResult<()> {
        let parts = RangeEntropyParts {
            len: 1,
            block_len: 1,
            scale_bits: 12,
            bin_lowers: vec![0],
            offset_widths: vec![0],
            weights: vec![1],
            block_offsets: vec![0, 8],
            payload: ByteBuffer::from(vec![0; 8]),
        };
        assert!(RangeEntropyCodec::try_from_parts(parts).is_err());
        Ok(())
    }
}
