// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared coarse models for fixed-bin scheme rejection.

#[derive(Clone, Copy, Debug)]
pub(crate) struct CoarseModel {
    pub(crate) range_ratio: f64,
    pub(crate) block_ratio: f64,
}

pub(crate) fn coarse_model(values: &[u64], source_width: usize) -> CoarseModel {
    CoarseModel {
        range_ratio: coarse_prefix_ratio(values, source_width),
        block_ratio: coarse_block_ratio(values, source_width),
    }
}

fn coarse_prefix_ratio(values: &[u64], source_width: usize) -> f64 {
    let Some((&minimum, &maximum)) = values.iter().min().zip(values.iter().max()) else {
        return 0.0;
    };
    let span_width = bit_width(maximum - minimum);
    let mut best_bits = values.len() * usize::from(span_width);
    for code_width in 1_u8..=6 {
        if code_width > span_width {
            break;
        }
        let bucket_count = 1usize << code_width;
        let shift = span_width - code_width;
        let mut counts = [0usize; 64];
        let mut minima = [u64::MAX; 64];
        let mut maxima = [0_u64; 64];
        for &value in values {
            let relative = value - minimum;
            let bucket = usize::try_from(relative >> shift).unwrap_or(bucket_count - 1);
            let bucket = bucket.min(bucket_count - 1);
            counts[bucket] += 1;
            minima[bucket] = minima[bucket].min(value);
            maxima[bucket] = maxima[bucket].max(value);
        }
        let offset_bits = (0..bucket_count)
            .filter(|&bucket| counts[bucket] != 0)
            .map(|bucket| counts[bucket] * usize::from(bit_width(maxima[bucket] - minima[bucket])))
            .sum::<usize>();
        best_bits = best_bits.min(values.len() * usize::from(code_width) + offset_bits);
    }

    if best_bits == 0 {
        f64::INFINITY
    } else {
        values.len() as f64 * source_width as f64 / best_bits as f64
    }
}

fn coarse_block_ratio(values: &[u64], source_width: usize) -> f64 {
    let block_bits = values
        .chunks(64)
        .map(|block| {
            let minimum = block.iter().copied().min().unwrap_or_default();
            let maximum = block.iter().copied().max().unwrap_or_default();
            block.len() * usize::from(bit_width(maximum - minimum))
        })
        .sum::<usize>();
    if block_bits == 0 {
        f64::INFINITY
    } else {
        values.len() as f64 * source_width as f64 / block_bits as f64
    }
}

fn bit_width(value: u64) -> u8 {
    u8::try_from(u64::BITS - value.leading_zeros()).unwrap_or(u8::MAX)
}
