// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::f64::consts::TAU;
use std::hint::black_box;
use std::time::Duration;
use std::time::Instant;

use vortex_error::VortexResult;
use vortex_range_entropy::RangeEntropyCodec;

const N: usize = 1 << 18;
const BLOCK_LEN: usize = 8192;
const ITERATIONS: usize = 200;

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) * (1.0 / ((1_u64 << 53) as f64))
    }

    fn normal(&mut self) -> f64 {
        let radius = (-2.0 * self.uniform().ln()).sqrt();
        radius * (TAU * self.uniform()).cos()
    }
}

fn ordered_f64(value: f64) -> u64 {
    let bits = value.to_bits();
    if bits & (1_u64 << 63) == 0 {
        bits ^ (1_u64 << 63)
    } else {
        !bits
    }
}

fn random_walk() -> Vec<u64> {
    let mut rng = Rng(0x4d59_5df4_d0f3_3173);
    let mut value = 0.0;
    (0..N)
        .map(|_| {
            value += rng.normal() * 0.01;
            ordered_f64(value)
        })
        .collect()
}

fn main() -> VortexResult<()> {
    let values = random_walk();
    black_box(RangeEntropyCodec::encode(&values, BLOCK_LEN)?);

    let mut durations = Vec::with_capacity(ITERATIONS);
    let mut encoded_size = 0usize;
    let mut bin_count = 0usize;
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        let encoded = RangeEntropyCodec::encode(black_box(&values), BLOCK_LEN)?;
        durations.push(start.elapsed());
        encoded_size = black_box(encoded.encoded_size());
        bin_count = black_box(encoded.bin_lowers().len());
    }

    durations.sort_unstable();
    let p10 = durations[ITERATIONS / 10];
    let median = durations[ITERATIONS / 2];
    let p90 = durations[ITERATIONS * 9 / 10];
    let input_bytes = (N * size_of::<u64>()) as f64;
    let throughput = input_bytes / median.as_secs_f64() / 1_000_000.0;
    print_result(encoded_size, bin_count, p10, median, p90, throughput);
    Ok(())
}

fn print_result(
    encoded_size: usize,
    bin_count: usize,
    p10: Duration,
    median: Duration,
    p90: Duration,
    throughput: f64,
) {
    println!(
        "random-walk values={N} blocks={} bins={bin_count} bytes={encoded_size}",
        N.div_ceil(BLOCK_LEN),
    );
    println!(
        "encode p10={:.3}ms median={:.3}ms p90={:.3}ms throughput={throughput:.1}MB/s",
        p10.as_secs_f64() * 1_000.0,
        median.as_secs_f64() * 1_000.0,
        p90.as_secs_f64() * 1_000.0,
    );
}
