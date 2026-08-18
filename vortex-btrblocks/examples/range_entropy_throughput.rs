// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::f64::consts::TAU;
use std::hint::black_box;
use std::time::Duration;
use std::time::Instant;

use pco::ChunkConfig;
use pco::DeltaSpec;
use pco::ModeSpec;
use pco::PagingSpec;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_btrblocks::SchemeExt;
use vortex_btrblocks::schemes::float::FloatQuantScheme;
use vortex_btrblocks::schemes::float::OrderedBlockResidualScheme;
use vortex_btrblocks::schemes::range_entropy::RangeEntropyScheme;
use vortex_error::VortexResult;
use vortex_float_quant::FloatQuant;
use vortex_float_quant::FloatQuantArraySlotsExt;
use vortex_float_quant::estimate_k;
use vortex_pco::Pco;
use vortex_range_entropy::BitSplitCodec;
use vortex_range_entropy::PatchedFoRCodec;
use vortex_range_entropy::RangeEntropy;
use vortex_range_entropy::RangeEntropyCodec;
use vortex_range_entropy::RangeGroupedCodec;
use vortex_range_entropy::RangePackedCodec;
use vortex_range_entropy::RangeTwoLevelCodec;
use vortex_session::VortexSession;

const N: usize = 1 << 18;
const VALUES_PER_PAGE: usize = 8192;
const DECODE_ITERATIONS: usize = 100;
const COMPRESS_ITERATIONS: usize = 40;
const SCALAR_ACCESS_ITERATIONS: usize = N * 8;
const ARRAY_SCALAR_ACCESS_ITERATIONS: usize = N;

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

#[expect(
    clippy::cast_possible_truncation,
    reason = "the benchmark models f32 values stored as f64"
)]
fn widen_f32(value: f64) -> f64 {
    value as f32 as f64
}

fn datasets() -> Vec<(&'static str, Vec<f64>)> {
    let mut rng = Rng(0x4d59_5df4_d0f3_3173);
    let gaussian = (0..N).map(|_| 1_000.0 + 17.0 * rng.normal()).collect();
    let lognormal = (0..N).map(|_| rng.normal().exp()).collect();
    let decimal = (0..N)
        .map(|_| (rng.next_u64() % 1_000_000) as f64 / 100.0)
        .collect();
    let widened_f32 = (0..N)
        .map(|_| widen_f32(1_000.0 + 17.0 * rng.normal()))
        .collect();
    let mut value = 0.0;
    let random_walk = (0..N)
        .map(|_| {
            value += rng.normal() * 0.01;
            value
        })
        .collect();
    let four_clusters = (0..N)
        .map(|_| {
            let center = match rng.next_u64() & 3 {
                0 => -1_000_000_000.0,
                1 => -1_000.0,
                2 => 1_000.0,
                _ => 1_000_000_000.0,
            };
            center + 17.0 * rng.normal()
        })
        .collect();

    vec![
        ("gaussian", gaussian),
        ("lognormal", lognormal),
        ("decimal", decimal),
        ("widened-f32", widened_f32),
        ("random-walk", random_walk),
        ("four-clusters", four_clusters),
    ]
}

fn pco_config(mode: ModeSpec, delta: DeltaSpec) -> ChunkConfig {
    ChunkConfig::default()
        .with_mode_spec(mode)
        .with_delta_spec(delta)
        .with_paging_spec(PagingSpec::EqualPagesUpTo(VALUES_PER_PAGE))
}

fn percentile(durations: &mut [Duration], numerator: usize, denominator: usize) -> Duration {
    durations.sort_unstable();
    durations[durations.len() * numerator / denominator]
}

fn encoding_tree(array: &ArrayRef) -> String {
    let children = array.children();
    if children.is_empty() {
        return array.encoding_id().to_string();
    }
    let children = children
        .iter()
        .map(encoding_tree)
        .collect::<Vec<_>>()
        .join(",");
    format!("{}({children})", array.encoding_id())
}

fn decode_once(array: &ArrayRef, session: &VortexSession) -> VortexResult<Duration> {
    let start = Instant::now();
    let decoded = array
        .clone()
        .execute::<PrimitiveArray>(&mut session.create_execution_ctx())?;
    black_box(decoded.nbytes());
    Ok(start.elapsed())
}

fn report_decode(
    dataset: &str,
    encoder: &str,
    array: &ArrayRef,
    durations: &mut [Duration],
) -> VortexResult<()> {
    let mut for_p10 = durations.to_vec();
    let mut for_median = durations.to_vec();
    let p10 = percentile(&mut for_p10, 1, 10);
    let median = percentile(&mut for_median, 1, 2);
    let p90 = percentile(durations, 9, 10);
    let throughput = (N * size_of::<f64>()) as f64 / median.as_secs_f64() / 1_000_000.0;
    println!(
        "decode\t{dataset}\t{encoder}\t{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{throughput:.1}",
        array.encoding_id(),
        array.nbytes(),
        p10.as_secs_f64() * 1_000.0,
        median.as_secs_f64() * 1_000.0,
        p90.as_secs_f64() * 1_000.0,
    );
    Ok(())
}

fn measure_decoders(
    dataset: &str,
    configs: &[(&str, &ArrayRef)],
    session: &VortexSession,
) -> VortexResult<()> {
    for _ in 0..3 {
        for (_, array) in configs {
            black_box(decode_once(array, session)?);
        }
    }

    let mut durations = vec![Vec::with_capacity(DECODE_ITERATIONS); configs.len()];
    for iteration in 0..DECODE_ITERATIONS {
        for offset in 0..configs.len() {
            let config_index = (iteration + offset) % configs.len();
            durations[config_index].push(decode_once(configs[config_index].1, session)?);
        }
    }

    for (config_index, (config, array)) in configs.iter().enumerate() {
        report_decode(dataset, config, array, &mut durations[config_index])?;
    }
    Ok(())
}

fn measure_range_entropy_codec(
    dataset: &str,
    values: &[f64],
    codec: &RangeEntropyCodec,
) -> VortexResult<()> {
    for _ in 0..3 {
        black_box(codec.decode()?);
    }
    let mut durations = Vec::with_capacity(DECODE_ITERATIONS);
    for _ in 0..DECODE_ITERATIONS {
        let start = Instant::now();
        black_box(codec.decode()?);
        durations.push(start.elapsed());
    }
    let p10 = percentile(&mut durations.to_vec(), 1, 10);
    let p90 = percentile(&mut durations.to_vec(), 9, 10);
    let median = percentile(&mut durations, 1, 2);
    let throughput =
        values.len() as f64 * size_of::<f64>() as f64 / median.as_secs_f64() / 1_000_000.0;
    println!(
        "decode\t{dataset}\trange-entropy-codec\tvortex.range_entropy_codec\t{}\t{:.3}\t{:.3}\t{:.3}\t{throughput:.1}",
        codec.encoded_size(),
        p10.as_secs_f64() * 1_000.0,
        median.as_secs_f64() * 1_000.0,
        p90.as_secs_f64() * 1_000.0,
    );
    Ok(())
}

fn measure_range_packed_codec(
    dataset: &str,
    values: &[f64],
    codec: &RangePackedCodec,
) -> VortexResult<()> {
    for _ in 0..3 {
        black_box(codec.decode()?);
    }
    let mut durations = Vec::with_capacity(DECODE_ITERATIONS);
    for _ in 0..DECODE_ITERATIONS {
        let start = Instant::now();
        black_box(codec.decode()?);
        durations.push(start.elapsed());
    }
    let p10 = percentile(&mut durations.to_vec(), 1, 10);
    let p90 = percentile(&mut durations.to_vec(), 9, 10);
    let median = percentile(&mut durations, 1, 2);
    let throughput =
        values.len() as f64 * size_of::<f64>() as f64 / median.as_secs_f64() / 1_000_000.0;
    println!(
        "decode\t{dataset}\trange-packed-codec\tvortex.range_packed_codec\t{}\t{:.3}\t{:.3}\t{:.3}\t{throughput:.1}",
        codec.encoded_size(),
        p10.as_secs_f64() * 1_000.0,
        median.as_secs_f64() * 1_000.0,
        p90.as_secs_f64() * 1_000.0,
    );
    Ok(())
}

fn measure_range_two_level_codec(
    dataset: &str,
    values: &[f64],
    codec: &RangeTwoLevelCodec,
) -> VortexResult<()> {
    for _ in 0..3 {
        black_box(codec.decode()?);
    }
    let mut durations = Vec::with_capacity(DECODE_ITERATIONS);
    for _ in 0..DECODE_ITERATIONS {
        let start = Instant::now();
        black_box(codec.decode()?);
        durations.push(start.elapsed());
    }
    let p10 = percentile(&mut durations.to_vec(), 1, 10);
    let p90 = percentile(&mut durations.to_vec(), 9, 10);
    let median = percentile(&mut durations, 1, 2);
    let throughput =
        values.len() as f64 * size_of::<f64>() as f64 / median.as_secs_f64() / 1_000_000.0;
    println!(
        "decode\t{dataset}\trange-two-level-codec\tvortex.range_two_level_codec\t{}\t{:.3}\t{:.3}\t{:.3}\t{throughput:.1}",
        codec.encoded_size(),
        p10.as_secs_f64() * 1_000.0,
        median.as_secs_f64() * 1_000.0,
        p90.as_secs_f64() * 1_000.0,
    );
    Ok(())
}

fn measure_range_grouped_codec(
    dataset: &str,
    values: &[f64],
    codec: &RangeGroupedCodec,
) -> VortexResult<()> {
    for _ in 0..3 {
        black_box(codec.decode()?);
    }
    let mut durations = Vec::with_capacity(DECODE_ITERATIONS);
    for _ in 0..DECODE_ITERATIONS {
        let start = Instant::now();
        black_box(codec.decode()?);
        durations.push(start.elapsed());
    }
    let p10 = percentile(&mut durations.to_vec(), 1, 10);
    let p90 = percentile(&mut durations.to_vec(), 9, 10);
    let median = percentile(&mut durations, 1, 2);
    let throughput =
        values.len() as f64 * size_of::<f64>() as f64 / median.as_secs_f64() / 1_000_000.0;
    println!(
        "decode\t{dataset}\trange-grouped-codec\tvortex.range_grouped_codec\t{}\t{:.3}\t{:.3}\t{:.3}\t{throughput:.1}",
        codec.encoded_size(),
        p10.as_secs_f64() * 1_000.0,
        median.as_secs_f64() * 1_000.0,
        p90.as_secs_f64() * 1_000.0,
    );
    Ok(())
}

fn measure_patched_for_codec(
    dataset: &str,
    config: &str,
    values: &[f64],
    codec: &PatchedFoRCodec,
) -> VortexResult<()> {
    for _ in 0..3 {
        black_box(codec.decode()?);
    }
    let mut durations = Vec::with_capacity(DECODE_ITERATIONS);
    for _ in 0..DECODE_ITERATIONS {
        let start = Instant::now();
        black_box(codec.decode()?);
        durations.push(start.elapsed());
    }
    let p10 = percentile(&mut durations.to_vec(), 1, 10);
    let p90 = percentile(&mut durations.to_vec(), 9, 10);
    let median = percentile(&mut durations, 1, 2);
    let throughput =
        values.len() as f64 * size_of::<f64>() as f64 / median.as_secs_f64() / 1_000_000.0;
    println!(
        "decode\t{dataset}\t{config}\tvortex.patched_for_codec\t{}\t{:.3}\t{:.3}\t{:.3}\t{throughput:.1}",
        codec.encoded_size(),
        p10.as_secs_f64() * 1_000.0,
        median.as_secs_f64() * 1_000.0,
        p90.as_secs_f64() * 1_000.0,
    );
    Ok(())
}

fn measure_ordered_float_patched_for_codec(
    dataset: &str,
    values: &[f64],
    codec: &PatchedFoRCodec,
) -> VortexResult<()> {
    let decode = || -> VortexResult<Vec<f64>> {
        Ok(codec
            .decode()?
            .into_iter()
            .map(|ordered| {
                let bits = if ordered & (1_u64 << 63) == 0 {
                    !ordered
                } else {
                    ordered ^ (1_u64 << 63)
                };
                f64::from_bits(bits)
            })
            .collect::<Vec<_>>())
    };
    for _ in 0..3 {
        black_box(decode()?);
    }
    let mut durations = Vec::with_capacity(DECODE_ITERATIONS);
    for _ in 0..DECODE_ITERATIONS {
        let start = Instant::now();
        black_box(decode()?);
        durations.push(start.elapsed());
    }
    let p10 = percentile(&mut durations.to_vec(), 1, 10);
    let p90 = percentile(&mut durations.to_vec(), 9, 10);
    let median = percentile(&mut durations, 1, 2);
    let throughput =
        values.len() as f64 * size_of::<f64>() as f64 / median.as_secs_f64() / 1_000_000.0;
    println!(
        "decode\t{dataset}\tordered-float+patched-for-single-base\tvortex.prototype_stack\t{}\t{:.3}\t{:.3}\t{:.3}\t{throughput:.1}",
        codec.encoded_size(),
        p10.as_secs_f64() * 1_000.0,
        median.as_secs_f64() * 1_000.0,
        p90.as_secs_f64() * 1_000.0,
    );

    for _ in 0..3 {
        black_box(codec.decode_ordered_f64()?);
    }
    let mut durations = Vec::with_capacity(DECODE_ITERATIONS);
    for _ in 0..DECODE_ITERATIONS {
        let start = Instant::now();
        black_box(codec.decode_ordered_f64()?);
        durations.push(start.elapsed());
    }
    let p10 = percentile(&mut durations.to_vec(), 1, 10);
    let p90 = percentile(&mut durations.to_vec(), 9, 10);
    let median = percentile(&mut durations, 1, 2);
    let throughput =
        values.len() as f64 * size_of::<f64>() as f64 / median.as_secs_f64() / 1_000_000.0;
    println!(
        "decode\t{dataset}\tordered-float+patched-for-single-base-fused\tvortex.prototype_fused\t{}\t{:.3}\t{:.3}\t{:.3}\t{throughput:.1}",
        codec.encoded_size(),
        p10.as_secs_f64() * 1_000.0,
        median.as_secs_f64() * 1_000.0,
        p90.as_secs_f64() * 1_000.0,
    );
    Ok(())
}

fn measure_patched_for_scalar_at(
    dataset: &str,
    config: &str,
    codec: &PatchedFoRCodec,
) -> VortexResult<()> {
    let mut checksum = 0_u64;
    for access in 0..N {
        let index = access.wrapping_mul(0x9e37_79b1) & (N - 1);
        checksum ^= black_box(codec.scalar_at(index)?);
    }
    black_box(checksum);

    let mut durations = Vec::with_capacity(9);
    for _ in 0..9 {
        let start = Instant::now();
        let mut checksum = 0_u64;
        for access in 0..SCALAR_ACCESS_ITERATIONS {
            let index = access.wrapping_mul(0x9e37_79b1) & (N - 1);
            checksum ^= black_box(codec.scalar_at(index)?);
        }
        durations.push(start.elapsed());
        black_box(checksum);
    }
    let median = percentile(&mut durations, 1, 2);
    let nanoseconds = median.as_secs_f64() * 1_000_000_000.0 / SCALAR_ACCESS_ITERATIONS as f64;
    let accesses_per_second = SCALAR_ACCESS_ITERATIONS as f64 / median.as_secs_f64() / 1_000_000.0;
    println!(
        "scalar-at\t{dataset}\t{config}\tvortex.patched_for_codec\t{}\t{nanoseconds:.2}\t{accesses_per_second:.1}",
        codec.encoded_size(),
    );
    Ok(())
}

fn measure_array_scalar_at(
    dataset: &str,
    config: &str,
    array: &ArrayRef,
    session: &VortexSession,
) -> VortexResult<()> {
    let mut durations = Vec::with_capacity(5);
    for _ in 0..5 {
        let mut ctx = session.create_execution_ctx();
        let start = Instant::now();
        let mut checksum = 0_u64;
        for access in 0..ARRAY_SCALAR_ACCESS_ITERATIONS {
            let index = access.wrapping_mul(0x9e37_79b1) & (N - 1);
            let value = array
                .execute_scalar(index, &mut ctx)?
                .as_primitive()
                .typed_value::<f64>()
                .ok_or_else(|| vortex_error::vortex_err!("benchmark value is null"))?;
            checksum ^= black_box(value.to_bits());
        }
        durations.push(start.elapsed());
        black_box(checksum);
    }
    let median = percentile(&mut durations, 1, 2);
    let nanoseconds =
        median.as_secs_f64() * 1_000_000_000.0 / ARRAY_SCALAR_ACCESS_ITERATIONS as f64;
    let accesses_per_second =
        ARRAY_SCALAR_ACCESS_ITERATIONS as f64 / median.as_secs_f64() / 1_000_000.0;
    println!(
        "scalar-at\t{dataset}\t{config}\t{}\t{}\t{nanoseconds:.2}\t{accesses_per_second:.1}",
        array.encoding_id(),
        array.nbytes(),
    );
    Ok(())
}

fn measure_bit_split_codec(
    dataset: &str,
    values: &[f64],
    codec: &BitSplitCodec,
) -> VortexResult<()> {
    for _ in 0..3 {
        black_box(codec.decode()?);
    }
    let mut durations = Vec::with_capacity(DECODE_ITERATIONS);
    for _ in 0..DECODE_ITERATIONS {
        let start = Instant::now();
        black_box(codec.decode()?);
        durations.push(start.elapsed());
    }
    let p10 = percentile(&mut durations.to_vec(), 1, 10);
    let p90 = percentile(&mut durations.to_vec(), 9, 10);
    let median = percentile(&mut durations, 1, 2);
    let throughput =
        values.len() as f64 * size_of::<f64>() as f64 / median.as_secs_f64() / 1_000_000.0;
    println!(
        "decode\t{dataset}\tbit-split-codec\tvortex.bit_split_codec\t{}\t{:.3}\t{:.3}\t{:.3}\t{throughput:.1}",
        codec.encoded_size(),
        p10.as_secs_f64() * 1_000.0,
        median.as_secs_f64() * 1_000.0,
        p90.as_secs_f64() * 1_000.0,
    );
    Ok(())
}

fn measure_bit_split_scalar_at(dataset: &str, codec: &BitSplitCodec) -> VortexResult<()> {
    let mut durations = Vec::with_capacity(9);
    for _ in 0..9 {
        let start = Instant::now();
        let mut checksum = 0_u64;
        for access in 0..SCALAR_ACCESS_ITERATIONS {
            let index = access.wrapping_mul(0x9e37_79b1) & (N - 1);
            checksum ^= black_box(codec.scalar_at(index)?);
        }
        durations.push(start.elapsed());
        black_box(checksum);
    }
    let median = percentile(&mut durations, 1, 2);
    let nanoseconds = median.as_secs_f64() * 1_000_000_000.0 / SCALAR_ACCESS_ITERATIONS as f64;
    let accesses_per_second = SCALAR_ACCESS_ITERATIONS as f64 / median.as_secs_f64() / 1_000_000.0;
    println!(
        "scalar-at\t{dataset}\tbit-split-codec\tvortex.bit_split_codec\t{}\t{nanoseconds:.2}\t{accesses_per_second:.1}",
        codec.encoded_size(),
    );
    Ok(())
}

fn measure_native_codec_encoders(dataset: &str, values: &[u64]) -> VortexResult<()> {
    for (name, encode) in [(
        "range-entropy-codec",
        RangeEntropyCodec::encode as fn(&[u64], usize) -> VortexResult<RangeEntropyCodec>,
    )] {
        let mut durations = Vec::with_capacity(COMPRESS_ITERATIONS);
        let mut size = 0;
        for _ in 0..COMPRESS_ITERATIONS {
            let start = Instant::now();
            let codec = encode(black_box(values), VALUES_PER_PAGE)?;
            durations.push(start.elapsed());
            size = black_box(codec.encoded_size());
        }
        report_native_codec_encoder(dataset, name, size, &mut durations);
    }

    let mut durations = Vec::with_capacity(COMPRESS_ITERATIONS);
    let mut size = 0;
    for _ in 0..COMPRESS_ITERATIONS {
        let start = Instant::now();
        let codec = RangePackedCodec::encode(black_box(values), VALUES_PER_PAGE)?;
        durations.push(start.elapsed());
        size = black_box(codec.encoded_size());
    }
    report_native_codec_encoder(dataset, "range-packed-codec", size, &mut durations);

    let mut durations = Vec::with_capacity(COMPRESS_ITERATIONS);
    let mut size = 0;
    for _ in 0..COMPRESS_ITERATIONS {
        let start = Instant::now();
        let codec = RangeTwoLevelCodec::encode(black_box(values), VALUES_PER_PAGE)?;
        durations.push(start.elapsed());
        size = black_box(codec.encoded_size());
    }
    report_native_codec_encoder(dataset, "range-two-level-codec", size, &mut durations);

    let mut durations = Vec::with_capacity(COMPRESS_ITERATIONS);
    let mut size = 0;
    for _ in 0..COMPRESS_ITERATIONS {
        let start = Instant::now();
        let codec = RangeGroupedCodec::encode(black_box(values), VALUES_PER_PAGE)?;
        durations.push(start.elapsed());
        size = black_box(codec.encoded_size());
    }
    report_native_codec_encoder(dataset, "range-grouped-codec", size, &mut durations);

    let mut durations = Vec::with_capacity(COMPRESS_ITERATIONS);
    let mut size = 0;
    for _ in 0..COMPRESS_ITERATIONS {
        let start = Instant::now();
        let codec = PatchedFoRCodec::encode(black_box(values))?;
        durations.push(start.elapsed());
        size = black_box(codec.encoded_size());
    }
    report_native_codec_encoder(dataset, "patched-for-codec", size, &mut durations);

    let mut durations = Vec::with_capacity(COMPRESS_ITERATIONS);
    let mut size = 0;
    for _ in 0..COMPRESS_ITERATIONS {
        let start = Instant::now();
        let codec = PatchedFoRCodec::encode_single_base(black_box(values))?;
        durations.push(start.elapsed());
        size = black_box(codec.encoded_size());
    }
    report_native_codec_encoder(dataset, "patched-for-single-base", size, &mut durations);

    let mut durations = Vec::with_capacity(COMPRESS_ITERATIONS);
    let mut size = 0;
    for _ in 0..COMPRESS_ITERATIONS {
        let start = Instant::now();
        let codec = BitSplitCodec::encode(black_box(values))?;
        durations.push(start.elapsed());
        size = black_box(codec.encoded_size());
    }
    report_native_codec_encoder(dataset, "bit-split-codec", size, &mut durations);
    Ok(())
}

fn measure_ordered_float_patched_for_encoder(dataset: &str, values: &[f64]) -> VortexResult<()> {
    let mut durations = Vec::with_capacity(COMPRESS_ITERATIONS);
    let mut size = 0;
    for _ in 0..COMPRESS_ITERATIONS {
        let start = Instant::now();
        let ordered = values
            .iter()
            .map(|value| {
                let bits = value.to_bits();
                if bits & (1_u64 << 63) == 0 {
                    bits ^ (1_u64 << 63)
                } else {
                    !bits
                }
            })
            .collect::<Vec<_>>();
        let codec = PatchedFoRCodec::encode_single_base(black_box(&ordered))?;
        durations.push(start.elapsed());
        size = black_box(codec.encoded_size());
    }
    report_native_codec_encoder(
        dataset,
        "ordered-float+patched-for-single-base",
        size,
        &mut durations,
    );
    Ok(())
}

fn report_native_codec_encoder(dataset: &str, name: &str, size: usize, durations: &mut [Duration]) {
    let p10 = percentile(&mut durations.to_vec(), 1, 10);
    let p90 = percentile(&mut durations.to_vec(), 9, 10);
    let median = percentile(durations, 1, 2);
    let throughput = (N * size_of::<u64>()) as f64 / median.as_secs_f64() / 1_000_000.0;
    println!(
        "codec-encode\t{dataset}\t{name}\tnative-codec\t{size}\t{:.3}\t{:.3}\t{:.3}\t{throughput:.1}",
        p10.as_secs_f64() * 1_000.0,
        median.as_secs_f64() * 1_000.0,
        p90.as_secs_f64() * 1_000.0,
    );
}

fn report_symbol_models(dataset: &str, codec: &RangeEntropyCodec) {
    let total_weight = (1_u64 << codec.scale_bits()) as f64;
    let mut probabilities = codec
        .weights()
        .iter()
        .map(|&weight| f64::from(weight) / total_weight)
        .collect::<Vec<_>>();
    let entropy = probabilities
        .iter()
        .map(|&probability| -probability * probability.log2())
        .sum::<f64>();
    probabilities.sort_by(|left, right| right.total_cmp(left));
    let bin_count = probabilities.len();
    let flat_width = usize::from(u8::try_from(codec.bin_lowers().len().ilog2()).unwrap_or(u8::MAX))
        + usize::from(!codec.bin_lowers().len().is_power_of_two());
    let mut best = (f64::INFINITY, 0usize, 0usize, 0.0);
    let minimum_tag_width = flat_width.min(2);
    for tag_width in minimum_tag_width..=flat_width {
        let direct_count = ((1usize << tag_width) - 1).min(bin_count);
        let escape_probability = 1.0 - probabilities[..direct_count].iter().sum::<f64>();
        let cold_count = bin_count - direct_count;
        let cold_width = if cold_count <= 1 {
            0
        } else {
            usize::from(u8::try_from(cold_count.ilog2()).unwrap_or(u8::MAX))
                + usize::from(!cold_count.is_power_of_two())
        };
        let cost = tag_width as f64 + escape_probability * cold_width as f64;
        if cost < best.0 {
            best = (cost, tag_width, cold_width, escape_probability);
        }
    }
    println!(
        "symbol-model\t{dataset}\tbins={bin_count}\tentropy={entropy:.3}\tflat={flat_width}\ttwo-level={:.3}\ttag={}\tcold={}\tescape={:.3}",
        best.0, best.1, best.2, best.3
    );
}

fn measure_float_quant_stages(
    dataset: &str,
    primitive: &PrimitiveArray,
    compressor: &BtrBlocksCompressor,
    session: &VortexSession,
) -> VortexResult<()> {
    let Some(k) = estimate_k(primitive.as_view()) else {
        return Ok(());
    };
    let encoded = FloatQuant::from_primitive(primitive.as_view(), k)?;
    let primary = encoded.primary().clone();
    let secondary = encoded.secondary().clone();
    let mut estimate_times = Vec::with_capacity(COMPRESS_ITERATIONS);
    let mut split_times = Vec::with_capacity(COMPRESS_ITERATIONS);
    let mut primary_times = Vec::with_capacity(COMPRESS_ITERATIONS);
    let mut secondary_times = Vec::with_capacity(COMPRESS_ITERATIONS);
    for _ in 0..COMPRESS_ITERATIONS {
        let start = Instant::now();
        black_box(estimate_k(primitive.as_view()));
        estimate_times.push(start.elapsed());

        let start = Instant::now();
        black_box(FloatQuant::from_primitive(primitive.as_view(), k)?);
        split_times.push(start.elapsed());

        let start = Instant::now();
        black_box(compressor.compress(&primary, &mut session.create_execution_ctx())?);
        primary_times.push(start.elapsed());

        let start = Instant::now();
        black_box(compressor.compress(&secondary, &mut session.create_execution_ctx())?);
        secondary_times.push(start.elapsed());
    }

    for (stage, durations) in [
        ("estimate-k", &mut estimate_times),
        ("split", &mut split_times),
        ("primary-child", &mut primary_times),
        ("secondary-child", &mut secondary_times),
    ] {
        let median = percentile(durations, 1, 2);
        println!(
            "stage\t{dataset}\t{stage}\tk={k}\t{:.3}",
            median.as_secs_f64() * 1_000.0
        );
    }
    Ok(())
}

fn compress_once(
    compressor: &BtrBlocksCompressor,
    values: &[f64],
    session: &VortexSession,
) -> VortexResult<(Duration, ArrayRef)> {
    let array = PrimitiveArray::from_iter(values.iter().copied()).into_array();
    let start = Instant::now();
    let compressed = compressor.compress(&array, &mut session.create_execution_ctx())?;
    Ok((start.elapsed(), compressed))
}

fn measure_compressors(
    dataset: &str,
    values: &[f64],
    configs: &[(&str, &BtrBlocksCompressor)],
    session: &VortexSession,
) -> VortexResult<()> {
    for _ in 0..2 {
        for (_, compressor) in configs {
            black_box(compress_once(compressor, values, session)?);
        }
    }

    let mut durations = (0..configs.len())
        .map(|_| Vec::with_capacity(COMPRESS_ITERATIONS))
        .collect::<Vec<_>>();
    let mut outputs = vec![None; configs.len()];
    for iteration in 0..COMPRESS_ITERATIONS {
        for offset in 0..configs.len() {
            let config_index = (iteration + offset) % configs.len();
            record_compression(
                configs[config_index].1,
                values,
                session,
                &mut durations[config_index],
                &mut outputs[config_index],
            )?;
        }
    }

    for (index, (name, _)) in configs.iter().enumerate() {
        report_compressor(dataset, name, outputs[index].take(), &mut durations[index])?;
    }
    Ok(())
}

fn record_compression(
    compressor: &BtrBlocksCompressor,
    values: &[f64],
    session: &VortexSession,
    durations: &mut Vec<Duration>,
    last_output: &mut Option<ArrayRef>,
) -> VortexResult<()> {
    let (elapsed, output) = compress_once(compressor, values, session)?;
    durations.push(elapsed);
    *last_output = Some(output);
    Ok(())
}

fn report_compressor(
    dataset: &str,
    config: &str,
    output: Option<ArrayRef>,
    durations: &mut [Duration],
) -> VortexResult<()> {
    let output =
        output.ok_or_else(|| vortex_error::vortex_err!("compressor produced no output"))?;
    let p10 = percentile(&mut durations.to_vec(), 1, 10);
    let p90 = percentile(&mut durations.to_vec(), 9, 10);
    let median = percentile(durations, 1, 2);
    let throughput = (N * size_of::<f64>()) as f64 / median.as_secs_f64() / 1_000_000.0;
    println!(
        "compress\t{dataset}\t{config}\t{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{throughput:.1}",
        output.encoding_id(),
        output.nbytes(),
        p10.as_secs_f64() * 1_000.0,
        median.as_secs_f64() * 1_000.0,
        p90.as_secs_f64() * 1_000.0,
    );
    Ok(())
}

#[expect(
    clippy::cognitive_complexity,
    reason = "the benchmark driver constructs and measures one fixed configuration matrix"
)]
fn main() -> VortexResult<()> {
    let dataset_filter = std::env::args().nth(1);
    let session = array_session();
    let baseline = BtrBlocksCompressorBuilder::default()
        .exclude_schemes([FloatQuantScheme.id(), OrderedBlockResidualScheme.id()])
        .build();
    let range_candidate = BtrBlocksCompressorBuilder::default()
        .exclude_schemes([FloatQuantScheme.id(), OrderedBlockResidualScheme.id()])
        .with_new_scheme(&RangeEntropyScheme)
        .build();
    let default_without_block_residual = BtrBlocksCompressorBuilder::default()
        .exclude_schemes([OrderedBlockResidualScheme.id()])
        .build();
    let float_candidate = BtrBlocksCompressor::default();
    let stacked_candidate = BtrBlocksCompressorBuilder::default()
        .with_new_scheme(&RangeEntropyScheme)
        .build();
    let compact = BtrBlocksCompressorBuilder::default().with_compact().build();

    println!("operation\tdataset\tconfig\tencoding\tbytes\tp10-ms\tmedian-ms\tp90-ms\tMB/s");
    for (name, values) in datasets() {
        if dataset_filter
            .as_deref()
            .is_some_and(|filter| filter != name)
        {
            continue;
        }
        let primitive = PrimitiveArray::from_iter(values.iter().copied());
        let expected = primitive.clone().into_array();
        let ordered_values = values
            .iter()
            .map(|value| {
                let bits = value.to_bits();
                if bits & (1_u64 << 63) == 0 {
                    bits ^ (1_u64 << 63)
                } else {
                    !bits
                }
            })
            .collect::<Vec<_>>();
        let native_codec = RangeEntropyCodec::encode(&ordered_values, VALUES_PER_PAGE)?;
        let range_packed_codec = RangePackedCodec::encode(&ordered_values, VALUES_PER_PAGE)?;
        let range_two_level_codec = RangeTwoLevelCodec::encode(&ordered_values, VALUES_PER_PAGE)?;
        let range_grouped_codec = RangeGroupedCodec::encode(&ordered_values, VALUES_PER_PAGE)?;
        let patched_for_codec = PatchedFoRCodec::encode(&ordered_values)?;
        let patched_for_single_base = PatchedFoRCodec::encode_single_base(&ordered_values)?;
        let bit_split_codec = BitSplitCodec::encode(&ordered_values)?;
        println!(
            "patched-model\t{name}\tblocks={}\tavg-bases={:.2}\tavg-width={:.2}\tpatch-rate={:.4}",
            patched_for_codec.block_count(),
            patched_for_codec.total_base_count() as f64
                / patched_for_codec.block_count().max(1) as f64,
            patched_for_codec.total_residual_width() as f64
                / patched_for_codec.block_count().max(1) as f64,
            patched_for_codec.patch_count() as f64 / ordered_values.len().max(1) as f64,
        );
        println!(
            "bit-split-model\t{name}\tavg-prefixes={:.2}\tavg-suffix-width={:.2}",
            bit_split_codec.average_prefix_count(),
            bit_split_codec.average_suffix_width(),
        );
        report_symbol_models(name, &native_codec);
        let native =
            RangeEntropy::from_primitive(primitive.as_view(), VALUES_PER_PAGE)?.into_array();
        let pco_classic = Pco::from_primitive_with_config(
            primitive.as_view(),
            &pco_config(ModeSpec::Classic, DeltaSpec::NoOp),
            pco::DEFAULT_MAX_PAGE_N,
            &mut session.create_execution_ctx(),
        )?
        .into_array();
        let pco_auto = Pco::from_primitive_with_config(
            primitive.as_view(),
            &pco_config(ModeSpec::Auto, DeltaSpec::Auto),
            pco::DEFAULT_MAX_PAGE_N,
            &mut session.create_execution_ctx(),
        )?
        .into_array();
        let default_encoded = baseline.compress(&expected, &mut session.create_execution_ctx())?;
        let range_encoded =
            range_candidate.compress(&expected, &mut session.create_execution_ctx())?;
        let float_encoded =
            float_candidate.compress(&expected, &mut session.create_execution_ctx())?;
        let default_without_block_residual_encoded = default_without_block_residual
            .compress(&expected, &mut session.create_execution_ctx())?;
        let stacked_encoded =
            stacked_candidate.compress(&expected, &mut session.create_execution_ctx())?;
        let compact_encoded = compact.compress(&expected, &mut session.create_execution_ctx())?;
        println!(
            "structure\t{name}\twithout-new-float={}\twithout-new-float+range={}\tdefault-off={}\tdefault-on={}\tdefault+range={}\tcompact={}",
            encoding_tree(&default_encoded),
            encoding_tree(&range_encoded),
            encoding_tree(&default_without_block_residual_encoded),
            encoding_tree(&float_encoded),
            encoding_tree(&stacked_encoded),
            encoding_tree(&compact_encoded)
        );

        let mut verify_ctx = session.create_execution_ctx();
        assert_eq!(range_packed_codec.decode()?, ordered_values);
        assert_eq!(range_two_level_codec.decode()?, ordered_values);
        assert_eq!(range_grouped_codec.decode()?, ordered_values);
        assert_eq!(patched_for_codec.decode()?, ordered_values);
        assert_eq!(patched_for_single_base.decode()?, ordered_values);
        assert_eq!(bit_split_codec.decode()?, ordered_values);
        assert_arrays_eq!(native, expected, &mut verify_ctx);
        assert_arrays_eq!(pco_classic, expected, &mut verify_ctx);
        assert_arrays_eq!(pco_auto, expected, &mut verify_ctx);
        assert_arrays_eq!(default_encoded, expected, &mut verify_ctx);
        assert_arrays_eq!(range_encoded, expected, &mut verify_ctx);
        assert_arrays_eq!(
            default_without_block_residual_encoded,
            expected,
            &mut verify_ctx
        );
        assert_arrays_eq!(float_encoded, expected, &mut verify_ctx);
        assert_arrays_eq!(stacked_encoded, expected, &mut verify_ctx);
        assert_arrays_eq!(compact_encoded, expected, &mut verify_ctx);

        measure_decoders(
            name,
            &[
                ("default-without-new-float", &default_encoded),
                ("default-without-new-float+range-entropy", &range_encoded),
                ("default-off", &default_without_block_residual_encoded),
                ("default-on", &float_encoded),
                ("default+range-entropy", &stacked_encoded),
                ("compact", &compact_encoded),
                ("range-entropy", &native),
                ("pco-classic", &pco_classic),
                ("pco-auto", &pco_auto),
            ],
            &session,
        )?;
        measure_array_scalar_at(
            name,
            "default-off",
            &default_without_block_residual_encoded,
            &session,
        )?;
        measure_array_scalar_at(name, "default-on", &float_encoded, &session)?;
        measure_range_entropy_codec(name, &values, &native_codec)?;
        measure_range_packed_codec(name, &values, &range_packed_codec)?;
        measure_range_two_level_codec(name, &values, &range_two_level_codec)?;
        measure_range_grouped_codec(name, &values, &range_grouped_codec)?;
        measure_patched_for_codec(name, "patched-for-codec", &values, &patched_for_codec)?;
        measure_patched_for_codec(
            name,
            "patched-for-single-base",
            &values,
            &patched_for_single_base,
        )?;
        measure_ordered_float_patched_for_codec(name, &values, &patched_for_single_base)?;
        measure_patched_for_scalar_at(name, "patched-for-codec", &patched_for_codec)?;
        measure_patched_for_scalar_at(name, "patched-for-single-base", &patched_for_single_base)?;
        measure_bit_split_codec(name, &values, &bit_split_codec)?;
        measure_bit_split_scalar_at(name, &bit_split_codec)?;
        measure_native_codec_encoders(name, &ordered_values)?;
        measure_ordered_float_patched_for_encoder(name, &values)?;
        measure_float_quant_stages(name, &primitive, &baseline, &session)?;
        measure_compressors(
            name,
            &values,
            &[
                ("default-without-new-float", &baseline),
                ("default-without-new-float+range-entropy", &range_candidate),
                ("default-off", &default_without_block_residual),
                ("default-on", &float_candidate),
                ("default+range-entropy", &stacked_candidate),
                ("compact", &compact),
            ],
            &session,
        )?;
    }
    Ok(())
}
