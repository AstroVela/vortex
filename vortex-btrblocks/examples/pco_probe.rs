// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::f64::consts::TAU;
use std::time::Duration;
use std::time::Instant;

use pco::ChunkConfig;
use pco::DeltaSpec;
use pco::ModeSpec;
use pco::PagingSpec;
use pco::metadata::DeltaEncoding;
use pco::metadata::Mode;
use pco::wrapped::FileCompressor;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::PrimitiveArray;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_btrblocks::schemes::float::PcoScheme;
use vortex_btrblocks::schemes::range_entropy::RangeEntropyScheme;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_pco::Pco;
use vortex_range_entropy::RangeEntropyCodec;

const N: usize = 1 << 18;
const VALUES_PER_PAGE: usize = 8192;
const STAGE_ITERATIONS: usize = 15;

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
    reason = "the probe models f32 values stored as f64"
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

    vec![
        ("gaussian", gaussian),
        ("lognormal", lognormal),
        ("decimal", decimal),
        ("widened-f32", widened_f32),
        ("random-walk", random_walk),
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

fn selected_config(dataset: &str) -> ChunkConfig {
    match dataset {
        "gaussian" | "lognormal" => pco_config(ModeSpec::Classic, DeltaSpec::NoOp),
        "decimal" => pco_config(ModeSpec::TryFloatMult(0.01), DeltaSpec::NoOp),
        "widened-f32" => pco_config(ModeSpec::TryFloatQuant(29), DeltaSpec::NoOp),
        "random-walk" => pco_config(ModeSpec::Classic, DeltaSpec::TryConsecutive(1)),
        _ => unreachable!("unknown synthetic dataset"),
    }
}

fn pco_stage_once(
    values: &[f64],
    config: &ChunkConfig,
) -> VortexResult<(Duration, Duration, usize)> {
    let compressor = FileCompressor::default();
    let prepare_start = Instant::now();
    let mut chunk = compressor
        .chunk_compressor(values, config)
        .map_err(|error| vortex_err!("{}", error))?;
    let prepare = prepare_start.elapsed();

    let emit_start = Instant::now();
    let mut encoded_bytes = 0usize;
    let mut metadata = Vec::with_capacity(chunk.meta_size_hint());
    chunk
        .write_meta(&mut metadata)
        .map_err(|error| vortex_err!("{}", error))?;
    encoded_bytes += metadata.len();
    for page_idx in 0..chunk.n_per_page().len() {
        let mut page = Vec::with_capacity(chunk.page_size_hint(page_idx));
        chunk
            .write_page(page_idx, &mut page)
            .map_err(|error| vortex_err!("{}", error))?;
        encoded_bytes += page.len();
    }
    let emit = emit_start.elapsed();
    Ok((prepare, emit, encoded_bytes))
}

fn report_pco_stages(
    dataset: &str,
    config_name: &str,
    values: &[f64],
    config: &ChunkConfig,
) -> VortexResult<()> {
    for _ in 0..3 {
        std::hint::black_box(pco_stage_once(std::hint::black_box(values), config)?);
    }

    let mut prepare_durations = Vec::with_capacity(STAGE_ITERATIONS);
    let mut emit_durations = Vec::with_capacity(STAGE_ITERATIONS);
    let mut total_durations = Vec::with_capacity(STAGE_ITERATIONS);
    let mut encoded_bytes = 0usize;
    for _ in 0..STAGE_ITERATIONS {
        let (prepare, emit, bytes) = pco_stage_once(std::hint::black_box(values), config)?;
        prepare_durations.push(prepare);
        emit_durations.push(emit);
        total_durations.push(prepare + emit);
        encoded_bytes = std::hint::black_box(bytes);
    }

    let prepare = percentile(&mut prepare_durations, 1, 2);
    let emit = percentile(&mut emit_durations, 1, 2);
    let total = percentile(&mut total_durations, 1, 2);
    let input_bytes = size_of_val(values);
    let throughput = input_bytes as f64 / total.as_secs_f64() / 1_000_000.0;
    println!(
        "pco-stage\t{dataset}\t{config_name}\t{encoded_bytes}\t{:.3}\t{:.3}\t{:.3}\t{throughput:.1}",
        prepare.as_secs_f64() * 1_000.0,
        emit.as_secs_f64() * 1_000.0,
        total.as_secs_f64() * 1_000.0,
    );
    Ok(())
}

fn report_auto_selection(dataset: &str, array: &PrimitiveArray) -> VortexResult<()> {
    let config = pco_config(ModeSpec::Auto, DeltaSpec::Auto);
    let compressor = FileCompressor::default();
    let chunk = compressor
        .chunk_compressor(array.as_slice::<f64>(), &config)
        .map_err(|error| vortex_err!("{}", error))?;
    let mode = match &chunk.meta().mode {
        Mode::Classic => "classic".to_owned(),
        Mode::IntMult(_) => "int-mult".to_owned(),
        Mode::FloatMult(_) => "float-mult".to_owned(),
        Mode::FloatQuant(bits) => format!("float-quant-{bits}"),
        Mode::Dict(_) => "dict".to_owned(),
        _ => "unknown".to_owned(),
    };
    let delta = match &chunk.meta().delta_encoding {
        DeltaEncoding::NoOp => "none".to_owned(),
        DeltaEncoding::Consecutive { order, .. } => format!("consecutive-{order}"),
        DeltaEncoding::Lookback { .. } => "lookback".to_owned(),
        DeltaEncoding::Conv1(_) => "conv1".to_owned(),
        _ => "unknown".to_owned(),
    };
    eprintln!("{dataset}: pco-auto selected {mode} with {delta}");
    Ok(())
}

fn report(
    dataset: &str,
    encoder: &str,
    input_bytes: u64,
    encode: impl FnOnce() -> VortexResult<ArrayRef>,
) -> VortexResult<()> {
    let start = Instant::now();
    let compressed = encode()?;
    let elapsed = start.elapsed();
    let encoded_bytes = compressed.nbytes();
    let bits_per_value = 8.0 * encoded_bytes as f64 / N as f64;
    let throughput = input_bytes as f64 / elapsed.as_secs_f64() / 1_000_000.0;
    println!("{dataset}\t{encoder}\t{encoded_bytes}\t{bits_per_value:.3}\t{throughput:.1}");
    Ok(())
}

fn ordered_f64(value: f64) -> u64 {
    let bits = value.to_bits();
    if bits & (1_u64 << 63) == 0 {
        bits ^ (1_u64 << 63)
    } else {
        !bits
    }
}

fn report_native(dataset: &str, values: &[f64], input_bytes: u64) -> VortexResult<()> {
    let latents: Vec<_> = values.iter().copied().map(ordered_f64).collect();
    let start = Instant::now();
    let compressed = RangeEntropyCodec::encode(&latents, VALUES_PER_PAGE)?;
    let elapsed = start.elapsed();
    let encoded_bytes = u64::try_from(compressed.encoded_size())?;
    let bits_per_value = 8.0 * encoded_bytes as f64 / N as f64;
    let throughput = input_bytes as f64 / elapsed.as_secs_f64() / 1_000_000.0;
    eprintln!(
        "{dataset}: native range entropy used {} bins",
        compressed.bin_lowers().len()
    );
    println!("{dataset}\trange-entropy\t{encoded_bytes}\t{bits_per_value:.3}\t{throughput:.1}");
    Ok(())
}

fn main() -> VortexResult<()> {
    let session = array_session();
    let default = BtrBlocksCompressor::default();
    let with_auto = BtrBlocksCompressorBuilder::default()
        .with_new_scheme(&PcoScheme)
        .build();
    let with_range_entropy = BtrBlocksCompressorBuilder::default()
        .with_new_scheme(&RangeEntropyScheme)
        .build();

    println!("dataset\tencoder\tbytes\tbits/value\tMB/s");
    println!("pco-stage columns: dataset, config, bytes, prepare-ms, emit-ms, total-ms, MB/s");
    for (name, values) in datasets() {
        let array = PrimitiveArray::from_iter(values.clone());
        let array_ref = array.clone().into_array();
        let input_bytes = array_ref.nbytes();
        report_auto_selection(name, &array)?;
        report_pco_stages(
            name,
            "auto",
            &values,
            &pco_config(ModeSpec::Auto, DeltaSpec::Auto),
        )?;
        for level in [0, 4, 8, 12] {
            let config = selected_config(name).with_compression_level(level);
            report_pco_stages(name, &format!("selected-level-{level}"), &values, &config)?;
        }
        report_native(name, &values, input_bytes)?;

        report(name, "btrblocks", input_bytes, || {
            default.compress(&array_ref, &mut session.create_execution_ctx())
        })?;
        report(name, "btrblocks+pco-auto", input_bytes, || {
            with_auto.compress(&array_ref, &mut session.create_execution_ctx())
        })?;
        report(name, "btrblocks+range-entropy", input_bytes, || {
            with_range_entropy.compress(&array_ref, &mut session.create_execution_ctx())
        })?;

        for (encoder, config) in [
            (
                "pco-classic",
                pco_config(ModeSpec::Classic, DeltaSpec::NoOp),
            ),
            (
                "pco-classic-delta1",
                pco_config(ModeSpec::Classic, DeltaSpec::TryConsecutive(1)),
            ),
            (
                "pco-classic-delta2",
                pco_config(ModeSpec::Classic, DeltaSpec::TryConsecutive(2)),
            ),
            ("pco-auto", pco_config(ModeSpec::Auto, DeltaSpec::Auto)),
        ] {
            report(name, encoder, input_bytes, || {
                Ok(Pco::from_primitive_with_config(
                    array.as_view(),
                    &config,
                    pco::DEFAULT_MAX_PAGE_N,
                    &mut session.create_execution_ctx(),
                )?
                .into_array())
            })?;
        }
    }

    Ok(())
}
