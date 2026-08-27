// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Standalone crossover benchmark for every hand-written CPU-intrinsic pattern in Vortex.

#![allow(clippy::exit)] // Invalid command-line arguments conventionally terminate with status 2.
#![allow(clippy::manual_isolate_lowest_one)] // Keep the scalar PEXT/PDEP baseline portable.

use std::hint::black_box;
use std::io::IsTerminal;
use std::time::Duration;
use std::time::Instant;

const LENGTHS: &[usize] = &[
    1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048, 4096, 8192,
    16_384,
];

#[derive(Clone, Copy)]
enum Format {
    Matrix,
    Markdown,
}

struct Options {
    min_time: Duration,
    samples: usize,
    format: Format,
}

fn options() -> Options {
    let mut min_time_ms = 10;
    let mut samples = 7;
    let mut format = Format::Matrix;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--min-time-ms" => {
                min_time_ms = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| usage())
            }
            "--samples" => {
                samples = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| usage())
            }
            "--format" => {
                format = match args.next().as_deref() {
                    Some("matrix") => Format::Matrix,
                    Some("markdown") => Format::Markdown,
                    _ => usage(),
                }
            }
            "-h" | "--help" => usage(),
            _ => usage(),
        }
    }
    Options {
        min_time: Duration::from_millis(min_time_ms),
        samples,
        format,
    }
}

fn usage() -> ! {
    eprintln!(
        "usage: intrinsics-thresholds [--min-time-ms N] [--samples N] \
[--format matrix|markdown]"
    );
    std::process::exit(2)
}

struct Row {
    case: &'static str,
    implementation: &'static str,
    len: usize,
    scalar_ns: f64,
    intrinsic_ns: f64,
}

impl Row {
    fn ratio(&self) -> f64 {
        self.intrinsic_ns / self.scalar_ns
    }
}

fn measure(mut f: impl FnMut() -> u64, options: &Options) -> f64 {
    let mut iterations = 1usize;
    loop {
        let start = Instant::now();
        for _ in 0..iterations {
            black_box(f());
        }
        if start.elapsed() >= options.min_time || iterations >= 1 << 30 {
            break;
        }
        iterations *= 2;
    }
    let mut samples = Vec::with_capacity(options.samples);
    for _ in 0..options.samples {
        let start = Instant::now();
        for _ in 0..iterations {
            black_box(f());
        }
        samples.push(start.elapsed().as_secs_f64() * 1e9 / iterations as f64);
    }
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn compare(
    rows: &mut Vec<Row>,
    case: &'static str,
    implementation: &'static str,
    len: usize,
    options: &Options,
    mut scalar: impl FnMut() -> u64,
    mut intrinsic: impl FnMut() -> u64,
) {
    assert_eq!(scalar(), intrinsic(), "{case}/{implementation} len={len}");
    let scalar_ns = measure(&mut scalar, options);
    let intrinsic_ns = measure(&mut intrinsic, options);
    rows.push(Row {
        case,
        implementation,
        len,
        scalar_ns,
        intrinsic_ns,
    });
}

fn words(len: usize) -> Vec<u64> {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state
        })
        .collect()
}

fn popcount_scalar(input: &[u64]) -> u64 {
    input.iter().map(|v| u64::from(v.count_ones())).sum()
}

#[cfg(target_arch = "x86_64")]
fn select_scalar(word: u64, mut rank: usize) -> u64 {
    let mut bits = word;
    while rank != 0 {
        bits &= bits - 1;
        rank -= 1;
    }
    u64::from(bits.trailing_zeros())
}

#[cfg(target_arch = "x86_64")]
fn extract_scalar(input: &[u64], masks: &[u64]) -> u64 {
    input
        .iter()
        .zip(masks)
        .fold(0, |sum, (&word, &mask)| sum ^ pext_scalar(word, mask))
}

#[cfg(target_arch = "x86_64")]
fn pext_scalar(mut source: u64, mut mask: u64) -> u64 {
    let mut result = 0;
    let mut out = 1;
    while mask != 0 {
        let bit = mask & mask.wrapping_neg();
        if source & bit != 0 {
            result |= out;
        }
        mask &= mask - 1;
        out <<= 1;
    }
    black_box(&mut source);
    result
}

#[cfg(target_arch = "x86_64")]
fn pdep_scalar(mut source: u64, mut mask: u64) -> u64 {
    let mut result = 0;
    while mask != 0 {
        let bit = mask & mask.wrapping_neg();
        if source & 1 != 0 {
            result |= bit;
        }
        source >>= 1;
        mask &= mask - 1;
    }
    result
}

fn pack_scalar(input: &[u8]) -> u64 {
    input
        .iter()
        .enumerate()
        .fold(0, |bits, (i, &value)| bits | (u64::from(value != 0) << i))
}

#[cfg(target_arch = "x86_64")]
fn gather_scalar(values: &[u32], indices: &[u32]) -> u64 {
    indices
        .iter()
        .map(|&index| u64::from(values[index as usize]))
        .fold(0, u64::wrapping_add)
}

#[cfg(target_arch = "x86_64")]
fn scan_chunks_scalar(input: &[u64], rank: u64) -> u64 {
    let mut remaining = rank;
    for (index, chunk) in input.as_chunks::<8>().0.iter().enumerate() {
        let count = chunk.iter().map(|word| u64::from(word.count_ones())).sum();
        if remaining < count {
            return index as u64;
        }
        remaining -= count;
    }
    input.len().div_ceil(8) as u64
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    #[target_feature(enable = "bmi2")]
    pub unsafe fn select(word: u64, rank: usize) -> u64 {
        unsafe { _tzcnt_u64(_pdep_u64(1 << rank, word)) }
    }

    #[target_feature(enable = "bmi2")]
    pub unsafe fn extract(input: &[u64], masks: &[u64]) -> u64 {
        input
            .iter()
            .zip(masks)
            .fold(0, |sum, (&v, &m)| sum ^ _pext_u64(v, m))
    }

    #[target_feature(enable = "bmi2")]
    pub unsafe fn deposit(input: &[u64], masks: &[u64]) -> u64 {
        input
            .iter()
            .zip(masks)
            .fold(0, |sum, (&v, &m)| sum ^ _pdep_u64(v, m))
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn popcount_avx2(input: &[u64]) -> u64 {
        let bytes =
            unsafe { std::slice::from_raw_parts(input.as_ptr().cast::<u8>(), input.len() * 8) };
        let lookup = _mm256_setr_epi8(
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2,
            3, 3, 4,
        );
        let nibble = _mm256_set1_epi8(0x0f);
        let zero = _mm256_setzero_si256();
        let mut total = 0_u64;
        let mut i = 0;
        while i + 32 <= bytes.len() {
            let value = unsafe { _mm256_loadu_si256(bytes.as_ptr().add(i).cast()) };
            let lo = _mm256_and_si256(value, nibble);
            let hi = _mm256_and_si256(_mm256_srli_epi16(value, 4), nibble);
            let counts = _mm256_add_epi8(
                _mm256_shuffle_epi8(lookup, lo),
                _mm256_shuffle_epi8(lookup, hi),
            );
            let sums = _mm256_sad_epu8(counts, zero);
            let mut lanes = [0_u64; 4];
            unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), sums) };
            total += lanes.into_iter().sum::<u64>();
            i += 32;
        }
        total + super::popcount_scalar(&input[i / 8..])
    }

    #[target_feature(enable = "avx512f,avx512vpopcntdq")]
    pub unsafe fn popcount_avx512(input: &[u64]) -> u64 {
        let mut total = 0;
        let mut i = 0;
        while i + 8 <= input.len() {
            let value = unsafe { _mm512_loadu_si512(input.as_ptr().add(i).cast()) };
            total += _mm512_reduce_add_epi64(_mm512_popcnt_epi64(value)) as u64;
            i += 8;
        }
        total + super::popcount_scalar(&input[i..])
    }

    #[target_feature(enable = "sse2")]
    pub unsafe fn pack_sse2(input: &[u8]) -> u64 {
        let zero = _mm_setzero_si128();
        let mut result = 0;
        for (i, chunk) in input.as_chunks::<16>().0.iter().enumerate() {
            let value = unsafe { _mm_loadu_si128(chunk.as_ptr().cast()) };
            result |=
                ((!_mm_movemask_epi8(_mm_cmpeq_epi8(value, zero)) as u64) & 0xffff) << (i * 16);
        }
        result
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn pack_avx2(input: &[u8]) -> u64 {
        let zero = _mm256_setzero_si256();
        let mut result = 0;
        for (i, chunk) in input.as_chunks::<32>().0.iter().enumerate() {
            let value = unsafe { _mm256_loadu_si256(chunk.as_ptr().cast()) };
            result |= ((!_mm256_movemask_epi8(_mm256_cmpeq_epi8(value, zero)) as u64)
                & 0xffff_ffff)
                << (i * 32);
        }
        result
    }

    #[target_feature(enable = "avx512f,avx512bw")]
    pub unsafe fn pack_avx512(input: &[u8]) -> u64 {
        debug_assert_eq!(input.len(), 64);
        let value = unsafe { _mm512_loadu_si512(input.as_ptr().cast()) };
        _mm512_test_epi8_mask(value, value)
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn gather_u32(values: &[u32], indices: &[u32]) -> u64 {
        let mut total = 0_u64;
        let (chunks, remainder) = indices.as_chunks::<8>();
        for chunk in chunks {
            let index = unsafe { _mm256_loadu_si256(chunk.as_ptr().cast()) };
            let gathered = unsafe {
                _mm256_mask_i32gather_epi32::<4>(
                    _mm256_setzero_si256(),
                    values.as_ptr().cast(),
                    index,
                    _mm256_set1_epi32(-1),
                )
            };
            let mut lanes = [0_u32; 8];
            unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), gathered) };
            total = total.wrapping_add(lanes.into_iter().map(u64::from).sum());
        }
        total.wrapping_add(super::gather_scalar(values, remainder))
    }

    #[target_feature(enable = "avx512f,avx512vpopcntdq")]
    pub unsafe fn scan_chunks(input: &[u64], rank: u64) -> u64 {
        let mut remaining = rank;
        for (index, chunk) in input.as_chunks::<8>().0.iter().enumerate() {
            let value = unsafe { _mm512_loadu_si512(chunk.as_ptr().cast()) };
            let count = _mm512_reduce_add_epi64(_mm512_popcnt_epi64(value)) as u64;
            if remaining < count {
                return index as u64;
            }
            remaining -= count;
        }
        input.len().div_ceil(8) as u64
    }
}

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use std::arch::aarch64::*;

    #[target_feature(enable = "neon")]
    pub unsafe fn popcount(input: &[u64]) -> u64 {
        let bytes =
            unsafe { std::slice::from_raw_parts(input.as_ptr().cast::<u8>(), size_of_val(input)) };
        let mut total = 0;
        let (chunks, remainder) = bytes.as_chunks::<16>();
        for chunk in chunks {
            let counts = unsafe { vcntq_u8(vld1q_u8(chunk.as_ptr())) };
            let sums = vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(counts)));
            total += vgetq_lane_u64::<0>(sums) + vgetq_lane_u64::<1>(sums);
        }
        total
            + remainder
                .iter()
                .map(|byte| u64::from(byte.count_ones()))
                .sum::<u64>()
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn pack(input: &[u8]) -> u64 {
        debug_assert_eq!(input.len(), 64);
        const SHIFTS: [i8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7];
        unsafe {
            let shifts = vld1q_s8(SHIFTS.as_ptr());
            let chunk0 = vshlq_u8(vld1q_u8(input.as_ptr()), shifts);
            let chunk1 = vshlq_u8(vld1q_u8(input.as_ptr().add(16)), shifts);
            let chunk2 = vshlq_u8(vld1q_u8(input.as_ptr().add(32)), shifts);
            let chunk3 = vshlq_u8(vld1q_u8(input.as_ptr().add(48)), shifts);
            let ab = vpaddq_u8(chunk0, chunk1);
            let cd = vpaddq_u8(chunk2, chunk3);
            let packed = vpaddq_u8(ab, cd);
            let packed = vpaddq_u8(packed, packed);
            vgetq_lane_u64::<0>(vreinterpretq_u64_u8(packed))
        }
    }
}

fn main() {
    let options = options();
    let mut rows = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        for &len in LENGTHS {
            let input = words(len);
            if is_x86_feature_detected!("avx2") {
                compare(
                    &mut rows,
                    "popcount",
                    "avx2",
                    len,
                    &options,
                    || popcount_scalar(&input),
                    || unsafe { x86::popcount_avx2(&input) },
                );
            }
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vpopcntdq") {
                compare(
                    &mut rows,
                    "popcount",
                    "avx512-vpopcntdq",
                    len,
                    &options,
                    || popcount_scalar(&input),
                    || unsafe { x86::popcount_avx512(&input) },
                );
                compare(
                    &mut rows,
                    "select-chunk-scan",
                    "avx512-vpopcntdq",
                    len,
                    &options,
                    || scan_chunks_scalar(&input, u64::MAX),
                    || unsafe { x86::scan_chunks(&input, u64::MAX) },
                );
            }
            let masks: Vec<_> = words(len)
                .into_iter()
                .map(|v| v & 0x5555_aaaa_0f0f_f0f0)
                .collect();
            if is_x86_feature_detected!("bmi2") {
                compare(
                    &mut rows,
                    "extract-words",
                    "bmi2-pext",
                    len,
                    &options,
                    || extract_scalar(&input, &masks),
                    || unsafe { x86::extract(&input, &masks) },
                );
                compare(
                    &mut rows,
                    "deposit-words",
                    "bmi2-pdep",
                    len,
                    &options,
                    || {
                        input
                            .iter()
                            .zip(&masks)
                            .fold(0, |s, (&v, &m)| s ^ pdep_scalar(v, m))
                    },
                    || unsafe { x86::deposit(&input, &masks) },
                );
            }
        }
        if is_x86_feature_detected!("bmi2") {
            for &len in LENGTHS {
                let input = words(len);
                compare(
                    &mut rows,
                    "select-word",
                    "bmi2-pdep",
                    len,
                    &options,
                    || {
                        input
                            .iter()
                            .map(|&v| select_scalar(v | 1, (v.count_ones() as usize).min(16) - 1))
                            .fold(0, u64::wrapping_add)
                    },
                    || unsafe {
                        input
                            .iter()
                            .map(|&v| x86::select(v | 1, (v.count_ones() as usize).min(16) - 1))
                            .fold(0, u64::wrapping_add)
                    },
                );
            }
        }
        for &len in LENGTHS.iter().filter(|&&len| len <= 64) {
            let input: Vec<u8> = words(len).into_iter().map(|v| (v & 1) as u8).collect();
            if len.is_multiple_of(16) {
                compare(
                    &mut rows,
                    "pack-bools",
                    "sse2",
                    len,
                    &options,
                    || pack_scalar(&input),
                    || unsafe { x86::pack_sse2(&input) },
                );
            }
            if len.is_multiple_of(32) && is_x86_feature_detected!("avx2") {
                compare(
                    &mut rows,
                    "pack-bools",
                    "avx2",
                    len,
                    &options,
                    || pack_scalar(&input),
                    || unsafe { x86::pack_avx2(&input) },
                );
            }
            if len == 64
                && is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512bw")
            {
                compare(
                    &mut rows,
                    "pack-bools",
                    "avx512-bw",
                    len,
                    &options,
                    || pack_scalar(&input),
                    || unsafe { x86::pack_avx512(&input) },
                );
            }
        }
        if is_x86_feature_detected!("avx2") {
            let values = words(65_536)
                .into_iter()
                .map(|value| {
                    let bytes = value.to_le_bytes();
                    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                })
                .collect::<Vec<_>>();
            for &len in LENGTHS {
                let indices = words(len)
                    .into_iter()
                    .map(|value| u32::try_from(value % 65_536).unwrap_or_default())
                    .collect::<Vec<_>>();
                compare(
                    &mut rows,
                    "take-u32-random",
                    "avx2-gather",
                    len,
                    &options,
                    || gather_scalar(&values, &indices),
                    || unsafe { x86::gather_u32(&values, &indices) },
                );
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        for &len in LENGTHS {
            let input = words(len);
            compare(
                &mut rows,
                "popcount",
                "neon",
                len,
                &options,
                || popcount_scalar(&input),
                || unsafe { aarch64::popcount(&input) },
            );
        }
        let input = words(64)
            .into_iter()
            .map(|value| (value & 1) as u8)
            .collect::<Vec<_>>();
        compare(
            &mut rows,
            "pack-bools",
            "neon",
            64,
            &options,
            || pack_scalar(&input),
            || unsafe { aarch64::pack(&input) },
        );
    }
    print_rows(&rows, options.format);
}

fn print_rows(rows: &[Row], format: Format) {
    match format {
        Format::Matrix => print_matrix(rows),
        Format::Markdown => print_markdown(rows),
    }
}

fn print_markdown(rows: &[Row]) {
    println!("| case | implementation | length | scalar ns | intrinsic ns | ratio |");
    println!("| --- | --- | ---: | ---: | ---: | ---: |");
    for row in rows {
        println!(
            "| {} | {} | {} | {:.2} | {:.2} | {:.3} |",
            row.case,
            row.implementation,
            row.len,
            row.scalar_ns,
            row.intrinsic_ns,
            row.ratio()
        );
    }
    println!("\n## Crossover thresholds\n");
    for (case, implementation, threshold) in thresholds(rows) {
        match threshold {
            Some(len) => println!("- `{case}/{implementation}`: **{len}** elements"),
            None => println!("- `{case}/{implementation}`: no sustained crossover measured"),
        }
    }
}

/// Renders one block per case: an implementation per row, a measured length per column.
fn print_matrix(rows: &[Row]) {
    let colour = std::io::stdout().is_terminal();
    let crossovers = thresholds(rows);
    let mut missing = false;

    println!("ratio = intrinsic ns / scalar ns; below 1.00 the intrinsic wins");
    for case in cases(rows) {
        let case_rows: Vec<&Row> = rows.iter().filter(|row| row.case == case).collect();
        let implementations = implementations(&case_rows);
        let lengths = lengths(&case_rows);

        let cell = |implementation: &str, len: usize| {
            case_rows
                .iter()
                .find(|row| row.implementation == implementation && row.len == len)
                .map(|row| format!("{:.2}", row.ratio()))
        };
        let label_width = implementations
            .iter()
            .map(|implementation| implementation.len())
            .max()
            .unwrap_or_default();
        let widths: Vec<usize> = lengths
            .iter()
            .map(|&len| {
                implementations
                    .iter()
                    .filter_map(|implementation| cell(implementation, len))
                    .map(|value| value.len())
                    .chain([abbreviate(len).len()])
                    .max()
                    .unwrap_or_default()
            })
            .collect();

        println!("\n{case}");
        let mut header = format!("  {:label_width$}", "");
        for (&len, width) in lengths.iter().zip(&widths) {
            header.push_str(&format!("  {:>width$}", abbreviate(len)));
        }
        println!("{header}   crossover");

        for implementation in implementations {
            let mut line = format!("  {implementation:label_width$}");
            for (&len, width) in lengths.iter().zip(&widths) {
                match cell(implementation, len) {
                    Some(value) => {
                        let wins = value.starts_with("0.");
                        line.push_str("  ");
                        line.push_str(&paint(&format!("{value:>width$}"), wins, colour));
                    }
                    None => {
                        missing = true;
                        line.push_str(&format!("  {:>width$}", "·"));
                    }
                }
            }

            let threshold = crossovers
                .iter()
                .find(|&&(other_case, other_implementation, _)| {
                    other_case == case && other_implementation == implementation
                })
                .and_then(|&(.., threshold)| threshold);
            println!("{line}   {:>9}", crossover(threshold));
        }
    }

    if missing {
        println!("\n`·` marks a length that this implementation did not measure.");
    } else {
        println!();
    }

    println!(
        "The crossover is the first length from which the intrinsic wins at every larger length."
    );
}

fn cases(rows: &[Row]) -> Vec<&'static str> {
    let mut cases = Vec::new();
    for row in rows {
        if !cases.contains(&row.case) {
            cases.push(row.case);
        }
    }
    cases
}

fn implementations(rows: &[&Row]) -> Vec<&'static str> {
    let mut implementations = Vec::new();
    for row in rows {
        if !implementations.contains(&row.implementation) {
            implementations.push(row.implementation);
        }
    }
    implementations
}

fn lengths(rows: &[&Row]) -> Vec<usize> {
    let mut lengths: Vec<usize> = rows.iter().map(|row| row.len).collect();
    lengths.sort_unstable();
    lengths.dedup();
    lengths
}

fn abbreviate(len: usize) -> String {
    if len >= 1024 && len.is_multiple_of(1024) {
        format!("{}K", len / 1024)
    } else {
        len.to_string()
    }
}

fn crossover(threshold: Option<usize>) -> String {
    threshold.map_or_else(|| "none".to_string(), |len| len.to_string())
}

fn paint(value: &str, wins: bool, colour: bool) -> String {
    match (colour, wins) {
        (true, true) => format!("\u{1b}[32m{value}\u{1b}[0m"),
        (true, false) => format!("\u{1b}[2m{value}\u{1b}[0m"),
        (false, _) => value.to_string(),
    }
}

/// The first length from which the intrinsic wins at every larger measured length.
fn thresholds(rows: &[Row]) -> Vec<(&'static str, &'static str, Option<usize>)> {
    let mut pairs: Vec<_> = rows
        .iter()
        .map(|row| (row.case, row.implementation))
        .collect();
    pairs.sort_unstable();
    pairs.dedup();

    pairs
        .into_iter()
        .map(|(case, implementation)| {
            let values: Vec<_> = rows
                .iter()
                .filter(|row| row.case == case && row.implementation == implementation)
                .collect();
            let threshold = (0..values.len())
                .find(|&start| {
                    values[start..]
                        .iter()
                        .all(|row| row.intrinsic_ns < row.scalar_ns)
                })
                .map(|start| values[start].len);

            (case, implementation, threshold)
        })
        .collect()
}
