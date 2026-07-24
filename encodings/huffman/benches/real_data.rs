// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compression ratio and decompression speed of the Huffman codec on real datasets
//! (ClickBench URLs, Wikipedia/enwik8), with zstd and FSST as reference points.
//!
//! Run with:
//!
//! ```bash
//! cargo bench -p vortex-huffman --features _test-harness --bench real_data
//! ```
//!
//! A compression-ratio summary table is printed before the divan timing runs.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::cast_precision_loss)]

mod datasets;

use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::BytesCount;

fn main() {
    print_ratio_summary();
    divan::main();
}

static WIKI: LazyLock<Vec<u8>> = LazyLock::new(datasets::wikipedia);
static URLS: LazyLock<Vec<u8>> = LazyLock::new(datasets::clickbench_urls);

static WIKI_HUFF: LazyLock<Vec<u8>> = LazyLock::new(|| vortex_huffman::compress(&WIKI));
static URLS_HUFF: LazyLock<Vec<u8>> = LazyLock::new(|| vortex_huffman::compress(&URLS));

const ZSTD_LEVEL: i32 = 3;
static WIKI_ZSTD: LazyLock<Vec<u8>> =
    LazyLock::new(|| zstd::bulk::compress(&WIKI, ZSTD_LEVEL).unwrap());
static URLS_ZSTD: LazyLock<Vec<u8>> =
    LazyLock::new(|| zstd::bulk::compress(&URLS, ZSTD_LEVEL).unwrap());

/// FSST corpora are the newline-split lines of each dataset (FSST is a string codec);
/// (compressor, per-line compressed payloads).
struct FsstCorpus {
    compressor: fsst::Compressor,
    lines: Vec<Vec<u8>>,
    compressed_len: usize,
    raw_len: usize,
}

fn fsst_prepare(data: &[u8]) -> FsstCorpus {
    let lines: Vec<&[u8]> = data.split(|&b| b == b'\n').collect();
    let compressor = fsst::Compressor::train(&lines);
    let compressed: Vec<Vec<u8>> = lines.iter().map(|line| compressor.compress(line)).collect();
    let compressed_len = compressed.iter().map(Vec::len).sum();
    FsstCorpus {
        compressor,
        lines: compressed,
        compressed_len,
        raw_len: data.len(),
    }
}

static WIKI_FSST: LazyLock<FsstCorpus> = LazyLock::new(|| fsst_prepare(&WIKI));
static URLS_FSST: LazyLock<FsstCorpus> = LazyLock::new(|| fsst_prepare(&URLS));

/// Order-0 Shannon entropy in bits per byte: the lower bound for any order-0
/// entropy coder like Huffman.
fn order0_entropy(data: &[u8]) -> f64 {
    let mut hist = [0u64; 256];
    for &byte in data {
        hist[usize::from(byte)] += 1;
    }
    let total = data.len() as f64;
    hist.iter()
        .filter(|&&count| count > 0)
        .map(|&count| {
            let p = count as f64 / total;
            -p * p.log2()
        })
        .sum()
}

fn print_ratio_summary() {
    println!(
        "## Compression ratio (raw / compressed), {} MiB per corpus\n",
        datasets::CORPUS_LEN / (1024 * 1024)
    );
    println!(
        "| dataset | order-0 entropy | huffman (this crate) | zstd-{ZSTD_LEVEL} | fsst | fsst+huffman |"
    );
    println!("| --- | --- | --- | --- | --- | --- |");
    for (name, raw, huff, zst, fsst_corpus) in [
        (
            "clickbench-urls",
            &*URLS,
            &*URLS_HUFF,
            &*URLS_ZSTD,
            &*URLS_FSST,
        ),
        ("wikipedia", &*WIKI, &*WIKI_HUFF, &*WIKI_ZSTD, &*WIKI_FSST),
    ] {
        // Verify round-trips before publishing numbers.
        assert_eq!(vortex_huffman::decompress(huff).unwrap(), *raw);
        let entropy = order0_entropy(raw);
        // Entropy-code the FSST output: the realistic stacking for string columns.
        let fsst_concat: Vec<u8> = fsst_corpus.lines.concat();
        let fsst_huff_len = vortex_huffman::compress(&fsst_concat).len();
        println!(
            "| {name} | {entropy:.3} bpb (bound {:.3}x) | {:.3}x ({} B) | {:.3}x ({} B) | {:.3}x ({} B) | {:.3}x ({} B) |",
            8.0 / entropy,
            raw.len() as f64 / huff.len() as f64,
            huff.len(),
            raw.len() as f64 / zst.len() as f64,
            zst.len(),
            fsst_corpus.raw_len as f64 / fsst_corpus.compressed_len as f64,
            fsst_corpus.compressed_len,
            fsst_corpus.raw_len as f64 / fsst_huff_len as f64,
            fsst_huff_len,
        );
    }
    println!();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Decompression speed (throughput counted in *uncompressed* bytes)
////////////////////////////////////////////////////////////////////////////////////////////////////

#[divan::bench(args = ["clickbench-urls", "wikipedia"])]
fn decompress_huffman(bencher: Bencher, dataset: &str) {
    let (raw, compressed) = match dataset {
        "clickbench-urls" => (&*URLS, &*URLS_HUFF),
        _ => (&*WIKI, &*WIKI_HUFF),
    };
    let mut out = vec![0u8; raw.len()];
    bencher.counter(BytesCount::new(raw.len())).bench_local(|| {
        vortex_huffman::decompress_into(divan::black_box(compressed), &mut out).unwrap()
    });
}

#[divan::bench(args = ["clickbench-urls", "wikipedia"])]
fn decompress_zstd(bencher: Bencher, dataset: &str) {
    let (raw, compressed) = match dataset {
        "clickbench-urls" => (&*URLS, &*URLS_ZSTD),
        _ => (&*WIKI, &*WIKI_ZSTD),
    };
    let mut out = vec![0u8; raw.len()];
    let mut zstd_decompressor = zstd::bulk::Decompressor::new().unwrap();
    bencher.counter(BytesCount::new(raw.len())).bench_local(|| {
        zstd_decompressor
            .decompress_to_buffer(divan::black_box(compressed), out.as_mut_slice())
            .unwrap()
    });
}

#[divan::bench(args = ["clickbench-urls", "wikipedia"])]
fn decompress_fsst(bencher: Bencher, dataset: &str) {
    let corpus = match dataset {
        "clickbench-urls" => &*URLS_FSST,
        _ => &*WIKI_FSST,
    };
    let decompressor = corpus.compressor.decompressor();
    bencher.counter(BytesCount::new(corpus.raw_len)).bench(|| {
        let mut total = 0usize;
        for line in &corpus.lines {
            total += decompressor.decompress(divan::black_box(line)).len();
        }
        total
    });
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Compression speed (secondary, for context)
////////////////////////////////////////////////////////////////////////////////////////////////////

#[divan::bench(args = ["clickbench-urls", "wikipedia"])]
fn compress_huffman(bencher: Bencher, dataset: &str) {
    let raw = match dataset {
        "clickbench-urls" => &*URLS,
        _ => &*WIKI,
    };
    bencher
        .counter(BytesCount::new(raw.len()))
        .bench(|| vortex_huffman::compress(divan::black_box(raw)));
}

#[divan::bench(args = ["clickbench-urls", "wikipedia"])]
fn compress_zstd(bencher: Bencher, dataset: &str) {
    let raw = match dataset {
        "clickbench-urls" => &*URLS,
        _ => &*WIKI,
    };
    bencher
        .counter(BytesCount::new(raw.len()))
        .bench(|| zstd::bulk::compress(divan::black_box(raw), ZSTD_LEVEL).unwrap());
}
