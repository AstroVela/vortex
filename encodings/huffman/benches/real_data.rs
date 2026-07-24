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
/// (compressor, per-line compressed payloads). `compressed_len` includes the symbol
/// table; like OnPair below, per-row offsets are excluded (the raw corpus carries its
/// row structure as newlines).
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
    let table_len = compressor.symbol_table().len() * 8 + compressor.symbol_lengths().len();
    let compressed_len = compressed.iter().map(Vec::len).sum::<usize>() + table_len;
    FsstCorpus {
        compressor,
        lines: compressed,
        compressed_len,
        raw_len: data.len(),
    }
}

static WIKI_FSST: LazyLock<FsstCorpus> = LazyLock::new(|| fsst_prepare(&WIKI));
static URLS_FSST: LazyLock<FsstCorpus> = LazyLock::new(|| fsst_prepare(&URLS));

/// OnPair corpora, over the same newline-split rows as FSST. `compressed_len` counts
/// the code stream (2 B/token) plus the compact dictionary (token bytes + u32
/// offsets); per-row offsets are excluded, as for FSST.
struct OnPairCorpus {
    dict: onpair::CompactDictionary,
    codes: Vec<onpair::Token>,
    compressed_len: usize,
    /// Concatenated decoded rows (the corpus minus its newlines), for verification
    /// and throughput accounting.
    decoded_len: usize,
    raw_len: usize,
}

fn onpair_prepare(data: &[u8]) -> OnPairCorpus {
    use onpair::Dictionary;

    // OnPair compresses a flat `(bytes, offsets)` row layout: strip the newline
    // separators and delimit the rows with offsets.
    let flat: Vec<u8> = data
        .split(|&b| b == b'\n')
        .flat_map(<[u8]>::iter)
        .copied()
        .collect();
    let mut offsets: Vec<u32> = vec![0];
    let mut acc = 0u32;
    for line in data.split(|&b| b == b'\n') {
        acc += u32::try_from(line.len()).expect("line fits u32");
        offsets.push(acc);
    }

    let config = onpair::Config {
        seed: Some(42),
        ..onpair::DEFAULT_CONFIG
    };
    let column = onpair::compress(&flat, &offsets, config).expect("onpair compress failed");
    let (dict, codes, _row_offsets) = column.into_raw();
    let compressed_len =
        codes.len() * size_of::<onpair::Token>() + dict.bytes().len() + dict.offsets().len() * 4;
    let decoded_len = onpair::decoded_len(&codes, dict.as_view());
    assert_eq!(decoded_len, flat.len());
    OnPairCorpus {
        dict,
        codes,
        compressed_len,
        decoded_len,
        raw_len: data.len(),
    }
}

static WIKI_ONPAIR: LazyLock<OnPairCorpus> = LazyLock::new(|| onpair_prepare(&WIKI));
static URLS_ONPAIR: LazyLock<OnPairCorpus> = LazyLock::new(|| onpair_prepare(&URLS));

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
        "| dataset | order-0 entropy | huffman (this crate) | zstd-{ZSTD_LEVEL} | fsst | fsst+huffman | onpair | onpair+huffman |"
    );
    println!("| --- | --- | --- | --- | --- | --- | --- | --- |");
    for (name, raw, huff, zst, fsst_corpus, onpair_corpus) in [
        (
            "clickbench-urls",
            &*URLS,
            &*URLS_HUFF,
            &*URLS_ZSTD,
            &*URLS_FSST,
            &*URLS_ONPAIR,
        ),
        (
            "wikipedia",
            &*WIKI,
            &*WIKI_HUFF,
            &*WIKI_ZSTD,
            &*WIKI_FSST,
            &*WIKI_ONPAIR,
        ),
    ] {
        // Verify round-trips before publishing numbers.
        assert_eq!(vortex_huffman::decompress(huff).unwrap(), *raw);
        let entropy = order0_entropy(raw);
        // Entropy-code the FSST/OnPair outputs: the realistic stacking for string
        // columns. For OnPair, the code stream and dictionary are entropy-coded.
        let fsst_concat: Vec<u8> = fsst_corpus.lines.concat();
        let fsst_huff_len = vortex_huffman::compress(&fsst_concat).len();
        let onpair_bytes: Vec<u8> = onpair_corpus
            .codes
            .iter()
            .flat_map(|code| code.to_le_bytes())
            .chain(onpair_corpus.dict.bytes().iter().copied())
            .collect();
        let onpair_huff_len =
            vortex_huffman::compress(&onpair_bytes).len() + onpair_corpus.dict.offsets().len() * 4;
        println!(
            "| {name} | {entropy:.3} bpb (bound {:.3}x) | {:.3}x ({} B) | {:.3}x ({} B) | {:.3}x ({} B) | {:.3}x ({} B) | {:.3}x ({} B) | {:.3}x ({} B) |",
            8.0 / entropy,
            raw.len() as f64 / huff.len() as f64,
            huff.len(),
            raw.len() as f64 / zst.len() as f64,
            zst.len(),
            fsst_corpus.raw_len as f64 / fsst_corpus.compressed_len as f64,
            fsst_corpus.compressed_len,
            fsst_corpus.raw_len as f64 / fsst_huff_len as f64,
            fsst_huff_len,
            onpair_corpus.raw_len as f64 / onpair_corpus.compressed_len as f64,
            onpair_corpus.compressed_len,
            onpair_corpus.raw_len as f64 / onpair_huff_len as f64,
            onpair_huff_len,
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
fn decompress_onpair(bencher: Bencher, dataset: &str) {
    use std::mem::MaybeUninit;

    use onpair::Dictionary;

    let corpus = match dataset {
        "clickbench-urls" => &*URLS_ONPAIR,
        _ => &*WIKI_ONPAIR,
    };
    // The wide dictionary is the reusable fast-path decode table; build it once,
    // like the FSST decompressor and the zstd context.
    let wide = corpus.dict.as_view().to_wide();
    let mut out = vec![MaybeUninit::<u8>::uninit(); corpus.decoded_len + onpair::DECODE_PADDING];
    bencher
        .counter(BytesCount::new(corpus.decoded_len))
        .bench_local(|| {
            onpair::try_decode_into(
                divan::black_box(&corpus.codes),
                wide.as_view(),
                out.as_mut_slice(),
            )
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
