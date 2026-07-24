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

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

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
    /// The `codes` child huffman-coded (the stacked layout's code stream).
    codes_huff: Vec<u8>,
    /// The low and high byte planes of `codes`, each huffman-coded separately.
    codes_huff_lo: Vec<u8>,
    codes_huff_hi: Vec<u8>,
    /// The dictionary token bytes huffman-coded.
    dict_huff: Vec<u8>,
    num_rows: usize,
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
    let code_bytes: Vec<u8> = codes.iter().flat_map(|code| code.to_le_bytes()).collect();
    let codes_huff = vortex_huffman::compress(&code_bytes);
    let lo_plane: Vec<u8> = codes.iter().map(|code| code.to_le_bytes()[0]).collect();
    let hi_plane: Vec<u8> = codes.iter().map(|code| code.to_le_bytes()[1]).collect();
    let codes_huff_lo = vortex_huffman::compress(&lo_plane);
    let codes_huff_hi = vortex_huffman::compress(&hi_plane);
    let dict_huff = vortex_huffman::compress(dict.bytes());
    // One-time verification that the stacked layout round-trips to the row bytes.
    {
        use std::mem::MaybeUninit;

        let decoded_code_bytes = vortex_huffman::decompress(&codes_huff).expect("codes round-trip");
        assert_eq!(decoded_code_bytes, code_bytes);
        let decoded_dict_bytes = vortex_huffman::decompress(&dict_huff).expect("dict round-trip");
        let rebuilt =
            onpair::CompactDictionary::validate(decoded_dict_bytes, dict.offsets().to_vec())
                .expect("dictionary must validate after round-trip");
        let mut buf = vec![MaybeUninit::<u8>::uninit(); decoded_len + onpair::DECODE_PADDING];
        let written = onpair::try_decode_into(&codes, rebuilt.as_view(), buf.as_mut_slice())
            .expect("stacked decode");
        assert_eq!(written, flat.len());
        let decoded: &[u8] = unsafe { std::slice::from_raw_parts(buf.as_ptr().cast(), written) };
        assert_eq!(decoded, flat.as_slice());
    }
    OnPairCorpus {
        dict,
        codes,
        codes_huff,
        codes_huff_lo,
        codes_huff_hi,
        dict_huff,
        num_rows: offsets.len() - 1,
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
        // columns. For OnPair, the code stream and dictionary token bytes are each
        // huffman-coded as separate children (see the array tree below).
        let fsst_concat: Vec<u8> = fsst_corpus.lines.concat();
        let fsst_huff_len = vortex_huffman::compress(&fsst_concat).len();
        let onpair_huff_len = onpair_corpus.codes_huff.len()
            + onpair_corpus.dict_huff.len()
            + onpair_corpus.dict.offsets().len() * 4;
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

    for (name, onpair_corpus) in [
        ("clickbench-urls", &*URLS_ONPAIR),
        ("wikipedia", &*WIKI_ONPAIR),
    ] {
        print_onpair_huffman_tree(name, onpair_corpus);
        print_codes_isolation(name, onpair_corpus);
    }
}

/// The `codes` child in isolation: what Huffman buys over the primitive encodings
/// Vortex would otherwise use for a bounded u16 token stream, against the
/// entropy bounds.
fn print_codes_isolation(name: &str, corpus: &OnPairCorpus) {
    let num_codes = corpus.codes.len();
    let raw_len = num_codes * size_of::<onpair::Token>();

    // The primitive alternative: FastLanes-style bit-packing to the token width.
    let token_bits = usize::BITS - (corpus.dict.num_tokens() - 1).leading_zeros();
    let bitpacked_len = (num_codes * token_bits as usize).div_ceil(8);

    // Entropy bounds: per-token (the bound for a token-alphabet entropy coder) and
    // per-byte-plane (the bound for what the byte-oriented planes can reach).
    let mut token_hist = vec![0u64; corpus.dict.num_tokens()];
    for &code in &corpus.codes {
        token_hist[usize::from(code)] += 1;
    }
    let total = num_codes as f64;
    let token_entropy: f64 = token_hist
        .iter()
        .filter(|&&count| count > 0)
        .map(|&count| {
            let p = count as f64 / total;
            -p * p.log2()
        })
        .sum();
    let token_bound_len = (token_entropy * total / 8.0).ceil() as u64;

    let plane_len = corpus.codes_huff_lo.len() + corpus.codes_huff_hi.len();
    let per_tok = |len: usize| len as f64 * 8.0 / total;
    println!("### {name}: codes child in isolation ({num_codes} tokens)\n");
    println!("| representation | size | bits/token | vs raw u16 |");
    println!("| --- | --- | --- | --- |");
    println!("| raw u16 primitive | {raw_len} B | 16.0 | 1.000x |");
    println!(
        "| bitpack-{token_bits} (FastLanes-style primitive) | {bitpacked_len} B | {token_bits}.0 | {:.3}x |",
        raw_len as f64 / bitpacked_len as f64,
    );
    println!(
        "| huffman, interleaved u16-LE bytes | {} B | {:.2} | {:.3}x |",
        corpus.codes_huff.len(),
        per_tok(corpus.codes_huff.len()),
        raw_len as f64 / corpus.codes_huff.len() as f64,
    );
    println!(
        "| huffman, split byte planes | {plane_len} B | {:.2} | {:.3}x |",
        per_tok(plane_len),
        raw_len as f64 / plane_len as f64,
    );
    println!(
        "| token-alphabet entropy bound | {token_bound_len} B | {token_entropy:.2} | {:.3}x |",
        raw_len as f64 / token_bound_len as f64,
    );
    println!();
}

/// The array tree an `onpair+huffman` encoding of the corpus would have in Vortex,
/// with concrete child sizes. Mirrors the children of the `OnPair` array in
/// `vortex-onpair`, with the two byte-payload children wrapped in Huffman.
fn print_onpair_huffman_tree(name: &str, corpus: &OnPairCorpus) {
    let num_tokens = corpus.dict.num_tokens();
    let code_bytes = corpus.codes.len() * size_of::<onpair::Token>();
    println!(
        "### {name}: onpair+huffman array tree ({} rows)\n",
        corpus.num_rows
    );
    println!("```text");
    println!(
        "OnPair(utf8, {} rows, {:.2} MiB of row bytes)",
        corpus.num_rows,
        corpus.decoded_len as f64 / (1024.0 * 1024.0)
    );
    println!(
        "├─ codes:        Huffman({} B) over PrimitiveArray<u16> x{} ({} B raw)",
        corpus.codes_huff.len(),
        corpus.codes.len(),
        code_bytes,
    );
    println!(
        "├─ dict_bytes:   Huffman({} B) over token bytes ({} B raw, {num_tokens} tokens)",
        corpus.dict_huff.len(),
        corpus.dict.bytes().len(),
    );
    println!(
        "├─ dict_offsets: PrimitiveArray<u32> x{} ({} B)",
        corpus.dict.offsets().len(),
        corpus.dict.offsets().len() * 4,
    );
    println!(
        "├─ row_offsets:  PrimitiveArray<u32> x{} (excluded from ratio accounting,",
        corpus.num_rows + 1,
    );
    println!("│                like FSST's; FastLanes-bitpacked in practice)");
    println!("╰─ validity:     all-valid");
    println!("```\n");
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

/// The codes child in isolation: huffman decode of the interleaved u16-LE byte
/// stream, throughput counted in decoded codes bytes.
#[divan::bench(args = ["clickbench-urls", "wikipedia"])]
fn decompress_huffman_codes(bencher: Bencher, dataset: &str) {
    let corpus = match dataset {
        "clickbench-urls" => &*URLS_ONPAIR,
        _ => &*WIKI_ONPAIR,
    };
    let raw_len = corpus.codes.len() * size_of::<onpair::Token>();
    let mut out = vec![0u8; raw_len];
    bencher.counter(BytesCount::new(raw_len)).bench_local(|| {
        vortex_huffman::decompress_into(divan::black_box(&corpus.codes_huff), &mut out).unwrap()
    });
}

/// The codes child as two huffman-coded byte planes: decode both planes and
/// reassemble the u16 tokens, throughput counted in decoded codes bytes.
#[divan::bench(args = ["clickbench-urls", "wikipedia"])]
fn decompress_huffman_codes_planes(bencher: Bencher, dataset: &str) {
    let corpus = match dataset {
        "clickbench-urls" => &*URLS_ONPAIR,
        _ => &*WIKI_ONPAIR,
    };
    let num_codes = corpus.codes.len();
    let mut lo_plane = vec![0u8; num_codes];
    let mut hi_plane = vec![0u8; num_codes];
    let mut codes: Vec<onpair::Token> = Vec::with_capacity(num_codes);
    bencher
        .counter(BytesCount::new(num_codes * size_of::<onpair::Token>()))
        .bench_local(|| {
            vortex_huffman::decompress_into(divan::black_box(&corpus.codes_huff_lo), &mut lo_plane)
                .unwrap();
            vortex_huffman::decompress_into(divan::black_box(&corpus.codes_huff_hi), &mut hi_plane)
                .unwrap();
            codes.clear();
            codes.extend(
                lo_plane
                    .iter()
                    .zip(hi_plane.iter())
                    .map(|(&lo, &hi)| onpair::Token::from_le_bytes([lo, hi])),
            );
            codes.len()
        });
}

/// Full stacked decode: huffman-decode the code stream and dictionary bytes, rebuild
/// the dictionary (validate + widen), then onpair-decode the rows. This is the whole
/// per-chunk scan path of the stacked layout; only the output buffers are reused
/// across iterations.
#[divan::bench(args = ["clickbench-urls", "wikipedia"])]
fn decompress_onpair_huffman(bencher: Bencher, dataset: &str) {
    use std::mem::MaybeUninit;

    use onpair::Dictionary;

    let corpus = match dataset {
        "clickbench-urls" => &*URLS_ONPAIR,
        _ => &*WIKI_ONPAIR,
    };
    let mut code_bytes = vec![0u8; corpus.codes.len() * size_of::<onpair::Token>()];
    let mut codes: Vec<onpair::Token> = Vec::with_capacity(corpus.codes.len());
    let dict_offsets = corpus.dict.offsets().to_vec();
    let mut out = vec![MaybeUninit::<u8>::uninit(); corpus.decoded_len + onpair::DECODE_PADDING];
    bencher
        .counter(BytesCount::new(corpus.decoded_len))
        .bench_local(|| {
            vortex_huffman::decompress_into(divan::black_box(&corpus.codes_huff), &mut code_bytes)
                .unwrap();
            codes.clear();
            codes.extend(
                code_bytes
                    .chunks_exact(2)
                    .map(|pair| onpair::Token::from_le_bytes([pair[0], pair[1]])),
            );
            let dict_bytes =
                vortex_huffman::decompress(divan::black_box(&corpus.dict_huff)).unwrap();
            let dict = onpair::CompactDictionary::validate(dict_bytes, dict_offsets.clone())
                .expect("dictionary must round-trip");
            let wide = dict.as_view().to_wide();
            onpair::try_decode_into(&codes, wide.as_view(), out.as_mut_slice()).unwrap()
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
