// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]

use rand::RngExt;
use rand::SeedableRng;
use rand::prelude::StdRng;
use rstest::rstest;
use vortex_error::VortexResult;

use crate::DEFAULT_BLOCK_LEN;
use crate::MAX_CODE_LEN;
use crate::build_code_lengths;
use crate::compress;
use crate::compress_with_block_len;
use crate::decompress;

fn roundtrip(input: &[u8], block_len: usize) -> VortexResult<()> {
    let compressed = compress_with_block_len(input, block_len);
    let decompressed = decompress(&compressed)?;
    assert_eq!(input, decompressed.as_slice());
    Ok(())
}

#[test]
fn empty() -> VortexResult<()> {
    roundtrip(&[], DEFAULT_BLOCK_LEN)
}

#[rstest]
#[case(1)]
#[case(2)]
#[case(3)]
#[case(7)]
#[case(1000)]
fn tiny_inputs(#[case] len: usize) -> VortexResult<()> {
    let input: Vec<u8> = (0..len).map(|i| (i % 7) as u8 * 31).collect();
    roundtrip(&input, DEFAULT_BLOCK_LEN)
}

#[test]
fn single_symbol_rle() -> VortexResult<()> {
    roundtrip(&[42u8; 100_000], DEFAULT_BLOCK_LEN)
}

#[test]
fn incompressible_raw_fallback() -> VortexResult<()> {
    let mut rng = StdRng::seed_from_u64(0);
    let input: Vec<u8> = (0..200_000).map(|_| rng.random()).collect();
    let compressed = compress(&input);
    // Random bytes must fall back to raw blocks: tiny overhead only.
    assert!(compressed.len() <= input.len() + input.len() / DEFAULT_BLOCK_LEN * 8 + 16);
    roundtrip(&input, DEFAULT_BLOCK_LEN)
}

#[test]
fn skewed_text_compresses() -> VortexResult<()> {
    let input: Vec<u8> = b"the quick brown fox jumps over the lazy dog "
        .iter()
        .cycle()
        .take(500_000)
        .copied()
        .collect();
    let compressed = compress(&input);
    assert!(compressed.len() < input.len() / 2 * 5 / 4);
    roundtrip(&input, DEFAULT_BLOCK_LEN)
}

#[rstest]
#[case(1)]
#[case(2)]
#[case(5)]
#[case(63)]
#[case(1024)]
#[case(65536)]
fn block_len_sweep(#[case] block_len: usize) -> VortexResult<()> {
    let mut rng = StdRng::seed_from_u64(1);
    let input: Vec<u8> = (0..70_001).map(|_| rng.random_range(b'a'..=b'z')).collect();
    roundtrip(&input, block_len)
}

#[test]
fn random_skewed_distributions() -> VortexResult<()> {
    let mut rng = StdRng::seed_from_u64(2);
    for _ in 0..20 {
        let alphabet = rng.random_range(2..=256usize);
        let len = rng.random_range(0..50_000usize);
        // Squaring skews the distribution towards low symbol values.
        let input: Vec<u8> = (0..len)
            .map(|_| {
                let uniform = rng.random_range(0.0..1.0f64);
                ((uniform * uniform * alphabet as f64) as usize).min(alphabet - 1) as u8
            })
            .collect();
        roundtrip(&input, DEFAULT_BLOCK_LEN)?;
    }
    Ok(())
}

#[test]
fn code_lengths_are_limited_and_kraft_valid() {
    let mut rng = StdRng::seed_from_u64(3);
    for _ in 0..50 {
        let mut hist = [0u64; 256];
        let alphabet = rng.random_range(2..=256usize);
        for slot in hist.iter_mut().take(alphabet) {
            *slot = rng.random_range(0..1_000_000);
        }
        if hist.iter().filter(|&&count| count > 0).count() < 2 {
            hist[0] = 1;
            hist[1] = 1;
        }
        let lens = build_code_lengths(&hist);
        let kraft: u64 = lens
            .iter()
            .filter(|&&len| len > 0)
            .map(|&len| 1u64 << (MAX_CODE_LEN - usize::from(len)))
            .sum();
        assert!(kraft <= 1 << MAX_CODE_LEN);
        for (sym, &len) in lens.iter().enumerate() {
            assert_eq!(hist[sym] > 0, len > 0, "symbol {sym}");
            assert!(usize::from(len) <= MAX_CODE_LEN);
        }
    }
}

#[test]
fn truncated_input_errors() {
    let compressed = compress(b"hello world hello world hello world");
    assert!(decompress(&compressed[..4]).is_err());
    assert!(decompress(&compressed[..compressed.len() - 1]).is_err());
    assert!(decompress(&[]).is_err());
}
