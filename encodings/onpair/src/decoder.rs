// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! The OnPair bulk decode loop, vendored from upstream `onpair` so the hot
//! path can be specialised per dictionary.
//!
//! Decoding is a gather-copy: each code names a dictionary token; the output
//! is those tokens concatenated. The general loop is the upstream
//! `onpair::try_decode_into` shape — a fixed 16-byte over-copy per token,
//! batched so an exactly-sized output buffer needs no per-store bounds check.
//!
//! On top of it sits a low-8 fast path ported from the CUDA
//! `onpair_shmem_4tpt_split8read` kernel. That kernel reads the common case
//! from an 8-byte-per-entry "first 8 bytes" table and touches the full
//! 16-byte rows only for the rare `len > 8` token, halving the hot dictionary
//! working set. A per-token `len > 8` branch does not survive contact with a
//! CPU branch predictor — trained dictionaries run 9–40% long tokens
//! dynamically, and at those rates the mispredictions cost more than the
//! smaller table saves — so the CPU port specialises per *dictionary*
//! instead: when every token is at most 8 bytes ([`ShortTokenDict`]), the
//! whole decode is one 1-byte length load, one 8-byte token load, and one
//! 8-byte over-store per token, with no offset indirection. Measured against
//! the 16-byte compact loop this wins 1.15–1.25x on uniform code streams over
//! 4K–64K-token dictionaries and more when the table is L1-resident; mixed
//! dictionaries keep the 16-byte loop, which benchmarks as fast as every
//! branchless hybrid tried.

use std::mem::MaybeUninit;

use onpair::CompactDictionaryView;
use onpair::DictionaryView;
use onpair::MAX_TOKEN_SIZE;
use onpair::OutputTooSmall;
use onpair::Token;
use vortex_error::VortexExpect;
use vortex_error::vortex_panic;

/// The malformed-code backstop, out of line so the decode loops carry only a
/// compare-and-branch: `vortex_panic!` expands error construction and
/// formatting inline, which measurably bloats the hot loop when written at
/// the panic site.
#[cold]
#[inline(never)]
fn code_out_of_range(code: usize, num_tokens: usize) -> ! {
    vortex_panic!("OnPair code {code} out of range for {num_tokens}-token dictionary")
}

/// An 8-byte-stride decode table for a dictionary whose tokens are all at
/// most 8 bytes — the CPU analogue of the CUDA kernel's `dict_s8` array.
///
/// Row `id` holds token `id`'s bytes zero-padded into a little-endian `u64`,
/// so a decode is a single independent load per token with no
/// `code → offset → bytes` indirection, and the table is half the size of the
/// 16-byte-stride wide dictionary.
#[derive(Debug, Clone)]
pub struct ShortTokenDict {
    /// Token bytes, zero-padded into one little-endian `u64` per token.
    rows: Vec<u64>,
    /// True token lengths, each in `1..=8`.
    lens: Vec<u8>,
}

impl ShortTokenDict {
    /// Build the table, or `None` if any token is longer than 8 bytes (the
    /// 16-byte general loop serves such dictionaries better — see the module
    /// docs).
    pub fn try_build(dict: CompactDictionaryView<'_>) -> Option<Self> {
        // Token ids fit `Token` by the dictionary's own bound of
        // `Token::MAX + 1` tokens.
        let ids = (0..dict.num_tokens()).map(|id| Token::try_from(id).vortex_expect("token id"));
        // Reject mixed dictionaries before allocating: one pass over the
        // offsets, which `validate_safety` has already walked once.
        if ids.clone().any(|id| dict.token_len(id) > 8) {
            return None;
        }
        let mut rows = vec![0u64; dict.num_tokens()];
        let mut lens = vec![0u8; dict.num_tokens()];
        for id in ids {
            let token = dict.token(id);
            let mut row = [0u8; 8];
            row[..token.len()].copy_from_slice(token);
            rows[id as usize] = u64::from_le_bytes(row);
            lens[id as usize] = u8::try_from(token.len()).vortex_expect("token len fits u8");
        }
        Some(Self { rows, lens })
    }
}

/// Decode `codes` against `dict` into `out`, returning the bytes written, or
/// `Err(`[`OutputTooSmall`]`)` if `out` cannot hold the result (in which case
/// nothing is written past the end of `out`). Never reads or writes out of
/// bounds; a buffer of exactly the decoded length suffices.
///
/// `short` routes to the low-8 loop and must be the table built from this
/// same `dict`; pass `None` for a dictionary with any token over 8 bytes.
///
/// # Panics
/// On a code that does not index the dictionary — malformed data is a panic
/// at the point of use, exactly like the upstream decoder this replaces.
pub fn try_decode_into(
    codes: &[u16],
    dict: CompactDictionaryView<'_>,
    short: Option<&ShortTokenDict>,
    out: &mut [MaybeUninit<u8>],
) -> Result<usize, OutputTooSmall> {
    match short {
        Some(short) => decode_short(codes, short, out),
        None => decode_general(codes, dict, out),
    }
}

/// The low-8 loop: an 8-byte over-store per token from [`ShortTokenDict`]
/// rows. Batching keeps the over-store within `out` without per-store checks:
/// each round issues `(out.len() - written) / 8` stores, and since every
/// token advances the cursor by at most 8 bytes, none can reach the end of
/// `out`; a sub-8-byte tail finishes with exact copies.
fn decode_short(
    codes: &[u16],
    short: &ShortTokenDict,
    out: &mut [MaybeUninit<u8>],
) -> Result<usize, OutputTooSmall> {
    let ntok = short.lens.len();
    let cap = out.len();
    let dst = out.as_mut_ptr().cast::<u8>();
    let mut written = 0usize;
    let mut consumed = 0usize;

    while consumed < codes.len() {
        let batch = (cap - written) / 8;
        if batch == 0 {
            break;
        }
        let end = (consumed + batch).min(codes.len());
        for &code in &codes[consumed..end] {
            let idx = code as usize;
            if idx >= ntok {
                code_out_of_range(idx, ntok);
            }
            // SAFETY: `idx < ntok` bounds both table reads. With `written0` the
            // cursor when this round took `batch = (cap - written0) / 8`, this
            // is store number `< batch`, and each preceding token advanced the
            // cursor by at most 8, so the 8-byte store ends at `<= written0 +
            // 8 * batch <= cap`, in bounds.
            unsafe {
                dst.add(written)
                    .cast::<u64>()
                    .write_unaligned(*short.rows.get_unchecked(idx));
                written += *short.lens.get_unchecked(idx) as usize;
            }
        }
        consumed = end;
    }

    // Tail: fewer than 8 bytes remain but codes are left. Exact copies (no
    // over-store), failing the moment a token would not fit.
    for &code in &codes[consumed..] {
        let idx = code as usize;
        if idx >= ntok {
            code_out_of_range(idx, ntok);
        }
        let len = short.lens[idx] as usize;
        if written + len > cap {
            return Err(OutputTooSmall);
        }
        let row = short.rows[idx].to_le_bytes();
        for (slot, &byte) in out[written..written + len].iter_mut().zip(&row[..len]) {
            slot.write(byte);
        }
        written += len;
    }
    Ok(written)
}

/// The general loop for dictionaries with tokens over 8 bytes: the upstream
/// `onpair::try_decode_into` shape, a fixed 16-byte over-copy per token
/// batched against `out` exactly as [`decode_short`] batches its 8-byte
/// stores.
fn decode_general(
    codes: &[u16],
    dict: CompactDictionaryView<'_>,
    out: &mut [MaybeUninit<u8>],
) -> Result<usize, OutputTooSmall> {
    let ntok = dict.num_tokens();
    let cap = out.len();
    let dst = out.as_mut_ptr().cast::<u8>();
    let mut written = 0usize;
    let mut consumed = 0usize;

    while consumed < codes.len() {
        let batch = (cap - written) / MAX_TOKEN_SIZE;
        if batch == 0 {
            break;
        }
        let end = (consumed + batch).min(codes.len());
        for &code in &codes[consumed..end] {
            if code as usize >= ntok {
                code_out_of_range(code as usize, ntok);
            }
            // SAFETY: `code < ntok`, so `token_ptr` is readable for
            // MAX_TOKEN_SIZE bytes (the dictionary blob is read-padded, per
            // `validate_safety`). The batch bound proves the 16-byte store
            // ends within `out`, by the same argument as `decode_short`.
            unsafe {
                let src = dict.token_ptr(code);
                dst.add(written)
                    .cast::<[u8; MAX_TOKEN_SIZE]>()
                    .write_unaligned(src.cast::<[u8; MAX_TOKEN_SIZE]>().read_unaligned());
                written += dict.token_len_unchecked(code);
            }
        }
        consumed = end;
    }

    // Tail: fewer than MAX_TOKEN_SIZE output bytes remain. Exact copies.
    for &code in &codes[consumed..] {
        if code as usize >= ntok {
            code_out_of_range(code as usize, ntok);
        }
        let token = dict.token(code);
        if written + token.len() > cap {
            return Err(OutputTooSmall);
        }
        for (slot, &byte) in out[written..written + token.len()].iter_mut().zip(token) {
            slot.write(byte);
        }
        written += token.len();
    }
    Ok(written)
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use onpair::MAX_TOKEN_SIZE;
    use vortex_error::VortexResult;

    use super::*;

    /// Read-padded compact `(bytes, offsets)` from tokens.
    fn storage(tokens: &[&[u8]]) -> (Vec<u8>, Vec<u32>) {
        let mut bytes = Vec::new();
        let mut offsets = vec![0u32];
        for token in tokens {
            bytes.extend_from_slice(token);
            offsets.push(bytes.len() as u32);
        }
        bytes.resize(bytes.len() + MAX_TOKEN_SIZE, 0);
        (bytes, offsets)
    }

    fn expected(tokens: &[&[u8]], codes: &[u16]) -> Vec<u8> {
        codes
            .iter()
            .flat_map(|&c| tokens[c as usize].iter().copied())
            .collect()
    }

    /// Decode through the vendored entry point into a buffer of `cap` bytes.
    fn decode(
        tokens: &[&[u8]],
        codes: &[u16],
        cap: usize,
    ) -> VortexResult<Result<Vec<u8>, OutputTooSmall>> {
        let (bytes, offsets) = storage(tokens);
        let dict = CompactDictionaryView::validate_safety(&bytes, &offsets)
            .map_err(|e| vortex_error::vortex_err!("invalid dictionary: {e}"))?;
        let short = ShortTokenDict::try_build(dict);
        let mut out = vec![MaybeUninit::uninit(); cap];
        Ok(
            try_decode_into(codes, dict, short.as_ref(), &mut out).map(|w| {
                out[..w]
                    .iter()
                    // SAFETY: `try_decode_into` initialised the first `w` bytes.
                    .map(|b| unsafe { b.assume_init() })
                    .collect()
            }),
        )
    }

    /// Round-trip on exact and padded buffers; asserts which loop was taken.
    fn check(tokens: &[&[u8]], codes: &[u16], expect_short: bool) -> VortexResult<()> {
        let (bytes, offsets) = storage(tokens);
        let dict = CompactDictionaryView::validate_safety(&bytes, &offsets)
            .map_err(|e| vortex_error::vortex_err!("invalid dictionary: {e}"))?;
        assert_eq!(ShortTokenDict::try_build(dict).is_some(), expect_short);

        let want = expected(tokens, codes);
        for cap in [want.len(), want.len() + MAX_TOKEN_SIZE] {
            let got = decode(tokens, codes, cap)?
                .unwrap_or_else(|e| panic!("decode failed at cap {cap}: {e}"));
            assert_eq!(got, want, "cap {cap}");
        }
        Ok(())
    }

    #[test]
    fn short_dict_round_trips_every_length_bucket() -> VortexResult<()> {
        let tokens: &[&[u8]] = &[b"a", b"bc", b"def", b"ghij", b"klmno", b"pqrstuvw"];
        let codes: Vec<u16> = (0..64).map(|i| (i % tokens.len()) as u16).collect();
        check(tokens, &codes, true)
    }

    #[test]
    fn short_dict_exact_buffer_takes_tail_per_token() -> VortexResult<()> {
        // Each single-token decode into an exactly-sized buffer runs entirely
        // through the exact-copy tail (`batch == 0` for len < 8).
        for len in 1..=8usize {
            let token = vec![b'a' + len as u8; len];
            let got = decode(&[token.as_slice()], &[0], len)?
                .unwrap_or_else(|e| panic!("len {len}: {e}"));
            assert_eq!(got, token, "len {len}");
        }
        Ok(())
    }

    #[test]
    fn mixed_dict_falls_back_to_general_loop() -> VortexResult<()> {
        let long = vec![b'z'; MAX_TOKEN_SIZE];
        let tokens: &[&[u8]] = &[b"a", b"bcdefghi", &long];
        let codes: Vec<u16> = (0..64).map(|i| (i % tokens.len()) as u16).collect();
        check(tokens, &codes, false)
    }

    #[test]
    fn eight_byte_tokens_stay_on_the_short_path() -> VortexResult<()> {
        // The boundary case: exactly 8 bytes is still "short"; 9 is not.
        let (bytes, offsets) = storage(&[b"12345678"]);
        let dict = CompactDictionaryView::validate_safety(&bytes, &offsets)
            .map_err(|e| vortex_error::vortex_err!("invalid dictionary: {e}"))?;
        assert!(ShortTokenDict::try_build(dict).is_some());

        let (bytes, offsets) = storage(&[b"123456789"]);
        let dict = CompactDictionaryView::validate_safety(&bytes, &offsets)
            .map_err(|e| vortex_error::vortex_err!("invalid dictionary: {e}"))?;
        assert!(ShortTokenDict::try_build(dict).is_none());
        Ok(())
    }

    #[test]
    fn rejects_buffer_one_byte_short() -> VortexResult<()> {
        // Short path.
        assert_eq!(decode(&[b"abcd"], &[0, 0], 7)?, Err(OutputTooSmall));
        // General path.
        let long = vec![b'z'; MAX_TOKEN_SIZE];
        let tokens: &[&[u8]] = &[b"abcd", &long];
        assert_eq!(decode(tokens, &[0, 1], 19)?, Err(OutputTooSmall));
        Ok(())
    }

    #[test]
    fn empty_codes_decode_to_empty() -> VortexResult<()> {
        assert_eq!(decode(&[b"ab"], &[], 0)?, Ok(Vec::new()));
        Ok(())
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn short_path_panics_on_out_of_range_code() {
        decode(&[b"ab"], &[0, 5], 64).unwrap().unwrap();
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn general_path_panics_on_out_of_range_code() {
        let long = vec![b'z'; MAX_TOKEN_SIZE];
        let tokens: &[&[u8]] = &[b"ab", &long];
        decode(tokens, &[0, 5], 64).unwrap().unwrap();
    }

    #[test]
    fn matches_upstream_decoder_on_both_paths() -> VortexResult<()> {
        let long = vec![b'y'; 12];
        let cases: &[&[&[u8]]] = &[&[b"a", b"bc", b"defgh"], &[b"a", b"bc", b"defgh", &long]];
        for tokens in cases {
            let (bytes, offsets) = storage(tokens);
            let dict = CompactDictionaryView::validate_safety(&bytes, &offsets)
                .map_err(|e| vortex_error::vortex_err!("invalid dictionary: {e}"))?;
            let codes: Vec<u16> = (0..64).map(|i| (i % tokens.len()) as u16).collect();

            let want_len = onpair::decoded_len(&codes, dict);
            let mut upstream = vec![MaybeUninit::uninit(); want_len];
            let upstream_written = onpair::try_decode_into(&codes, dict, &mut upstream)
                .unwrap_or_else(|e| panic!("upstream decode failed: {e}"));

            let got = decode(tokens, &codes, want_len)?
                .unwrap_or_else(|e| panic!("vendored decode failed: {e}"));
            assert_eq!(got.len(), upstream_written);
            // SAFETY: upstream initialised the first `upstream_written` bytes.
            let upstream_bytes = upstream[..upstream_written]
                .iter()
                .map(|b| unsafe { b.assume_init() })
                .collect::<Vec<_>>();
            assert_eq!(got, upstream_bytes);
        }
        Ok(())
    }
}
