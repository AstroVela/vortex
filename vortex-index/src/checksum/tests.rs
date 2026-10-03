// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use base16ct::HexDisplay;
use rstest::rstest;
use sha2::Digest;
use sha2::Sha256 as LegacySha256;

use super::Sha256;
use super::sha256;

#[rstest]
#[case(
    b"",
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)]
#[case(
    b"abc",
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
)]
fn test_known_sha256_vectors(#[case] bytes: &[u8], #[case] expected: &str) {
    assert_eq!(sha256(bytes), format!("sha256:{expected}"));
}

#[rstest]
fn test_checksum_compatibility_across_padding_and_io_boundaries(
    #[values(0, 1, 55, 56, 63, 64, 65, 127, 128, 65_535, 65_536, 65_537, 131_089)] len: usize,
    #[values(1, 7, 64, 65_536)] chunk_size: usize,
) {
    let bytes = (0_u8..=250).cycle().take(len).collect::<Vec<_>>();
    let expected = format!("sha256:{:x}", HexDisplay(&LegacySha256::digest(&bytes)));
    let mut streamed = Sha256::new();
    streamed.update(&[]);
    for chunk in bytes.chunks(chunk_size) {
        streamed.update(chunk);
    }
    streamed.update(&[]);
    assert_eq!(sha256(&bytes), expected);
    assert_eq!(streamed.finalize(), expected);
}
