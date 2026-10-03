// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use base16ct::HexDisplay;
use divan::Bencher;
use divan::black_box;
use divan::counter::BytesCount;
use sha2::Digest;
use sha2::Sha256;
use vortex_index::file::file_version;

const SIZES: &[usize] = &[64 * 1024, 1024 * 1024, 8 * 1024 * 1024];

fn main() {
    divan::main();
}

#[divan::bench(args = SIZES)]
fn current_checksum(bencher: Bencher, size: usize) {
    let bytes = vec![42; size];
    bencher
        .counter(BytesCount::new(size))
        .bench(|| file_version(black_box(&bytes)));
}

#[divan::bench(args = SIZES)]
fn legacy_checksum(bencher: Bencher, size: usize) {
    let bytes = vec![42; size];
    bencher.counter(BytesCount::new(size)).bench(|| {
        format!(
            "sha256:{:x}",
            HexDisplay(&Sha256::digest(black_box(&bytes)))
        )
    });
}
