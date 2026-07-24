// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Real-world benchmark datasets, downloaded once and cached locally.
//!
//! - ClickBench URLs: the `URL` column of the first partition of the ClickBench `hits`
//!   dataset (real web-analytics URLs), newline-joined.
//! - Wikipedia: `enwik8`, the first 10^8 bytes of an English Wikipedia XML dump (the
//!   Hutter Prize / large-text-compression-benchmark corpus).
//!
//! Both corpora are truncated to [`CORPUS_LEN`] so benchmark iterations stay fast; the
//! per-block codec's ratio is insensitive to the truncation.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::fs;
use std::io::Read;
use std::path::PathBuf;

const ENWIK8_URL: &str = "https://mattmahoney.net/dc/enwik8.zip";
const HITS_URL: &str =
    "https://datasets.clickhouse.com/hits_compatible/athena_partitioned/hits_0.parquet";

/// Bytes of each corpus actually fed to the codecs.
pub const CORPUS_LEN: usize = 32 * 1024 * 1024;

fn cache_dir() -> PathBuf {
    let dir = std::env::var_os("VORTEX_HUFFMAN_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").expect("HOME not set"))
                .join(".cache")
                .join("vortex-huffman-bench")
        });
    fs::create_dir_all(&dir).expect("failed to create dataset cache dir");
    dir
}

fn download(url: &str, dest: &PathBuf) {
    if dest.exists() {
        return;
    }
    eprintln!("downloading {url} -> {}", dest.display());
    let response = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .expect("failed to build http client")
        .get(url)
        .send()
        .and_then(reqwest::blocking::Response::error_for_status)
        .unwrap_or_else(|e| panic!("failed to download {url}: {e}"));
    let bytes = response.bytes().expect("failed to read response body");
    let tmp = dest.with_extension("tmp");
    fs::write(&tmp, &bytes).expect("failed to write download");
    fs::rename(&tmp, dest).expect("failed to move download into place");
}

/// First [`CORPUS_LEN`] bytes of enwik8 (Wikipedia XML dump).
pub fn wikipedia() -> Vec<u8> {
    let cached = cache_dir().join("wikipedia.bin");
    if let Ok(data) = fs::read(&cached) {
        return data;
    }
    let zip_path = cache_dir().join("enwik8.zip");
    download(ENWIK8_URL, &zip_path);
    let file = fs::File::open(&zip_path).expect("failed to open enwik8.zip");
    let mut archive = zip::ZipArchive::new(file).expect("failed to open zip archive");
    let mut entry = archive.by_name("enwik8").expect("enwik8 not in archive");
    let mut data = vec![0u8; CORPUS_LEN];
    entry
        .read_exact(&mut data)
        .expect("enwik8 shorter than corpus length");
    fs::write(&cached, &data).expect("failed to cache wikipedia corpus");
    data
}

/// Newline-joined `URL` column values from the first ClickBench hits partition,
/// truncated to [`CORPUS_LEN`] bytes.
pub fn clickbench_urls() -> Vec<u8> {
    use arrow_array::Array;
    use arrow_array::cast::AsArray;
    use parquet::arrow::ProjectionMask;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let cached = cache_dir().join("clickbench_urls.bin");
    if let Ok(data) = fs::read(&cached) {
        return data;
    }
    let parquet_path = cache_dir().join("hits_0.parquet");
    download(HITS_URL, &parquet_path);

    let file = fs::File::open(&parquet_path).expect("failed to open hits parquet");
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).expect("invalid parquet");
    let url_idx = builder
        .schema()
        .fields()
        .iter()
        .position(|f| f.name() == "URL")
        .expect("no URL column in hits parquet");
    let mask = ProjectionMask::roots(builder.parquet_schema(), [url_idx]);
    let reader = builder
        .with_projection(mask)
        .build()
        .expect("failed to build parquet reader");

    let mut data = Vec::with_capacity(CORPUS_LEN + 4096);
    'outer: for batch in reader {
        let batch = batch.expect("failed to read record batch");
        // The reader may surface the column as Utf8, Utf8View, or Binary depending on
        // parquet metadata; normalize to BinaryView.
        let column = arrow_cast::cast(batch.column(0), &arrow_schema::DataType::BinaryView)
            .expect("failed to cast URL column");
        let urls = column.as_binary_view();
        for row in 0..urls.len() {
            data.extend_from_slice(urls.value(row));
            data.push(b'\n');
            if data.len() >= CORPUS_LEN {
                break 'outer;
            }
        }
    }
    assert!(
        data.len() >= CORPUS_LEN,
        "hits partition smaller than corpus length"
    );
    data.truncate(CORPUS_LEN);
    fs::write(&cached, &data).expect("failed to cache url corpus");
    data
}
