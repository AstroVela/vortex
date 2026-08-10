// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::io::Cursor;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use arrow_array::RecordBatch;
use arrow_schema::Schema;
use async_trait::async_trait;
use bytes::Bytes;
use parking_lot::Mutex;
use parquet::arrow::ArrowWriter;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use parquet::basic::ZstdLevel;
use parquet::file::properties::WriterProperties;
use vortex_bench::Format;
use vortex_bench::compress::Compressor;
use vortex_bench::compress::read_projection;

/// A Parquet file read into Arrow batches, keyed by the path it came from.
type DecodedInput = (PathBuf, Arc<Schema>, Vec<RecordBatch>);

/// Compressor implementation for Parquet format with ZSTD compression.
///
/// A compressor is built once per (dataset, format) and then driven through every compress
/// and decompress iteration, so the decoded batches and the compressed bytes are retained
/// here rather than rebuilt on each call. Neither is inside a timed region — rebuilding them
/// per iteration cost time without changing any reported number.
pub struct ParquetCompressor {
    compression: Compression,
    decoded: Mutex<Option<DecodedInput>>,
    compressed: Mutex<Option<(PathBuf, Bytes)>>,
}

impl ParquetCompressor {
    pub fn new() -> Self {
        Self::with_compression(Compression::ZSTD(ZstdLevel::default()))
    }

    pub fn with_compression(compression: Compression) -> Self {
        Self {
            compression,
            decoded: Mutex::new(None),
            compressed: Mutex::new(None),
        }
    }

    /// The Parquet input read into Arrow batches. Read on first use for a given path.
    ///
    /// `RecordBatch` is `Arc`-backed, so handing out a clone shares buffers rather than
    /// copying the data.
    fn decoded(&self, parquet_path: &Path) -> anyhow::Result<(Arc<Schema>, Vec<RecordBatch>)> {
        if let Some((path, schema, batches)) = self.decoded.lock().as_ref()
            && path == parquet_path
        {
            return Ok((Arc::clone(schema), batches.clone()));
        }
        let file = File::open(parquet_path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let schema = Arc::clone(builder.schema());
        let batches: Vec<RecordBatch> = builder.build()?.collect::<Result<Vec<_>, _>>()?;
        *self.decoded.lock() = Some((
            parquet_path.to_owned(),
            Arc::clone(&schema),
            batches.clone(),
        ));
        Ok((schema, batches))
    }

    /// The input compressed to a Parquet buffer, for `decompress` to read back.
    fn compressed(&self, parquet_path: &Path) -> anyhow::Result<Bytes> {
        if let Some((path, bytes)) = self.compressed.lock().as_ref()
            && path == parquet_path
        {
            return Ok(bytes.clone());
        }
        let (schema, batches) = self.decoded(parquet_path)?;
        let mut buf = Vec::new();
        parquet_compress_write(batches, schema, self.compression, &mut buf)?;
        let bytes = Bytes::from(buf);
        *self.compressed.lock() = Some((parquet_path.to_owned(), bytes.clone()));
        Ok(bytes)
    }
}

impl Default for ParquetCompressor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Compressor for ParquetCompressor {
    fn format(&self) -> Format {
        Format::Parquet
    }

    async fn compress(&self, parquet_path: &Path) -> anyhow::Result<(u64, Duration)> {
        let (schema, batches) = self.decoded(parquet_path)?;

        // Compress with our compression settings
        let mut buf = Vec::new();
        let start = Instant::now();
        let size = parquet_compress_write(batches, schema, self.compression, &mut buf)?;
        let elapsed = start.elapsed();
        Ok((size as u64, elapsed))
    }

    async fn decompress(&self, parquet_path: &Path) -> anyhow::Result<Duration> {
        let buf = self.compressed(parquet_path)?;

        // Now decompress
        let timer = Instant::now();
        parquet_decompress_read(buf)?;
        Ok(timer.elapsed())
    }
}

#[inline(never)]
pub fn parquet_compress_write(
    batches: Vec<RecordBatch>,
    schema: Arc<Schema>,
    compression: Compression,
    buf: &mut Vec<u8>,
) -> anyhow::Result<usize> {
    let mut buf = Cursor::new(buf);
    let writer_properties = WriterProperties::builder()
        .set_compression(compression)
        .build();
    let mut writer = ArrowWriter::try_new(&mut buf, schema, Some(writer_properties))?;
    for batch in batches {
        writer.write(&batch)?;
    }
    writer.flush()?;
    let n_bytes = writer.bytes_written();
    writer.close()?;
    Ok(n_bytes)
}

#[inline(never)]
pub fn parquet_decompress_read(buf: Bytes) -> anyhow::Result<usize> {
    let mut builder = ParquetRecordBatchReaderBuilder::try_new(buf)?;
    if let Some(cols) = read_projection(builder.schema().fields().len()) {
        // Project the given top-level (root) columns.
        let mask = ProjectionMask::roots(builder.parquet_schema(), cols.iter().copied());
        builder = builder.with_projection(mask);
    }
    let reader = builder.build()?;
    let mut nbytes = 0;
    for batch in reader {
        nbytes += batch?.get_array_memory_size()
    }

    Ok(nbytes)
}
