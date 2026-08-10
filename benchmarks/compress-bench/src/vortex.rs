// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io::Cursor;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use futures::pin_mut;
use parking_lot::Mutex;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::dtype::FieldNames;
use vortex::expr::root;
use vortex::expr::select;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex_arrow::ToArrowType;
use vortex_bench::Format;
use vortex_bench::SESSION;
use vortex_bench::compress::Compressor;
use vortex_bench::compress::read_projection;
use vortex_bench::conversions::parquet_to_vortex_chunks;

/// Compressor implementation for Vortex format.
///
/// A compressor is built once per (dataset, format) and then driven through every compress
/// and decompress iteration, so the decoded input and the compressed bytes are retained
/// here rather than rebuilt on each call. Neither is inside a timed region — rebuilding
/// them per iteration cost time without changing any reported number.
#[derive(Default)]
pub struct VortexCompressor {
    decoded: Mutex<Option<(PathBuf, ArrayRef)>>,
    compressed: Mutex<Option<(PathBuf, Bytes)>>,
}

impl VortexCompressor {
    /// The Parquet input decoded to Vortex arrays. Decoded on first use for a given path.
    async fn decoded(&self, parquet_path: &Path) -> Result<ArrayRef> {
        if let Some((path, array)) = self.decoded.lock().as_ref()
            && path == parquet_path
        {
            return Ok(array.clone());
        }
        let array = parquet_to_vortex_chunks(parquet_path.to_path_buf())
            .await?
            .into_array();
        *self.decoded.lock() = Some((parquet_path.to_owned(), array.clone()));
        Ok(array)
    }

    /// The input compressed to a Vortex buffer, for `decompress` to read back.
    async fn compressed(&self, parquet_path: &Path) -> Result<Bytes> {
        if let Some((path, bytes)) = self.compressed.lock().as_ref()
            && path == parquet_path
        {
            return Ok(bytes.clone());
        }
        let uncompressed = self.decoded(parquet_path).await?;
        let mut buf = Vec::new();
        let mut cursor = Cursor::new(&mut buf);
        SESSION
            .write_options()
            .write(&mut cursor, uncompressed.to_array_stream())
            .await?;
        let bytes = Bytes::from(buf);
        *self.compressed.lock() = Some((parquet_path.to_owned(), bytes.clone()));
        Ok(bytes)
    }
}

#[async_trait]
impl Compressor for VortexCompressor {
    fn format(&self) -> Format {
        Format::OnDiskVortex
    }

    async fn compress(&self, parquet_path: &Path) -> Result<(u64, Duration)> {
        let uncompressed = self.decoded(parquet_path).await?;

        let mut buf = Vec::new();
        let start = Instant::now();
        let mut cursor = Cursor::new(&mut buf);
        SESSION
            .write_options()
            .write(&mut cursor, uncompressed.to_array_stream())
            .await?;
        let elapsed = start.elapsed();

        Ok((buf.len() as u64, elapsed))
    }

    async fn decompress(&self, parquet_path: &Path) -> Result<Duration> {
        let data = self.compressed(parquet_path).await?;

        // Now decompress
        let start = Instant::now();
        let mut scan = SESSION.open_options().open_buffer(data)?.scan()?;
        let source_dtype = scan.dtype()?;
        let root_columns = source_dtype
            .as_struct_fields_opt()
            .map_or(0, |fields| fields.nfields());
        if let Some(cols) = read_projection(root_columns) {
            // Columns are named "0".."num_columns-1"; project the given subset.
            let names: FieldNames = cols.iter().map(|i| i.to_string()).collect();
            let projection = select(names, root())
                .optimize_recursive(&source_dtype)?
                .bind(&source_dtype)?;
            scan = scan.with_projection(projection);
        }
        let schema = Arc::new(scan.dtype()?.to_arrow_schema()?);

        let stream = scan.into_record_batch_stream(schema)?;
        pin_mut!(stream);

        while let Some(batch) = stream.next().await {
            let _batch = batch?;
        }
        Ok(start.elapsed())
    }
}
