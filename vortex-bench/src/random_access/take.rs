// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter::once;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::PrimitiveArray;
use arrow_array::types::Int64Type;
use arrow_select::concat::concat_batches;
use arrow_select::take::take_record_batch;
use async_trait::async_trait;
use futures::stream;
use itertools::Itertools;
use object_store::path::Path as ObjectStorePath;
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use parquet::arrow::arrow_reader::ArrowReaderMetadata;
use parquet::arrow::arrow_reader::ArrowReaderOptions;
use parquet::arrow::async_reader::AsyncFileReader;
use parquet::arrow::async_reader::ParquetObjectReader;
use parquet::file::metadata::PageIndexPolicy;
use stream::StreamExt;
use tokio::fs::File;
use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::stream::ArrayStreamExt;
use vortex::buffer::Buffer;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::VortexFile;
use vortex::scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex::utils::aliases::hash_map::HashMap;

use crate::Format;
use crate::SESSION;
use crate::random_access::RandomAccessor;
use crate::random_access::RandomAccessorRet;
use crate::random_access::RemoteDataDir;

/// Random accessor for Vortex format files.
///
/// The file handle is opened at construction time and reused across `take()` calls.
pub struct VortexRandomAccessor {
    name: String,
    format: Format,
    file: VortexFile,
}

impl VortexRandomAccessor {
    /// Open a Vortex file and return a ready-to-use accessor.
    pub async fn open(
        path: impl AsRef<Path>,
        name: impl Into<String>,
        format: Format,
    ) -> anyhow::Result<Self> {
        let file = SESSION
            .open_options()
            .with_layout_reader_cache()
            .open_path(path.as_ref())
            .await?;
        Ok(Self {
            name: name.into(),
            format,
            file,
        })
    }

    /// Open a Vortex file stored in an object store and return a ready-to-use accessor.
    pub async fn open_object_store(
        remote: &RemoteDataDir,
        path: &Path,
        name: impl Into<String>,
        format: Format,
    ) -> anyhow::Result<Self> {
        let file = SESSION
            .open_options()
            .with_layout_reader_cache()
            .open_object_store(remote.store(), &remote.key(path)?)
            .await?;
        Ok(Self {
            name: name.into(),
            format,
            file,
        })
    }
}

#[async_trait]
impl RandomAccessor for VortexRandomAccessor {
    fn format(&self) -> Format {
        self.format
    }

    fn name(&self) -> &str {
        &self.name
    }

    async fn take(&self, indices: &[u64]) -> anyhow::Result<RandomAccessorRet> {
        let indices_buf: Buffer<u64> = Buffer::from(indices.to_vec());
        let array = self
            .file
            .scan()?
            .with_row_indices(StrictSortedBuffer::try_new(indices_buf)?)
            .into_array_stream()?
            .read_all()
            .await?;

        // We canonicalize / decompress for equivalence to Arrow's `RecordBatch`es.
        let mut ctx = SESSION.create_execution_ctx();
        let canonical = array.execute::<Canonical>(&mut ctx)?.into_array();
        Ok(RandomAccessorRet::ArrayRef(canonical))
    }
}

/// Random accessor for Parquet format files.
///
/// Parquet footer and row group offsets are parsed at construction time and
/// reused to map indices to row groups in each `take()` call.
pub struct ParquetRandomAccessor {
    name: String,
    /// Cumulative row offsets per row group (length = num_row_groups + 1).
    row_group_offsets: Vec<i64>,
    /// Cached Arrow reader metadata (footer) to avoid re-parsing on each take.
    arrow_metadata: ArrowReaderMetadata,
    /// Where to re-open the file from on each take.
    source: ParquetSource,
}

/// Backing store of a [`ParquetRandomAccessor`].
enum ParquetSource {
    /// Path to a local Parquet file.
    Local(PathBuf),
    /// Reader for a Parquet file held in an object store.
    Object(ParquetObjectReader),
}

impl ParquetRandomAccessor {
    /// Open a Parquet file, parse the footer, and return a ready-to-use accessor.
    pub async fn open(path: PathBuf, name: impl Into<String>) -> anyhow::Result<Self> {
        let mut file = File::open(&path).await?;
        let arrow_metadata = load_metadata(&mut file).await?;
        Ok(Self::new(name, arrow_metadata, ParquetSource::Local(path)))
    }

    /// Open a Parquet file stored in an object store and return a ready-to-use accessor.
    pub async fn open_object_store(
        remote: &RemoteDataDir,
        path: &Path,
        name: impl Into<String>,
    ) -> anyhow::Result<Self> {
        let mut reader = ParquetObjectReader::new(
            Arc::clone(remote.store()),
            ObjectStorePath::from(remote.key(path)?),
        );
        let arrow_metadata = load_metadata(&mut reader).await?;
        Ok(Self::new(
            name,
            arrow_metadata,
            ParquetSource::Object(reader),
        ))
    }

    fn new(
        name: impl Into<String>,
        arrow_metadata: ArrowReaderMetadata,
        source: ParquetSource,
    ) -> Self {
        let row_group_offsets = once(0)
            .chain(
                arrow_metadata
                    .metadata()
                    .row_groups()
                    .iter()
                    .map(|rg| rg.num_rows()),
            )
            .scan(0i64, |acc, x| {
                *acc += x;
                Some(*acc)
            })
            .collect::<Vec<_>>();

        Self {
            name: name.into(),
            row_group_offsets,
            arrow_metadata,
            source,
        }
    }
}

/// Parse the Parquet footer, including the page index, from any async reader.
async fn load_metadata<T: AsyncFileReader + Send>(
    reader: &mut T,
) -> anyhow::Result<ArrowReaderMetadata> {
    let options = ArrowReaderOptions::new().with_page_index_policy(PageIndexPolicy::Required);
    Ok(ArrowReaderMetadata::load_async(reader, options).await?)
}

#[async_trait]
impl RandomAccessor for ParquetRandomAccessor {
    fn format(&self) -> Format {
        Format::Parquet
    }

    fn name(&self) -> &str {
        &self.name
    }

    async fn take(&self, indices: &[u64]) -> anyhow::Result<RandomAccessorRet> {
        // Map indices to row groups.
        let mut row_groups = HashMap::new();
        for &idx in indices {
            let row_group_idx = self
                .row_group_offsets
                .binary_search(&(idx as i64))
                .unwrap_or_else(|e| e - 1);
            row_groups
                .entry(row_group_idx)
                .or_insert_with(Vec::new)
                .push((idx as i64) - self.row_group_offsets[row_group_idx]);
        }

        let sorted_row_group_keys = row_groups.keys().copied().sorted().collect_vec();
        let row_group_indices = sorted_row_group_keys
            .iter()
            .map(|i| row_groups[i].clone())
            .collect_vec();

        // Re-open the file but reuse cached metadata (avoids re-parsing the footer).
        match &self.source {
            ParquetSource::Local(path) => {
                let file = File::open(path).await?;
                take_row_groups(
                    file,
                    self.arrow_metadata.clone(),
                    sorted_row_group_keys,
                    &row_group_indices,
                )
                .await
            }
            ParquetSource::Object(reader) => {
                take_row_groups(
                    reader.clone(),
                    self.arrow_metadata.clone(),
                    sorted_row_group_keys,
                    &row_group_indices,
                )
                .await
            }
        }
    }
}

/// Read `row_groups` from `reader` and take `row_group_indices` within each of them.
async fn take_row_groups<T>(
    reader: T,
    metadata: ArrowReaderMetadata,
    row_groups: Vec<usize>,
    row_group_indices: &[Vec<i64>],
) -> anyhow::Result<RandomAccessorRet>
where
    T: AsyncFileReader + Unpin + Send + 'static,
{
    let builder = ParquetRecordBatchStreamBuilder::new_with_metadata(reader, metadata);

    let reader = builder
        .with_row_groups(row_groups)
        // FIXME(ngates): our indices code assumes the batch size == the row group sizes
        .with_batch_size(10_000_000)
        .build()?;

    let schema = Arc::clone(reader.schema());

    let batches = reader
        .enumerate()
        .map(|(idx, batch)| {
            let batch = batch.unwrap();
            let indices = PrimitiveArray::<Int64Type>::from(row_group_indices[idx].clone());
            take_record_batch(&batch, &indices).unwrap()
        })
        .collect::<Vec<_>>()
        .await;

    let result = concat_batches(&schema, &batches)?;
    Ok(RandomAccessorRet::RecordBatch(result))
}
