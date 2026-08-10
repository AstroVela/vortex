// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter::once;
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::PrimitiveArray;
use arrow_array::types::Int64Type;
use arrow_select::concat::concat_batches;
use arrow_select::take::take_record_batch;
use async_trait::async_trait;
use futures::stream;
use itertools::Itertools;
use parking_lot::Mutex;
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use parquet::arrow::arrow_reader::ArrowReaderMetadata;
use parquet::arrow::arrow_reader::ArrowReaderOptions;
use parquet::file::metadata::PageIndexPolicy;
use stream::StreamExt;
use tokio::fs::File;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::stream::ArrayStreamExt;
use vortex::buffer::Buffer;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::VortexFile;
use vortex::layout::scan::repeated_scan::RepeatedScan;
use vortex::scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex::utils::aliases::hash_map::HashMap;

use crate::Format;
use crate::SESSION;
use crate::random_access::RandomAccessor;
use crate::random_access::RandomAccessorRet;

/// A scan prepared for a fixed set of row indices.
struct PreparedTake {
    indices: Arc<[u64]>,
    scan: Arc<RepeatedScan<ArrayRef>>,
}

/// Random accessor for Vortex format files.
///
/// The file handle is opened at construction time and reused across `take()` calls, as is the
/// [`RepeatedScan`] prepared by the first `take()`.
pub struct VortexRandomAccessor {
    name: String,
    format: Format,
    file: VortexFile,
    /// Scan prepared by the first `take()` and reused while the indices are unchanged.
    ///
    /// A prepared scan pins its selection, and with row indices the split ranges are derived
    /// from that selection, so a new index set requires preparing again.
    prepared: Mutex<Option<PreparedTake>>,
}

impl VortexRandomAccessor {
    /// Open a Vortex file and return a ready-to-use accessor.
    pub async fn open(
        path: impl AsRef<std::path::Path>,
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
            prepared: Mutex::new(None),
        })
    }

    /// Return the prepared scan for `indices`, preparing one if the cache does not hold it.
    fn prepared_scan(&self, indices: &[u64]) -> anyhow::Result<Arc<RepeatedScan<ArrayRef>>> {
        let mut prepared = self.prepared.lock();

        if let Some(prepared) = prepared.as_ref()
            && prepared.indices.as_ref() == indices
        {
            return Ok(Arc::clone(&prepared.scan));
        }

        let indices_buf: Buffer<u64> = Buffer::from(indices.to_vec());
        let scan = Arc::new(
            self.file
                .scan()?
                .with_row_indices(StrictSortedBuffer::try_new(indices_buf)?)
                .prepare()?,
        );
        *prepared = Some(PreparedTake {
            indices: Arc::from(indices),
            scan: Arc::clone(&scan),
        });

        Ok(scan)
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
        let array = self
            .prepared_scan(indices)?
            .execute_array_stream(None)?
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
    /// Path to the Parquet file (for re-opening on each take).
    path: PathBuf,
}

impl ParquetRandomAccessor {
    /// Open a Parquet file, parse the footer, and return a ready-to-use accessor.
    pub async fn open(path: PathBuf, name: impl Into<String>) -> anyhow::Result<Self> {
        let mut file = File::open(&path).await?;
        let options = ArrowReaderOptions::new().with_page_index_policy(PageIndexPolicy::Required);
        let arrow_metadata = ArrowReaderMetadata::load_async(&mut file, options).await?;

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

        Ok(Self {
            name: name.into(),
            row_group_offsets,
            arrow_metadata,
            path,
        })
    }
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
        let file = File::open(&self.path).await?;
        let builder =
            ParquetRecordBatchStreamBuilder::new_with_metadata(file, self.arrow_metadata.clone());

        let reader = builder
            .with_row_groups(sorted_row_group_keys)
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
}

#[cfg(test)]
mod tests {
    use vortex::array::IntoArray as _;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::stream::ArrayStreamAdapter;
    use vortex::file::WriteOptionsSessionExt;

    use super::*;

    /// Write `0..count` as a single-column Vortex file and return its path.
    async fn write_vortex_file(dir: &tempfile::TempDir, count: i32) -> anyhow::Result<PathBuf> {
        let array = Buffer::from((0..count).collect::<Vec<i32>>()).into_array();
        let mut buf = Vec::new();
        SESSION
            .write_options()
            .write(
                &mut buf,
                ArrayStreamAdapter::new(array.dtype().clone(), stream::iter([Ok(array)])),
            )
            .await?;

        let path = dir.path().join("data.vortex");
        std::fs::write(&path, buf)?;
        Ok(path)
    }

    async fn take_values(
        accessor: &VortexRandomAccessor,
        indices: &[u64],
    ) -> anyhow::Result<Vec<i32>> {
        let RandomAccessorRet::ArrayRef(array) = accessor.take(indices).await? else {
            anyhow::bail!("expected an ArrayRef from the Vortex accessor")
        };
        let mut ctx = SESSION.create_execution_ctx();
        let primitive = array.execute::<PrimitiveArray>(&mut ctx)?;
        Ok(primitive.as_slice::<i32>().to_vec())
    }

    #[tokio::test]
    async fn prepared_scan_is_reused_for_repeated_indices() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let path = write_vortex_file(&dir, 1_000).await?;
        let accessor = VortexRandomAccessor::open(&path, "test", Format::OnDiskVortex).await?;

        let indices = [3u64, 7, 42];
        let first = accessor.prepared_scan(&indices)?;
        let second = accessor.prepared_scan(&indices)?;
        assert!(
            Arc::ptr_eq(&first, &second),
            "repeated takes must reuse the prepared scan"
        );

        // The reused scan must keep returning the rows it was prepared for.
        assert_eq!(take_values(&accessor, &indices).await?, vec![3, 7, 42]);
        assert_eq!(take_values(&accessor, &indices).await?, vec![3, 7, 42]);

        Ok(())
    }

    #[tokio::test]
    async fn changed_indices_prepare_a_new_scan() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let path = write_vortex_file(&dir, 1_000).await?;
        let accessor = VortexRandomAccessor::open(&path, "test", Format::OnDiskVortex).await?;

        let first = accessor.prepared_scan(&[3u64, 7, 42])?;
        let second = accessor.prepared_scan(&[4u64, 8, 43])?;
        assert!(
            !Arc::ptr_eq(&first, &second),
            "a new index set must prepare a new scan"
        );

        // A prepared scan pins its selection, so the second index set must not
        // return the rows cached for the first.
        assert_eq!(
            take_values(&accessor, &[4u64, 8, 43]).await?,
            vec![4, 8, 43]
        );
        assert_eq!(
            take_values(&accessor, &[3u64, 7, 42]).await?,
            vec![3, 7, 42]
        );

        Ok(())
    }
}
