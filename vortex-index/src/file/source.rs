// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::sync::Arc;

use async_stream::try_stream;
use async_trait::async_trait;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::stream::BoxStream;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::root;
use vortex_array::expr::select;
use vortex_array::stream::ArrayStreamExt;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_file::VortexFile;
use vortex_scan::strict_sorted_buffer::StrictSortedBuffer;

use crate::IndexSource;
use crate::RowAddress;
use crate::Snapshot;
use crate::SourceBatch;

pub(crate) struct FileSource {
    pub(crate) snapshot: Arc<Snapshot>,
    pub(crate) dtype: DType,
    pub(crate) files: BTreeMap<u64, VortexFile>,
}

impl FileSource {
    fn projection(&self, fields: &[String]) -> VortexResult<BoundExpression> {
        let mut seen = BTreeSet::new();
        for field in fields {
            if !seen.insert(field) {
                vortex_bail!("Duplicate source projection field: {}", field);
            }
        }
        select(
            FieldNames::from_iter(fields.iter().map(String::as_str)),
            root(),
        )
        .bind(&self.dtype)
    }
}

#[async_trait]
impl IndexSource for FileSource {
    fn snapshot(&self) -> &Snapshot {
        &self.snapshot
    }

    /// Scan files in request order and rows in physical order within each file.
    ///
    /// File IDs must be distinct and present in the snapshot. Projection preserves
    /// requested field order; duplicate or unknown fields are errors. An empty
    /// projection returns zero-column structs with aligned row addresses.
    fn scan(
        &self,
        files: &[u64],
        fields: &[String],
    ) -> VortexResult<BoxStream<'static, VortexResult<SourceBatch>>> {
        let projection = self.projection(fields)?;
        let mut seen = BTreeSet::new();
        let inputs = files
            .iter()
            .map(|id| {
                if !seen.insert(*id) {
                    vortex_bail!("Duplicate source scan file: {}", id);
                }
                let file = self
                    .files
                    .get(id)
                    .ok_or_else(|| vortex_err!("Unknown source scan file: {}", id))?;
                Ok((*id, file.clone()))
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let snapshot = Arc::clone(&self.snapshot);
        Ok(try_stream! {
            for (file_id, file) in inputs {
                let mut offset = 0_u64;
                let mut stream = file
                    .scan()?
                    .with_projection(projection.clone())
                    .with_ordered(true)
                    .into_stream()?
                    .boxed();
                while let Some(data) = stream.try_next().await? {
                    let end = offset.checked_add(data.len() as u64)
                        .ok_or_else(|| vortex_err!("Source scan row offset overflow"))?;
                    if end > file.row_count() {
                        Err(vortex_err!("Source scan returned excess rows for file {}", file_id))?;
                    }
                    let rows = (offset..end)
                        .map(|row_offset| RowAddress { file_id, row_offset })
                        .collect();
                    offset = end;
                    yield SourceBatch::try_new(&snapshot, rows, data)?;
                }
                if offset != file.row_count() {
                    Err(vortex_err!("Source scan returned too few rows for file {}", file_id))?;
                }
            }
        }
        .boxed())
    }

    async fn take(&self, rows: &[RowAddress], fields: &[String]) -> VortexResult<SourceBatch> {
        let projection = self.projection(fields)?;
        let mut grouped = BTreeMap::<u64, Vec<u64>>::new();
        for row in rows {
            self.snapshot.validate_row(*row)?;
            grouped.entry(row.file_id).or_default().push(row.row_offset);
        }

        let mut chunks = Vec::with_capacity(grouped.len());
        let mut read_order = Vec::new();
        for (file_id, mut offsets) in grouped {
            offsets.sort_unstable();
            offsets.dedup();
            let count = offsets.len();
            read_order.extend(offsets.iter().map(|row_offset| RowAddress {
                file_id,
                row_offset: *row_offset,
            }));
            let file = self
                .files
                .get(&file_id)
                .ok_or_else(|| vortex_err!("Unknown source take file: {}", file_id))?;
            let data = file
                .scan()?
                .with_projection(projection.clone())
                .with_row_indices(StrictSortedBuffer::try_new(Buffer::from(offsets))?)
                .with_ordered(true)
                .into_array_stream()?
                .read_all()
                .await?;
            if data.len() != count {
                vortex_bail!(
                    "Source take returned an unexpected row count for file {}",
                    file_id
                );
            }
            chunks.push(data);
        }

        // IO uses sorted unique file-local selections. Restore caller order and
        // multiplicity with one gather over the concatenated selected batches.
        let positions = rows
            .iter()
            .map(|row| {
                read_order
                    .binary_search(row)
                    .map(|idx| idx as u64)
                    .map_err(|_| vortex_err!("Source take could not resolve row {:?}", row))
            })
            .collect::<VortexResult<Buffer<u64>>>()?;
        let data = ChunkedArray::try_new(chunks, projection.dtype().clone())?
            .into_array()
            .take(positions.into_array())?;
        SourceBatch::try_new(&self.snapshot, rows.to_vec(), data)
    }
}
