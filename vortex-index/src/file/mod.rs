// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Content-verified local Vortex files, available with the `file` feature.
//!
//! This initial adapter pins the complete encoded files in memory. It is intended
//! for bounded, frozen collections without deletion masks, not out-of-core ANN.
//!
//! ```no_run
//! use vortex_array::dtype::DType;
//! use vortex_error::VortexResult;
//! use vortex_index::{IndexSource, RowAddress, Snapshot};
//! use vortex_index::file::LocalFileSource;
//! use vortex_session::VortexSession;
//!
//! # async fn example(snapshot: Snapshot, dtype: DType, session: VortexSession) -> VortexResult<()> {
//! // The catalog supplies the expected SHA-256 inventory and schema identity.
//! let source = LocalFileSource::open(snapshot, dtype, 64 * 1024 * 1024, session).await?;
//! let rows = [RowAddress { file_id: 10, row_offset: 3 }];
//! let batch = source.take(&rows, &["embedding".into()]).await?;
//! assert_eq!(batch.rows, rows);
//! # Ok(())
//! # }
//! ```

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

use async_stream::try_stream;
use async_trait::async_trait;
use base16ct::HexDisplay;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::stream::BoxStream;
use sha2::Digest;
use sha2::Sha256;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::root;
use vortex_array::expr::select;
use vortex_array::stream::ArrayStreamExt;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::VortexFile;
use vortex_io::session::RuntimeSessionExt;
use vortex_scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex_session::VortexSession;

use crate::IndexSource;
use crate::RowAddress;
use crate::Snapshot;
use crate::SourceBatch;

/// SHA-256 identity of the complete encoded file, including footer and metadata.
pub fn file_version(bytes: &[u8]) -> String {
    format!("sha256:{:x}", HexDisplay(&Sha256::digest(bytes)))
}

/// Fingerprint of the logical schema, including field order and nullability.
///
/// This experimental scheme hashes Vortex's compact DType JSON representation.
/// Its version tag must change if that serialization changes; it is not a stable
/// Vortex file-format identity or an external table's field-ID scheme.
pub fn schema_fingerprint(dtype: &DType) -> VortexResult<String> {
    let json = serde_json::to_vec(dtype).map_err(|err| vortex_err!("{}", err))?;
    Ok(format!("vortex-dtype-json-v1:{}", file_version(&json)))
}

/// A fixed collection of content-verified local Vortex files without tombstones.
///
/// Every physical row is visible. All files must have the same non-nullable
/// top-level struct dtype; individual fields may be nullable or nested.
/// `SourceFile::uri` is an absolute native filesystem path, not a URL.
/// Subsequent scans read only pinned bytes, so path replacement, deletion, or
/// in-place writes cannot change an already opened source.
pub struct LocalFileSource {
    snapshot: Arc<Snapshot>,
    dtype: DType,
    files: BTreeMap<u64, VortexFile>,
}

impl LocalFileSource {
    /// Verify the entire inventory and pin its encoded contents before returning.
    ///
    /// `max_pinned_bytes` bounds the sum of encoded file lengths, not total RSS:
    /// allocations, decoded batches and `take` results require additional memory.
    /// The caller supplies trusted expected identities, not versions inferred
    /// from whichever files happen to exist at open time. File versions must use
    /// [`file_version`] and the schema identity must use
    /// [`schema_fingerprint`]. Empty inventories still require an explicit dtype.
    /// The session must have file encodings/layouts and a live runtime configured.
    /// Blocking file reads run on that runtime's blocking executor.
    pub async fn open(
        snapshot: Snapshot,
        dtype: DType,
        max_pinned_bytes: usize,
        session: VortexSession,
    ) -> VortexResult<Self> {
        snapshot.validate()?;
        let DType::Struct(fields, Nullability::NonNullable) = &dtype else {
            vortex_bail!("Local file sources require a non-nullable struct dtype");
        };
        if fields.names().iter().collect::<BTreeSet<_>>().len() != fields.nfields() {
            vortex_bail!("Local file sources require unique top-level field names");
        }
        if snapshot.schema_fingerprint != schema_fingerprint(&dtype)? {
            vortex_bail!("Local file source schema fingerprint does not match its dtype");
        }
        for file in &snapshot.files {
            if !Path::new(&file.uri).is_absolute() {
                vortex_bail!("Local file source requires an absolute path: {}", file.uri);
            }
            if !file.version.strip_prefix("sha256:").is_some_and(|digest| {
                digest.len() == 64
                    && digest
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            }) {
                vortex_bail!(
                    "Local file source requires a lowercase SHA-256 version: {}",
                    file.id
                );
            }
        }

        session
            .handle()
            .spawn_blocking(move || {
                let mut remaining = max_pinned_bytes;
                let mut files = BTreeMap::new();
                for entry in &snapshot.files {
                    let file = File::open(&entry.uri).map_err(|err| {
                        vortex_err!("Cannot open source file {}: {}", entry.id, err)
                    })?;
                    let metadata = file.metadata()?;
                    if !metadata.is_file() {
                        vortex_bail!("Source {} is not a regular file", entry.id);
                    }
                    if metadata.len() > remaining as u64 {
                        vortex_bail!("Source {} exceeds the pinned byte budget", entry.id);
                    }
                    let mut bytes = Vec::new();
                    // Bound the read too: the file can grow after the metadata check.
                    file.take((remaining as u64).saturating_add(1))
                        .read_to_end(&mut bytes)?;
                    if bytes.len() > remaining {
                        vortex_bail!("Source {} exceeds the pinned byte budget", entry.id);
                    }
                    remaining -= bytes.len();
                    if file_version(&bytes) != entry.version {
                        vortex_bail!("Source file {} content version mismatch", entry.id);
                    }
                    let opened = session
                        .open_options()
                        .open_buffer(ByteBuffer::from(bytes))?;
                    if opened.dtype() != &dtype || opened.row_count() != entry.row_count {
                        vortex_bail!("Source file {} schema or row count mismatch", entry.id);
                    }
                    files.insert(entry.id, opened);
                }
                Ok(Self {
                    snapshot: Arc::new(snapshot),
                    dtype,
                    files,
                })
            })
            .await
    }

    /// The verified, unprojected dtype shared by all files.
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

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
impl IndexSource for LocalFileSource {
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

#[cfg(test)]
mod tests;
