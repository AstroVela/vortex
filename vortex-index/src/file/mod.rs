// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Content-verified local Vortex files, available with the `file` feature.
//!
//! This initial adapter pins the complete encoded files in memory. It is intended
//! for bounded, frozen collections, not out-of-core ANN. With the `snapshot`
//! feature, `VisibleSource` can apply table-owned deletion/version visibility.
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

use async_trait::async_trait;
use futures::stream::BoxStream;
#[cfg(unix)]
use rustix::fs as unix_fs;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_file::OpenOptionsSessionExt;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;

pub(crate) mod source;

use self::source::FileSource;
use crate::IndexSource;
use crate::RowAddress;
use crate::Snapshot;
use crate::SourceBatch;
use crate::checksum::sha256;

/// SHA-256 identity of the complete encoded file, including footer and metadata.
pub fn file_version(bytes: &[u8]) -> String {
    sha256(bytes)
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
    source: FileSource,
    pinned_bytes: usize,
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
    /// Files may omit their embedded dtype; when present it must match `dtype`.
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
                    // Inspect the opened descriptor, but do not wait for a FIFO writer.
                    #[cfg(unix)]
                    let opened_file = unix_fs::open(
                        entry.uri.as_str(),
                        unix_fs::OFlags::RDONLY
                            | unix_fs::OFlags::NONBLOCK
                            | unix_fs::OFlags::NOCTTY
                            | unix_fs::OFlags::CLOEXEC,
                        unix_fs::Mode::empty(),
                    )
                    .map(File::from);
                    #[cfg(not(unix))]
                    let opened_file = File::open(&entry.uri);
                    let file = opened_file.map_err(|err| {
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
                        .with_fallback_dtype(dtype.clone())
                        .open_buffer(ByteBuffer::from(bytes))?;
                    if opened.dtype() != &dtype || opened.row_count() != entry.row_count {
                        vortex_bail!("Source file {} schema or row count mismatch", entry.id);
                    }
                    files.insert(entry.id, opened);
                }
                Ok(Self {
                    source: FileSource {
                        snapshot: Arc::new(snapshot),
                        dtype,
                        files,
                    },
                    pinned_bytes: max_pinned_bytes - remaining,
                })
            })
            .await
    }

    /// The verified, unprojected dtype shared by all files.
    pub fn dtype(&self) -> &DType {
        &self.source.dtype
    }

    /// Total encoded bytes retained by this source, excluding decoded data and metadata.
    pub fn pinned_bytes(&self) -> usize {
        self.pinned_bytes
    }
}

#[async_trait]
impl IndexSource for LocalFileSource {
    fn snapshot(&self) -> &Snapshot {
        self.source.snapshot()
    }

    fn scan(
        &self,
        files: &[u64],
        fields: &[String],
    ) -> VortexResult<BoxStream<'static, VortexResult<SourceBatch>>> {
        self.source.scan(files, fields)
    }

    async fn take(&self, rows: &[RowAddress], fields: &[String]) -> VortexResult<SourceBatch> {
        self.source.take(rows, fields).await
    }
}

#[cfg(test)]
pub(crate) mod tests;
