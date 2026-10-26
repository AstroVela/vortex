// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Projected, version-pinned Vortex reads from a catalog-selected object store.
//!
//! The catalog supplies store-qualified locations in [`SourceFile::uri`] and
//! identities produced by [`object_version`]. Opening never adopts the latest version.
//! Every remote read requests and checks the expected version/ETag and size. An
//! ETag-only object that changes fails subsequent uncached reads; cached bytes
//! still belong to the original snapshot. Backends must implement strong object
//! version or conditional-read semantics. Mutable filesystem payloads are rejected.
//!
//! This adapter reads source data, not SPFresh's native local checkpoint files.
//!
//! ```no_run
//! use std::sync::Arc;
//! use object_store::ObjectStore;
//! use vortex_array::dtype::DType;
//! use vortex_error::VortexResult;
//! use vortex_index::{IndexSource, RowAddress, Snapshot};
//! use vortex_index::cloud::{CloudSourceOptions, ObjectStoreFileSource};
//! use vortex_session::VortexSession;
//!
//! # async fn example(snapshot: Snapshot, dtype: DType, store: Arc<dyn ObjectStore>,
//! #     session: VortexSession) -> VortexResult<()> {
//! // Catalog URIs begin with s3://bucket/, followed by literal keys in this store.
//! let source = ObjectStoreFileSource::open(
//!     snapshot, dtype, store, "s3://bucket/", session, CloudSourceOptions::default(),
//! ).await?;
//! let batch = source.take(
//!     &[RowAddress { file_id: 10, row_offset: 3 }], &["title".into()],
//! ).await?;
//! assert_eq!(batch.data.len(), 1);
//! # Ok(())
//! # }
//! ```

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::sync::Arc;

use async_trait::async_trait;
use futures::FutureExt;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use object_store::GetOptions;
use object_store::GetRange;
use object_store::GetResultPayload;
use object_store::ObjectMeta;
use object_store::ObjectStore;
use object_store::path::Path;
use serde::Deserialize;
use serde::Serialize;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_file::OpenOptionsSessionExt;
use vortex_io::CoalesceConfig;
use vortex_io::VortexReadAt;
use vortex_io::runtime::Handle;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::segments::MokaSegmentCache;
use vortex_session::VortexSession;

use crate::IndexSource;
use crate::RowAddress;
use crate::Snapshot;
use crate::SourceBatch;
use crate::SourceFile;
use crate::file::schema_fingerprint;
use crate::file::source::FileSource;

const VERSION_PREFIX: &str = "object-store-v1:";

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Identity {
    version: Option<String>,
    etag: Option<String>,
    size: u64,
}

impl Identity {
    fn validate(&self) -> VortexResult<()> {
        if self.version.as_ref().is_some_and(|value| {
            value.trim().is_empty() || value.contains(['\0', '\r', '\n']) || value == "null"
        }) || self.etag.as_ref().is_some_and(|value| {
            value.trim().is_empty()
                || value.contains(['\0', '\r', '\n', ','])
                || value.trim() == "*"
                || value.starts_with("W/")
        }) || (self.version.is_none() && self.etag.is_none())
        {
            vortex_bail!("Object identity requires an immutable version or a single strong ETag");
        }
        Ok(())
    }

    fn options(&self) -> GetOptions {
        GetOptions {
            version: self.version.clone(),
            if_match: self.etag.clone(),
            ..Default::default()
        }
    }

    fn check(&self, path: &Path, meta: &ObjectMeta) -> VortexResult<()> {
        if &meta.location != path
            || meta.size != self.size
            || self
                .version
                .as_ref()
                .is_some_and(|v| meta.version.as_ref() != Some(v))
            || self
                .etag
                .as_ref()
                .is_some_and(|v| meta.e_tag.as_ref() != Some(v))
        {
            vortex_bail!("Object response does not match the pinned identity: {path}");
        }
        Ok(())
    }
}

/// Encode catalog-observed object metadata as a [`SourceFile::version`].
///
/// Persist this alongside the store-qualified location and row count when publishing the
/// source snapshot. An ETag is an opaque conditional-read token, not a checksum.
/// S3's mutable `null` version is omitted; such objects require a strong ETag.
pub fn object_version(meta: &ObjectMeta) -> VortexResult<String> {
    let identity = Identity {
        version: meta.version.clone().filter(|value| value != "null"),
        etag: meta.e_tag.clone(),
        size: meta.size,
    };
    identity.validate()?;
    Ok(format!(
        "{VERSION_PREFIX}{}",
        serde_json::to_string(&identity).map_err(|err| vortex_err!("{err}"))?
    ))
}

/// Explicit source IO and cache budgets; these do not bound total process RSS.
#[derive(Clone, Copy, Debug)]
pub struct CloudSourceOptions {
    /// Maximum number of files whose metadata may be opened in one source.
    pub max_files: usize,
    /// Maximum bytes in an individual coalesced read, including footer reads.
    pub max_read_bytes: usize,
    /// Maximum concurrent range reads per file.
    pub concurrency: usize,
    /// Retained encoded segment budget per file, shared by queries on this source.
    /// Zero disables the additional cache. Footer metadata has separate ownership.
    pub cache_bytes_per_file: u64,
}

impl Default for CloudSourceOptions {
    fn default() -> Self {
        Self {
            max_files: 4096,
            max_read_bytes: 64 * 1024 * 1024,
            concurrency: 16,
            cache_bytes_per_file: 0,
        }
    }
}

/// A fixed snapshot of object-store Vortex files with lazy projected data reads.
///
/// One catalog-selected store owns all locations under the supplied URI prefix. All
/// physical rows are exposed; wrap this in `snapshot::VisibleSource` to apply the
/// table's visibility. Source handles and their per-file caches may be shared
/// between queries. Independent sources do not share segment IDs or caches.
pub struct ObjectStoreFileSource {
    source: FileSource,
}

impl ObjectStoreFileSource {
    /// Validate expected object identities, schemas and physical row counts.
    ///
    /// Opening reads footer ranges without eagerly downloading large files.
    /// Credentials and store resolution belong to the caller (for example `vortex-cloud`). The
    /// runtime must remain alive for the source's scans and ordered row retrieval.
    ///
    /// `uri_prefix` identifies this store's root, for example `s3://bucket/`. It
    /// must end in `/` and remain bound to the same store namespace. A source URI
    /// is this exact prefix followed by a literal object key, without URL decoding.
    /// Include any configured store prefix in this identity. Persisting qualified
    /// locations prevents reuse across stores whose opaque version tokens collide.
    pub async fn open(
        snapshot: Snapshot,
        dtype: DType,
        store: Arc<dyn ObjectStore>,
        uri_prefix: &str,
        session: VortexSession,
        options: CloudSourceOptions,
    ) -> VortexResult<Self> {
        snapshot.validate()?;
        validate_uri_prefix(uri_prefix)?;
        let DType::Struct(fields, Nullability::NonNullable) = &dtype else {
            vortex_bail!("Cloud sources require a non-nullable struct dtype");
        };
        if fields.names().iter().collect::<BTreeSet<_>>().len() != fields.nfields()
            || snapshot.schema_fingerprint != schema_fingerprint(&dtype)?
        {
            vortex_bail!("Cloud source schema fingerprint or field names are invalid");
        }
        if options.max_files == 0
            || snapshot.files.len() > options.max_files
            || options.max_read_bytes == 0
            || options.concurrency == 0
        {
            vortex_bail!("Invalid cloud source budgets");
        }
        let mut files = BTreeMap::new();
        for entry in &snapshot.files {
            let reader = PinnedReadAt::new(
                Arc::clone(&store),
                uri_prefix,
                entry,
                session.handle(),
                options,
            )?;
            let response = store
                .get_opts(
                    &reader.path,
                    GetOptions {
                        head: true,
                        ..reader.identity.options()
                    },
                )
                .await?;
            reader.identity.check(&reader.path, &response.meta)?;
            let mut open = session
                .open_options()
                .with_fallback_dtype(dtype.clone())
                .with_layout_reader_cache();
            if options.cache_bytes_per_file > 0 {
                open = open.with_segment_cache(Arc::new(MokaSegmentCache::new(
                    options.cache_bytes_per_file,
                )));
            }
            let opened = open.open(Arc::new(reader)).await?;
            if opened.dtype() != &dtype || opened.row_count() != entry.row_count {
                vortex_bail!("Cloud source {} schema or row count mismatch", entry.id);
            }
            files.insert(entry.id, opened);
        }
        Ok(Self {
            source: FileSource {
                snapshot: Arc::new(snapshot),
                dtype,
                files,
            },
        })
    }

    /// Unprojected dtype verified against every file in the pinned snapshot.
    pub fn dtype(&self) -> &DType {
        &self.source.dtype
    }
}

#[async_trait]
impl IndexSource for ObjectStoreFileSource {
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

struct PinnedReadAt {
    store: Arc<dyn ObjectStore>,
    path: Path,
    identity: Identity,
    handle: Handle,
    options: CloudSourceOptions,
}

fn validate_uri_prefix(prefix: &str) -> VortexResult<()> {
    let Some((scheme, root)) = prefix.split_once("://") else {
        vortex_bail!("Cloud source URI prefix must identify its store namespace");
    };
    if scheme.is_empty()
        || root.len() <= 1
        || root.starts_with('/')
        || !prefix.ends_with('/')
        || prefix.chars().any(char::is_control)
    {
        vortex_bail!("Invalid cloud source URI prefix");
    }
    Ok(())
}

impl PinnedReadAt {
    fn new(
        store: Arc<dyn ObjectStore>,
        uri_prefix: &str,
        file: &SourceFile,
        handle: Handle,
        options: CloudSourceOptions,
    ) -> VortexResult<Self> {
        let key = file.uri.strip_prefix(uri_prefix).ok_or_else(|| {
            vortex_err!("Source location is outside the selected store namespace")
        })?;
        let path = Path::parse(key).map_err(|err| vortex_err!("{err}"))?;
        if path.as_ref() != key || key.is_empty() || file.version.len() > 16 * 1024 {
            vortex_bail!(
                "Cloud sources require literal object keys and bounded version identities"
            );
        }
        let json = file
            .version
            .strip_prefix(VERSION_PREFIX)
            .ok_or_else(|| vortex_err!("Cloud source requires an object-store-v1 identity"))?;
        let identity: Identity = serde_json::from_str(json).map_err(|err| vortex_err!("{err}"))?;
        identity.validate()?;
        Ok(Self {
            store,
            path,
            identity,
            handle,
            options,
        })
    }
}

impl VortexReadAt for PinnedReadAt {
    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        Some(CoalesceConfig::new(
            0,
            self.options.max_read_bytes.min(16 * 1024 * 1024) as u64,
        ))
    }

    fn concurrency(&self) -> usize {
        self.options.concurrency
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let size = self.identity.size;
        async move { Ok(size) }.boxed()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let store = Arc::clone(&self.store);
        let path = self.path.clone();
        let identity = self.identity.clone();
        let max_read_bytes = self.options.max_read_bytes;
        self.handle
            .spawn_io(async move {
                let end = offset
                    .checked_add(length as u64)
                    .ok_or_else(|| vortex_err!("Cloud read range overflow"))?;
                if end > identity.size || length > max_read_bytes {
                    vortex_bail!("Cloud read exceeds object bounds or the range byte budget");
                }
                if length == 0 {
                    return Ok(BufferHandle::new_host(
                        ByteBuffer::empty().aligned(alignment),
                    ));
                }
                let range = offset..end;
                let response = store
                    .get_opts(
                        &path,
                        GetOptions {
                            range: Some(GetRange::Bounded(range.clone())),
                            ..identity.options()
                        },
                    )
                    .await?;
                identity.check(&path, &response.meta)?;
                if response.range != range {
                    vortex_bail!("Object store returned a different byte range: {path}");
                }
                #[allow(
                    unreachable_patterns,
                    reason = "File payloads exist when another crate enables object_store/fs"
                )]
                let mut stream = match response.payload {
                    GetResultPayload::Stream(stream) => stream,
                    _ => vortex_bail!(
                        "Cloud snapshot reads require immutable streamed object payloads"
                    ),
                };
                let mut bytes = Vec::new();
                bytes
                    .try_reserve_exact(length)
                    .map_err(|err| vortex_err!("Cloud read allocation: {err}"))?;
                while let Some(chunk) = stream.next().await {
                    let chunk = chunk?;
                    if chunk.len() > length - bytes.len() {
                        vortex_bail!("Object store returned excess range bytes: {path}");
                    }
                    bytes.extend_from_slice(&chunk);
                }
                if bytes.len() != length {
                    vortex_bail!("Object store returned a short range: {path}");
                }
                Ok(BufferHandle::new_host(
                    ByteBuffer::from(bytes).aligned(alignment),
                ))
            })
            .boxed()
    }
}

#[cfg(test)]
mod tests;
