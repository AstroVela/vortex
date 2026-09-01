// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(vortex_vane_distributed)]
use std::ops::Range;
use std::sync::Arc;
use std::sync::LazyLock;

#[cfg(vortex_vane_distributed)]
use async_trait::async_trait;
#[cfg(vortex_vane_distributed)]
use futures::TryStreamExt;
#[cfg(vortex_vane_distributed)]
use futures::future::BoxFuture;
use itertools::Itertools;
#[cfg(vortex_vane_distributed)]
use object_store::Error as ObjectStoreError;
#[cfg(vortex_vane_distributed)]
use object_store::ObjectMeta;
#[cfg(vortex_vane_distributed)]
use object_store::ObjectStore;
#[cfg(vortex_vane_distributed)]
use object_store::ObjectStoreExt;
#[cfg(vortex_vane_distributed)]
use object_store::local::LocalFileSystem;
#[cfg(vortex_vane_distributed)]
use object_store::path::Path as ObjectPath;
use object_store::registry::ObjectStoreRegistry;
use url::Url;
#[cfg(vortex_vane_distributed)]
use vortex::array::buffer::BufferHandle;
#[cfg(vortex_vane_distributed)]
use vortex::buffer::Alignment;
use vortex::cloud::Registry;
#[cfg(vortex_vane_distributed)]
use vortex::dtype::DType;
#[cfg(vortex_vane_distributed)]
use vortex::error::VortexError;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
#[cfg(vortex_vane_distributed)]
use vortex::file::VortexFile;
#[cfg(vortex_vane_distributed)]
use vortex::file::VortexOpenOptions;
#[cfg(not(vortex_vane_distributed))]
use vortex::file::multi::MultiFileDataSource;
#[cfg(vortex_vane_distributed)]
use vortex::file::multi::open_cached;
use vortex::file::multi::parse_uri_or_path;
#[cfg(vortex_vane_distributed)]
use vortex::io::CoalesceConfig;
#[cfg(vortex_vane_distributed)]
use vortex::io::VortexReadAt;
use vortex::io::compat::Compat;
use vortex::io::filesystem::FileSystemRef;
use vortex::io::object_store::ObjectStoreFileSystem;
#[cfg(vortex_vane_distributed)]
use vortex::io::object_store::ObjectStoreReadAt;
use vortex::io::runtime::BlockingRuntime;
#[cfg(vortex_vane_distributed)]
use vortex::layout::LayoutReaderRef;
#[cfg(vortex_vane_distributed)]
use vortex::layout::scan::multi::LayoutReaderFactory;
use vortex::layout::scan::multi::MultiLayoutDataSource;

use crate::RUNTIME;
use crate::SESSION;
#[cfg(vortex_vane_distributed)]
use crate::bound_object_store::BoundObjectStore;
use crate::duckdb::BindInputRef;
use crate::duckdb::ExtractedValue;

/// Process-wide registry, so repeated scans against the same bucket share one client.
static REGISTRY: LazyLock<Registry> = LazyLock::new(Registry::new);

/// One exact file selected by the coordinator bind. `source_url` identifies
/// the filesystem mount while `path` is the literal path inside that mount.
/// Keeping both avoids reconstructing ambiguous URLs for stores such as hf://.
#[cfg(vortex_vane_distributed)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundFile {
    pub source_url: String,
    pub path: String,
    pub size: u64,
    pub e_tag: Option<String>,
    pub version: Option<String>,
}

#[cfg(vortex_vane_distributed)]
pub struct BoundMultiFileScan {
    pub data_source: MultiLayoutDataSource,
    pub files: Vec<BoundFile>,
}

#[cfg(vortex_vane_distributed)]
struct ResolvedFileSystem {
    filesystem: FileSystemRef,
    store: Arc<dyn ObjectStore>,
    path: String,
}

#[cfg(not(vortex_vane_distributed))]
fn resolve_filesystem(glob_url: &Url) -> VortexResult<(FileSystemRef, String)> {
    // Compat makes us use tokio which is very bad for local reads on
    // high-core machines because reads go into blocking pool
    if glob_url.scheme() == "file" {
        let path = glob_url.path().to_string();
        return Ok((
            Arc::new(ObjectStoreFileSystem::local(RUNTIME.handle())),
            path,
        ));
    }

    // The full URL goes through the shared registry, which reports the glob as a path *within*
    // the store it returns. For most schemes the store is mounted at the URL authority, so the
    // path is the whole URL path — but not for all of them: an `hf://` store is rooted at a
    // repository and revision, which occupy path segments. Only the registry knows how deep the
    // store is mounted, so globbing anything other than the path it reports would address the
    // wrong keys. Going through the registry also means DuckDB resolves the same set of schemes
    // as the Python and Java bindings, including the OpenDAL-backed ones when the `opendal`
    // feature is on. The registry caches one client per store prefix, so repeated scans against
    // the same bucket or repository share a client even though the filesystem wrapper is rebuilt.
    let (object_store, path) = REGISTRY.resolve(glob_url)?;

    let path = path.to_string();

    Ok((
        Arc::new(ObjectStoreFileSystem::new(
            Arc::new(Compat::new(object_store)),
            RUNTIME.handle(),
        )),
        path,
    ))
}

#[cfg(vortex_vane_distributed)]
fn resolve_filesystem(glob_url: &Url) -> VortexResult<ResolvedFileSystem> {
    let (store, path): (Arc<dyn ObjectStore>, String) = if glob_url.scheme() == "file" {
        (
            Arc::new(LocalFileSystem::new()),
            glob_url.path().trim_start_matches('/').to_string(),
        )
    } else {
        let (store, path) = REGISTRY.resolve(glob_url)?;
        (
            Arc::new(Compat::new(store)),
            path.to_string().trim_start_matches('/').to_string(),
        )
    };
    let filesystem = Arc::new(ObjectStoreFileSystem::new(
        Arc::clone(&store),
        RUNTIME.handle(),
    ));
    Ok(ResolvedFileSystem {
        filesystem,
        store,
        path,
    })
}

#[cfg(vortex_vane_distributed)]
struct BoundReadAt {
    inner: ObjectStoreReadAt,
    uri: Arc<str>,
}

#[cfg(vortex_vane_distributed)]
impl BoundReadAt {
    fn new(file: &BoundFile, store: Arc<dyn ObjectStore>, path: ObjectPath) -> Self {
        Self {
            inner: ObjectStoreReadAt::new(store, path, RUNTIME.handle()),
            uri: Arc::from(format!(
                "vane-bound:{:?}",
                (&file.source_url, &file.path, &file.e_tag, &file.version)
            )),
        }
    }
}

#[cfg(vortex_vane_distributed)]
impl VortexReadAt for BoundReadAt {
    fn uri(&self) -> Option<&Arc<str>> {
        Some(&self.uri)
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        self.inner.coalesce_config()
    }

    fn concurrency(&self) -> usize {
        self.inner.concurrency()
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        self.inner.size()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        self.inner.read_at(offset, length, alignment)
    }
}

#[cfg(vortex_vane_distributed)]
pub(crate) fn validate_bound_file(file: &BoundFile) -> VortexResult<()> {
    let source_url = Url::parse(&file.source_url).map_err(|error| {
        vortex_err!(
            "Invalid bound Vortex source URL '{}': {error}",
            file.source_url
        )
    })?;
    if source_url.to_string() != file.source_url {
        vortex_bail!(
            "Bound Vortex source URL is not canonical: {}",
            file.source_url
        );
    }
    if file.path.is_empty()
        || file.path.starts_with('/')
        || file.path.ends_with('/')
        || file.path.contains('\0')
        || file
            .path
            .split('/')
            .any(|segment| segment.is_empty() || segment == "." || segment == "..")
    {
        vortex_bail!("Bound Vortex path is not canonical: {}", file.path);
    }
    if file.size == u64::MAX {
        vortex_bail!("Bound Vortex file has an invalid size: {}", file.path);
    }
    if file
        .e_tag
        .as_ref()
        .is_some_and(|e_tag| e_tag.is_empty() || e_tag.contains('\0'))
    {
        vortex_bail!("Bound Vortex file has an invalid ETag: {}", file.path);
    }
    if file
        .version
        .as_ref()
        .is_some_and(|version| version.is_empty() || version.contains('\0'))
    {
        vortex_bail!("Bound Vortex file has an invalid version: {}", file.path);
    }
    if file.e_tag.is_none() && file.version.is_none() {
        vortex_bail!(
            "Bound Vortex file has no immutable object identity: {}",
            file.path
        );
    }
    Ok(())
}

#[cfg(vortex_vane_distributed)]
fn object_path(path: &str) -> VortexResult<ObjectPath> {
    ObjectPath::parse(path)
        .map_err(|error| vortex_err!("Bound Vortex path is not an object key: {path}: {error}"))
}

#[cfg(vortex_vane_distributed)]
fn bind_object(source_url: &Url, meta: ObjectMeta) -> VortexResult<BoundFile> {
    let file = BoundFile {
        source_url: source_url.to_string(),
        path: meta.location.to_string(),
        size: meta.size,
        e_tag: meta.e_tag,
        version: meta.version,
    };
    validate_bound_file(&file)?;
    Ok(file)
}

#[cfg(vortex_vane_distributed)]
async fn verify_file(
    file: &BoundFile,
    store: &Arc<dyn ObjectStore>,
) -> VortexResult<(ObjectMeta, Arc<dyn ObjectStore>, ObjectPath)> {
    validate_bound_file(file)?;
    let path = object_path(&file.path)?;
    let bound_store = Arc::new(BoundObjectStore::new(
        Arc::clone(store),
        file.e_tag.clone(),
        file.version.clone(),
    )) as Arc<dyn ObjectStore>;
    let meta = match bound_store.head(&path).await {
        Ok(meta) => meta,
        Err(ObjectStoreError::NotFound { .. }) => {
            vortex_bail!("Bound Vortex file no longer exists: {}", file.path)
        }
        Err(error) => return Err(error.into()),
    };
    if meta.location != path {
        vortex_bail!(
            "Bound Vortex file identity changed: expected {}, got {}",
            file.path,
            meta.location
        );
    }
    if meta.size != file.size {
        vortex_bail!(
            "Bound Vortex file size changed for {}: expected {}, got {}",
            file.path,
            file.size,
            meta.size
        );
    }
    if file.e_tag.is_some() && meta.e_tag != file.e_tag {
        vortex_bail!(
            "Bound Vortex file ETag changed for {}: expected {:?}, got {:?}",
            file.path,
            file.e_tag,
            meta.e_tag
        );
    }
    if file.version.is_some() && meta.version != file.version {
        vortex_bail!(
            "Bound Vortex file version changed for {}: expected {:?}, got {:?}",
            file.path,
            file.version,
            meta.version
        );
    }
    Ok((meta, bound_store, path))
}

#[cfg(vortex_vane_distributed)]
pub(crate) async fn open_bound_file(file: &BoundFile) -> VortexResult<VortexFile> {
    let source_url = Url::parse(&file.source_url).map_err(|error| {
        vortex_err!(
            "Invalid bound Vortex source URL '{}': {error}",
            file.source_url
        )
    })?;
    let resolved = resolve_filesystem(&source_url)?;
    let (_meta, bound_store, path) = verify_file(file, &resolved.store).await?;
    let source = Arc::new(BoundReadAt::new(file, bound_store, path));
    open_cached(
        &SESSION,
        source,
        &file.path,
        Some(file.size),
        &|options: VortexOpenOptions| options,
    )
    .await
}

#[cfg(vortex_vane_distributed)]
struct BoundFileReaderFactory {
    file: BoundFile,
}

#[cfg(vortex_vane_distributed)]
#[async_trait]
impl LayoutReaderFactory for BoundFileReaderFactory {
    async fn open(&self) -> VortexResult<Option<LayoutReaderRef>> {
        Ok(Some(open_bound_file(&self.file).await?.layout_reader()?))
    }
}

/// Build a reader over immutable file fragments selected by a distributed worker assignment.
#[cfg(vortex_vane_distributed)]
pub fn build_bound_fragment_scan(
    files: &[BoundFile],
    row_ranges: &[Range<u64>],
    empty_dtype: Option<DType>,
) -> VortexResult<MultiLayoutDataSource> {
    if files.len() != row_ranges.len() {
        vortex_bail!(
            "Distributed Vortex fragment file count {} differs from row-range count {}",
            files.len(),
            row_ranges.len()
        );
    }
    let dtype = empty_dtype.ok_or_else(|| vortex_err!("Distributed fragment schema is missing"))?;
    let factories = files
        .iter()
        .cloned()
        .map(|file| Arc::new(BoundFileReaderFactory { file }) as Arc<dyn LayoutReaderFactory>)
        .collect();
    MultiLayoutDataSource::new_deferred_ranges(
        dtype,
        factories,
        row_ranges.to_vec(),
        Vec::new(),
        &SESSION,
    )
}

/// Build a reader over an already selected file set. No glob is evaluated
/// here, so an empty assignment stays empty and a worker cannot discover
/// files that were not part of the coordinator bind.
#[cfg(vortex_vane_distributed)]
pub fn build_bound_file_scan(
    files: &[BoundFile],
    empty_dtype: Option<DType>,
) -> VortexResult<MultiLayoutDataSource> {
    if files.is_empty() {
        let dtype = empty_dtype.ok_or_else(|| vortex_err!("No files matched the Vortex scan"))?;
        return Ok(MultiLayoutDataSource::new_deferred(
            dtype,
            Vec::new(),
            Vec::new(),
            &SESSION,
        ));
    }

    RUNTIME.block_on(async {
        let first = open_bound_file(&files[0]).await?.layout_reader()?;
        let remaining = files[1..]
            .iter()
            .cloned()
            .map(|file| Arc::new(BoundFileReaderFactory { file }) as Arc<dyn LayoutReaderFactory>)
            .collect();
        let byte_sizes = files.iter().map(|file| Some(file.size)).collect();
        Ok(MultiLayoutDataSource::new_with_first(
            first, remaining, byte_sizes, &SESSION,
        ))
    })
}

/// Shared bind logic for both single-glob and multi-glob variants.
#[cfg(vortex_vane_distributed)]
pub fn bind_multi_file_scan(input: &BindInputRef) -> VortexResult<BoundMultiFileScan> {
    let glob_url_parameter = input
        .get_parameter(0)
        .ok_or_else(|| vortex_err!("Missing file glob parameter"))?;

    // The input to the table function can either be a single glob, or a List of glob patterns.
    let glob_strings: Vec<String> = match glob_url_parameter.extract() {
        ExtractedValue::Varchar(glob) => {
            vec![glob.to_string()]
        }
        ExtractedValue::List(globs) => globs
            .into_iter()
            .map(|glob| {
                let ExtractedValue::Varchar(string) = glob.extract() else {
                    vortex_bail!("list element must be Varchar type")
                };

                Ok(string.to_string())
            })
            .try_collect()?,
        _ => vortex_bail!("Invalid argument to read_vortex table function"),
    };

    // Parse each glob URL and resolve its filesystem.
    let mut glob_urls: Vec<Url> = Vec::with_capacity(glob_strings.len());
    for glob_str in &glob_strings {
        glob_urls.push(parse_uri_or_path(glob_str)?);
    }

    let files = RUNTIME.block_on(async {
        let mut files = Vec::new();
        for glob_url in &glob_urls {
            let resolved = resolve_filesystem(glob_url)?;
            let mut listings = resolved
                .filesystem
                .glob(&resolved.path)?
                .try_collect::<Vec<_>>()
                .await?;
            // FileSystem::list does not promise an order. Freeze a canonical
            // order per user-supplied glob so file_index and split_id remain
            // stable across independent binds and retries.
            listings.sort();
            for listing in listings {
                let path = object_path(&listing.path)?;
                let meta = resolved.store.head(&path).await?;
                if meta.location != path {
                    vortex_bail!(
                        "Bound Vortex file identity changed while binding: expected {}, got {}",
                        listing.path,
                        meta.location
                    );
                }
                let file = bind_object(glob_url, meta)?;
                files.push(file);
            }
        }
        Ok::<_, VortexError>(files)
    })?;
    if files.is_empty() {
        vortex_bail!("No files matched the glob pattern(s): {:?}", glob_strings);
    }
    let data_source = build_bound_file_scan(&files, None)?;
    Ok(BoundMultiFileScan { data_source, files })
}

/// Original Vortex multi-file binding path for ordinary DuckDB builds.
#[cfg(not(vortex_vane_distributed))]
pub fn bind_multi_file_scan(input: &BindInputRef) -> VortexResult<MultiLayoutDataSource> {
    let glob_url_parameter = input
        .get_parameter(0)
        .ok_or_else(|| vortex_err!("Missing file glob parameter"))?;

    let glob_strings: Vec<String> = match glob_url_parameter.extract() {
        ExtractedValue::Varchar(glob) => vec![glob.to_string()],
        ExtractedValue::List(globs) => globs
            .into_iter()
            .map(|glob| {
                let ExtractedValue::Varchar(string) = glob.extract() else {
                    vortex_bail!("list element must be Varchar type")
                };
                Ok(string.to_string())
            })
            .try_collect()?,
        _ => vortex_bail!("Invalid argument to read_vortex table function"),
    };

    let mut glob_urls: Vec<Url> = Vec::with_capacity(glob_strings.len());
    for glob_str in glob_strings {
        glob_urls.push(parse_uri_or_path(&glob_str)?);
    }
    let resolved = glob_urls
        .iter()
        .map(resolve_filesystem)
        .collect::<VortexResult<Vec<_>>>()?;

    RUNTIME.block_on(async {
        let mut builder = MultiFileDataSource::new(SESSION.clone());
        for (fs, glob) in resolved {
            builder = builder.with_glob(&glob, Some(fs));
        }
        builder.build().await
    })
}
