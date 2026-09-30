// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Explicit, local static-index SQL operations (`index` feature, Unix).
//!
//! Extensions register provider factories before loading SQL functions. Each
//! operation creates a private scratch lease. Reference files are immutable,
//! exclusively published only after build, seal, and qualified reopen. They
//! describe frozen whole-file coverage, not a mutable table catalog.

mod ffi;
#[cfg(test)]
mod tests;

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::fs;
use std::fs::File;
use std::io::Read;
use std::io::Write;
use std::num::NonZeroUsize;
use std::path::Component;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::LazyLock;

use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use futures::stream::BoxStream;
use parking_lot::RwLock;
use rustix::fs as unix_fs;
use serde::Deserialize;
use serde::Serialize;
use serde::de::DeserializeSeed;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::FixedSizeListArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinArray;
use vortex::array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex::array::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::array::validity::Validity;
use vortex::buffer::ByteBuffer;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::dtype::serde::DTypeSerde;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::file::OpenOptionsSessionExt;
use vortex::session::VortexSession;
use vortex_index::DistanceMetric;
use vortex_index::IndexBuildRequest;
use vortex_index::IndexMetadata;
use vortex_index::IndexProvider;
use vortex_index::IndexRegistry;
use vortex_index::IndexSource;
use vortex_index::RowAddress;
use vortex_index::RowFilter;
use vortex_index::SearchMode;
use vortex_index::Snapshot;
use vortex_index::SourceBatch;
use vortex_index::SourceFile;
use vortex_index::VectorSearchOptions;
use vortex_index::file::LocalFileSource;
use vortex_index::file::file_version;
use vortex_index::file::schema_fingerprint;
use vortex_index::store::LocalGeneration;
use vortex_index::store::LocalIndexStore;
use vortex_index::store::LocalStoreLimits;

use crate::SESSION;

const MAX_SOURCE_BYTES: usize = 512 * 1024 * 1024;
const MAX_REFERENCE_BYTES: usize = 16 * 1024 * 1024;
const MAX_OPTIONS_BYTES: usize = 64 * 1024;
const MAX_K: usize = 10_000;
const STORE_LIMITS: LocalStoreLimits = LocalStoreLimits {
    max_artifact_bytes: 1024 * 1024 * 1024,
    max_manifest_bytes: MAX_REFERENCE_BYTES,
};

/// Construct an explicitly selected provider using this operation's scratch root.
///
/// The root exists, is private and absolute, and remains alive until all provider
/// handles have closed. The factory must not retain its path beyond the operation.
pub type IndexProviderFactory = fn(VortexSession, &Path) -> VortexResult<Arc<dyn IndexProvider>>;

static FACTORIES: LazyLock<RwLock<BTreeMap<String, IndexProviderFactory>>> =
    LazyLock::new(|| RwLock::new(BTreeMap::new()));

/// Register a process-wide SQL backend before loading an extension.
///
/// Re-registering the same factory supports multiple databases in one process;
/// a different factory cannot replace an existing backend. No provider is added
/// by default, and this crate does not link a native index implementation.
pub fn register_index_provider_factory(
    id: &str,
    factory: IndexProviderFactory,
) -> VortexResult<()> {
    if id.trim().is_empty() || id.contains('\0') {
        vortex_bail!("Invalid SQL index backend identity");
    }
    let mut factories = FACTORIES.write();
    if let Some(existing) = factories.get(id) {
        if !std::ptr::fn_addr_eq(*existing, factory) {
            vortex_bail!("SQL index backend is already registered: {id}");
        }
    } else {
        factories.insert(id.into(), factory);
    }
    Ok(())
}

fn provider(id: &str, scratch: &Path) -> VortexResult<Arc<dyn IndexProvider>> {
    let factory = FACTORIES
        .read()
        .get(id)
        .copied()
        .ok_or_else(|| vortex_err!("Unknown SQL index backend: {id}"))?;
    let provider = factory(SESSION.clone(), scratch)?;
    if provider.id() != id {
        vortex_bail!("SQL index factory returned a different backend");
    }
    Ok(provider)
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Reference {
    format_version: u32,
    snapshot: Snapshot,
    #[serde(deserialize_with = "deserialize_dtype")]
    dtype: DType,
    generation: LocalGeneration,
}

fn deserialize_dtype<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<DType, D::Error> {
    DTypeSerde::<DType>::new(&SESSION).deserialize(deserializer)
}

#[derive(Clone)]
enum Request {
    Build {
        files: Vec<String>,
        reference: PathBuf,
        field: String,
        backend: String,
        options: Bytes,
    },
    Search {
        reference: PathBuf,
        identity: String,
        descriptor: Reference,
        query: Vec<f32>,
        k: NonZeroUsize,
        options: Bytes,
    },
}

impl Request {
    fn result_dtype(&self) -> VortexResult<DType> {
        match self {
            Self::Build { .. } => Ok(DType::Struct(
                vortex::dtype::StructFields::from_iter([
                    ("reference", DType::Utf8(Nullability::NonNullable)),
                    ("generation", DType::Utf8(Nullability::NonNullable)),
                    (
                        "rows",
                        DType::Primitive(PType::U64, Nullability::NonNullable),
                    ),
                ]),
                Nullability::NonNullable,
            )),
            Self::Search { descriptor, .. } => Ok(DType::Struct(
                vortex::dtype::StructFields::from_iter([
                    (
                        "rank",
                        DType::Primitive(PType::U64, Nullability::NonNullable),
                    ),
                    (
                        "file_id",
                        DType::Primitive(PType::U64, Nullability::NonNullable),
                    ),
                    (
                        "row_offset",
                        DType::Primitive(PType::U64, Nullability::NonNullable),
                    ),
                    (
                        "distance",
                        DType::Primitive(PType::F32, Nullability::NonNullable),
                    ),
                    ("row", descriptor.dtype.clone()),
                ]),
                Nullability::NonNullable,
            )),
        }
    }

    async fn execute(&self) -> VortexResult<StructArray> {
        match self {
            Self::Build {
                files,
                reference,
                field,
                backend,
                options,
            } => build(files, reference, field, backend, options.clone()).await,
            Self::Search {
                reference,
                identity,
                descriptor,
                query,
                k,
                options,
            } => {
                if file_version(&read_regular(reference, MAX_REFERENCE_BYTES)?) != *identity {
                    vortex_bail!("Index reference changed after bind; rebind the query");
                }
                search(reference, descriptor, query, *k, options.clone()).await
            }
        }
    }
}

fn local_path(path: &Path, directory: bool) -> VortexResult<()> {
    if !path.is_absolute() {
        vortex_bail!("SQL indexes require absolute local paths");
    }
    let mut current = PathBuf::new();
    for component in path.components() {
        match component {
            Component::RootDir | Component::Normal(_) => current.push(component),
            _ => vortex_bail!("SQL index paths cannot contain parent components"),
        }
        let metadata = fs::symlink_metadata(&current)?;
        if metadata.file_type().is_symlink() {
            vortex_bail!("SQL index paths cannot contain symlinks");
        }
    }
    if directory && !fs::metadata(path)?.is_dir() {
        vortex_bail!("SQL index root is not a directory");
    }
    Ok(())
}

fn root(reference: &Path) -> VortexResult<&Path> {
    if !reference.is_absolute() || reference.file_name().is_none() {
        vortex_bail!("Index reference requires an absolute local filename");
    }
    let parent = reference
        .parent()
        .ok_or_else(|| vortex_err!("Index reference parent"))?;
    local_path(parent, true)?;
    Ok(parent)
}

fn read_regular(path: &Path, limit: usize) -> VortexResult<Vec<u8>> {
    local_path(path, false)?;
    let file = File::from(
        unix_fs::open(
            path,
            unix_fs::OFlags::RDONLY
                | unix_fs::OFlags::NOFOLLOW
                | unix_fs::OFlags::NONBLOCK
                | unix_fs::OFlags::CLOEXEC
                | unix_fs::OFlags::NOCTTY,
            unix_fs::Mode::empty(),
        )
        .map_err(|error| vortex_err!("Cannot open index input: {error}"))?,
    );
    if !file.metadata()?.is_file() || file.metadata()?.len() > u64::try_from(limit)? {
        vortex_bail!("Index input is not a regular file within the byte limit");
    }
    let mut bytes = Vec::new();
    file.take(u64::try_from(limit)?.saturating_add(1))
        .read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        vortex_bail!("Index input exceeds the byte limit");
    }
    Ok(bytes)
}

fn read_reference(path: &Path) -> VortexResult<(Reference, String)> {
    root(path)?;
    let bytes = read_regular(path, MAX_REFERENCE_BYTES)?;
    let reference: Reference = serde_json::from_slice(&bytes)
        .map_err(|error| vortex_err!("Invalid index reference: {error}"))?;
    if reference.format_version != 1
        || reference.snapshot.schema_fingerprint != schema_fingerprint(&reference.dtype)?
        || !matches!(reference.dtype, DType::Struct(_, Nullability::NonNullable))
    {
        vortex_bail!("Invalid SQL index reference format or schema");
    }
    reference.snapshot.validate()?;
    Ok((reference, file_version(&bytes)))
}

async fn build(
    files: &[String],
    reference: &Path,
    field: &str,
    backend: &str,
    options: Bytes,
) -> VortexResult<StructArray> {
    let root = root(reference)?;
    if fs::symlink_metadata(reference).is_ok() {
        vortex_bail!("Index reference already exists");
    }
    let scratch = tempfile::Builder::new()
        .prefix(".index-scratch-")
        .tempdir_in(root)?;
    let provider = provider(backend, scratch.path())?;
    let builder = provider
        .builder()
        .ok_or_else(|| vortex_err!("Backend has no initial builder"))?;
    if !provider.supports_version(1) {
        vortex_bail!("SQL initial construction requires backend format version 1");
    }
    let mut inventory = Vec::new();
    let mut dtype = None;
    let mut remaining = MAX_SOURCE_BYTES;
    for (index, path) in files.iter().enumerate() {
        let bytes = read_regular(Path::new(path), remaining)?;
        remaining -= bytes.len();
        let version = file_version(&bytes);
        let opened = SESSION
            .open_options()
            .open_buffer(ByteBuffer::from(bytes))?;
        if dtype.as_ref().is_some_and(|dtype| opened.dtype() != dtype) {
            vortex_bail!("Index source files must share the same schema");
        }
        dtype = Some(opened.dtype().clone());
        inventory.push(SourceFile {
            id: u64::try_from(index)?
                .checked_add(1)
                .ok_or_else(|| vortex_err!("File ID overflow"))?,
            uri: path.clone(),
            version,
            row_count: opened.row_count(),
        });
    }
    let dtype = dtype.ok_or_else(|| vortex_err!("Index construction requires source files"))?;
    let snapshot = Snapshot {
        dataset_id: reference.to_string_lossy().into_owned(),
        version: file_version(
            &serde_json::to_vec(&inventory).map_err(|error| vortex_err!("{error}"))?,
        ),
        schema_fingerprint: schema_fingerprint(&dtype)?,
        files: inventory,
    };
    let source = Arc::new(
        LocalFileSource::open(
            snapshot.clone(),
            dtype.clone(),
            MAX_SOURCE_BYTES,
            SESSION.clone(),
        )
        .await?,
    );
    let build_source = Arc::new(BuildSource {
        source: Arc::clone(&source),
        field: field.into(),
    });
    let generation = format!(
        "generation-{}",
        scratch
            .path()
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| vortex_err!("Scratch generation name"))?
    );
    let store = Arc::new(LocalIndexStore::create(root, &generation, STORE_LIMITS)?);
    let expected = IndexMetadata {
        format_version: 1,
        name: field.into(),
        generation: generation.clone(),
        backend: backend.into(),
        backend_version: 1,
        snapshot: snapshot.clone(),
        fields: vec![field.into()],
        covered_files: snapshot.files.iter().map(|file| file.id).collect(),
        artifacts: vec![],
    };
    let metadata = builder
        .build(
            IndexBuildRequest {
                metadata: expected.clone(),
                backend_options: options,
            },
            build_source,
            Arc::clone(&store) as Arc<dyn vortex_index::IndexStore>,
        )
        .await?;
    let mut built_identity = metadata.clone();
    built_identity.artifacts.clear();
    if built_identity != expected {
        vortex_bail!("Index builder changed the requested identity or coverage");
    }
    let descriptor = store.seal(&metadata)?;
    let (reopened, reopened_metadata) =
        LocalIndexStore::open(root, &descriptor, &snapshot, STORE_LIMITS)?;
    let mut registry = IndexRegistry::default();
    registry.register(provider)?;
    let handle = registry
        .open(&reopened_metadata, &snapshot, Arc::new(reopened))
        .await?;
    if handle
        .as_vector()
        .is_none_or(|vector| vector.spec().metric != DistanceMetric::SquaredL2)
    {
        vortex_bail!("SQL static indexes require squared-L2 vector capability");
    }
    drop(handle);
    // Recheck the paths before publishing the pinned source, rather than infer a
    // new snapshot from whichever source files are now present.
    drop(
        LocalFileSource::open(
            snapshot.clone(),
            dtype.clone(),
            MAX_SOURCE_BYTES,
            SESSION.clone(),
        )
        .await?,
    );
    let count = snapshot
        .files
        .iter()
        .try_fold(0u64, |count, file| count.checked_add(file.row_count))
        .ok_or_else(|| vortex_err!("Source row count overflow"))?;
    let reference_bytes = serde_json::to_vec(&Reference {
        format_version: 1,
        snapshot: snapshot.clone(),
        dtype,
        generation: descriptor,
    })
    .map_err(|error| vortex_err!("{error}"))?;
    if reference_bytes.len() > MAX_REFERENCE_BYTES {
        vortex_bail!("Index reference exceeds the byte limit");
    }
    let mut pending = tempfile::NamedTempFile::new_in(root)?;
    pending.write_all(&reference_bytes)?;
    pending.as_file().sync_all()?;
    pending
        .persist_noclobber(reference)
        .map_err(|error| vortex_err!("Cannot publish index reference: {error}"))?;
    File::open(root)?.sync_all()?;
    StructArray::try_from_iter([
        (
            "reference",
            VarBinArray::from_strs(vec![reference.to_string_lossy().as_ref()]).into_array(),
        ),
        (
            "generation",
            VarBinArray::from_strs(vec![generation.as_str()]).into_array(),
        ),
        ("rows", PrimitiveArray::from_iter([count]).into_array()),
    ])
}

async fn search(
    reference: &Path,
    descriptor: &Reference,
    query: &[f32],
    k: NonZeroUsize,
    options: Bytes,
) -> VortexResult<StructArray> {
    let root = root(reference)?;
    let source = LocalFileSource::open(
        descriptor.snapshot.clone(),
        descriptor.dtype.clone(),
        MAX_SOURCE_BYTES,
        SESSION.clone(),
    )
    .await?;
    let (store, metadata) = LocalIndexStore::open(
        root,
        &descriptor.generation,
        source.snapshot(),
        STORE_LIMITS,
    )?;
    if metadata.uncovered_files().next().is_some() || metadata.fields.len() != 1 {
        vortex_bail!("SQL static search requires full file coverage and one vector field");
    }
    let scratch = tempfile::Builder::new()
        .prefix(".index-scratch-")
        .tempdir_in(root)?;
    let mut registry = IndexRegistry::default();
    registry.register(provider(&metadata.backend, scratch.path())?)?;
    let index = registry
        .open(&metadata, source.snapshot(), Arc::new(store))
        .await?;
    let vector = index
        .as_vector()
        .ok_or_else(|| vortex_err!("Index has no vector capability"))?;
    if vector.spec().metric != DistanceMetric::SquaredL2 {
        vortex_bail!("SQL static search supports squared L2 only");
    }
    let filter = RowFilter::try_new(source.snapshot().clone(), None, BTreeSet::new())?;
    let hits = vector
        .search(
            query,
            &VectorSearchOptions {
                k,
                mode: SearchMode::Approximate,
                backend_options: options,
            },
            &filter,
        )
        .await?;
    if hits.len() > k.get() {
        vortex_bail!("Index returned more than k hits");
    }
    let mut seen = BTreeSet::new();
    for hit in &hits {
        source.snapshot().validate_row(hit.row)?;
        if !hit.distance.is_finite() || !metadata.covers(hit.row.file_id) || !seen.insert(hit.row) {
            vortex_bail!("Index returned invalid vector hits");
        }
    }
    if hits.windows(2).any(|pair| {
        pair[0]
            .distance
            .total_cmp(&pair[1].distance)
            .then_with(|| pair[0].row.cmp(&pair[1].row))
            .is_gt()
    }) {
        vortex_bail!("Index returned unordered vector hits");
    }
    let rows = hits.iter().map(|hit| hit.row).collect::<Vec<_>>();
    let fields = descriptor
        .dtype
        .as_struct_fields()
        .names()
        .iter()
        .map(|field| field.to_string())
        .collect::<Vec<_>>();
    let taken = source.take(&rows, &fields).await?;
    if taken.rows != rows || taken.data.dtype() != &descriptor.dtype {
        vortex_bail!("Index source take did not preserve ranked rows or schema");
    }
    StructArray::try_from_iter([
        (
            "rank",
            PrimitiveArray::from_iter((1..=hits.len()).map(|rank| rank as u64)).into_array(),
        ),
        (
            "file_id",
            PrimitiveArray::from_iter(hits.iter().map(|hit| hit.row.file_id)).into_array(),
        ),
        (
            "row_offset",
            PrimitiveArray::from_iter(hits.iter().map(|hit| hit.row.row_offset)).into_array(),
        ),
        (
            "distance",
            PrimitiveArray::from_iter(hits.iter().map(|hit| hit.distance)).into_array(),
        ),
        ("row", taken.data),
    ])
}

struct BuildSource {
    source: Arc<LocalFileSource>,
    field: String,
}

#[async_trait]
impl IndexSource for BuildSource {
    fn snapshot(&self) -> &Snapshot {
        self.source.snapshot()
    }

    fn scan(
        &self,
        files: &[u64],
        fields: &[String],
    ) -> VortexResult<BoxStream<'static, VortexResult<SourceBatch>>> {
        if fields != [self.field.clone()] {
            vortex_bail!("SQL vector builder requires its selected field only");
        }
        let field = self.field.clone();
        let snapshot = self.source.snapshot().clone();
        let mut ctx = SESSION.create_execution_ctx();
        Ok(Box::pin(self.source.scan(files, fields)?.map(
            move |batch| {
                let batch = batch?;
                let data = batch.data.execute::<StructArray>(&mut ctx)?;
                let lists = data
                    .unmasked_field(0)
                    .clone()
                    .execute::<FixedSizeListArray>(&mut ctx)?;
                let elements = lists
                    .elements()
                    .clone()
                    .execute::<PrimitiveArray>(&mut ctx)?;
                if elements.ptype() != PType::F32
                    || !lists
                        .validity()?
                        .execute_mask(lists.len(), &mut ctx)?
                        .all_true()
                    || !elements
                        .validity()?
                        .execute_mask(elements.len(), &mut ctx)?
                        .all_true()
                {
                    vortex_bail!(
                        "SQL static index requires Float32 vectors without NULL rows or elements"
                    );
                }
                let values =
                    PrimitiveArray::new(elements.to_buffer::<f32>(), Validity::NonNullable);
                let vectors = FixedSizeListArray::new(
                    values.into_array(),
                    lists.list_size(),
                    Validity::NonNullable,
                    lists.len(),
                );
                SourceBatch::try_new(
                    &snapshot,
                    batch.rows,
                    StructArray::try_from_iter([(field.as_str(), vectors.into_array())])?
                        .into_array(),
                )
            },
        )))
    }

    async fn take(&self, rows: &[RowAddress], fields: &[String]) -> VortexResult<SourceBatch> {
        self.source.take(rows, fields).await
    }
}
