// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::env;
use std::fs;
use std::os::unix::fs::symlink;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use async_trait::async_trait;
use bytes::Bytes;
use futures::stream;
use futures::stream::BoxStream;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::assert_arrays_eq;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_index::FlatIndex;
use vortex_index::IndexArtifact;
use vortex_index::IndexBuildRequest;
use vortex_index::IndexBuilder;
use vortex_index::IndexMetadata;
use vortex_index::IndexProvider;
use vortex_index::IndexRegistry;
use vortex_index::IndexSource;
use vortex_index::IndexStore;
use vortex_index::LocalArtifactLease;
use vortex_index::LocalIndexFiles;
use vortex_index::RowAddress;
use vortex_index::SearchMode;
use vortex_index::Snapshot;
use vortex_index::SourceBatch;
use vortex_index::SourceFile;
use vortex_index::VectorIndex;
use vortex_index::VectorSearchOptions;
use vortex_index::file::LocalFileSource;
use vortex_index::store::LocalIndexStore;
use vortex_io::runtime::single::block_on;
use vortex_session::VortexSession;

use super::BUDGET;
use super::Control;
use super::LIMITS;
use super::data;
use super::filter;
use super::options;
use super::provider;
use super::rows;
use super::session;
use super::snapshot;
use super::vector;
use crate::SPFRESH_ID;
use crate::SpFreshBuildLimits;
use crate::SpFreshBuildOptions;
use crate::SpFreshIndexBuilder;
use crate::tests::metadata;

fn config() -> SpFreshBuildOptions {
    SpFreshBuildOptions {
        format_version: 1,
        dimension: 8,
        head_count: 64,
        posting_page_limit: 12,
        replicas: 4,
    }
}

fn request(
    metadata: IndexMetadata,
    config: &SpFreshBuildOptions,
) -> VortexResult<IndexBuildRequest> {
    Ok(IndexBuildRequest {
        metadata,
        backend_options: Bytes::from(
            serde_json::to_vec(config).map_err(|err| vortex_err!("{err}"))?,
        ),
    })
}

fn builder(root: &Path, session: &VortexSession) -> VortexResult<SpFreshIndexBuilder> {
    SpFreshIndexBuilder::try_new(
        root.join("scratch"),
        session.clone(),
        SpFreshBuildLimits::default(),
    )
}

async fn prepare(
    root: &Path,
    session: &VortexSession,
) -> VortexResult<(Arc<LocalIndexStore>, IndexMetadata, Control)> {
    fs::create_dir(root.join("scratch"))?;
    fs::create_dir(root.join("store"))?;
    let snapshot = snapshot(root, session).await?;
    let source = Arc::new(
        LocalFileSource::open(
            snapshot.clone(),
            data(&[0])?.dtype().clone(),
            BUDGET,
            session.clone(),
        )
        .await?,
    );
    let store = Arc::new(LocalIndexStore::create(
        root.join("store"),
        "generation-1",
        LIMITS,
    )?);
    let mut registry = IndexRegistry::default();
    registry.register(Arc::new(
        provider(root)?.with_builder(builder(root, session)?),
    ))?;
    let metadata = registry
        .provider(SPFRESH_ID)?
        .builder()
        .ok_or_else(|| vortex_err!("Missing initial build capability"))?
        .build(
            request(metadata(snapshot.clone()), &config())?,
            source,
            Arc::clone(&store) as Arc<dyn IndexStore>,
        )
        .await?;
    assert_eq!(metadata.artifacts.len(), 8);
    assert_eq!(fs::read_dir(root.join("scratch"))?.count(), 0);
    // Construction does not seal the store or make an index visible.
    assert!(
        provider(root)?
            .open(&metadata, Arc::clone(&store) as Arc<dyn IndexStore>)
            .await
            .is_err()
    );
    let generation = store.seal(&metadata)?;
    Ok((
        store,
        metadata,
        Control {
            snapshot,
            generation,
        },
    ))
}

#[test]
fn test_static_builder_file_roundtrip_recall_and_take() -> VortexResult<()> {
    block_on(|handle| async move {
        let session = session(handle)?;
        let dir = tempfile::tempdir()?;
        let root = dir.path();
        let (store, metadata, control) = prepare(root, &session).await?;
        drop(store);
        let (store, reopened) = LocalIndexStore::open(
            root.join("store"),
            &control.generation,
            &control.snapshot,
            LIMITS,
        )?;
        assert_eq!(reopened, metadata);
        let index = provider(root)?.open(&reopened, Arc::new(store)).await?;
        let vector_index = index
            .as_vector()
            .ok_or_else(|| vortex_err!("Vector capability"))?;
        let mut flat_metadata = metadata.clone();
        flat_metadata.backend = FlatIndex::ID.into();
        flat_metadata.artifacts.clear();
        let flat = FlatIndex::try_new(
            flat_metadata,
            vector_index.spec(),
            rows(),
            Buffer::from_iter((0..256u16).flat_map(vector)),
        )?;
        let source = LocalFileSource::open(
            metadata.snapshot.clone(),
            data(&[0])?.dtype().clone(),
            BUDGET,
            session.clone(),
        )
        .await?;
        let filter = filter(&metadata.snapshot)?;
        let queries = [7, 80, 129, 242].map(|id| Buffer::from(vector(id)));
        let batches = vector_index
            .search_batch(&queries, &options(10)?, &filter)
            .await?;
        let mut recall = 0;
        for (query, hits) in queries.iter().zip(batches) {
            assert_eq!(
                hits,
                vector_index.search(query, &options(10)?, &filter).await?
            );
            let exact = flat
                .search(
                    query,
                    &VectorSearchOptions {
                        mode: SearchMode::Exact,
                        ..options(256)?
                    },
                    &filter,
                )
                .await?;
            recall += hits
                .iter()
                .filter(|hit| exact.iter().take(10).any(|truth| truth.row == hit.row))
                .count();
            for hit in &hits {
                let truth = exact
                    .iter()
                    .find(|truth| truth.row == hit.row)
                    .ok_or_else(|| vortex_err!("Missing exact hit"))?;
                assert!((hit.distance - truth.distance).abs() < 1e-5);
            }
            let addresses = hits.iter().map(|hit| hit.row).collect::<Vec<_>>();
            let ids = addresses
                .iter()
                .map(|row| u16::try_from(row.row_offset * 2 + u64::from(row.file_id == 11)))
                .collect::<Result<Vec<_>, _>>()?;
            let batch = source
                .take(&addresses, &["id".into(), "embedding".into()])
                .await?;
            assert_eq!(batch.rows, addresses);
            assert_arrays_eq!(
                batch.data,
                data(&ids)?.into_array(),
                &mut session.create_execution_ctx()
            );
        }
        assert!(recall >= 36, "recall@10 below 0.9: {recall}/40");
        drop(index);
        assert_eq!(fs::read_dir(root.join("scratch"))?.count(), 0);
        Ok(())
    })
}

const CHILD: &str = "tests::native::builder::test_static_builder_child";

#[test]
fn test_static_builder_partial_coverage_at_input_limit() -> VortexResult<()> {
    block_on(|handle| async move {
        let session = session(handle)?;
        let dir = tempfile::tempdir()?;
        let root = dir.path();
        fs::create_dir(root.join("scratch"))?;
        fs::create_dir(root.join("store"))?;
        let snapshot = snapshot(root, &session).await?;
        let source = Arc::new(
            LocalFileSource::open(
                snapshot.clone(),
                data(&[0])?.dtype().clone(),
                BUDGET,
                session.clone(),
            )
            .await?,
        );
        let mut metadata = metadata(snapshot);
        metadata.covered_files = vec![11];
        let store = Arc::new(LocalIndexStore::create(
            root.join("store"),
            "generation-1",
            LIMITS,
        )?);
        let builder = SpFreshIndexBuilder::try_new(
            root.join("scratch"),
            session,
            SpFreshBuildLimits {
                max_rows: 128,
                max_vector_bytes: 128 * 8 * 4,
                ..SpFreshBuildLimits::default()
            },
        )?;
        let metadata = builder
            .build(
                request(
                    metadata,
                    &SpFreshBuildOptions {
                        head_count: 32,
                        ..config()
                    },
                )?,
                source,
                Arc::clone(&store) as Arc<dyn IndexStore>,
            )
            .await?;
        store.seal(&metadata)?;
        assert_eq!(
            metadata
                .uncovered_files()
                .map(|file| file.id)
                .collect::<Vec<_>>(),
            vec![29]
        );
        let index = provider(root)?.open(&metadata, store).await?;
        let hits = index
            .as_vector()
            .ok_or_else(|| vortex_err!("Vector capability"))?
            .search(&vector(7), &options(10)?, &filter(&metadata.snapshot)?)
            .await?;
        assert_eq!(hits.first().map(|hit| hit.row), Some(rows()[7]));
        assert!(hits.iter().all(|hit| hit.row.file_id == 11));
        Ok(())
    })
}

#[test]
fn test_static_builder_cross_process_reopen() -> VortexResult<()> {
    let dir = tempfile::tempdir()?;
    for mode in ["build", "read", "read"] {
        let output = Command::new(env::current_exe()?)
            .args(["--exact", CHILD])
            .env("VORTEX_STATIC_BUILD_ROOT", dir.path())
            .env("VORTEX_STATIC_BUILD_MODE", mode)
            .output()?;
        if !output.status.success() {
            vortex_bail!(
                "Static build child {mode}: {}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
    Ok(())
}

#[test]
fn test_static_builder_child() -> VortexResult<()> {
    let Ok(root) = env::var("VORTEX_STATIC_BUILD_ROOT") else {
        return Ok(());
    };
    block_on(|handle| async move {
        let root = Path::new(&root);
        let session = session(handle)?;
        if env::var("VORTEX_STATIC_BUILD_MODE").map_err(|err| vortex_err!("{err}"))? == "build" {
            let (_, _, control) = prepare(root, &session).await?;
            fs::write(
                root.join("built.json"),
                serde_json::to_vec(&control).map_err(|err| vortex_err!("{err}"))?,
            )?;
        } else {
            let control: Control = serde_json::from_slice(&fs::read(root.join("built.json"))?)
                .map_err(|err| vortex_err!("{err}"))?;
            let (store, metadata) = LocalIndexStore::open(
                root.join("store"),
                &control.generation,
                &control.snapshot,
                LIMITS,
            )?;
            let index = provider(root)?.open(&metadata, Arc::new(store)).await?;
            let hits = index
                .as_vector()
                .ok_or_else(|| vortex_err!("Vector capability"))?
                .search(&vector(7), &options(5)?, &filter(&control.snapshot)?)
                .await?;
            assert_eq!(hits.first().map(|hit| hit.row), Some(rows()[7]));
            assert_eq!(hits.first().map(|hit| hit.distance), Some(0.0));
            let source = LocalFileSource::open(
                control.snapshot,
                data(&[0])?.dtype().clone(),
                BUDGET,
                session.clone(),
            )
            .await?;
            let taken = source
                .take(&[rows()[7]], &["id".into(), "embedding".into()])
                .await?;
            assert_arrays_eq!(
                taken.data,
                data(&[7])?.into_array(),
                &mut session.create_execution_ctx()
            );
        }
        assert_eq!(fs::read_dir(root.join("scratch"))?.count(), 0);
        Ok(())
    })
}

struct TestSource {
    snapshot: Snapshot,
    rows: Vec<RowAddress>,
    data: ArrayRef,
    fail: bool,
    scans: AtomicUsize,
}

#[async_trait]
impl IndexSource for TestSource {
    fn snapshot(&self) -> &Snapshot {
        &self.snapshot
    }
    fn scan(
        &self,
        _files: &[u64],
        _fields: &[String],
    ) -> VortexResult<BoxStream<'static, VortexResult<SourceBatch>>> {
        self.scans.fetch_add(1, Ordering::SeqCst);
        let mut batches = Vec::new();
        if self.rows.len() != self.data.len() {
            batches.push(Ok(SourceBatch {
                rows: self.rows.clone(),
                data: self.data.clone(),
            }));
        } else {
            batches.push(Ok(SourceBatch {
                rows: Vec::new(),
                data: self.data.slice(0..0)?,
            }));
            for (chunk, rows) in self.rows.chunks(127).enumerate() {
                let start = chunk * 127;
                batches.push(Ok(SourceBatch {
                    rows: rows.to_vec(),
                    data: self.data.slice(start..start + rows.len())?,
                }));
            }
        }
        if self.fail {
            batches.push(Err(vortex_err!("Injected scan failure")));
        }
        Ok(Box::pin(stream::iter(batches)))
    }
    async fn take(&self, _rows: &[RowAddress], _fields: &[String]) -> VortexResult<SourceBatch> {
        vortex_bail!("Unused test take")
    }
}

fn projected(values: Vec<f32>, dimension: u32, validity: Validity) -> VortexResult<ArrayRef> {
    let count = values.len() / dimension as usize;
    let lists = FixedSizeListArray::try_new(
        Buffer::from(values).into_array(),
        dimension,
        Validity::NonNullable,
        count,
    )?;
    Ok(StructArray::try_new(
        ["embedding"].into(),
        vec![lists.into_array()],
        count,
        validity,
    )?
    .into_array())
}

fn source(snapshot: Snapshot) -> VortexResult<TestSource> {
    Ok(TestSource {
        snapshot,
        rows: rows(),
        data: projected(
            (0..256u16).flat_map(vector).collect(),
            8,
            Validity::NonNullable,
        )?,
        fail: false,
        scans: AtomicUsize::new(0),
    })
}

#[test]
fn test_static_builder_rejects_inputs_without_importing() -> VortexResult<()> {
    block_on(|handle| async move {
        let session = session(handle)?;
        let dir = tempfile::tempdir()?;
        let root = dir.path();
        fs::create_dir(root.join("scratch"))?;
        fs::create_dir(root.join("store"))?;
        let snapshot = snapshot(root, &session).await?;
        let builder = builder(root, &session)?;
        for case in 0..13 {
            let mut input = source(snapshot.clone())?;
            match case {
                0 => input.rows[255] = input.rows[0],
                1 => {
                    input.rows.pop();
                }
                2 => {
                    input.rows.pop();
                    input.data = projected(
                        (0..255u16).flat_map(vector).collect(),
                        8,
                        Validity::NonNullable,
                    )?;
                }
                3 => input.rows[0].row_offset = 128,
                4 => input.data = projected(vec![0.0; 256 * 4], 4, Validity::NonNullable)?,
                5 => input.data = projected(vec![0.0; 256 * 8], 8, Validity::AllInvalid)?,
                6 | 7 => {
                    let mut values = (0..256u16).flat_map(vector).collect::<Vec<_>>();
                    values[0] = if case == 6 { f32::NAN } else { f32::INFINITY };
                    input.data = projected(values, 8, Validity::NonNullable)?;
                }
                8 => input.fail = true,
                9..=11 => {
                    let elements = match case {
                        9 => Buffer::from(vec![0.0f64; 256 * 8]).into_array(),
                        10 => PrimitiveArray::from_option_iter((0..256 * 8).map(|_| Some(0.0f32)))
                            .into_array(),
                        _ => Buffer::from(vec![0.0f32; 256 * 8]).into_array(),
                    };
                    let validity = if case == 11 {
                        Validity::AllInvalid
                    } else {
                        Validity::NonNullable
                    };
                    let lists = FixedSizeListArray::try_new(elements, 8, validity, 256)?;
                    input.data = StructArray::try_new(
                        ["embedding"].into(),
                        vec![lists.into_array()],
                        256,
                        Validity::NonNullable,
                    )?
                    .into_array();
                }
                _ => input.data = data(&(0..256u16).collect::<Vec<_>>())?.into_array(),
            }
            let generation = format!("invalid-{case}");
            let store = Arc::new(LocalIndexStore::create(
                root.join("store"),
                &generation,
                LIMITS,
            )?);
            let mut metadata = metadata(snapshot.clone());
            metadata.generation = generation.clone();
            assert!(
                builder
                    .build(request(metadata, &config())?, Arc::new(input), store)
                    .await
                    .is_err(),
                "case {case}"
            );
            assert_eq!(fs::read_dir(root.join("scratch"))?.count(), 0);
            assert_eq!(
                fs::read_dir(root.join("store").join(generation).join("artifacts"))?.count(),
                0
            );
        }
        Ok(())
    })
}

struct FailedImport {
    store: Arc<LocalIndexStore>,
    imports: AtomicUsize,
    local: bool,
}

#[async_trait]
impl IndexStore for FailedImport {
    async fn read(&self, artifact: &IndexArtifact) -> VortexResult<Bytes> {
        self.store.read(artifact).await
    }
    async fn write(&self, path: &str, bytes: Bytes) -> VortexResult<IndexArtifact> {
        self.store.write(path, bytes).await
    }
    fn as_local_files(&self) -> Option<&dyn LocalIndexFiles> {
        self.local.then_some(self)
    }
}

impl LocalIndexFiles for FailedImport {
    fn import_file(&self, path: &str, source: &Path) -> VortexResult<IndexArtifact> {
        if self.imports.fetch_add(1, Ordering::SeqCst) == 1 {
            vortex_bail!("Injected native import failure");
        }
        self.store.import_file(path, source)
    }
    fn materialize(
        &self,
        artifacts: &[IndexArtifact],
        root: &Path,
        bytes: u64,
    ) -> VortexResult<Box<dyn LocalArtifactLease>> {
        self.store.materialize(artifacts, root, bytes)
    }
}

#[test]
fn test_static_builder_failed_import_and_output_limit_preserve_generation() -> VortexResult<()> {
    block_on(|handle| async move {
        let session = session(handle)?;
        let dir = tempfile::tempdir()?;
        let root = dir.path();
        let (original, old_metadata, _) = prepare(root, &session).await?;
        let input = Arc::new(source(old_metadata.snapshot.clone())?);
        let mut target = metadata(old_metadata.snapshot.clone());
        target.generation = "failed-build".into();
        let store = Arc::new(LocalIndexStore::create(
            root.join("store"),
            &target.generation,
            LIMITS,
        )?);
        let tiny = SpFreshIndexBuilder::try_new(
            root.join("scratch"),
            session.clone(),
            SpFreshBuildLimits {
                max_native_artifact_bytes: 1,
                ..SpFreshBuildLimits::default()
            },
        )?;
        let error = tiny
            .build(
                request(target.clone(), &config())?,
                Arc::clone(&input) as Arc<dyn IndexSource>,
                Arc::clone(&store) as Arc<dyn IndexStore>,
            )
            .await
            .err()
            .ok_or_else(|| vortex_err!("Expected output limit error"))?;
        assert!(error.to_string().contains("native artifact byte limit"));
        assert_eq!(
            fs::read_dir(root.join("store/failed-build/artifacts"))?.count(),
            0
        );
        assert_eq!(fs::read_dir(root.join("scratch"))?.count(), 0);

        let no_local = Arc::new(FailedImport {
            store: Arc::clone(&store),
            imports: AtomicUsize::new(0),
            local: false,
        });
        let scans = input.scans.load(Ordering::SeqCst);
        assert!(
            builder(root, &session)?
                .build(
                    request(target.clone(), &config())?,
                    Arc::clone(&input) as Arc<dyn IndexSource>,
                    no_local
                )
                .await
                .is_err()
        );
        assert_eq!(input.scans.load(Ordering::SeqCst), scans);

        let failing = Arc::new(FailedImport {
            store,
            imports: AtomicUsize::new(0),
            local: true,
        });
        let error = builder(root, &session)?
            .build(request(target, &config())?, input, failing)
            .await
            .err()
            .ok_or_else(|| vortex_err!("Expected import failure"))?;
        assert!(error.to_string().contains("Injected native import failure"));
        assert!(
            root.join("store/failed-build/artifacts/spfresh/native/vectors.bin")
                .is_file()
        );
        assert_eq!(fs::read_dir(root.join("scratch"))?.count(), 0);
        let index = provider(root)?.open(&old_metadata, original).await?;
        let hits = index
            .as_vector()
            .ok_or_else(|| vortex_err!("Vector capability"))?
            .search(&vector(7), &options(5)?, &filter(&old_metadata.snapshot)?)
            .await?;
        assert_eq!(hits.first().map(|hit| hit.row), Some(rows()[7]));
        drop(index);
        assert_eq!(fs::read_dir(root.join("scratch"))?.count(), 0);
        Ok(())
    })
}

#[test]
fn test_static_builder_scratch_and_limit_validation() -> VortexResult<()> {
    block_on(|handle| async move {
        let session = session(handle)?;
        let dir = tempfile::tempdir()?;
        let root = dir.path();
        fs::create_dir(root.join("scratch"))?;
        symlink(root.join("scratch"), root.join("alias"))?;
        fs::write(root.join("file"), b"not a directory")?;
        for path in [
            "relative".into(),
            root.join("alias"),
            root.join("file"),
            root.join("missing"),
            root.join("scratch/../scratch"),
        ] {
            assert!(
                SpFreshIndexBuilder::try_new(path, session.clone(), SpFreshBuildLimits::default())
                    .is_err()
            );
        }
        for limits in [
            SpFreshBuildLimits {
                max_rows: 0,
                ..SpFreshBuildLimits::default()
            },
            SpFreshBuildLimits {
                max_vector_bytes: 0,
                ..SpFreshBuildLimits::default()
            },
            SpFreshBuildLimits {
                max_vector_bytes: u64::MAX,
                ..SpFreshBuildLimits::default()
            },
            SpFreshBuildLimits {
                max_native_artifact_bytes: 0,
                ..SpFreshBuildLimits::default()
            },
        ] {
            assert!(
                SpFreshIndexBuilder::try_new(root.join("scratch"), session.clone(), limits)
                    .is_err()
            );
        }
        Ok(())
    })
}

#[test]
fn test_static_builder_validates_request_before_scan() -> VortexResult<()> {
    block_on(|handle| async move {
        let session = session(handle)?;
        let dir = tempfile::tempdir()?;
        let root = dir.path();
        fs::create_dir(root.join("scratch"))?;
        fs::create_dir(root.join("store"))?;
        let snapshot = snapshot(root, &session).await?;
        let source = Arc::new(source(snapshot.clone())?);
        let builder = builder(root, &session)?;
        let store = Arc::new(LocalIndexStore::create(
            root.join("store"),
            "generation-1",
            LIMITS,
        )?);
        for case in 0..10 {
            let mut request = request(metadata(snapshot.clone()), &config())?;
            let mut options = config();
            match case {
                0 => options.format_version = 2,
                1 => options.head_count = 0,
                2 => options.head_count = 256,
                3 => options.replicas = 0,
                4 => options.replicas = 9,
                5 => options.posting_page_limit = 0,
                6 => options.dimension = 4097,
                7 => request.metadata.snapshot.version = "different".into(),
                8 => request.metadata.backend = "other".into(),
                _ => request.metadata.covered_files.clear(),
            }
            request.backend_options =
                Bytes::from(serde_json::to_vec(&options).map_err(|err| vortex_err!("{err}"))?);
            assert!(
                builder
                    .build(
                        request,
                        Arc::clone(&source) as Arc<dyn IndexSource>,
                        Arc::clone(&store) as Arc<dyn IndexStore>
                    )
                    .await
                    .is_err()
            );
        }
        for bytes in [
            Bytes::new(),
            Bytes::from_static(b"{\"unknown\":1}"),
            Bytes::from(vec![b' '; 4097]),
        ] {
            let mut request = request(metadata(snapshot.clone()), &config())?;
            request.backend_options = bytes;
            assert!(
                builder
                    .build(
                        request,
                        Arc::clone(&source) as Arc<dyn IndexSource>,
                        Arc::clone(&store) as Arc<dyn IndexStore>
                    )
                    .await
                    .is_err()
            );
        }
        for limits in [
            SpFreshBuildLimits {
                max_rows: 255,
                ..SpFreshBuildLimits::default()
            },
            SpFreshBuildLimits {
                max_vector_bytes: 256 * 8 * 4 - 1,
                ..SpFreshBuildLimits::default()
            },
        ] {
            let bounded =
                SpFreshIndexBuilder::try_new(root.join("scratch"), session.clone(), limits)?;
            assert!(
                bounded
                    .build(
                        request(metadata(snapshot.clone()), &config())?,
                        Arc::clone(&source) as Arc<dyn IndexSource>,
                        Arc::clone(&store) as Arc<dyn IndexStore>
                    )
                    .await
                    .is_err()
            );
        }
        assert_eq!(source.scans.load(Ordering::SeqCst), 0);
        assert_eq!(fs::read_dir(root.join("scratch"))?.count(), 0);
        Ok(())
    })
}

#[test]
fn test_static_builder_real_multipage_postings() -> VortexResult<()> {
    block_on(|handle| async move {
        let session = session(handle)?;
        let dir = tempfile::tempdir()?;
        let root = dir.path();
        fs::create_dir(root.join("scratch"))?;
        fs::create_dir(root.join("store"))?;
        let count = 30_000u32;
        let snapshot = Snapshot {
            dataset_id: "multipage".into(),
            version: "v1".into(),
            schema_fingerprint: "f32-8".into(),
            files: vec![SourceFile {
                id: 11,
                uri: "test://multipage".into(),
                version: "v1".into(),
                row_count: u64::from(count),
            }],
        };
        let values = (0..count)
            .flat_map(|id| {
                let mut state = id + 1;
                (0..8).map(move |_| {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    f32::from((state >> 16) as u16) / 65536.0
                })
            })
            .collect::<Vec<_>>();
        let query = values[7 * 8..8 * 8].to_vec();
        let source = Arc::new(TestSource {
            snapshot: snapshot.clone(),
            rows: (0..u64::from(count))
                .map(|row_offset| RowAddress {
                    file_id: 11,
                    row_offset,
                })
                .collect(),
            data: projected(values, 8, Validity::NonNullable)?,
            fail: false,
            scans: AtomicUsize::new(0),
        });
        let store = Arc::new(LocalIndexStore::create(
            root.join("store"),
            "generation-1",
            LIMITS,
        )?);
        let config = SpFreshBuildOptions {
            head_count: 60,
            replicas: 1,
            posting_page_limit: 128,
            ..config()
        };
        let metadata = builder(root, &session)?
            .build(
                request(metadata(snapshot), &config)?,
                source,
                Arc::clone(&store) as Arc<dyn IndexStore>,
            )
            .await?;
        store.seal(&metadata)?;
        let artifact = metadata
            .artifacts
            .iter()
            .find(|artifact| artifact.path == "spfresh/native/postings.bin")
            .ok_or_else(|| vortex_err!("Missing posting artifact"))?;
        let bytes = store.read(artifact).await?;
        let max_pages = bytes[16..16 + 60 * 12]
            .chunks_exact(12)
            .map(|entry| u16::from_le_bytes([entry[10], entry[11]]))
            .max()
            .ok_or_else(|| vortex_err!("No postings"))?;
        assert!(max_pages > 2, "must exercise real multipage postings");
        let index = provider(root)?.open(&metadata, store).await?;
        let vector = index
            .as_vector()
            .ok_or_else(|| vortex_err!("Vector capability"))?;
        let filter = filter(&metadata.snapshot)?;
        let hits = vector.search(&query, &options(5)?, &filter).await?;
        assert_eq!(hits.first().map(|hit| hit.row.row_offset), Some(7));
        assert_eq!(hits.first().map(|hit| hit.distance), Some(0.0));
        for pages in [1, 127, 129] {
            let invalid = VectorSearchOptions {
                backend_options: Bytes::from(format!(
                    r#"{{"max_check":4096,"internal_results":64,"search_pages":{pages}}}"#
                )),
                ..options(5)?
            };
            assert!(vector.search(&query, &invalid, &filter).await.is_err());
            assert!(vector.search_batch(&[], &invalid, &filter).await.is_err());
        }
        assert_eq!(vector.search(&query, &options(5)?, &filter).await?, hits);
        Ok(())
    })
}
