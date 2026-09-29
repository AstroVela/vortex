// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::fs::OpenOptions;
use std::num::NonZeroUsize;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::thread;

use async_trait::async_trait;
use bytes::Bytes;
use futures::executor::block_on as sync;
use serde::Deserialize;
use serde::Serialize;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::StructArray;
use vortex_array::assert_arrays_eq;
use vortex_array::session::ArraySessionExt;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_edition::Edition;
use vortex_edition::EditionId;
use vortex_edition::EditionInclusion;
use vortex_edition::EditionSessionExt;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_file::WriteOptionsSessionExt;
use vortex_index::FlatIndex;
use vortex_index::IndexArtifact;
use vortex_index::IndexMetadata;
use vortex_index::IndexProvider;
use vortex_index::IndexRegistry;
use vortex_index::IndexSource;
use vortex_index::IndexStore;
use vortex_index::LocalArtifactLease;
use vortex_index::LocalIndexFiles;
use vortex_index::RowAddress;
use vortex_index::RowFilter;
use vortex_index::SearchHit;
use vortex_index::SearchMode;
use vortex_index::Snapshot;
use vortex_index::SourceFile;
use vortex_index::VectorIndex;
use vortex_index::VectorSearchOptions;
use vortex_index::file::LocalFileSource;
use vortex_index::file::file_version;
use vortex_index::file::schema_fingerprint;
use vortex_index::store::LocalGeneration;
use vortex_index::store::LocalIndexStore;
use vortex_index::store::LocalStoreLimits;
use vortex_io::runtime::Handle;
use vortex_io::runtime::single::block_on;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

use crate::SpFreshBundle;
use crate::SpFreshLimits;
use crate::SpFreshProvider;
use crate::bundle::DESCRIPTOR;
use crate::bundle::ROWS;
use crate::ffi::Native;
use crate::import_bundle;

mod builder;

const BUDGET: usize = 64 * 1024 * 1024;
const LIMITS: LocalStoreLimits = LocalStoreLimits {
    max_artifact_bytes: BUDGET,
    max_manifest_bytes: 1024 * 1024,
};
const BUNDLE: SpFreshBundle = SpFreshBundle {
    dimension: 8,
    rows: 256,
    posting_page_limit: 12,
};

struct ObservedStore {
    inner: Arc<LocalIndexStore>,
    closed_before_lease: Arc<AtomicBool>,
    local: bool,
}

#[derive(Debug)]
struct ObservedLease {
    inner: Box<dyn LocalArtifactLease>,
    closed_before_lease: Arc<AtomicBool>,
}

impl LocalArtifactLease for ObservedLease {
    fn path(&self) -> &Path {
        self.inner.path()
    }
}

impl Drop for ObservedLease {
    fn drop(&mut self) {
        let Ok(files) = fs::read_dir("/proc/self/fd") else {
            return;
        };
        let still_open = files
            .filter_map(Result::ok)
            .filter_map(|file| fs::read_link(file.path()).ok())
            .any(|path| path.starts_with(self.path()));
        self.closed_before_lease
            .store(!still_open, Ordering::SeqCst);
    }
}

#[async_trait]
impl IndexStore for ObservedStore {
    async fn read(&self, artifact: &IndexArtifact) -> VortexResult<Bytes> {
        self.inner.read(artifact).await
    }
    async fn write(&self, path: &str, bytes: Bytes) -> VortexResult<IndexArtifact> {
        self.inner.write(path, bytes).await
    }
    fn as_local_files(&self) -> Option<&dyn LocalIndexFiles> {
        self.local.then_some(self)
    }
}

impl LocalIndexFiles for ObservedStore {
    fn import_file(&self, path: &str, source: &Path) -> VortexResult<IndexArtifact> {
        self.inner.import_file(path, source)
    }
    fn materialize(
        &self,
        artifacts: &[IndexArtifact],
        scratch: &Path,
        budget: u64,
    ) -> VortexResult<Box<dyn LocalArtifactLease>> {
        Ok(Box::new(ObservedLease {
            inner: self.inner.materialize(artifacts, scratch, budget)?,
            closed_before_lease: Arc::clone(&self.closed_before_lease),
        }))
    }
}

#[derive(Serialize, Deserialize)]
struct Control {
    snapshot: Snapshot,
    generation: LocalGeneration,
}

fn vector(id: u16) -> Vec<f32> {
    (0..8u16)
        .map(|component| f32::from((id * 97 + component * 53 + id * component) % 997) / 997.0)
        .collect()
}

fn rows() -> Vec<RowAddress> {
    (0..256u64)
        .map(|id| RowAddress {
            file_id: if id % 2 == 0 { 29 } else { 11 },
            row_offset: id / 2,
        })
        .collect()
}

fn data(ids: &[u16]) -> VortexResult<StructArray> {
    StructArray::try_new(
        ["id", "embedding"].into(),
        vec![
            Buffer::from_iter(ids.iter().map(|id| i32::from(*id))).into_array(),
            FixedSizeListArray::try_new(
                Buffer::from_iter(ids.iter().flat_map(|id| vector(*id))).into_array(),
                8,
                Validity::NonNullable,
                ids.len(),
            )?
            .into_array(),
        ],
        ids.len(),
        Validity::NonNullable,
    )
}

fn session(handle: Handle) -> VortexResult<VortexSession> {
    let session = array_session()
        .with::<LayoutSession>()
        .with::<RuntimeSession>()
        .with_handle(handle);
    vortex_file::register_default_encodings(&session);
    let edition = EditionId::new("spfresh-test", 2026, 9, 0);
    session
        .editions()
        .declare_edition(Edition {
            id: edition,
            min_vortex_version: None,
        })
        .map_err(|err| vortex_err!("{err}"))?;
    let encodings = session
        .arrays()
        .registry()
        .read(|map| map.keys().copied().collect::<Vec<_>>());
    for encoding in encodings {
        session
            .editions()
            .declare_inclusion(EditionInclusion::new(&encoding, edition))
            .map_err(|err| vortex_err!("{err}"))?;
    }
    session
        .enable_edition(edition)
        .map_err(|err| vortex_err!("{err}"))?;
    Ok(session)
}

async fn snapshot(root: &Path, session: &VortexSession) -> VortexResult<Snapshot> {
    let mut files = Vec::new();
    for (file, parity) in [(11, 1u16), (29, 0u16)] {
        let ids = (0..128u16).map(|id| 2 * id + parity).collect::<Vec<_>>();
        let array = data(&ids)?.into_array();
        let mut encoded = Vec::new();
        session
            .write_options()
            .with_strategy(Arc::new(FlatLayoutStrategy::default()))
            .write(&mut encoded, array.to_array_stream())
            .await?;
        let path = root.join(format!("{file}.vortex"));
        fs::write(&path, &encoded)?;
        files.push(SourceFile {
            id: file,
            uri: path
                .to_str()
                .ok_or_else(|| vortex_err!("Test path"))?
                .into(),
            version: file_version(&encoded),
            row_count: 128,
        });
    }
    Ok(Snapshot {
        dataset_id: "spfresh-integration".into(),
        version: "snapshot-1".into(),
        schema_fingerprint: schema_fingerprint(data(&[0])?.dtype())?,
        files,
    })
}

fn build_native(root: &Path) -> VortexResult<()> {
    let mut input = Vec::new();
    input.extend_from_slice(&256i32.to_le_bytes());
    input.extend_from_slice(&8i32.to_le_bytes());
    for value in (0..256u16).flat_map(vector) {
        input.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(root.join("input.bin"), input)?;
    let output = Command::new(env!("VORTEX_SPFRESH_FIXTURE"))
        .arg(root.join("input.bin"))
        .arg(root.join("native"))
        .output()?;
    if !output.status.success() {
        return Err(vortex_err!(
            "Native fixture failed: {}\n{}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(())
}

async fn prepare(
    root: &Path,
    session: &VortexSession,
) -> VortexResult<(Arc<LocalIndexStore>, IndexMetadata, Control)> {
    fs::create_dir(root.join("store"))?;
    fs::create_dir(root.join("scratch"))?;
    let snapshot = snapshot(root, session).await?;
    build_native(root)?;
    let store = Arc::new(LocalIndexStore::create(
        root.join("store"),
        "generation-1",
        LIMITS,
    )?);
    let metadata = import_bundle(
        super::metadata(snapshot.clone()),
        store.as_ref(),
        &root.join("native"),
        BUNDLE,
        &rows(),
    )
    .await?;
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

fn provider(root: &Path) -> VortexResult<SpFreshProvider> {
    SpFreshProvider::try_new(root.join("scratch"), SpFreshLimits::default())
}

fn options(k: usize) -> VortexResult<VectorSearchOptions> {
    Ok(VectorSearchOptions {
        k: NonZeroUsize::new(k).ok_or_else(|| vortex_err!("Zero test k"))?,
        mode: SearchMode::Approximate,
        backend_options: Bytes::new(),
    })
}

fn filter(snapshot: &Snapshot) -> VortexResult<RowFilter> {
    RowFilter::try_new(snapshot.clone(), None, BTreeSet::new())
}

#[test]
fn test_native_parity_batch_flat_recall_and_multifile_take() -> VortexResult<()> {
    block_on(|handle| async move {
        let session = session(handle)?;
        let dir = tempfile::tempdir()?;
        let root = dir.path();
        let (store, metadata, _) = prepare(root, &session).await?;
        let raw = Native::open(&root.join("native"), BUNDLE)?;
        let queries = [7u16, 80, 129, 242].map(|id| Buffer::from(vector(id)));
        let flat_queries = queries
            .iter()
            .flat_map(|query| query.iter().copied())
            .collect::<Vec<_>>();
        let native = raw.search(&flat_queries, 10, 4096, 64, 12)?;
        drop(raw);
        fs::remove_dir_all(root.join("native"))?;
        fs::remove_file(root.join("input.bin"))?;

        let mut registry = IndexRegistry::default();
        registry.register(Arc::new(provider(root)?))?;
        let index = registry.open(&metadata, &metadata.snapshot, store).await?;
        let vector_index = index
            .as_vector()
            .ok_or_else(|| vortex_err!("Missing vector capability"))?;
        assert!(!vector_index.supports_exact());
        assert!(!vector_index.supports_filter());
        let filter = filter(&metadata.snapshot)?;
        let results = vector_index
            .search_batch(&queries, &options(10)?, &filter)
            .await?;
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
        let mapping = rows();
        let mut recall = 0usize;
        for (query, (hits, native)) in queries.iter().zip(results.iter().zip(native)) {
            let mut expected = native
                .into_iter()
                .map(|(id, distance)| SearchHit {
                    row: mapping[id as usize],
                    distance,
                })
                .collect::<Vec<_>>();
            expected.sort_unstable_by(|left, right| {
                left.distance
                    .total_cmp(&right.distance)
                    .then(left.row.cmp(&right.row))
            });
            expected.dedup_by_key(|hit| hit.row);
            assert_eq!(*hits, expected);
            assert_eq!(
                *hits,
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
            for hit in hits {
                let truth = exact
                    .iter()
                    .find(|truth| truth.row == hit.row)
                    .ok_or_else(|| vortex_err!("Native result absent from Flat reference"))?;
                assert!((hit.distance - truth.distance).abs() < 1e-5);
            }
            let addresses = hits.iter().map(|hit| hit.row).collect::<Vec<_>>();
            let ids = addresses
                .iter()
                .map(|row| u16::try_from(row.row_offset * 2 + u64::from(row.file_id == 11)))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|err| vortex_err!("{err}"))?;
            let taken = source
                .take(&addresses, &["id".into(), "embedding".into()])
                .await?;
            assert_eq!(taken.rows, addresses);
            assert_arrays_eq!(
                taken.data,
                data(&ids)?.into_array(),
                &mut session.create_execution_ctx()
            );
        }
        assert!(recall >= 36, "recall@10 below 0.9: {recall}/40");
        Ok(())
    })
}

#[test]
fn test_query_rejections_and_empty_batch() -> VortexResult<()> {
    block_on(|handle| async move {
        let dir = tempfile::tempdir()?;
        let (store, metadata, _) = prepare(dir.path(), &session(handle)?).await?;
        let index = provider(dir.path())?.open(&metadata, store).await?;
        let index = index
            .as_vector()
            .ok_or_else(|| vortex_err!("Vector capability"))?;
        let filter = filter(&metadata.snapshot)?;
        assert!(index.search(&[0.0], &options(5)?, &filter).await.is_err());
        assert!(
            index
                .search(&[f32::NAN; 8], &options(5)?, &filter)
                .await
                .is_err()
        );
        assert!(
            index
                .search(&[f32::INFINITY; 8], &options(5)?, &filter)
                .await
                .is_err()
        );
        let exact = VectorSearchOptions {
            mode: SearchMode::Exact,
            ..options(5)?
        };
        assert!(index.search(&vector(7), &exact, &filter).await.is_err());
        assert!(index.search_batch(&[], &exact, &filter).await.is_err());
        let restricted = RowFilter::try_new(metadata.snapshot.clone(), None, [rows()[0]].into())?;
        assert!(
            index
                .search(&vector(7), &options(5)?, &restricted)
                .await
                .is_err()
        );
        let allow_none = RowFilter::try_new(
            metadata.snapshot.clone(),
            Some(BTreeSet::new()),
            BTreeSet::new(),
        )?;
        assert!(
            index
                .search(&vector(7), &options(5)?, &allow_none)
                .await
                .is_err()
        );
        let mut different = metadata.snapshot.clone();
        different.version = "new".into();
        let stale = RowFilter::try_new(different, None, BTreeSet::new())?;
        assert!(
            index
                .search(&vector(7), &options(5)?, &stale)
                .await
                .is_err()
        );
        for json in [
            r#"{"unknown":1}"#,
            r#"{"max_check":4096,"internal_results":2,"search_pages":12}"#,
            r#"{"max_check":1,"internal_results":64,"search_pages":12}"#,
            r#"{"max_check":4096,"internal_results":64,"search_pages":13}"#,
            r#"{"max_check":4096,"internal_results":64,"search_pages":11}"#,
            r#"{"max_check":4096,"internal_results":64,"search_pages":0}"#,
        ] {
            let invalid = VectorSearchOptions {
                backend_options: Bytes::copy_from_slice(json.as_bytes()),
                ..options(5)?
            };
            assert!(index.search(&vector(7), &invalid, &filter).await.is_err());
        }
        assert!(
            index
                .search(&vector(7), &options(4097)?, &filter)
                .await
                .is_err()
        );
        assert!(
            index
                .search_batch(&[], &options(5)?, &filter)
                .await?
                .is_empty()
        );
        let batch = vec![Buffer::from(vector(7)); 257];
        assert!(
            index
                .search_batch(&batch, &options(5)?, &filter)
                .await
                .is_err()
        );
        let hits = index.search(&vector(7), &options(300)?, &filter).await?;
        assert!(hits.len() <= 256);
        Ok(())
    })
}

#[test]
fn test_read_failure_is_an_error_and_native_closes_before_lease() -> VortexResult<()> {
    block_on(|handle| async move {
        let dir = tempfile::tempdir()?;
        let root = dir.path();
        let (store, metadata, _) = prepare(root, &session(handle)?).await?;
        let closed = Arc::new(AtomicBool::new(false));
        let provider = provider(root)?;
        let no_local = Arc::new(ObservedStore {
            inner: Arc::clone(&store),
            closed_before_lease: Arc::clone(&closed),
            local: false,
        });
        assert!(provider.open(&metadata, no_local).await.is_err());
        assert_eq!(fs::read_dir(root.join("scratch"))?.count(), 0);
        let observed = Arc::new(ObservedStore {
            inner: store,
            closed_before_lease: Arc::clone(&closed),
            local: true,
        });
        let index = provider.open(&metadata, observed).await?;
        let lease = fs::read_dir(root.join("scratch"))?
            .next()
            .ok_or_else(|| vortex_err!("Missing test lease"))??
            .path();
        let posting = lease.join("spfresh/native/postings.bin");
        // Deliberate fault injection into private scratch, outside the ownership contract.
        let mut permissions = fs::metadata(&posting)?.permissions();
        permissions.set_mode(0o600);
        fs::set_permissions(&posting, permissions)?;
        OpenOptions::new().write(true).open(posting)?.set_len(0)?;
        let vector_index = index
            .as_vector()
            .ok_or_else(|| vortex_err!("Vector capability"))?;
        let filter = filter(&metadata.snapshot)?;
        for _ in 0..10 {
            assert!(
                vector_index
                    .search(&vector(7), &options(5)?, &filter)
                    .await
                    .is_err()
            );
        }
        drop(index);
        assert!(closed.load(Ordering::SeqCst));
        assert_eq!(fs::read_dir(root.join("scratch"))?.count(), 0);
        Ok(())
    })
}

#[test]
fn test_cross_process_reopen_without_source_and_environment() -> VortexResult<()> {
    let dir = tempfile::tempdir()?;
    for mode in ["build", "read", "read"] {
        let output = Command::new(env::current_exe()?)
            .args(["--exact", "tests::native::test_child"])
            .env("VORTEX_SPFRESH_CHILD_ROOT", dir.path())
            .env("VORTEX_SPFRESH_CHILD_MODE", mode)
            .env("SPFRESH_SPDK_MEM_FILE", "/does/not/exist")
            .env("SPFRESH_SPDK_PERSIST_PATH", "/does/not/exist")
            .output()?;
        assert!(
            output.status.success(),
            "child {mode}: {}\n{}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(())
}

#[test]
fn test_child() -> VortexResult<()> {
    let Ok(root) = env::var("VORTEX_SPFRESH_CHILD_ROOT") else {
        return Ok(());
    };
    let root = Path::new(&root);
    block_on(|handle| async move {
        if env::var("VORTEX_SPFRESH_CHILD_MODE").map_err(|err| vortex_err!("{err}"))? == "build" {
            let (_, _, control) = prepare(root, &session(handle)?).await?;
            fs::write(
                root.join("control.json"),
                serde_json::to_vec(&control).map_err(|err| vortex_err!("{err}"))?,
            )?;
            fs::remove_dir_all(root.join("native"))?;
            fs::remove_file(root.join("input.bin"))?;
        } else {
            let control: Control = serde_json::from_slice(&fs::read(root.join("control.json"))?)
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
            drop(index);
            assert_eq!(fs::read_dir(root.join("scratch"))?.count(), 0);
        }
        Ok(())
    })
}

#[test]
fn test_concurrent_handles_repeat_and_lease_lifetime() -> VortexResult<()> {
    block_on(|handle| async move {
        let dir = tempfile::tempdir()?;
        let root = dir.path();
        let (store, metadata, _) = prepare(root, &session(handle)?).await?;
        let provider = provider(root)?;
        let first = provider
            .open(&metadata, Arc::clone(&store) as Arc<dyn IndexStore>)
            .await?;
        let second = provider
            .open(&metadata, Arc::clone(&store) as Arc<dyn IndexStore>)
            .await?;
        assert_eq!(fs::read_dir(root.join("scratch"))?.count(), 2);
        // Already-open handles depend on their private copies, not the source/store.
        drop(store);
        fs::remove_dir_all(root.join("store"))?;
        fs::remove_dir_all(root.join("native"))?;
        let mut threads = Vec::new();
        for worker in 0..4 {
            let index = if worker % 2 == 0 {
                Arc::clone(&first)
            } else {
                Arc::clone(&second)
            };
            let filter = filter(&metadata.snapshot)?;
            threads.push(thread::spawn(move || -> VortexResult<()> {
                let index = index.as_vector().ok_or_else(|| vortex_err!("Vector capability"))?;
                for repeat in 0..20 {
                    let internal = if repeat % 2 == 0 { 16 } else { 128 };
                    let options = VectorSearchOptions {
                        backend_options: Bytes::from(format!(r#"{{"max_check":4096,"internal_results":{internal},"search_pages":12}}"#)),
                        ..options(5)?
                    };
                    let hits = sync(index.search(&vector(7), &options, &filter))?;
                    assert_eq!(hits.first().map(|hit| hit.row), Some(rows()[7]));
                }
                Ok(())
            }));
        }
        for thread in threads {
            thread
                .join()
                .map_err(|_| vortex_err!("Native search thread panicked"))??;
        }
        drop(first);
        assert_eq!(fs::read_dir(root.join("scratch"))?.count(), 1);
        drop(second);
        assert_eq!(fs::read_dir(root.join("scratch"))?.count(), 0);
        Ok(())
    })
}

#[test]
fn test_inventory_metadata_limits_and_corruption() -> VortexResult<()> {
    block_on(|handle| async move {
        let dir = tempfile::tempdir()?;
        let root = dir.path();
        let (store, metadata, _) = prepare(root, &session(handle)?).await?;
        let provider = provider(root)?;
        assert!(provider.builder().is_none());
        for change in 0..4 {
            let mut invalid = metadata.clone();
            match change {
                0 => {
                    invalid.artifacts.pop();
                }
                1 => {
                    invalid.snapshot.version = "different".into();
                }
                2 => {
                    invalid.covered_files.pop();
                }
                _ => {
                    invalid.fields = vec!["other".into()];
                }
            }
            assert!(
                provider
                    .open(&invalid, Arc::clone(&store) as Arc<dyn IndexStore>)
                    .await
                    .is_err()
            );
            assert_eq!(fs::read_dir(root.join("scratch"))?.count(), 0);
        }
        let bounded = SpFreshProvider::try_new(
            root.join("scratch"),
            SpFreshLimits {
                max_rows: 255,
                ..SpFreshLimits::default()
            },
        )?;
        assert!(
            bounded
                .open(&metadata, Arc::clone(&store) as Arc<dyn IndexStore>)
                .await
                .is_err()
        );
        let bounded = SpFreshProvider::try_new(
            root.join("scratch"),
            SpFreshLimits {
                max_materialized_bytes: 1,
                ..SpFreshLimits::default()
            },
        )?;
        assert!(
            bounded
                .open(&metadata, Arc::clone(&store) as Arc<dyn IndexStore>)
                .await
                .is_err()
        );
        for artifact in &metadata.artifacts {
            let path = root
                .join("store/generation-1/artifacts")
                .join(&artifact.path);
            let original = fs::read(&path)?;
            let mut corrupt = original.clone();
            corrupt[0] ^= 1;
            fs::write(&path, corrupt)?;
            assert!(
                provider
                    .open(&metadata, Arc::clone(&store) as Arc<dyn IndexStore>)
                    .await
                    .is_err(),
                "{}",
                artifact.path
            );
            assert_eq!(fs::read_dir(root.join("scratch"))?.count(), 0);
            fs::write(path, original)?;
        }
        // Valid checksums are not sufficient: reject semantic mapping/descriptor errors.
        for (generation, target) in [
            ("bad-map", ROWS),
            ("bad-descriptor", DESCRIPTOR),
            ("bad-native", "spfresh/native/postings.bin"),
            ("bad-graph", "spfresh/native/graph.bin"),
            ("bad-tree", "spfresh/native/tree.bin"),
        ] {
            let bad = Arc::new(LocalIndexStore::create(
                root.join("store"),
                generation,
                LIMITS,
            )?);
            let mut invalid = metadata.clone();
            invalid.generation = generation.into();
            invalid.artifacts.clear();
            for artifact in &metadata.artifacts {
                let mut bytes = store.read(artifact).await?.to_vec();
                if artifact.path == target {
                    if target == ROWS {
                        bytes[16..32].copy_from_slice(&[0; 16]);
                    } else if target == DESCRIPTOR {
                        let mut descriptor: serde_json::Value =
                            serde_json::from_slice(&bytes).map_err(|err| vortex_err!("{err}"))?;
                        descriptor["bundle"]["dimension"] = 7.into();
                        bytes =
                            serde_json::to_vec(&descriptor).map_err(|err| vortex_err!("{err}"))?;
                    } else if generation == "bad-graph" {
                        bytes[8..12].copy_from_slice(&i32::MIN.to_le_bytes());
                    } else if generation == "bad-tree" {
                        // Fixture has one tree; root child start cannot point back to itself.
                        bytes[16..20].copy_from_slice(&0i32.to_le_bytes());
                    } else {
                        bytes[..4].copy_from_slice(&i32::MAX.to_le_bytes());
                    }
                }
                invalid
                    .artifacts
                    .push(bad.write(&artifact.path, Bytes::from(bytes)).await?);
            }
            bad.seal(&invalid)?;
            assert!(provider.open(&invalid, bad).await.is_err());
            assert_eq!(fs::read_dir(root.join("scratch"))?.count(), 0);
        }
        Ok(())
    })
}
