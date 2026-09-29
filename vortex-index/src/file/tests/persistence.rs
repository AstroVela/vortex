// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Test-only persisted Flat provider. This is not a supported backend or format.

use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::num::NonZeroUsize;
use std::os::unix::process::ExitStatusExt;
use std::path::Path;
use std::process::Command;
use std::process::Stdio;
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use std::time::Instant;

use async_trait::async_trait;
use bytes::Bytes;
use futures::TryStreamExt;
use serde::Deserialize;
use serde::Serialize;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_io::runtime::single::block_on;
use vortex_session::VortexSession;

use super::BUDGET;
use super::Fixture;
use super::data;
use super::fields;
use super::row;
use super::test_session;
use crate::DistanceMetric;
use crate::FlatIndex;
use crate::Index;
use crate::IndexBuildRequest;
use crate::IndexBuilder;
use crate::IndexMetadata;
use crate::IndexProvider;
use crate::IndexRegistry;
use crate::IndexSource;
use crate::IndexStore;
use crate::RowAddress;
use crate::RowFilter;
use crate::SearchHit;
use crate::SearchMode;
use crate::Snapshot;
use crate::VectorIndex;
use crate::VectorSearchOptions;
use crate::VectorSpec;
use crate::file::LocalFileSource;
use crate::store::LocalGeneration;
use crate::store::LocalIndexStore;
use crate::store::LocalStoreLimits;

const PROVIDER: &str = "test.flat.persisted";
const CHILD: &str = "file::tests::persistence::test_persistence_child";
const CHILD_ROOT: &str = "VORTEX_INDEX_TEST_ROOT";
const CHILD_MODE: &str = "VORTEX_INDEX_TEST_MODE";
const LIMITS: LocalStoreLimits = LocalStoreLimits {
    max_artifact_bytes: BUDGET,
    max_manifest_bytes: 1024 * 1024,
};

#[derive(Serialize, Deserialize)]
struct Control {
    snapshot: Snapshot,
}

#[derive(Serialize, Deserialize)]
struct Vectors {
    rows: Vec<RowAddress>,
    values: Vec<f32>,
}

#[derive(Debug)]
struct PersistedFlat {
    metadata: IndexMetadata,
    flat: FlatIndex,
}

impl PersistedFlat {
    fn new(metadata: IndexMetadata, vectors: Vectors) -> VortexResult<Self> {
        let mut memory = metadata.clone();
        memory.backend = FlatIndex::ID.into();
        memory.artifacts.clear();
        let flat = FlatIndex::try_new(
            memory,
            VectorSpec {
                dimension: NonZeroUsize::new(2).ok_or_else(|| vortex_err!("Test dimension"))?,
                metric: DistanceMetric::SquaredL2,
            },
            vectors.rows,
            Buffer::from(vectors.values),
        )?;
        Ok(Self { metadata, flat })
    }
}

impl Index for PersistedFlat {
    fn metadata(&self) -> &IndexMetadata {
        &self.metadata
    }
    fn as_vector(&self) -> Option<&dyn VectorIndex> {
        Some(self)
    }
}

#[async_trait]
impl VectorIndex for PersistedFlat {
    fn spec(&self) -> VectorSpec {
        self.flat.spec()
    }
    fn supports_exact(&self) -> bool {
        true
    }
    fn supports_filter(&self) -> bool {
        true
    }

    async fn search(
        &self,
        query: &[f32],
        options: &VectorSearchOptions,
        filter: &RowFilter,
    ) -> VortexResult<Vec<SearchHit>> {
        self.flat.search(query, options, filter).await
    }
}

struct ReferenceProvider {
    session: VortexSession,
}

#[async_trait]
impl IndexProvider for ReferenceProvider {
    fn id(&self) -> &str {
        PROVIDER
    }
    fn supports_version(&self, version: u32) -> bool {
        version == 1
    }
    fn builder(&self) -> Option<&dyn IndexBuilder> {
        Some(self)
    }

    async fn open(
        &self,
        metadata: &IndexMetadata,
        store: Arc<dyn IndexStore>,
    ) -> VortexResult<Arc<dyn Index>> {
        metadata.validate_for(&metadata.snapshot)?;
        if metadata.backend != PROVIDER
            || metadata.backend_version != 1
            || metadata.artifacts.len() != 1
        {
            vortex_bail!("Invalid test reference metadata");
        }
        let bytes = store.read(&metadata.artifacts[0]).await?;
        let vectors: Vectors =
            serde_json::from_slice(&bytes).map_err(|err| vortex_err!("{}", err))?;
        Ok(Arc::new(PersistedFlat::new(metadata.clone(), vectors)?))
    }
}

#[async_trait]
impl IndexBuilder for ReferenceProvider {
    async fn build(
        &self,
        request: IndexBuildRequest,
        source: Arc<dyn IndexSource>,
        store: Arc<dyn IndexStore>,
    ) -> VortexResult<IndexMetadata> {
        let mut metadata = request.metadata;
        metadata.validate_for(source.snapshot())?;
        if metadata.backend != PROVIDER
            || metadata.backend_version != 1
            || !metadata.artifacts.is_empty()
            || metadata.fields != fields(&["embedding"])
            || !request.backend_options.is_empty()
        {
            vortex_bail!("Unsupported test reference build request");
        }
        let mut vectors = Vectors {
            rows: Vec::new(),
            values: Vec::new(),
        };
        let mut ctx = self.session.create_execution_ctx();
        let mut stream = source.scan(&metadata.covered_files, &metadata.fields)?;
        while let Some(batch) = stream.try_next().await? {
            let data = batch.data.execute::<StructArray>(&mut ctx)?;
            let lists = data
                .unmasked_field(0)
                .clone()
                .execute::<FixedSizeListArray>(&mut ctx)?;
            let expected = DType::FixedSizeList(
                Arc::new(DType::Primitive(PType::F32, Nullability::NonNullable)),
                2,
                Nullability::NonNullable,
            );
            if lists.dtype() != &expected {
                vortex_bail!("Test provider requires non-nullable 2D f32 vectors");
            }
            let values = lists
                .elements()
                .clone()
                .execute::<PrimitiveArray>(&mut ctx)?;
            vectors.rows.extend(batch.rows);
            vectors.values.extend_from_slice(values.as_slice::<f32>());
        }
        // Check dense coverage, dimensions and finite values before persisting.
        let bytes = serde_json::to_vec(&vectors).map_err(|err| vortex_err!("{}", err))?;
        PersistedFlat::new(metadata.clone(), vectors)?;
        metadata.artifacts.push(
            store
                .write("reference/vectors.json", Bytes::from(bytes))
                .await?,
        );
        Ok(metadata)
    }
}

fn json<T: Serialize>(value: &T) -> VortexResult<Vec<u8>> {
    serde_json::to_vec(value).map_err(|err| vortex_err!("{}", err))
}

fn run_child(root: &Path, mode: &str) -> VortexResult<()> {
    let output = Command::new(env::current_exe()?)
        .args(["--exact", CHILD, "--nocapture"])
        .env(CHILD_ROOT, root)
        .env(CHILD_MODE, mode)
        .output()?;
    if !output.status.success() {
        vortex_bail!(
            "Persistence child {} failed: {}\n{}",
            mode,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(())
}

#[test]
fn test_build_exit_reopen_search_and_take_in_separate_processes() -> VortexResult<()> {
    block_on(|handle| async move {
        let fixture = Fixture::new(handle).await?;
        let root = fixture.dir.path().canonicalize()?;
        fs::write(
            root.join("control.json"),
            json(&Control {
                snapshot: fixture.snapshot,
            })?,
        )?;
        run_child(&root, "build")?;
        run_child(&root, "search")?;
        // A second fresh reader must not need to modify or rebuild the generation.
        run_child(&root, "search")?;
        Ok(())
    })
}

#[test]
fn test_killed_builder_leaves_only_an_unpublished_generation() -> VortexResult<()> {
    block_on(|handle| async move {
        let fixture = Fixture::new(handle).await?;
        let root = fixture.dir.path().canonicalize()?;
        fs::write(
            root.join("control.json"),
            json(&Control {
                snapshot: fixture.snapshot,
            })?,
        )?;
        let mut child = Command::new(env::current_exe()?)
            .args(["--exact", CHILD, "--nocapture"])
            .env(CHILD_ROOT, &root)
            .env(CHILD_MODE, "interrupt")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()?;
        let deadline = Instant::now() + Duration::from_secs(15);
        let ready = root.join("ready.json");
        let ready_result = (|| -> VortexResult<()> {
            while !ready.exists() {
                if let Some(status) = child.try_wait()? {
                    vortex_bail!("Interrupted builder exited before ready: {}", status);
                }
                if Instant::now() >= deadline {
                    vortex_bail!("Interrupted builder did not become ready");
                }
                thread::sleep(Duration::from_millis(10));
            }
            Ok(())
        })();
        // Always reap the child, even when readiness fails.
        let killed = child.kill();
        let status = child.wait();
        ready_result?;
        killed?;
        assert_eq!(status?.signal(), Some(9));
        assert!(!root.join("descriptor.json").exists());
        assert!(!root.join("persisted-flat/manifest.json").exists());
        assert!(LocalIndexStore::create(&root, "persisted-flat", LIMITS).is_err());
        Ok(())
    })
}

#[test]
fn test_persistence_child() -> VortexResult<()> {
    let Some(root) = env::var_os(CHILD_ROOT) else {
        return Ok(());
    };
    let root = Path::new(&root);
    let mode = env::var(CHILD_MODE).map_err(|err| vortex_err!("{}", err))?;
    block_on(|handle| async move {
        let control: Control = serde_json::from_slice(&fs::read(root.join("control.json"))?)
            .map_err(|err| vortex_err!("{}", err))?;
        let session = test_session(handle)?;
        let source = Arc::new(
            LocalFileSource::open(
                control.snapshot.clone(),
                data(&[])?.dtype().clone(),
                BUDGET,
                session.clone(),
            )
            .await?,
        );
        let mut registry = IndexRegistry::default();
        registry.register(Arc::new(ReferenceProvider {
            session: session.clone(),
        }))?;
        match mode.as_str() {
            "build" | "interrupt" => {
                let store = Arc::new(LocalIndexStore::create(root, "persisted-flat", LIMITS)?);
                let request = IndexBuildRequest {
                    metadata: IndexMetadata {
                        format_version: 1,
                        name: "embedding-index".into(),
                        generation: "persisted-flat".into(),
                        backend: PROVIDER.into(),
                        backend_version: 1,
                        snapshot: control.snapshot,
                        fields: fields(&["embedding"]),
                        covered_files: vec![10, 30],
                        artifacts: vec![],
                    },
                    backend_options: Bytes::new(),
                };
                let provider = registry.provider(PROVIDER)?;
                let builder = provider
                    .builder()
                    .ok_or_else(|| vortex_err!("Missing builder"))?;
                let metadata = builder
                    .build(request, source, Arc::clone(&store) as Arc<dyn IndexStore>)
                    .await?;
                if mode == "interrupt" {
                    fs::write(root.join("ready.json"), json(&metadata)?)?;
                    loop {
                        thread::park();
                    }
                }
                let descriptor = store.seal(&metadata)?;
                fs::write(root.join("descriptor.json"), json(&descriptor)?)?;
            }
            "search" => {
                let descriptor: LocalGeneration =
                    serde_json::from_slice(&fs::read(root.join("descriptor.json"))?)
                        .map_err(|err| vortex_err!("{}", err))?;
                let (store, metadata) =
                    LocalIndexStore::open(root, &descriptor, &control.snapshot, LIMITS)?;
                let index = registry
                    .open(&metadata, source.snapshot(), Arc::new(store))
                    .await?;
                let index = index
                    .as_vector()
                    .ok_or_else(|| vortex_err!("Missing vector index"))?;
                let options = VectorSearchOptions {
                    k: NonZeroUsize::new(5).ok_or_else(|| vortex_err!("Test k"))?,
                    mode: SearchMode::Exact,
                    backend_options: Bytes::new(),
                };
                let filter = RowFilter::try_new(control.snapshot, None, BTreeSet::new())?;
                let hits = index.search(&[25., 0.], &options, &filter).await?;
                let rows: Vec<_> = hits.iter().map(|hit| hit.row).collect();
                assert_eq!(
                    rows,
                    vec![row(30, 0), row(30, 1), row(30, 2), row(10, 4), row(10, 3)]
                );
                assert_eq!(
                    hits.iter().map(|hit| hit.distance).collect::<Vec<_>>(),
                    vec![25., 36., 49., 121., 144.]
                );
                let batch = source.take(&rows, &fields(&["id"])).await?;
                assert_eq!(batch.rows, rows);
                assert_arrays_eq!(
                    batch.data,
                    data(&[30, 31, 32, 14, 13])?.project(&["id".into()])?,
                    &mut session.create_execution_ctx()
                );
                let filter = RowFilter::try_new(
                    source.snapshot().clone(),
                    Some([row(10, 4), row(30, 0)].into()),
                    [row(30, 0)].into(),
                )?;
                let hits = index.search(&[25., 0.], &options, &filter).await?;
                assert_eq!(
                    hits.iter().map(|hit| hit.row).collect::<Vec<_>>(),
                    vec![row(10, 4)]
                );
            }
            _ => vortex_bail!("Unknown persistence child mode"),
        }
        Ok(())
    })
}
