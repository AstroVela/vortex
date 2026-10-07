// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::fs;
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Arc;
use std::sync::Barrier;
use std::thread;

use async_trait::async_trait;
use bytes::Bytes;
use futures::executor::block_on;
use futures::stream;
use futures::stream::BoxStream;
use rstest::rstest;
use vortex::VortexSessionDefault;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_index::FlatIndex;
use vortex_index::IndexBuildRequest;
use vortex_index::IndexBuilder;
use vortex_index::IndexMetadata;
use vortex_index::IndexProvider;
use vortex_index::IndexSource;
use vortex_index::IndexStore;
use vortex_index::RowAddress;
use vortex_index::RowFilter;
use vortex_index::SearchMode;
use vortex_index::Snapshot;
use vortex_index::SourceBatch;
use vortex_index::VectorIndex;
use vortex_index::VectorSearchOptions;
use vortex_index::store::LocalIndexStore;
use vortex_index::store::LocalStoreLimits;
use vortex_session::VortexSession;

use super::metadata;
use super::rows;
use crate::HnswBuildLimits;
use crate::HnswBuildOptions;
use crate::HnswIndexBuilder;
use crate::HnswLimits;
use crate::HnswProvider;
use crate::ffi::Native;
use crate::validate;

const LIMITS: LocalStoreLimits = LocalStoreLimits {
    max_artifact_bytes: 1024 * 1024,
    max_manifest_bytes: 1024 * 1024,
};

fn config() -> HnswBuildOptions {
    HnswBuildOptions {
        format_version: 1,
        dimension: 8,
        m: 8,
        ef_construction: 64,
        seed: 100,
        threads: 1,
    }
}

fn vector(id: u32) -> Vec<f32> {
    vector_at_dimension(id, 8)
}

fn vector_at_dimension(id: u32, dimension: u32) -> Vec<f32> {
    (0..dimension)
        .map(|component| ((id * 97 + component * 53 + id * component) % 997) as f32 / 997.0)
        .collect()
}

fn options(k: usize, json: &str) -> VortexResult<VectorSearchOptions> {
    Ok(VectorSearchOptions {
        k: NonZeroUsize::new(k).ok_or_else(|| vortex_err!("Zero k"))?,
        mode: SearchMode::Approximate,
        backend_options: Bytes::copy_from_slice(json.as_bytes()),
    })
}

struct Source {
    metadata: IndexMetadata,
    data: ArrayRef,
}

#[async_trait]
impl IndexSource for Source {
    fn snapshot(&self) -> &Snapshot {
        &self.metadata.snapshot
    }
    fn scan(
        &self,
        files: &[u64],
        fields: &[String],
    ) -> VortexResult<BoxStream<'static, VortexResult<SourceBatch>>> {
        if files != self.metadata.covered_files || fields != self.metadata.fields {
            vortex_bail!("Unexpected projection");
        }
        Ok(Box::pin(stream::iter([Ok(SourceBatch {
            rows: rows(256),
            data: self.data.clone(),
        })])))
    }
    async fn take(&self, _rows: &[RowAddress], _fields: &[String]) -> VortexResult<SourceBatch> {
        vortex_bail!("Test source does not implement take");
    }
}

fn source(values: ArrayRef) -> VortexResult<Arc<dyn IndexSource>> {
    let data = StructArray::try_new(
        ["embedding"].into(),
        vec![values],
        256,
        Validity::NonNullable,
    )?
    .into_array();
    Ok(Arc::new(Source {
        metadata: metadata(256),
        data,
    }))
}

fn data() -> VortexResult<Arc<dyn IndexSource>> {
    source(
        FixedSizeListArray::try_new(
            PrimitiveArray::new(
                Buffer::from_iter((0..256).flat_map(vector)),
                Validity::NonNullable,
            )
            .into_array(),
            8,
            Validity::NonNullable,
            256,
        )?
        .into_array(),
    )
}

fn builder(root: &Path) -> VortexResult<HnswIndexBuilder> {
    HnswIndexBuilder::try_new(
        root.to_path_buf(),
        VortexSession::default(),
        HnswBuildLimits::default(),
    )
}

fn request(json: Bytes) -> IndexBuildRequest {
    IndexBuildRequest {
        metadata: metadata(256),
        backend_options: json,
    }
}

async fn prepare(root: &Path) -> VortexResult<(Arc<LocalIndexStore>, IndexMetadata)> {
    fs::create_dir(root.join("store"))?;
    let store = Arc::new(LocalIndexStore::create(
        root.join("store"),
        "generation-1",
        LIMITS,
    )?);
    let built = builder(root)?
        .build(
            request(Bytes::from(
                serde_json::to_vec(&config()).map_err(|err| vortex_err!("{err}"))?,
            )),
            data()?,
            Arc::clone(&store) as Arc<dyn IndexStore>,
        )
        .await?;
    assert_eq!(built.artifacts.len(), 3);
    let provider = HnswProvider::try_new(root.to_path_buf(), HnswLimits::default())?;
    assert!(
        provider
            .open(&built, Arc::clone(&store) as Arc<dyn IndexStore>)
            .await
            .is_err()
    );
    let generation = store.seal(&built)?;
    let (reopened, read_metadata) =
        LocalIndexStore::open(root.join("store"), &generation, &built.snapshot, LIMITS)?;
    assert_eq!(built, read_metadata);
    Ok((Arc::new(reopened), built))
}

#[test]
fn build_seal_reopen_parity_and_lease_cleanup() -> VortexResult<()> {
    block_on(async {
        let root = tempfile::tempdir()?;
        let (store, built) = prepare(root.path()).await?;
        let provider = HnswProvider::try_new(root.path().to_path_buf(), HnswLimits::default())?
            .with_builder(builder(root.path())?);
        assert!(provider.builder().is_some());
        let index = provider.open(&built, store).await?;
        let vector_index = index
            .as_vector()
            .ok_or_else(|| vortex_err!("Missing vector capability"))?;
        assert!(!vector_index.supports_exact());
        assert!(!vector_index.supports_filter());
        let filter = RowFilter::try_new(built.snapshot.clone(), None, BTreeSet::new())?;
        let mut flat_metadata = built;
        flat_metadata.backend = FlatIndex::ID.into();
        flat_metadata.artifacts.clear();
        let flat = FlatIndex::try_new(
            flat_metadata,
            vector_index.spec(),
            rows(256),
            Buffer::from_iter((0..256).flat_map(vector)),
        )?;
        for id in [7, 80, 129, 242] {
            let hits = vector_index
                .search(&vector(id), &options(10, "{\"ef\":256}")?, &filter)
                .await?;
            let exact = flat
                .search(
                    &vector(id),
                    &VectorSearchOptions {
                        mode: SearchMode::Exact,
                        ..options(10, "")?
                    },
                    &filter,
                )
                .await?;
            assert_eq!(hits.len(), exact.len());
            for (hit, truth) in hits.iter().zip(exact) {
                assert_eq!(hit.row, truth.row);
                assert!((hit.distance - truth.distance).abs() < 1e-5);
            }
        }
        assert!(
            vector_index
                .search_batch(&[], &options(10, "")?, &filter)
                .await?
                .is_empty()
        );
        let too_many = vector_index
            .search(&vector(7), &options(300, "")?, &filter)
            .await?;
        assert_eq!(too_many.len(), 256);
        drop(index);
        assert_eq!(fs::read_dir(root.path())?.count(), 1);
        Ok(())
    })
}

#[rstest]
#[case("{\"ef\":9}")]
#[case("{\"ef\":1048577}")]
#[case("{\"ef\":64,\"unknown\":1}")]
#[case("{\"ef\":-1}")]
#[case("{}")]
#[case("null")]
fn rejects_query_options(#[case] json: &str) -> VortexResult<()> {
    block_on(async {
        let root = tempfile::tempdir()?;
        let (store, built) = prepare(root.path()).await?;
        let index = HnswProvider::try_new(root.path().to_path_buf(), HnswLimits::default())?
            .open(&built, store)
            .await?;
        let filter = RowFilter::try_new(built.snapshot, None, BTreeSet::new())?;
        assert!(
            index
                .as_vector()
                .ok_or_else(|| vortex_err!("Vector capability"))?
                .search(&vector(7), &options(10, json)?, &filter)
                .await
                .is_err()
        );
        Ok(())
    })
}

#[test]
fn rejects_modes_filters_vectors_and_limits() -> VortexResult<()> {
    block_on(async {
        let root = tempfile::tempdir()?;
        let (store, built) = prepare(root.path()).await?;
        let limits = HnswLimits {
            max_materialized_bytes: 16,
            ..HnswLimits::default()
        };
        assert!(
            HnswProvider::try_new(root.path().to_path_buf(), limits)?
                .open(&built, Arc::clone(&store) as Arc<dyn IndexStore>)
                .await
                .is_err()
        );
        let index = HnswProvider::try_new(root.path().to_path_buf(), HnswLimits::default())?
            .open(&built, store)
            .await?;
        let vector_index = index
            .as_vector()
            .ok_or_else(|| vortex_err!("Vector capability"))?;
        let filter = RowFilter::try_new(built.snapshot.clone(), None, BTreeSet::new())?;
        assert!(
            vector_index
                .search(&[0.0], &options(10, "")?, &filter)
                .await
                .is_err()
        );
        assert!(
            vector_index
                .search(&[f32::NAN; 8], &options(10, "")?, &filter)
                .await
                .is_err()
        );
        assert!(
            vector_index
                .search(
                    &vector(7),
                    &VectorSearchOptions {
                        mode: SearchMode::Exact,
                        ..options(10, "")?
                    },
                    &filter
                )
                .await
                .is_err()
        );
        let restricted = RowFilter::try_new(built.snapshot, None, BTreeSet::from([rows(256)[7]]))?;
        assert!(
            vector_index
                .search(&vector(7), &options(10, "")?, &restricted)
                .await
                .is_err()
        );
        Ok(())
    })
}

#[rstest]
#[case(false)]
#[case(true)]
fn rejects_wrong_or_nullable_vector_dtype(#[case] nullable: bool) -> VortexResult<()> {
    block_on(async {
        let root = tempfile::tempdir()?;
        let values = PrimitiveArray::new(
            Buffer::from(vec![0f32; 256]),
            if nullable {
                Validity::AllValid
            } else {
                Validity::NonNullable
            },
        )
        .into_array();
        let store: Arc<dyn IndexStore> = Arc::new(LocalIndexStore::create(
            root.path(),
            "generation-1",
            LIMITS,
        )?);
        assert!(
            builder(root.path())?
                .build(
                    request(Bytes::from(
                        serde_json::to_vec(&config()).map_err(|err| vortex_err!("{err}"))?
                    )),
                    source(values)?,
                    store
                )
                .await
                .is_err()
        );
        Ok(())
    })
}

#[rstest]
#[case(0, u64::MAX)]
#[case(8, 1)]
#[case(24, u64::MAX)]
#[case(40, u64::MAX)]
#[case(72, 65)]
#[case(88, 4097)]
#[case(96, u64::MAX)]
fn rejects_native_structure_before_deserialization(
    #[case] offset: usize,
    #[case] value: u64,
) -> VortexResult<()> {
    let root = tempfile::tempdir()?;
    let path = root.path().join("index.bin");
    let bundle = config().bundle(256);
    Native::build(
        &path,
        &(0..256).flat_map(vector).collect::<Vec<_>>(),
        bundle,
        &config(),
    )?;
    validate::native_file(&path, bundle)?;
    let mut bytes = fs::read(&path)?;
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    fs::write(&path, bytes)?;
    assert!(Native::open(&path, bundle).is_err());
    Ok(())
}

#[test]
fn concurrent_search_ef_is_handle_local() -> VortexResult<()> {
    block_on(async {
        let root = tempfile::tempdir()?;
        let (store, built) = prepare(root.path()).await?;
        let index = HnswProvider::try_new(root.path().to_path_buf(), HnswLimits::default())?
            .open(&built, store)
            .await?;
        let filter = RowFilter::try_new(built.snapshot, None, BTreeSet::new())?;
        thread::scope(|scope| -> VortexResult<()> {
            let handles = [32, 64, 128, 256].map(|ef| {
                let index = Arc::clone(&index);
                let filter = filter.clone();
                scope.spawn(move || -> VortexResult<()> {
                    for _ in 0..20 {
                        let hits = block_on(
                            index
                                .as_vector()
                                .ok_or_else(|| vortex_err!("Vector capability"))?
                                .search(
                                    &vector(7),
                                    &options(10, &format!("{{\"ef\":{ef}}}"))?,
                                    &filter,
                                ),
                        )?;
                        assert_eq!(hits[0].row, rows(256)[7]);
                        assert_eq!(hits[0].distance, 0.0);
                    }
                    Ok(())
                })
            });
            for handle in handles {
                handle
                    .join()
                    .map_err(|_| vortex_err!("Search thread panicked"))??;
            }
            Ok(())
        })
    })
}

#[rstest]
#[case(1)]
#[case(3)]
#[case(4)]
#[case(5)]
#[case(8)]
#[case(15)]
#[case(16)]
#[case(17)]
#[case(20)]
#[case(31)]
#[case(32)]
#[case(128)]
#[case(129)]
fn native_distance_kernels_match_scalar_l2(#[case] dimension: u32) -> VortexResult<()> {
    let root = tempfile::tempdir()?;
    let path = root.path().join("index.bin");
    let config = HnswBuildOptions {
        dimension,
        ..config()
    };
    let bundle = config.bundle(256);
    let vectors = (0..256)
        .flat_map(|id| vector_at_dimension(id, dimension))
        .collect::<Vec<_>>();
    Native::build(&path, &vectors, bundle, &config)?;
    let index = Native::open(&path, bundle)?;
    for id in [7, 257] {
        let query = vector_at_dimension(id, dimension);
        let hits = index.search(&query, 256, 256)?;
        assert_eq!(hits.len(), 256);
        assert_eq!(
            hits.iter().map(|hit| hit.0).collect::<BTreeSet<_>>().len(),
            256
        );
        for (row, distance) in hits {
            let expected = query
                .iter()
                .zip(vector_at_dimension(row, dimension))
                .map(|(left, right)| (left - right).powi(2))
                .sum::<f32>();
            assert!((distance - expected).abs() <= 1e-5 * expected.max(1.0));
        }
    }
    Ok(())
}

#[rstest]
#[case(17)]
#[case(128)]
#[case(129)]
fn independent_build_open_and_search_can_overlap(#[case] dimension: u32) -> VortexResult<()> {
    let root = tempfile::tempdir()?;
    let path = root.path().join("index.bin");
    let config = HnswBuildOptions {
        dimension,
        ..config()
    };
    let bundle = config.bundle(256);
    let vectors = (0..256)
        .flat_map(|id| vector_at_dimension(id, dimension))
        .collect::<Vec<_>>();
    Native::build(&path, &vectors, bundle, &config)?;
    let index = Native::open(&path, bundle)?;
    let query = vector_at_dimension(7, dimension);
    let expected = index.search(&query, 10, 256)?;
    assert_eq!(expected[0], (7, 0.0));
    let barrier = Barrier::new(4);
    thread::scope(|scope| -> VortexResult<()> {
        let handles = (0..4)
            .map(|worker| {
                let barrier = &barrier;
                let index = &index;
                let query = &query;
                let expected = &expected;
                let path = &path;
                let root = root.path();
                let config = &config;
                let vectors = &vectors;
                scope.spawn(move || -> VortexResult<()> {
                    barrier.wait();
                    for _ in 0..16 {
                        if worker < 2 {
                            let built_path = root.join(format!("built-{worker}.bin"));
                            Native::build(&built_path, vectors, bundle, config)?;
                            let opened = Native::open(&built_path, bundle)?;
                            assert_eq!(opened.search(query, 10, 256)?, *expected);
                        } else if worker == 2 {
                            let opened = Native::open(path, bundle)?;
                            assert_eq!(opened.search(query, 10, 256)?, *expected);
                        } else {
                            for _ in 0..32 {
                                assert_eq!(index.search(query, 10, 256)?, *expected);
                            }
                        }
                    }
                    Ok(())
                })
            })
            .collect::<Vec<_>>();
        for handle in handles {
            handle
                .join()
                .map_err(|_| vortex_err!("Native worker panicked"))??;
        }
        Ok(())
    })
}

#[test]
fn equal_distances_are_ordered_by_physical_address() -> VortexResult<()> {
    block_on(async {
        let root = tempfile::tempdir()?;
        let store = Arc::new(LocalIndexStore::create(
            root.path(),
            "generation-1",
            LIMITS,
        )?);
        let values = FixedSizeListArray::try_new(
            PrimitiveArray::new(Buffer::from(vec![0f32; 256 * 8]), Validity::NonNullable)
                .into_array(),
            8,
            Validity::NonNullable,
            256,
        )?
        .into_array();
        let built = builder(root.path())?
            .build(
                request(Bytes::from(
                    serde_json::to_vec(&config()).map_err(|err| vortex_err!("{err}"))?,
                )),
                source(values)?,
                Arc::clone(&store) as Arc<dyn IndexStore>,
            )
            .await?;
        store.seal(&built)?;
        let index = HnswProvider::try_new(root.path().to_path_buf(), HnswLimits::default())?
            .open(&built, store)
            .await?;
        let filter = RowFilter::try_new(built.snapshot, None, BTreeSet::new())?;
        let hits = index
            .as_vector()
            .ok_or_else(|| vortex_err!("Vector capability"))?
            .search(&[0f32; 8], &options(256, "{\"ef\":256}")?, &filter)
            .await?;
        let mut expected = hits.iter().map(|hit| hit.row).collect::<Vec<_>>();
        expected.sort_unstable();
        assert!(!hits.is_empty());
        assert!(hits.len() <= 256);
        assert_eq!(hits.iter().map(|hit| hit.row).collect::<Vec<_>>(), expected);
        assert!(hits.iter().all(|hit| hit.distance == 0.0));
        Ok(())
    })
}

#[rstest]
#[case("format_version", 2)]
#[case("dimension", 0)]
#[case("m", 1)]
#[case("ef_construction", 7)]
#[case("threads", 0)]
#[case::parallel_two("threads", 2)]
#[case::parallel_four("threads", 4)]
#[case::parallel_eight("threads", 8)]
#[case("threads", 9)]
#[case("seed", u32::MAX)]
#[case("unknown", 1)]
fn invalid_build_options_are_rejected(#[case] key: &str, #[case] value: u32) -> VortexResult<()> {
    block_on(async {
        let root = tempfile::tempdir()?;
        let mut json = serde_json::to_value(config()).map_err(|err| vortex_err!("{err}"))?;
        json[key] = value.into();
        let store: Arc<dyn IndexStore> = Arc::new(LocalIndexStore::create(
            root.path(),
            "generation-1",
            LIMITS,
        )?);
        assert!(
            builder(root.path())?
                .build(
                    request(Bytes::from(
                        serde_json::to_vec(&json).map_err(|err| vortex_err!("{err}"))?
                    )),
                    data()?,
                    store,
                )
                .await
                .is_err()
        );
        assert!(!root.path().join("generation-1/manifest.json").exists());
        Ok(())
    })
}

#[rstest]
#[case(128, 8192, 1048576)]
#[case(256, 8191, 1048576)]
#[case(256, 8192, 16)]
fn construction_limits_leave_no_published_generation(
    #[case] max_rows: u32,
    #[case] max_vector_bytes: u64,
    #[case] max_native_artifact_bytes: u64,
) -> VortexResult<()> {
    block_on(async {
        let root = tempfile::tempdir()?;
        let builder = HnswIndexBuilder::try_new(
            root.path().to_path_buf(),
            VortexSession::default(),
            HnswBuildLimits {
                max_rows,
                max_vector_bytes,
                max_native_artifact_bytes,
            },
        )?;
        let store: Arc<dyn IndexStore> = Arc::new(LocalIndexStore::create(
            root.path(),
            "generation-1",
            LIMITS,
        )?);
        assert!(
            builder
                .build(
                    request(Bytes::from(
                        serde_json::to_vec(&config()).map_err(|err| vortex_err!("{err}"))?
                    )),
                    data()?,
                    store
                )
                .await
                .is_err()
        );
        assert!(!root.path().join("generation-1/manifest.json").exists());
        assert_eq!(fs::read_dir(root.path())?.count(), 1);
        Ok(())
    })
}

#[test]
fn truncated_file_and_non_finite_native_vectors_are_rejected() -> VortexResult<()> {
    let root = tempfile::tempdir()?;
    let path = root.path().join("index.bin");
    let bundle = config().bundle(256);
    Native::build(
        &path,
        &(0..256).flat_map(vector).collect::<Vec<_>>(),
        bundle,
        &config(),
    )?;
    let original = fs::read(&path)?;
    fs::write(&path, &original[..original.len() - 1])?;
    assert!(Native::open(&path, bundle).is_err());
    let mut changed = original;
    let offset = 96 + config().m as usize * 8 + 4;
    changed[offset..offset + 4].copy_from_slice(&f32::NAN.to_le_bytes());
    fs::write(&path, changed)?;
    assert!(Native::open(&path, bundle).is_err());
    Ok(())
}
