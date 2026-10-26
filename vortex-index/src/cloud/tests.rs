// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
#[cfg(feature = "snapshot")]
use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::atomic::AtomicU8;
use std::sync::atomic::Ordering;

use bytes::Bytes;
use futures::TryStreamExt;
use futures::stream;
use object_store::CopyOptions;
use object_store::GetResult;
use object_store::ListResult;
use object_store::MultipartUpload;
use object_store::ObjectStoreExt;
use object_store::PutMultipartOptions;
use object_store::PutOptions;
use object_store::PutPayload;
use object_store::PutResult;
use object_store::memory::InMemory;
use parking_lot::Mutex;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::assert_arrays_eq;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::runtime::single::block_on;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::layouts::struct_::writer::StructStrategy;

use super::*;
#[cfg(feature = "snapshot")]
use crate::DistanceMetric;
#[cfg(feature = "snapshot")]
use crate::FlatIndex;
#[cfg(feature = "snapshot")]
use crate::IndexMetadata;
#[cfg(feature = "snapshot")]
use crate::RowFilter;
#[cfg(feature = "snapshot")]
use crate::SearchMode;
#[cfg(feature = "snapshot")]
use crate::VectorSpec;
use crate::file::tests::data;
use crate::file::tests::test_session;
#[cfg(feature = "snapshot")]
use crate::snapshot::SnapshotIndex;
#[cfg(feature = "snapshot")]
use crate::snapshot::SnapshotVectorSearch;
#[cfg(feature = "snapshot")]
use crate::snapshot::VisibleSource;

#[derive(Clone, Debug)]
struct Request {
    range: Option<Range<u64>>,
    head: bool,
    etag: Option<String>,
    version: Option<String>,
}

#[derive(Debug, Default)]
struct TestStore {
    inner: InMemory,
    requests: Mutex<Vec<Request>>,
    fault: AtomicU8,
    historical: Option<Arc<InMemory>>,
}

impl fmt::Display for TestStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TestStore")
    }
}

#[async_trait]
impl ObjectStore for TestStore {
    async fn put_opts(
        &self,
        path: &Path,
        payload: PutPayload,
        options: PutOptions,
    ) -> object_store::Result<PutResult> {
        self.inner.put_opts(path, payload, options).await
    }
    async fn put_multipart_opts(
        &self,
        path: &Path,
        options: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(path, options).await
    }
    async fn get_opts(
        &self,
        path: &Path,
        mut options: GetOptions,
    ) -> object_store::Result<GetResult> {
        let range = match &options.range {
            Some(GetRange::Bounded(range)) => Some(range.clone()),
            _ => None,
        };
        self.requests.lock().push(Request {
            range,
            head: options.head,
            etag: options.if_match.clone(),
            version: options.version.clone(),
        });
        let fault = self.fault.load(Ordering::SeqCst);
        if fault == 1 {
            options.if_match = None;
        }
        let is_read = !options.head;
        let versioned = options.version.as_deref() == Some("v1") && self.historical.is_some();
        let store = if versioned {
            self.historical.as_deref().unwrap_or(&self.inner)
        } else {
            &self.inner
        };
        let mut result = store.get_opts(path, options).await?;
        if versioned {
            result.meta.version = Some("v1".into());
        }
        if is_read {
            match fault {
                2 | 3 => {
                    let length =
                        usize::try_from(result.range.end - result.range.start).map_err(|e| {
                            object_store::Error::Generic {
                                store: "test",
                                source: Box::new(e),
                            }
                        })?;
                    let length = if fault == 2 { length - 1 } else { length + 1 };
                    result.payload = GetResultPayload::Stream(
                        stream::once(async move { Ok(Bytes::from(vec![0; length])) }).boxed(),
                    );
                }
                4 => result.range.start += 1,
                5 => result.meta.size += 1,
                _ => {}
            }
        }
        Ok(result)
    }
    fn delete_stream(
        &self,
        paths: BoxStream<'static, object_store::Result<Path>>,
    ) -> BoxStream<'static, object_store::Result<Path>> {
        self.inner.delete_stream(paths)
    }
    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.inner.list(prefix)
    }
    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> object_store::Result<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }
    async fn copy_opts(
        &self,
        from: &Path,
        to: &Path,
        options: CopyOptions,
    ) -> object_store::Result<()> {
        self.inner.copy_opts(from, to, options).await
    }
}

impl TestStore {
    fn requests(&self) -> Vec<Request> {
        self.requests.lock().clone()
    }
}

pub(crate) async fn put_vortex(
    store: &dyn ObjectStore,
    session: &VortexSession,
    id: u64,
    array: ArrayRef,
) -> VortexResult<SourceFile> {
    let row_count = array.len() as u64;
    let mut bytes = Vec::new();
    session
        .write_options()
        .with_strategy(Arc::new(StructStrategy::new(
            Arc::new(FlatLayoutStrategy::default()),
            Arc::new(FlatLayoutStrategy::default()),
        )))
        .write(&mut bytes, array.to_array_stream())
        .await?;
    let path = Path::parse(format!("data/{id}~test.vortex")).map_err(|e| vortex_err!("{e}"))?;
    store.put(&path, bytes.into()).await?;
    let meta = store.head(&path).await?;
    Ok(SourceFile {
        id,
        uri: format!("memory://test/{path}"),
        version: object_version(&meta)?,
        row_count,
    })
}

fn snapshot(files: Vec<SourceFile>, dtype: &DType) -> VortexResult<Snapshot> {
    Ok(Snapshot {
        dataset_id: "cloud-test".into(),
        version: "snapshot-1".into(),
        schema_fingerprint: schema_fingerprint(dtype)?,
        files,
    })
}

#[test]
fn test_projected_cloud_reads_and_ordered_take() -> VortexResult<()> {
    block_on(|handle| async move {
        let session = test_session(handle)?;
        let store = Arc::new(TestStore::default());
        let payload = "large unrequested column".repeat(4096);
        let array = StructArray::try_new(
            ["id", "payload"].into(),
            vec![
                Buffer::from_iter(0..64_i32).into_array(),
                VarBinArray::from_iter_nonnull(
                    std::iter::repeat_n(payload.as_str(), 64),
                    DType::Utf8(Nullability::NonNullable),
                )
                .into_array(),
            ],
            64,
            Validity::NonNullable,
        )?
        .into_array();
        let dtype = array.dtype().clone();
        let replacement = array.clone();
        let file = put_vortex(store.as_ref(), &session, 3, array).await?;
        let source = ObjectStoreFileSource::open(
            snapshot(vec![file], &dtype)?,
            dtype,
            Arc::clone(&store) as Arc<dyn ObjectStore>,
            "memory://test/",
            session.clone(),
            CloudSourceOptions {
                cache_bytes_per_file: 1024 * 1024,
                ..Default::default()
            },
        )
        .await?;
        let rows = [
            RowAddress {
                file_id: 3,
                row_offset: 62,
            },
            RowAddress {
                file_id: 3,
                row_offset: 1,
            },
            RowAddress {
                file_id: 3,
                row_offset: 62,
            },
        ];
        let fields = ["id".into()];
        let batch = source.take(&rows, &fields).await?;
        assert_eq!(batch.rows, rows);
        let expected = StructArray::try_new(
            ["id"].into(),
            vec![Buffer::from(vec![62_i32, 1, 62]).into_array()],
            3,
            Validity::NonNullable,
        )?;
        assert_arrays_eq!(batch.data, expected, &mut session.create_execution_ctx());
        let reads = store.requests().iter().filter(|r| !r.head).count();
        let batch = source.take(&rows, &fields).await?;
        assert_eq!(batch.rows, rows);
        assert_eq!(store.requests().iter().filter(|r| !r.head).count(), reads);
        let requests = store.requests();
        assert!(
            requests
                .iter()
                .filter(|r| !r.head)
                .all(|r| r.range.is_some() && r.etag.is_some())
        );
        let bytes: u64 = requests
            .iter()
            .filter_map(|r| r.range.as_ref())
            .map(|r| r.end - r.start)
            .sum();
        assert!(
            bytes < 128 * 1024,
            "projected reads downloaded {bytes} bytes"
        );
        drop(requests);
        let empty = source.take(&[], &[]).await?;
        assert_eq!(empty.data.len(), 0);
        let projected = source.scan(&[3], &[])?.try_collect::<Vec<_>>().await?;
        assert_eq!(projected.iter().map(|b| b.rows.len()).sum::<usize>(), 64);
        // The large payload has not been cached. Replacing the object cannot mix
        // a new payload with the already pinned footer and id column.
        put_vortex(store.as_ref(), &session, 3, replacement).await?;
        assert!(source.take(&rows, &["payload".into()]).await.is_err());
        Ok(())
    })
}

#[cfg(feature = "snapshot")]
#[test]
fn test_cloud_snapshot_merges_generations_and_delta_then_reads_payload() -> VortexResult<()> {
    block_on(|handle| async move {
        let session = test_session(handle)?;
        let store = Arc::new(InMemory::new());
        let dtype = data(&[1])?.dtype().clone();
        let mut files = Vec::new();
        for (id, ids) in [(10, [1, 2]), (20, [3, 4])] {
            files.push(put_vortex(store.as_ref(), &session, id, data(&ids)?.into_array()).await?);
        }
        let base = snapshot(files, &dtype)?;
        let mut current = base.clone();
        current.version = "snapshot-2".into();
        current
            .files
            .push(put_vortex(store.as_ref(), &session, 30, data(&[2, 6])?.into_array()).await?);
        let row = |file_id, row_offset| RowAddress {
            file_id,
            row_offset,
        };
        let spec = VectorSpec {
            dimension: NonZeroUsize::MIN.saturating_add(1),
            metric: DistanceMetric::SquaredL2,
        };
        let mut indexes = Vec::new();
        for (id, values) in [(10, [1.0_f32, 0.0, 2.0, 0.0]), (20, [3.0, 0.0, 4.0, 0.0])] {
            let index = FlatIndex::try_new(
                IndexMetadata {
                    format_version: 1,
                    name: "embedding".into(),
                    generation: format!("index-{id}"),
                    backend: FlatIndex::ID.into(),
                    backend_version: 1,
                    snapshot: base.clone(),
                    fields: vec!["embedding".into()],
                    covered_files: vec![id],
                    artifacts: Vec::new(),
                },
                spec,
                vec![row(id, 0), row(id, 1)],
                Buffer::from(values.to_vec()),
            )?;
            indexes.push(SnapshotIndex::try_new(
                Arc::new(index),
                RowFilter::try_new(base.clone(), None, BTreeSet::new())?,
                Bytes::new(),
            )?);
        }
        let source = Arc::new(
            ObjectStoreFileSource::open(
                current.clone(),
                dtype,
                store,
                "memory://test/",
                session.clone(),
                CloudSourceOptions {
                    cache_bytes_per_file: 1024 * 1024,
                    ..Default::default()
                },
            )
            .await?,
        );
        let visibility = RowFilter::try_new(
            current.clone(),
            None,
            BTreeSet::from([row(10, 1), row(20, 1)]),
        )?;
        let search = SnapshotVectorSearch::try_new(
            VisibleSource::try_new(
                Arc::clone(&source) as Arc<dyn IndexSource>,
                visibility.clone(),
            )?,
            "embedding".into(),
            spec,
            indexes,
            vec![30],
            session.clone(),
        )?;
        let oracle = SnapshotVectorSearch::try_new(
            VisibleSource::try_new(source, visibility)?,
            "embedding".into(),
            spec,
            Vec::new(),
            vec![10, 20, 30],
            session.clone(),
        )?;
        let predicate = RowFilter::try_new(current, None, BTreeSet::new())?;
        let k = NonZeroUsize::MIN.saturating_add(2);
        let hits = search
            .search(&[2.0, 0.0], k, SearchMode::Exact, &predicate)
            .await?;
        assert_eq!(
            hits,
            oracle
                .search(&[2.0, 0.0], k, SearchMode::Exact, &predicate)
                .await?
        );
        assert_eq!(
            hits.iter().map(|hit| hit.row).collect::<Vec<_>>(),
            vec![row(30, 0), row(10, 0), row(20, 0)]
        );
        let rows = [hits[2].row, hits[0].row, hits[1].row, hits[0].row];
        let batch = search
            .source()
            .take(&rows, &["id".into(), "note".into()])
            .await?;
        assert_eq!(batch.rows, rows);
        assert_arrays_eq!(
            batch.data,
            data(&[3, 2, 1, 2])?.project(&["id".into(), "note".into()])?,
            &mut session.create_execution_ctx()
        );
        Ok(())
    })
}

#[rstest]
#[case(false)]
#[case(true)]
fn test_replacement_is_rejected_even_if_store_ignores_conditions(
    #[case] ignore_condition: bool,
) -> VortexResult<()> {
    block_on(|handle| async move {
        let store = Arc::new(TestStore::default());
        let path = Path::from("data");
        store
            .put(&path, Bytes::from_static(b"old data").into())
            .await?;
        let file = SourceFile {
            id: 1,
            uri: format!("memory://test/{path}"),
            version: object_version(&store.head(&path).await?)?,
            row_count: 1,
        };
        let reader = PinnedReadAt::new(
            Arc::clone(&store) as Arc<dyn ObjectStore>,
            "memory://test/",
            &file,
            handle,
            CloudSourceOptions::default(),
        )?;
        let first = reader.read_at(0, 3, Alignment::new(1)).await?;
        assert_eq!(first.to_host().await.as_slice(), b"old");
        store
            .put(&path, Bytes::from_static(b"new data").into())
            .await?;
        store
            .fault
            .store(u8::from(ignore_condition), Ordering::SeqCst);
        assert!(reader.read_at(0, 3, Alignment::new(1)).await.is_err());
        assert_eq!(first.to_host().await.as_slice(), b"old");
        Ok(())
    })
}

#[test]
fn test_versioned_reads_survive_latest_object_replacement() -> VortexResult<()> {
    block_on(|handle| async move {
        let historical = Arc::new(InMemory::new());
        let path = Path::from("data");
        historical
            .put(&path, Bytes::from_static(b"old").into())
            .await?;
        let mut meta = historical.head(&path).await?;
        meta.version = Some("v1".into());
        let store = Arc::new(TestStore {
            historical: Some(historical),
            ..Default::default()
        });
        store.put(&path, Bytes::from_static(b"new").into()).await?;
        let file = SourceFile {
            id: 1,
            uri: format!("memory://test/{path}"),
            version: object_version(&meta)?,
            row_count: 1,
        };
        let reader = PinnedReadAt::new(
            Arc::clone(&store) as Arc<dyn ObjectStore>,
            "memory://test/",
            &file,
            handle,
            CloudSourceOptions::default(),
        )?;
        assert_eq!(
            reader
                .read_at(0, 3, Alignment::new(8))
                .await?
                .to_host()
                .await
                .as_slice(),
            b"old"
        );
        assert!(
            store
                .requests()
                .iter()
                .all(|r| r.version.as_deref() == Some("v1"))
        );
        Ok(())
    })
}

#[rstest]
#[case(2)]
#[case(3)]
#[case(4)]
#[case(5)]
fn test_malformed_range_responses_and_budgets(#[case] fault: u8) -> VortexResult<()> {
    block_on(|handle| async move {
        let store = Arc::new(TestStore::default());
        let path = Path::from("data");
        store
            .put(&path, Bytes::from_static(b"abcdefgh").into())
            .await?;
        let file = SourceFile {
            id: 1,
            uri: format!("memory://test/{path}"),
            version: object_version(&store.head(&path).await?)?,
            row_count: 1,
        };
        let reader = PinnedReadAt::new(
            Arc::clone(&store) as Arc<dyn ObjectStore>,
            "memory://test/",
            &file,
            handle,
            CloudSourceOptions {
                max_read_bytes: 4,
                ..Default::default()
            },
        )?;
        store.fault.store(fault, Ordering::SeqCst);
        assert!(reader.read_at(0, 4, Alignment::new(1)).await.is_err());
        assert!(reader.read_at(0, 5, Alignment::new(1)).await.is_err());
        assert!(reader.read_at(7, 2, Alignment::new(1)).await.is_err());
        assert!(
            reader
                .read_at(u64::MAX, 2, Alignment::new(1))
                .await
                .is_err()
        );
        store.fault.store(0, Ordering::SeqCst);
        store.delete(&path).await?;
        assert!(reader.read_at(0, 4, Alignment::new(1)).await.is_err());
        Ok(())
    })
}

#[test]
fn test_cloud_open_checks_catalog_schema_rows_and_identity() -> VortexResult<()> {
    block_on(|handle| async move {
        let session = test_session(handle)?;
        let store = Arc::new(InMemory::new());
        let array = data(&[1, 2])?.into_array();
        let dtype = array.dtype().clone();
        let file = put_vortex(store.as_ref(), &session, 1, array).await?;
        let original = snapshot(vec![file], &dtype)?;
        for case in 0..7 {
            let mut bad = original.clone();
            match case {
                0 => bad.files[0].row_count += 1,
                1 => bad.schema_fingerprint = "different-schema".into(),
                2 => bad.files[0].version = "sha256:wrong-kind".into(),
                3 => {
                    bad.files[0].version =
                        "object-store-v1:{\"etag\":\"*\",\"version\":null,\"size\":1}".into()
                }
                4 => {
                    bad.files[0].version =
                        "object-store-v1:{\"etag\":null,\"version\":null,\"size\":1}".into()
                }
                5 => bad.files[0].uri = "memory://another-store/data/1~test.vortex".into(),
                _ => {
                    bad.files[0].version =
                        "object-store-v1:{\"etag\":null,\"version\":\"ignored-version\",\"size\":1}"
                            .into()
                }
            }
            assert!(
                ObjectStoreFileSource::open(
                    bad,
                    dtype.clone(),
                    Arc::clone(&store) as Arc<dyn ObjectStore>,
                    "memory://test/",
                    session.clone(),
                    CloudSourceOptions::default()
                )
                .await
                .is_err(),
                "case {case}"
            );
        }
        let source = ObjectStoreFileSource::open(
            original,
            dtype,
            store,
            "memory://test/",
            session,
            CloudSourceOptions::default(),
        )
        .await?;
        assert!(source.scan(&[1, 1], &["id".into()]).is_err());
        assert!(source.scan(&[2], &[]).is_err());
        assert!(source.scan(&[1], &["id".into(), "id".into()]).is_err());
        assert!(
            source
                .take(
                    &[RowAddress {
                        file_id: 1,
                        row_offset: 2
                    }],
                    &[]
                )
                .await
                .is_err()
        );
        Ok(())
    })
}
