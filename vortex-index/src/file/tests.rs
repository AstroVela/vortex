// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::fs;
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Arc;
#[cfg(unix)]
use std::sync::mpsc;
#[cfg(unix)]
use std::thread;
#[cfg(unix)]
use std::time::Duration;

use bytes::Bytes;
use futures::TryStreamExt;
#[cfg(unix)]
use rustix::fs as unix_fs;
use tempfile::TempDir;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
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
use vortex_io::runtime::Handle;
use vortex_io::runtime::single::block_on;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

use super::LocalFileSource;
use super::file_version;
use super::schema_fingerprint;
use crate::DistanceMetric;
use crate::FlatIndex;
use crate::IndexMetadata;
use crate::IndexSource;
use crate::RowAddress;
use crate::RowFilter;
use crate::SearchMode;
use crate::Snapshot;
use crate::SourceFile;
use crate::VectorIndex;
use crate::VectorSearchOptions;
use crate::VectorSpec;

const BUDGET: usize = 64 * 1024 * 1024;

fn row(file_id: u64, row_offset: u64) -> RowAddress {
    RowAddress {
        file_id,
        row_offset,
    }
}

fn fields(names: &[&str]) -> Vec<String> {
    names.iter().map(|name| (*name).to_owned()).collect()
}

fn data(ids: &[i32]) -> VortexResult<StructArray> {
    let elements = ids
        .iter()
        .map(|id| {
            i16::try_from(*id)
                .map(|id| [f32::from(id), 0.0])
                .map_err(|err| vortex_err!("{}", err))
        })
        .collect::<VortexResult<Vec<_>>>()?;
    StructArray::try_new(
        ["id", "embedding", "note"].into(),
        vec![
            Buffer::copy_from(ids).into_array(),
            FixedSizeListArray::try_new(
                Buffer::from_iter(elements.into_iter().flatten()).into_array(),
                2,
                Validity::NonNullable,
                ids.len(),
            )?
            .into_array(),
            PrimitiveArray::from_option_iter(ids.iter().map(|id| (id % 2 == 0).then_some(*id)))
                .into_array(),
        ],
        ids.len(),
        Validity::NonNullable,
    )
}

async fn write_file(
    session: &VortexSession,
    dir: &Path,
    id: u64,
    data: ArrayRef,
) -> VortexResult<SourceFile> {
    let row_count = data.len() as u64;
    let mut bytes = Vec::new();
    session
        .write_options()
        .with_strategy(Arc::new(FlatLayoutStrategy::default()))
        .write(&mut bytes, data.to_array_stream())
        .await?;
    let path = dir.join(format!("{id}.vortex"));
    fs::write(&path, &bytes)?;
    Ok(SourceFile {
        id,
        uri: path
            .to_str()
            .ok_or_else(|| vortex_err!("Non-UTF8 test path"))?
            .into(),
        version: file_version(&bytes),
        row_count,
    })
}

fn test_session(handle: Handle) -> VortexResult<VortexSession> {
    let session = array_session()
        .with::<LayoutSession>()
        .with::<RuntimeSession>()
        .with_handle(handle);
    vortex_file::register_default_encodings(&session);
    let edition = EditionId::new("index-test", 2026, 7, 0);
    let editions = session.editions();
    editions
        .declare_edition(Edition {
            id: edition,
            min_vortex_version: None,
        })
        .map_err(|err| vortex_err!("{}", err))?;
    let encodings = session
        .arrays()
        .registry()
        .read(|map| map.keys().copied().collect::<Vec<_>>());
    for encoding in encodings {
        editions
            .declare_inclusion(EditionInclusion::new(&encoding, edition))
            .map_err(|err| vortex_err!("{}", err))?;
    }
    session
        .enable_edition(edition)
        .map_err(|err| vortex_err!("{}", err))?;
    Ok(session)
}

struct Fixture {
    dir: TempDir,
    snapshot: Snapshot,
    dtype: DType,
    session: VortexSession,
}

impl Fixture {
    async fn new(handle: Handle) -> VortexResult<Self> {
        let session = test_session(handle)?;
        let dir = tempfile::tempdir()?;
        let first = data(&[10, 11, 12, 13, 14])?.into_array();
        let dtype = first.dtype().clone();
        let snapshot = Snapshot {
            dataset_id: "file-source-tests".into(),
            version: "all-rows-visible-v1".into(),
            schema_fingerprint: schema_fingerprint(&dtype)?,
            files: vec![
                write_file(&session, dir.path(), 10, first).await?,
                write_file(&session, dir.path(), 30, data(&[30, 31, 32])?.into_array()).await?,
            ],
        };
        Ok(Self {
            dir,
            snapshot,
            dtype,
            session,
        })
    }

    async fn open(&self) -> VortexResult<LocalFileSource> {
        LocalFileSource::open(
            self.snapshot.clone(),
            self.dtype.clone(),
            BUDGET,
            self.session.clone(),
        )
        .await
    }
}

#[cfg(all(unix, feature = "local-store"))]
mod persistence;

fn assert_error<T>(result: VortexResult<T>, message: &str) -> VortexResult<()> {
    let err = result
        .err()
        .ok_or_else(|| vortex_err!("Expected error containing: {}", message))?;
    assert!(err.to_string().contains(message), "{err}");
    Ok(())
}

#[test]
fn test_file_and_schema_identities() -> VortexResult<()> {
    assert_eq!(
        file_version(b"abc"),
        "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    let original = data(&[])?.dtype().clone();
    let reordered = data(&[])?.project(&["note".into(), "embedding".into(), "id".into()])?;
    assert_ne!(
        schema_fingerprint(&original)?,
        schema_fingerprint(reordered.dtype())?
    );
    assert_ne!(
        schema_fingerprint(&original)?,
        schema_fingerprint(&original.as_nullable())?
    );
    assert_eq!(
        schema_fingerprint(&original)?,
        schema_fingerprint(data(&[1, 2])?.dtype())?
    );
    Ok(())
}

#[cfg(unix)]
#[test]
fn test_fifo_open_rejects_without_waiting_for_writer() -> VortexResult<()> {
    block_on(|handle| async move {
        let mut fixture = Fixture::new(handle).await?;
        fixture.snapshot.files.truncate(1);
        let path = fixture.snapshot.files[0].uri.clone();
        fs::remove_file(&path)?;
        unix_fs::mkfifoat(
            unix_fs::CWD,
            path.as_str(),
            unix_fs::Mode::RUSR | unix_fs::Mode::WUSR,
        )
        .map_err(std::io::Error::from)?;

        let (finished, receiver) = mpsc::channel();
        // Release a blocking opener on regression, keeping the test suite live.
        // A correct implementation returns before any writer is opened.
        let guard = thread::spawn(move || -> VortexResult<bool> {
            if receiver.recv_timeout(Duration::from_secs(2)).is_ok() {
                return Ok(false);
            }
            let writer = unix_fs::open(
                path.as_str(),
                unix_fs::OFlags::RDWR | unix_fs::OFlags::NONBLOCK | unix_fs::OFlags::CLOEXEC,
                unix_fs::Mode::empty(),
            )
            .map_err(std::io::Error::from)?;
            let _ = receiver.recv_timeout(Duration::from_secs(5));
            drop(writer);
            Ok(true)
        });
        let result = fixture.open().await;
        let _ = finished.send(());
        let needed_writer = guard
            .join()
            .map_err(|_| vortex_err!("FIFO guard panicked"))??;
        assert_error(result, "not a regular file")?;
        assert!(
            !needed_writer,
            "Opening a FIFO waited for a writer before rejecting it"
        );
        Ok(())
    })
}

#[test]
fn test_external_dtype_file_scan_and_take() -> VortexResult<()> {
    block_on(|handle| async move {
        let mut fixture = Fixture::new(handle).await?;
        let mut bytes = Vec::new();
        fixture
            .session
            .write_options()
            .with_strategy(Arc::new(FlatLayoutStrategy::default()))
            .exclude_dtype()
            .write(
                &mut bytes,
                data(&[30, 31, 32])?.into_array().to_array_stream(),
            )
            .await?;
        let file = &mut fixture.snapshot.files[1];
        fs::write(&file.uri, &bytes)?;
        file.version = file_version(&bytes);

        let source = fixture.open().await?;
        let batches: Vec<_> = source
            .scan(&[10, 30], &fields(&["embedding", "note", "id"]))?
            .try_collect()
            .await?;
        let projection = ["embedding".into(), "note".into(), "id".into()];
        let expected = data(&[10, 11, 12, 13, 14, 30, 31, 32])?.project(&projection)?;
        let actual = ChunkedArray::try_new(
            batches.into_iter().map(|batch| batch.data),
            expected.dtype().clone(),
        )?;
        let mut ctx = fixture.session.create_execution_ctx();
        assert_arrays_eq!(actual, expected, &mut ctx);
        let rows = [row(30, 2), row(10, 0), row(30, 0), row(30, 2)];
        let batch = source
            .take(&rows, &fields(&["embedding", "note", "id"]))
            .await?;
        assert_eq!(batch.rows, rows);
        assert_arrays_eq!(
            batch.data,
            data(&[32, 10, 30, 32])?.project(&projection)?,
            &mut ctx
        );
        Ok(())
    })
}

#[test]
fn test_external_dtype_does_not_override_embedded_schema() -> VortexResult<()> {
    block_on(|handle| async move {
        let mut fixture = Fixture::new(handle).await?;
        let dtypes = fixture
            .dtype
            .as_struct_fields_opt()
            .ok_or_else(|| vortex_err!("Test schema must be a struct"))?
            .fields()
            .collect();
        fixture.dtype = DType::Struct(
            StructFields::new(["renamed_id", "embedding", "note"].into(), dtypes),
            Nullability::NonNullable,
        );
        fixture.snapshot.schema_fingerprint = schema_fingerprint(&fixture.dtype)?;
        assert_error(fixture.open().await, "schema or row count mismatch")
    })
}

#[test]
fn test_projected_scan_preserves_file_order_and_addresses() -> VortexResult<()> {
    block_on(|handle| async move {
        let fixture = Fixture::new(handle).await?;
        let source = fixture.open().await?;
        assert_eq!(source.snapshot(), &fixture.snapshot);
        assert_eq!(source.dtype(), &fixture.dtype);
        let batches: Vec<_> = source
            .scan(&[30, 10], &fields(&["note", "id"]))?
            .try_collect()
            .await?;
        let rows: Vec<_> = batches
            .iter()
            .flat_map(|batch| batch.rows.iter().copied())
            .collect();
        assert_eq!(
            rows,
            (0..3)
                .map(|offset| row(30, offset))
                .chain((0..5).map(|offset| row(10, offset)))
                .collect::<Vec<_>>()
        );
        let expected =
            data(&[30, 31, 32, 10, 11, 12, 13, 14])?.project(&["note".into(), "id".into()])?;
        let actual = ChunkedArray::try_new(
            batches.into_iter().map(|batch| batch.data),
            expected.dtype().clone(),
        )?;
        assert_arrays_eq!(
            actual,
            expected,
            &mut fixture.session.create_execution_ctx()
        );
        Ok(())
    })
}

#[test]
fn test_take_restores_cross_file_order_duplicates_and_nested_values() -> VortexResult<()> {
    block_on(|handle| async move {
        let fixture = Fixture::new(handle).await?;
        let source = fixture.open().await?;
        let rows = vec![
            row(30, 2),
            row(10, 4),
            row(30, 0),
            row(10, 0),
            row(30, 2),
            row(10, 4),
        ];
        let batch = source
            .take(&rows, &fields(&["embedding", "note", "id"]))
            .await?;
        assert_eq!(batch.rows, rows);
        let expected = data(&[32, 14, 30, 10, 32, 14])?.project(&[
            "embedding".into(),
            "note".into(),
            "id".into(),
        ])?;
        assert_arrays_eq!(
            batch.data,
            expected,
            &mut fixture.session.create_execution_ctx()
        );
        Ok(())
    })
}

#[test]
fn test_empty_requests_and_zero_column_projection() -> VortexResult<()> {
    block_on(|handle| async move {
        let fixture = Fixture::new(handle).await?;
        let source = fixture.open().await?;
        assert!(
            source
                .scan(&[], &fields(&["id"]))?
                .try_collect::<Vec<_>>()
                .await?
                .is_empty()
        );
        let empty = source.take(&[], &fields(&["embedding", "note"])).await?;
        assert!(empty.rows.is_empty());
        let expected = data(&[])?.project(&["embedding".into(), "note".into()])?;
        assert_arrays_eq!(
            empty.data,
            expected,
            &mut fixture.session.create_execution_ctx()
        );
        let batches: Vec<_> = source.scan(&[30], &[])?.try_collect().await?;
        assert_eq!(
            batches.iter().map(|batch| batch.rows.len()).sum::<usize>(),
            3
        );
        for batch in batches {
            assert_eq!(batch.data.dtype(), data(&[])?.project(&[])?.dtype());
            assert_eq!(batch.rows.len(), batch.data.len());
        }
        let rows = [row(30, 1), row(10, 3), row(30, 1)];
        let batch = source.take(&rows, &[]).await?;
        assert_eq!(batch.rows, rows);
        assert_arrays_eq!(
            batch.data,
            data(&[31, 13, 31])?.project(&[])?,
            &mut fixture.session.create_execution_ctx()
        );
        Ok(())
    })
}

#[test]
fn test_empty_inventory_and_empty_file() -> VortexResult<()> {
    block_on(|handle| async move {
        let mut fixture = Fixture::new(handle).await?;
        fixture.snapshot.files.clear();
        let source = LocalFileSource::open(
            fixture.snapshot.clone(),
            fixture.dtype.clone(),
            0,
            fixture.session.clone(),
        )
        .await?;
        assert_eq!(source.pinned_bytes(), 0);
        assert!(source.take(&[], &fields(&["id"])).await?.data.is_empty());
        assert!(
            source
                .scan(&[], &[])?
                .try_collect::<Vec<_>>()
                .await?
                .is_empty()
        );
        fixture.snapshot.files.push(
            write_file(
                &fixture.session,
                fixture.dir.path(),
                80,
                data(&[])?.into_array(),
            )
            .await?,
        );
        let source = fixture.open().await?;
        let batches: Vec<_> = source.scan(&[80], &fields(&["id"]))?.try_collect().await?;
        assert_eq!(
            batches.iter().map(|batch| batch.data.len()).sum::<usize>(),
            0
        );
        assert!(source.take(&[row(80, 0)], &fields(&["id"])).await.is_err());
        Ok(())
    })
}

#[test]
fn test_scan_addresses_across_multiple_batches() -> VortexResult<()> {
    block_on(|handle| async move {
        let mut fixture = Fixture::new(handle).await?;
        fixture.snapshot.files = vec![
            write_file(
                &fixture.session,
                fixture.dir.path(),
                90,
                data(&(0..150_000).map(|id| id % 30_000).collect::<Vec<_>>())?.into_array(),
            )
            .await?,
        ];
        let source = fixture.open().await?;
        let batches: Vec<_> = source.scan(&[90], &fields(&["id"]))?.try_collect().await?;
        assert!(batches.len() > 1);
        let rows: Vec<_> = batches
            .iter()
            .flat_map(|batch| batch.rows.iter().copied())
            .collect();
        assert_eq!(
            rows,
            (0..150_000)
                .map(|offset| row(90, offset))
                .collect::<Vec<_>>()
        );
        let requested = [
            row(90, 149_999),
            row(90, 0),
            row(90, 100_000),
            row(90, 99_999),
            row(90, 0),
        ];
        let taken = source.take(&requested, &fields(&["id"])).await?;
        assert_eq!(taken.rows, requested);
        assert_arrays_eq!(
            taken.data,
            data(&[29_999, 0, 10_000, 9_999, 0])?.project(&["id".into()])?,
            &mut fixture.session.create_execution_ctx()
        );
        Ok(())
    })
}

#[test]
fn test_invalid_requests_fail_even_when_empty() -> VortexResult<()> {
    block_on(|handle| async move {
        let fixture = Fixture::new(handle).await?;
        let source = fixture.open().await?;
        assert_error(
            source.scan(&[10, 10], &fields(&["id"])),
            "Duplicate source scan file",
        )?;
        assert_error(source.scan(&[10, 999], &[]), "Unknown source scan file")?;
        for files in [vec![], vec![10]] {
            assert!(source.scan(&files, &fields(&["missing"])).is_err());
            assert_error(
                source.scan(&files, &fields(&["id", "id"])),
                "Duplicate source projection",
            )?;
        }
        assert!(source.take(&[], &fields(&["missing"])).await.is_err());
        assert_error(
            source.take(&[], &fields(&["id", "id"])).await,
            "Duplicate source projection",
        )?;
        for invalid in [row(999, 0), row(10, 5), row(30, u64::MAX)] {
            assert_error(
                source.take(&[row(10, 0), invalid], &[]).await,
                "outside the snapshot",
            )?;
        }
        Ok(())
    })
}

#[test]
fn test_open_rejects_stale_descriptors_and_unsupported_locations() -> VortexResult<()> {
    block_on(|handle| async move {
        let fixture = Fixture::new(handle).await?;
        let mut variants = vec![fixture.snapshot.clone(); 7];
        variants[0].schema_fingerprint = "stale-schema".into();
        variants[1].files[0].row_count += 1;
        variants[2].files[0].version = file_version(b"other bytes");
        variants[3].files[0].uri = "relative.vortex".into();
        variants[4].files[0].uri = "s3://bucket/file.vortex".into();
        variants[5].files[0].version = "etag:abc".into();
        variants[6].files.swap(0, 1);
        for snapshot in variants {
            assert!(
                LocalFileSource::open(
                    snapshot,
                    fixture.dtype.clone(),
                    BUDGET,
                    fixture.session.clone()
                )
                .await
                .is_err()
            );
        }
        let wrong_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        assert_error(
            LocalFileSource::open(fixture.snapshot, wrong_dtype, BUDGET, fixture.session).await,
            "non-nullable struct",
        )?;
        Ok(())
    })
}

#[test]
fn test_open_checks_each_file_schema_and_complete_inventory_budget() -> VortexResult<()> {
    block_on(|handle| async move {
        let mut fixture = Fixture::new(handle).await?;
        let total = fixture
            .snapshot
            .files
            .iter()
            .map(|file| fs::metadata(&file.uri).map(|meta| meta.len()))
            .collect::<std::io::Result<Vec<_>>>()?
            .into_iter()
            .sum::<u64>();
        let total = usize::try_from(total)?;
        LocalFileSource::open(
            fixture.snapshot.clone(),
            fixture.dtype.clone(),
            total,
            fixture.session.clone(),
        )
        .await?;
        assert_error(
            LocalFileSource::open(
                fixture.snapshot.clone(),
                fixture.dtype.clone(),
                total - 1,
                fixture.session.clone(),
            )
            .await,
            "pinned byte budget",
        )?;
        fixture.snapshot.files[1] = write_file(
            &fixture.session,
            fixture.dir.path(),
            30,
            Buffer::from(vec![1_i32, 2, 3]).into_array(),
        )
        .await?;
        assert_error(fixture.open().await, "schema or row count mismatch")?;
        Ok(())
    })
}

#[test]
fn test_missing_truncated_and_corrupt_files_are_errors() -> VortexResult<()> {
    block_on(|handle| async move {
        let mut fixture = Fixture::new(handle).await?;
        let path = fixture.snapshot.files[1].uri.clone();
        fs::remove_file(&path)?;
        assert_error(fixture.open().await, "Cannot open source file 30")?;
        let bytes = b"not a vortex file";
        fs::write(&path, bytes)?;
        assert_error(fixture.open().await, "content version mismatch")?;
        fixture.snapshot.files[1].version = file_version(bytes);
        assert!(fixture.open().await.is_err());
        fs::remove_file(&path)?;
        fs::create_dir(&path)?;
        assert_error(fixture.open().await, "not a regular file")?;
        Ok(())
    })
}

#[test]
fn test_pinned_bytes_survive_replacement_in_place_writes_and_deletion() -> VortexResult<()> {
    block_on(|handle| async move {
        let fixture = Fixture::new(handle).await?;
        let source = fixture.open().await?;
        let first = &fixture.snapshot.files[0].uri;
        let second = &fixture.snapshot.files[1].uri;
        let pinned_bytes =
            usize::try_from(fs::metadata(first)?.len() + fs::metadata(second)?.len())?;
        assert_eq!(source.pinned_bytes(), pinned_bytes);
        let replacement = fixture.dir.path().join("replacement.vortex");
        fs::write(&replacement, fs::read(second)?)?;
        fs::rename(&replacement, first)?;
        fs::write(second, b"changed in place")?;
        assert!(fixture.open().await.is_err());
        fs::remove_file(first)?;
        fs::remove_file(second)?;
        assert_eq!(source.pinned_bytes(), pinned_bytes);
        let batch = source
            .take(&[row(30, 2), row(10, 0)], &fields(&["id"]))
            .await?;
        assert_arrays_eq!(
            batch.data,
            data(&[32, 10])?.project(&["id".into()])?,
            &mut fixture.session.create_execution_ctx()
        );
        let batches: Vec<_> = source
            .scan(&[10, 30], &fields(&["id"]))?
            .try_collect()
            .await?;
        assert_eq!(
            batches.iter().map(|batch| batch.rows.len()).sum::<usize>(),
            8
        );
        Ok(())
    })
}

#[test]
fn test_file_scan_flat_search_and_ranked_take() -> VortexResult<()> {
    block_on(|handle| async move {
        let fixture = Fixture::new(handle).await?;
        let source = fixture.open().await?;
        let mut rows = Vec::new();
        let mut vectors = Vec::new();
        let mut ctx = fixture.session.create_execution_ctx();
        let mut stream = source.scan(&[10, 30], &fields(&["embedding"]))?;
        while let Some(batch) = stream.try_next().await? {
            rows.extend(batch.rows);
            let data = batch.data.execute::<StructArray>(&mut ctx)?;
            let lists = data
                .unmasked_field(0)
                .clone()
                .execute::<FixedSizeListArray>(&mut ctx)?;
            let values = lists
                .elements()
                .clone()
                .execute::<PrimitiveArray>(&mut ctx)?;
            vectors.extend_from_slice(values.as_slice::<f32>());
        }
        let metadata = IndexMetadata {
            format_version: 1,
            name: "embedding-index".into(),
            generation: "test-generation".into(),
            backend: FlatIndex::ID.into(),
            backend_version: 1,
            snapshot: source.snapshot().clone(),
            fields: fields(&["embedding"]),
            covered_files: vec![10, 30],
            artifacts: vec![],
        };
        let spec = VectorSpec {
            dimension: NonZeroUsize::new(2).ok_or_else(|| vortex_err!("test dimension"))?,
            metric: DistanceMetric::SquaredL2,
        };
        let index = FlatIndex::try_new(metadata, spec, rows, Buffer::from(vectors))?;
        let options = VectorSearchOptions {
            k: NonZeroUsize::new(5).ok_or_else(|| vortex_err!("test k"))?,
            mode: SearchMode::Exact,
            backend_options: Bytes::new(),
        };
        let filter = RowFilter::try_new(source.snapshot().clone(), None, BTreeSet::new())?;
        let hits = index.search(&[25.0, 0.0], &options, &filter).await?;
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
            &mut ctx
        );
        Ok(())
    })
}
