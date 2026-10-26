// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering as AtomicOrdering;

use parking_lot::Mutex;
use rstest::rstest;
use tempfile::TempDir;
use vortex_array::assert_arrays_eq;
use vortex_io::runtime::Handle;
use vortex_io::runtime::single::block_on;

use super::*;
use crate::FlatIndex;
use crate::IndexMetadata;
use crate::SourceFile;
use crate::VectorIndex;
use crate::file::LocalFileSource;
use crate::file::schema_fingerprint;
use crate::file::tests::data;
use crate::file::tests::test_session;
use crate::file::tests::write_file;

fn row(file_id: u64, row_offset: u64) -> RowAddress {
    RowAddress {
        file_id,
        row_offset,
    }
}
fn all(snapshot: &Snapshot) -> VortexResult<RowFilter> {
    RowFilter::try_new(snapshot.clone(), None, BTreeSet::new())
}
fn spec() -> VectorSpec {
    VectorSpec {
        dimension: NonZeroUsize::MIN.saturating_add(1),
        metric: DistanceMetric::SquaredL2,
    }
}

struct TracedSource {
    inner: LocalFileSource,
    scans: Mutex<Vec<(Vec<u64>, Vec<String>)>>,
}

#[async_trait]
impl IndexSource for TracedSource {
    fn snapshot(&self) -> &Snapshot {
        self.inner.snapshot()
    }
    fn scan(
        &self,
        files: &[u64],
        fields: &[String],
    ) -> VortexResult<BoxStream<'static, VortexResult<SourceBatch>>> {
        self.scans.lock().push((files.to_vec(), fields.to_vec()));
        self.inner.scan(files, fields)
    }
    async fn take(&self, rows: &[RowAddress], fields: &[String]) -> VortexResult<SourceBatch> {
        self.inner.take(rows, fields).await
    }
}

#[derive(Debug)]
struct TracedIndex {
    flat: FlatIndex,
    calls: AtomicUsize,
    exact: bool,
    filtering: bool,
    fail: bool,
}

impl Index for TracedIndex {
    fn metadata(&self) -> &IndexMetadata {
        self.flat.metadata()
    }
    fn as_vector(&self) -> Option<&dyn VectorIndex> {
        Some(self)
    }
}
#[async_trait]
impl VectorIndex for TracedIndex {
    fn spec(&self) -> VectorSpec {
        self.flat.spec()
    }
    fn supports_exact(&self) -> bool {
        self.exact
    }
    fn supports_filter(&self) -> bool {
        self.filtering
    }
    async fn search(
        &self,
        query: &[f32],
        options: &VectorSearchOptions,
        filter: &RowFilter,
    ) -> VortexResult<Vec<SearchHit>> {
        self.calls.fetch_add(1, AtomicOrdering::SeqCst);
        if self.fail {
            vortex_bail!("injected provider failure");
        }
        assert_eq!(filter.snapshot(), &self.metadata().snapshot);
        assert!(self.filtering || !filter.is_restricted());
        assert!(self.exact || options.mode == SearchMode::Approximate);
        self.flat.search(query, options, filter).await
    }
}

struct Fixture {
    dir: TempDir,
    session: VortexSession,
    dtype: DType,
    base: Snapshot,
    current: Snapshot,
}

impl Fixture {
    async fn new(handle: Handle) -> VortexResult<Self> {
        let session = test_session(handle)?;
        let dir = tempfile::tempdir()?;
        let array = data(&[1, 2, 3])?.into_array();
        let dtype = array.dtype().clone();
        let file = write_file(&session, dir.path(), 10, array).await?;
        let base = Snapshot {
            dataset_id: "snapshot-search".into(),
            version: "base".into(),
            schema_fingerprint: schema_fingerprint(&dtype)?,
            files: vec![file],
        };
        let mut current = base.clone();
        current.version = "appended-and-deleted".into();
        current
            .files
            .push(write_file(&session, dir.path(), 30, data(&[2, 4])?.into_array()).await?);
        Ok(Self {
            dir,
            session,
            dtype,
            base,
            current,
        })
    }
    fn index(&self) -> VortexResult<TracedIndex> {
        let metadata = IndexMetadata {
            format_version: 1,
            name: "embedding".into(),
            generation: "base-index".into(),
            backend: FlatIndex::ID.into(),
            backend_version: 1,
            snapshot: self.base.clone(),
            fields: vec!["embedding".into()],
            covered_files: vec![10],
            artifacts: Vec::new(),
        };
        Ok(TracedIndex {
            flat: FlatIndex::try_new(
                metadata,
                spec(),
                vec![row(10, 0), row(10, 1), row(10, 2)],
                Buffer::from(vec![1.0_f32, 0.0, 2.0, 0.0, 3.0, 0.0]),
            )?,
            calls: AtomicUsize::new(0),
            exact: true,
            filtering: true,
            fail: false,
        })
    }
    async fn source(&self) -> VortexResult<Arc<TracedSource>> {
        Ok(Arc::new(TracedSource {
            inner: LocalFileSource::open(
                self.current.clone(),
                self.dtype.clone(),
                16 * 1024 * 1024,
                self.session.clone(),
            )
            .await?,
            scans: Mutex::new(Vec::new()),
        }))
    }
    async fn searcher(
        &self,
        visibility: RowFilter,
        index: Option<SnapshotIndex>,
    ) -> VortexResult<(SnapshotVectorSearch, Arc<TracedSource>)> {
        let source = self.source().await?;
        let delta_files = if index.is_some() {
            vec![30]
        } else {
            self.current.files.iter().map(|file| file.id).collect()
        };
        let search = SnapshotVectorSearch::try_new(
            VisibleSource::try_new(Arc::clone(&source) as Arc<dyn IndexSource>, visibility)?,
            "embedding".into(),
            spec(),
            index.into_iter().collect(),
            delta_files,
            self.session.clone(),
        )?;
        Ok((search, source))
    }
}

#[test]
fn test_reuses_base_scans_delta_and_hides_superseded_and_deleted_rows() -> VortexResult<()> {
    block_on(|handle| async move {
        let fixture = Fixture::new(handle).await?;
        let index = Arc::new(fixture.index()?);
        let entry = SnapshotIndex::try_new(
            Arc::clone(&index) as Arc<dyn Index>,
            all(&fixture.base)?,
            Bytes::new(),
        )?;
        let visibility = RowFilter::try_new(
            fixture.current.clone(),
            None,
            BTreeSet::from([row(10, 0), row(10, 1)]),
        )?;
        let (search, source) = fixture.searcher(visibility, Some(entry)).await?;
        let hits = search
            .search(
                &[2.0, 0.0],
                NonZeroUsize::MIN.saturating_add(2),
                SearchMode::Approximate,
                &all(&fixture.current)?,
            )
            .await?;
        assert_eq!(
            hits,
            vec![
                SearchHit {
                    row: row(30, 0),
                    distance: 0.0
                },
                SearchHit {
                    row: row(10, 2),
                    distance: 1.0
                },
                SearchHit {
                    row: row(30, 1),
                    distance: 4.0
                }
            ]
        );
        assert_eq!(index.calls.load(AtomicOrdering::SeqCst), 1);
        assert_eq!(
            *source.scans.lock(),
            vec![(vec![30], vec!["embedding".to_owned()])]
        );
        let rows: Vec<_> = hits.iter().map(|hit| hit.row).collect();
        let batch = search
            .source()
            .take(&rows, &["id".into(), "note".into()])
            .await?;
        assert_arrays_eq!(
            batch.data,
            data(&[2, 3, 4])?.project(&["id".into(), "note".into()])?,
            &mut fixture.session.create_execution_ctx()
        );
        assert!(
            search
                .source()
                .take(&[row(10, 1)], &["id".into()])
                .await
                .is_err()
        );
        let base_filter = all(&fixture.base)?;
        assert!(
            search
                .search(
                    &[2.0, 0.0],
                    NonZeroUsize::MIN.saturating_add(2),
                    SearchMode::Exact,
                    &base_filter
                )
                .await
                .is_err()
        );
        Ok(())
    })
}

#[rstest]
#[case(false, true, SearchMode::Exact)]
#[case(true, false, SearchMode::Approximate)]
fn test_unsupported_capabilities_fail_without_scanning(
    #[case] exact: bool,
    #[case] filtering: bool,
    #[case] mode: SearchMode,
) -> VortexResult<()> {
    block_on(|handle| async move {
        let fixture = Fixture::new(handle).await?;
        let mut backend = fixture.index()?;
        backend.exact = exact;
        backend.filtering = filtering;
        let backend = Arc::new(backend);
        let entry = SnapshotIndex::try_new(
            Arc::clone(&backend) as Arc<dyn Index>,
            all(&fixture.base)?,
            Bytes::new(),
        )?;
        let visibility = RowFilter::try_new(
            fixture.current.clone(),
            None,
            BTreeSet::from([row(10, 0), row(10, 1), row(30, 0)]),
        )?;
        let (search, source) = fixture.searcher(visibility, Some(entry)).await?;
        let result = search
            .search(
                &[1.0, 0.0],
                NonZeroUsize::MIN.saturating_add(1),
                mode,
                &all(&fixture.current)?,
            )
            .await;
        assert!(result.is_err());
        assert_eq!(backend.calls.load(AtomicOrdering::SeqCst), 0);
        assert!(source.scans.lock().is_empty());
        Ok(())
    })
}

#[rstest]
#[case(0)]
#[case(1)]
#[case(2)]
fn test_changed_removed_or_newly_visible_coverage_is_rejected(
    #[case] change: u8,
) -> VortexResult<()> {
    block_on(|handle| async move {
        let mut fixture = Fixture::new(handle).await?;
        let backend = Arc::new(fixture.index()?);
        let build_visibility = if change == 2 {
            RowFilter::try_new(fixture.base.clone(), None, BTreeSet::from([row(10, 1)]))?
        } else {
            all(&fixture.base)?
        };
        let entry = SnapshotIndex::try_new(
            Arc::clone(&backend) as Arc<dyn Index>,
            build_visibility,
            Bytes::new(),
        )?;
        match change {
            0 => {
                fixture.current.files[0] = write_file(
                    &fixture.session,
                    fixture.dir.path(),
                    10,
                    data(&[7, 8, 9])?.into_array(),
                )
                .await?
            }
            1 => {
                fixture.current.files.remove(0);
            }
            _ => {}
        }
        assert!(
            fixture
                .searcher(all(&fixture.current)?, Some(entry))
                .await
                .is_err()
        );
        assert_eq!(backend.calls.load(AtomicOrdering::SeqCst), 0);
        Ok(())
    })
}

#[test]
fn test_query_predicate_applies_to_both_branches_and_provider_errors_propagate() -> VortexResult<()>
{
    block_on(|handle| async move {
        let fixture = Fixture::new(handle).await?;
        let entry = SnapshotIndex::try_new(
            Arc::new(fixture.index()?),
            all(&fixture.base)?,
            Bytes::new(),
        )?;
        let (search, _) = fixture
            .searcher(all(&fixture.current)?, Some(entry))
            .await?;
        let predicate = RowFilter::try_new(
            fixture.current.clone(),
            Some(BTreeSet::from([row(10, 2), row(30, 1)])),
            BTreeSet::new(),
        )?;
        let k = NonZeroUsize::MIN.saturating_add(1);
        assert_eq!(
            search
                .search(&[2.0, 0.0], k, SearchMode::Approximate, &predicate)
                .await?
                .iter()
                .map(|hit| hit.row)
                .collect::<Vec<_>>(),
            vec![row(10, 2), row(30, 1)]
        );
        let empty = RowFilter::try_new(
            fixture.current.clone(),
            Some(BTreeSet::new()),
            BTreeSet::new(),
        )?;
        assert!(
            search
                .search(&[2.0, 0.0], k, SearchMode::Approximate, &empty)
                .await?
                .is_empty()
        );
        assert!(
            search
                .search(&[f32::NAN, 0.0], k, SearchMode::Approximate, &predicate)
                .await
                .is_err()
        );
        assert!(
            search
                .search(&[0.0], k, SearchMode::Approximate, &predicate)
                .await
                .is_err()
        );
        let mut backend = fixture.index()?;
        backend.fail = true;
        let entry = SnapshotIndex::try_new(Arc::new(backend), all(&fixture.base)?, Bytes::new())?;
        let (search, _) = fixture
            .searcher(all(&fixture.current)?, Some(entry))
            .await?;
        let error = search
            .search(&[2.0, 0.0], k, SearchMode::Approximate, &predicate)
            .await
            .err()
            .ok_or_else(|| vortex_err!("Expected provider failure"))?;
        assert!(error.to_string().contains("injected provider failure"));
        Ok(())
    })
}

#[test]
fn test_visibility_preserves_physical_addresses_and_take_duplicates() -> VortexResult<()> {
    block_on(|handle| async move {
        let fixture = Fixture::new(handle).await?;
        let mask = RowFilter::try_new(
            fixture.current.clone(),
            Some(BTreeSet::from([row(10, 2), row(30, 1)])),
            BTreeSet::new(),
        )?;
        let (search, _) = fixture.searcher(mask, None).await?;
        let batches = search
            .source()
            .scan(&[30, 10], &["id".into()])?
            .try_collect::<Vec<_>>()
            .await?;
        assert_eq!(
            batches
                .iter()
                .flat_map(|b| b.rows.clone())
                .collect::<Vec<_>>(),
            vec![row(30, 1), row(10, 2)]
        );
        let rows = [row(30, 1), row(10, 2), row(30, 1)];
        let batch = search.source().take(&rows, &["id".into()]).await?;
        assert_eq!(batch.rows, rows);
        assert_arrays_eq!(
            batch.data,
            data(&[4, 3, 4])?.project(&["id".into()])?,
            &mut fixture.session.create_execution_ctx()
        );
        Ok(())
    })
}

#[test]
fn test_rejects_ambiguous_coverage_and_schema() -> VortexResult<()> {
    block_on(|handle| async move {
        let fixture = Fixture::new(handle).await?;
        let source = fixture.source().await?;
        let entries = (0..2)
            .map(|_| {
                SnapshotIndex::try_new(
                    Arc::new(fixture.index()?),
                    all(&fixture.base)?,
                    Bytes::new(),
                )
            })
            .collect::<VortexResult<Vec<_>>>()?;
        assert!(
            SnapshotVectorSearch::try_new(
                VisibleSource::try_new(
                    Arc::clone(&source) as Arc<dyn IndexSource>,
                    all(&fixture.current)?
                )?,
                "embedding".into(),
                spec(),
                entries,
                vec![30],
                fixture.session.clone()
            )
            .is_err()
        );
        let entry = SnapshotIndex::try_new(
            Arc::new(fixture.index()?),
            all(&fixture.base)?,
            Bytes::new(),
        )?;
        assert!(
            SnapshotVectorSearch::try_new(
                VisibleSource::try_new(
                    Arc::clone(&source) as Arc<dyn IndexSource>,
                    all(&fixture.current)?
                )?,
                "different-field".into(),
                spec(),
                vec![entry],
                vec![30],
                fixture.session.clone()
            )
            .is_err()
        );
        assert!(VisibleSource::try_new(source, all(&fixture.base)?).is_err());
        assert!(
            SnapshotIndex::try_new(
                Arc::new(fixture.index()?),
                all(&fixture.current)?,
                Bytes::new()
            )
            .is_err()
        );
        Ok(())
    })
}

#[rstest]
#[case(vec![])]
#[case(vec![30, 30])]
#[case(vec![10, 30])]
#[case(vec![99])]
fn test_delta_inventory_must_complete_base_without_overlap(
    #[case] delta_files: Vec<u64>,
) -> VortexResult<()> {
    block_on(|handle| async move {
        let fixture = Fixture::new(handle).await?;
        let source = fixture.source().await?;
        let entry = SnapshotIndex::try_new(
            Arc::new(fixture.index()?),
            all(&fixture.base)?,
            Bytes::new(),
        )?;
        assert!(
            SnapshotVectorSearch::try_new(
                VisibleSource::try_new(
                    Arc::clone(&source) as Arc<dyn IndexSource>,
                    all(&fixture.current)?
                )?,
                "embedding".into(),
                spec(),
                vec![entry],
                delta_files,
                fixture.session.clone(),
            )
            .is_err()
        );
        assert!(source.scans.lock().is_empty());
        Ok(())
    })
}

#[test]
fn test_visibility_subset_proof_against_exhaustive_rows() -> VortexResult<()> {
    let snapshot = Snapshot {
        dataset_id: "subset".into(),
        version: "v1".into(),
        schema_fingerprint: "schema".into(),
        files: vec![
            SourceFile {
                id: 1,
                uri: "a".into(),
                version: "v1".into(),
                row_count: 3,
            },
            SourceFile {
                id: 2,
                uri: "b".into(),
                version: "v1".into(),
                row_count: 1,
            },
        ],
    };
    let rows = [row(1, 0), row(1, 1), row(1, 2), row(2, 0)];
    let mut masks = Vec::new();
    for allow_bits in 0..=16 {
        for exclude_bits in 0..16 {
            let select = |bits: usize| {
                rows.iter()
                    .enumerate()
                    .filter(|(i, _)| bits & (1 << i) != 0)
                    .map(|(_, row)| *row)
                    .collect()
            };
            masks.push(RowFilter::try_new(
                snapshot.clone(),
                (allow_bits != 16).then(|| select(allow_bits)),
                select(exclude_bits),
            )?);
        }
    }
    for old in &masks {
        for current in &masks {
            let expected = rows[..3]
                .iter()
                .all(|row| !current.includes(*row) || old.includes(*row));
            assert_eq!(current.covered_subset_of(old, &[1]), expected);
        }
    }
    Ok(())
}

#[test]
fn test_local_snapshot_merges_generations_and_delta_then_reads_payload() -> VortexResult<()> {
    block_on(|handle| async move {
        let session = test_session(handle)?;
        let dir = tempfile::tempdir()?;
        let dtype = data(&[1])?.dtype().clone();
        let mut files = Vec::new();
        for (id, ids) in [(10, [1, 2]), (20, [3, 4])] {
            files.push(write_file(&session, dir.path(), id, data(&ids)?.into_array()).await?);
        }
        let base = Snapshot {
            dataset_id: "local-multi-base".into(),
            version: "base".into(),
            schema_fingerprint: schema_fingerprint(&dtype)?,
            files,
        };
        let mut current = base.clone();
        current.version = "snapshot-2".into();
        current
            .files
            .push(write_file(&session, dir.path(), 30, data(&[2, 6])?.into_array()).await?);
        let spec = spec();
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
            LocalFileSource::open(current.clone(), dtype, 16 * 1024 * 1024, session.clone())
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
