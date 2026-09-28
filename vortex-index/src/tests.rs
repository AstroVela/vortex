// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::num::NonZeroUsize;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::executor::block_on;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_scan::selection::Selection;

use crate::CandidateGuarantee;
use crate::DistanceMetric;
use crate::FlatIndex;
use crate::Index;
use crate::IndexArtifact;
use crate::IndexMetadata;
use crate::IndexProvider;
use crate::IndexRegistry;
use crate::IndexStore;
use crate::RowAddress;
use crate::RowFilter;
use crate::ScalarCandidates;
use crate::SearchMode;
use crate::Snapshot;
use crate::SourceBatch;
use crate::SourceFile;
use crate::VectorIndex;
use crate::VectorSearchOptions;
use crate::VectorSpec;

fn snapshot() -> Snapshot {
    Snapshot {
        dataset_id: "dataset-a".into(),
        version: "snapshot-1".into(),
        schema_fingerprint: "embedding-f32x2-v1".into(),
        files: [10, 30]
            .map(|id| SourceFile {
                id,
                uri: format!("data/{id}.vortex"),
                version: format!("immutable-object-{id}"),
                row_count: 2,
            })
            .to_vec(),
    }
}

fn metadata() -> IndexMetadata {
    IndexMetadata {
        format_version: 1,
        name: "embedding_idx".into(),
        generation: "generation-1".into(),
        backend: FlatIndex::ID.into(),
        backend_version: 1,
        snapshot: snapshot(),
        fields: vec!["embedding".into()],
        covered_files: vec![10, 30],
        artifacts: Vec::new(),
    }
}

fn row(file_id: u64, row_offset: u64) -> RowAddress {
    RowAddress {
        file_id,
        row_offset,
    }
}

fn rows() -> Vec<RowAddress> {
    vec![row(10, 0), row(10, 1), row(30, 0), row(30, 1)]
}

fn values() -> Buffer<f32> {
    Buffer::copy_from([0., 0., 1., 0., 0., 2., 1., 1.])
}

fn spec() -> VortexResult<VectorSpec> {
    Ok(VectorSpec {
        dimension: NonZeroUsize::new(2).ok_or_else(|| vortex_err!("Invalid test dimension"))?,
        metric: DistanceMetric::SquaredL2,
    })
}

fn options(k: usize) -> VortexResult<VectorSearchOptions> {
    Ok(VectorSearchOptions {
        k: NonZeroUsize::new(k).ok_or_else(|| vortex_err!("Invalid test k"))?,
        mode: SearchMode::Exact,
        backend_options: Bytes::new(),
    })
}

fn all_rows() -> VortexResult<RowFilter> {
    RowFilter::try_new(snapshot(), None, BTreeSet::new())
}

fn index() -> VortexResult<FlatIndex> {
    FlatIndex::try_new(metadata(), spec()?, rows(), values())
}

#[test]
fn test_metadata_round_trip() -> VortexResult<()> {
    let original = metadata();
    let json = serde_json::to_vec(&original).map_err(|err| vortex_err!("{}", err))?;
    let decoded: IndexMetadata =
        serde_json::from_slice(&json).map_err(|err| vortex_err!("{}", err))?;
    decoded.validate_for(&snapshot())?;
    assert_eq!(original, decoded);
    Ok(())
}

#[test]
fn test_snapshot_rejects_ambiguous_files() {
    let mut changed = snapshot();
    changed.files.reverse();
    assert!(changed.validate().is_err());
    changed = snapshot();
    changed.files[1].id = 10;
    assert!(changed.validate().is_err());
    changed = snapshot();
    changed.files[0].version.clear();
    assert!(changed.validate().is_err());
}

#[test]
fn test_snapshot_identity_covers_data_schema_and_visibility() {
    let mut variants = vec![snapshot(); 6];
    variants[0].dataset_id = "different-dataset".into();
    variants[1].version = "new-deletion-mask".into();
    variants[2].schema_fingerprint = "new-schema".into();
    variants[3].files[0].version = "replacement-at-same-path".into();
    variants[4].files[0].row_count += 1;
    variants[5].files[0].uri = "other.vortex".into();
    for changed in variants {
        assert!(metadata().validate_for(&changed).is_err());
    }
}

#[test]
fn test_invalid_metadata() {
    let mut variants = vec![metadata(); 8];
    variants[0].format_version = 2;
    variants[1].backend_version = 0;
    variants[2].fields.clear();
    variants[3].fields.push("embedding".into());
    variants[4].covered_files = vec![10, 10];
    variants[5].covered_files = vec![10, 99];
    variants[6].generation = "bad\0generation".into();
    variants[7].name.clear();
    for changed in variants {
        assert!(changed.validate_for(&snapshot()).is_err());
    }
}

#[test]
fn test_artifact_confinement_and_duplicates() -> VortexResult<()> {
    let mut meta = metadata();
    meta.artifacts = vec![IndexArtifact {
        path: "native/graph.bin".into(),
        size: 32,
        checksum: "sha256:example".into(),
    }];
    meta.validate_for(&snapshot())?;
    for path in [
        "",
        "/absolute",
        "../graph",
        "native/../graph",
        "native//graph",
        "./graph",
        "C:\\graph",
        "s3://bucket/file",
        "a\0b",
    ] {
        let mut invalid = meta.clone();
        invalid.artifacts[0].path = path.into();
        assert!(
            invalid.validate_for(&snapshot()).is_err(),
            "accepted {path:?}"
        );
    }
    meta.artifacts.push(meta.artifacts[0].clone());
    assert!(meta.validate_for(&snapshot()).is_err());
    Ok(())
}

#[test]
fn test_partial_coverage_retains_scan_work() -> VortexResult<()> {
    let mut meta = metadata();
    meta.covered_files = vec![10];
    meta.validate_for(&snapshot())?;
    assert_eq!(
        meta.uncovered_files()
            .map(|file| file.id)
            .collect::<Vec<_>>(),
        vec![30]
    );
    let flat = FlatIndex::try_new(
        meta,
        spec()?,
        rows()[..2].to_vec(),
        Buffer::copy_from([0., 0., 1., 0.]),
    )?;
    let hits = block_on(flat.search(&[0., 0.], &options(10)?, &all_rows()?))?;
    assert_eq!(hits.len(), 2);
    assert!(hits.iter().all(|hit| hit.row.file_id == 10));
    Ok(())
}

#[test]
fn test_flat_validates_dense_coverage_and_vectors() -> VortexResult<()> {
    let mut duplicate = rows();
    duplicate[1] = duplicate[0];
    assert!(FlatIndex::try_new(metadata(), spec()?, duplicate, values()).is_err());
    assert!(
        FlatIndex::try_new(
            metadata(),
            spec()?,
            rows()[..3].to_vec(),
            Buffer::copy_from([0.; 6])
        )
        .is_err()
    );
    assert!(FlatIndex::try_new(metadata(), spec()?, rows(), Buffer::copy_from([0.; 7])).is_err());
    assert!(
        FlatIndex::try_new(
            metadata(),
            spec()?,
            rows(),
            Buffer::copy_from([f32::NAN; 8])
        )
        .is_err()
    );
    let mut unsupported = spec()?;
    unsupported.metric = DistanceMetric::Cosine;
    assert!(FlatIndex::try_new(metadata(), unsupported, rows(), values()).is_err());
    let mut invalid = rows();
    invalid[0] = row(10, 2);
    assert!(FlatIndex::try_new(metadata(), spec()?, invalid, values()).is_err());
    Ok(())
}

#[test]
fn test_exact_search_and_deterministic_ties() -> VortexResult<()> {
    let flat = index()?;
    let hits = block_on(flat.search(&[1., 0.], &options(4)?, &all_rows()?))?;
    assert_eq!(
        hits.iter().map(|hit| hit.row).collect::<Vec<_>>(),
        vec![row(10, 1), row(10, 0), row(30, 1), row(30, 0)]
    );
    assert_eq!(
        hits.iter().map(|hit| hit.distance).collect::<Vec<_>>(),
        vec![0., 1., 1., 5.]
    );
    assert!(flat.supports_exact() && flat.supports_filter());
    assert!(flat.as_scalar().is_none());
    Ok(())
}

#[test]
fn test_filter_runs_before_top_k() -> VortexResult<()> {
    let filter = RowFilter::try_new(snapshot(), None, BTreeSet::from([row(10, 0)]))?;
    let hits = block_on(index()?.search(&[0., 0.], &options(2)?, &filter))?;
    assert_eq!(
        hits.iter().map(|hit| hit.row).collect::<Vec<_>>(),
        vec![row(10, 1), row(30, 1)]
    );
    Ok(())
}

#[test]
fn test_allow_list_empty_and_exclusion_precedence() -> VortexResult<()> {
    let filter = RowFilter::try_new(
        snapshot(),
        Some(BTreeSet::from([row(10, 0), row(30, 1)])),
        BTreeSet::from([row(10, 0)]),
    )?;
    let flat = index()?;
    let hits = block_on(flat.search(&[0., 0.], &options(2)?, &filter))?;
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].row, row(30, 1));
    let empty = RowFilter::try_new(snapshot(), Some(BTreeSet::new()), BTreeSet::new())?;
    assert!(block_on(flat.search(&[0., 0.], &options(2)?, &empty))?.is_empty());
    assert!(empty.is_restricted());
    assert!(!all_rows()?.is_restricted());
    Ok(())
}

#[test]
fn test_query_and_filter_validation() -> VortexResult<()> {
    let flat = index()?;
    for query in [&[0.][..], &[0., f32::NAN], &[f32::INFINITY, 0.]] {
        assert!(block_on(flat.search(query, &options(2)?, &all_rows()?)).is_err());
    }
    let mut stale = snapshot();
    stale.version = "snapshot-2".into();
    let filter = RowFilter::try_new(stale, None, BTreeSet::new())?;
    assert!(block_on(flat.search(&[0., 0.], &options(2)?, &filter)).is_err());
    assert!(
        RowFilter::try_new(
            snapshot(),
            Some(BTreeSet::from([row(99, 0)])),
            BTreeSet::new()
        )
        .is_err()
    );
    assert!(RowFilter::try_new(snapshot(), None, BTreeSet::from([row(10, 2)])).is_err());
    let mut configured = options(2)?;
    configured.backend_options = Bytes::from_static(b"unexpected");
    assert!(block_on(flat.search(&[0., 0.], &configured, &all_rows()?)).is_err());
    assert!(block_on(flat.search(&[f32::MAX, 0.], &options(2)?, &all_rows()?)).is_err());
    Ok(())
}

#[test]
fn test_batch_query_order_and_empty_batch() -> VortexResult<()> {
    let flat = index()?;
    let queries = [Buffer::copy_from([0., 0.]), Buffer::copy_from([1., 1.])];
    let hits = block_on(flat.search_batch(&queries, &options(1)?, &all_rows()?))?;
    assert_eq!(hits[0][0].row, row(10, 0));
    assert_eq!(hits[1][0].row, row(30, 1));
    assert!(block_on(flat.search_batch(&[], &options(1)?, &all_rows()?))?.is_empty());
    let mut approximate = options(1)?;
    approximate.mode = SearchMode::Approximate;
    assert_eq!(
        block_on(flat.search_batch(&queries, &approximate, &all_rows()?))?,
        hits
    );
    Ok(())
}

#[test]
fn test_scan_selections_are_per_file_and_sorted() -> VortexResult<()> {
    let candidates = ScalarCandidates {
        snapshot: snapshot(),
        rows: vec![row(30, 1), row(10, 1), row(10, 0), row(10, 1)],
        guarantee: CandidateGuarantee::Superset,
    };
    let selections = candidates.selections(&snapshot())?;
    assert_eq!(selections.len(), 2);
    for (file, expected) in [(10, vec![0, 1]), (30, vec![1])] {
        let Some(Selection::IncludeByIndex(indices)) = selections.get(&file) else {
            vortex_bail!("Expected sorted row selection");
        };
        assert_eq!(indices.as_ref(), expected.as_slice());
    }
    let mut stale = snapshot();
    stale.version = "different-snapshot".into();
    assert!(candidates.selections(&stale).is_err());
    Ok(())
}

#[test]
fn test_source_batch_alignment_and_duplicate_take_rows() -> VortexResult<()> {
    let data = PrimitiveArray::from_iter([10i64, 10]).into_array();
    let batch = SourceBatch::try_new(&snapshot(), vec![row(10, 0), row(10, 0)], data.clone())?;
    assert_eq!(batch.rows.len(), 2);
    assert!(SourceBatch::try_new(&snapshot(), vec![row(10, 0)], data.clone()).is_err());
    assert!(SourceBatch::try_new(&snapshot(), vec![row(10, 0), row(99, 0)], data).is_err());
    Ok(())
}

// Registry tests use a prebuilt in-memory provider. No persistence adapter is implied.
struct ReferenceProvider(Arc<FlatIndex>);

#[async_trait]
impl IndexProvider for ReferenceProvider {
    fn id(&self) -> &str {
        FlatIndex::ID
    }
    fn supports_version(&self, version: u32) -> bool {
        version == 1
    }
    async fn open(
        &self,
        _: &IndexMetadata,
        _: Arc<dyn IndexStore>,
    ) -> VortexResult<Arc<dyn Index>> {
        Ok(Arc::clone(&self.0) as Arc<dyn Index>)
    }
}

struct NoStorage;

#[async_trait]
impl IndexStore for NoStorage {
    async fn read(&self, _: &IndexArtifact) -> VortexResult<Bytes> {
        vortex_bail!("The in-memory reference does not access storage")
    }
    async fn write(&self, _: &str, _: Bytes) -> VortexResult<IndexArtifact> {
        vortex_bail!("The in-memory reference does not access storage")
    }
}

#[test]
fn test_registry_dispatch_and_duplicate_registration() -> VortexResult<()> {
    let mut registry = IndexRegistry::default();
    let provider = Arc::new(ReferenceProvider(Arc::new(index()?)));
    registry.register(Arc::clone(&provider) as Arc<dyn IndexProvider>)?;
    assert!(registry.register(provider).is_err());
    assert!(registry.provider("unknown.backend").is_err());
    assert!(registry.provider(FlatIndex::ID)?.builder().is_none());
    let handle = block_on(registry.open(&metadata(), &snapshot(), Arc::new(NoStorage)))?;
    let vector = handle
        .as_vector()
        .ok_or_else(|| vortex_err!("Missing vector capability"))?;
    let result = block_on(vector.search(&[0., 0.], &options(1)?, &all_rows()?))?;
    assert_eq!(result[0].row, row(10, 0));
    Ok(())
}

#[test]
fn test_registry_rejects_wrong_versions_snapshots_and_generations() -> VortexResult<()> {
    let mut registry = IndexRegistry::default();
    registry.register(Arc::new(ReferenceProvider(Arc::new(index()?))))?;
    let mut wrong = metadata();
    wrong.backend_version = 2;
    assert!(block_on(registry.open(&wrong, &snapshot(), Arc::new(NoStorage))).is_err());
    wrong = metadata();
    wrong.generation = "generation-not-returned-by-provider".into();
    assert!(block_on(registry.open(&wrong, &snapshot(), Arc::new(NoStorage))).is_err());
    let mut changed = snapshot();
    changed.version = "new-snapshot".into();
    assert!(block_on(registry.open(&metadata(), &changed, Arc::new(NoStorage))).is_err());
    Ok(())
}
