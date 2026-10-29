// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use futures::TryStreamExt;
use vortex_fts::AttributeType;
use vortex_fts::TANTIVY_ID;
use vortex_fts::TantivyBuildOptions;
use vortex_fts::TantivyProvider;
use vortex_index::Index;
use vortex_index::attributes::AttributeBlockKind;
use vortex_index::attributes::AttributePredicate as Predicate;
use vortex_index::attributes::AttributeValue;

use super::*;
use crate::HnswPayload;

async fn attributes(root: &Path) -> VortexResult<Arc<dyn Index>> {
    let mut metadata = metadata(256);
    metadata.backend = TANTIVY_ID.into();
    metadata.backend_version = 2;
    metadata.generation = "attributes".into();
    metadata.fields = vec!["category".into(), "price".into()];
    let data = StructArray::from_fields(&[
        (
            "category",
            PrimitiveArray::new(
                Buffer::from_iter((0..256).map(|id| id % 8u64)),
                Validity::NonNullable,
            )
            .into_array(),
        ),
        (
            "price",
            PrimitiveArray::new(
                Buffer::from_iter((0..256).map(|id| f64::from(id / 4))),
                Validity::NonNullable,
            )
            .into_array(),
        ),
    ])?
    .into_array();
    let provider = TantivyProvider::try_new(VortexSession::default(), root.to_owned())?;
    let store = Arc::new(LocalIndexStore::create(root, "attributes", LIMITS)?);
    let metadata = provider
        .build(
            IndexBuildRequest {
                metadata: metadata.clone(),
                backend_options: Bytes::from(
                    serde_json::to_vec(&TantivyBuildOptions {
                        text_fields: vec![],
                        attribute_fields: [
                            ("category".into(), AttributeType::UInt),
                            ("price".into(), AttributeType::Float),
                        ]
                        .into(),
                    })
                    .map_err(|err| vortex_err!(External: err))?,
                ),
            },
            Arc::new(Source { metadata, data }),
            Arc::clone(&store) as Arc<dyn IndexStore>,
        )
        .await?;
    let generation = store.seal(&metadata)?;
    let (store, metadata) = LocalIndexStore::open(root, &generation, &metadata.snapshot, LIMITS)?;
    provider.open(&metadata, Arc::new(store)).await
}

fn payload_config() -> HnswBuildOptions {
    HnswBuildOptions {
        m: 2,
        payload_m: 8,
        payload_fields: [
            ("category".into(), AttributeBlockKind::Equality),
            ("price".into(), AttributeBlockKind::NumericRange),
        ]
        .into(),
        payload_target_rows: 32,
        ..config()
    }
}

#[test]
fn test_tantivy_payload_build_reload_and_filtered_batch() -> VortexResult<()> {
    block_on(async {
        let root = tempfile::tempdir()?;
        let attributes = attributes(root.path()).await?;
        let store = Arc::new(LocalIndexStore::create(
            root.path(),
            "generation-1",
            LIMITS,
        )?);
        let metadata = builder(root.path())?
            .with_attributes(Arc::clone(&attributes))?
            .build(
                request(Bytes::from(
                    serde_json::to_vec(&payload_config())
                        .map_err(|err| vortex_err!(External: err))?,
                )),
                data()?,
                Arc::clone(&store) as Arc<dyn IndexStore>,
            )
            .await?;
        let generation = store.seal(&metadata)?;
        let (store, metadata) =
            LocalIndexStore::open(root.path(), &generation, &metadata.snapshot, LIMITS)?;
        let payload = HnswPayload::decode(
            &store
                .read(
                    metadata
                        .artifacts
                        .iter()
                        .find(|artifact| artifact.path == crate::bundle::PAYLOAD)
                        .ok_or_else(|| vortex_err!("Missing payload"))?,
                )
                .await?,
            256,
            100_000,
            256,
        )?;
        assert!(payload.edge_count() > 0);
        let provider = HnswProvider::try_new(root.path().to_owned(), HnswLimits::default())?;
        let store: Arc<dyn IndexStore> = Arc::new(store);
        let index = provider.open(&metadata, Arc::clone(&store)).await?;
        let vectors = index
            .as_vector()
            .ok_or_else(|| vortex_err!("Missing vectors"))?;
        let mut flat_metadata = metadata.clone();
        flat_metadata.backend = FlatIndex::ID.into();
        flat_metadata.backend_version = 1;
        flat_metadata.artifacts.clear();
        let flat = FlatIndex::try_new(
            flat_metadata,
            vectors.spec(),
            rows(256),
            Buffer::from_iter((0..256).flat_map(vector)),
        )?;
        let visibility = RowFilter::try_new(
            metadata.snapshot.clone(),
            None,
            [rows(256)[3], rows(256)[11]].into(),
        )?;
        let predicates = [
            Predicate::Eq("category".into(), AttributeValue::UInt(3)),
            Predicate::And(vec![
                Predicate::In(
                    "category".into(),
                    vec![AttributeValue::UInt(2), AttributeValue::UInt(3)],
                ),
                Predicate::Range(
                    "price".into(),
                    AttributeValue::Float(10.0.try_into()?),
                    AttributeValue::Float(40.0.try_into()?),
                ),
            ]),
            Predicate::Or(vec![]),
            Predicate::And(vec![]),
        ];
        let queries: Vec<Buffer<f32>> = [3, 47, 101, 255]
            .into_iter()
            .map(|id| vector(id).into())
            .collect();
        for predicate in predicates {
            let filter = attributes
                .as_attributes()
                .ok_or_else(|| vortex_err!("Missing attributes"))?
                .filter(&predicate, &visibility)
                .await?;
            let actual = vectors
                .search_batch(&queries, &options(10, "{\"ef\":256}")?, &filter)
                .await?;
            let exact = flat
                .search_batch(
                    &queries,
                    &VectorSearchOptions {
                        mode: SearchMode::Exact,
                        ..options(10, "")?
                    },
                    &filter,
                )
                .await?;
            for ((query, hits), expected) in queries.iter().zip(&actual).zip(exact) {
                assert_eq!(hits.len(), expected.len());
                for (hit, expected) in hits.iter().zip(expected) {
                    assert_eq!(hit.row, expected.row);
                    assert!((hit.distance - expected.distance).abs() < 1e-5);
                    assert!(filter.includes(hit.row));
                }
                assert_eq!(
                    *hits,
                    vectors
                        .search(query, &options(10, "{\"ef\":256}")?, &filter)
                        .await?
                );
            }
            // Reopen uses only persisted artifacts, with identical filtered results.
            let reopened = provider.open(&metadata, Arc::clone(&store)).await?;
            assert_eq!(
                actual,
                reopened
                    .as_vector()
                    .ok_or_else(|| vortex_err!("Missing vectors"))?
                    .search_batch(&queries, &options(10, "{\"ef\":256}")?, &filter)
                    .await?
            );
        }
        let all = RowFilter::try_new(metadata.snapshot.clone(), None, BTreeSet::new())?;
        let one_row = rows(256)[19];
        let singleton = RowFilter::try_new(
            metadata.snapshot.clone(),
            Some([one_row].into()),
            BTreeSet::new(),
        )?;
        let singleton_hits = vectors
            .search(&vector(3), &options(10, "")?, &singleton)
            .await?;
        assert_eq!(singleton_hits.len(), 1);
        assert_eq!(singleton_hits[0].row, one_row);
        let excluded = RowFilter::try_new(
            metadata.snapshot.clone(),
            Some([one_row].into()),
            [one_row].into(),
        )?;
        assert!(
            vectors
                .search(&vector(3), &options(10, "")?, &excluded)
                .await?
                .is_empty()
        );
        assert!(
            vectors
                .search(&[f32::NAN; 8], &options(10, "")?, &excluded)
                .await
                .is_err()
        );
        let none = RowFilter::try_new(
            metadata.snapshot.clone(),
            None,
            rows(256).into_iter().collect(),
        )?;
        assert!(
            vectors
                .search(&vector(3), &options(10, "")?, &none)
                .await?
                .is_empty()
        );
        thread::scope(|scope| -> VortexResult<()> {
            let handles: Vec<_> = (0..4)
                .map(|_| {
                    scope.spawn(|| -> VortexResult<()> {
                        for _ in 0..20 {
                            assert_eq!(
                                block_on(vectors.search(
                                    &vector(3),
                                    &options(10, "")?,
                                    &singleton
                                ))?,
                                singleton_hits
                            );
                        }
                        Ok(())
                    })
                })
                .collect();
            for handle in handles {
                handle
                    .join()
                    .map_err(|_| vortex_err!("Filtered search thread panicked"))??;
            }
            Ok(())
        })?;
        let mut wrong = metadata.snapshot.clone();
        wrong.version = "other".into();
        assert!(
            vectors
                .search(
                    &vector(0),
                    &options(10, "")?,
                    &RowFilter::try_new(wrong, Some(BTreeSet::new()), BTreeSet::new())?
                )
                .await
                .is_err()
        );
        assert!(
            vectors
                .search(
                    &vector(0),
                    &VectorSearchOptions {
                        mode: SearchMode::Exact,
                        ..options(10, "")?
                    },
                    &all
                )
                .await
                .is_err()
        );
        // A required payload file cannot be omitted or rebound through metadata.
        let mut missing = metadata.clone();
        missing
            .artifacts
            .retain(|artifact| artifact.path != crate::bundle::PAYLOAD);
        assert!(provider.open(&missing, Arc::clone(&store)).await.is_err());
        let mut unsupported = metadata;
        unsupported.backend_version = 2;
        assert!(provider.open(&unsupported, store).await.is_err());
        Ok(())
    })
}

#[rstest]
#[case(crate::bundle::INDEX)]
#[case(crate::bundle::ROWS)]
#[case(crate::bundle::PAYLOAD)]
fn test_descriptor_rejects_rebound_artifacts(#[case] changed_path: &str) -> VortexResult<()> {
    block_on(async {
        let root = tempfile::tempdir()?;
        let (original, mut metadata) = prepare(root.path()).await?;
        metadata.generation = "rebound".into();
        let copied = Arc::new(LocalIndexStore::create(root.path(), "rebound", LIMITS)?);
        let mut inventory = Vec::new();
        for artifact in &metadata.artifacts {
            let mut bytes = original.read(artifact).await?.to_vec();
            if artifact.path == changed_path {
                if changed_path == crate::bundle::PAYLOAD {
                    let mut offsets = vec![1; 257];
                    offsets[0] = 0;
                    bytes = HnswPayload::try_new(offsets, vec![1])?.encode(256)?;
                } else if changed_path == crate::bundle::ROWS {
                    // A different but complete label mapping passes row validation.
                    for offset in 0..16 {
                        bytes.swap(16 + offset, 32 + offset);
                    }
                } else {
                    let last = bytes.len() - 1;
                    bytes[last] ^= 1;
                }
            }
            inventory.push(copied.write(&artifact.path, Bytes::from(bytes)).await?);
        }
        metadata.artifacts = inventory;
        let generation = copied.seal(&metadata)?;
        let (copied, metadata) =
            LocalIndexStore::open(root.path(), &generation, &metadata.snapshot, LIMITS)?;
        let error = HnswProvider::try_new(root.path().to_owned(), HnswLimits::default())?
            .open(&metadata, Arc::new(copied))
            .await
            .expect_err("Rebound graph must fail");
        assert!(error.to_string().contains("artifact binding mismatch"));
        Ok(())
    })
}

#[test]
fn test_payload_requires_bound_attributes_and_hard_budgets() -> VortexResult<()> {
    block_on(async {
        let root = tempfile::tempdir()?;
        let attributes = attributes(root.path()).await?;
        let options = Bytes::from(
            serde_json::to_vec(&payload_config()).map_err(|err| vortex_err!(External: err))?,
        );
        let store: Arc<dyn IndexStore> = Arc::new(LocalIndexStore::create(
            root.path(),
            "generation-1",
            LIMITS,
        )?);
        assert!(
            builder(root.path())?
                .build(request(options.clone()), data()?, Arc::clone(&store))
                .await
                .is_err()
        );
        let limited = HnswIndexBuilder::try_new(
            root.path().to_owned(),
            VortexSession::default(),
            HnswBuildLimits {
                max_payload_edges: 1,
                ..HnswBuildLimits::default()
            },
        )?
        .with_attributes(Arc::clone(&attributes))?;
        assert!(
            limited
                .build(request(options.clone()), data()?, Arc::clone(&store))
                .await
                .is_err()
        );
        let mut source = Source {
            metadata: metadata(256),
            data: data()?
                .scan(&[11, 29], &["embedding".into()])?
                .try_next()
                .await?
                .ok_or_else(|| vortex_err!("Missing batch"))?
                .data,
        };
        source.metadata.snapshot.version = "other".into();
        let mut request = request(options);
        request.metadata.snapshot.version = "other".into();
        assert!(
            builder(root.path())?
                .with_attributes(attributes)?
                .build(request, Arc::new(source), store)
                .await
                .is_err()
        );
        Ok(())
    })
}
