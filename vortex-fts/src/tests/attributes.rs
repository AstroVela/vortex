// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Arc;

use bytes::Bytes;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::BoolArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::buffer::Buffer;
use vortex::buffer::buffer;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::session::VortexSession;
use vortex_index::DistanceMetric;
use vortex_index::FlatIndex;
use vortex_index::Index;
use vortex_index::IndexBuildRequest;
use vortex_index::IndexBuilder;
use vortex_index::IndexMetadata;
use vortex_index::IndexProvider;
use vortex_index::IndexStore;
use vortex_index::RowAddress;
use vortex_index::RowFilter;
use vortex_index::SearchMode;
use vortex_index::Snapshot;
use vortex_index::SourceFile;
use vortex_index::VectorIndex;
use vortex_index::VectorSearchOptions;
use vortex_index::VectorSpec;
use vortex_index::attributes::AttributePredicate as P;
use vortex_index::attributes::AttributeValue as V;
use vortex_index::attributes::FloatValue;
use vortex_index::file::LocalFileSource;
use vortex_index::file::file_version;
use vortex_index::file::schema_fingerprint;
use vortex_index::store::LocalIndexStore;
use vortex_index::store::LocalStoreLimits;

use super::write_file;
use crate::AttributeType as T;
use crate::TANTIVY_ID;
use crate::TantivyBuildOptions;
use crate::TantivyProvider;

const LIMITS: LocalStoreLimits = LocalStoreLimits {
    max_artifact_bytes: 64 * 1024 * 1024,
    max_manifest_bytes: 1024 * 1024,
};

async fn build(
    root: &Path,
    data: ArrayRef,
    options: TantivyBuildOptions,
    partial: bool,
) -> VortexResult<Arc<dyn Index>> {
    let session = VortexSession::default();
    let mut files = Vec::new();
    for id in [3, 7] {
        let path = root.join(format!("{id}.vortex"));
        write_file(&session, &path, data.clone()).await?;
        files.push(SourceFile {
            id,
            uri: path.to_string_lossy().into(),
            version: file_version(&std::fs::read(path)?),
            row_count: data.len() as u64,
        });
    }
    let snapshot = Snapshot {
        dataset_id: "attributes".into(),
        version: "v1".into(),
        schema_fingerprint: schema_fingerprint(data.dtype())?,
        files,
    };
    let source = Arc::new(
        LocalFileSource::open(
            snapshot.clone(),
            data.dtype().clone(),
            4 * 1024 * 1024,
            session.clone(),
        )
        .await?,
    );
    let provider = TantivyProvider::try_new(session, root.to_owned())?;
    let store = Arc::new(LocalIndexStore::create(root, "generation", LIMITS)?);
    let fields: BTreeSet<_> = options
        .text_fields
        .iter()
        .chain(options.attribute_fields.keys())
        .cloned()
        .collect();
    let metadata = provider
        .build(
            IndexBuildRequest {
                metadata: IndexMetadata {
                    format_version: 1,
                    name: "search".into(),
                    generation: "generation".into(),
                    backend: TANTIVY_ID.into(),
                    backend_version: 2,
                    snapshot: snapshot.clone(),
                    fields: fields.into_iter().collect(),
                    covered_files: if partial { vec![3] } else { vec![3, 7] },
                    artifacts: vec![],
                },
                backend_options: Bytes::from(
                    serde_json::to_vec(&options).map_err(|err| vortex_err!(External: err))?,
                ),
            },
            source,
            Arc::clone(&store) as Arc<dyn IndexStore>,
        )
        .await?;
    let generation = store.seal(&metadata)?;
    drop(store);
    let (store, metadata) = LocalIndexStore::open(root, &generation, &snapshot, LIMITS)?;
    provider.open(&metadata, Arc::new(store)).await
}

fn row(file_id: u64, row_offset: u64) -> RowAddress {
    RowAddress {
        file_id,
        row_offset,
    }
}
fn string(value: &str) -> V {
    V::String(value.into())
}
fn float(value: f64) -> VortexResult<V> {
    Ok(V::Float(FloatValue::try_from(value)?))
}
fn all(index: &dyn Index) -> VortexResult<RowFilter> {
    RowFilter::try_new(index.metadata().snapshot.clone(), None, BTreeSet::new())
}
fn offsets(filter: &RowFilter) -> Vec<u64> {
    (0..filter.snapshot().files[0].row_count)
        .filter(|offset| filter.includes(row(3, *offset)))
        .collect()
}

#[tokio::test]
async fn test_shared_tantivy_attributes_text_and_vector_prefilter() -> VortexResult<()> {
    let dir = tempfile::tempdir()?;
    let data = StructArray::from_fields(&[
        (
            "category",
            VarBinViewArray::from_iter_nullable_str([
                Some("tech"),
                Some("Tech"),
                Some("tech news"),
                None,
                Some(""),
                None,
            ])
            .into_array(),
        ),
        ("signed", buffer![-10i64, 2, 3, 4, 5, 6].into_array()),
        ("unsigned", buffer![u64::MAX, 2, 3, 4, 5, 6].into_array()),
        (
            "price",
            buffer![-0.0f64, 0.0, 2.5, 3.0, 4.0, 9.0].into_array(),
        ),
        (
            "active",
            BoolArray::from_iter([true, false, true, false, true, true]).into_array(),
        ),
    ])?;
    let data = data.into_array();
    let options = TantivyBuildOptions {
        text_fields: vec!["category".into()],
        attribute_fields: [
            ("category".into(), T::String),
            ("signed".into(), T::Int),
            ("unsigned".into(), T::UInt),
            ("price".into(), T::Float),
            ("active".into(), T::Bool),
        ]
        .into(),
    };
    let index = build(&dir.path().canonicalize()?, data, options, false).await?;
    let attributes = index
        .as_attributes()
        .ok_or_else(|| vortex_err!("Missing attribute capability"))?;
    let visible = all(index.as_ref())?;
    let cases = vec![
        (P::Eq("category".into(), string("tech")), vec![0]),
        (P::Eq("category".into(), string("Tech")), vec![1]),
        (P::Eq("category".into(), string("tech news")), vec![2]),
        (P::Eq("category".into(), string("")), vec![4]),
        (P::Exists("category".into()), vec![0, 1, 2, 4]),
        (
            P::Range("category".into(), string("Tech"), string("tech")),
            vec![0, 1],
        ),
        (
            P::In(
                "category".into(),
                vec![string("Tech"), string("tech news"), string("Tech")],
            ),
            vec![1, 2],
        ),
        (
            P::Range("signed".into(), V::Int(-10), V::Int(3)),
            vec![0, 1, 2],
        ),
        (
            P::Range("unsigned".into(), V::UInt(5), V::UInt(u64::MAX)),
            vec![0, 4, 5],
        ),
        (P::Eq("price".into(), float(0.0)?), vec![0, 1]),
        (
            P::Range("price".into(), float(-0.0)?, float(2.5)?),
            vec![0, 1, 2],
        ),
        (P::Eq("active".into(), V::Bool(true)), vec![0, 2, 4, 5]),
        (
            P::And(vec![
                P::Exists("category".into()),
                P::Or(vec![
                    P::Eq("signed".into(), V::Int(2)),
                    P::Eq("signed".into(), V::Int(4)),
                ]),
            ]),
            vec![1],
        ),
        (P::And(vec![]), vec![0, 1, 2, 3, 4, 5]),
        (P::Or(vec![]), vec![]),
        (P::In("category".into(), vec![]), vec![]),
    ];
    for (predicate, expected) in cases {
        let result = attributes.filter(&predicate, &visible).await?;
        assert_eq!(offsets(&result), expected, "{predicate:?}");
        for offset in 0..6 {
            assert_eq!(result.includes(row(7, offset)), expected.contains(&offset));
        }
    }
    let text = index
        .as_full_text()
        .ok_or_else(|| vortex_err!("Missing text capability"))?;
    let text_filter = text.filter_text("tech", &visible).await?;
    assert_eq!(offsets(&text_filter), vec![0, 1, 2]);
    assert_eq!(
        text.search_text(
            "tech",
            NonZeroUsize::new(1).unwrap(),
            &Bytes::new(),
            &visible
        )
        .await?
        .len(),
        1
    );
    let visibility = RowFilter::try_new(
        visible.snapshot().clone(),
        Some([row(3, 0), row(3, 1), row(7, 1)].into()),
        [row(3, 1)].into(),
    )?;
    let result = attributes.filter(&P::And(vec![]), &visibility).await?;
    assert!(
        result.includes(row(3, 0)) && result.includes(row(7, 1)) && !result.includes(row(3, 1))
    );
    assert_eq!(
        offsets(&text.filter_text("tech", &visibility).await?),
        vec![0]
    );
    for predicate in [
        P::Eq("signed".into(), V::UInt(2)),
        P::Exists("missing".into()),
        P::Range("signed".into(), V::Int(5), V::Int(1)),
        P::In("signed".into(), vec![V::Int(2), string("2")]),
    ] {
        assert!(attributes.filter(&predicate, &visible).await.is_err());
    }
    let mut deep = P::And(vec![]);
    for _ in 0..66 {
        deep = P::And(vec![deep]);
    }
    assert!(attributes.filter(&deep, &visible).await.is_err());
    let mut stale = visible.snapshot().clone();
    stale.version = "new".into();
    let stale = RowFilter::try_new(stale, None, BTreeSet::new())?;
    assert!(attributes.filter(&P::And(vec![]), &stale).await.is_err());
    assert!(text.filter_text("tech", &stale).await.is_err());

    // Vector row order differs from Tantivy's doc IDs. The nearest unmatched row
    // must not consume k; prefilter before exact retrieval provides the oracle.
    let mut metadata = index.metadata().clone();
    metadata.backend = FlatIndex::ID.into();
    metadata.backend_version = 1;
    metadata.artifacts.clear();
    metadata.fields = vec!["embedding".into()];
    let rows: Vec<_> = [7, 3]
        .into_iter()
        .flat_map(|id| (0..6).rev().map(move |offset| row(id, offset)))
        .collect();
    let values = rows
        .iter()
        .map(|row| row.row_offset as f32)
        .collect::<Vec<_>>();
    let vectors = FlatIndex::try_new(
        metadata,
        VectorSpec {
            dimension: NonZeroUsize::new(1).unwrap(),
            metric: DistanceMetric::SquaredL2,
        },
        rows,
        Buffer::from(values),
    )?;
    let allowed = attributes
        .filter(&P::Eq("category".into(), string("tech news")), &visible)
        .await?;
    let options = VectorSearchOptions {
        k: NonZeroUsize::new(1).unwrap(),
        mode: SearchMode::Exact,
        backend_options: Bytes::new(),
    };
    assert_eq!(
        vectors.search(&[0.0], &options, &allowed).await?[0].row,
        row(3, 2)
    );
    Ok(())
}

#[tokio::test]
async fn test_prefilter_not_truncated_by_full_text_limit() -> VortexResult<()> {
    let dir = tempfile::tempdir()?;
    let data = StructArray::from_fields(&[(
        "category",
        VarBinViewArray::from_iter_str(std::iter::repeat_n("tech", 5_001)).into_array(),
    )])?
    .into_array();
    let index = build(
        &dir.path().canonicalize()?,
        data,
        TantivyBuildOptions {
            text_fields: vec!["category".into()],
            attribute_fields: [("category".into(), T::String)].into(),
        },
        false,
    )
    .await?;
    let visible = all(index.as_ref())?;
    let attr = index
        .as_attributes()
        .unwrap()
        .filter(&P::Eq("category".into(), string("tech")), &visible)
        .await?;
    let text = index
        .as_full_text()
        .unwrap()
        .filter_text("tech", &visible)
        .await?;
    for file in [3, 7] {
        for offset in 0..5_001 {
            assert!(attr.includes(row(file, offset)) && text.includes(row(file, offset)));
        }
    }
    Ok(())
}

#[tokio::test]
async fn test_partial_coverage_cannot_be_an_ann_allow_list() -> VortexResult<()> {
    let dir = tempfile::tempdir()?;
    let data = StructArray::from_fields(&[(
        "category",
        VarBinViewArray::from_iter_str(["tech"]).into_array(),
    )])?
    .into_array();
    let index = build(
        &dir.path().canonicalize()?,
        data,
        TantivyBuildOptions {
            text_fields: vec!["category".into()],
            attribute_fields: [("category".into(), T::String)].into(),
        },
        true,
    )
    .await?;
    let visible = all(index.as_ref())?;
    assert!(
        index
            .as_attributes()
            .unwrap()
            .filter(&P::And(vec![]), &visible)
            .await
            .is_err()
    );
    assert!(
        index
            .as_full_text()
            .unwrap()
            .filter_text("tech", &visible)
            .await
            .is_err()
    );
    Ok(())
}

#[tokio::test]
async fn test_attribute_only_raw_terms_and_invalid_source_values() -> VortexResult<()> {
    let long = "a".repeat(200);
    let dir = tempfile::tempdir()?;
    let data = StructArray::from_fields(&[(
        "category",
        VarBinViewArray::from_iter_str([long.as_str(), "x:y AND \"z\""]).into_array(),
    )])?
    .into_array();
    let config = TantivyBuildOptions {
        text_fields: vec![],
        attribute_fields: [("category".into(), T::String)].into(),
    };
    let index = build(&dir.path().canonicalize()?, data, config.clone(), false).await?;
    assert!(index.as_full_text().is_none());
    let visible = all(index.as_ref())?;
    let attributes = index.as_attributes().unwrap();
    assert_eq!(
        offsets(
            &attributes
                .filter(&P::Eq("category".into(), string(&long)), &visible)
                .await?
        ),
        vec![0]
    );
    assert_eq!(
        offsets(
            &attributes
                .filter(&P::Eq("category".into(), string("x:y AND \"z\"")), &visible)
                .await?
        ),
        vec![1]
    );
    let excessive = "a".repeat(tantivy::tokenizer::MAX_TOKEN_LEN + 1);
    assert!(
        attributes
            .filter(&P::Eq("category".into(), string(&excessive)), &visible)
            .await
            .is_err()
    );
    let dir = tempfile::tempdir()?;
    let data = StructArray::from_fields(&[(
        "category",
        VarBinViewArray::from_iter_str([excessive]).into_array(),
    )])?
    .into_array();
    assert!(
        build(&dir.path().canonicalize()?, data, config, false)
            .await
            .is_err()
    );
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let dir = tempfile::tempdir()?;
        let data = StructArray::from_fields(&[("price", Buffer::from(vec![value]).into_array())])?
            .into_array();
        let config = TantivyBuildOptions {
            text_fields: vec![],
            attribute_fields: [("price".into(), T::Float)].into(),
        };
        assert!(
            build(&dir.path().canonicalize()?, data, config, false)
                .await
                .is_err()
        );
        assert!(FloatValue::try_from(value).is_err());
    }
    let dir = tempfile::tempdir()?;
    let data = StructArray::from_fields(&[("price", buffer![2i64].into_array())])?.into_array();
    let config = TantivyBuildOptions {
        text_fields: vec![],
        attribute_fields: [("price".into(), T::Float)].into(),
    };
    assert!(
        build(&dir.path().canonicalize()?, data, config, false)
            .await
            .is_err()
    );
    Ok(())
}
