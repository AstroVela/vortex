// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs;
use std::path::Path;

use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::dtype::FieldNames;
use vortex::array::validity::Validity;
use vortex::buffer::buffer;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::file::WriteOptionsSessionExt;
use vortex::session::VortexSession;
use vortex_array::assert_arrays_eq;

use super::*;

async fn write_file(session: &VortexSession, path: &Path, array: ArrayRef) -> VortexResult<()> {
    let mut bytes = Vec::new();
    session
        .write_options()
        .write(&mut bytes, array.to_array_stream())
        .await?;
    fs::write(path, bytes)?;
    Ok(())
}

fn table() -> VortexResult<ArrayRef> {
    Ok(StructArray::from_fields(&[
        (
            "title",
            VarBinViewArray::from_iter_str([
                "Rust search",
                "Vortex engine",
                "Other",
                "",
                "Masked rust",
            ])
            .into_array(),
        ),
        (
            "body",
            VarBinViewArray::from_iter_nullable_str([
                Some("fast full text search"),
                Some("a full text engine in rust"),
                None,
                Some(""),
                Some("masked secret"),
            ])
            .into_array(),
        ),
        ("id", buffer![10u64, 11, 12, 13, 14].into_array()),
    ])?
    .into_array())
}

#[tokio::test]
async fn test_queries_persistence_and_fetch() -> VortexResult<()> {
    let session = VortexSession::default();
    let dir = tempfile::tempdir()?;
    let source = dir.path().join("source.vortex");
    let destination = dir.path().join("fts");
    let original = table()?;
    write_file(&session, &source, original.clone()).await?;
    build_index(
        &session,
        &source,
        &destination,
        &["title".into(), "body".into()],
    )
    .await?;
    let index = SearchIndex::open(&destination)?;
    for (query, expected) in [
        ("search", 1),
        ("\"full text\"", 2),
        ("rust AND engine", 1),
        ("title:rust", 2),
        ("body:rust", 1),
        ("rust NOT masked", 2),
        ("unfindable", 0),
        ("*", 5),
        ("title:RUST", 2),
        ("\"search fast\"", 0),
        ("\"full engine\"", 0),
    ] {
        let result = index.search(&source, query, 10)?;
        assert_eq!(result.total_hits, expected, "{query}");
        assert_eq!(result.hits.len(), expected);
        assert!(result.hits.windows(2).all(|h| h[0].score >= h[1].score));
    }
    let result = index.search(&source, "engine", 1)?;
    assert_eq!(result.hits[0].row_id, 1);
    let limited = index.search(&source, "rust", 1)?;
    assert_eq!(limited.total_hits, 3);
    assert_eq!(limited.hits.len(), 1);
    assert!(index.search(&source, "rust", 0)?.hits.is_empty());
    assert!(index.search(&source, "\"unterminated", 10).is_err());
    assert!(index.search(&source, "unknown:rust", 10).is_err());
    assert!(index.search(&source, " ", 10).is_err());
    assert!(index.search(&source, "*", 10_001).is_err());
    let fetched = index.read_rows(&session, &source, &[0, 1, 2, 3, 4]).await?;
    assert_arrays_eq!(original, fetched, &mut session.create_execution_ctx());
    assert_eq!(index.read_rows(&session, &source, &[]).await?.len(), 0);
    assert!(index.read_rows(&session, &source, &[2, 1]).await.is_err());
    assert!(index.read_rows(&session, &source, &[1, 1]).await.is_err());
    assert!(index.read_rows(&session, &source, &[5]).await.is_err());
    // Index identity is content based, so moving/copying an unchanged file is supported.
    let copied = dir.path().join("copied.vortex");
    fs::copy(&source, &copied)?;
    assert_eq!(index.search(&copied, "search", 10)?.total_hits, 1);
    Ok(())
}

#[tokio::test]
async fn test_nullable_columns_across_scan_batches() -> VortexResult<()> {
    let session = VortexSession::default();
    let dir = tempfile::tempdir()?;
    let source = dir.path().join("source.vortex");
    let destination = dir.path().join("fts");
    let batch = StructArray::new(
        FieldNames::from(["text"]),
        vec![
            VarBinViewArray::from_iter_nullable_str([Some("visible rust"), None, None, Some("")])
                .into_array(),
        ],
        4,
        Validity::NonNullable,
    )
    .into_array();
    let dtype = batch.dtype().clone();
    let chunks = vec![batch; 3000];
    let array = ChunkedArray::try_new(chunks, dtype)?.into_array();
    write_file(&session, &source, array).await?;
    let metadata = build_index(&session, &source, &destination, &["text".into()]).await?;
    assert_eq!(metadata.row_count, 12000);
    let index = SearchIndex::open(&destination)?;
    assert_eq!(index.search(&source, "secret", 10)?.total_hits, 0);
    let results = index.search(&source, "rust", 10000)?;
    assert_eq!(results.total_hits, 3000);
    let mut rows = results
        .hits
        .iter()
        .map(|hit| hit.row_id)
        .collect::<Vec<_>>();
    rows.sort_unstable();
    assert_eq!(rows, (0..12000).step_by(4).collect::<Vec<_>>());
    assert_eq!(index.read_rows(&session, &source, &rows).await?.len(), 3000);
    Ok(())
}

#[tokio::test]
async fn test_empty_file() -> VortexResult<()> {
    let session = VortexSession::default();
    let dir = tempfile::tempdir()?;
    let source = dir.path().join("empty.vortex");
    let destination = dir.path().join("fts");
    let array = StructArray::from_fields(&[(
        "text",
        VarBinViewArray::from_iter_str(Vec::<String>::new()).into_array(),
    )])?
    .into_array();
    write_file(&session, &source, array).await?;
    build_index(&session, &source, &destination, &["text".into()]).await?;
    let index = SearchIndex::open(destination)?;
    assert_eq!(index.search(&source, "*", 10)?.total_hits, 0);
    Ok(())
}

#[tokio::test]
async fn test_invalid_columns_do_not_create_destination() -> VortexResult<()> {
    let session = VortexSession::default();
    let dir = tempfile::tempdir()?;
    let source = dir.path().join("source.vortex");
    let destination = dir.path().join("fts");
    write_file(&session, &source, table()?).await?;
    for columns in [
        vec![],
        vec!["missing"],
        vec!["id"],
        vec!["title", "title"],
        vec!["_vortex_row_id"],
        vec!["a.b"],
    ] {
        let columns = columns.into_iter().map(str::to_string).collect::<Vec<_>>();
        assert!(
            build_index(&session, &source, &destination, &columns)
                .await
                .is_err()
        );
        assert!(!destination.exists());
    }
    Ok(())
}

#[tokio::test]
async fn test_stale_source_and_no_overwrite() -> VortexResult<()> {
    let session = VortexSession::default();
    let dir = tempfile::tempdir()?;
    let source = dir.path().join("source.vortex");
    let destination = dir.path().join("fts");
    write_file(&session, &source, table()?).await?;
    build_index(&session, &source, &destination, &["title".into()]).await?;
    assert!(
        build_index(&session, &source, &destination, &["title".into()])
            .await
            .is_err()
    );
    let index = SearchIndex::open(&destination)?;
    assert_eq!(index.search(&source, "search", 10)?.total_hits, 1);
    // Same size with changed bytes must also invalidate the identity.
    let mut bytes = fs::read(&source)?;
    bytes[4] ^= 1;
    fs::write(&source, bytes)?;
    assert!(index.search(&source, "search", 10).is_err());
    assert!(index.read_rows(&session, &source, &[0]).await.is_err());
    Ok(())
}

#[tokio::test]
async fn test_incomplete_and_incompatible_indexes() -> VortexResult<()> {
    let session = VortexSession::default();
    let dir = tempfile::tempdir()?;
    assert!(SearchIndex::open(dir.path()).is_err());
    let source = dir.path().join("source.vortex");
    let destination = dir.path().join("fts");
    write_file(&session, &source, table()?).await?;
    build_index(&session, &source, &destination, &["title".into()]).await?;
    let path = destination.join("manifest.json");
    let mut metadata: serde_json::Value =
        serde_json::from_slice(&fs::read(&path)?).map_err(|err| vortex_err!(External: err))?;
    for (key, value) in [
        ("format_version", serde_json::json!(999)),
        ("row_count", serde_json::json!(100)),
        ("tokenizer", serde_json::json!("unknown")),
        ("columns", serde_json::json!(["body"])),
    ] {
        let previous = metadata[key].clone();
        metadata[key] = value;
        fs::write(
            &path,
            serde_json::to_vec(&metadata).map_err(|err| vortex_err!(External: err))?,
        )?;
        assert!(SearchIndex::open(&destination).is_err(), "{key}");
        metadata[key] = previous;
    }
    Ok(())
}

#[test]
fn test_parent_null_masks_child_text() -> VortexResult<()> {
    // The current Vortex file writer rejects nullable root structs. Exercise the indexing
    // boundary directly so parent nulls remain safe when reading externally produced files.
    let session = VortexSession::default();
    let batch = StructArray::new(
        FieldNames::from(["text"]),
        vec![VarBinViewArray::from_iter_str(["visible", "secret"]).into_array()],
        2,
        Validity::from_iter([true, false]),
    );
    let (schema, row_field, text_fields) = make_schema(&["text".into()]);
    let index = Index::create_in_ram(schema);
    let mut writer = index
        .writer_with_num_threads(1, 15_000_000)
        .map_err(|err| vortex_err!(External: err))?;
    assert_eq!(
        index_batch(
            &mut writer,
            row_field,
            &text_fields,
            &batch,
            &mut session.create_execution_ctx(),
            0
        )?,
        2
    );
    writer.commit().map_err(|err| vortex_err!(External: err))?;
    let reader: IndexReader = index.reader().map_err(|err| vortex_err!(External: err))?;
    let parser = QueryParser::for_index(&index, text_fields);
    for (query, count) in [("visible", 1), ("secret", 0), ("*", 2)] {
        let query = parser
            .parse_query(query)
            .map_err(|err| vortex_err!(External: err))?;
        assert_eq!(
            reader
                .searcher()
                .search(&query, &Count)
                .map_err(|err| vortex_err!(External: err))?,
            count
        );
    }
    Ok(())
}

#[tokio::test]
async fn test_bm25_and_analyzer_contract() -> VortexResult<()> {
    let session = VortexSession::default();
    let dir = tempfile::tempdir()?;
    let source = dir.path().join("source.vortex");
    let destination = dir.path().join("fts");
    let long_indexed = "a".repeat(39);
    let long_dropped = "b".repeat(40);
    let array = StructArray::from_fields(&[(
        "text",
        VarBinViewArray::from_iter_str([
            "rust rust rust",
            "rust python python",
            "中文检索",
            &long_indexed,
            &long_dropped,
        ])
        .into_array(),
    )])?
    .into_array();
    write_file(&session, &source, array).await?;
    build_index(&session, &source, &destination, &["text".into()]).await?;
    let index = SearchIndex::open(destination)?;
    let hits = index.search(&source, "RUST", 10)?.hits;
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].row_id, 0);
    assert!(hits[0].score > hits[1].score);
    for (query, count) in [
        ("中文检索", 1),
        ("中文", 0),
        (long_indexed.as_str(), 1),
        (long_dropped.as_str(), 0),
    ] {
        assert_eq!(index.search(&source, query, 10)?.total_hits, count);
    }
    Ok(())
}
