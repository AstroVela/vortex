// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Runnable end-to-end example; all files live in an automatically cleaned temporary directory.

use std::fs;

use vortex::VortexSessionDefault;
use vortex::array::IntoArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::error::VortexResult;
use vortex::file::WriteOptionsSessionExt;
use vortex::session::VortexSession;
use vortex_fts::SearchIndex;
use vortex_fts::build_index;

#[tokio::main]
async fn main() -> VortexResult<()> {
    let session = VortexSession::default();
    let temporary = tempfile::tempdir()?;
    let source = temporary.path().join("articles.vortex");
    let destination = temporary.path().join("articles.fts");
    let articles = StructArray::from_fields(&[
        (
            "title",
            VarBinViewArray::from_iter_str(["Vortex", "Tantivy"]).into_array(),
        ),
        (
            "body",
            VarBinViewArray::from_iter_str(["compressed columnar data", "fast full text search"])
                .into_array(),
        ),
    ])?;
    let mut bytes = Vec::new();
    session
        .write_options()
        .write(&mut bytes, articles.into_array().to_array_stream())
        .await?;
    fs::write(&source, bytes)?;
    build_index(
        &session,
        &source,
        &destination,
        &["title".into(), "body".into()],
    )
    .await?;
    let index = SearchIndex::open(&destination)?;
    let result = index.search(&source, "\"full text\"", 10)?;
    assert_eq!(result.total_hits, 1);
    assert_eq!(result.hits[0].row_id, 1);
    let mut row_ids = result.hits.iter().map(|hit| hit.row_id).collect::<Vec<_>>();
    row_ids.sort_unstable();
    let rows = index.read_rows(&session, &source, &row_ids).await?;
    assert_eq!(rows.len(), 1);
    println!("Matched {} row(s)", result.total_hits);
    for hit in result.hits {
        println!("row_id={}, score={}", hit.row_id, hit.score);
    }
    Ok(())
}
