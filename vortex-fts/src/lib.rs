// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![deny(missing_docs)]
//! Persistent Tantivy sidecar indexes for immutable local Vortex files.
//!
//! Build with [`build_index`], reopen with [`SearchIndex::open`], and query with
//! [`SearchIndex::search`]. Use [`SearchIndex::read_rows`] to fetch matching Vortex rows.
//! Source files must not be modified during any operation. Search and row retrieval check a full
//! SHA-256 digest, so replacing a file with different contents requires rebuilding its index.

mod manifest;
mod provider;
pub use provider::AttributeType;
pub use provider::TANTIVY_ID;
pub use provider::TantivyBuildOptions;
pub use provider::TantivyProvider;
mod schema;
#[cfg(test)]
mod tests;

use std::fs;
use std::path::Path;

use futures::TryStreamExt;
use serde::Deserialize;
use serde::Serialize;
use tantivy::Index;
use tantivy::IndexReader;
use tantivy::IndexWriter;
use tantivy::TantivyDocument;
use tantivy::collector::Count;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::Field;
use tantivy::schema::Value;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::array::expr::root;
use vortex::array::expr::select;
use vortex::array::stream::ArrayStreamExt;
use vortex::buffer::Buffer;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::file::OpenOptionsSessionExt;
use vortex::layout::scan::split_by::SplitBy;
use vortex::scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex::session::VortexSession;

pub use crate::manifest::IndexMetadata;
use crate::manifest::SourceIdentity;
use crate::schema::make_schema;
use crate::schema::validate_columns;

/// Build a complete sidecar in a new directory. Existing directories are never overwritten.
///
/// Columns must be unique, top-level UTF-8 fields. One document is created per physical row,
/// including null rows. Text is not duplicated in the stored document. Indexing uses a single
/// Tantivy worker with a 64 MB budget, plus the current decoded Vortex scan batch.
///
/// `manifest.json` is published last. An interrupted/failed build may leave an incomplete
/// directory, which readers reject; remove that directory or use a new path before retrying.
/// This async function also performs blocking hashing/indexing; run it on a dedicated worker
/// when embedding in a latency-sensitive async service.
pub async fn build_index(
    session: &VortexSession,
    source: impl AsRef<Path>,
    destination: impl AsRef<Path>,
    columns: &[String],
) -> VortexResult<IndexMetadata> {
    let source = source.as_ref();
    let destination = destination.as_ref();
    let identity = SourceIdentity::read(source)?;
    let file = session.open_options().open_path(source).await?;
    validate_columns(file.dtype(), columns)?;
    let (schema, row_field, text_fields) = make_schema(columns);
    fs::create_dir(destination)?;
    let staging = tempfile::tempdir_in(destination)?;
    let index =
        Index::create_in_dir(staging.path(), schema).map_err(|err| vortex_err!(External: err))?;
    let mut writer = index
        .writer_with_num_threads::<TantivyDocument>(1, 64_000_000)
        .map_err(|err| vortex_err!(External: err))?;
    let projection = select(
        columns.iter().map(String::as_str).collect::<Vec<_>>(),
        root(),
    )
    .bind(file.dtype())?;
    let mut stream = file
        .scan()?
        .with_projection(projection)
        .with_ordered(true)
        .with_concurrency(1)
        .with_split_by(SplitBy::RowCount(8192))
        .into_array_stream()?;
    let mut ctx = session.create_execution_ctx();
    let mut row_id = 0u64;
    while let Some(batch) = stream.try_next().await? {
        let batch = batch.execute::<StructArray>(&mut ctx)?;
        row_id = index_batch(
            &mut writer,
            row_field,
            &text_fields,
            &batch,
            &mut ctx,
            row_id,
        )?;
    }
    if row_id != file.row_count() {
        vortex_bail!("scan row count differs from the Vortex footer");
    }
    writer.commit().map_err(|err| vortex_err!(External: err))?;
    writer
        .wait_merging_threads()
        .map_err(|err| vortex_err!(External: err))?;
    identity.verify(source)?;
    drop(index);
    fs::rename(staging.path(), destination.join("index"))?;
    let metadata = IndexMetadata::new(identity, row_id, columns.to_vec());
    metadata.publish(destination)?;
    Ok(metadata)
}

// Input is the validated projection: fields have the same order as the Tantivy text fields.
fn index_batch(
    writer: &mut IndexWriter<TantivyDocument>,
    row_field: Field,
    text_fields: &[Field],
    batch: &StructArray,
    ctx: &mut ExecutionCtx,
    mut row_id: u64,
) -> VortexResult<u64> {
    let row_validity = batch.validity()?.execute_mask(batch.len(), ctx)?;
    let mut decoded = Vec::with_capacity(text_fields.len());
    for i in 0..text_fields.len() {
        let values = batch
            .unmasked_field(i)
            .clone()
            .execute::<VarBinViewArray>(ctx)?;
        let validity = values.validity()?.execute_mask(values.len(), ctx)?;
        decoded.push((values, validity));
    }
    for (offset, row_valid) in row_validity.iter().enumerate() {
        let mut doc = TantivyDocument::default();
        doc.add_u64(row_field, row_id);
        if row_valid {
            for (field, (values, validity)) in text_fields.iter().zip(&decoded) {
                if validity.value(offset) {
                    let bytes = values.bytes_at(offset);
                    let text =
                        std::str::from_utf8(&bytes).map_err(|err| vortex_err!(External: err))?;
                    doc.add_text(*field, text);
                }
            }
        }
        writer
            .add_document(doc)
            .map_err(|err| vortex_err!(External: err))?;
        row_id = row_id
            .checked_add(1)
            .ok_or_else(|| vortex_err!("row ID overflow"))?;
    }
    Ok(row_id)
}

/// One match in BM25 relevance order. Row IDs are zero-based physical Vortex row positions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    /// Physical row position in the immutable source file.
    pub row_id: u64,
    /// Tantivy BM25 score; scores are comparable within the same query/index only.
    pub score: f32,
}

/// Total matching documents and the requested highest-scoring hits.
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResults {
    /// Total matches, independent of the requested hit limit.
    pub total_hits: usize,
    /// Hits sorted by decreasing relevance. Tie ordering is unspecified.
    pub hits: Vec<SearchHit>,
}

/// A validated, immutable Tantivy reader. Reuse it for multiple searches.
pub struct SearchIndex {
    index: Index,
    reader: IndexReader,
    metadata: IndexMetadata,
    row_field: Field,
    text_fields: Vec<Field>,
}

impl SearchIndex {
    /// Open a completed sidecar and validate its version, schema, and document count.
    pub fn open(directory: impl AsRef<Path>) -> VortexResult<Self> {
        let directory = directory.as_ref();
        let metadata = IndexMetadata::read(directory)?;
        let (schema, row_field, text_fields) = make_schema(&metadata.columns);
        let index = Index::open_in_dir(directory.join("index"))
            .map_err(|err| vortex_err!(External: err))?;
        if index.schema() != schema {
            vortex_bail!("sidecar schema does not match manifest; rebuild the index");
        }
        let reader: IndexReader = index.reader().map_err(|err| vortex_err!(External: err))?;
        if reader.searcher().num_docs() != metadata.row_count {
            vortex_bail!("sidecar document count does not match manifest; rebuild the index");
        }
        Ok(Self {
            index,
            reader,
            metadata,
            row_field,
            text_fields,
        })
    }

    /// Metadata describing the indexed columns, row count, tokenizer, and source digest.
    pub fn metadata(&self) -> &IndexMetadata {
        &self.metadata
    }

    /// Search using Tantivy query syntax and BM25. The default operator is OR.
    ///
    /// A zero limit returns only the total count. Invalid queries return an error, never a
    /// partial query. Every call hashes the source, requiring O(source bytes) I/O; keep source
    /// and index immutable for the duration of the call. The maximum limit is 10,000.
    pub fn search(
        &self,
        source: impl AsRef<Path>,
        query: &str,
        limit: usize,
    ) -> VortexResult<SearchResults> {
        if limit > 10_000 {
            vortex_bail!("search limit must be at most 10000");
        }
        if query.trim().is_empty() {
            vortex_bail!("query must not be empty; use '*' to match all rows");
        }
        self.metadata.source.verify(source.as_ref())?;
        let parser = QueryParser::for_index(&self.index, self.text_fields.clone());
        let query = parser
            .parse_query(query)
            .map_err(|err| vortex_err!(External: err))?;
        let searcher = self.reader.searcher();
        if limit == 0 {
            let total_hits = searcher
                .search(&query, &Count)
                .map_err(|err| vortex_err!(External: err))?;
            return Ok(SearchResults {
                total_hits,
                hits: Vec::new(),
            });
        }
        let (total_hits, docs) = searcher
            .search(&query, &(Count, TopDocs::with_limit(limit)))
            .map_err(|err| vortex_err!(External: err))?;
        let hits = docs
            .into_iter()
            .map(|(score, address)| {
                let doc: TantivyDocument = searcher
                    .doc(address)
                    .map_err(|err| vortex_err!(External: err))?;
                let row_id = doc
                    .get_first(self.row_field)
                    .and_then(|value| value.as_u64())
                    .ok_or_else(|| vortex_err!("index document is missing its row ID"))?;
                if row_id >= self.metadata.row_count {
                    vortex_bail!("index document has an out-of-range row ID");
                }
                Ok(SearchHit { row_id, score })
            })
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(SearchResults { total_hits, hits })
    }

    /// Read selected physical rows after validating the source digest.
    ///
    /// IDs must be strictly increasing, unique, and in range. To retrieve search results,
    /// sort their row IDs first; output follows file order, not relevance order. An empty
    /// selection returns an empty array with the source schema.
    pub async fn read_rows(
        &self,
        session: &VortexSession,
        source: impl AsRef<Path>,
        row_ids: &[u64],
    ) -> VortexResult<ArrayRef> {
        let source = source.as_ref();
        let indices = StrictSortedBuffer::try_new(Buffer::from(row_ids.to_vec()))?;
        if row_ids
            .last()
            .is_some_and(|id| *id >= self.metadata.row_count)
        {
            vortex_bail!("row ID is outside the indexed file");
        }
        self.metadata.source.verify(source)?;
        let file = session.open_options().open_path(source).await?;
        let rows = file
            .scan()?
            .with_row_indices(indices)
            .with_ordered(true)
            .into_array_stream()?
            .read_all()
            .await?;
        self.metadata.source.verify(source)?;
        Ok(rows)
    }
}
