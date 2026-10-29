// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::BinaryHeap;
use std::fs;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::TryStreamExt;
use serde::Deserialize;
use serde::Serialize;
use tantivy::Index as NativeIndex;
use tantivy::IndexReader;
use tantivy::TERMINATED;
use tantivy::TantivyDocument;
use tantivy::query::EnableScoring;
use tantivy::query::Query;
use tantivy::query::QueryParser;
use tantivy::schema::FAST;
use tantivy::schema::Field;
use tantivy::schema::Schema;
use tantivy::schema::TEXT;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::session::VortexSession;
use vortex_index::AttributeIndex;
use vortex_index::FullTextHit;
use vortex_index::FullTextIndex;
use vortex_index::Index;
use vortex_index::IndexBuildRequest;
use vortex_index::IndexBuilder;
use vortex_index::IndexMetadata;
use vortex_index::IndexProvider;
use vortex_index::IndexSource;
use vortex_index::IndexStore;
use vortex_index::LocalArtifactLease;
use vortex_index::RowAddress;
use vortex_index::RowFilter;
use vortex_index::attributes::AttributePredicate;

use crate::schema::validate_columns;
use crate::schema::validate_names;

mod attributes;
use attributes::AttributeFields;
pub use attributes::AttributeType;
pub use attributes::TantivyBuildOptions;

/// Registry identity of the immutable multi-file Tantivy provider.
pub const TANTIVY_ID: &str = "tantivy.static";
const DESCRIPTOR: &str = "tantivy/descriptor.json";
const MAX_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_ROWS: usize = 1_000_000;
const FILE_ID: &str = "_vortex_file_id";
const ROW_ID: &str = "_vortex_row_id";

/// Generic-index provider for a whole snapshot, with a shared BM25 corpus across its files.
/// Synchronous Tantivy and filesystem work must run on an application worker thread.
#[derive(Debug)]
pub struct TantivyProvider {
    session: VortexSession,
    scratch: PathBuf,
}

impl TantivyProvider {
    /// Construct in an existing absolute scratch directory owned by the caller.
    pub fn try_new(session: VortexSession, scratch: PathBuf) -> VortexResult<Self> {
        if !scratch.is_absolute() || !scratch.is_dir() {
            vortex_bail!("Tantivy requires an existing absolute scratch directory");
        }
        Ok(Self { session, scratch })
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Descriptor {
    metadata: IndexMetadata,
    analyzer: String,
    rows: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    options: Option<TantivyBuildOptions>,
}

fn validate(metadata: &IndexMetadata) -> VortexResult<()> {
    metadata.validate_for(&metadata.snapshot)?;
    validate_names(&metadata.fields)?;
    if metadata.backend != TANTIVY_ID
        || !matches!(metadata.backend_version, 1 | 2)
        || metadata.fields.len() > 16
        || metadata.fields.iter().any(|name| name == FILE_ID)
    {
        vortex_bail!("Invalid Tantivy backend, format or text fields");
    }
    Ok(())
}

fn schema(
    fields: &[String],
    options: Option<&TantivyBuildOptions>,
) -> (Schema, Field, Field, Vec<Field>, Option<AttributeFields>) {
    let mut schema = Schema::builder();
    let file = schema.add_u64_field(FILE_ID, FAST);
    let row = schema.add_u64_field(ROW_ID, FAST);
    let text = fields
        .iter()
        .map(|name| schema.add_text_field(name, TEXT))
        .collect();
    let attributes = options.map(|options| AttributeFields::add(&mut schema, options));
    (schema.build(), file, row, text, attributes)
}

#[async_trait]
impl IndexProvider for TantivyProvider {
    fn id(&self) -> &str {
        TANTIVY_ID
    }
    fn supports_version(&self, version: u32) -> bool {
        matches!(version, 1 | 2)
    }
    fn builder(&self) -> Option<&dyn IndexBuilder> {
        Some(self)
    }
    fn local_materialization_limit(&self) -> Option<u64> {
        Some(MAX_BYTES)
    }

    async fn open(
        &self,
        metadata: &IndexMetadata,
        store: Arc<dyn IndexStore>,
    ) -> VortexResult<Arc<dyn Index>> {
        validate(metadata)?;
        if !metadata
            .artifacts
            .iter()
            .any(|artifact| artifact.path == DESCRIPTOR)
            || metadata.artifacts.iter().any(|artifact| {
                artifact.path != DESCRIPTOR && !artifact.path.starts_with("tantivy/index/")
            })
        {
            vortex_bail!("Invalid Tantivy artifact inventory");
        }
        let lease = store
            .as_local_files()
            .ok_or_else(|| vortex_err!("Tantivy requires local artifacts"))?
            .materialize(&metadata.artifacts, &self.scratch, MAX_BYTES)?;
        let descriptor: Descriptor =
            serde_json::from_slice(&fs::read(lease.path().join(DESCRIPTOR))?)
                .map_err(|err| vortex_err!(External: err))?;
        let mut expected = metadata.clone();
        expected.artifacts.clear();
        if descriptor.metadata != expected
            || descriptor.analyzer != "tantivy-0.25-default"
            || descriptor.rows > MAX_ROWS as u64
        {
            vortex_bail!("Tantivy descriptor does not match its generation");
        }
        let index = NativeIndex::open_in_dir(lease.path().join("tantivy/index"))
            .map_err(|err| vortex_err!(External: err))?;
        let options = descriptor.options;
        if (metadata.backend_version == 2) != options.is_some() {
            vortex_bail!("Tantivy format and field roles disagree");
        }
        if let Some(options) = &options {
            options.validate(metadata)?;
        }
        let text_names = options
            .as_ref()
            .map_or(&metadata.fields, |options| &options.text_fields)
            .clone();
        let (expected_schema, _, _, fields, attributes) = schema(&text_names, options.as_ref());
        if index.schema() != expected_schema {
            vortex_bail!("Tantivy schema mismatch");
        }
        let reader: IndexReader = index.reader().map_err(|err| vortex_err!(External: err))?;
        if reader.searcher().num_docs() != descriptor.rows {
            vortex_bail!("Tantivy row count mismatch");
        }
        Ok(Arc::new(Opened {
            index,
            reader,
            fields,
            text_names,
            attributes,
            metadata: metadata.clone(),
            _lease: lease,
        }))
    }
}

#[async_trait]
impl IndexBuilder for TantivyProvider {
    async fn build(
        &self,
        request: IndexBuildRequest,
        source: Arc<dyn IndexSource>,
        store: Arc<dyn IndexStore>,
    ) -> VortexResult<IndexMetadata> {
        let mut metadata = request.metadata;
        validate(&metadata)?;
        metadata.validate_for(source.snapshot())?;
        if !metadata.artifacts.is_empty() || request.backend_options.len() > 64 * 1024 {
            vortex_bail!("Tantivy requires a new artifact inventory and bounded build options");
        }
        let options = if metadata.backend_version == 1 {
            if !request.backend_options.is_empty() {
                vortex_bail!("Tantivy v1 requires empty build options");
            }
            None
        } else {
            let options: TantivyBuildOptions = serde_json::from_slice(&request.backend_options)
                .map_err(|err| vortex_err!(External: err))?;
            options.validate(&metadata)?;
            Some(options)
        };
        let text_names = options
            .as_ref()
            .map_or(&metadata.fields, |options| &options.text_fields);
        let local = store
            .as_local_files()
            .ok_or_else(|| vortex_err!("Tantivy requires local artifacts"))?;
        let temporary = tempfile::tempdir_in(&self.scratch)?;
        let (schema, file_field, row_field, text_fields, attributes) =
            schema(text_names, options.as_ref());
        let index = NativeIndex::create_in_dir(temporary.path(), schema)
            .map_err(|err| vortex_err!(External: err))?;
        let mut writer = index
            .writer_with_num_threads::<TantivyDocument>(1, 64_000_000)
            .map_err(|err| vortex_err!(External: err))?;
        let mut stream = source.scan(&metadata.covered_files, &metadata.fields)?;
        let mut ctx = self.session.create_execution_ctx();
        let mut seen = BTreeSet::new();
        while let Some(batch) = stream.try_next().await? {
            if batch.rows.len() != batch.data.len() {
                vortex_bail!("Source row mapping length mismatch");
            }
            if !text_names.is_empty() {
                validate_columns(batch.data.dtype(), text_names)?;
            }
            let data = batch.data.execute::<StructArray>(&mut ctx)?;
            let valid = data.validity()?.execute_mask(data.len(), &mut ctx)?;
            let mut columns = Vec::new();
            for name in text_names {
                let array = data
                    .unmasked_field_by_name(name)?
                    .clone()
                    .execute::<VarBinViewArray>(&mut ctx)?;
                let mask = array.validity()?.execute_mask(array.len(), &mut ctx)?;
                columns.push((array, mask));
            }
            let attribute_columns = attributes
                .as_ref()
                .map(|attributes| attributes.decode(&data, &mut ctx))
                .transpose()?;
            for ((offset, address), valid) in batch.rows.into_iter().enumerate().zip(valid.iter()) {
                metadata.snapshot.validate_row(address)?;
                if !metadata.covers(address.file_id)
                    || !seen.insert(address)
                    || seen.len() > MAX_ROWS
                {
                    vortex_bail!("Invalid, duplicate or excessive source rows");
                }
                let mut doc = TantivyDocument::default();
                doc.add_u64(file_field, address.file_id);
                doc.add_u64(row_field, address.row_offset);
                if valid {
                    for (field, (array, mask)) in text_fields.iter().zip(&columns) {
                        if mask.value(offset) {
                            let bytes = array.bytes_at(offset);
                            doc.add_text(
                                *field,
                                std::str::from_utf8(&bytes)
                                    .map_err(|err| vortex_err!(External: err))?,
                            );
                        }
                    }
                }
                if let (Some(attributes), Some(columns)) = (&attributes, &attribute_columns) {
                    attributes.add_values(&mut doc, columns, offset);
                }
                writer
                    .add_document(doc)
                    .map_err(|err| vortex_err!(External: err))?;
            }
        }
        writer.commit().map_err(|err| vortex_err!(External: err))?;
        writer
            .wait_merging_threads()
            .map_err(|err| vortex_err!(External: err))?;
        drop(index);
        let descriptor = Descriptor {
            metadata: metadata.clone(),
            analyzer: "tantivy-0.25-default".into(),
            rows: seen.len() as u64,
            options,
        };
        let mut files = fs::read_dir(temporary.path())?.collect::<Result<Vec<_>, _>>()?;
        files.sort_by_key(|entry| entry.file_name());
        let mut bytes = 0u64;
        for file in files {
            let name = file
                .file_name()
                .into_string()
                .map_err(|_| vortex_err!("Invalid index filename"))?;
            // Tantivy lock files are process state and must never be part of a sealed generation.
            if name.starts_with('.') {
                continue;
            }
            bytes = bytes
                .checked_add(file.metadata()?.len())
                .ok_or_else(|| vortex_err!("Index byte overflow"))?;
            if bytes > MAX_BYTES {
                vortex_bail!("Tantivy exceeds artifact budget");
            }
            metadata
                .artifacts
                .push(local.import_file(&format!("tantivy/index/{name}"), &file.path())?);
        }
        metadata.artifacts.push(
            store
                .write(
                    DESCRIPTOR,
                    Bytes::from(
                        serde_json::to_vec(&descriptor)
                            .map_err(|err| vortex_err!(External: err))?,
                    ),
                )
                .await?,
        );
        Ok(metadata)
    }
}

struct Opened {
    index: NativeIndex,
    reader: IndexReader,
    fields: Vec<Field>,
    text_names: Vec<String>,
    attributes: Option<AttributeFields>,
    metadata: IndexMetadata,
    _lease: Box<dyn LocalArtifactLease>,
}

impl std::fmt::Debug for Opened {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TantivyIndex").finish_non_exhaustive()
    }
}
impl Index for Opened {
    fn metadata(&self) -> &IndexMetadata {
        &self.metadata
    }
    fn as_full_text(&self) -> Option<&dyn FullTextIndex> {
        (!self.fields.is_empty()).then_some(self)
    }
    fn as_attributes(&self) -> Option<&dyn AttributeIndex> {
        self.attributes
            .as_ref()
            .map(|_| self as &dyn AttributeIndex)
    }
}

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct QueryOptions {
    field_boosts: BTreeMap<String, f32>,
}

#[derive(Clone)]
struct Ranked(FullTextHit);
impl PartialEq for Ranked {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}
impl Eq for Ranked {}
impl PartialOrd for Ranked {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Ranked {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .score
            .total_cmp(&other.0.score)
            .then_with(|| other.0.row.cmp(&self.0.row))
    }
}

#[async_trait]
impl FullTextIndex for Opened {
    async fn filter_text(&self, query: &str, visibility: &RowFilter) -> VortexResult<RowFilter> {
        if query.trim().is_empty() || query.len() > 64 * 1024 || self.fields.is_empty() {
            vortex_bail!("Invalid full-text prefilter query");
        }
        let query = QueryParser::for_index(&self.index, self.fields.clone())
            .parse_query(query)
            .map_err(|err| vortex_err!(External: err))?;
        self.collect_filter(query.as_ref(), visibility)
    }
    async fn search_text(
        &self,
        query: &str,
        k: NonZeroUsize,
        options: &Bytes,
        filter: &RowFilter,
    ) -> VortexResult<Vec<FullTextHit>> {
        if filter.snapshot() != &self.metadata.snapshot
            || k.get() > 10_000
            || query.trim().is_empty()
            || query.len() > 64 * 1024
            || options.len() > 64 * 1024
        {
            vortex_bail!("Invalid full-text snapshot, query or result limit");
        }
        let options: QueryOptions = if options.is_empty() {
            QueryOptions::default()
        } else {
            serde_json::from_slice(options).map_err(|err| vortex_err!(External: err))?
        };
        let mut parser = QueryParser::for_index(&self.index, self.fields.clone());
        for (name, boost) in options.field_boosts {
            let position = self
                .text_names
                .iter()
                .position(|field| field == &name)
                .ok_or_else(|| vortex_err!("Unknown boosted field: {name}"))?;
            if !boost.is_finite() || boost <= 0.0 {
                vortex_bail!("Field boosts must be finite and positive");
            }
            parser.set_field_boost(self.fields[position], boost);
        }
        let query = parser
            .parse_query(query)
            .map_err(|err| vortex_err!(External: err))?;
        let searcher = self.reader.searcher();
        let weight = query
            .weight(EnableScoring::enabled_from_searcher(&searcher))
            .map_err(|err| vortex_err!(External: err))?;
        let mut heap: BinaryHeap<Reverse<Ranked>> = BinaryHeap::new();
        for segment in searcher.segment_readers() {
            let files = segment
                .fast_fields()
                .u64(FILE_ID)
                .map_err(|err| vortex_err!(External: err))?;
            let offsets = segment
                .fast_fields()
                .u64(ROW_ID)
                .map_err(|err| vortex_err!(External: err))?;
            let mut scorer = weight
                .scorer(segment, 1.0)
                .map_err(|err| vortex_err!(External: err))?;
            while scorer.doc() != TERMINATED {
                let doc = scorer.doc();
                let row = RowAddress {
                    file_id: files
                        .first(doc)
                        .ok_or_else(|| vortex_err!("Missing file ID"))?,
                    row_offset: offsets
                        .first(doc)
                        .ok_or_else(|| vortex_err!("Missing row offset"))?,
                };
                self.metadata.snapshot.validate_row(row)?;
                if !self.metadata.covers(row.file_id) {
                    vortex_bail!("Index row outside coverage");
                }
                if segment.is_deleted(doc) {
                    vortex_bail!("Immutable Tantivy generation contains deleted documents");
                }
                if filter.includes(row) {
                    let score = scorer.score();
                    if !score.is_finite() {
                        vortex_bail!("Non-finite relevance score");
                    }
                    let candidate = Ranked(FullTextHit { row, score });
                    if heap.len() < k.get() {
                        heap.push(Reverse(candidate));
                    } else if heap.peek().is_some_and(|worst| candidate > worst.0) {
                        heap.pop();
                        heap.push(Reverse(candidate));
                    }
                }
                scorer.advance();
            }
        }
        let mut hits = heap.into_iter().map(|hit| hit.0).collect::<Vec<_>>();
        hits.sort_by(|left, right| right.cmp(left));
        Ok(hits.into_iter().map(|hit| hit.0).collect())
    }
}

#[async_trait]
impl AttributeIndex for Opened {
    async fn blocks(
        &self,
        request: &vortex_index::attributes::AttributeBlockRequest,
    ) -> VortexResult<vortex_index::attributes::AttributeBlockPlan> {
        self.attributes
            .as_ref()
            .ok_or_else(|| vortex_err!("Generation has no attribute fields"))?
            .blocks(&self.metadata, &self.reader.searcher(), request)
    }

    async fn filter(
        &self,
        predicate: &AttributePredicate,
        visibility: &RowFilter,
    ) -> VortexResult<RowFilter> {
        let attributes = self.attributes.as_ref().ok_or_else(|| {
            vortex_err!("Generation has no attribute fields; rebuild with Tantivy v2")
        })?;
        let query = attributes.query(predicate, 0, &mut 4096)?;
        self.collect_filter(query.as_ref(), visibility)
    }
}

impl Opened {
    fn collect_filter(&self, query: &dyn Query, visibility: &RowFilter) -> VortexResult<RowFilter> {
        if visibility.snapshot() != &self.metadata.snapshot {
            vortex_bail!("Tantivy prefilter snapshot mismatch");
        }
        if self.metadata.uncovered_files().next().is_some() {
            vortex_bail!("Tantivy prefilters require complete snapshot file coverage");
        }
        let searcher = self.reader.searcher();
        // Enumerate every match with scoring disabled: BM25 top-k cannot be an ANN prefilter.
        let weight = query
            .weight(EnableScoring::disabled_from_searcher(&searcher))
            .map_err(|err| vortex_err!(External: err))?;
        let mut allowed = BTreeSet::new();
        for segment in searcher.segment_readers() {
            let files = segment
                .fast_fields()
                .u64(FILE_ID)
                .map_err(|err| vortex_err!(External: err))?;
            let offsets = segment
                .fast_fields()
                .u64(ROW_ID)
                .map_err(|err| vortex_err!(External: err))?;
            let mut scorer = weight
                .scorer(segment, 1.0)
                .map_err(|err| vortex_err!(External: err))?;
            while scorer.doc() != TERMINATED {
                let doc = scorer.doc();
                let row = RowAddress {
                    file_id: files
                        .first(doc)
                        .ok_or_else(|| vortex_err!("Missing file ID"))?,
                    row_offset: offsets
                        .first(doc)
                        .ok_or_else(|| vortex_err!("Missing row offset"))?,
                };
                self.metadata.snapshot.validate_row(row)?;
                if !self.metadata.covers(row.file_id) || segment.is_deleted(doc) {
                    vortex_bail!("Invalid row in immutable Tantivy generation");
                }
                if visibility.includes(row) {
                    allowed.insert(row);
                }
                scorer.advance();
            }
        }
        RowFilter::try_new(
            self.metadata.snapshot.clone(),
            Some(allowed),
            BTreeSet::new(),
        )
    }
}
