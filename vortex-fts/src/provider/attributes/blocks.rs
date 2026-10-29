// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeMap;

use tantivy::DocSet;
use tantivy::Searcher;
use tantivy::TERMINATED;
use tantivy::schema::IndexRecordOption;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex_index::IndexMetadata;
use vortex_index::RowAddress;
use vortex_index::attributes::AttributeBlock;
use vortex_index::attributes::AttributeBlockKind;
use vortex_index::attributes::AttributeBlockPlan;
use vortex_index::attributes::AttributeBlockRequest;
use vortex_index::attributes::AttributePredicate;
use vortex_index::attributes::AttributeValue;
use vortex_index::attributes::FloatValue;

use super::super::FILE_ID;
use super::super::ROW_ID;
use super::AttributeFields;
use super::AttributeType;

fn decode(kind: AttributeType, bytes: &[u8]) -> VortexResult<AttributeValue> {
    if kind == AttributeType::String {
        return Ok(AttributeValue::String(
            std::str::from_utf8(bytes)
                .map_err(|err| vortex_err!(External: err))?
                .into(),
        ));
    }
    let encoded = u64::from_be_bytes(
        bytes
            .try_into()
            .map_err(|_| vortex_err!("Invalid numeric attribute term"))?,
    );
    Ok(match kind {
        AttributeType::Bool if encoded <= 1 => AttributeValue::Bool(encoded != 0),
        AttributeType::Int => AttributeValue::Int(tantivy::u64_to_i64(encoded)),
        AttributeType::UInt => AttributeValue::UInt(encoded),
        AttributeType::Float => {
            AttributeValue::Float(FloatValue::try_from(tantivy::u64_to_f64(encoded))?)
        }
        _ => vortex_bail!("Invalid attribute term type or value"),
    })
}

impl AttributeFields {
    pub(in super::super) fn blocks(
        &self,
        metadata: &IndexMetadata,
        searcher: &Searcher,
        request: &AttributeBlockRequest,
    ) -> VortexResult<AttributeBlockPlan> {
        request.validate(&metadata.snapshot)?;
        if request.covered_files.iter().any(|id| !metadata.covers(*id)) {
            vortex_bail!("Attribute blocks require complete requested file coverage");
        }
        let mut blocks = Vec::new();
        let mut scanned = 0u64;
        let mut emitted = 0u64;
        for (name, policy) in &request.fields {
            let (field, kind) = self.field(name)?;
            if *policy == AttributeBlockKind::NumericRange
                && !matches!(
                    kind,
                    AttributeType::Int | AttributeType::UInt | AttributeType::Float
                )
            {
                vortex_bail!("Numeric blocks require a numeric attribute: {name}");
            }
            let mut groups: BTreeMap<AttributeValue, Vec<RowAddress>> = BTreeMap::new();
            // Read term dictionaries and postings directly. No ranked retrieval,
            // full-document decoding, or separate query for every distinct value.
            for segment in searcher.segment_readers() {
                let files = segment
                    .fast_fields()
                    .u64(FILE_ID)
                    .map_err(|err| vortex_err!(External: err))?;
                let offsets = segment
                    .fast_fields()
                    .u64(ROW_ID)
                    .map_err(|err| vortex_err!(External: err))?;
                let inverted = segment
                    .inverted_index(field)
                    .map_err(|err| vortex_err!(External: err))?;
                let mut terms = inverted
                    .terms()
                    .stream()
                    .map_err(|err| vortex_err!(External: err))?;
                while terms.advance() {
                    let value = decode(kind, terms.key())?;
                    if !groups.contains_key(&value) && groups.len() >= request.max_blocks as usize {
                        vortex_bail!("Attribute term limit exceeded");
                    }
                    let rows = groups.entry(value).or_default();
                    let mut postings = inverted
                        .read_postings_from_terminfo(terms.value(), IndexRecordOption::Basic)
                        .map_err(|err| vortex_err!(External: err))?;
                    while postings.doc() != TERMINATED {
                        let doc = postings.doc();
                        let row = RowAddress {
                            file_id: files
                                .first(doc)
                                .ok_or_else(|| vortex_err!("Missing file ID"))?,
                            row_offset: offsets
                                .first(doc)
                                .ok_or_else(|| vortex_err!("Missing row offset"))?,
                        };
                        metadata.snapshot.validate_row(row)?;
                        if !metadata.covers(row.file_id) || segment.is_deleted(doc) {
                            vortex_bail!("Invalid row in immutable attribute postings");
                        }
                        if request.covered_files.binary_search(&row.file_id).is_ok() {
                            if scanned == request.max_memberships {
                                vortex_bail!("Attribute posting limit exceeded");
                            }
                            scanned += 1;
                            rows.push(row);
                        }
                        postings.advance();
                    }
                }
            }
            groups.retain(|_, rows| !rows.is_empty());
            for rows in groups.values_mut() {
                rows.sort_unstable();
                if rows.windows(2).any(|pair| pair[0] == pair[1]) {
                    vortex_bail!("Duplicate row in attribute postings");
                }
            }
            let groups: Vec<_> = groups.into_iter().collect();
            let mut start = 0;
            while start < groups.len() {
                let mut end = start + 1;
                let mut count = groups[start].1.len();
                if *policy == AttributeBlockKind::NumericRange {
                    while end < groups.len() && count < request.target_rows as usize {
                        count += groups[end].1.len();
                        end += 1;
                    }
                }
                if count > request.max_block_rows as usize {
                    vortex_bail!("Attribute block exceeds row limit; equal values cannot be split");
                }
                if count >= request.min_rows as usize {
                    if blocks.len() >= request.max_blocks as usize
                        || count as u64 > request.max_memberships - emitted
                    {
                        vortex_bail!("Attribute block/membership limit exceeded");
                    }
                    emitted += count as u64;
                    let predicate = match policy {
                        AttributeBlockKind::Equality => {
                            AttributePredicate::Eq(name.clone(), groups[start].0.clone())
                        }
                        AttributeBlockKind::NumericRange => AttributePredicate::Range(
                            name.clone(),
                            groups[start].0.clone(),
                            groups[end - 1].0.clone(),
                        ),
                    };
                    let mut rows = Vec::with_capacity(count);
                    for (_, group) in &groups[start..end] {
                        rows.extend_from_slice(group);
                    }
                    rows.sort_unstable();
                    if rows.windows(2).any(|pair| pair[0] == pair[1]) {
                        vortex_bail!("Scalar attribute has multiple values per physical row");
                    }
                    blocks.push(AttributeBlock { predicate, rows });
                }
                if end == groups.len() {
                    break;
                }
                let mut advanced = 0;
                loop {
                    advanced += groups[start].1.len();
                    start += 1;
                    if *policy == AttributeBlockKind::Equality
                        || start == end
                        || advanced >= request.target_rows as usize / 2
                    {
                        break;
                    }
                }
            }
        }
        Ok(AttributeBlockPlan {
            snapshot: metadata.snapshot.clone(),
            request: request.clone(),
            blocks,
        })
    }
}
