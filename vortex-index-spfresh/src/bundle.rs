// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::Path;

use bytes::Bytes;
use serde::Deserialize;
use serde::Serialize;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_index::IndexMetadata;
use vortex_index::IndexStore;
use vortex_index::RowAddress;
use vortex_index::Snapshot;

use crate::SPFRESH_FORMAT_VERSION;
use crate::SPFRESH_ID;
use crate::SPFRESH_REVISION;

pub(crate) const DESCRIPTOR: &str = "spfresh/bundle.json";
pub(crate) const ROWS: &str = "spfresh/rows.bin";
pub(crate) const NATIVE_FILES: [&str; 6] = [
    "vectors.bin",
    "tree.bin",
    "graph.bin",
    "deletes.bin",
    "head_ids.bin",
    "postings.bin",
];
pub(crate) const ROW_MAGIC: &[u8; 8] = b"VXSFROW1";

/// Native binary shape asserted by the producer of a frozen bundle.
///
/// Only unquantized Float32 squared-L2, BKT heads, excluded heads, a single static
/// posting file and no compression, rearrangement, delta encoding or deletes are
/// supported. The native revision must be exactly [`SPFRESH_REVISION`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpFreshBundle {
    /// Number of Float32 components in each vector (1..=4096).
    pub dimension: u32,
    /// Dense native IDs are exactly 0..rows; deleted or missing IDs are unsupported.
    pub rows: u32,
    /// Page limit used by the producer (1..=4096), in 4096-byte pages.
    pub posting_page_limit: u32,
}

impl SpFreshBundle {
    pub(crate) fn validate(self) -> VortexResult<()> {
        if !(1..=4096).contains(&self.dimension)
            || !(1..=i32::MAX as u32).contains(&self.rows)
            || !(1..=4096).contains(&self.posting_page_limit)
        {
            vortex_bail!("Unsupported SPFresh bundle dimensions, row count or page limit");
        }
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Descriptor {
    pub format_version: u32,
    pub revision: String,
    pub value_type: String,
    pub metric: String,
    pub bundle: SpFreshBundle,
    pub snapshot: Snapshot,
    pub fields: Vec<String>,
    pub covered_files: Vec<u64>,
}

pub(crate) fn validate_metadata(metadata: &IndexMetadata) -> VortexResult<()> {
    metadata.validate_for(&metadata.snapshot)?;
    if metadata.backend != SPFRESH_ID
        || metadata.backend_version != SPFRESH_FORMAT_VERSION
        || metadata.fields.len() != 1
    {
        vortex_bail!("SPFresh requires its supported backend version and exactly one vector field");
    }
    Ok(())
}

pub(crate) fn validate_rows(
    metadata: &IndexMetadata,
    bundle: SpFreshBundle,
    rows: &[RowAddress],
) -> VortexResult<()> {
    bundle.validate()?;
    let covered = metadata
        .snapshot
        .files
        .iter()
        .filter(|file| metadata.covers(file.id));
    let count = covered.clone().try_fold(0u64, |sum, file| {
        sum.checked_add(file.row_count)
            .ok_or_else(|| vortex_err!("Coverage row count overflow"))
    })?;
    if count != u64::from(bundle.rows) || rows.len() as u64 != count {
        vortex_bail!("SPFresh mapping must cover every physical row in exactly the covered files");
    }
    let mut sorted = rows.to_vec();
    sorted.sort_unstable();
    let expected = covered.flat_map(|file| {
        (0..file.row_count).map(move |row_offset| RowAddress {
            file_id: file.id,
            row_offset,
        })
    });
    if !sorted.into_iter().eq(expected) {
        vortex_bail!("SPFresh row mapping has duplicates, missing rows or uncovered addresses");
    }
    Ok(())
}

/// Import closed native files and persist a dense native-ID-to-row mapping.
///
/// `source` must be an absolute, symlink-free directory containing the six named
/// binaries described in the crate README. No INI, external paths or environment
/// configuration are imported. The caller attests that this trusted bundle was
/// produced using the pinned build, with the stated schema and native ID order.
///
/// Requires empty artifact metadata and a private store with local-file support.
/// Validates full physical coverage before writing. The returned metadata is NOT
/// sealed or published: the owner seals it, then opens with the provider to verify
/// native structure before catalog publication. Failures can leave private artifacts.
/// All IO is blocking, even though this function is async. Mapping memory is O(rows).
pub async fn import_bundle(
    mut metadata: IndexMetadata,
    store: &dyn IndexStore,
    source: &Path,
    bundle: SpFreshBundle,
    rows: &[RowAddress],
) -> VortexResult<IndexMetadata> {
    validate_metadata(&metadata)?;
    if !metadata.artifacts.is_empty() || !source.is_absolute() {
        vortex_bail!("SPFresh import requires empty artifacts and an absolute source directory");
    }
    validate_rows(&metadata, bundle, rows)?;
    let local = store
        .as_local_files()
        .ok_or_else(|| vortex_err!("SPFresh requires local-file support"))?;
    let descriptor = Descriptor {
        format_version: SPFRESH_FORMAT_VERSION,
        revision: SPFRESH_REVISION.into(),
        value_type: "float32".into(),
        metric: "squared_l2".into(),
        bundle,
        snapshot: metadata.snapshot.clone(),
        fields: metadata.fields.clone(),
        covered_files: metadata.covered_files.clone(),
    };
    let json =
        serde_json::to_vec(&descriptor).map_err(|err| vortex_err!("SPFresh descriptor: {err}"))?;
    if json.len() > 1024 * 1024 {
        vortex_bail!("SPFresh descriptor exceeds 1 MiB");
    }
    let capacity = rows
        .len()
        .checked_mul(16)
        .and_then(|len| len.checked_add(16))
        .ok_or_else(|| vortex_err!("SPFresh mapping size overflow"))?;
    let mut mapping = Vec::with_capacity(capacity);
    mapping.extend_from_slice(ROW_MAGIC);
    mapping.extend_from_slice(&u64::from(bundle.rows).to_le_bytes());
    for row in rows {
        mapping.extend_from_slice(&row.file_id.to_le_bytes());
        mapping.extend_from_slice(&row.row_offset.to_le_bytes());
    }
    for name in NATIVE_FILES {
        metadata
            .artifacts
            .push(local.import_file(&format!("spfresh/native/{name}"), &source.join(name))?);
    }
    metadata
        .artifacts
        .push(store.write(DESCRIPTOR, Bytes::from(json)).await?);
    metadata
        .artifacts
        .push(store.write(ROWS, Bytes::from(mapping)).await?);
    Ok(metadata)
}
