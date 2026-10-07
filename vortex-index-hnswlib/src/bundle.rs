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

use crate::HNSWLIB_FORMAT_VERSION;
use crate::HNSWLIB_ID;
use crate::HNSWLIB_REVISION;

pub(crate) const DESCRIPTOR: &str = "hnswlib/bundle.json";
pub(crate) const ROWS: &str = "hnswlib/rows.bin";
pub(crate) const INDEX: &str = "hnswlib/index.bin";
pub(crate) const ROW_MAGIC: &[u8; 8] = b"VXHNROW1";

/// Shape of a closed, unquantized native index with dense external labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HnswBundle {
    /// Number of Float32 components, 1..=4096.
    pub dimension: u32,
    /// Dense external labels are exactly 0..rows, without deletions.
    pub rows: u32,
    /// Graph degree parameter, 2..=64 (level zero uses twice this capacity).
    pub m: u32,
    /// Construction search width, m..=4096.
    pub ef_construction: u32,
}

impl HnswBundle {
    pub(crate) fn validate(self) -> VortexResult<()> {
        if !(1..=4096).contains(&self.dimension)
            || !(1..=i32::MAX as u32).contains(&self.rows)
            || !(2..=64).contains(&self.m)
            || !(self.m..=4096).contains(&self.ef_construction)
        {
            vortex_bail!("Invalid hnswlib bundle shape or construction parameters");
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
    pub bundle: HnswBundle,
    pub snapshot: Snapshot,
    pub fields: Vec<String>,
    pub covered_files: Vec<u64>,
}

pub(crate) fn validate_metadata(metadata: &IndexMetadata) -> VortexResult<()> {
    metadata.validate_for(&metadata.snapshot)?;
    if metadata.backend != HNSWLIB_ID
        || metadata.backend_version != HNSWLIB_FORMAT_VERSION
        || metadata.fields.len() != 1
    {
        vortex_bail!("hnswlib requires its supported backend version and one vector field");
    }
    Ok(())
}

pub(crate) fn validate_rows(
    metadata: &IndexMetadata,
    bundle: HnswBundle,
    rows: &[RowAddress],
) -> VortexResult<()> {
    bundle.validate()?;
    let covered = metadata
        .snapshot
        .files
        .iter()
        .filter(|file| metadata.covers(file.id));
    let count = covered
        .clone()
        .try_fold(0u64, |total, file| total.checked_add(file.row_count))
        .ok_or_else(|| vortex_err!("hnswlib coverage count overflow"))?;
    if count != u64::from(bundle.rows) || rows.len() as u64 != count {
        vortex_bail!("hnswlib mapping must cover every physical row in the covered files");
    }
    let mut sorted = rows.to_vec();
    sorted.sort_unstable();
    if !sorted.into_iter().eq(covered.flat_map(|file| {
        (0..file.row_count).map(move |row_offset| RowAddress {
            file_id: file.id,
            row_offset,
        })
    })) {
        vortex_bail!("hnswlib mapping contains duplicate, missing or uncovered rows");
    }
    Ok(())
}

/// Import a trusted, closed pinned native index and persist its external-label mapping.
///
/// The source must be absolute, symlink-free and quiescent, produced by the pinned
/// revision with the stated bundle shape. This writes private artifacts, not a
/// sealed or published generation. The owner must seal and reopen through the
/// provider, which checks native structure before loading it. Blocking IO requires
/// a blocking worker; errors may leave unpublished artifacts in this private store.
pub async fn import_bundle(
    mut metadata: IndexMetadata,
    store: &dyn IndexStore,
    native_index: &Path,
    bundle: HnswBundle,
    rows: &[RowAddress],
) -> VortexResult<IndexMetadata> {
    validate_metadata(&metadata)?;
    if !metadata.artifacts.is_empty() || !native_index.is_absolute() {
        vortex_bail!("hnswlib import requires empty artifacts and an absolute native file");
    }
    validate_rows(&metadata, bundle, rows)?;
    let local = store
        .as_local_files()
        .ok_or_else(|| vortex_err!("hnswlib requires a local-file store"))?;
    let descriptor = Descriptor {
        format_version: HNSWLIB_FORMAT_VERSION,
        revision: HNSWLIB_REVISION.into(),
        value_type: "float32".into(),
        metric: "squared_l2".into(),
        bundle,
        snapshot: metadata.snapshot.clone(),
        fields: metadata.fields.clone(),
        covered_files: metadata.covered_files.clone(),
    };
    let json =
        serde_json::to_vec(&descriptor).map_err(|err| vortex_err!("hnswlib descriptor: {err}"))?;
    if json.len() > 1024 * 1024 {
        vortex_bail!("hnswlib descriptor exceeds 1 MiB");
    }
    let capacity = rows
        .len()
        .checked_mul(16)
        .and_then(|size| size.checked_add(16))
        .ok_or_else(|| vortex_err!("hnswlib mapping size overflow"))?;
    let mut mapping = Vec::with_capacity(capacity);
    mapping.extend_from_slice(ROW_MAGIC);
    mapping.extend_from_slice(&u64::from(bundle.rows).to_le_bytes());
    for row in rows {
        mapping.extend_from_slice(&row.file_id.to_le_bytes());
        mapping.extend_from_slice(&row.row_offset.to_le_bytes());
    }
    metadata
        .artifacts
        .push(local.import_file(INDEX, native_index)?);
    metadata
        .artifacts
        .push(store.write(DESCRIPTOR, Bytes::from(json)).await?);
    metadata
        .artifacts
        .push(store.write(ROWS, Bytes::from(mapping)).await?);
    Ok(metadata)
}
