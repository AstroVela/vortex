// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;

use serde::Deserialize;
use serde::Serialize;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

/// A physical row address, meaningful only within its enclosing snapshot.
///
/// File IDs are assigned by the source, not inferred from listing order. A
/// rewrite/compaction must change the snapshot; these are not stable logical IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RowAddress {
    /// Source-assigned file identity within the dataset.
    pub file_id: u64,
    /// Zero-based physical offset in the file, before filtering.
    pub row_offset: u64,
}

/// Identity of an immutable source file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceFile {
    /// Source-assigned identity, independent of scan partition numbering.
    pub id: u64,
    /// Location interpreted by the source implementation.
    pub uri: String,
    /// Content digest or immutable object version, not just path or modification time.
    pub version: String,
    /// Physical row count, including any rows hidden by the snapshot's deletion mask.
    pub row_count: u64,
}

/// A source's immutable, ordered file inventory and visibility version.
///
/// The source must pin these versions while reading. Validation checks the
/// descriptor, not the underlying objects; adapters must enforce that pinning.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Snapshot {
    /// Namespace separating otherwise identical row addresses in different datasets.
    pub dataset_id: String,
    /// Changes with data, file layout, or deletion visibility.
    pub version: String,
    /// Fingerprint of the logical schema, including column identity and type.
    pub schema_fingerprint: String,
    /// File inventory in strictly increasing file-ID order.
    pub files: Vec<SourceFile>,
}

impl Snapshot {
    /// Reject ambiguous identities and noncanonical file inventories.
    pub fn validate(&self) -> VortexResult<()> {
        for value in [&self.dataset_id, &self.version, &self.schema_fingerprint] {
            nonempty(value)?;
        }
        if self.files.windows(2).any(|pair| pair[0].id >= pair[1].id) {
            vortex_bail!("Snapshot file IDs must be unique and strictly sorted");
        }
        for file in &self.files {
            nonempty(&file.uri)?;
            nonempty(&file.version)?;
        }
        Ok(())
    }

    /// Find a file by identity in a validated snapshot.
    pub fn file(&self, id: u64) -> Option<&SourceFile> {
        self.files
            .binary_search_by_key(&id, |file| file.id)
            .ok()
            .map(|idx| &self.files[idx])
    }

    /// Check that a row belongs to the snapshot's physical address space.
    pub fn validate_row(&self, row: RowAddress) -> VortexResult<()> {
        if self
            .file(row.file_id)
            .is_none_or(|file| row.row_offset >= file.row_count)
        {
            vortex_bail!(
                "Row address is outside the snapshot: file={}, offset={}",
                row.file_id,
                row.row_offset
            );
        }
        Ok(())
    }
}

/// An immutable backend artifact, relative to one index generation's store.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexArtifact {
    /// Portable relative path; no parent components, URI scheme, or backslashes.
    pub path: String,
    /// Exact byte length.
    pub size: u64,
    /// Algorithm-tagged content digest, such as `sha256:<hex>`.
    pub checksum: String,
}

/// Common index metadata; algorithm-specific details belong in backend artifacts.
///
/// This is a draft sidecar schema, not part of the Vortex file format. Deserializers
/// must call [`Self::validate_for`] before using it. Publication belongs to the
/// source's catalog/transaction layer, not to an index builder.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexMetadata {
    /// Common metadata format version; currently 1.
    pub format_version: u32,
    /// Logical index name within the dataset.
    pub name: String,
    /// Unique immutable generation, never reused for different artifacts.
    pub generation: String,
    /// Case-sensitive provider identifier.
    pub backend: String,
    /// Backend artifact format version.
    pub backend_version: u32,
    /// Exact source snapshot against which this generation was built.
    pub snapshot: Snapshot,
    /// Top-level logical field names; their identities/types are bound by the schema fingerprint.
    pub fields: Vec<String>,
    /// Strictly sorted file IDs covered in full for visible, eligible rows.
    pub covered_files: Vec<u64>,
    /// Complete artifact inventory; reference-only in-memory indexes may leave it empty.
    pub artifacts: Vec<IndexArtifact>,
}

impl IndexMetadata {
    /// Validate the metadata and require exact source/visibility identity.
    ///
    /// A newer snapshot is rejected even if some files are unchanged. Coverage
    /// reuse across snapshots requires a future table-layer compatibility proof.
    pub fn validate_for(&self, snapshot: &Snapshot) -> VortexResult<()> {
        snapshot.validate()?;
        self.snapshot.validate()?;
        if self.format_version != 1 || self.backend_version == 0 {
            vortex_bail!("Unsupported index metadata or backend format version");
        }
        for value in [&self.name, &self.generation, &self.backend] {
            nonempty(value)?;
        }
        if self.snapshot != *snapshot {
            vortex_bail!("Index snapshot does not match the source snapshot");
        }
        let mut fields = BTreeSet::new();
        if self.fields.is_empty() {
            vortex_bail!("An index must declare its fields");
        }
        for field in &self.fields {
            nonempty(field)?;
            if !fields.insert(field) {
                vortex_bail!("Duplicate indexed field: {}", field);
            }
        }
        if self.covered_files.windows(2).any(|pair| pair[0] >= pair[1]) {
            vortex_bail!("Index coverage must be unique and strictly sorted");
        }
        for id in &self.covered_files {
            if snapshot.file(*id).is_none() {
                vortex_bail!("Index covers an unknown file: {}", id);
            }
        }
        let mut paths = BTreeSet::new();
        for artifact in &self.artifacts {
            validate_artifact_path(&artifact.path)?;
            nonempty(&artifact.checksum)?;
            if !paths.insert(&artifact.path) {
                vortex_bail!("Duplicate index artifact: {}", artifact.path);
            }
        }
        Ok(())
    }

    /// Whether this generation covers the entire file.
    pub fn covers(&self, file_id: u64) -> bool {
        self.covered_files.binary_search(&file_id).is_ok()
    }

    /// Files requiring a scan in addition to this index's search.
    pub fn uncovered_files(&self) -> impl Iterator<Item = &SourceFile> {
        self.snapshot
            .files
            .iter()
            .filter(|file| !self.covers(file.id))
    }
}

pub(crate) fn nonempty(value: &str) -> VortexResult<()> {
    if value.trim().is_empty() || value.contains('\0') {
        vortex_bail!("Index identities must be nonempty and contain no NUL");
    }
    Ok(())
}

pub(crate) fn validate_artifact_path(path: &str) -> VortexResult<()> {
    nonempty(path)?;
    if path.contains(['\\', ':']) || path.split('/').any(|part| matches!(part, "" | "." | "..")) {
        vortex_bail!(
            "Index artifact must have a portable relative path: {}",
            path
        );
    }
    Ok(())
}
