// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::io::Read;
use std::io::Write;
use std::path::Path;

use serde::Deserialize;
use serde::Serialize;
use sha2::Digest;
use sha2::Sha256;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;

use crate::schema::validate_names;

const FORMAT_VERSION: u32 = 1;
const TOKENIZER: &str = "tantivy-0.25-default";

/// Metadata committed only after the Tantivy index has been completed.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IndexMetadata {
    /// Version of the sidecar manifest and schema contract.
    pub format_version: u32,
    /// Number of physical source rows, including rows with null/empty text.
    pub row_count: u64,
    /// Indexed top-level UTF-8 columns, in schema order.
    pub columns: Vec<String>,
    /// Pinned analyzer contract; changes require a rebuilt index.
    pub tokenizer: String,
    pub(crate) source: SourceIdentity,
}

impl IndexMetadata {
    pub(crate) fn new(source: SourceIdentity, row_count: u64, columns: Vec<String>) -> Self {
        Self {
            format_version: FORMAT_VERSION,
            row_count,
            columns,
            tokenizer: TOKENIZER.into(),
            source,
        }
    }

    pub(crate) fn read(directory: &Path) -> VortexResult<Self> {
        let manifest: Self = serde_json::from_reader(File::open(directory.join("manifest.json"))?)
            .map_err(|err| vortex_err!(External: err))?;
        if manifest.format_version != FORMAT_VERSION || manifest.tokenizer != TOKENIZER {
            vortex_bail!("unsupported sidecar format or tokenizer; rebuild the index");
        }
        validate_names(&manifest.columns)?;
        Ok(manifest)
    }

    pub(crate) fn publish(&self, directory: &Path) -> VortexResult<()> {
        let mut temporary = tempfile::NamedTempFile::new_in(directory)?;
        serde_json::to_writer_pretty(&mut temporary, self)
            .map_err(|err| vortex_err!(External: err))?;
        temporary.write_all(b"\n")?;
        temporary.as_file().sync_all()?;
        temporary
            .persist_noclobber(directory.join("manifest.json"))
            .map_err(|err| vortex_err!(External: err))?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SourceIdentity {
    bytes: u64,
    sha256: String,
}

impl SourceIdentity {
    pub(crate) fn read(path: &Path) -> VortexResult<Self> {
        let mut file = File::open(path)?;
        if !file.metadata()?.is_file() {
            vortex_bail!("source must be a regular local file");
        }
        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; 1024 * 1024];
        let mut bytes = 0u64;
        loop {
            let read = file.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
            bytes += read as u64;
        }
        Ok(Self {
            bytes,
            sha256: hasher
                .finalize()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect(),
        })
    }

    pub(crate) fn verify(&self, path: &Path) -> VortexResult<()> {
        if *self != Self::read(path)? {
            vortex_bail!("source file has changed or does not match this index; rebuild the index");
        }
        Ok(())
    }
}
