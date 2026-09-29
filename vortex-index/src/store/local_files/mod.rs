// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::fs::Permissions;
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use tempfile::TempDir;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use super::LocalIndexStore;
use super::State;
use super::artifact_parent;
use super::copy_and_hash;
use super::create_file;
use super::open_directory;
use super::open_file;
use super::open_root;
use super::verify;
use crate::IndexArtifact;
use crate::LocalArtifactLease;
use crate::LocalIndexFiles;
use crate::metadata::validate_artifact_path;

impl LocalIndexFiles for LocalIndexStore {
    fn import_file(&self, path: &str, source: &Path) -> VortexResult<IndexArtifact> {
        validate_artifact_path(path)?;
        if !source.is_absolute() {
            vortex_bail!("Import source must be absolute");
        }
        let parent = source
            .parent()
            .ok_or_else(|| vortex_err!("Import source must name a file"))?;
        let name = source
            .file_name()
            .ok_or_else(|| vortex_err!("Import source must name a file"))?;
        let mut source = open_file(&open_root(parent)?, name)?;
        let size = source.metadata()?.len();
        if size > u64::try_from(self.limits.max_artifact_bytes)? {
            vortex_bail!("Artifact exceeds the byte limit: {}", path);
        }
        self.write_artifact(path, |file| {
            let checksum = copy_and_hash(&mut source, size, |chunk| {
                file.write_all(chunk)?;
                Ok(())
            })
            .map_err(|err| err.with_context(format!("Importing artifact {path}")))?;
            Ok(IndexArtifact {
                path: path.to_owned(),
                size,
                checksum,
            })
        })
    }

    fn materialize(
        &self,
        artifacts: &[IndexArtifact],
        scratch_root: &Path,
        max_bytes: u64,
    ) -> VortexResult<Box<dyn LocalArtifactLease>> {
        {
            let state = self.state.lock();
            let State::Sealed(inventory) = &*state else {
                vortex_bail!("Only a sealed generation can be materialized");
            };
            let mut paths = BTreeSet::new();
            let mut total = 0u64;
            for artifact in artifacts {
                if inventory.get(&artifact.path) != Some(artifact) {
                    vortex_bail!("Artifact does not match the generation inventory");
                }
                if !paths.insert(&artifact.path) {
                    vortex_bail!("Duplicate materialization path: {}", artifact.path);
                }
                total = total
                    .checked_add(artifact.size)
                    .ok_or_else(|| vortex_err!("Materialization byte count overflow"))?;
                if total > max_bytes {
                    vortex_bail!("Materialization exceeds the total byte limit");
                }
            }
        }
        let root = open_root(scratch_root)?;
        // The scratch namespace must remain owner-managed, including creation
        // and TempDir cleanup. Native code receives copies, never canonical paths.
        let dir = tempfile::Builder::new()
            .prefix(".vortex-index-")
            .permissions(Permissions::from_mode(0o700))
            .tempdir_in(scratch_root)?;
        let name = dir
            .path()
            .file_name()
            .ok_or_else(|| vortex_err!("Missing scratch directory name"))?;
        let directory = open_directory(&root, name)?;
        for artifact in artifacts {
            let (parent, name) = artifact_parent(&directory, &artifact.path, true)?;
            let mut file = create_file(&parent, name)?;
            verify(
                &self.artifacts,
                artifact,
                self.limits.max_artifact_bytes,
                |chunk| {
                    file.write_all(chunk)?;
                    Ok(())
                },
            )?;
            file.set_permissions(Permissions::from_mode(0o400))?;
        }
        Ok(Box::new(ArtifactLease(dir)))
    }
}

#[derive(Debug)]
struct ArtifactLease(TempDir);

impl LocalArtifactLease for ArtifactLease {
    fn path(&self) -> &Path {
        self.0.path()
    }
}
