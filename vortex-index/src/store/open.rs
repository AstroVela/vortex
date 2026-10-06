// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::OsStr;
use std::fs::File;
use std::os::unix::fs::MetadataExt;
use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use parking_lot::Mutex;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use super::ARTIFACTS;
use super::LocalGeneration;
use super::LocalIndexStore;
use super::LocalStoreLimits;
use super::MANIFEST;
use super::State;
use super::inventory;
use super::local_files::ArtifactLease;
use super::local_files::copy_artifacts;
use super::local_files::validate_selection;
use super::open_directory;
use super::open_root;
use super::read_bytes;
use super::validate_generation;
use super::verify;
use crate::IndexArtifact;
use crate::IndexMetadata;
use crate::IndexStore;
use crate::LocalArtifactLease;
use crate::LocalIndexFiles;
use crate::Snapshot;

/// Authenticated generation metadata whose artifact contents are not yet verified.
///
/// This is not a readable or sealed store. Consuming it through [`Self::verify`]
/// or [`Self::materialize`] verifies every declared artifact before exposing a
/// store. Directory descriptors pin path traversal; file contents can still change
/// and must match the authenticated lengths and checksums when read.
pub struct LocalIndexOpen {
    directory: File,
    artifacts: File,
    metadata: IndexMetadata,
    limits: LocalStoreLimits,
}

impl LocalIndexOpen {
    pub(super) fn new(
        root: &Path,
        descriptor: &LocalGeneration,
        snapshot: &Snapshot,
        limits: LocalStoreLimits,
    ) -> VortexResult<Self> {
        validate_generation(&descriptor.generation)?;
        snapshot.validate()?;
        if descriptor.manifest.path != MANIFEST {
            vortex_bail!("Invalid local generation manifest path");
        }
        let root = open_root(root)?;
        let directory = open_directory(&root, OsStr::new(&descriptor.generation))?;
        let bytes = read_bytes(&directory, &descriptor.manifest, limits.max_manifest_bytes)?;
        let metadata: IndexMetadata = serde_json::from_slice(&bytes)
            .map_err(|err| vortex_err!("Invalid manifest: {}", err))?;
        metadata.validate_for(snapshot)?;
        if metadata.generation != descriptor.generation {
            vortex_bail!("Manifest generation does not match its descriptor");
        }
        let artifacts = open_directory(&directory, OsStr::new(ARTIFACTS))?;
        Ok(Self {
            directory,
            artifacts,
            metadata,
            limits,
        })
    }

    /// Metadata bound to the trusted manifest and source snapshot.
    ///
    /// Artifact identities describe expected contents, not completed verification.
    pub fn metadata(&self) -> &IndexMetadata {
        &self.metadata
    }

    /// Verify all artifacts and reopen the canonical generation as a sealed store.
    ///
    /// Later reads and materializations verify again, including after mutations.
    pub fn verify(self) -> VortexResult<(LocalIndexStore, IndexMetadata)> {
        for artifact in &self.metadata.artifacts {
            verify(
                &self.artifacts,
                artifact,
                self.limits.max_artifact_bytes,
                |_| Ok(()),
            )?;
        }
        Ok((
            LocalIndexStore {
                directory: self.directory,
                artifacts: self.artifacts,
                generation: self.metadata.generation.clone(),
                limits: self.limits,
                state: Mutex::new(State::Sealed(inventory(&self.metadata))),
            },
            self.metadata,
        ))
    }

    /// Copy and verify the entire generation once into an immutable private store.
    ///
    /// `scratch_root` and byte limits have the same contract as
    /// [`LocalIndexFiles::materialize`]. All artifacts are verified before success;
    /// failure removes partial copies. Subsequent canonical mutations cannot change
    /// this store. Its reads still verify private-file contents.
    ///
    /// The first complete-inventory materialization into this same scratch root
    /// transfers the already verified lease without copying or hashing again.
    /// Other materializations create independent verified copies. The store and
    /// transferred lease keep the private directory alive until both are dropped.
    pub fn materialize(
        self,
        scratch_root: &Path,
        max_bytes: u64,
    ) -> VortexResult<(Arc<dyn IndexStore>, IndexMetadata)> {
        let inventory = inventory(&self.metadata);
        validate_selection(&inventory, &self.metadata.artifacts, max_bytes)?;
        let lease = copy_artifacts(
            &self.artifacts,
            &self.metadata.artifacts,
            scratch_root,
            self.limits.max_artifact_bytes,
        )?;
        let directory = open_root(lease.path())?;
        let store = LocalIndexStore {
            artifacts: directory.try_clone()?,
            directory,
            generation: self.metadata.generation.clone(),
            limits: self.limits,
            state: Mutex::new(State::Sealed(inventory)),
        };
        Ok((
            Arc::new(MaterializedStore {
                store,
                lease,
                available: Mutex::new(true),
            }),
            self.metadata,
        ))
    }
}

struct MaterializedStore {
    store: LocalIndexStore,
    lease: ArtifactLease,
    available: Mutex<bool>,
}

#[async_trait]
impl IndexStore for MaterializedStore {
    async fn read(&self, artifact: &IndexArtifact) -> VortexResult<Bytes> {
        self.store.read(artifact).await
    }

    async fn write(&self, _path: &str, _data: Bytes) -> VortexResult<IndexArtifact> {
        vortex_bail!("A materialized generation is read-only");
    }

    fn as_local_files(&self) -> Option<&dyn LocalIndexFiles> {
        Some(self)
    }
}

impl LocalIndexFiles for MaterializedStore {
    fn import_file(&self, _path: &str, _source: &Path) -> VortexResult<IndexArtifact> {
        vortex_bail!("A materialized generation is read-only");
    }

    fn materialize(
        &self,
        artifacts: &[IndexArtifact],
        scratch_root: &Path,
        max_bytes: u64,
    ) -> VortexResult<Box<dyn LocalArtifactLease>> {
        let complete = {
            let state = self.store.state.lock();
            let State::Sealed(inventory) = &*state else {
                vortex_bail!("Materialized generation must be sealed");
            };
            validate_selection(inventory, artifacts, max_bytes)?;
            artifacts.len() == inventory.len()
        };
        if complete && self.lease.path().parent() == Some(scratch_root) {
            let mut available = self.available.lock();
            if *available {
                let actual = open_root(self.lease.path())?.metadata()?;
                let expected = self.store.artifacts.metadata()?;
                if (actual.dev(), actual.ino()) != (expected.dev(), expected.ino()) {
                    vortex_bail!("Materialized directory changed before lease transfer");
                }
                *available = false;
                return Ok(Box::new(self.lease.clone()));
            }
        }
        self.store.materialize(artifacts, scratch_root, max_bytes)
    }
}
