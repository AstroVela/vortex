// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Immutable local index generations (Unix, `local-store` feature).
//!
//! A generation is created once, written privately, then sealed with a manifest.
//! The returned descriptor must be persisted by the owning catalog together with
//! its expected source snapshot. Sealing is not a catalog commit. There is no
//! directory discovery, mutable "latest" pointer, writer resume, or reclamation.
//!
//! The root must already be durable, be absolute and have no symlink components. It
//! and its generations must be exclusively managed by the owner: this is not a
//! sandbox against processes that can rename directories or modify their bytes.
//! Descriptor-relative, no-follow opens reject symlinks, and nonblocking opens
//! reject special files without waiting on a FIFO. Reads verify content anew;
//! only the returned bytes, not the on-disk files, are pinned against mutation.
//!
//! Successful writes sync files and containing directories. Sealing syncs a
//! temporary manifest, installs it with an atomic, no-clobber hard link, then
//! syncs the generation directory. This requires a local filesystem supporting
//! these operations and their durability guarantees; network filesystems are
//! not qualified. A failed seal may leave a complete but unreferenced generation;
//! a failed write poisons its writer. Neither can publish an index to readers.
//!
//! IO is blocking, including the async [`IndexStore`] methods. Callers should use
//! a blocking worker. Whole-object reads have explicit byte limits; this is not
//! a range-IO or native-backend file-handle API.

use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::fs::File;
use std::io::Read;
use std::io::Write;
use std::mem;
use std::path::Component;
use std::path::Path;

use async_trait::async_trait;
use base16ct::HexDisplay;
use bytes::Bytes;
use parking_lot::Mutex;
use rustix::fs as unix_fs;
use rustix::io::Errno;
use serde::Deserialize;
use serde::Serialize;
use sha2::Digest;
use sha2::Sha256;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::IndexArtifact;
use crate::IndexMetadata;
use crate::IndexStore;
use crate::Snapshot;
use crate::metadata::validate_artifact_path;

const ARTIFACTS: &str = "artifacts";
const MANIFEST: &str = "manifest.json";
const PENDING: &str = "manifest.pending";

/// Explicit limits for whole-object IO, not a process RSS budget.
#[derive(Debug, Clone, Copy)]
pub struct LocalStoreLimits {
    /// Maximum size of each backend artifact, including writes and verification.
    pub max_artifact_bytes: usize,
    /// Maximum serialized manifest size, checked before parsing or installation.
    pub max_manifest_bytes: usize,
}

/// Trusted catalog reference to one sealed generation, not a publication record.
///
/// The manifest's SHA-256 binds all metadata and artifact identities. Callers
/// must not reconstruct this descriptor by hashing arbitrary files during open.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalGeneration {
    /// Single-component directory name; must match the index metadata generation.
    pub generation: String,
    /// Identity returned by seal for the generation's `manifest.json`.
    pub manifest: IndexArtifact,
}

enum State {
    Private(BTreeMap<String, IndexArtifact>),
    Sealed(BTreeMap<String, IndexArtifact>),
    Failed,
}

/// One private writer or one read-only sealed generation.
///
/// Creation is exclusive across processes. Only the creating handle can write;
/// reopening always requires a sealed descriptor and produces a read-only store.
/// Cloned `Arc`s share writer state, so writes and seal cannot race each other.
pub struct LocalIndexStore {
    directory: File,
    artifacts: File,
    generation: String,
    limits: LocalStoreLimits,
    state: Mutex<State>,
}

impl LocalIndexStore {
    /// Exclusively create a new generation under an existing, owner-managed root.
    ///
    /// Existing names (even abandoned generations) are never reused or removed.
    pub fn create(
        root: impl AsRef<Path>,
        generation: impl Into<String>,
        limits: LocalStoreLimits,
    ) -> VortexResult<Self> {
        let generation = generation.into();
        validate_generation(&generation)?;
        let root = open_root(root.as_ref())?;
        make_directory(&root, &generation)?;
        let directory = open_directory(&root, OsStr::new(&generation))?;
        make_directory(&directory, ARTIFACTS)?;
        let artifacts = open_directory(&directory, OsStr::new(ARTIFACTS))?;
        Ok(Self {
            directory,
            artifacts,
            generation,
            limits,
            state: Mutex::new(State::Private(BTreeMap::new())),
        })
    }

    /// Verify a trusted manifest, its snapshot and all artifacts before reopening.
    ///
    /// Artifact verification is streamed without retaining their contents. Later
    /// reads verify again, failing on mutation rather than adopting new contents.
    pub fn open(
        root: impl AsRef<Path>,
        descriptor: &LocalGeneration,
        snapshot: &Snapshot,
        limits: LocalStoreLimits,
    ) -> VortexResult<(Self, IndexMetadata)> {
        validate_generation(&descriptor.generation)?;
        snapshot.validate()?;
        if descriptor.manifest.path != MANIFEST {
            vortex_bail!("Invalid local generation manifest path");
        }
        let root = open_root(root.as_ref())?;
        let directory = open_directory(&root, OsStr::new(&descriptor.generation))?;
        let bytes = read_bytes(&directory, &descriptor.manifest, limits.max_manifest_bytes)?;
        let metadata: IndexMetadata = serde_json::from_slice(&bytes)
            .map_err(|err| vortex_err!("Invalid manifest: {}", err))?;
        metadata.validate_for(snapshot)?;
        if metadata.generation != descriptor.generation {
            vortex_bail!("Manifest generation does not match its descriptor");
        }
        let artifacts = open_directory(&directory, OsStr::new(ARTIFACTS))?;
        for artifact in &metadata.artifacts {
            verify(&artifacts, artifact, limits.max_artifact_bytes, |_| {})?;
        }
        let inventory = inventory(&metadata);
        Ok((
            Self {
                directory,
                artifacts,
                generation: descriptor.generation.clone(),
                limits,
                state: Mutex::new(State::Sealed(inventory)),
            },
            metadata,
        ))
    }

    /// Durably seal exactly the successful writes with validated index metadata.
    ///
    /// A successful return makes this handle read-only and returns the manifest
    /// identity for a separate catalog commit. The catalog still must validate
    /// source identity and publication conflicts. IO failures poison the writer;
    /// retry with a new generation, not the same directory name.
    pub fn seal(&self, metadata: &IndexMetadata) -> VortexResult<LocalGeneration> {
        let mut state = self.state.lock();
        let State::Private(written) = &*state else {
            vortex_bail!("Only a private generation can be sealed");
        };
        metadata.validate_for(&metadata.snapshot)?;
        if metadata.generation != self.generation || inventory(metadata) != *written {
            vortex_bail!("Manifest must match the generation and its complete written inventory");
        }
        let bytes = serde_json::to_vec(metadata).map_err(|err| vortex_err!("{}", err))?;
        if bytes.len() > self.limits.max_manifest_bytes {
            vortex_bail!("Manifest exceeds the byte limit");
        }
        // Any IO error from here leaves the writer unusable, including a sync
        // failure after the manifest has become visible in the private directory.
        *state = State::Failed;
        for artifact in &metadata.artifacts {
            verify(
                &self.artifacts,
                artifact,
                self.limits.max_artifact_bytes,
                |_| {},
            )?;
        }
        write_new(&self.directory, PENDING, &bytes)?;
        unix_fs::linkat(
            &self.directory,
            PENDING,
            &self.directory,
            MANIFEST,
            unix_fs::AtFlags::empty(),
        )
        .map_err(|err| vortex_err!("Cannot install sealed manifest: {}", err))?;
        #[cfg(test)]
        tests::inject_fault(tests::Fault::ManifestInstalled)?;
        unix_fs::unlinkat(&self.directory, PENDING, unix_fs::AtFlags::empty())
            .map_err(|err| vortex_err!("Cannot remove pending manifest: {}", err))?;
        self.directory.sync_all()?;
        *state = State::Sealed(inventory(metadata));
        Ok(LocalGeneration {
            generation: self.generation.clone(),
            manifest: artifact(MANIFEST, &bytes)?,
        })
    }
}

#[async_trait]
impl IndexStore for LocalIndexStore {
    async fn read(&self, artifact: &IndexArtifact) -> VortexResult<Bytes> {
        {
            let state = self.state.lock();
            let (State::Private(inventory) | State::Sealed(inventory)) = &*state else {
                vortex_bail!("Local generation has failed");
            };
            if inventory.get(&artifact.path) != Some(artifact) {
                vortex_bail!("Artifact does not match the generation inventory");
            }
        }
        read_bytes(&self.artifacts, artifact, self.limits.max_artifact_bytes)
    }

    async fn write(&self, path: &str, data: Bytes) -> VortexResult<IndexArtifact> {
        validate_artifact_path(path)?;
        if data.len() > self.limits.max_artifact_bytes {
            vortex_bail!("Artifact exceeds the byte limit");
        }
        let mut state = self.state.lock();
        let State::Private(inventory) = &mut *state else {
            vortex_bail!("Only a private generation can be written");
        };
        if inventory.contains_key(path) {
            vortex_bail!("Artifact already exists: {}", path);
        }
        let artifact = artifact(path, &data)?;
        // Failed writes may have created directories or partial files. Never
        // allow sealing a writer after such a failure or silently reuse a name.
        let State::Private(mut inventory) = mem::replace(&mut *state, State::Failed) else {
            vortex_bail!("Only a private generation can be written");
        };
        let (parent, name) = artifact_parent(&self.artifacts, path, true)?;
        write_new(&parent, name, &data)?;
        inventory.insert(path.to_owned(), artifact.clone());
        *state = State::Private(inventory);
        Ok(artifact)
    }
}

fn inventory(metadata: &IndexMetadata) -> BTreeMap<String, IndexArtifact> {
    metadata
        .artifacts
        .iter()
        .map(|artifact| (artifact.path.clone(), artifact.clone()))
        .collect()
}

fn validate_generation(generation: &str) -> VortexResult<()> {
    validate_artifact_path(generation)?;
    if generation.contains('/') {
        vortex_bail!("Local generation must be a single path component");
    }
    Ok(())
}

fn open_root(path: &Path) -> VortexResult<File> {
    if !path.is_absolute() {
        vortex_bail!("Local store root must be absolute");
    }
    let mut directory = File::from(
        unix_fs::open("/", directory_flags(), unix_fs::Mode::empty())
            .map_err(|err| vortex_err!("Cannot open filesystem root: {}", err))?,
    );
    for component in path.components() {
        match component {
            Component::RootDir => {}
            Component::Normal(name) => directory = open_directory(&directory, name)?,
            _ => vortex_bail!("Local store root must not contain parent components"),
        }
    }
    Ok(directory)
}

fn directory_flags() -> unix_fs::OFlags {
    unix_fs::OFlags::RDONLY
        | unix_fs::OFlags::DIRECTORY
        | unix_fs::OFlags::NOFOLLOW
        | unix_fs::OFlags::CLOEXEC
}

fn open_directory(parent: &File, name: &OsStr) -> VortexResult<File> {
    Ok(File::from(
        unix_fs::openat(parent, name, directory_flags(), unix_fs::Mode::empty())
            .map_err(|err| vortex_err!("Cannot open local directory {:?}: {}", name, err))?,
    ))
}

fn make_directory(parent: &File, name: &str) -> VortexResult<()> {
    unix_fs::mkdirat(parent, name, unix_fs::Mode::RWXU)
        .map_err(|err| vortex_err!("Cannot create local directory {}: {}", name, err))?;
    open_directory(parent, OsStr::new(name))?.sync_all()?;
    parent.sync_all()?;
    Ok(())
}

fn artifact_parent<'a>(root: &File, path: &'a str, create: bool) -> VortexResult<(File, &'a str)> {
    validate_artifact_path(path)?;
    let mut directory = root.try_clone()?;
    let mut parts = path.split('/').peekable();
    while let Some(part) = parts.next() {
        if parts.peek().is_none() {
            return Ok((directory, part));
        }
        if create {
            match unix_fs::mkdirat(&directory, part, unix_fs::Mode::RWXU) {
                Ok(()) => directory.sync_all()?,
                Err(Errno::EXIST) => {}
                Err(err) => vortex_bail!("Cannot create artifact directory {}: {}", part, err),
            }
        }
        directory = open_directory(&directory, OsStr::new(part))?;
    }
    vortex_bail!("Empty artifact path")
}

fn file_flags() -> unix_fs::OFlags {
    unix_fs::OFlags::NOFOLLOW
        | unix_fs::OFlags::NONBLOCK
        | unix_fs::OFlags::NOCTTY
        | unix_fs::OFlags::CLOEXEC
}

fn write_new(parent: &File, name: &str, data: &[u8]) -> VortexResult<()> {
    let mut file = File::from(
        unix_fs::openat(
            parent,
            name,
            file_flags()
                | unix_fs::OFlags::WRONLY
                | unix_fs::OFlags::CREATE
                | unix_fs::OFlags::EXCL,
            unix_fs::Mode::RUSR | unix_fs::Mode::WUSR,
        )
        .map_err(|err| vortex_err!("Cannot create artifact {}: {}", name, err))?,
    );
    file.write_all(data)?;
    #[cfg(test)]
    tests::inject_fault(tests::Fault::FileWritten)?;
    file.sync_all()?;
    #[cfg(test)]
    tests::inject_fault(tests::Fault::FileSynced)?;
    parent.sync_all()?;
    Ok(())
}

fn artifact(path: &str, bytes: &[u8]) -> VortexResult<IndexArtifact> {
    Ok(IndexArtifact {
        path: path.to_owned(),
        size: u64::try_from(bytes.len())?,
        checksum: format!("sha256:{:x}", HexDisplay(&Sha256::digest(bytes))),
    })
}

fn read_bytes(root: &File, artifact: &IndexArtifact, limit: usize) -> VortexResult<Bytes> {
    let mut bytes = Vec::new();
    verify(root, artifact, limit, |chunk| {
        bytes.extend_from_slice(chunk)
    })?;
    Ok(Bytes::from(bytes))
}

fn verify(
    root: &File,
    artifact: &IndexArtifact,
    limit: usize,
    mut consume: impl FnMut(&[u8]),
) -> VortexResult<()> {
    if artifact.size > u64::try_from(limit)? {
        vortex_bail!("Artifact exceeds the byte limit: {}", artifact.path);
    }
    if !artifact
        .checksum
        .strip_prefix("sha256:")
        .is_some_and(|hex| {
            hex.len() == 64
                && hex
                    .bytes()
                    .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
        })
    {
        vortex_bail!("Invalid SHA-256 artifact checksum: {}", artifact.path);
    }
    let (parent, name) = artifact_parent(root, &artifact.path, false)?;
    let mut file = File::from(
        unix_fs::openat(
            &parent,
            name,
            file_flags() | unix_fs::OFlags::RDONLY,
            unix_fs::Mode::empty(),
        )
        .map_err(|err| vortex_err!("Cannot open artifact {}: {}", artifact.path, err))?,
    );
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        vortex_bail!("Artifact is not a regular file: {}", artifact.path);
    }
    if metadata.len() != artifact.size {
        vortex_bail!("Artifact length mismatch: {}", artifact.path);
    }
    let mut remaining = artifact.size;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    while remaining != 0 {
        let count = usize::try_from(remaining.min(buffer.len() as u64))?;
        let count = file.read(&mut buffer[..count])?;
        if count == 0 {
            vortex_bail!("Truncated artifact: {}", artifact.path);
        }
        remaining -= count as u64;
        hasher.update(&buffer[..count]);
        consume(&buffer[..count]);
    }
    if file.read(&mut buffer[..1])? != 0 {
        vortex_bail!(
            "Artifact grew beyond its declared length: {}",
            artifact.path
        );
    }
    if format!("sha256:{:x}", HexDisplay(&hasher.finalize())) != artifact.checksum {
        vortex_bail!("Artifact checksum mismatch: {}", artifact.path);
    }
    Ok(())
}

#[cfg(test)]
mod tests;
