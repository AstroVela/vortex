// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::fs;
use std::os::unix::fs::MetadataExt;
use std::os::unix::fs::symlink;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;

use bytes::Bytes;
use futures::executor::block_on;
use tempfile::TempDir;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::FAULT;
use super::Fault;
use super::LIMITS;
use super::READ_BYTES;
use super::assert_error;
use super::assert_rejects_fifo;
use super::metadata;
use crate::IndexMetadata;
use crate::IndexStore;
use crate::store::COPY_BUFFER_BYTES;
use crate::store::LocalGeneration;
use crate::store::LocalIndexOpen;
use crate::store::LocalIndexStore;
use crate::store::LocalStoreLimits;

struct Fixture {
    root: TempDir,
    scratch: PathBuf,
    descriptor: LocalGeneration,
    metadata: IndexMetadata,
    data: Vec<u8>,
}

impl Fixture {
    fn new() -> VortexResult<Self> {
        let root = tempfile::tempdir()?;
        let scratch = root.path().join("scratch");
        fs::create_dir(&scratch)?;
        let store = LocalIndexStore::create(root.path(), "gen-1", LIMITS)?;
        let data = vec![42; COPY_BUFFER_BYTES * 2 + 17];
        let first = block_on(store.write("native/data", Bytes::copy_from_slice(&data)))?;
        let second = block_on(store.write("empty", Bytes::new()))?;
        let metadata = metadata(vec![first, second]);
        let descriptor = store.seal(&metadata)?;
        Ok(Self {
            root,
            scratch,
            descriptor,
            metadata,
            data,
        })
    }

    fn prepare(&self) -> VortexResult<LocalIndexOpen> {
        LocalIndexStore::prepare_open(
            self.root.path(),
            &self.descriptor,
            &self.metadata.snapshot,
            LIMITS,
        )
    }

    fn canonical(&self) -> PathBuf {
        self.root.path().join("gen-1/artifacts/native/data")
    }

    fn materialize(&self) -> VortexResult<Arc<dyn IndexStore>> {
        let (store, metadata) = self
            .prepare()?
            .materialize(&self.scratch, self.data.len() as u64)?;
        assert_eq!(metadata, self.metadata);
        Ok(store)
    }
}

struct ReadCounter;

impl ReadCounter {
    fn new() -> Self {
        READ_BYTES.set(Some(0));
        Self
    }
}

impl Drop for ReadCounter {
    fn drop(&mut self) {
        READ_BYTES.set(None);
    }
}

#[rstest::rstest]
#[case::ordinary(false, 2)]
#[case::verified_materialization(true, 1)]
fn test_complete_open_reads_artifacts_once(
    #[case] materialized: bool,
    #[case] passes: u64,
) -> VortexResult<()> {
    let fixture = Fixture::new()?;
    let _counter = ReadCounter::new();
    let pending = fixture.prepare()?;
    assert_eq!(pending.metadata(), &fixture.metadata);
    assert_eq!(READ_BYTES.get(), Some(fixture.descriptor.manifest.size));
    let store: Arc<dyn IndexStore> = if materialized {
        pending
            .materialize(&fixture.scratch, fixture.data.len() as u64)?
            .0
    } else {
        Arc::new(pending.verify()?.0)
    };
    let files = store
        .as_local_files()
        .ok_or_else(|| vortex_err!("Missing local files"))?;
    let lease = files.materialize(
        &fixture.metadata.artifacts,
        &fixture.scratch,
        fixture.data.len() as u64,
    )?;
    assert_eq!(
        READ_BYTES.get(),
        Some(fixture.descriptor.manifest.size + passes * fixture.data.len() as u64)
    );
    assert_eq!(fs::read_dir(&fixture.scratch)?.count(), 1);
    let private = lease.path().join("native/data");
    assert_ne!(
        fs::metadata(&private)?.ino(),
        fs::metadata(fixture.canonical())?.ino()
    );
    assert_eq!(fs::metadata(lease.path())?.mode() & 0o777, 0o700);
    assert_eq!(fs::metadata(&private)?.mode() & 0o777, 0o400);
    assert_eq!(fs::read(&private)?, fixture.data);
    let path = lease.path().to_owned();
    drop(store);
    assert!(path.is_dir());
    drop(lease);
    assert!(!path.exists());
    Ok(())
}

#[rstest::rstest]
#[case::missing("missing")]
#[case::truncated("truncated")]
#[case::grown("grown")]
#[case::corrupt("corrupt")]
#[case::directory("directory")]
#[case::symlink("symlink")]
fn test_changes_after_manifest_validation_are_rejected(#[case] change: &str) -> VortexResult<()> {
    let fixture = Fixture::new()?;
    let pending = fixture.prepare()?;
    let path = fixture.canonical();
    match change {
        "truncated" => fs::write(&path, b"short")?,
        "grown" => fs::write(&path, vec![42; fixture.data.len() + 1])?,
        "corrupt" => fs::write(&path, vec![43; fixture.data.len()])?,
        _ => {
            fs::remove_file(&path)?;
            match change {
                "directory" => fs::create_dir(&path)?,
                "symlink" => symlink(fixture.root.path().join("gen-1/artifacts/empty"), &path)?,
                _ => {}
            }
        }
    }
    assert!(pending.materialize(&fixture.scratch, u64::MAX).is_err());
    assert_eq!(fs::read_dir(&fixture.scratch)?.count(), 0);
    Ok(())
}

#[test]
fn test_fifo_and_copy_failure_cleanup() -> VortexResult<()> {
    let fixture = Fixture::new()?;
    let pending = fixture.prepare()?;
    fs::remove_file(fixture.canonical())?;
    assert_rejects_fifo(&fixture.canonical(), || {
        pending.materialize(&fixture.scratch, u64::MAX)
    })?;
    assert_eq!(fs::read_dir(&fixture.scratch)?.count(), 0);
    fs::remove_file(fixture.canonical())?;
    fs::write(fixture.canonical(), &fixture.data)?;
    let pending = fixture.prepare()?;
    FAULT.set(Some(Fault::ChunkCopied));
    let failed = pending.materialize(&fixture.scratch, u64::MAX);
    FAULT.set(None);
    assert_error(failed, "Injected")?;
    assert_eq!(fs::read_dir(&fixture.scratch)?.count(), 0);
    drop(fixture.materialize()?);
    assert_eq!(fs::read_dir(&fixture.scratch)?.count(), 0);
    Ok(())
}

#[test]
fn test_limits_and_invalid_requests_do_not_consume_the_lease() -> VortexResult<()> {
    let fixture = Fixture::new()?;
    assert_error(
        fixture
            .prepare()?
            .materialize(&fixture.scratch, fixture.data.len() as u64 - 1),
        "total byte limit",
    )?;
    let pending = LocalIndexStore::prepare_open(
        fixture.root.path(),
        &fixture.descriptor,
        &fixture.metadata.snapshot,
        LocalStoreLimits {
            max_artifact_bytes: fixture.data.len() - 1,
            ..LIMITS
        },
    )?;
    assert_error(
        pending.materialize(&fixture.scratch, u64::MAX),
        "byte limit",
    )?;
    assert_eq!(fs::read_dir(&fixture.scratch)?.count(), 0);
    let store = fixture.materialize()?;
    let files = store
        .as_local_files()
        .ok_or_else(|| vortex_err!("Missing local files"))?;
    let mut invalid = fixture.metadata.artifacts.clone();
    invalid[0].checksum.push('0');
    assert_error(
        files.materialize(&invalid, &fixture.scratch, u64::MAX),
        "inventory",
    )?;
    let duplicate = vec![fixture.metadata.artifacts[0].clone(); 2];
    assert_error(
        files.materialize(&duplicate, &fixture.scratch, u64::MAX),
        "Duplicate",
    )?;
    assert_error(
        files.materialize(&fixture.metadata.artifacts, &fixture.scratch, 0),
        "total byte limit",
    )?;
    let _counter = ReadCounter::new();
    let lease = files.materialize(
        &fixture.metadata.artifacts,
        &fixture.scratch,
        fixture.data.len() as u64,
    )?;
    assert_eq!(READ_BYTES.get(), Some(0));
    drop(store);
    drop(lease);
    assert_eq!(fs::read_dir(&fixture.scratch)?.count(), 0);
    Ok(())
}

#[test]
fn test_private_store_survives_canonical_removal_and_rechecks_private_reads() -> VortexResult<()> {
    let fixture = Fixture::new()?;
    let store = fixture.materialize()?;
    fs::remove_dir_all(fixture.root.path().join("gen-1"))?;
    let item = &fixture.metadata.artifacts[0];
    assert_eq!(block_on(store.read(item))?, fixture.data);
    assert!(block_on(store.write("new", Bytes::new())).is_err());
    let files = store
        .as_local_files()
        .ok_or_else(|| vortex_err!("Missing local files"))?;
    assert!(files.import_file("new", &fixture.canonical()).is_err());
    let first = files.materialize(&fixture.metadata.artifacts, &fixture.scratch, u64::MAX)?;
    let second = files.materialize(&fixture.metadata.artifacts, &fixture.scratch, u64::MAX)?;
    assert_ne!(first.path(), second.path());
    fs::remove_file(first.path().join(&item.path))?;
    fs::write(first.path().join(&item.path), vec![43; fixture.data.len()])?;
    assert_error(block_on(store.read(item)), "checksum mismatch")?;
    assert_eq!(fs::read(second.path().join(&item.path))?, fixture.data);
    drop((first, second, store));
    assert_eq!(fs::read_dir(&fixture.scratch)?.count(), 0);
    Ok(())
}

#[test]
fn test_subset_and_other_root_leave_the_complete_lease_available() -> VortexResult<()> {
    let fixture = Fixture::new()?;
    let store = fixture.materialize()?;
    let files = store
        .as_local_files()
        .ok_or_else(|| vortex_err!("Missing local files"))?;
    let subset = files.materialize(&fixture.metadata.artifacts[..1], &fixture.scratch, u64::MAX)?;
    assert!(!subset.path().join("empty").exists());
    let other = tempfile::tempdir()?;
    let elsewhere = files.materialize(&fixture.metadata.artifacts, other.path(), u64::MAX)?;
    let _counter = ReadCounter::new();
    let mut reversed = fixture.metadata.artifacts.clone();
    reversed.reverse();
    let complete = files.materialize(&reversed, &fixture.scratch, u64::MAX)?;
    assert_eq!(READ_BYTES.get(), Some(0));
    assert_ne!(complete.path(), subset.path());
    assert!(elsewhere.path().starts_with(other.path()));
    assert!(complete.path().join("empty").exists());
    drop((subset, elsewhere, complete, store));
    assert_eq!(fs::read_dir(&fixture.scratch)?.count(), 0);
    assert_eq!(fs::read_dir(other.path())?.count(), 0);
    Ok(())
}

#[test]
fn test_concurrent_calls_transfer_once_and_own_independent_copies() -> VortexResult<()> {
    let fixture = Fixture::new()?;
    let store = fixture.materialize()?;
    let initial = fs::read_dir(&fixture.scratch)?
        .next()
        .transpose()?
        .ok_or_else(|| vortex_err!("Missing private directory"))?
        .path();
    let workers = (0..8)
        .map(|_| {
            let store = Arc::clone(&store);
            let artifacts = fixture.metadata.artifacts.clone();
            let scratch = fixture.scratch.clone();
            thread::spawn(move || {
                store
                    .as_local_files()
                    .ok_or_else(|| vortex_err!("Missing local files"))?
                    .materialize(&artifacts, &scratch, u64::MAX)
            })
        })
        .collect::<Vec<_>>();
    let leases = workers
        .into_iter()
        .map(|worker| {
            worker
                .join()
                .map_err(|_| vortex_err!("Materializer panicked"))?
        })
        .collect::<VortexResult<Vec<_>>>()?;
    assert_eq!(
        leases
            .iter()
            .filter(|lease| lease.path() == initial)
            .count(),
        1
    );
    assert_eq!(
        leases
            .iter()
            .map(|lease| lease.path())
            .collect::<BTreeSet<_>>()
            .len(),
        8
    );
    let inodes = leases
        .iter()
        .map(|lease| Ok(fs::metadata(lease.path().join("native/data"))?.ino()))
        .collect::<VortexResult<BTreeSet<_>>>()?;
    assert_eq!(inodes.len(), 8);
    drop(leases);
    assert!(initial.exists());
    drop(store);
    assert_eq!(fs::read_dir(&fixture.scratch)?.count(), 0);
    Ok(())
}

#[test]
fn test_transfer_rejects_replaced_scratch_directory() -> VortexResult<()> {
    let fixture = Fixture::new()?;
    let store = fixture.materialize()?;
    let initial = fs::read_dir(&fixture.scratch)?
        .next()
        .transpose()?
        .ok_or_else(|| vortex_err!("Missing private directory"))?
        .path();
    let moved = initial.with_extension("moved");
    fs::rename(&initial, &moved)?;
    fs::create_dir(&initial)?;
    let files = store
        .as_local_files()
        .ok_or_else(|| vortex_err!("Missing local files"))?;
    assert_error(
        files.materialize(&fixture.metadata.artifacts, &fixture.scratch, u64::MAX),
        "directory changed",
    )?;
    fs::remove_dir(&initial)?;
    symlink(&moved, &initial)?;
    assert!(
        files
            .materialize(&fixture.metadata.artifacts, &fixture.scratch, u64::MAX)
            .is_err()
    );
    fs::remove_file(&initial)?;
    fs::rename(&moved, &initial)?;
    drop(files.materialize(&fixture.metadata.artifacts, &fixture.scratch, u64::MAX)?);
    drop(store);
    assert_eq!(fs::read_dir(&fixture.scratch)?.count(), 0);
    Ok(())
}
