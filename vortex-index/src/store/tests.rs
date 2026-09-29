// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cell::Cell;
use std::fs;
use std::os::unix::fs::symlink;
use std::path::Path;
use std::sync::Arc;
use std::sync::Barrier;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use bytes::Bytes;
use futures::executor::block_on;
use rustix::fs as unix_fs;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::LocalGeneration;
use super::LocalIndexStore;
use super::LocalStoreLimits;
use super::MANIFEST;
use super::PENDING;
use super::artifact;
use crate::IndexArtifact;
use crate::IndexMetadata;
use crate::IndexStore;
use crate::Snapshot;
use crate::SourceFile;

const LIMITS: LocalStoreLimits = LocalStoreLimits {
    max_artifact_bytes: 1024 * 1024,
    max_manifest_bytes: 16 * 1024,
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Fault {
    ChunkCopied,
    FileWritten,
    FileSynced,
    ManifestInstalled,
}

thread_local! {
    static FAULT: Cell<Option<Fault>> = const { Cell::new(None) };
}

pub(super) fn inject_fault(point: Fault) -> VortexResult<()> {
    if FAULT.get() == Some(point) {
        FAULT.set(None);
        return Err(vortex_err!("Injected local store IO failure"));
    }
    Ok(())
}

fn metadata(artifacts: Vec<IndexArtifact>) -> IndexMetadata {
    IndexMetadata {
        format_version: 1,
        name: "vectors".into(),
        generation: "gen-1".into(),
        backend: "test.persisted".into(),
        backend_version: 1,
        snapshot: Snapshot {
            dataset_id: "dataset".into(),
            version: "snapshot-1".into(),
            schema_fingerprint: "schema-1".into(),
            files: vec![SourceFile {
                id: 10,
                uri: "test:source".into(),
                version: "source-1".into(),
                row_count: 3,
            }],
        },
        fields: vec!["embedding".into()],
        covered_files: vec![10],
        artifacts,
    }
}

fn descriptor(metadata: &IndexMetadata) -> VortexResult<LocalGeneration> {
    Ok(LocalGeneration {
        generation: metadata.generation.clone(),
        manifest: artifact(
            MANIFEST,
            &serde_json::to_vec(metadata).map_err(|err| vortex_err!("{}", err))?,
        )?,
    })
}

fn assert_error<T>(result: VortexResult<T>, message: &str) -> VortexResult<()> {
    let error = result
        .err()
        .ok_or_else(|| vortex_err!("Expected error: {}", message))?;
    assert!(error.to_string().contains(message), "{error}");
    Ok(())
}

#[test]
fn test_write_seal_and_reopen() -> VortexResult<()> {
    block_on(async {
        let dir = tempfile::tempdir()?;
        let root = dir.path().canonicalize()?;
        let store = LocalIndexStore::create(&root, "gen-1", LIMITS)?;
        let first = store
            .write("native/first.bin", Bytes::from_static(b"abc"))
            .await?;
        let second = store.write("native/second.bin", Bytes::new()).await?;
        assert_eq!(
            first.checksum,
            "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(store.read(&first).await?, b"abc"[..]);
        assert_eq!(store.read(&second).await?, b""[..]);
        assert!(
            store
                .write(&first.path, Bytes::from_static(b"new"))
                .await
                .is_err()
        );
        let metadata = metadata(vec![second, first.clone()]);
        let sealed = store.seal(&metadata)?;
        assert_eq!(sealed, descriptor(&metadata)?);
        assert!(!root.join("gen-1").join(PENDING).exists());
        assert!(store.seal(&metadata).is_err());
        assert!(store.write("third", Bytes::new()).await.is_err());
        drop(store);
        let (opened, actual) = LocalIndexStore::open(&root, &sealed, &metadata.snapshot, LIMITS)?;
        assert_eq!(actual, metadata);
        assert_eq!(opened.read(&first).await?, b"abc"[..]);
        assert!(opened.write("third", Bytes::new()).await.is_err());
        assert!(LocalIndexStore::create(&root, "gen-1", LIMITS).is_err());
        Ok(())
    })
}

#[test]
fn test_seal_requires_exact_complete_inventory() -> VortexResult<()> {
    block_on(async {
        let dir = tempfile::tempdir()?;
        let root = dir.path().canonicalize()?;
        let store = LocalIndexStore::create(&root, "gen-1", LIMITS)?;
        let first = store.write("first", Bytes::from_static(b"abc")).await?;
        let second = store.write("second", Bytes::from_static(b"def")).await?;
        let valid = metadata(vec![first.clone(), second]);
        let mut cases = vec![valid.clone(); 5];
        cases[0].artifacts.pop();
        cases[1].artifacts.push(first.clone());
        cases[2].artifacts[0].checksum = artifact("first", b"xyz")?.checksum;
        cases[3].generation = "another".into();
        cases[4].artifacts[0].size += 1;
        for invalid in cases {
            assert!(store.seal(&invalid).is_err());
            assert!(!root.join("gen-1").join(MANIFEST).exists());
        }
        let mut unknown = first;
        unknown.path = "unknown".into();
        assert!(store.read(&unknown).await.is_err());
        store.seal(&valid)?;
        Ok(())
    })
}

#[test]
fn test_reopen_requires_trusted_manifest_and_snapshot() -> VortexResult<()> {
    let dir = tempfile::tempdir()?;
    let root = dir.path().canonicalize()?;
    let store = LocalIndexStore::create(&root, "gen-1", LIMITS)?;
    let metadata = metadata(vec![]);
    let sealed = store.seal(&metadata)?;
    let mut snapshot = metadata.snapshot.clone();
    snapshot.version = "snapshot-2".into();
    assert_error(
        LocalIndexStore::open(&root, &sealed, &snapshot, LIMITS),
        "snapshot",
    )?;
    let mut invalid = sealed.clone();
    invalid.manifest.checksum = "sha256:invalid".into();
    assert_error(
        LocalIndexStore::open(&root, &invalid, &metadata.snapshot, LIMITS),
        "checksum",
    )?;
    invalid = sealed.clone();
    invalid.manifest.path = "../manifest.json".into();
    assert!(LocalIndexStore::open(&root, &invalid, &metadata.snapshot, LIMITS).is_err());
    fs::rename(root.join("gen-1"), root.join("gen-2"))?;
    invalid = sealed;
    invalid.generation = "gen-2".into();
    assert_error(
        LocalIndexStore::open(&root, &invalid, &metadata.snapshot, LIMITS),
        "generation",
    )?;
    Ok(())
}

#[test]
fn test_missing_truncated_corrupt_and_growing_artifacts() -> VortexResult<()> {
    block_on(async {
        let dir = tempfile::tempdir()?;
        let root = dir.path().canonicalize()?;
        let store = LocalIndexStore::create(&root, "gen-1", LIMITS)?;
        let artifact = store.write("vectors", Bytes::from_static(b"abc")).await?;
        let metadata = metadata(vec![artifact.clone()]);
        let sealed = store.seal(&metadata)?;
        let path = root.join("gen-1/artifacts/vectors");
        for bytes in [b"ab".as_slice(), b"xyz".as_slice(), b"abcd".as_slice()] {
            fs::write(&path, bytes)?;
            assert!(store.read(&artifact).await.is_err());
            assert!(LocalIndexStore::open(&root, &sealed, &metadata.snapshot, LIMITS).is_err());
        }
        fs::remove_file(&path)?;
        assert!(store.read(&artifact).await.is_err());
        assert!(LocalIndexStore::open(&root, &sealed, &metadata.snapshot, LIMITS).is_err());
        fs::create_dir(&path)?;
        assert_error(store.read(&artifact).await, "not a regular file")?;
        Ok(())
    })
}

#[test]
fn test_corrupt_manifest_is_not_adopted() -> VortexResult<()> {
    let dir = tempfile::tempdir()?;
    let root = dir.path().canonicalize()?;
    let store = LocalIndexStore::create(&root, "gen-1", LIMITS)?;
    let metadata = metadata(vec![]);
    let sealed = store.seal(&metadata)?;
    let path = root.join("gen-1").join(MANIFEST);
    let mut changed = metadata.clone();
    changed.name = "changed".into();
    fs::write(
        &path,
        serde_json::to_vec(&changed).map_err(|err| vortex_err!("{}", err))?,
    )?;
    assert!(LocalIndexStore::open(&root, &sealed, &metadata.snapshot, LIMITS).is_err());
    fs::write(&path, b"{invalid")?;
    assert!(LocalIndexStore::open(&root, &sealed, &metadata.snapshot, LIMITS).is_err());
    fs::remove_file(&path)?;
    assert!(LocalIndexStore::open(&root, &sealed, &metadata.snapshot, LIMITS).is_err());
    Ok(())
}

#[test]
fn test_manifest_parser_and_metadata_validation_after_digest_check() -> VortexResult<()> {
    let dir = tempfile::tempdir()?;
    let root = dir.path().canonicalize()?;
    let store = LocalIndexStore::create(&root, "gen-1", LIMITS)?;
    let valid = metadata(vec![]);
    let mut sealed = store.seal(&valid)?;
    let path = root.join("gen-1").join(MANIFEST);
    let mut unsupported = valid.clone();
    unsupported.format_version += 1;
    let mut duplicate_fields = valid.clone();
    duplicate_fields.fields.push("embedding".into());
    let mut wrong_generation = valid.clone();
    wrong_generation.generation = "other".into();
    let mut invalid_path = valid.clone();
    invalid_path.artifacts.push(artifact("../escape", b"abc")?);
    let cases = [
        unsupported,
        duplicate_fields,
        wrong_generation,
        invalid_path,
    ]
    .into_iter()
    .map(|metadata| serde_json::to_vec(&metadata).map_err(|err| vortex_err!("{}", err)))
    .collect::<VortexResult<Vec<_>>>()?;
    for bytes in cases.into_iter().chain([b"{invalid-json".to_vec()]) {
        fs::write(&path, &bytes)?;
        sealed.manifest = artifact(MANIFEST, &bytes)?;
        assert!(LocalIndexStore::open(&root, &sealed, &valid.snapshot, LIMITS).is_err());
    }
    Ok(())
}

#[test]
fn test_seal_rechecks_artifact_content() -> VortexResult<()> {
    block_on(async {
        let dir = tempfile::tempdir()?;
        let root = dir.path().canonicalize()?;
        let store = LocalIndexStore::create(&root, "gen-1", LIMITS)?;
        let artifact = store.write("data", Bytes::from_static(b"abc")).await?;
        fs::write(root.join("gen-1/artifacts/data"), b"xyz")?;
        assert_error(store.seal(&metadata(vec![artifact])), "checksum mismatch")?;
        assert!(!root.join("gen-1").join(MANIFEST).exists());
        assert!(store.write("other", Bytes::new()).await.is_err());
        Ok(())
    })
}

#[test]
fn test_byte_limits_on_write_seal_and_open() -> VortexResult<()> {
    block_on(async {
        let dir = tempfile::tempdir()?;
        let root = dir.path().canonicalize()?;
        let limits = LocalStoreLimits {
            max_artifact_bytes: 3,
            ..LIMITS
        };
        let store = LocalIndexStore::create(&root, "gen-1", limits)?;
        assert_error(
            store.write("too-big", Bytes::from_static(b"abcd")).await,
            "byte limit",
        )?;
        let first = store.write("first", Bytes::from_static(b"abc")).await?;
        let metadata = metadata(vec![first]);
        let sealed = store.seal(&metadata)?;
        let limits = LocalStoreLimits {
            max_artifact_bytes: 2,
            ..LIMITS
        };
        assert_error(
            LocalIndexStore::open(&root, &sealed, &metadata.snapshot, limits),
            "byte limit",
        )?;
        let limits = LocalStoreLimits {
            max_manifest_bytes: 3,
            ..LIMITS
        };
        assert_error(
            LocalIndexStore::open(&root, &sealed, &metadata.snapshot, limits),
            "byte limit",
        )?;
        let store = LocalIndexStore::create(&root, "gen-2", limits)?;
        let mut metadata = metadata;
        metadata.generation = "gen-2".into();
        metadata.artifacts.clear();
        assert_error(store.seal(&metadata), "byte limit")?;
        assert!(!root.join("gen-2").join(MANIFEST).exists());
        Ok(())
    })
}

#[test]
fn test_rejects_traversal_and_symlinks_at_every_level() -> VortexResult<()> {
    block_on(async {
        let dir = tempfile::tempdir()?;
        let root = dir.path().canonicalize()?;
        let outside = tempfile::tempdir()?;
        let store = LocalIndexStore::create(&root, "gen-1", LIMITS)?;
        for path in [
            "",
            "../escape",
            "/absolute",
            "a/../../b",
            "a//b",
            "a/./b",
            "a\\b",
            "s3:x",
        ] {
            assert!(store.write(path, Bytes::new()).await.is_err());
        }
        for generation in ["../escape", "a/b", ".", "..", "/absolute"] {
            assert!(LocalIndexStore::create(&root, generation, LIMITS).is_err());
        }
        assert!(LocalIndexStore::create("relative", "gen", LIMITS).is_err());
        symlink(outside.path(), root.join("alias"))?;
        assert!(LocalIndexStore::create(root.join("alias"), "gen", LIMITS).is_err());
        symlink(outside.path(), root.join("gen-alias"))?;
        let mut alias = descriptor(&metadata(vec![]))?;
        alias.generation = "gen-alias".into();
        assert!(LocalIndexStore::open(&root, &alias, &metadata(vec![]).snapshot, LIMITS).is_err());
        symlink(outside.path(), root.join("gen-1/artifacts/nested"))?;
        assert!(store.write("nested/escape", Bytes::new()).await.is_err());
        assert!(!outside.path().join("escape").exists());
        assert!(store.seal(&metadata(vec![])).is_err());
        Ok(())
    })
}

#[test]
fn test_leaf_symlink_never_overwrites_or_reads_target() -> VortexResult<()> {
    block_on(async {
        let dir = tempfile::tempdir()?;
        let root = dir.path().canonicalize()?;
        let target = root.join("target");
        fs::write(&target, b"secret")?;
        let store = LocalIndexStore::create(&root, "gen-1", LIMITS)?;
        let artifact = store.write("data", Bytes::from_static(b"secret")).await?;
        let metadata = metadata(vec![artifact.clone()]);
        let sealed = store.seal(&metadata)?;
        fs::remove_file(root.join("gen-1/artifacts/data"))?;
        symlink(&target, root.join("gen-1/artifacts/data"))?;
        assert!(store.read(&artifact).await.is_err());
        assert!(LocalIndexStore::open(&root, &sealed, &metadata.snapshot, LIMITS).is_err());
        let writer = LocalIndexStore::create(&root, "gen-2", LIMITS)?;
        symlink(&target, root.join("gen-2/artifacts/data"))?;
        assert!(
            writer
                .write("data", Bytes::from_static(b"damage"))
                .await
                .is_err()
        );
        assert_eq!(fs::read(&target)?, b"secret");
        Ok(())
    })
}

#[test]
fn test_fifo_read_is_nonblocking() -> VortexResult<()> {
    let dir = tempfile::tempdir()?;
    let root = dir.path().canonicalize()?;
    let store = LocalIndexStore::create(&root, "gen-1", LIMITS)?;
    let artifact = block_on(store.write("fifo", Bytes::new()))?;
    let path = root.join("gen-1/artifacts/fifo");
    fs::remove_file(&path)?;
    assert_rejects_fifo(&path, || block_on(store.read(&artifact)))
}

fn assert_rejects_fifo<T>(path: &Path, read: impl FnOnce() -> VortexResult<T>) -> VortexResult<()> {
    unix_fs::mkfifoat(
        unix_fs::CWD,
        path,
        unix_fs::Mode::RUSR | unix_fs::Mode::WUSR,
    )
    .map_err(|err| vortex_err!("{}", err))?;
    let path = path.to_owned();
    let (done, wait) = mpsc::channel();
    let watchdog = thread::spawn(move || -> VortexResult<bool> {
        if wait.recv_timeout(Duration::from_secs(2)).is_ok() {
            return Ok(false);
        }
        let writer = unix_fs::open(
            &path,
            unix_fs::OFlags::RDWR | unix_fs::OFlags::NONBLOCK,
            unix_fs::Mode::empty(),
        )
        .map_err(|err| vortex_err!("{}", err))?;
        let _received = wait.recv_timeout(Duration::from_secs(2));
        drop(writer);
        Ok(true)
    });
    let result = read();
    let _sent = done.send(());
    let released = watchdog
        .join()
        .map_err(|_| vortex_err!("Watchdog panicked"))??;
    assert!(!released, "FIFO needed a writer to unblock its reader");
    assert_error(result, "not a regular file")
}

#[test]
fn test_failed_write_and_interrupted_seal_are_not_reopenable() -> VortexResult<()> {
    block_on(async {
        let dir = tempfile::tempdir()?;
        let root = dir.path().canonicalize()?;
        let store = LocalIndexStore::create(&root, "gen-1", LIMITS)?;
        // A partial artifact left by an interrupted writer cannot be overwritten.
        fs::write(root.join("gen-1/artifacts/partial"), b"ab")?;
        assert!(
            store
                .write("partial", Bytes::from_static(b"abc"))
                .await
                .is_err()
        );
        assert!(store.write("other", Bytes::new()).await.is_err());
        assert!(store.seal(&metadata(vec![])).is_err());
        assert!(LocalIndexStore::create(&root, "gen-1", LIMITS).is_err());
        let metadata = metadata(vec![]);
        let descriptor = descriptor(&metadata)?;
        // Neither a partial nor a complete pending manifest is a sealed one.
        for bytes in [
            b"{\"format_version\":".to_vec(),
            serde_json::to_vec(&metadata).map_err(|err| vortex_err!("{}", err))?,
        ] {
            fs::write(root.join("gen-1").join(PENDING), bytes)?;
            assert!(LocalIndexStore::open(&root, &descriptor, &metadata.snapshot, LIMITS).is_err());
        }
        Ok(())
    })
}

mod local_files;

#[test]
fn test_manifest_install_does_not_clobber_and_poisoned_writer_cannot_retry() -> VortexResult<()> {
    let dir = tempfile::tempdir()?;
    let root = dir.path().canonicalize()?;
    let store = LocalIndexStore::create(&root, "gen-1", LIMITS)?;
    let path = root.join("gen-1").join(MANIFEST);
    fs::write(&path, b"existing")?;
    assert!(store.seal(&metadata(vec![])).is_err());
    assert_eq!(fs::read(&path)?, b"existing");
    fs::remove_file(&path)?;
    assert!(store.seal(&metadata(vec![])).is_err());
    Ok(())
}

#[test]
fn test_io_failures_poison_writer_and_never_return_a_sealed_descriptor() -> VortexResult<()> {
    block_on(async {
        for point in [Fault::FileWritten, Fault::FileSynced] {
            let dir = tempfile::tempdir()?;
            let root = dir.path().canonicalize()?;
            let store = LocalIndexStore::create(&root, "gen-1", LIMITS)?;
            FAULT.set(Some(point));
            assert_error(
                store.write("data", Bytes::from_static(b"abc")).await,
                "Injected",
            )?;
            assert!(store.seal(&metadata(vec![])).is_err());
            assert!(store.write("other", Bytes::new()).await.is_err());
            assert!(!root.join("gen-1").join(MANIFEST).exists());
        }
        for point in [
            Fault::FileWritten,
            Fault::FileSynced,
            Fault::ManifestInstalled,
        ] {
            let dir = tempfile::tempdir()?;
            let root = dir.path().canonicalize()?;
            let store = LocalIndexStore::create(&root, "gen-1", LIMITS)?;
            let artifact = store.write("data", Bytes::from_static(b"abc")).await?;
            let metadata = metadata(vec![artifact]);
            FAULT.set(Some(point));
            assert_error(store.seal(&metadata), "Injected")?;
            assert!(store.seal(&metadata).is_err());
            assert!(store.write("other", Bytes::new()).await.is_err());
            let open =
                LocalIndexStore::open(&root, &descriptor(&metadata)?, &metadata.snapshot, LIMITS);
            if point == Fault::ManifestInstalled {
                // Complete but unreferenced: seal returned no descriptor to publish.
                open?;
            } else {
                assert!(open.is_err());
            }
        }
        Ok(())
    })
}

#[test]
fn test_concurrent_creation_and_duplicate_writes_have_one_winner() -> VortexResult<()> {
    let dir = tempfile::tempdir()?;
    let root = dir.path().canonicalize()?;
    let barrier = Arc::new(Barrier::new(2));
    let handles: Vec<_> = (0..2)
        .map(|_| {
            let root = root.clone();
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                LocalIndexStore::create(root, "gen-1", LIMITS)
            })
        })
        .collect();
    let mut winners = Vec::new();
    for handle in handles {
        if let Ok(store) = handle.join().map_err(|_| vortex_err!("Creator panicked"))? {
            winners.push(store);
        }
    }
    assert_eq!(winners.len(), 1);
    let store = Arc::new(winners.pop().ok_or_else(|| vortex_err!("Missing writer"))?);
    let handles: Vec<_> = (0..2)
        .map(|_| {
            let store = Arc::clone(&store);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                block_on(store.write("data", Bytes::from_static(b"abc")))
            })
        })
        .collect();
    let mut artifacts = Vec::new();
    for handle in handles {
        if let Ok(artifact) = handle.join().map_err(|_| vortex_err!("Writer panicked"))? {
            artifacts.push(artifact);
        }
    }
    assert_eq!(artifacts.len(), 1);
    store.seal(&metadata(artifacts))?;
    Ok(())
}

#[test]
fn test_write_and_seal_are_serialized() -> VortexResult<()> {
    for _ in 0..16 {
        let dir = tempfile::tempdir()?;
        let root = dir.path().canonicalize()?;
        let store = Arc::new(LocalIndexStore::create(&root, "gen-1", LIMITS)?);
        let barrier = Arc::new(Barrier::new(2));
        let writer = {
            let store = Arc::clone(&store);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                block_on(store.write("data", Bytes::from_static(b"abc")))
            })
        };
        barrier.wait();
        let sealed = store.seal(&metadata(vec![]));
        let written = writer.join().map_err(|_| vortex_err!("Writer panicked"))?;
        let (descriptor, expected) = match (sealed, written) {
            (Ok(descriptor), Err(_)) => (descriptor, metadata(vec![])),
            (Err(_), Ok(artifact)) => {
                let expected = metadata(vec![artifact]);
                (store.seal(&expected)?, expected)
            }
            _ => return Err(vortex_err!("Write and empty seal did not serialize")),
        };
        let (_, actual) = LocalIndexStore::open(&root, &descriptor, &expected.snapshot, LIMITS)?;
        assert_eq!(actual, expected);
    }
    Ok(())
}
