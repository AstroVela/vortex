// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::io;
use std::io::Cursor;
use std::io::Read;
use std::os::unix::fs::MetadataExt;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::fs::symlink;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::sync::Barrier;
use std::thread;

use bytes::Bytes;
use futures::executor::block_on;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use super::FAULT;
use super::Fault;
use super::LIMITS;
use super::artifact;
use super::assert_error;
use super::assert_rejects_fifo;
use super::metadata;
use crate::IndexStore;
use crate::LocalArtifactLease;
use crate::LocalIndexFiles;
use crate::store::COPY_BUFFER_BYTES;
use crate::store::LocalGeneration;
use crate::store::LocalIndexStore;
use crate::store::LocalStoreLimits;
use crate::store::copy_and_hash;

const CHILD: &str = "store::tests::local_files::test_local_files_child";
const CHILD_ROOT: &str = "VORTEX_INDEX_LOCAL_FILES_TEST_ROOT";
const CHILD_MODE: &str = "VORTEX_INDEX_LOCAL_FILES_TEST_MODE";

#[test]
fn test_import_seal_materialize_and_lease_lifetime() -> VortexResult<()> {
    let dir = tempfile::tempdir()?;
    let root = dir.path().canonicalize()?;
    let scratch = root.join("scratch");
    fs::create_dir(&scratch)?;
    let source = root.join("source.bin");
    let data = vec![42; 3 * COPY_BUFFER_BYTES + 17];
    fs::write(&source, &data)?;
    let store = LocalIndexStore::create(&root, "gen-1", LIMITS)?;
    let files = (&store as &dyn IndexStore)
        .as_local_files()
        .ok_or_else(|| vortex_err!("Missing local files capability"))?;
    let imported = files.import_file("native/graph.bin", &source)?;
    assert_eq!(imported, artifact("native/graph.bin", &data)?);
    assert!(files.import_file(&imported.path, &source).is_err());
    fs::write(&source, [])?;
    let empty = files.import_file("empty", &source)?;
    assert_eq!(empty, artifact("empty", &[])?);
    fs::remove_file(&source)?;
    assert_eq!(block_on(store.read(&imported))?, data);
    assert!(files.materialize(&[], &scratch, u64::MAX).is_err());
    let metadata = metadata(vec![imported.clone(), empty]);
    let descriptor = store.seal(&metadata)?;
    assert!(
        files
            .import_file("again", &root.join("gen-1/artifacts/empty"))
            .is_err()
    );
    drop(store);

    let (store, actual) = LocalIndexStore::open(&root, &descriptor, &metadata.snapshot, LIMITS)?;
    assert_eq!(actual, metadata);
    let store: Arc<dyn IndexStore> = Arc::new(store);
    let files = store
        .as_local_files()
        .ok_or_else(|| vortex_err!("Missing local files capability"))?;
    let first = files.materialize(&actual.artifacts, &scratch, imported.size)?;
    let second = files.materialize(&actual.artifacts, &scratch, imported.size)?;
    assert_ne!(first.path(), second.path());
    let copy = first.path().join(&imported.path);
    let canonical = root.join("gen-1/artifacts").join(&imported.path);
    assert_ne!(fs::metadata(&copy)?.ino(), fs::metadata(&canonical)?.ino());
    assert_eq!(fs::metadata(first.path())?.mode() & 0o777, 0o700);
    assert_eq!(fs::metadata(&copy)?.mode() & 0o777, 0o400);
    assert_eq!(fs::read(first.path().join("empty"))?, b"");
    assert_eq!(fs::read(&copy)?, data);

    // A misbehaving native reader can change its own copy, not another lease
    // or the canonical generation. Permissions alone are not a same-user sandbox.
    fs::set_permissions(&copy, fs::Permissions::from_mode(0o600))?;
    fs::write(&copy, b"changed by native reader")?;
    assert_eq!(fs::read(&canonical)?, data);
    assert_eq!(fs::read(second.path().join(&imported.path))?, data);
    let first_path = first.path().to_owned();
    drop(first);
    assert!(!first_path.exists());

    fs::write(&canonical, b"changed after opening")?;
    assert_eq!(fs::read(second.path().join(&imported.path))?, data);
    assert!(
        files
            .materialize(&actual.artifacts, &scratch, imported.size)
            .is_err()
    );

    drop(store);
    fs::remove_dir_all(root.join("gen-1"))?;
    let second: Arc<dyn LocalArtifactLease> = second.into();
    let native_owner = Arc::clone(&second);
    let second_path = second.path().to_owned();
    drop(second);
    assert_eq!(fs::read(native_owner.path().join(&imported.path))?, data);
    drop(native_owner);
    assert!(!second_path.exists());
    assert_eq!(fs::read_dir(&scratch)?.count(), 0);
    Ok(())
}

#[test]
fn test_import_preflight_rejects_invalid_sources_paths_and_limits() -> VortexResult<()> {
    let dir = tempfile::tempdir()?;
    let root = dir.path().canonicalize()?;
    fs::create_dir(root.join("input"))?;
    let source = root.join("input/file");
    fs::write(&source, b"abc")?;
    symlink(&source, root.join("link"))?;
    symlink(root.join("input"), root.join("alias"))?;
    let store = LocalIndexStore::create(
        &root,
        "gen-1",
        LocalStoreLimits {
            max_artifact_bytes: 3,
            ..LIMITS
        },
    )?;
    for invalid in [
        Path::new("relative").to_owned(),
        Path::new("/").to_owned(),
        root.join("missing"),
        root.join("input"),
        root.join("link"),
        root.join("alias/file"),
        root.join("input/../input/file"),
    ] {
        assert!(
            store.import_file("native/file", &invalid).is_err(),
            "{invalid:?}"
        );
    }
    for invalid in [
        "",
        "../escape",
        "/absolute",
        "a//b",
        "a/./b",
        "a\\b",
        "s3:x",
    ] {
        assert!(store.import_file(invalid, &source).is_err(), "{invalid}");
    }
    fs::write(&source, b"abcd")?;
    assert_error(store.import_file("large", &source), "byte limit")?;
    assert_eq!(fs::read_dir(root.join("gen-1/artifacts"))?.count(), 0);
    fs::write(&source, b"abc")?;
    let imported = store.import_file("native/file", &source)?;
    store.seal(&metadata(vec![imported]))?;
    Ok(())
}

#[test]
fn test_import_destination_no_clobber_and_symlink_confinement() -> VortexResult<()> {
    for kind in ["existing", "leaf-link", "parent-link"] {
        let dir = tempfile::tempdir()?;
        let root = dir.path().canonicalize()?;
        let source = root.join("source");
        let outside = root.join("outside");
        fs::create_dir(&outside)?;
        fs::write(outside.join("data"), b"keep")?;
        fs::write(&source, b"new")?;
        let store = LocalIndexStore::create(&root, "gen-1", LIMITS)?;
        let destination = root.join("gen-1/artifacts/data");
        let path = match kind {
            "existing" => {
                fs::write(&destination, b"keep")?;
                "data"
            }
            "leaf-link" => {
                symlink(outside.join("data"), &destination)?;
                "data"
            }
            _ => {
                symlink(&outside, root.join("gen-1/artifacts/nested"))?;
                "nested/data"
            }
        };
        assert!(store.import_file(path, &source).is_err());
        assert_eq!(fs::read(outside.join("data"))?, b"keep");
        if kind == "existing" {
            assert_eq!(fs::read(&destination)?, b"keep");
        }
        assert!(store.import_file("retry", &source).is_err());
        assert!(store.seal(&metadata(vec![])).is_err());
    }
    Ok(())
}

#[test]
fn test_import_and_materialize_reject_fifos_without_waiting() -> VortexResult<()> {
    let dir = tempfile::tempdir()?;
    let root = dir.path().canonicalize()?;
    let store = LocalIndexStore::create(&root, "gen-1", LIMITS)?;
    let source = root.join("source-fifo");
    assert_rejects_fifo(&source, || store.import_file("unused", &source))?;
    let artifact = block_on(store.write("fifo", Bytes::new()))?;
    store.seal(&metadata(vec![artifact.clone()]))?;
    let scratch = root.join("scratch");
    fs::create_dir(&scratch)?;
    let path = root.join("gen-1/artifacts/fifo");
    fs::remove_file(&path)?;
    assert_rejects_fifo(&path, || store.materialize(&[artifact], &scratch, 0))?;
    assert_eq!(fs::read_dir(&scratch)?.count(), 0);
    Ok(())
}

#[test]
fn test_import_io_failures_poison_writer() -> VortexResult<()> {
    for fault in [Fault::ChunkCopied, Fault::FileWritten, Fault::FileSynced] {
        let dir = tempfile::tempdir()?;
        let root = dir.path().canonicalize()?;
        let source = root.join("source");
        fs::write(&source, vec![42; COPY_BUFFER_BYTES + 1])?;
        let store = LocalIndexStore::create(&root, "gen-1", LIMITS)?;
        FAULT.set(Some(fault));
        let result = store.import_file("native/data", &source);
        FAULT.set(None);
        assert_error(result, "Injected")?;
        assert!(store.import_file("retry", &source).is_err());
        assert!(block_on(store.write("retry", Bytes::new())).is_err());
        assert!(store.seal(&metadata(vec![])).is_err());
        assert!(store.materialize(&[], &root, 0).is_err());
        assert!(!root.join("gen-1/manifest.json").exists());
        if fault == Fault::ChunkCopied {
            assert_eq!(
                fs::metadata(root.join("gen-1/artifacts/native/data"))?.len(),
                COPY_BUFFER_BYTES as u64
            );
        }
    }
    Ok(())
}

#[test]
fn test_materialize_validates_inventory_and_total_budget_before_copying() -> VortexResult<()> {
    let dir = tempfile::tempdir()?;
    let root = dir.path().canonicalize()?;
    let scratch = root.join("scratch");
    fs::create_dir(&scratch)?;
    let store = LocalIndexStore::create(&root, "gen-1", LIMITS)?;
    let first = block_on(store.write("first", Bytes::from_static(b"abc")))?;
    let second = block_on(store.write("second", Bytes::from_static(b"def")))?;
    let metadata = metadata(vec![first.clone(), second]);
    store.seal(&metadata)?;
    let mut invalid = vec![first.clone(); 4];
    invalid[0].path = "unknown".into();
    invalid[1].path = "../escape".into();
    invalid[2].size += 1;
    invalid[3].checksum = artifact("first", b"bad")?.checksum;
    for artifact in invalid {
        assert_error(
            store.materialize(&[artifact], &scratch, u64::MAX),
            "inventory",
        )?;
    }
    assert_error(
        store.materialize(&[first.clone(), first.clone()], &scratch, u64::MAX),
        "Duplicate",
    )?;
    assert_error(
        store.materialize(&metadata.artifacts, &scratch, 5),
        "total byte limit",
    )?;
    assert_eq!(fs::read_dir(&scratch)?.count(), 0);
    let subset = store.materialize(&[first], &scratch, 3)?;
    assert!(!subset.path().join("second").exists());
    assert_eq!(fs::read(subset.path().join("first"))?, b"abc");
    let empty = store.materialize(&[], &scratch, 0)?;
    assert_eq!(fs::read_dir(empty.path())?.count(), 0);
    drop((subset, empty));
    assert_eq!(fs::read_dir(&scratch)?.count(), 0);
    Ok(())
}

#[test]
fn test_materialize_corruption_and_copy_failures_remove_partial_leases() -> VortexResult<()> {
    let dir = tempfile::tempdir()?;
    let root = dir.path().canonicalize()?;
    let scratch = root.join("scratch");
    fs::create_dir(&scratch)?;
    let store = LocalIndexStore::create(&root, "gen-1", LIMITS)?;
    let good = block_on(store.write("nested/good", Bytes::from_static(b"abc")))?;
    let bad = block_on(store.write("bad", Bytes::from_static(b"def")))?;
    let metadata = metadata(vec![good, bad]);
    store.seal(&metadata)?;
    let path = root.join("gen-1/artifacts/bad");
    for bytes in [b"x".as_slice(), b"bad", b"grow"] {
        fs::write(&path, bytes)?;
        assert!(store.materialize(&metadata.artifacts, &scratch, 6).is_err());
        assert_eq!(fs::read_dir(&scratch)?.count(), 0);
    }
    fs::remove_file(&path)?;
    assert!(store.materialize(&metadata.artifacts, &scratch, 6).is_err());
    fs::create_dir(&path)?;
    assert!(store.materialize(&metadata.artifacts, &scratch, 6).is_err());
    fs::remove_dir(&path)?;
    fs::write(root.join("outside"), b"def")?;
    symlink(root.join("outside"), &path)?;
    assert!(store.materialize(&metadata.artifacts, &scratch, 6).is_err());
    fs::remove_file(&path)?;
    fs::write(&path, b"def")?;
    FAULT.set(Some(Fault::ChunkCopied));
    let result = store.materialize(&metadata.artifacts, &scratch, 6);
    FAULT.set(None);
    assert_error(result, "Injected")?;
    assert_eq!(fs::read_dir(&scratch)?.count(), 0);
    let lease = store.materialize(&metadata.artifacts, &scratch, 6)?;
    assert_eq!(fs::read(lease.path().join("bad"))?, b"def");
    drop(lease);
    assert_eq!(fs::read_dir(&scratch)?.count(), 0);
    Ok(())
}

#[test]
fn test_materialize_rejects_scratch_symlinks_and_parent_components() -> VortexResult<()> {
    let dir = tempfile::tempdir()?;
    let root = dir.path().canonicalize()?;
    let outside = tempfile::tempdir()?;
    fs::create_dir(outside.path().join("nested"))?;
    symlink(outside.path(), root.join("alias"))?;
    let store = LocalIndexStore::create(&root, "gen-1", LIMITS)?;
    store.seal(&metadata(vec![]))?;
    for path in [
        root.join("alias"),
        root.join("alias/nested"),
        root.join("gen-1/.."),
        Path::new("relative").to_owned(),
    ] {
        assert!(store.materialize(&[], &path, 0).is_err(), "{path:?}");
    }
    assert_eq!(fs::read_dir(outside.path())?.count(), 1);
    assert_eq!(fs::read_dir(outside.path().join("nested"))?.count(), 0);
    Ok(())
}

#[test]
fn test_concurrent_materializations_have_independent_lifetimes() -> VortexResult<()> {
    let dir = tempfile::tempdir()?;
    let root = dir.path().canonicalize()?;
    let scratch = root.join("scratch");
    fs::create_dir(&scratch)?;
    let store = Arc::new(LocalIndexStore::create(&root, "gen-1", LIMITS)?);
    let artifact = block_on(store.write("data", Bytes::from_static(b"abc")))?;
    store.seal(&metadata(vec![artifact.clone()]))?;
    let workers: Vec<_> = (0..8)
        .map(|_| {
            let store = Arc::clone(&store);
            let artifact = artifact.clone();
            let scratch = scratch.clone();
            thread::spawn(move || store.materialize(&[artifact], &scratch, 3))
        })
        .collect();
    let leases = workers
        .into_iter()
        .map(|worker| {
            worker
                .join()
                .map_err(|_| vortex_err!("Materializer panicked"))?
        })
        .collect::<VortexResult<Vec<_>>>()?;
    let paths: BTreeSet<_> = leases.iter().map(|lease| lease.path()).collect();
    assert_eq!(paths.len(), 8);
    for lease in &leases {
        assert_eq!(fs::read(lease.path().join("data"))?, b"abc");
    }
    drop(leases);
    assert_eq!(fs::read_dir(&scratch)?.count(), 0);
    assert_eq!(block_on(store.read(&artifact))?, b"abc"[..]);
    Ok(())
}

#[test]
fn test_import_and_seal_are_serialized() -> VortexResult<()> {
    for _ in 0..16 {
        let dir = tempfile::tempdir()?;
        let root = dir.path().canonicalize()?;
        let source = root.join("source");
        fs::write(&source, vec![42; COPY_BUFFER_BYTES * 2 + 1])?;
        let store = Arc::new(LocalIndexStore::create(&root, "gen-1", LIMITS)?);
        let barrier = Arc::new(Barrier::new(2));
        let writer = {
            let store = Arc::clone(&store);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                store.import_file("native/data", &source)
            })
        };
        barrier.wait();
        let sealed = store.seal(&metadata(vec![]));
        let written = writer
            .join()
            .map_err(|_| vortex_err!("Importer panicked"))?;
        let (descriptor, expected) = match (sealed, written) {
            (Ok(descriptor), Err(_)) => (descriptor, metadata(vec![])),
            (Err(_), Ok(artifact)) => {
                let expected = metadata(vec![artifact]);
                (store.seal(&expected)?, expected)
            }
            _ => vortex_bail!("Import and empty seal did not serialize"),
        };
        let (_, actual) = LocalIndexStore::open(&root, &descriptor, &expected.snapshot, LIMITS)?;
        assert_eq!(actual, expected);
    }
    Ok(())
}

#[test]
fn test_streaming_copy_is_bounded_and_handles_short_interrupted_reads() -> VortexResult<()> {
    struct Reader<'a> {
        data: Cursor<&'a [u8]>,
        interrupted: bool,
    }
    impl Read for Reader<'_> {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            assert!(buffer.len() <= COPY_BUFFER_BYTES);
            if !self.interrupted {
                self.interrupted = true;
                return Err(io::ErrorKind::Interrupted.into());
            }
            let count = buffer.len().min(113);
            self.data.read(&mut buffer[..count])
        }
    }
    let data = vec![42; COPY_BUFFER_BYTES * 3 + 17];
    let mut reader = Reader {
        data: Cursor::new(&data),
        interrupted: false,
    };
    let mut consumed = 0;
    let mut calls = 0;
    let digest = copy_and_hash(&mut reader, data.len() as u64, |chunk| {
        assert!(chunk.len() <= COPY_BUFFER_BYTES);
        assert_eq!(chunk, &data[consumed..consumed + chunk.len()]);
        consumed += chunk.len();
        calls += 1;
        Ok(())
    })?;
    assert_eq!(digest, artifact("data", &data)?.checksum);
    assert_eq!(consumed, data.len());
    assert_eq!(calls, 4);
    Ok(())
}

#[test]
fn test_streaming_copy_rejects_truncation_growth_and_consumer_errors() -> VortexResult<()> {
    for size in [0, 2, 4] {
        assert!(copy_and_hash(&mut b"abc".as_slice(), size, |_| Ok(())).is_err());
    }
    assert_eq!(
        copy_and_hash(&mut b"".as_slice(), 0, |_| Ok(()))?,
        artifact("empty", &[])?.checksum
    );
    assert_error(
        copy_and_hash(&mut b"abc".as_slice(), 3, |_| {
            Err(vortex_err!("consumer failed"))
        }),
        "consumer failed",
    )?;
    Ok(())
}

fn run_child(root: &Path, mode: &str) -> VortexResult<()> {
    let output = Command::new(env::current_exe()?)
        .args(["--exact", CHILD, "--nocapture"])
        .env(CHILD_ROOT, root)
        .env(CHILD_MODE, mode)
        .output()?;
    if !output.status.success() {
        vortex_bail!(
            "Local files child {} failed: {}\n{}",
            mode,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(())
}

#[test]
fn test_native_files_build_exit_and_reopen_in_separate_processes() -> VortexResult<()> {
    let dir = tempfile::tempdir()?;
    let root = dir.path().canonicalize()?;
    fs::create_dir(root.join("scratch"))?;
    fs::write(root.join("source"), vec![42; COPY_BUFFER_BYTES * 3 + 17])?;
    run_child(&root, "build")?;
    fs::remove_file(root.join("source"))?;
    run_child(&root, "read")?;
    run_child(&root, "read")?;
    assert_eq!(fs::read_dir(root.join("scratch"))?.count(), 0);
    Ok(())
}

#[test]
fn test_local_files_child() -> VortexResult<()> {
    let Some(root) = env::var_os(CHILD_ROOT) else {
        return Ok(());
    };
    let root = Path::new(&root);
    let mode = env::var(CHILD_MODE).map_err(|err| vortex_err!("{}", err))?;
    match mode.as_str() {
        "build" => {
            let store = LocalIndexStore::create(root, "gen-1", LIMITS)?;
            let files = (&store as &dyn IndexStore)
                .as_local_files()
                .ok_or_else(|| vortex_err!("Missing local files"))?;
            let imported = files.import_file("native/graph", &root.join("source"))?;
            let descriptor = store.seal(&metadata(vec![imported]))?;
            fs::write(
                root.join("descriptor.json"),
                serde_json::to_vec(&descriptor).map_err(|err| vortex_err!("{}", err))?,
            )?;
        }
        "read" => {
            let descriptor: LocalGeneration =
                serde_json::from_slice(&fs::read(root.join("descriptor.json"))?)
                    .map_err(|err| vortex_err!("{}", err))?;
            let (store, metadata) =
                LocalIndexStore::open(root, &descriptor, &metadata(vec![]).snapshot, LIMITS)?;
            let store: Arc<dyn IndexStore> = Arc::new(store);
            let lease = store
                .as_local_files()
                .ok_or_else(|| vortex_err!("Missing local files"))?
                .materialize(
                    &metadata.artifacts,
                    &root.join("scratch"),
                    LIMITS.max_artifact_bytes as u64,
                )?;
            drop(store);
            assert_eq!(
                fs::read(lease.path().join("native/graph"))?,
                vec![42; COPY_BUFFER_BYTES * 3 + 17]
            );
            let path = lease.path().to_owned();
            drop(lease);
            assert!(!path.exists());
        }
        _ => vortex_bail!("Unknown local files child mode"),
    }
    Ok(())
}
