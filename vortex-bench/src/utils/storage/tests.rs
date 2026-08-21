// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::Path;
use std::path::PathBuf;

use rstest::rstest;

use super::*;

/// A trimmed `/proc/self/mountinfo` from a `c6id.metal` benchmark runner: an EBS root, the
/// instance-store RAID at `/mnt/ephemeral`, and the runner work directory overlaid onto it.
const RUNNER_MOUNTINFO: &str = concat!(
    "24 30 259:1 / / rw,relatime shared:1 - ext4 /dev/nvme4n1p1 rw,discard,commit=30\n",
    "31 24 0:23 / /dev/shm rw,nosuid,nodev shared:2 - tmpfs tmpfs rw\n",
    "40 24 9:0 / /mnt/ephemeral rw,noatime shared:3 - ext4 /dev/md0 rw,stripe=256\n",
    "41 24 0:57 / /home/runner rw,relatime shared:4 - overlay overlay ",
    "rw,lowerdir=/home/runner,upperdir=/mnt/ephemeral/overlay/home-runner/upper,",
    "workdir=/mnt/ephemeral/overlay/home-runner/work\n",
    "42 24 0:58 / /mnt/with\\040space rw,relatime shared:5 - ext4 /dev/nvme5n1 rw\n",
);

#[rstest]
#[case::root("/etc/hostname", "/", "ext4", "/dev/nvme4n1p1")]
#[case::ephemeral("/mnt/ephemeral/data", "/mnt/ephemeral", "ext4", "/dev/md0")]
#[case::escaped_mount_point("/mnt/with space/x", "/mnt/with space", "ext4", "/dev/nvme5n1")]
fn resolves_plain_mounts(
    #[case] path: &str,
    #[case] mount_point: &str,
    #[case] fs_type: &str,
    #[case] source: &str,
) {
    let mounts = parse_mountinfo(RUNNER_MOUNTINFO);
    let info = describe_with_mounts(Path::new(path), &mounts, 0);
    assert_eq!(info.mount_point, PathBuf::from(mount_point));
    assert_eq!(info.fs_type, fs_type);
    assert_eq!(info.source, source);
    assert!(info.resolved_via.is_empty());
}

#[test]
fn resolves_overlay_to_its_upper_dir() {
    let mounts = parse_mountinfo(RUNNER_MOUNTINFO);
    let path = Path::new("/home/runner/_work/vortex/vortex/vortex-bench/data");
    let info = describe_with_mounts(path, &mounts, 0);

    // The workspace looks like it lives on the overlay, but its bytes land on the RAID.
    assert_eq!(info.path, path);
    assert_eq!(info.source, "/dev/md0");
    assert_eq!(info.mount_point, PathBuf::from("/mnt/ephemeral"));
    assert_eq!(info.resolved_via, vec![PathBuf::from("/home/runner")]);
}

#[test]
fn longest_mount_point_wins_over_shorter_prefix() {
    let mounts = parse_mountinfo(RUNNER_MOUNTINFO);
    let info = describe_with_mounts(Path::new("/mnt/ephemeral"), &mounts, 0);
    assert_eq!(info.source, "/dev/md0");
}

#[test]
fn stacked_overlays_terminate() {
    let stacked = (0..=MAX_OVERLAY_DEPTH)
        .map(|i| {
            format!(
                "{} 24 0:{i} / /o{i} rw - overlay overlay rw,upperdir=/o{},workdir=/w\n",
                100 + i,
                i + 1
            )
        })
        .collect::<String>();
    let mounts = parse_mountinfo(&stacked);
    let info = describe_with_mounts(Path::new("/o0/x"), &mounts, 0);
    assert_eq!(info.resolved_via.len(), MAX_OVERLAY_DEPTH);
    assert_eq!(info.kind, StorageKind::Unknown);
}

#[rstest]
#[case::instance_store(MODEL_INSTANCE_STORE, StorageKind::LocalNvme)]
#[case::ebs(MODEL_EBS, StorageKind::NetworkBlockStore)]
#[case::other("Samsung SSD 990 PRO", StorageKind::Unknown)]
fn classifies_by_device_model(#[case] model: &str, #[case] expected: StorageKind) {
    let devices = vec![device(model)];
    assert_eq!(classify("ext4", &devices), expected);
}

#[test]
fn classifies_uniform_raid_members_as_one_kind() {
    let devices = vec![device(MODEL_INSTANCE_STORE); 4];
    assert_eq!(classify("ext4", &devices), StorageKind::LocalNvme);
}

#[test]
fn mixed_raid_members_are_not_classified() {
    let devices = vec![device(MODEL_INSTANCE_STORE), device(MODEL_EBS)];
    assert_eq!(classify("ext4", &devices), StorageKind::Unknown);
}

#[rstest]
#[case::tmpfs("tmpfs", StorageKind::Memory)]
#[case::no_devices("ext4", StorageKind::Unknown)]
fn classifies_deviceless_mounts(#[case] fs_type: &str, #[case] expected: StorageKind) {
    assert_eq!(classify(fs_type, &[]), expected);
}

#[test]
fn describes_a_real_path() {
    // Every platform this runs on either exposes mountinfo or is expected to return None; a
    // resolved answer must at least name a mount point that is an ancestor of the path.
    let cwd = std::env::current_dir().unwrap();
    if let Some(info) = describe_path(&cwd) {
        assert!(cwd.starts_with(&info.mount_point) || !info.resolved_via.is_empty());
    } else if cfg!(target_os = "linux") {
        panic!("/proc/self/mountinfo should be readable on Linux");
    }
}

#[test]
fn describes_a_path_that_does_not_exist_yet() {
    let missing = std::env::current_dir()
        .unwrap()
        .join("definitely/not/created/yet");
    if let Some(info) = describe_path(&missing) {
        assert_ne!(info.fs_type, "unknown");
    }
}

fn device(model: &str) -> BlockDevice {
    BlockDevice {
        name: "nvme0n1".to_string(),
        model: Some(model.to_string()),
        rotational: Some(false),
        size_bytes: Some(1_600_000_000_000),
    }
}
