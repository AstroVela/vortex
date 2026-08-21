// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Resolve which physical block device backs a benchmark path.
//!
//! Benchmarks tagged [`STORAGE_NVME`](super::constants::STORAGE_NVME) assume their data sits on
//! machine-local NVMe (an EC2 instance store), not on a network-attached volume such as EBS.
//! Nothing in the benchmark harness pins the data directory to a particular device: it lives
//! inside the checkout, so the answer depends entirely on where the CI runner's work directory
//! happens to be mounted. [`describe_path`] resolves that at runtime so every run records the
//! device it actually read from.

use std::fmt::Display;
use std::fmt::Formatter;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Once;

use serde::Deserialize;
use serde::Serialize;
use tracing::debug;
use tracing::info;
use tracing::warn;

use super::file::data_dir;

/// AWS reports the instance store and EBS through the NVMe `model` field.
const MODEL_INSTANCE_STORE: &str = "Amazon EC2 NVMe Instance Storage";
const MODEL_EBS: &str = "Amazon Elastic Block Store";

/// Coarse classification of the storage backing a path.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum StorageKind {
    /// Machine-local NVMe, e.g. an EC2 instance store on a `*d` instance family.
    LocalNvme,
    /// Network-attached block storage (EBS).
    NetworkBlockStore,
    /// In-memory filesystem (`tmpfs`, `ramfs`).
    Memory,
    /// A real device that is neither of the above, or one we could not classify.
    Unknown,
}

impl Display for StorageKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            StorageKind::LocalNvme => "local-nvme",
            StorageKind::NetworkBlockStore => "network-block-store",
            StorageKind::Memory => "memory",
            StorageKind::Unknown => "unknown",
        };
        f.write_str(s)
    }
}

/// A leaf block device backing a path, with the sysfs attributes used to classify it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockDevice {
    /// Kernel name, e.g. `nvme0n1`.
    pub name: String,
    /// Contents of `/sys/block/<name>/device/model`, when readable.
    pub model: Option<String>,
    /// Contents of `/sys/block/<name>/queue/rotational`, when readable.
    pub rotational: Option<bool>,
    /// Device size in bytes, derived from `/sys/block/<name>/size` (512-byte sectors).
    pub size_bytes: Option<u64>,
}

/// Where a path physically lives.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StorageInfo {
    /// The path that was inspected.
    pub path: PathBuf,
    /// Mount point covering `path`.
    pub mount_point: PathBuf,
    /// Filesystem type of that mount, e.g. `ext4` or `overlay`.
    pub fs_type: String,
    /// Mount source, e.g. `/dev/md0` or `overlay`.
    pub source: String,
    /// For stacked filesystems, the chain of mount points walked to reach a real device.
    pub resolved_via: Vec<PathBuf>,
    /// Leaf block devices backing the mount, expanded through MD/RAID members.
    pub devices: Vec<BlockDevice>,
    /// Classification derived from `devices`.
    pub kind: StorageKind,
}

impl Display for StorageInfo {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} -> {} on {} ({})",
            self.path.display(),
            self.source,
            self.mount_point.display(),
            self.fs_type
        )?;
        for via in &self.resolved_via {
            write!(f, " via {}", via.display())?;
        }
        if !self.devices.is_empty() {
            let devices = self
                .devices
                .iter()
                .map(|d| match &d.model {
                    Some(model) => format!("{} [{}]", d.name, model),
                    None => d.name.clone(),
                })
                .collect::<Vec<_>>()
                .join(", ");
            write!(f, " backed by {devices}")?;
        }
        write!(f, ", storage={}", self.kind)
    }
}

/// Resolve the storage backing `path`, walking up to the nearest existing ancestor when the
/// path itself has not been created yet.
///
/// Returns `None` when the platform does not expose `/proc/self/mountinfo`, i.e. everywhere
/// except Linux. Failure to classify is reported as [`StorageKind::Unknown`] rather than an
/// error: this is diagnostic output and must never fail a benchmark.
pub fn describe_path(path: &Path) -> Option<StorageInfo> {
    let mounts = read_mountinfo()?;
    let mut info = describe_with_mounts(&existing_ancestor(path), &mounts, 0);
    info.path = path.to_path_buf();
    Some(info)
}

/// Number of stacked filesystems (`overlay` on `overlay` on ...) to walk before giving up.
const MAX_OVERLAY_DEPTH: usize = 4;

fn describe_with_mounts(path: &Path, mounts: &[MountEntry], depth: usize) -> StorageInfo {
    let Some(mount) = longest_prefix_mount(path, mounts) else {
        return StorageInfo {
            path: path.to_path_buf(),
            mount_point: PathBuf::from("/"),
            fs_type: "unknown".to_string(),
            source: "unknown".to_string(),
            resolved_via: Vec::new(),
            devices: Vec::new(),
            kind: StorageKind::Unknown,
        };
    };

    // An overlay's data lives in its upper directory, which is itself on some other mount.
    if depth < MAX_OVERLAY_DEPTH
        && let Some(upper) = mount.upper_dir.as_ref()
    {
        // Resolve the upper directory against the mount table rather than the filesystem: it is
        // often not readable by the benchmark user, and its mount is all we need.
        let mut inner = describe_with_mounts(upper, mounts, depth + 1);
        inner.path = path.to_path_buf();
        inner.resolved_via.insert(0, mount.mount_point.clone());
        return inner;
    }

    let devices = leaf_devices(&mount.source);
    let kind = classify(&mount.fs_type, &devices);
    StorageInfo {
        path: path.to_path_buf(),
        mount_point: mount.mount_point.clone(),
        fs_type: mount.fs_type.clone(),
        source: mount.source.clone(),
        resolved_via: Vec::new(),
        devices,
        kind,
    }
}

fn classify(fs_type: &str, devices: &[BlockDevice]) -> StorageKind {
    if matches!(fs_type, "tmpfs" | "ramfs") {
        return StorageKind::Memory;
    }
    if devices.is_empty() {
        return StorageKind::Unknown;
    }
    let kinds = devices
        .iter()
        .map(|d| match d.model.as_deref() {
            Some(MODEL_INSTANCE_STORE) => StorageKind::LocalNvme,
            Some(MODEL_EBS) => StorageKind::NetworkBlockStore,
            _ => StorageKind::Unknown,
        })
        .collect::<Vec<_>>();
    // A RAID spanning device classes is not one kind of storage; report it as unknown rather
    // than picking whichever member happened to come first.
    if kinds.iter().all(|k| *k == kinds[0]) {
        kinds[0]
    } else {
        StorageKind::Unknown
    }
}

/// Expand a mount source such as `/dev/md0` into the leaf devices it is built from.
fn leaf_devices(source: &str) -> Vec<BlockDevice> {
    let Some(name) = source.strip_prefix("/dev/") else {
        return Vec::new();
    };
    // sysfs uses `!` where the device name contains a slash, e.g. `/dev/dm-0` vs `nvme0n1`.
    let name = name.replace('/', "!");
    let block = Path::new("/sys/block").join(&name);
    if !block.exists() {
        // Partitions live under their parent disk, e.g. /sys/block/nvme0n1/nvme0n1p1.
        if let Some(parent) = partition_parent(&name) {
            return leaf_devices(&format!("/dev/{parent}"));
        }
        return Vec::new();
    }

    // MD arrays list their members under `slaves/`.
    let slaves = block.join("slaves");
    if let Ok(entries) = fs::read_dir(&slaves) {
        let mut members = entries
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        if !members.is_empty() {
            members.sort();
            return members
                .iter()
                .flat_map(|m| leaf_devices(&format!("/dev/{m}")))
                .collect();
        }
    }

    vec![read_device(&name, &block)]
}

fn partition_parent(name: &str) -> Option<String> {
    let link = fs::read_link(Path::new("/sys/class/block").join(name)).ok()?;
    let parent = link.parent()?.file_name()?.to_string_lossy().into_owned();
    (parent != "block" && parent != name).then_some(parent)
}

fn read_device(name: &str, block: &Path) -> BlockDevice {
    let read_trimmed = |rel: &str| {
        fs::read_to_string(block.join(rel))
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    };
    BlockDevice {
        name: name.replace('!', "/"),
        model: read_trimmed("device/model"),
        rotational: read_trimmed("queue/rotational").map(|s| s == "1"),
        size_bytes: read_trimmed("size")
            .and_then(|s| s.parse::<u64>().ok())
            .map(|sectors| sectors * 512),
    }
}

/// Walk up until a component exists, so a not-yet-created data directory still resolves.
fn existing_ancestor(path: &Path) -> PathBuf {
    let mut candidate: &Path = path;
    loop {
        if let Ok(canonical) = candidate.canonicalize() {
            return canonical;
        }
        match candidate.parent() {
            Some(parent) => candidate = parent,
            None => return path.to_path_buf(),
        }
    }
}

#[derive(Clone, Debug)]
struct MountEntry {
    mount_point: PathBuf,
    fs_type: String,
    source: String,
    upper_dir: Option<PathBuf>,
}

fn read_mountinfo() -> Option<Vec<MountEntry>> {
    let raw = fs::read_to_string("/proc/self/mountinfo").ok()?;
    Some(parse_mountinfo(&raw))
}

/// Parse `/proc/self/mountinfo`.
///
/// Fields up to the `-` separator are mount metadata (mount point is field 5); after it come the
/// filesystem type, the mount source, and the super options. Path fields are octal-escaped.
fn parse_mountinfo(raw: &str) -> Vec<MountEntry> {
    raw.lines()
        .filter_map(|line| {
            let (before, after) = line.split_once(" - ")?;
            let mount_point = before.split(' ').nth(4)?;
            let mut after = after.split(' ');
            let fs_type = after.next()?;
            let source = after.next()?;
            let super_options = after.next().unwrap_or("");
            let upper_dir = super_options
                .split(',')
                .find_map(|opt| opt.strip_prefix("upperdir="))
                .map(|dir| PathBuf::from(unescape_octal(dir)));
            Some(MountEntry {
                mount_point: PathBuf::from(unescape_octal(mount_point)),
                fs_type: fs_type.to_string(),
                source: unescape_octal(source),
                upper_dir,
            })
        })
        .collect()
}

/// Decode the `\040`-style octal escapes the kernel uses for spaces, tabs, newlines and `\`.
fn unescape_octal(s: &str) -> String {
    if !s.contains('\\') {
        return s.to_string();
    }
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\'
            && i + 3 < bytes.len()
            && let Ok(code) = u8::from_str_radix(&s[i + 1..i + 4], 8)
        {
            out.push(code as char);
            i += 4;
        } else {
            out.push(bytes[i] as char);
            i += 1;
        }
    }
    out
}

/// The mount covering `path` is the one with the longest mount point that prefixes it. Later
/// entries win ties, since a mount stacked onto the same point shadows the earlier one.
fn longest_prefix_mount<'a>(path: &Path, mounts: &'a [MountEntry]) -> Option<&'a MountEntry> {
    mounts
        .iter()
        .filter(|m| path.starts_with(&m.mount_point))
        .max_by_key(|m| m.mount_point.components().count())
}

/// Log the storage backing the benchmark data directory, once per process.
///
/// Benchmarks that claim to read from local NVMe only do so because the CI runner's work
/// directory happens to sit on the instance store; this records what was actually used, and
/// warns in CI when it is not local NVMe. Call it after logging has been initialized.
pub fn log_data_dir_storage() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let dir = data_dir();
        let Some(info) = describe_path(&dir) else {
            debug!(path = %dir.display(), "benchmark data storage could not be resolved");
            return;
        };
        info!(
            path = %info.path.display(),
            mount_point = %info.mount_point.display(),
            fs_type = %info.fs_type,
            source = %info.source,
            devices = %info
                .devices
                .iter()
                .map(|d| d.name.as_str())
                .collect::<Vec<_>>()
                .join(","),
            storage = %info.kind,
            "benchmark data storage: {info}"
        );
        if std::env::var_os("CI").is_some() && info.kind != StorageKind::LocalNvme {
            warn!(
                storage = %info.kind,
                source = %info.source,
                "benchmark data is not on local NVMe; timings are not comparable to NVMe runs"
            );
        }
    });
}

#[cfg(test)]
mod tests;
