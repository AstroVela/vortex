// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeMap;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::Mutex;
use tempfile::TempDir;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex_index::DistanceMetric;
use vortex_index::Index;
use vortex_index::IndexMetadata;
use vortex_index::IndexRegistry;
use vortex_index::IndexStore;
use vortex_index::file::LocalFileSource;

use super::ValidationMode;
use super::provider;

pub(super) const MAX_RETAINED_HANDLES: usize = 8;
const MAX_RETAINED_ARTIFACT_BYTES: u64 = 256 * 1024 * 1024;
const MAX_RETAINED_SOURCE_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Default)]
struct Usage {
    handles: usize,
    artifact_bytes: u64,
    source_bytes: u64,
}

pub(super) struct CacheBudget {
    usage: Mutex<Usage>,
    max_handles: usize,
    max_artifact_bytes: u64,
    max_source_bytes: u64,
}

impl Default for CacheBudget {
    fn default() -> Self {
        Self {
            usage: Mutex::default(),
            max_handles: MAX_RETAINED_HANDLES,
            max_artifact_bytes: MAX_RETAINED_ARTIFACT_BYTES,
            max_source_bytes: MAX_RETAINED_SOURCE_BYTES,
        }
    }
}

impl CacheBudget {
    fn reserve(self: &Arc<Self>, artifact_bytes: u64, source_bytes: u64) -> Option<Reservation> {
        let mut usage = self.usage.lock();
        let total = usage.artifact_bytes.checked_add(artifact_bytes)?;
        let source_total = usage.source_bytes.checked_add(source_bytes)?;
        if usage.handles >= self.max_handles
            || total > self.max_artifact_bytes
            || source_total > self.max_source_bytes
        {
            return None;
        }
        usage.handles += 1;
        usage.artifact_bytes = total;
        usage.source_bytes = source_total;
        Some(Reservation {
            budget: Arc::clone(self),
            artifact_bytes,
            source_bytes,
        })
    }
}

struct Reservation {
    budget: Arc<CacheBudget>,
    artifact_bytes: u64,
    source_bytes: u64,
}

impl Drop for Reservation {
    fn drop(&mut self) {
        let mut usage = self.budget.usage.lock();
        usage.handles -= 1;
        usage.artifact_bytes -= self.artifact_bytes;
        usage.source_bytes -= self.source_bytes;
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct CacheKey {
    pub reference: PathBuf,
    pub identity: String,
    pub validation_mode: ValidationMode,
}

pub(super) struct OpenedIndex {
    // Close native handles and their artifact leases before removing the root.
    pub index: Arc<dyn Index>,
    pub source: Option<Arc<LocalFileSource>>,
    _scratch: TempDir,
    _reservation: Option<Reservation>,
}

impl OpenedIndex {
    async fn open(
        root: &Path,
        metadata: &IndexMetadata,
        store: Arc<dyn IndexStore>,
        reservation: Option<Reservation>,
        source: Option<Arc<LocalFileSource>>,
    ) -> VortexResult<Arc<Self>> {
        let scratch = tempfile::Builder::new()
            .prefix(".index-scratch-")
            .permissions(fs::Permissions::from_mode(0o700))
            .tempdir_in(root)?;
        let mut registry = IndexRegistry::default();
        registry.register(provider(&metadata.backend, scratch.path())?)?;
        let index = registry.open(metadata, &metadata.snapshot, store).await?;
        if index
            .as_vector()
            .is_none_or(|vector| vector.spec().metric != DistanceMetric::SquaredL2)
        {
            vortex_bail!("SQL static search requires a squared L2 vector index");
        }
        Ok(Arc::new(Self {
            index,
            source,
            _scratch: scratch,
            _reservation: reservation,
        }))
    }
}

pub(super) struct PreparedIndexCache {
    entries: Mutex<BTreeMap<CacheKey, Arc<OpenedIndex>>>,
    budget: Arc<CacheBudget>,
}

impl PreparedIndexCache {
    pub fn new(budget: Arc<CacheBudget>) -> Self {
        Self {
            entries: Mutex::default(),
            budget,
        }
    }

    pub fn snapshot(&self, key: &CacheKey) -> Option<Arc<OpenedIndex>> {
        self.entries.lock().get(key).cloned()
    }

    // Cache misses require fully verified inputs. Strict hits still verify anew;
    // snapshot hits use the separately retained source through snapshot().
    pub async fn open(
        cache: Option<&Self>,
        key: &CacheKey,
        root: &Path,
        metadata: &IndexMetadata,
        store: Arc<dyn IndexStore>,
        source: Option<Arc<LocalFileSource>>,
    ) -> VortexResult<(Arc<OpenedIndex>, bool)> {
        if (key.validation_mode == ValidationMode::Snapshot) != source.is_some() {
            vortex_bail!("Index cache validation mode does not match its source");
        }
        if let Some(cache) = cache
            && let Some(opened) = cache.entries.lock().get(key).cloned()
        {
            if opened.index.metadata() != metadata {
                vortex_bail!("Cached index metadata does not match the verified generation");
            }
            return Ok((opened, true));
        }
        let bytes = metadata
            .artifacts
            .iter()
            .try_fold(0_u64, |sum, artifact| sum.checked_add(artifact.size))
            .ok_or_else(|| vortex::error::vortex_err!("Index artifact byte count overflow"))?;
        let source_bytes = source
            .as_ref()
            .map(|source| u64::try_from(source.pinned_bytes()))
            .transpose()?
            .unwrap_or(0);
        let reservation = cache.and_then(|cache| cache.budget.reserve(bytes, source_bytes));
        let retain = reservation.is_some();
        if source.is_some() && !retain {
            vortex_bail!(
                "Prepared snapshot exceeds the connection's retained handle or byte budget"
            );
        }
        let opened = OpenedIndex::open(root, metadata, store, reservation, source).await?;
        if let Some(cache) = cache.filter(|_| retain) {
            // Concurrent misses may open twice; only one entry stays retained.
            return Ok((
                Arc::clone(cache.entries.lock().entry(key.clone()).or_insert(opened)),
                false,
            ));
        }
        Ok((opened, false))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use parking_lot::Mutex;

    use super::CacheBudget;

    #[test]
    fn test_reservations_bound_count_and_bytes_and_release_on_drop() {
        let budget = Arc::new(CacheBudget {
            usage: Mutex::default(),
            max_handles: 2,
            max_artifact_bytes: 10,
            max_source_bytes: 10,
        });
        let first = budget.reserve(6, 6);
        assert!(first.is_some());
        assert!(budget.reserve(5, 0).is_none());
        assert!(budget.reserve(0, 5).is_none());
        let second = budget.reserve(4, 4);
        assert!(second.is_some());
        assert!(budget.reserve(0, 0).is_none());
        drop(first);
        assert!(budget.reserve(6, 6).is_some());
        drop(second);
        assert!(budget.reserve(10, 10).is_some());
        assert_eq!(budget.usage.lock().handles, 0);
        assert_eq!(budget.usage.lock().artifact_bytes, 0);
        assert_eq!(budget.usage.lock().source_bytes, 0);
    }

    #[test]
    fn test_reservation_overflow_falls_back_without_changing_usage() {
        let budget = Arc::new(CacheBudget {
            usage: Mutex::default(),
            max_handles: 2,
            max_artifact_bytes: u64::MAX,
            max_source_bytes: u64::MAX,
        });
        let first = budget.reserve(u64::MAX, u64::MAX);
        assert!(first.is_some());
        assert!(budget.reserve(1, 0).is_none());
        assert!(budget.reserve(0, 1).is_none());
        assert_eq!(budget.usage.lock().handles, 1);
        assert_eq!(budget.usage.lock().artifact_bytes, u64::MAX);
        drop(first);
        assert_eq!(budget.usage.lock().handles, 0);
    }
}
