// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeMap;
use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::Index;
use crate::IndexMetadata;
use crate::IndexProvider;
use crate::IndexStore;
use crate::Snapshot;
use crate::metadata::nonempty;

/// Explicit backend registry. No native dependency or provider is loaded by default.
///
/// ```
/// use vortex_index::IndexRegistry;
///
/// let registry = IndexRegistry::default();
/// // Adding this crate alone does not link or register SPFresh.
/// assert!(registry.provider("vortex.spfresh").is_err());
/// ```
#[derive(Default)]
pub struct IndexRegistry {
    providers: BTreeMap<String, Arc<dyn IndexProvider>>,
}

impl IndexRegistry {
    /// Register a backend without silently replacing another implementation.
    pub fn register(&mut self, provider: Arc<dyn IndexProvider>) -> VortexResult<()> {
        nonempty(provider.id())?;
        if self.providers.contains_key(provider.id()) {
            vortex_bail!("Index provider is already registered: {}", provider.id());
        }
        self.providers.insert(provider.id().to_owned(), provider);
        Ok(())
    }

    /// Find an explicitly registered provider; unknown backends are errors.
    pub fn provider(&self, id: &str) -> VortexResult<&Arc<dyn IndexProvider>> {
        match self.providers.get(id) {
            Some(provider) => Ok(provider),
            None => vortex_bail!("Unknown index provider: {}", id),
        }
    }

    /// Validate snapshot and format support before opening any backend artifacts.
    ///
    /// Whether an unavailable index triggers a scan fallback or fails an explicit
    /// index request is a planner decision. This API never returns an empty result
    /// to disguise an incompatible or missing index.
    pub async fn open(
        &self,
        metadata: &IndexMetadata,
        snapshot: &Snapshot,
        store: Arc<dyn IndexStore>,
    ) -> VortexResult<Arc<dyn Index>> {
        metadata.validate_for(snapshot)?;
        let provider = self.provider(&metadata.backend)?;
        if !provider.supports_version(metadata.backend_version) {
            vortex_bail!(
                "Unsupported artifact version {} for {}",
                metadata.backend_version,
                metadata.backend
            );
        }
        let index = provider.open(metadata, store).await?;
        if index.metadata() != metadata {
            vortex_bail!("Index provider returned a different generation or descriptor");
        }
        Ok(index)
    }
}
