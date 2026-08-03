// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use once_cell::sync::OnceCell;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::plan::PlanRef;

type ChildInitializer = dyn Fn(usize) -> VortexResult<Option<PlanRef>> + 'static + Send + Sync;

/// Lazily initializes and caches a fixed number of logical plan-child slots.
#[derive(Clone)]
pub(crate) struct LazyPlanChildren {
    initializer: Arc<ChildInitializer>,
    cache: Arc<[OnceCell<Option<PlanRef>>]>,
}

impl LazyPlanChildren {
    /// Creates lazy plan-child slots backed by `initializer`.
    pub(crate) fn new(
        len: usize,
        initializer: impl Fn(usize) -> VortexResult<Option<PlanRef>> + 'static + Send + Sync,
    ) -> Self {
        Self {
            initializer: Arc::new(initializer),
            cache: (0..len).map(|_| OnceCell::new()).collect::<Vec<_>>().into(),
        }
    }

    /// Returns the number of logical child slots without initializing any child.
    pub(crate) fn len(&self) -> usize {
        self.cache.len()
    }

    /// Returns a child, initializing and caching its slot on first access.
    pub(crate) fn get(&self, index: usize) -> VortexResult<Option<PlanRef>> {
        let Some(cell) = self.cache.get(index) else {
            vortex_bail!(
                "Plan child index out of bounds: {index} of {}",
                self.cache.len()
            )
        };
        Ok(cell.get_or_try_init(|| (self.initializer)(index))?.clone())
    }

    /// Lazily transforms each present child into a new child collection.
    pub(crate) fn map(
        &self,
        transform: impl Fn(usize, PlanRef) -> VortexResult<PlanRef> + 'static + Send + Sync,
    ) -> Self {
        let source = self.clone();
        Self::new(self.len(), move |index| {
            source
                .get(index)?
                .map(|child| transform(index, child))
                .transpose()
        })
    }
}
