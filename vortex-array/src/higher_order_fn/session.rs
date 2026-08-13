// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;

use vortex_session::ArcSwapMap;
use vortex_session::SessionExt;
use vortex_session::SessionGuard;
use vortex_session::SessionVar;
use vortex_session::registry::Id;

use crate::higher_order_fn::HigherOrderFunctionPluginRef;
use crate::higher_order_fn::HigherOrderFunctionVTable;
use crate::higher_order_fn::fns::list_transform::ListTransform;

/// Registry of higher-order function vtables.
pub type HigherOrderFunctionRegistry = ArcSwapMap<Id, HigherOrderFunctionPluginRef>;

/// Session state for higher-order function vtables.
#[derive(Clone, Debug)]
pub struct HigherOrderFunctionSession {
    registry: HigherOrderFunctionRegistry,
}

impl HigherOrderFunctionSession {
    pub fn registry(&self) -> &HigherOrderFunctionRegistry {
        &self.registry
    }

    /// Register a vtable, replacing any existing vtable with the same ID.
    pub fn register<V: HigherOrderFunctionVTable>(&self, vtable: V) {
        self.registry.insert(vtable.id(), Arc::new(vtable));
    }
}

impl Default for HigherOrderFunctionSession {
    fn default() -> Self {
        let this = Self {
            registry: HigherOrderFunctionRegistry::default(),
        };
        this.register(ListTransform);
        this
    }
}

impl SessionVar for HigherOrderFunctionSession {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Extension trait for accessing higher-order-function session state.
pub trait HigherOrderFunctionSessionExt: SessionExt {
    fn higher_order_functions(&self) -> SessionGuard<'_, HigherOrderFunctionSession> {
        self.get::<HigherOrderFunctionSession>()
    }
}

impl<S: SessionExt> HigherOrderFunctionSessionExt for S {}
