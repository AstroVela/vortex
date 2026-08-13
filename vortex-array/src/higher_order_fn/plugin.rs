// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::higher_order_fn::HigherOrderFunctionId;
use crate::higher_order_fn::HigherOrderFunctionRef;
use crate::higher_order_fn::HigherOrderFunctionVTable;

/// A session-registered higher-order-function vtable.
pub trait HigherOrderFunctionPlugin: 'static + Send + Sync {
    fn id(&self) -> HigherOrderFunctionId;
    fn deserialize(
        &self,
        metadata: &[u8],
        session: &VortexSession,
    ) -> VortexResult<HigherOrderFunctionRef>;
}

impl std::fmt::Debug for dyn HigherOrderFunctionPlugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("HigherOrderFunctionPlugin")
            .field(&self.id())
            .finish()
    }
}

impl<V: HigherOrderFunctionVTable> HigherOrderFunctionPlugin for V {
    fn id(&self) -> HigherOrderFunctionId {
        V::id(self)
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        session: &VortexSession,
    ) -> VortexResult<HigherOrderFunctionRef> {
        let function = HigherOrderFunctionRef::new(self.clone());
        function.deserialize(metadata, session)?;
        Ok(function)
    }
}

pub type HigherOrderFunctionPluginRef = Arc<dyn HigherOrderFunctionPlugin>;
