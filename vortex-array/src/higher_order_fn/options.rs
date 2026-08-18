// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;
use std::hash::Hasher;

use vortex_error::VortexResult;

use crate::higher_order_fn::typed::DynHigherOrderFunction;

/// An opaque handle to the options of a higher-order function.
pub struct HigherOrderFunctionOptions<'a> {
    pub(super) inner: &'a dyn DynHigherOrderFunction,
}

impl Display for HigherOrderFunctionOptions<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.options_display(f)
    }
}

impl Debug for HigherOrderFunctionOptions<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.options_debug(f)
    }
}

impl PartialEq for HigherOrderFunctionOptions<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.inner.id() == other.inner.id() && self.inner.options_eq(other.inner.options_any())
    }
}

impl Eq for HigherOrderFunctionOptions<'_> {}

impl Hash for HigherOrderFunctionOptions<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.id().hash(state);
        self.inner.options_hash(state);
    }
}

impl HigherOrderFunctionOptions<'_> {
    /// Serialize these options to a byte vector.
    pub fn serialize(&self) -> VortexResult<Option<Vec<u8>>> {
        self.inner.options_serialize()
    }

    /// Return the underlying typed options as [`Any`].
    pub fn as_any(&self) -> &dyn Any {
        self.inner.options_any()
    }
}
