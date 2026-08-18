// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Higher-order function vtable machinery.
//!
//! Higher-order functions combine ordinary expression children with lambdas. They are distinct
//! from scalar functions because they establish lambda parameter types and evaluation bindings.

use vortex_session::registry::Id;

mod erased;
pub use erased::HigherOrderFunctionRef;

mod options;
pub use options::HigherOrderFunctionOptions;

mod typed;
pub use typed::TypedHigherOrderFunctionInstance;

mod lambda;
pub use lambda::LambdaCall;

mod plugin;
pub use plugin::*;

pub mod session;

mod vtable;
pub use vtable::*;

pub mod fns;

/// A globally unique identifier for a higher-order function.
pub type HigherOrderFunctionId = Id;
