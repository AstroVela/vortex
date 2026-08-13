// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Built-in Vortex scalar function implementations.
//!
//! The scalar function machinery — [`ScalarFnVTable`](vortex_array::scalar_fn::ScalarFnVTable),
//! the kernel traits that encodings implement, and the expression tree that references scalar
//! functions — lives in `vortex-array`. This crate holds implementations that `vortex-array`
//! itself does not depend on, so that the set of built-in functions can grow without growing the
//! core array crate.
//!
//! Register them on a session with [`register_scalar_fns`].

use vortex_array::scalar_fn::session::ScalarFnSessionExt;
use vortex_session::VortexSession;

pub mod exprs;
pub mod fns;

/// Register this crate's scalar functions on `session`.
///
/// [`VortexSession::default`](vortex_session::VortexSession) does not know about this crate, so a
/// session that should resolve these functions by id — when deserializing an expression, for
/// example — must call this.
pub fn register_scalar_fns(session: &VortexSession) {
    let scalar_fns = session.scalar_fns();
    scalar_fns.register(fns::byte_length::ByteLength);
    scalar_fns.register(fns::case_when::CaseWhen);
    scalar_fns.register(fns::ext_storage::ExtStorage);
    scalar_fns.register(fns::list_length::ListLength);
    scalar_fns.register(fns::list_sum::ListSum);
}

/// A [`vortex_array::array_session`] with this crate's scalar functions registered.
pub fn scalar_fn_session() -> VortexSession {
    let session = vortex_array::array_session();
    register_scalar_fns(&session);
    session
}
