// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Experimental support for strict scalar functions computed one row at a time.
//!
//! This module is experimental and has no compatibility guarantees. External users must enable
//! the `unstable_row_fns` Cargo feature before importing it.
//!
//! A [`RowFn`] describes the typed operation while the framework owns columnar concerns such as
//! decoding, constant handling, null propagation, allocation, and validity. Its
//! [`RowFn::dispatch`] implementation uses a [`RowVisitor`] to select an [`ElementTuple`] and
//! output contract for each supported dtype combination. An [`OutputElement`] returns one owned
//! value per row, an [`OutputSink`] writes through a shared builder, and a [`RowKernel`] can attach
//! an associated dense output representation while retaining scalar semantics for other execution
//! policies.
//!
//! Unlike a general strict function, a [`RowFn`] cannot produce null from valid inputs.
//!
//! A _partially valid_ batch contains both valid and invalid rows. _Skip-invalid_ runs the kernel
//! only for valid rows without changing row positions.
//!
//! Prepared visits move work derived from constant operands outside the hot loop. Deferred visits
//! reduce compact failure evidence without constructing errors in that loop. Eligible kernels
//! first evaluate all payloads. If the reduced evidence reports an error for a partially valid
//! batch, execution retries only valid rows.

mod execute;

mod batch;

mod row_fn;
pub use row_fn::RowFn;

mod types;
pub use types::ArgColumn;
pub use types::ArgView;
pub use types::DenseRows;
pub use types::ElementTuple;
pub use types::FailureEvidence;
pub use types::IndexedElementTuple;
pub use types::InitializedElement;
pub use types::InputElement;
pub use types::OutputElement;
pub use types::OutputSink;
pub use types::PackedBoolOutput;
pub use types::RowKernel;
pub use types::RowKernelOutput;
pub use types::SinkResult;
pub use types::UninitElementSink;
pub use types::VecOutput;
pub use types::ViewLen;

mod visitor;
pub use visitor::RowVisitor;

mod vtable;
pub use vtable::execute_rows;
pub use vtable::row_fn_return_dtype;
