// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Input views and planning metadata passed to a row kernel.

use crate::ArrayRef;
use crate::dtype::DType;
use crate::scalar_fn::ExecutionArgs;

/// The arguments handed to one kernel invocation.
///
/// `arrays` may be filtered or sliced, while `dtypes` and `output_dtype` always describe the
/// original planned batch. Keeping them together prevents an execution path from accidentally
/// pairing an input view with unrelated planning metadata.
#[derive(Clone, Copy)]
pub struct KernelArgs<'a> {
    /// The executor-facing view, including the row count for this invocation.
    pub execution: &'a dyn ExecutionArgs,

    /// The same inputs as concrete arrays for encoding-aware rewrites.
    pub arrays: &'a [ArrayRef],

    /// The original input dtypes used to select the row implementation.
    pub dtypes: &'a [DType],

    /// The non-nullable dtype built by the selected output capability.
    pub output_dtype: &'a DType,
}
