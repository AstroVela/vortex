// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Input views and planning metadata passed to a row kernel.

use crate::ArrayRef;
use crate::dtype::DType;

/// The arguments handed to one kernel invocation.
///
/// `arrays` may be filtered or sliced, while `dtypes` and `output_dtype` always describe the
/// original planned batch. Keeping them together prevents an execution path from pairing an input
/// view with unrelated planning metadata.
#[derive(Clone, Copy)]
pub struct KernelArgs<'a> {
    /// The input arrays for this kernel invocation.
    pub arrays: &'a [ArrayRef],

    /// The number of rows in this kernel invocation.
    pub row_count: usize,

    /// The original input dtypes used to select the row implementation.
    pub dtypes: &'a [DType],

    /// The non-nullable dtype built by the selected output capability.
    pub output_dtype: &'a DType,
}
