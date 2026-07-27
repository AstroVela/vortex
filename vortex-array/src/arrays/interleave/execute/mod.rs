// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execution logic for [`Interleave`], dispatched on the value type.
//!
//! All values share a type (validated in [`Interleave::check`]), so the
//! physical gather kernel is chosen from the first value. The selector types are an orthogonal
//! concern handled within each kernel. Fixed-width nested types are rebuilt recursively around
//! interleaved canonical children.
//!
//! [`Interleave::check`]: super::Interleave::check
//! [`bool`]: module@crate::arrays::interleave::execute::bool

mod bool;
mod decimal;
mod extension;
mod fixed_size_list;
mod null;
mod primitive;
mod selectors;
mod struct_;

use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use super::Interleave;
use super::InterleaveArrayExt;
use crate::array::Array;
use crate::dtype::DType;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionResult;

/// Executes an [`InterleaveArray`](super::InterleaveArray) by dispatching on the value type.
pub(super) fn execute(
    array: Array<Interleave>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ExecutionResult> {
    let value_dtype = array.value(0).dtype().clone();
    match value_dtype {
        DType::Null => null::execute(array, ctx),
        DType::Bool(..) => bool::execute(array, ctx),
        DType::Primitive(..) => primitive::execute(array, ctx),
        DType::Decimal(..) => decimal::execute(array, ctx),
        dtype @ DType::FixedSizeList(..) if dtype.element_size().is_some() => {
            fixed_size_list::execute(array, ctx)
        }
        dtype @ DType::Struct(..) if dtype.element_size().is_some() => struct_::execute(array, ctx),
        dtype @ DType::Extension(..) if dtype.element_size().is_some() => {
            extension::execute(array, ctx)
        }
        value_dtype => {
            vortex_panic!(
                "interleave execution is not implemented for value dtype {}",
                value_dtype
            )
        }
    }
}
