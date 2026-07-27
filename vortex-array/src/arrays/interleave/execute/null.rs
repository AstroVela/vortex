// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::super::Interleave;
use super::super::InterleaveArrayExt;
use super::selectors::validate_interleave;
use crate::array::Array;
use crate::arrays::NullArray;
use crate::arrays::Primitive;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionResult;
use crate::require_child;

pub(super) fn execute(
    array: Array<Interleave>,
    _ctx: &mut ExecutionCtx,
) -> VortexResult<ExecutionResult> {
    let mut array = array;
    array = require_child!(array, array.array_indices(), 0 => Primitive);
    array = require_child!(array, array.row_indices(), 1 => Primitive);
    validate_interleave(&array)?;
    Ok(ExecutionResult::done(NullArray::new(array.len())))
}
