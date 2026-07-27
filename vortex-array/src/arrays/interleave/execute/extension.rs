// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::super::Interleave;
use super::super::InterleaveArray;
use super::super::InterleaveArrayExt;
use super::selectors::validate_interleave;
use crate::IntoArray;
use crate::array::Array;
use crate::arrays::Extension;
use crate::arrays::ExtensionArray;
use crate::arrays::Primitive;
use crate::arrays::extension::ExtensionArrayExt;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionResult;
use crate::require_child;

pub(super) fn execute(
    array: Array<Interleave>,
    _ctx: &mut ExecutionCtx,
) -> VortexResult<ExecutionResult> {
    let num_values = array.num_values();
    let mut array = array;
    array = require_child!(array, array.array_indices(), 0 => Primitive);
    array = require_child!(array, array.row_indices(), 1 => Primitive);
    for i in 0..num_values {
        array = require_child!(array, array.value(i), i + 2 => Extension);
    }
    validate_interleave(&array)?;

    let first = array.value(0).as_::<Extension>();
    let storage = InterleaveArray::try_new(
        (0..num_values)
            .map(|i| array.value(i).as_::<Extension>().storage_array().clone())
            .collect(),
        array.array_indices().clone(),
        array.row_indices().clone(),
    )?
    .into_array();
    let ext_dtype = first
        .ext_dtype()
        .with_nullability(storage.dtype().nullability());
    let output = ExtensionArray::try_new(ext_dtype, storage)?;
    Ok(ExecutionResult::done(output))
}
