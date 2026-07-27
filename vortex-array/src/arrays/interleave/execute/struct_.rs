// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::super::Interleave;
use super::super::InterleaveArray;
use super::super::InterleaveArrayExt;
use super::selectors::validate_interleave;
use crate::IntoArray;
use crate::array::Array;
use crate::arrays::Primitive;
use crate::arrays::Struct;
use crate::arrays::StructArray;
use crate::arrays::struct_::StructArrayExt;
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
        array = require_child!(array, array.value(i), i + 2 => Struct);
    }
    validate_interleave(&array)?;

    let first = array.value(0).as_::<Struct>();
    let mut fields = Vec::with_capacity(first.struct_fields().nfields());
    for field_idx in 0..first.struct_fields().nfields() {
        let values = (0..num_values)
            .map(|i| {
                array
                    .value(i)
                    .as_::<Struct>()
                    .unmasked_field(field_idx)
                    .clone()
            })
            .collect();
        fields.push(
            InterleaveArray::try_new(
                values,
                array.array_indices().clone(),
                array.row_indices().clone(),
            )?
            .into_array(),
        );
    }

    let output = StructArray::try_new_with_dtype(
        fields,
        first.struct_fields().clone(),
        array.len(),
        array.as_ref().validity()?,
    )?;
    Ok(ExecutionResult::done(output))
}
