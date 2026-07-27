// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Optimized [`Interleave`] implementation for decimal values.

use vortex_error::VortexResult;

use super::super::Interleave;
use super::super::InterleaveArrayExt;
use super::primitive::gather;
use crate::ArrayRef;
use crate::IntoArray;
use crate::array::Array;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::arrays::Primitive;
use crate::dtype::NativeDecimalType;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionResult;
use crate::match_each_decimal_value_type;
use crate::match_each_unsigned_integer_ptype;
use crate::require_child;
use crate::validity::Validity;

pub(super) fn execute(
    array: Array<Interleave>,
    _ctx: &mut ExecutionCtx,
) -> VortexResult<ExecutionResult> {
    let num_values = array.num_values();

    let mut array = array;
    array = require_child!(array, array.array_indices(), 0 => Primitive);
    array = require_child!(array, array.row_indices(), 1 => Primitive);
    for i in 0..num_values {
        array = require_child!(array, array.value(i), i + 2 => Decimal);
    }

    let first = array.value(0).as_::<Decimal>();
    let decimal_dtype = first.decimal_dtype();
    let validity = array.as_ref().validity()?;
    let output = match_each_decimal_value_type!(first.values_type(), |T| {
        execute_typed::<T>(&array, decimal_dtype, validity)?
    });

    Ok(ExecutionResult::done(output))
}

fn execute_typed<T: NativeDecimalType>(
    array: &Array<Interleave>,
    decimal_dtype: crate::dtype::DecimalDType,
    validity: Validity,
) -> VortexResult<ArrayRef> {
    let value_buffers = (0..array.num_values())
        .map(|i| array.value(i).as_::<Decimal>().buffer::<T>())
        .collect::<Vec<_>>();
    let array_indices = array.array_indices().as_::<Primitive>();
    let row_indices = array.row_indices().as_::<Primitive>();
    let values = match_each_unsigned_integer_ptype!(array_indices.ptype(), |A| {
        match_each_unsigned_integer_ptype!(row_indices.ptype(), |R| {
            gather(
                &value_buffers,
                array_indices.as_slice::<A>(),
                row_indices.as_slice::<R>(),
            )?
        })
    });
    Ok(DecimalArray::new(values, decimal_dtype, validity).into_array())
}
