// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Optimized [`Interleave`] implementation for primitive values.

use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;

use super::super::Interleave;
use super::super::InterleaveArrayExt;
use super::selectors::validate_selectors;
use crate::ArrayRef;
use crate::IntoArray;
use crate::array::Array;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::dtype::NativePType;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionResult;
use crate::match_each_native_ptype;
use crate::match_each_unsigned_integer_ptype;
use crate::require_child;
use crate::validity::Validity;

/// Gathers primitive values under unsigned `array_indices` / `row_indices` selectors.
pub(super) fn execute(
    array: Array<Interleave>,
    _ctx: &mut ExecutionCtx,
) -> VortexResult<ExecutionResult> {
    let num_values = array.num_values();

    let mut array = array;
    array = require_child!(array, array.array_indices(), 0 => Primitive);
    array = require_child!(array, array.row_indices(), 1 => Primitive);
    for i in 0..num_values {
        array = require_child!(array, array.value(i), i + 2 => Primitive);
    }

    let validity = array.as_ref().validity()?;
    let output = match_each_native_ptype!(array.value(0).as_::<Primitive>().ptype(), |T| {
        execute_typed::<T>(&array, validity)?
    });

    Ok(ExecutionResult::done(output))
}

fn execute_typed<T: NativePType>(
    array: &Array<Interleave>,
    validity: Validity,
) -> VortexResult<ArrayRef> {
    let value_buffers = (0..array.num_values())
        .map(|i| array.value(i).as_::<Primitive>().to_buffer::<T>())
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
    Ok(PrimitiveArray::new(values, validity).into_array())
}

pub(super) fn gather<T: Copy, A: AsPrimitive<usize>, R: AsPrimitive<usize>>(
    value_buffers: &[Buffer<T>],
    branches: &[A],
    rows: &[R],
) -> VortexResult<Buffer<T>> {
    let len = validate_selectors(
        value_buffers.len(),
        |branch| value_buffers[branch].len(),
        branches,
        rows,
    )?;

    let mut result = BufferMut::with_capacity(len);
    let output = result.spare_capacity_mut().as_mut_ptr().cast::<T>();

    for i in 0..len {
        let branch = branches[i].as_();
        let row = rows[i].as_();
        // SAFETY: `validate_selectors` proved both selector bounds, and `result` reserved `len`
        // elements. Each output position is written exactly once.
        unsafe {
            output
                .add(i)
                .write(*value_buffers.get_unchecked(branch).get_unchecked(row));
        }
    }

    // SAFETY: The loop initialized exactly `len` elements.
    unsafe { result.set_len(len) };
    Ok(result.freeze())
}
