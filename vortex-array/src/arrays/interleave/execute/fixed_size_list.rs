// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ptr;

use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::super::Interleave;
use super::super::InterleaveArray;
use super::super::InterleaveArrayExt;
use super::selectors::validate_selectors;
use crate::ArrayRef;
use crate::IntoArray;
use crate::array::Array;
use crate::arrays::FixedSizeList;
use crate::arrays::FixedSizeListArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::fixed_size_list::FixedSizeListArrayExt;
use crate::dtype::NativePType;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionResult;
use crate::match_each_native_ptype;
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
        array = require_child!(array, array.value(i), i + 2 => FixedSizeList);
    }

    let first = array.value(0).as_::<FixedSizeList>();
    let list_size = first.list_size();
    if let Some(output) = execute_nonnullable_primitive_elements(&array, list_size)? {
        return Ok(ExecutionResult::done(output));
    }

    let array_indices = array.array_indices().as_::<Primitive>();
    let row_indices = array.row_indices().as_::<Primitive>();
    let (element_array_indices, element_row_indices) =
        match_each_unsigned_integer_ptype!(array_indices.ptype(), |A| {
            match_each_unsigned_integer_ptype!(row_indices.ptype(), |R| {
                expand_selectors(
                    num_values,
                    |branch| array.value(branch).len(),
                    array_indices.as_slice::<A>(),
                    row_indices.as_slice::<R>(),
                    list_size as usize,
                )?
            })
        });

    let elements = InterleaveArray::try_new(
        (0..num_values)
            .map(|i| array.value(i).as_::<FixedSizeList>().elements().clone())
            .collect(),
        element_array_indices.into_array(),
        element_row_indices.into_array(),
    )?
    .into_array();
    let output =
        FixedSizeListArray::try_new(elements, list_size, array.as_ref().validity()?, array.len())?;
    Ok(ExecutionResult::done(output))
}

fn execute_nonnullable_primitive_elements(
    array: &Array<Interleave>,
    list_size: u32,
) -> VortexResult<Option<ArrayRef>> {
    let first = array.value(0).as_::<FixedSizeList>();
    let Some(first_elements) = first.elements().as_opt::<Primitive>() else {
        return Ok(None);
    };
    if first_elements.dtype().is_nullable()
        || (1..array.num_values()).any(|i| {
            !array
                .value(i)
                .as_::<FixedSizeList>()
                .elements()
                .is::<Primitive>()
        })
    {
        return Ok(None);
    }

    let elements = match_each_native_ptype!(first_elements.ptype(), |T| {
        gather_primitive_blocks::<T>(array, list_size as usize)?
    });
    let output =
        FixedSizeListArray::try_new(elements, list_size, array.as_ref().validity()?, array.len())?
            .into_array();
    Ok(Some(output))
}

fn gather_primitive_blocks<T: NativePType>(
    array: &Array<Interleave>,
    list_size: usize,
) -> VortexResult<ArrayRef> {
    let value_buffers = (0..array.num_values())
        .map(|i| {
            array
                .value(i)
                .as_::<FixedSizeList>()
                .elements()
                .as_::<Primitive>()
                .to_buffer::<T>()
        })
        .collect::<Vec<_>>();
    let value_lengths = (0..array.num_values())
        .map(|i| array.value(i).len())
        .collect::<Vec<_>>();
    let array_indices = array.array_indices().as_::<Primitive>();
    let row_indices = array.row_indices().as_::<Primitive>();
    let values = match_each_unsigned_integer_ptype!(array_indices.ptype(), |A| {
        match_each_unsigned_integer_ptype!(row_indices.ptype(), |R| {
            gather_blocks(
                &value_buffers,
                &value_lengths,
                array_indices.as_slice::<A>(),
                row_indices.as_slice::<R>(),
                list_size,
            )?
        })
    });
    Ok(PrimitiveArray::new(values, Validity::NonNullable).into_array())
}

fn gather_blocks<T: Copy, A: AsPrimitive<usize>, R: AsPrimitive<usize>>(
    value_buffers: &[Buffer<T>],
    value_lengths: &[usize],
    branches: &[A],
    rows: &[R],
    list_size: usize,
) -> VortexResult<Buffer<T>> {
    let len = validate_selectors(
        value_buffers.len(),
        |branch| value_lengths[branch],
        branches,
        rows,
    )?;
    let elements_len = len.checked_mul(list_size).ok_or_else(|| {
        vortex_err!(
            "interleave FixedSizeList output length overflow: {len} lists of size {list_size}"
        )
    })?;
    let mut result = BufferMut::with_capacity(elements_len);
    let output = result.spare_capacity_mut().as_mut_ptr().cast::<T>();

    for i in 0..len {
        let branch = branches[i].as_();
        let row = rows[i].as_();
        let source_offset = row * list_size;
        let output_offset = i * list_size;
        // SAFETY: selector validation proved the selected row is in bounds. Every canonical FSL
        // element buffer contains exactly `value_len * list_size` elements, the output reserved
        // `len * list_size`, and the selected input/output blocks do not overlap.
        unsafe {
            ptr::copy_nonoverlapping(
                value_buffers[branch].as_ptr().add(source_offset),
                output.add(output_offset),
                list_size,
            );
        }
    }

    // SAFETY: each selected row initialized one disjoint block of `list_size` elements.
    unsafe { result.set_len(elements_len) };
    Ok(result.freeze())
}

fn expand_selectors<A, R>(
    num_values: usize,
    value_len: impl Fn(usize) -> usize,
    branches: &[A],
    rows: &[R],
    list_size: usize,
) -> VortexResult<(Buffer<u64>, Buffer<u64>)>
where
    A: AsPrimitive<usize>,
    R: AsPrimitive<usize>,
{
    let len = validate_selectors(num_values, value_len, branches, rows)?;
    let elements_len = len.checked_mul(list_size).ok_or_else(|| {
        vortex_err!(
            "interleave FixedSizeList output length overflow: {len} lists of size {list_size}"
        )
    })?;
    let mut element_branches = BufferMut::with_capacity(elements_len);
    let mut element_rows = BufferMut::with_capacity(elements_len);

    for i in 0..len {
        let branch = branches[i].as_();
        let first_row = rows[i].as_().checked_mul(list_size).ok_or_else(|| {
            vortex_err!(
                "interleave FixedSizeList row offset overflow: row {} with list size {list_size}",
                rows[i].as_()
            )
        })?;
        let branch = u64::try_from(branch)
            .map_err(|_| vortex_err!("interleave array index does not fit in u64"))?;
        for offset in 0..list_size {
            element_branches.push(branch);
            element_rows.push(u64::try_from(first_row + offset).map_err(|_| {
                vortex_err!("interleave FixedSizeList element index does not fit in u64")
            })?);
        }
    }

    Ok((element_branches.freeze(), element_rows.freeze()))
}
