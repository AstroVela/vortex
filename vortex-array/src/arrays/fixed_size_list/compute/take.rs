// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::RecursiveCanonical;
use crate::array::ArrayView;
use crate::arrays::ConstantArray;
use crate::arrays::FixedSizeList;
use crate::arrays::FixedSizeListArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::TakeSlicesArray;
use crate::arrays::dict::TakeExecute;
use crate::arrays::fixed_size_list::FixedSizeListArrayExt;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::builders::builder_with_capacity;
use crate::dtype::IntegerPType;
use crate::executor::ExecutionCtx;
use crate::match_each_unsigned_integer_ptype;
use crate::validity::Validity;

/// Take implementation for [`FixedSizeListArray`].
///
/// `FixedSizeListArray` must rebuild its elements array because selected lists need to become
/// packed from offset 0. The FSL layer translates selected list rows into ordered element runs
/// and delegates the execution strategy to the elements child via materialized `take_slices`.
impl TakeExecute for FixedSizeList {
    fn take(
        array: ArrayView<'_, FixedSizeList>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        match_each_unsigned_integer_ptype!(indices.dtype().as_ptype().to_unsigned(), |I| {
            take_with_indices::<I>(array, indices, ctx)
        })
        .map(Some)
    }
}

fn take_with_indices<I: IntegerPType>(
    array: ArrayView<'_, FixedSizeList>,
    indices: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let list_size = array.list_size() as usize;

    let indices_array = indices.clone().execute::<PrimitiveArray>(ctx)?;
    // Bit-identical reinterpret so `as_slice::<I>` (with unsigned `I`) matches. Negative signed
    // inputs become large unsigned values and are rejected by the bounds check below.
    let indices_array = indices_array.reinterpret_cast(indices_array.ptype().to_unsigned());

    if list_size == 0 {
        return take_degenerate_fsl::<I>(array, indices, indices_array.as_view(), ctx);
    }

    if array.is_empty() {
        return take_empty_non_degenerate_fsl::<I>(array, indices, indices_array.as_view(), ctx);
    }

    take_non_empty_non_degenerate_fsl::<I>(array, indices_array.as_view(), ctx)
}

fn take_degenerate_fsl<I: IntegerPType>(
    array: ArrayView<'_, FixedSizeList>,
    indices: &ArrayRef,
    indices_array: ArrayView<'_, Primitive>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    vortex_ensure!(
        array.elements().is_empty(),
        "degenerate list must have empty elements"
    );

    validate_valid_indices::<I>(&indices_array, array.as_ref().len(), ctx)?;
    let new_validity = array.validity()?.take(indices)?;
    let new_len = indices_array.len();

    // SAFETY: degenerate FSL inputs have no elements, valid index payloads were checked against
    // the source length, and `Validity::take` produces validity for `new_len`.
    Ok(unsafe {
        FixedSizeListArray::new_unchecked(
            array.elements().clone(),
            array.list_size(),
            new_validity,
            new_len,
        )
    }
    .into_array())
}

fn take_empty_non_degenerate_fsl<I: IntegerPType>(
    array: ArrayView<'_, FixedSizeList>,
    indices: &ArrayRef,
    indices_array: ArrayView<'_, Primitive>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    debug_assert_ne!(array.list_size(), 0);
    debug_assert!(array.is_empty());

    validate_valid_indices::<I>(&indices_array, 0, ctx)?;

    let list_size = array.list_size() as usize;
    let new_len = indices_array.len();
    let expected_elements_len = take_elements_len(new_len, list_size)?;
    let new_elements = default_elements(array, expected_elements_len);
    ensure_elements_len(new_elements.len(), expected_elements_len)?;
    let new_validity = if new_len == 0 {
        array.validity()?.take(indices)?
    } else {
        Validity::AllInvalid
    };

    // SAFETY: an empty non-degenerate source has no usable element slices. Valid indices have
    // already been rejected, so non-empty output is all null and backed by default child values.
    Ok(unsafe {
        FixedSizeListArray::new_unchecked(new_elements, array.list_size(), new_validity, new_len)
    }
    .into_array())
}

fn take_non_empty_non_degenerate_fsl<I: IntegerPType>(
    array: ArrayView<'_, FixedSizeList>,
    indices_array: ArrayView<'_, Primitive>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    debug_assert_ne!(array.list_size(), 0);
    debug_assert!(!array.is_empty());

    if array.dtype().is_nullable() || indices_array.dtype().is_nullable() {
        take_nullable_non_empty_fsl::<I>(array, indices_array, ctx)
    } else {
        take_non_nullable_non_empty_fsl::<I>(array, indices_array, ctx)
    }
}

fn take_non_nullable_non_empty_fsl<I: IntegerPType>(
    array: ArrayView<'_, FixedSizeList>,
    indices_array: ArrayView<'_, Primitive>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let list_size = array.list_size() as usize;
    let array_len = array.as_ref().len();
    let indices: &[I] = indices_array.as_slice::<I>();
    let new_len = indices.len();
    let expected_elements_len = take_elements_len(new_len, list_size)?;
    let mut starts = Vec::with_capacity(new_len);

    for &data_idx in indices {
        let data_idx = index_to_usize(data_idx)?;
        let start = list_start_u64(data_idx, list_size, array_len)?;
        starts.push(start);
    }

    let new_elements = take_element_runs(
        array.elements(),
        starts,
        list_size,
        expected_elements_len,
        ctx,
    )?;
    ensure_elements_len(new_elements.len(), expected_elements_len)?;

    // SAFETY: `starts` contains one checked run of `list_size` elements for each output row,
    // `new_elements` has `new_len * list_size` elements, and non-nullable validity has no length.
    Ok(unsafe {
        FixedSizeListArray::new_unchecked(
            new_elements,
            array.list_size(),
            Validity::NonNullable,
            new_len,
        )
    }
    .into_array())
}

fn take_nullable_non_empty_fsl<I: IntegerPType>(
    array: ArrayView<'_, FixedSizeList>,
    indices_array: ArrayView<'_, Primitive>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let list_size = array.list_size() as usize;
    let array_len = array.as_ref().len();
    let indices: &[I] = indices_array.as_slice::<I>();
    let new_len = indices.len();
    let expected_elements_len = take_elements_len(new_len, list_size)?;

    let array_validity = array
        .fixed_size_list_validity()
        .execute_mask(array.as_ref().len(), ctx)?;
    let indices_validity = indices_validity_mask(&indices_array, ctx)?;

    let mut starts = Vec::with_capacity(new_len);
    let mut new_validity_builder = BitBufferMut::with_capacity(new_len);

    // Null output rows still need placeholder child elements so the FSL elements length stays
    // `rows * list_size`. This path has a non-empty source, so 0 is in bounds and validity hides it.
    for (&data_idx, is_index_valid) in indices.iter().zip(indices_validity.iter()) {
        if !is_index_valid {
            starts.push(0);
            new_validity_builder.append(false);
            continue;
        }

        let data_idx = index_to_usize(data_idx)?;
        let start = list_start_u64(data_idx, list_size, array_len)?;
        if !array_validity.value(data_idx) {
            starts.push(0);
            new_validity_builder.append(false);
            continue;
        }

        starts.push(start);
        new_validity_builder.append(true);
    }

    let new_elements = take_element_runs(
        array.elements(),
        starts,
        list_size,
        expected_elements_len,
        ctx,
    )?;
    ensure_elements_len(new_elements.len(), expected_elements_len)?;

    let new_validity = Validity::from(new_validity_builder.freeze());
    debug_assert!(new_validity.maybe_len().is_none_or(|vl| vl == new_len));

    // SAFETY: `new_elements` has `new_len * list_size` elements. `new_validity_builder` appends
    // exactly one bit per output row, and `Validity::from` preserves that length when needed.
    Ok(unsafe {
        FixedSizeListArray::new_unchecked(new_elements, array.list_size(), new_validity, new_len)
    }
    .into_array())
}

fn indices_validity_mask(
    indices_array: &ArrayView<'_, Primitive>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Mask> {
    indices_array
        .validity()?
        .execute_mask(indices_array.as_ref().len(), ctx)
}

fn validate_valid_indices<I: IntegerPType>(
    indices_array: &ArrayView<'_, Primitive>,
    array_len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let indices: &[I] = indices_array.as_slice::<I>();
    let indices_validity = indices_validity_mask(indices_array, ctx)?;

    for (&data_idx, is_index_valid) in indices.iter().zip(indices_validity.iter()) {
        if is_index_valid {
            check_index_in_bounds(index_to_usize(data_idx)?, array_len)?;
        }
    }
    Ok(())
}

fn take_elements_len(new_len: usize, list_size: usize) -> VortexResult<usize> {
    new_len.checked_mul(list_size).ok_or_else(|| {
        vortex_err!(
            "FixedSizeList take output length overflow: {new_len} lists of size {list_size}"
        )
    })
}

fn ensure_elements_len(actual: usize, expected: usize) -> VortexResult<()> {
    vortex_ensure!(
        actual == expected,
        "FixedSizeList take elements length {actual} does not match expected length {expected}"
    );
    Ok(())
}

fn list_start_u64(data_idx: usize, list_size: usize, array_len: usize) -> VortexResult<u64> {
    check_index_in_bounds(data_idx, array_len)?;

    let start = data_idx.checked_mul(list_size).ok_or_else(|| {
        vortex_err!(
            "FixedSizeList take element range overflow for index {data_idx} and list size {list_size}"
        )
    })?;
    start.checked_add(list_size).ok_or_else(|| {
        vortex_err!(
            "FixedSizeList take element range overflow for index {data_idx} and list size {list_size}"
        )
        })?;
    Ok(start as u64)
}

fn check_index_in_bounds(data_idx: usize, array_len: usize) -> VortexResult<()> {
    if data_idx >= array_len {
        vortex_bail!(OutOfBounds: data_idx, 0, array_len);
    }
    Ok(())
}

fn default_elements(array: ArrayView<'_, FixedSizeList>, len: usize) -> ArrayRef {
    let mut builder = builder_with_capacity(array.elements().dtype(), len);
    builder.append_defaults(len);
    builder.finish()
}

fn take_element_runs(
    elements: &ArrayRef,
    starts: Vec<u64>,
    length: usize,
    output_len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let run_count = starts.len();
    let starts = PrimitiveArray::from_iter(starts).into_array();
    let lengths = ConstantArray::new(length as u64, run_count).into_array();

    // SAFETY: callers produced one start per output row after validating list indices against the
    // source FSL length. `length` is the fixed list size, represented as a non-nullable unsigned
    // constant array, and `output_len` was computed as `run_count * length`.
    Ok(
        unsafe { TakeSlicesArray::new_unchecked(elements.clone(), starts, lengths, output_len) }
            .into_array()
            .execute::<RecursiveCanonical>(ctx)?
            .0
            .into_array(),
    )
}

fn index_to_usize<I: IntegerPType>(index: I) -> VortexResult<usize> {
    index
        .to_usize()
        .ok_or_else(|| vortex_err!("FixedSizeList take index {index} does not fit in usize"))
}
