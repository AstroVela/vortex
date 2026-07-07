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
use crate::array::ArrayView;
use crate::arrays::BoolArray;
use crate::arrays::FixedSizeList;
use crate::arrays::FixedSizeListArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::bool::BoolArrayExt;
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
/// packed from offset 0. The FSL layer translates selected list rows into ordered element ranges
/// and delegates the execution strategy to the elements child via `take_slices`.
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
    // Reinterpret to unsigned so `as_slice::<I>` (with unsigned `I`) matches; values are unchanged.
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
        take_non_nullable_non_empty_fsl::<I>(array, indices_array)
    }
}

fn take_non_nullable_non_empty_fsl<I: IntegerPType>(
    array: ArrayView<'_, FixedSizeList>,
    indices_array: ArrayView<'_, Primitive>,
) -> VortexResult<ArrayRef> {
    let list_size = array.list_size() as usize;
    let array_len = array.as_ref().len();
    let indices: &[I] = indices_array.as_slice::<I>();
    let new_len = indices.len();
    let expected_elements_len = take_elements_len(new_len, list_size)?;
    let mut slices = Vec::with_capacity(new_len);

    for &data_idx in indices {
        let data_idx = index_to_usize(data_idx)?;
        slices.push(list_range(data_idx, list_size, array_len)?);
    }

    let new_elements = array.elements().take_slices(slices)?;
    ensure_elements_len(new_elements.len(), expected_elements_len)?;

    // SAFETY: `slices` contains one checked range of `list_size` elements for each output row,
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

    let null_elements = (0, list_size);
    let mut slices = Vec::with_capacity(new_len);
    let mut new_validity_builder = BitBufferMut::with_capacity(new_len);

    for (&data_idx, is_index_valid) in indices.iter().zip(indices_validity.iter()) {
        if !is_index_valid {
            slices.push(null_elements);
            new_validity_builder.append(false);
            continue;
        }

        let data_idx = index_to_usize(data_idx)?;
        let range = list_range(data_idx, list_size, array_len)?;
        if !array_validity.value(data_idx) {
            slices.push(null_elements);
            new_validity_builder.append(false);
            continue;
        }

        slices.push(range);
        new_validity_builder.append(true);
    }

    let new_elements = array.elements().take_slices(slices)?;
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
    let indices_len = indices_array.as_ref().len();
    match indices_array.validity()? {
        Validity::NonNullable | Validity::AllValid => Ok(Mask::new_true(indices_len)),
        Validity::AllInvalid => Ok(Mask::new_false(indices_len)),
        Validity::Array(a) => Ok(a.execute::<BoolArray>(ctx)?.execute_mask(ctx)),
    }
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

fn list_range(data_idx: usize, list_size: usize, array_len: usize) -> VortexResult<(usize, usize)> {
    check_index_in_bounds(data_idx, array_len)?;

    let start = data_idx.checked_mul(list_size).ok_or_else(|| {
        vortex_err!(
            "FixedSizeList take element range overflow for index {data_idx} and list size {list_size}"
        )
    })?;
    let end = start.checked_add(list_size).ok_or_else(|| {
        vortex_err!(
            "FixedSizeList take element range overflow for index {data_idx} and list size {list_size}"
        )
    })?;
    Ok((start, end))
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

fn index_to_usize<I: IntegerPType>(index: I) -> VortexResult<usize> {
    index
        .to_usize()
        .ok_or_else(|| vortex_err!("FixedSizeList take index {index} does not fit in usize"))
}
