// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
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
use crate::dtype::IntegerPType;
use crate::dtype::NativePType;
use crate::executor::ExecutionCtx;
use crate::match_each_native_ptype;
use crate::match_each_unsigned_integer_ptype;
use crate::match_smallest_offset_type;
use crate::validity::Validity;

/// Use range copies when the child values are already a contiguous primitive buffer.
///
/// This deliberately does not canonicalize encoded children just to make range slicing possible:
/// the existing child `take` path lets those encodings handle the gather without eagerly
/// decompressing the full child.
const RANGE_TAKE_MIN_PRIMITIVE_LIST_BYTES: usize = 4;

/// Take implementation for [`FixedSizeListArray`].
///
/// Unlike `ListView`, `FixedSizeListArray` must rebuild the elements array because it requires
/// that elements start at offset 0 and be perfectly packed without gaps. We either use a bulk child
/// `take` over expanded element indices or copy selected child ranges directly, depending on the
/// child encoding and list width.
impl TakeExecute for FixedSizeList {
    fn take(
        array: ArrayView<'_, FixedSizeList>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let max_element_idx = array.elements().len();
        // Indices are non-negative; dispatch over the 4 unsigned widths (the executed array is
        // reinterpreted to unsigned in `take_with_indices`). `E` is already unsigned.
        match_each_unsigned_integer_ptype!(indices.dtype().as_ptype().to_unsigned(), |I| {
            match_smallest_offset_type!(max_element_idx, |E| {
                take_with_indices::<I, E>(array, indices, ctx)
            })
        })
        .map(Some)
    }
}

/// Dispatches to the appropriate take implementation based on list size and nullability.
fn take_with_indices<I: IntegerPType, E: IntegerPType>(
    array: ArrayView<'_, FixedSizeList>,
    indices: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let list_size = array.list_size() as usize;

    let indices_array = indices.clone().execute::<PrimitiveArray>(ctx)?;
    // Reinterpret to unsigned so `as_slice::<I>` (with unsigned `I`) matches; values are unchanged.
    let indices_array = indices_array.reinterpret_cast(indices_array.ptype().to_unsigned());

    // Make sure to handle degenerate case where lists have size 0 (these can take fast paths).
    if list_size == 0 {
        debug_assert!(
            array.elements().is_empty(),
            "degenerate list must have empty elements"
        );

        // Since there are no elements to take, we just need to take on the validity map.
        let new_validity = array.validity()?.take(indices)?;
        let new_len = indices_array.len();

        Ok(
            // SAFETY: list_size is 0, elements array is empty, and validity has the correct length.
            unsafe {
                FixedSizeListArray::new_unchecked(
                    array.elements().clone(), // Remember that this is an empty array.
                    array.list_size(),
                    new_validity,
                    new_len,
                )
            }
            .into_array(),
        )
    } else {
        // The result's nullability is the union of the input nullabilities.
        if array.dtype().is_nullable() || indices_array.dtype().is_nullable() {
            let indices_array = indices_array.as_view();
            if should_take_fsl_with_ranges(&array, list_size) {
                take_nullable_fsl_by_ranges::<I, E>(array, indices_array, ctx)
            } else {
                take_nullable_fsl::<I, E>(array, indices_array, ctx)
            }
        } else {
            let indices_array = indices_array.as_view();
            if should_take_fsl_with_ranges(&array, list_size) {
                take_non_nullable_fsl_by_ranges::<I, E>(array, indices_array)
            } else {
                take_non_nullable_fsl::<I, E>(array, indices_array)
            }
        }
    }
}

fn should_take_fsl_with_ranges(array: &ArrayView<'_, FixedSizeList>, list_size: usize) -> bool {
    let element_dtype = array
        .dtype()
        .as_fixed_size_list_element_opt()
        .vortex_expect("FixedSizeList dtype must have an element dtype");

    if !element_dtype.is_nullable() && array.elements().is::<Primitive>() {
        return element_dtype.element_size().is_some_and(|element_size| {
            element_size
                .checked_mul(list_size)
                .is_some_and(|list_byte_width| {
                    list_byte_width >= RANGE_TAKE_MIN_PRIMITIVE_LIST_BYTES
                })
        });
    }

    false
}

/// Takes from an array when both the array and indices are non-nullable.
fn take_non_nullable_fsl<I: IntegerPType, E: IntegerPType>(
    array: ArrayView<'_, FixedSizeList>,
    indices_array: ArrayView<'_, Primitive>,
) -> VortexResult<ArrayRef> {
    let list_size = array.list_size() as usize;
    let indices: &[I] = indices_array.as_slice::<I>();
    let new_len = indices.len();

    // Build the element indices directly without validity tracking.
    let mut elements_indices = BufferMut::<E>::with_capacity(new_len * list_size);

    // Build the element indices for each list.
    for data_idx in indices {
        let data_idx = data_idx
            .to_usize()
            .unwrap_or_else(|| vortex_panic!("Failed to convert index to usize: {}", data_idx));

        let list_start = data_idx * list_size;
        let list_end = (data_idx + 1) * list_size;

        // Expand the list into individual element indices.
        for i in list_start..list_end {
            // SAFETY: We've allocated enough space for enough indices for all `new_len` lists (that each consist of `list_size = list_end - list_start` elements), so we know we have enough capacity.
            unsafe {
                elements_indices.push_unchecked(E::from_usize(i).vortex_expect("i < list_end"))
            };
        }
    }

    let elements_indices = elements_indices.freeze();
    debug_assert_eq!(elements_indices.len(), new_len * list_size);

    let elements_indices_array = PrimitiveArray::new(elements_indices, Validity::NonNullable);
    let new_elements = array.elements().take(elements_indices_array.into_array())?;
    debug_assert_eq!(new_elements.len(), new_len * list_size);

    // Both inputs are non-nullable, so the result is non-nullable.
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

/// Takes from an array when both the array and indices are non-nullable, copying each selected
/// list as a contiguous range instead of expanding it into per-element gather indices.
fn take_non_nullable_fsl_by_ranges<I: IntegerPType, E: IntegerPType>(
    array: ArrayView<'_, FixedSizeList>,
    indices_array: ArrayView<'_, Primitive>,
) -> VortexResult<ArrayRef> {
    if let Some(new_elements) =
        take_primitive_non_nullable_elements_by_ranges::<I>(array, indices_array)?
    {
        let new_len = indices_array.len();
        return Ok(unsafe {
            FixedSizeListArray::new_unchecked(
                new_elements,
                array.list_size(),
                Validity::NonNullable,
                new_len,
            )
        }
        .into_array());
    }

    take_non_nullable_fsl::<I, E>(array, indices_array)
}

fn take_primitive_non_nullable_elements_by_ranges<I: IntegerPType>(
    array: ArrayView<'_, FixedSizeList>,
    indices_array: ArrayView<'_, Primitive>,
) -> VortexResult<Option<ArrayRef>> {
    let Some(elements) = array.elements().as_typed::<Primitive>() else {
        return Ok(None);
    };
    if !PrimitiveArrayExt::validity(&elements).definitely_no_nulls() {
        return Ok(None);
    }

    match_each_native_ptype!(elements.ptype(), |T| {
        Ok(Some(
            take_primitive_non_nullable_elements_by_ranges_typed::<I, T>(
                array,
                indices_array,
                elements,
            ),
        ))
    })
}

fn take_primitive_non_nullable_elements_by_ranges_typed<I, T>(
    array: ArrayView<'_, FixedSizeList>,
    indices_array: ArrayView<'_, Primitive>,
    elements: ArrayView<'_, Primitive>,
) -> ArrayRef
where
    I: IntegerPType,
    T: NativePType,
{
    let list_size = array.list_size() as usize;
    let indices: &[I] = indices_array.as_slice::<I>();
    let mut values = BufferMut::<T>::with_capacity(indices.len() * list_size);
    let source = elements.as_slice::<T>();

    for &data_idx in indices {
        let data_idx = index_to_usize(data_idx);
        let list_start = data_idx * list_size;
        values.extend_from_slice(&source[list_start..list_start + list_size]);
    }

    PrimitiveArray::new(values.freeze(), Validity::from(elements.nullability())).into_array()
}

/// Takes from an array when either the array or indices are nullable.
fn take_nullable_fsl<I: IntegerPType, E: IntegerPType>(
    array: ArrayView<'_, FixedSizeList>,
    indices_array: ArrayView<'_, Primitive>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let list_size = array.list_size() as usize;
    let indices: &[I] = indices_array.as_slice::<I>();
    let new_len = indices.len();

    let array_validity = array
        .fixed_size_list_validity()
        .execute_mask(array.as_ref().len(), ctx)
        .vortex_expect("Failed to compute validity mask");
    let indices_len = indices_array.as_ref().len();
    let indices_validity = match indices_array
        .validity()
        .vortex_expect("Failed to compute validity mask")
    {
        Validity::NonNullable | Validity::AllValid => Mask::new_true(indices_len),
        Validity::AllInvalid => Mask::new_false(indices_len),
        Validity::Array(a) => a.execute::<BoolArray>(ctx)?.execute_mask(ctx),
    };

    // We must use placeholder zeros for null lists to maintain the array length without
    // propagating nullability to the element array's take operation.
    let mut elements_indices = BufferMut::<E>::with_capacity(new_len * list_size);
    let mut new_validity_builder = BitBufferMut::with_capacity(new_len);

    // Build the element indices while tracking which lists are null.
    for (data_idx, is_index_valid) in indices.iter().zip(indices_validity.iter()) {
        let data_idx = data_idx
            .to_usize()
            .unwrap_or_else(|| vortex_panic!("Failed to convert index to usize: {}", data_idx));

        // The list is null if the index is null or the indexed element is null.
        if !is_index_valid || !array_validity.value(data_idx) {
            // Append placeholder zeros for null lists. These will be masked by the validity array.
            // We cannot use append_nulls here as explained above.
            unsafe { elements_indices.push_n_unchecked(E::zero(), list_size) };
            new_validity_builder.append(false);
        } else {
            // Append the actual element indices for this list.
            let list_start = data_idx * list_size;
            let list_end = (data_idx + 1) * list_size;

            // Expand the list into individual element indices.
            for i in list_start..list_end {
                // SAFETY: We've allocated enough space for enough indices for all `new_len` lists (that each consist of `list_size = list_end - list_start` elements), so we know we have enough capacity.
                unsafe {
                    elements_indices.push_unchecked(E::from_usize(i).vortex_expect("i < list_end"))
                };
            }

            new_validity_builder.append(true);
        }
    }

    let elements_indices = elements_indices.freeze();
    debug_assert_eq!(elements_indices.len(), new_len * list_size);

    let elements_indices_array = PrimitiveArray::new(elements_indices, Validity::NonNullable);
    let new_elements = array.elements().take(elements_indices_array.into_array())?;
    debug_assert_eq!(new_elements.len(), new_len * list_size);

    // At least one input was nullable, so the result is nullable.
    let new_validity = Validity::from(new_validity_builder.freeze());
    debug_assert!(new_validity.maybe_len().is_none_or(|vl| vl == new_len));

    Ok(unsafe {
        FixedSizeListArray::new_unchecked(new_elements, array.list_size(), new_validity, new_len)
    }
    .into_array())
}

/// Takes from an array when either the array or indices are nullable, copying each selected
/// non-null list as a contiguous range.
fn take_nullable_fsl_by_ranges<I: IntegerPType, E: IntegerPType>(
    array: ArrayView<'_, FixedSizeList>,
    indices_array: ArrayView<'_, Primitive>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let indices: &[I] = indices_array.as_slice::<I>();
    let new_len = indices.len();

    let array_validity = array
        .fixed_size_list_validity()
        .execute_mask(array.as_ref().len(), ctx)
        .vortex_expect("Failed to compute validity mask");
    let indices_len = indices_array.as_ref().len();
    let indices_validity = match indices_array
        .validity()
        .vortex_expect("Failed to compute validity mask")
    {
        Validity::NonNullable | Validity::AllValid => Mask::new_true(indices_len),
        Validity::AllInvalid => Mask::new_false(indices_len),
        Validity::Array(a) => a.execute::<BoolArray>(ctx)?.execute_mask(ctx),
    };

    if let Some((new_elements, new_validity)) = take_primitive_nullable_elements_by_ranges::<I>(
        array,
        indices_array,
        &array_validity,
        &indices_validity,
    )? {
        debug_assert!(new_validity.maybe_len().is_none_or(|vl| vl == new_len));
        return Ok(unsafe {
            FixedSizeListArray::new_unchecked(
                new_elements,
                array.list_size(),
                new_validity,
                new_len,
            )
        }
        .into_array());
    }

    take_nullable_fsl::<I, E>(array, indices_array, ctx)
}

fn take_primitive_nullable_elements_by_ranges<I: IntegerPType>(
    array: ArrayView<'_, FixedSizeList>,
    indices_array: ArrayView<'_, Primitive>,
    array_validity: &Mask,
    indices_validity: &Mask,
) -> VortexResult<Option<(ArrayRef, Validity)>> {
    let Some(elements) = array.elements().as_typed::<Primitive>() else {
        return Ok(None);
    };
    if !PrimitiveArrayExt::validity(&elements).definitely_no_nulls() {
        return Ok(None);
    }

    match_each_native_ptype!(elements.ptype(), |T| {
        Ok(Some(
            take_primitive_nullable_elements_by_ranges_typed::<I, T>(
                array,
                indices_array,
                elements,
                array_validity,
                indices_validity,
            ),
        ))
    })
}

fn take_primitive_nullable_elements_by_ranges_typed<I, T>(
    array: ArrayView<'_, FixedSizeList>,
    indices_array: ArrayView<'_, Primitive>,
    elements: ArrayView<'_, Primitive>,
    array_validity: &Mask,
    indices_validity: &Mask,
) -> (ArrayRef, Validity)
where
    I: IntegerPType,
    T: NativePType,
{
    let list_size = array.list_size() as usize;
    let indices: &[I] = indices_array.as_slice::<I>();
    let mut values = BufferMut::<T>::with_capacity(indices.len() * list_size);
    let mut new_validity_builder = BitBufferMut::with_capacity(indices.len());
    let source = elements.as_slice::<T>();

    for (&data_idx, is_index_valid) in indices.iter().zip(indices_validity.iter()) {
        let data_idx = index_to_usize(data_idx);
        if !is_index_valid || !array_validity.value(data_idx) {
            values.push_n(T::default(), list_size);
            new_validity_builder.append(false);
        } else {
            let list_start = data_idx * list_size;
            values.extend_from_slice(&source[list_start..list_start + list_size]);
            new_validity_builder.append(true);
        }
    }

    let new_elements =
        PrimitiveArray::new(values.freeze(), Validity::from(elements.nullability())).into_array();
    let new_validity = Validity::from(new_validity_builder.freeze());

    (new_elements, new_validity)
}

fn index_to_usize<I: IntegerPType>(index: I) -> usize {
    index
        .to_usize()
        .unwrap_or_else(|| vortex_panic!("Failed to convert index to usize: {}", index))
}
