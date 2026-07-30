// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! What the tensor scalar functions add to the row-function machinery: an element type that reads
//! a tensor row, and the width rule they share.

use std::marker::PhantomData;

use num_traits::Float;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::scalar_fn::InputElement;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;

use crate::utils::FlatElements;
use crate::utils::extract_flat_elements;
use crate::utils::validate_tensor_float_input;
use crate::utils::validate_tensor_float_inputs;

/// The width rule the tensor scalar functions share: every argument is the same float tensor dtype,
/// and the width is its element ptype.
pub(crate) fn tensor_element_ptype(args: &[DType]) -> VortexResult<PType> {
    Ok(validate_tensor_float_inputs(args)?.element_ptype())
}

/// Marker for tensor-valued input elements: accepts any tensor-like extension column whose
/// elements are `T`, and presents each row as its flat elements, `&[T]`.
pub struct TensorRow<T>(PhantomData<T>);

impl<T: Float + NativePType> InputElement for TensorRow<T> {
    type Column = FlatElements;
    type Elem<'a> = &'a [T];

    // Tensor storage is a fully materialized non-nullable primitive buffer, so the elements behind
    // a null row are arbitrary values rather than an unresolvable reference.
    const DENSE_SAFE: bool = true;
    // Tensor storage is a primitive buffer; reading it cannot fail on account of its values.
    const DECODE_FALLIBLE: bool = false;

    fn validate(dtype: &DType) -> VortexResult<()> {
        let tensor_match = validate_tensor_float_input(dtype)?;
        let expected = T::PTYPE;
        vortex_ensure_eq!(
            tensor_match.element_ptype(),
            expected,
            "expected a tensor of {expected} elements, got {dtype}",
        );
        Ok(())
    }

    fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Column> {
        let list_size = validate_tensor_float_input(array.dtype())?.list_size() as usize;
        let ext: ExtensionArray = array.execute(ctx)?;
        extract_flat_elements(ext.storage_array(), list_size, ctx)
    }

    fn get(column: &Self::Column, index: usize) -> &[T] {
        column.row::<T>(index)
    }
}
