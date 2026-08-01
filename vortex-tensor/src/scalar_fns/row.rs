// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! What the tensor scalar functions add to the row-function machinery: an element type that reads a
//! tensor row, a sink that writes one, and the width rule they share.

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
use vortex_array::scalar_fn::OutputSink;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;

use crate::scalar_fns::l2_denorm::build_tensor_array;
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

/// The decoded form of a [`TensorRow`] column: one flat typed buffer plus the stride to read it at.
///
/// Typed at decode time rather than per row. `FlatElements::row` re-derives its typed slice on every
/// call, which costs a ptype check and a buffer downcast per row; a row loop reads every row, so it
/// pays that once here instead.
pub struct TensorRows<T> {
    /// Every row's elements, back to back.
    elements: Buffer<T>,

    /// Elements per row, the length of each row slice.
    list_size: usize,

    /// `list_size` for a full column and `0` for constant-backed storage, so `index * stride` pins a
    /// constant to its single materialized row without a branch in the loop.
    stride: usize,
}

impl<T: Float + NativePType> InputElement for TensorRow<T> {
    type Column = TensorRows<T>;
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
        let flat = extract_flat_elements(ext.storage_array(), list_size, ctx)?;

        Ok(TensorRows {
            list_size: flat.list_size(),
            stride: flat.row_stride(),
            elements: flat.into_buffer::<T>(),
        })
    }

    fn decode_null_tolerant(
        array: ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Self::Column>> {
        Self::decode(array, ctx).map(Some)
    }

    fn get(column: &Self::Column, index: usize) -> &[T] {
        let start = index * column.stride;
        &column.elements.as_slice()[start..start + column.list_size]
    }
}

/// A sink that builds a tensor column shaped like the function's *first* argument, presenting each
/// row as the `&mut [T]` slice of the flat buffer to write.
///
/// A tensor row is exactly what an [`OutputElement`](vortex_array::scalar_fn::OutputElement) cannot
/// carry: its dtype is not a property of `T` alone (the shape lives in the extension metadata) and one
/// owned value per row would mean one allocation per row. Both fall out of writing into a flat buffer
/// allocated once for the batch.
pub struct TensorSink<T> {
    /// The non-nullable tensor extension dtype being built, which [`finish`](OutputSink::finish)
    /// rebuilds the extension array from.
    dtype: DType,

    /// Elements per row, the stride into `elements`.
    list_size: usize,

    /// The row count, kept rather than divided back out of `elements` so a zero-width tensor stays
    /// unambiguous.
    rows: usize,

    /// The flat backing buffer for every row, written in place.
    elements: BufferMut<T>,
}

impl<T: Float + NativePType> OutputSink for TensorSink<T> {
    type Row<'a> = &'a mut [T];

    fn sink_dtype(args: &[DType]) -> VortexResult<DType> {
        let dtype = args
            .first()
            .ok_or_else(|| vortex_err!("a tensor sink takes its shape from a tensor argument"))?;
        let expected = T::PTYPE;
        vortex_ensure_eq!(
            validate_tensor_float_input(dtype)?.element_ptype(),
            expected,
            "expected a tensor of {expected} elements, got {dtype}",
        );
        Ok(dtype.as_nonnullable())
    }

    fn with_capacity(rows: usize, dtype: &DType) -> VortexResult<Self> {
        let list_size = validate_tensor_float_input(dtype)?.list_size() as usize;
        let total = rows.checked_mul(list_size).ok_or_else(|| {
            vortex_err!("tensor sink of {rows} rows x {list_size} elements overflows usize")
        })?;

        // `zeroed` rather than uninitialized spare capacity: the row loop overwrites every element, so
        // this is only to keep the buffer safely indexable. Large allocations come back zeroed from the
        // allocator, so it costs no separate pass.
        Ok(Self {
            dtype: dtype.clone(),
            list_size,
            rows,
            elements: BufferMut::zeroed(total),
        })
    }

    fn row(&mut self, index: usize) -> &mut [T] {
        &mut self.elements.as_mut_slice()[index * self.list_size..][..self.list_size]
    }

    fn finish(self) -> VortexResult<ArrayRef> {
        build_tensor_array(
            self.dtype,
            self.list_size,
            self.rows,
            Validity::NonNullable,
            self.elements.freeze(),
        )
    }
}

/// Test-only probe recording which operands the last `prepare` step saw as batch-constant, so a
/// test can assert its inputs took the stride-0 decode path rather than merely producing the right
/// values through the varying path.
#[cfg(test)]
pub(crate) mod probe {
    use std::cell::Cell;

    thread_local! {
        /// Bitmask of the constant operands the last `prepare` saw (bit 0 for the lhs, bit 1 for
        /// the rhs). Thread-local rather than a process global so concurrent tests in one process
        /// (plain `cargo test`) cannot race it; execution runs on the calling thread.
        pub(crate) static SEEN_CONSTANTS: Cell<u8> = const { Cell::new(u8::MAX) };
    }

    /// Record which operands `prepare` saw as constant.
    pub(crate) fn record(lhs_constant: bool, rhs_constant: bool) {
        SEEN_CONSTANTS.set(u8::from(lhs_constant) | (u8::from(rhs_constant) << 1));
    }
}
