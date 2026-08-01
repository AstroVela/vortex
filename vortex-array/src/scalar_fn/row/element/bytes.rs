// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The byte-string element family: `Utf8` and `Binary` columns, read as bytes or as lengths.

use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::VarBinViewArray;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::scalar_fn::InputElement;
use crate::scalar_fn::OutputElement;

/// Marker for byte-string input elements: accepts `Utf8` or `Binary` columns and presents each
/// row as `&[u8]`.
///
/// Resolving a row means following the offset in its view into a data buffer, which is only
/// meaningful for valid rows, so this element forces
/// [`NullHandling::Filter`](crate::scalar_fn::NullHandling::Filter). Use [`BytesLen`] instead when
/// only the length is needed.
pub struct Bytes;

/// Decoded column form of a [`Bytes`] input: the canonical views array plus its resolved data
/// buffers, supporting cheap per-row byte access.
pub struct BytesColumn {
    /// The canonical views array, read one view per row.
    array: VarBinViewArray,

    /// The array's data buffers, hoisted out of the row loop. These could be re-derived per row
    /// from `array`, but [`Bytes::get`](InputElement::get) runs once per row and resolving a buffer
    /// by index **must** stay a slice index rather than a lookup.
    buffers: Vec<ByteBuffer>,
}

impl InputElement for Bytes {
    type Column = BytesColumn;
    type Elem<'a> = &'a [u8];

    // A view behind a null row may point outside its buffer, or name a buffer that does not
    // exist: `VarBinViewArray` only validates the views of valid rows.
    const DENSE_SAFE: bool = false;
    const DECODE_FALLIBLE: bool = false;

    fn validate(dtype: &DType) -> VortexResult<()> {
        validate_byte_column(dtype)
    }

    fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Column> {
        let array = array.execute::<VarBinViewArray>(ctx)?;
        let buffers = (0..array.data_buffers().len())
            .map(|idx| array.buffer(idx).clone())
            .collect();
        Ok(BytesColumn { array, buffers })
    }

    fn decode_null_tolerant(
        array: ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Self::Column>> {
        Self::decode(array, ctx).map(Some)
    }

    fn get(column: &Self::Column, index: usize) -> &[u8] {
        let view = &column.array.views()[index];
        if view.is_inlined() {
            view.as_inlined().value()
        } else {
            let view = view.as_view();
            &column.buffers[view.buffer_index as usize].as_slice()[view.as_range()]
        }
    }
}

/// Marker for the byte *length* of a `Utf8` or `Binary` row, presented as `usize`.
///
/// Every view stores its own length, so this reads one field and never resolves the row's bytes:
/// cheaper than [`Bytes`], and safe to read densely. Prefer it whenever the length is all a
/// function needs.
pub struct BytesLen;

impl InputElement for BytesLen {
    type Column = VarBinViewArray;
    type Elem<'a> = usize;

    // The length lives in the view itself, so a view behind a null row yields an arbitrary length
    // rather than an unresolvable pointer.
    const DENSE_SAFE: bool = true;
    const DECODE_FALLIBLE: bool = false;

    fn validate(dtype: &DType) -> VortexResult<()> {
        validate_byte_column(dtype)
    }

    fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Column> {
        array.execute::<VarBinViewArray>(ctx)
    }

    fn decode_null_tolerant(
        array: ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Self::Column>> {
        Self::decode(array, ctx).map(Some)
    }

    fn get(column: &Self::Column, index: usize) -> usize {
        column.views()[index].len() as usize
    }
}

/// Shared dtype check for the byte-string element types.
fn validate_byte_column(dtype: &DType) -> VortexResult<()> {
    vortex_ensure!(
        matches!(dtype, DType::Utf8(_) | DType::Binary(_)),
        "expected a Utf8 or Binary column, got {dtype}",
    );
    Ok(())
}

impl OutputElement for String {
    fn element_dtype() -> DType {
        DType::Utf8(Nullability::NonNullable)
    }

    fn build(values: Vec<Self>) -> ArrayRef {
        VarBinViewArray::from_iter_str(values).into_array()
    }

    fn placeholder() -> Self {
        String::new()
    }
}
