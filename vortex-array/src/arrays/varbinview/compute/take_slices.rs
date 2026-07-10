// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use itertools::Itertools as _;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBinView;
use crate::arrays::VarBinViewArray;
use crate::arrays::take_slices::TakeSlicesKernel;
use crate::arrays::take_slices::check_index_arrays;
use crate::arrays::take_slices::index_value_to_usize;
use crate::arrays::take_slices::validate_index_ranges;
use crate::arrays::varbinview::BinaryView;
use crate::buffer::BufferHandle;
use crate::dtype::IntegerPType;
use crate::executor::ExecutionCtx;
use crate::match_each_unsigned_integer_ptype;

impl TakeSlicesKernel for VarBinView {
    fn take_slices(
        array: ArrayView<'_, Self>,
        starts: &ArrayRef,
        lengths: &ArrayRef,
        output_len: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        check_index_arrays(starts, lengths)?;

        match_each_unsigned_integer_ptype!(starts.dtype().as_ptype(), |S| {
            match_each_unsigned_integer_ptype!(lengths.dtype().as_ptype(), |L| {
                take_slices_typed::<S, L>(array, starts, lengths, output_len, ctx)
            })
        })
        .map(Some)
    }
}

fn take_slices_typed<S, L>(
    array: ArrayView<'_, VarBinView>,
    starts: &ArrayRef,
    lengths: &ArrayRef,
    output_len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    S: IntegerPType,
    L: IntegerPType,
{
    let starts = starts.clone().execute::<PrimitiveArray>(ctx)?;
    let lengths = lengths.clone().execute::<PrimitiveArray>(ctx)?;
    let views = gather_views(
        array.views(),
        starts.as_slice::<S>(),
        lengths.as_slice::<L>(),
        output_len,
    )?;

    let starts = starts.into_array();
    let lengths = lengths.into_array();
    let validity = array
        .validity()?
        .take_slices(&starts, &lengths, output_len)?;

    // SAFETY: ranges were validated against the source views, and copied views still reference the
    // same backing data buffers.
    unsafe {
        Ok(VarBinViewArray::new_handle_unchecked(
            BufferHandle::new_host(views.into_byte_buffer()),
            Arc::clone(array.data_buffers()),
            array.dtype().clone(),
            validity,
        )
        .into_array())
    }
}

fn gather_views<S, L>(
    source: &[BinaryView],
    starts: &[S],
    lengths: &[L],
    output_len: usize,
) -> VortexResult<Buffer<BinaryView>>
where
    S: IntegerPType,
    L: IntegerPType,
{
    validate_index_ranges(source.len(), starts, lengths, output_len)?;

    let mut views = BufferMut::<BinaryView>::with_capacity(output_len);
    for (&start, &length) in starts.iter().zip_eq(lengths) {
        let start = index_value_to_usize("start", start)?;
        let length = index_value_to_usize("length", length)?;
        let end = start + length;
        views.extend_from_slice(&source[start..end]);
    }

    Ok(views.freeze())
}
