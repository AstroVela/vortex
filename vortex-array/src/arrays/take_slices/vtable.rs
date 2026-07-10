// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools as _;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayParts;
use crate::ArrayRef;
use crate::EmptyArrayData;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::array::VTable;
use crate::array::ValidityVTable;
use crate::array::with_empty_buffers;
use crate::arrays::PrimitiveArray;
use crate::arrays::take_slices::TakeSlicesArrayExt;
use crate::arrays::take_slices::array::CHILD_SLOT;
use crate::arrays::take_slices::array::LENGTHS_SLOT;
use crate::arrays::take_slices::array::NUM_SLOTS;
use crate::arrays::take_slices::array::SLOT_NAMES;
use crate::arrays::take_slices::array::STARTS_SLOT;
use crate::arrays::take_slices::check_index_arrays;
use crate::arrays::take_slices::checked_range_end;
use crate::arrays::take_slices::index_value_to_usize;
use crate::arrays::take_slices::validate_index_ranges;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::builders::builder_with_capacity_in;
use crate::dtype::DType;
use crate::dtype::IntegerPType;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionResult;
use crate::match_each_unsigned_integer_ptype;
use crate::scalar::Scalar;
use crate::serde::ArrayChildren;
use crate::validity::Validity;

/// A [`TakeSlices`]-encoded Vortex array.
pub type TakeSlicesArray = Array<TakeSlices>;

/// Contiguous-range gather selection encoding.
///
/// Like [`crate::arrays::Slice`], this is a lazy compute encoding and is not serialized as a file
/// encoding. Execute it before a file-writing path when a materialized physical representation is
/// required.
#[derive(Clone, Debug)]
pub struct TakeSlices;

impl VTable for TakeSlices {
    type TypedArrayData = EmptyArrayData;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.take_slices");
        *ID
    }

    fn validate(
        &self,
        _data: &Self::TypedArrayData,
        dtype: &DType,
        _len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == NUM_SLOTS,
            "TakeSlicesArray expected {NUM_SLOTS} slots, found {}",
            slots.len()
        );
        vortex_ensure!(
            slots[CHILD_SLOT].is_some(),
            "TakeSlicesArray child slot must be present"
        );
        vortex_ensure!(
            slots[STARTS_SLOT].is_some(),
            "TakeSlicesArray starts slot must be present"
        );
        vortex_ensure!(
            slots[LENGTHS_SLOT].is_some(),
            "TakeSlicesArray lengths slot must be present"
        );
        let child = slots[CHILD_SLOT]
            .as_ref()
            .vortex_expect("validated child slot");
        vortex_ensure!(
            child.dtype() == dtype,
            "TakeSlicesArray dtype {} does not match outer dtype {}",
            child.dtype(),
            dtype
        );
        let starts = slots[STARTS_SLOT]
            .as_ref()
            .vortex_expect("validated starts slot");
        let lengths = slots[LENGTHS_SLOT]
            .as_ref()
            .vortex_expect("validated lengths slot");
        check_index_arrays(starts, lengths)?;
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, _idx: usize) -> BufferHandle {
        vortex_panic!("TakeSlicesArray has no buffers")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        with_empty_buffers(self, array, buffers)
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        vortex_bail!("TakeSlices array is not serializable")
    }

    fn deserialize(
        &self,
        _dtype: &DType,
        _len: usize,
        _metadata: &[u8],
        _buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_bail!("TakeSlices array is not serializable")
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let mut builder = builder_with_capacity_in(ctx.allocator(), array.dtype(), array.len());
        append_selected_ranges(array.as_view(), builder.as_mut(), ctx)?;
        Ok(ExecutionResult::done(builder.finish()))
    }
}

impl OperationsVTable<TakeSlices> for TakeSlices {
    fn scalar_at(
        array: ArrayView<'_, TakeSlices>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        scalar_at_selected_range(array, index, ctx)?
            .ok_or_else(|| vortex_err!("TakeSlicesArray scalar index {index} out of bounds"))
    }
}

impl ValidityVTable<TakeSlices> for TakeSlices {
    fn validity(array: ArrayView<'_, TakeSlices>) -> VortexResult<Validity> {
        array
            .child()
            .validity()?
            .take_slices(array.starts(), array.lengths(), array.len())
    }
}

fn append_selected_ranges(
    array: ArrayView<'_, TakeSlices>,
    builder: &mut dyn ArrayBuilder,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let starts = array.starts();
    let lengths = array.lengths();
    check_index_arrays(starts, lengths)?;

    match_each_unsigned_integer_ptype!(starts.dtype().as_ptype(), |S| {
        match_each_unsigned_integer_ptype!(lengths.dtype().as_ptype(), |L| {
            let starts = starts.clone().execute::<PrimitiveArray>(ctx)?;
            let lengths = lengths.clone().execute::<PrimitiveArray>(ctx)?;
            append_array_ranges(
                array.child(),
                starts.as_slice::<S>(),
                lengths.as_slice::<L>(),
                array.len(),
                builder,
                ctx,
            )
        })
    })
}

fn append_array_ranges<S, L>(
    child: &ArrayRef,
    starts: &[S],
    lengths: &[L],
    output_len: usize,
    builder: &mut dyn ArrayBuilder,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()>
where
    S: IntegerPType,
    L: IntegerPType,
{
    validate_index_ranges(child.len(), starts, lengths, output_len)?;

    for (&start, &length) in starts.iter().zip_eq(lengths) {
        let start = index_value_to_usize("start", start)?;
        let length = index_value_to_usize("length", length)?;
        let end = start + length;
        child.slice(start..end)?.append_to_builder(builder, ctx)?;
    }

    Ok(())
}

fn scalar_at_selected_range(
    array: ArrayView<'_, TakeSlices>,
    index: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Scalar>> {
    let starts = array.starts();
    let lengths = array.lengths();
    check_index_arrays(starts, lengths)?;

    match_each_unsigned_integer_ptype!(starts.dtype().as_ptype(), |S| {
        match_each_unsigned_integer_ptype!(lengths.dtype().as_ptype(), |L| {
            let starts = starts.clone().execute::<PrimitiveArray>(ctx)?;
            let lengths = lengths.clone().execute::<PrimitiveArray>(ctx)?;
            scalar_from_array_ranges(
                array.child(),
                index,
                starts.as_slice::<S>(),
                lengths.as_slice::<L>(),
                ctx,
            )
        })
    })
}

fn scalar_from_array_ranges<S, L>(
    child: &ArrayRef,
    index: usize,
    starts: &[S],
    lengths: &[L],
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Scalar>>
where
    S: IntegerPType,
    L: IntegerPType,
{
    let mut logical_start = 0usize;
    for (&start, &length) in starts.iter().zip_eq(lengths) {
        let start = index_value_to_usize("start", start)?;
        let length = index_value_to_usize("length", length)?;
        if let Some(scalar) = scalar_from_range(child, logical_start, index, start, length, ctx)? {
            return Ok(Some(scalar));
        }
        logical_start = logical_start
            .checked_add(length)
            .ok_or_else(|| vortex_err!("TakeSlicesArray logical length overflow"))?;
    }
    Ok(None)
}

fn scalar_from_range(
    child: &ArrayRef,
    logical_start: usize,
    index: usize,
    start: usize,
    length: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Scalar>> {
    let end = checked_range_end(start, length)?;
    vortex_ensure!(
        end <= child.len(),
        "TakeSlicesArray range {start}..{end} exceeds child array length {}",
        child.len()
    );
    let logical_end = logical_start
        .checked_add(length)
        .ok_or_else(|| vortex_err!("TakeSlicesArray logical length overflow"))?;
    if index < logical_end {
        return child
            .execute_scalar(start + (index - logical_start), ctx)
            .map(Some);
    }
    Ok(None)
}
