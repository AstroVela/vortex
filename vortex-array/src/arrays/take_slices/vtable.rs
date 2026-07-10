// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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
use crate::arrays::take_slices::check_selector_arrays;
use crate::arrays::take_slices::selector_constant;
use crate::arrays::take_slices::selector_to_usize;
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
        check_selector_arrays(starts, lengths)?;
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
        let produced_len = append_selected_ranges(array.as_view(), builder.as_mut(), ctx)?;
        vortex_ensure!(
            produced_len == array.len(),
            "TakeSlicesArray produced length {produced_len} does not match declared length {}",
            array.len()
        );
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
) -> VortexResult<usize> {
    let starts = array.starts();
    let lengths = array.lengths();
    check_selector_arrays(starts, lengths)?;

    match_each_unsigned_integer_ptype!(starts.dtype().as_ptype(), |S| {
        match_each_unsigned_integer_ptype!(lengths.dtype().as_ptype(), |L| {
            let start = selector_constant::<S>("start", starts)?;
            let length = selector_constant::<L>("length", lengths)?;
            match (start, length) {
                (Some(start), Some(length)) => append_ranges::<S, L, _, _>(
                    array.child(),
                    starts.len(),
                    |_| start,
                    |_| length,
                    builder,
                    ctx,
                ),
                (Some(start), None) => {
                    let lengths = lengths.clone().execute::<PrimitiveArray>(ctx)?;
                    let lengths = lengths.as_slice::<L>();
                    append_ranges::<S, L, _, _>(
                        array.child(),
                        starts.len(),
                        |_| start,
                        |index| lengths[index],
                        builder,
                        ctx,
                    )
                }
                (None, Some(length)) => {
                    let starts = starts.clone().execute::<PrimitiveArray>(ctx)?;
                    let starts = starts.as_slice::<S>();
                    append_ranges::<S, L, _, _>(
                        array.child(),
                        starts.len(),
                        |index| starts[index],
                        |_| length,
                        builder,
                        ctx,
                    )
                }
                (None, None) => {
                    let starts = starts.clone().execute::<PrimitiveArray>(ctx)?;
                    let lengths = lengths.clone().execute::<PrimitiveArray>(ctx)?;
                    let starts = starts.as_slice::<S>();
                    let lengths = lengths.as_slice::<L>();
                    append_ranges::<S, L, _, _>(
                        array.child(),
                        starts.len(),
                        |index| starts[index],
                        |index| lengths[index],
                        builder,
                        ctx,
                    )
                }
            }
        })
    })
}

fn append_ranges<S, L, StartAt, LengthAt>(
    child: &ArrayRef,
    len: usize,
    mut start_at: StartAt,
    mut length_at: LengthAt,
    builder: &mut dyn ArrayBuilder,
    ctx: &mut ExecutionCtx,
) -> VortexResult<usize>
where
    S: IntegerPType,
    L: IntegerPType,
    StartAt: FnMut(usize) -> S,
    LengthAt: FnMut(usize) -> L,
{
    let mut produced_len = 0usize;
    for index in 0..len {
        let start = selector_to_usize("start", start_at(index))?;
        let length = selector_to_usize("length", length_at(index))?;
        let end = start.checked_add(length).ok_or_else(|| {
            vortex_err!("TakeSlicesArray range overflow for start {start} and length {length}")
        })?;
        child.slice(start..end)?.append_to_builder(builder, ctx)?;
        produced_len = produced_len
            .checked_add(length)
            .ok_or_else(|| vortex_err!("TakeSlicesArray produced length overflow"))?;
    }
    Ok(produced_len)
}

fn scalar_at_selected_range(
    array: ArrayView<'_, TakeSlices>,
    index: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Scalar>> {
    let starts = array.starts();
    let lengths = array.lengths();
    check_selector_arrays(starts, lengths)?;

    match_each_unsigned_integer_ptype!(starts.dtype().as_ptype(), |S| {
        match_each_unsigned_integer_ptype!(lengths.dtype().as_ptype(), |L| {
            let start = selector_constant::<S>("start", starts)?;
            let length = selector_constant::<L>("length", lengths)?;
            match (start, length) {
                (Some(start), Some(length)) => scalar_from_ranges::<S, L, _, _>(
                    array.child(),
                    starts.len(),
                    index,
                    |_| start,
                    |_| length,
                    ctx,
                ),
                (Some(start), None) => {
                    let lengths = lengths.clone().execute::<PrimitiveArray>(ctx)?;
                    let lengths = lengths.as_slice::<L>();
                    scalar_from_ranges::<S, L, _, _>(
                        array.child(),
                        starts.len(),
                        index,
                        |_| start,
                        |range_index| lengths[range_index],
                        ctx,
                    )
                }
                (None, Some(length)) => {
                    let starts = starts.clone().execute::<PrimitiveArray>(ctx)?;
                    let starts = starts.as_slice::<S>();
                    scalar_from_ranges::<S, L, _, _>(
                        array.child(),
                        starts.len(),
                        index,
                        |range_index| starts[range_index],
                        |_| length,
                        ctx,
                    )
                }
                (None, None) => {
                    let starts = starts.clone().execute::<PrimitiveArray>(ctx)?;
                    let lengths = lengths.clone().execute::<PrimitiveArray>(ctx)?;
                    let starts = starts.as_slice::<S>();
                    let lengths = lengths.as_slice::<L>();
                    scalar_from_ranges::<S, L, _, _>(
                        array.child(),
                        starts.len(),
                        index,
                        |range_index| starts[range_index],
                        |range_index| lengths[range_index],
                        ctx,
                    )
                }
            }
        })
    })
}

fn scalar_from_ranges<S, L, StartAt, LengthAt>(
    child: &ArrayRef,
    len: usize,
    index: usize,
    mut start_at: StartAt,
    mut length_at: LengthAt,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Scalar>>
where
    S: IntegerPType,
    L: IntegerPType,
    StartAt: FnMut(usize) -> S,
    LengthAt: FnMut(usize) -> L,
{
    let mut logical_start = 0usize;
    for range_index in 0..len {
        let start = selector_to_usize("start", start_at(range_index))?;
        let length = selector_to_usize("length", length_at(range_index))?;
        let end = start.checked_add(length).ok_or_else(|| {
            vortex_err!("TakeSlicesArray range overflow for start {start} and length {length}")
        })?;
        vortex_ensure!(
            end <= child.len(),
            "TakeSlicesArray range {start}..{end} exceeds child array length {}",
            child.len()
        );
        let logical_end = logical_start
            .checked_add(length)
            .ok_or_else(|| vortex_err!("TakeSlicesArray length overflow"))?;
        if index < logical_end {
            return child
                .execute_scalar(start + (index - logical_start), ctx)
                .map(Some);
        }
        logical_start = logical_end;
    }
    Ok(None)
}
