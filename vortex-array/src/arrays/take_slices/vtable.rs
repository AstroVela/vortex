// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::hash::Hasher;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayParts;
use crate::ArrayRef;
use crate::EqMode;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::array::VTable;
use crate::array::ValidityVTable;
use crate::array::with_empty_buffers;
use crate::arrays::take_slices::TakeSlicesArrayExt;
use crate::arrays::take_slices::array::CHILD_SLOT;
use crate::arrays::take_slices::array::NUM_SLOTS;
use crate::arrays::take_slices::array::SLOT_NAMES;
use crate::arrays::take_slices::array::TakeSlicesData;
use crate::arrays::take_slices::rules::PARENT_RULES;
use crate::arrays::take_slices::rules::RULES;
use crate::buffer::BufferHandle;
use crate::builders::builder_with_capacity_in;
use crate::dtype::DType;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionResult;
use crate::scalar::Scalar;
use crate::serde::ArrayChildren;
use crate::validity::Validity;

/// A [`TakeSlices`]-encoded Vortex array.
pub type TakeSlicesArray = Array<TakeSlices>;

/// Child-range sequence selection encoding.
#[derive(Clone, Debug)]
pub struct TakeSlices;

impl ArrayHash for TakeSlicesData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        self.slices.hash(state);
    }
}

impl ArrayEq for TakeSlicesData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.slices == other.slices
    }
}

impl VTable for TakeSlices {
    type TypedArrayData = TakeSlicesData;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.take_slices");
        *ID
    }

    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
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
        let child = slots[CHILD_SLOT]
            .as_ref()
            .vortex_expect("validated child slot");
        vortex_ensure!(
            child.dtype() == dtype,
            "TakeSlicesArray dtype {} does not match outer dtype {}",
            child.dtype(),
            dtype
        );
        let mut computed_len = 0usize;
        for &(start, end) in data.slices() {
            vortex_ensure!(
                start < end,
                "TakeSlicesArray range must be non-empty: {start}..{end}"
            );
            vortex_ensure!(
                end <= child.len(),
                "TakeSlicesArray range {start}..{end} exceeds child length {}",
                child.len()
            );
            computed_len = computed_len
                .checked_add(end - start)
                .ok_or_else(|| vortex_err!("TakeSlicesArray length overflow"))?;
        }
        vortex_ensure!(
            data.len() == computed_len,
            "TakeSlicesArray metadata length {} does not match computed range length {}",
            data.len(),
            computed_len
        );
        vortex_ensure!(
            computed_len == len,
            "TakeSlicesArray computed length {} does not match outer length {}",
            computed_len,
            len
        );
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
        for range in array.slice_ranges() {
            let slice = array.child().slice(range)?;
            slice.append_to_builder(builder.as_mut(), ctx)?;
        }
        Ok(ExecutionResult::done(builder.finish()))
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn reduce(array: ArrayView<'_, Self>) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array)
    }
}

impl OperationsVTable<TakeSlices> for TakeSlices {
    fn scalar_at(
        array: ArrayView<'_, TakeSlices>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let mut logical_start = 0usize;
        for &(start, end) in array.slices() {
            let len = end - start;
            let logical_end = logical_start + len;
            if index < logical_end {
                return array
                    .child()
                    .execute_scalar(start + (index - logical_start), ctx);
            }
            logical_start = logical_end;
        }

        vortex_panic!("TakeSlicesArray scalar index {index} out of bounds")
    }
}

impl ValidityVTable<TakeSlices> for TakeSlices {
    fn validity(array: ArrayView<'_, TakeSlices>) -> VortexResult<Validity> {
        array.child().validity()?.take_slices(array.slices())
    }
}
