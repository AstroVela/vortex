// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_alp::ALP;
use vortex_alp::ALPArrayExt;
use vortex_alp::ALPFloat;
use vortex_array::ArrayRef;
use vortex_array::ArrayVTable;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::PType;
use vortex_array::dtype::half::f16;
use vortex_array::kernel::ExecuteParentKernel;
use vortex_array::optimizer::kernels::ArrayKernelsExt;
use vortex_block_residual::OrderedFloat;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use crate::RangePacked;

pub(super) fn initialize(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_execute_parent_kernel(
        OrderedFloat.id(),
        RangePacked,
        OrderedFloatRangePackedKernel,
    );
    kernels.register_execute_parent_kernel(ALP.id(), RangePacked, ALPRangePackedKernel);
}

#[derive(Debug)]
struct OrderedFloatRangePackedKernel;

impl ExecuteParentKernel<RangePacked> for OrderedFloatRangePackedKernel {
    type Parent = OrderedFloat;

    #[expect(
        clippy::cast_possible_truncation,
        reason = "OrderedFloat validates the child integer width against the float width"
    )]
    fn execute_parent(
        &self,
        array: ArrayView<'_, RangePacked>,
        parent: ArrayView<'_, OrderedFloat>,
        child_idx: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if child_idx != 0 {
            return Ok(None);
        }

        let validity = array.validity()?;
        let decoded = match parent.dtype().as_ptype() {
            PType::F16 => PrimitiveArray::new::<f16>(
                RangePacked::decode_mapped(
                    array,
                    |value| f16::from_bits(unordered_u16(value as u16)),
                    f16::ZERO,
                )?,
                validity,
            ),
            PType::F32 => PrimitiveArray::new::<f32>(
                RangePacked::decode_mapped(
                    array,
                    |value| f32::from_bits(unordered_u32(value as u32)),
                    0.0,
                )?,
                validity,
            ),
            PType::F64 => PrimitiveArray::new::<f64>(
                RangePacked::decode_mapped(
                    array,
                    |value| f64::from_bits(unordered_u64(value)),
                    0.0,
                )?,
                validity,
            ),
            ptype => vortex_bail!("OrderedFloat RangePacked kernel does not support {ptype}"),
        };
        Ok(Some(decoded.into_array()))
    }
}

#[derive(Debug)]
struct ALPRangePackedKernel;

impl ExecuteParentKernel<RangePacked> for ALPRangePackedKernel {
    type Parent = ALP;

    #[expect(
        clippy::cast_possible_truncation,
        reason = "ALP validates the child integer width against the float width"
    )]
    fn execute_parent(
        &self,
        array: ArrayView<'_, RangePacked>,
        parent: ArrayView<'_, ALP>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if child_idx != 0 {
            return Ok(None);
        }

        let validity = array.validity()?;
        let exponents = parent.exponents();
        let decoded = match parent.dtype().as_ptype() {
            PType::F32 => PrimitiveArray::new::<f32>(
                RangePacked::decode_mapped(
                    array,
                    |ordered| {
                        let encoded = ((ordered as u32) ^ (1_u32 << 31)) as i32;
                        <f32 as ALPFloat>::decode_single(encoded, exponents)
                    },
                    0.0,
                )?,
                validity,
            ),
            PType::F64 => PrimitiveArray::new::<f64>(
                RangePacked::decode_mapped(
                    array,
                    |ordered| {
                        let encoded = (ordered ^ (1_u64 << 63)) as i64;
                        <f64 as ALPFloat>::decode_single(encoded, exponents)
                    },
                    0.0,
                )?,
                validity,
            ),
            ptype => vortex_bail!("ALP RangePacked kernel does not support {ptype}"),
        };
        let decoded = if let Some(patches) = parent.patches() {
            decoded.patch(&patches, ctx)?
        } else {
            decoded
        };
        Ok(Some(decoded.into_array()))
    }
}

#[inline]
fn unordered_u16(value: u16) -> u16 {
    if value & (1_u16 << 15) == 0 {
        !value
    } else {
        value ^ (1_u16 << 15)
    }
}

#[inline]
fn unordered_u32(value: u32) -> u32 {
    if value & (1_u32 << 31) == 0 {
        !value
    } else {
        value ^ (1_u32 << 31)
    }
}

#[inline]
fn unordered_u64(value: u64) -> u64 {
    if value & (1_u64 << 63) == 0 {
        !value
    } else {
        value ^ (1_u64 << 63)
    }
}
