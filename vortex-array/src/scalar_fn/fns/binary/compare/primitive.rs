// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Primitive comparison execution through [`RowFn`] and fused lane kernels.
//!
//! Production uses the RowFn implementation on x86-64, where it has explicit SIMD kernels. Other
//! targets retain the fused columnar implementation until RowFn has a competitive target-specific
//! kernel.

#[cfg(any(not(target_arch = "x86_64"), test, feature = "_test-harness"))]
mod columnar;
mod simd;

use half::f16;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::dtype::PType;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::VecExecutionArgs;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::scalar_fn::unstable::row::RowFn;
use crate::scalar_fn::unstable::row::RowVisitor;
use crate::scalar_fn::unstable::row::execute_rows;

/// Compare two primitive arrays of the same [`PType`].
///
/// Floats compare with Vortex's total ordering, including signed zero and ordered NaN bit
/// patterns. Equality is bitwise.
pub(super) fn compare_primitive(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: CompareOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    #[cfg(target_arch = "x86_64")]
    {
        compare_primitive_rows(lhs, rhs, op, ctx)
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        columnar::compare_primitive(lhs, rhs, op, ctx)
    }
}

/// Compare primitives through the retained columnar benchmark baseline.
#[cfg(any(test, feature = "_test-harness"))]
pub(crate) fn compare_primitive_columnar(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: CompareOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    columnar::compare_primitive(lhs, rhs, op, ctx)
}

pub(crate) fn compare_primitive_rows(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: CompareOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let args = VecExecutionArgs::new(vec![lhs.clone(), rhs.clone()], lhs.len());

    execute_rows(&PrimitiveCompare, &op, &args, ctx)
}

/// Internal row execution for primitive comparison operators.
#[derive(Clone)]
struct PrimitiveCompare;

impl RowFn for PrimitiveCompare {
    type Options = CompareOperator;

    const ARG_NAMES: &'static [&'static str] = &["lhs", "rhs"];
    const INFALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        // `PrimitiveCompare` is a private implementation detail of `Binary`: it is never registered
        // or serialized independently. Reusing the public ID keeps execution errors attributed to
        // `Binary`. If this type becomes registrable, it needs its own ID and persistence contract.
        ScalarFnVTable::id(&Binary)
    }

    fn dispatch<V: RowVisitor<Self::Options>>(
        &self,
        op: &Self::Options,
        args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        let ptype =
            PType::try_from(args.first().ok_or_else(|| {
                vortex_err!("a comparison operator takes two operands, got none")
            })?)?;

        match ptype {
            PType::U8 => visit_compare_simd::<u8, V>(*op, visitor),
            PType::U16 => visit_compare_simd::<u16, V>(*op, visitor),
            PType::U32 => visit_compare_simd::<u32, V>(*op, visitor),
            PType::I64 => visit_compare_simd::<i64, V>(*op, visitor),
            PType::U64 => visit_compare_simd::<u64, V>(*op, visitor),
            PType::I8 => visit_compare_simd::<i8, V>(*op, visitor),
            PType::I16 => visit_compare_simd::<i16, V>(*op, visitor),
            PType::I32 => visit_compare_simd::<i32, V>(*op, visitor),
            PType::F16 => visit_compare_simd::<f16, V>(*op, visitor),
            PType::F32 => visit_compare_simd::<f32, V>(*op, visitor),
            PType::F64 => visit_compare_simd::<f64, V>(*op, visitor),
        }
    }
}

fn visit_compare_simd<T, V>(op: CompareOperator, visitor: V) -> VortexResult<V::VisitResult>
where
    T: simd::SimdCompare,
    V: RowVisitor<CompareOperator>,
{
    visitor.visit_kernel::<(T, T), _>(simd::PrimitiveComparisonKernel::new(op))
}
