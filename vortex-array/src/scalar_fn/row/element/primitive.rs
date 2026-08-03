// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::PrimitiveArray;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::scalar_fn::InputElement;
use crate::scalar_fn::OutputElement;
use crate::validity::Validity;

impl<T: NativePType> InputElement for T {
    type Column = Buffer<T>;
    type Varying<'a> = &'a [T];
    type Elem<'a> = T;

    // Every lane of the buffer holds a `T`, valid or not.
    const DENSE_SAFE: bool = true;
    const DECODE_FALLIBLE: bool = false;

    fn validate(dtype: &DType) -> VortexResult<()> {
        let expected = T::PTYPE;
        let DType::Primitive(ptype, _) = dtype else {
            vortex_bail!("expected a {expected} column, got {dtype}");
        };
        vortex_ensure_eq!(
            *ptype,
            expected,
            "expected a {expected} column, got {dtype}"
        );
        Ok(())
    }

    fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Column> {
        Ok(array.execute::<PrimitiveArray>(ctx)?.into_buffer::<T>())
    }

    fn decode_null_tolerant(
        array: ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Self::Column>> {
        Self::decode(array, ctx).map(Some)
    }

    fn get(column: &Self::Column, index: usize) -> T {
        column[index]
    }

    fn varying(column: &Self::Column) -> Self::Varying<'_> {
        column.as_slice()
    }

    fn varying_len(column: &Self::Varying<'_>) -> usize {
        column.len()
    }

    fn get_varying<'a>(column: &Self::Varying<'a>, index: usize) -> T
    where
        Self: 'a,
    {
        column[index]
    }
}

impl<T: NativePType> OutputElement for T {
    fn element_dtype() -> DType {
        DType::Primitive(T::PTYPE, Nullability::NonNullable)
    }

    fn build(values: Vec<Self>) -> ArrayRef {
        PrimitiveArray::new(values, Validity::NonNullable).into_array()
    }

    #[inline(always)]
    fn build_from_fn(row_count: usize, mut apply: impl FnMut(usize) -> Self) -> ArrayRef {
        let mut values = BufferMut::with_capacity(row_count);
        for (index, slot) in values.spare_capacity_mut()[..row_count]
            .iter_mut()
            .enumerate()
        {
            slot.write(apply(index));
        }

        // SAFETY: the loop above initializes every one of the `row_count` slots.
        unsafe { values.set_len(row_count) };
        PrimitiveArray::new(values.freeze(), Validity::NonNullable).into_array()
    }

    #[inline(always)]
    fn try_build_from_fn(
        row_count: usize,
        mut apply: impl FnMut(usize) -> VortexResult<Self>,
    ) -> VortexResult<ArrayRef> {
        let mut values = BufferMut::with_capacity(row_count);
        for (index, slot) in values.spare_capacity_mut()[..row_count]
            .iter_mut()
            .enumerate()
        {
            slot.write(apply(index)?);
        }

        // SAFETY: successful completion of the loop initializes every one of the `row_count`
        // slots. `NativePType` values need no drop, so an early error may safely leave the buffer's
        // logical length at zero.
        unsafe { values.set_len(row_count) };
        Ok(PrimitiveArray::new(values.freeze(), Validity::NonNullable).into_array())
    }

    fn placeholder() -> Self {
        T::default()
    }
}
