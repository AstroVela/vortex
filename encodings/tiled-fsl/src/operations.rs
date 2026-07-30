// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_native_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::TiledFixedSizeList;
use crate::TiledFixedSizeListArrayExt;
use crate::TiledFixedSizeListArraySlotsExt;
use crate::geometry::physical_offset;

impl OperationsVTable<TiledFixedSizeList> for TiledFixedSizeList {
    fn scalar_at(
        array: ArrayView<'_, TiledFixedSizeList>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let list_size = array.list_size() as usize;
        let indices = (0..list_size)
            .map(|dimension| {
                let offset =
                    physical_offset(array.len(), list_size, array.geometry(), index, dimension)?;
                Ok(u64::try_from(offset)?)
            })
            .collect::<VortexResult<Buffer<u64>>>()?;
        let row = array
            .elements()
            .take(PrimitiveArray::from_iter(indices).into_array())?
            .execute::<PrimitiveArray>(ctx)?;
        let element_dtype = row.dtype().clone();
        let element_nullability = element_dtype.nullability();
        let mask = row.validity()?.execute_mask(row.len(), ctx)?;

        let children = match_each_native_ptype!(row.ptype(), |T| {
            let values = row.as_slice::<T>();
            match mask {
                Mask::AllTrue(_) => values
                    .iter()
                    .copied()
                    .map(|value| Scalar::primitive(value, element_nullability))
                    .collect(),
                Mask::AllFalse(_) => (0..values.len())
                    .map(|_| Scalar::null(element_dtype.clone()))
                    .collect(),
                Mask::Values(validity) => values
                    .iter()
                    .copied()
                    .zip(validity.bit_buffer().iter())
                    .map(|(value, is_valid)| {
                        if is_valid {
                            Scalar::primitive(value, Nullability::Nullable)
                        } else {
                            Scalar::null(element_dtype.clone())
                        }
                    })
                    .collect(),
            }
        });

        Ok(Scalar::fixed_size_list(
            element_dtype,
            children,
            array.dtype().nullability(),
        ))
    }
}
