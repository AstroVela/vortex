// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BufferMut;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::take_slices::TakeSlicesExecute;
use crate::executor::ExecutionCtx;
use crate::match_each_native_ptype;

impl TakeSlicesExecute for Primitive {
    fn take_slices(
        array: ArrayView<'_, Self>,
        slices: &[(usize, usize)],
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let validity = array.validity()?.take_slices(slices)?;
        match_each_native_ptype!(array.ptype(), |T| {
            let source = array.as_slice::<T>();
            let len = slices.iter().map(|(start, end)| end - start).sum();
            let mut values = BufferMut::<T>::with_capacity(len);

            for &(start, end) in slices {
                values.extend_from_slice(&source[start..end]);
            }

            Ok(Some(
                PrimitiveArray::new(values.freeze(), validity).into_array(),
            ))
        })
    }
}
