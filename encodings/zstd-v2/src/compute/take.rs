// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::match_each_integer_ptype;
use vortex_error::VortexResult;

use crate::ZstdV2;
use crate::array::unsliced_validity;

impl TakeExecute for ZstdV2 {
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let indices = indices.clone().execute::<PrimitiveArray>(ctx)?;
        let index_mask = indices.validity()?.execute_mask(indices.len(), ctx)?;
        let rows: Vec<usize> = match_each_integer_ptype!(indices.ptype(), |P| {
            indices
                .as_slice::<P>()
                .iter()
                .map(|index| index.as_())
                .collect()
        });

        let validity = unsliced_validity(array);
        array
            .data()
            .take(array.dtype(), &validity, &rows, &index_mask, ctx)
            .map(Some)
    }
}
