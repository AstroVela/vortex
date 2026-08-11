// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::dict::TakeReduce;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use crate::DenseUnion;
use crate::DenseUnionArrayExt;
use crate::DenseUnionArraySlotsExt;

impl TakeReduce for DenseUnion {
    fn take(array: ArrayView<'_, Self>, indices: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        let type_ids = array.type_ids().take(indices.clone())?;
        let fill_scalar = Scalar::zero_value(&indices.dtype().as_nonnullable());
        let offset_indices = indices.clone().fill_null(fill_scalar)?;
        let offsets = array.offsets().take(offset_indices)?;

        DenseUnion::try_new(
            type_ids,
            offsets,
            array.variants().clone(),
            array.iter_children().cloned(),
        )
        .map(|array| Some(array.into_array()))
    }
}
