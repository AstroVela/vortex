// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::arrays::Extension;
use crate::arrays::extension::ExtensionArrayExt;
use crate::dtype::DType;
use crate::scalar::Scalar;

impl OperationsVTable<Extension> for Extension {
    fn scalar_at(
        array: ArrayView<'_, Extension>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let storage_scalar = array.storage_array().execute_scalar(index, ctx)?;
        Scalar::try_new(
            DType::Extension(array.ext_dtype().clone()),
            storage_scalar.into_value(),
        )
    }
}
