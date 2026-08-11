// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::MinMaxPartial;
use super::MinMaxResult;
use super::min_max;
use crate::ExecutionCtx;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::arrays::ExtensionArray;
use crate::arrays::extension::ExtensionArrayExt;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::scalar::Scalar;

pub(super) fn accumulate_extension(
    partial: &mut MinMaxPartial,
    array: &ExtensionArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let non_nullable_ext_dtype = array.ext_dtype().with_nullability(Nullability::NonNullable);
    let Some(MinMaxResult { min, max }) = min_max(
        array.storage_array(),
        ctx,
        NumericalAggregateOpts::default(),
    )?
    else {
        return Ok(());
    };

    let ext_dtype = DType::Extension(non_nullable_ext_dtype);
    let local = match (
        Scalar::try_new(ext_dtype.clone(), min.into_value()),
        Scalar::try_new(ext_dtype, max.into_value()),
    ) {
        (Ok(min), Ok(max)) => Some(MinMaxResult { min, max }),
        _ => None,
    };
    partial.merge(local);
    Ok(())
}
