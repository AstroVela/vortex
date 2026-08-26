// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::dtype::DType;
use crate::expr::BoundExpression;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::fns::cast::Cast;

mod binary;
mod cast;
mod conditional;
mod nulls;
mod structural;

pub(crate) use binary::BinaryBoolean;
pub(crate) use binary::BinaryNullComparison;
pub(crate) use binary::FindBetween;
pub(crate) use cast::CastLiteralOrIdentity;
pub(crate) use conditional::ConstantMask;
pub(crate) use conditional::ConstantZip;
pub(crate) use nulls::CaseWhenToFillNull;
pub(crate) use nulls::RemoveRedundantFillNull;
pub(crate) use structural::GetItemFromPack;
pub(crate) use structural::MergeToPack;
pub(crate) use structural::SelectFromPack;

fn preserve_dtype(replacement: BoundExpression, dtype: &DType) -> VortexResult<BoundExpression> {
    if replacement.dtype() == dtype {
        return Ok(replacement);
    }
    Cast.try_new_bound_expr(dtype.clone(), [replacement])
}
