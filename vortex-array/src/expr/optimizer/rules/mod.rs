// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::BoundExpressionOptimizer;
use crate::dtype::DType;
use crate::expr::BoundExpression;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::fns::cast::Cast;

mod binary;
mod cast;
mod conditional;
mod nulls;
mod structural;

pub(super) fn register(optimizer: &mut BoundExpressionOptimizer) {
    binary::register(optimizer);
    cast::register(optimizer);
    structural::register(optimizer);
    nulls::register(optimizer);
    conditional::register(optimizer);
}

fn preserve_dtype(replacement: BoundExpression, dtype: &DType) -> VortexResult<BoundExpression> {
    if replacement.dtype() == dtype {
        return Ok(replacement);
    }
    Cast.try_new_bound_expr(dtype.clone(), [replacement])
}
