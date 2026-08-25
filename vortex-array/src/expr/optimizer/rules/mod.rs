// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::RuleRegistry;
use crate::dtype::DType;
use crate::expr::BoundExpression;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::fns::cast::Cast;

mod binary;
mod cast;
mod conditional;
mod nulls;
mod structural;

pub(super) fn register(registry: &mut RuleRegistry) {
    binary::register(registry);
    cast::register(registry);
    structural::register(registry);
    nulls::register(registry);
    conditional::register(registry);
}

fn preserve_dtype(replacement: BoundExpression, dtype: &DType) -> VortexResult<BoundExpression> {
    if replacement.dtype() == dtype {
        return Ok(replacement);
    }
    Cast.try_new_bound_expr(dtype.clone(), [replacement])
}
