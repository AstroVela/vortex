// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::expr::BoundExpression;
use crate::expr::ExprApplyCtx;
use crate::expr::Expression;

impl ArrayRef {
    /// Apply a bound expression to this array, producing a new array in constant time.
    pub fn apply_bound(self, expr: &BoundExpression) -> VortexResult<ArrayRef> {
        expr.apply(&mut ExprApplyCtx::new(self))
    }

    /// Apply the expression to this array, producing a new array in constant time.
    pub fn apply(self, expr: &Expression) -> VortexResult<ArrayRef> {
        let bound = expr.bind(self.dtype())?;
        self.apply_bound(&bound)
    }
}
