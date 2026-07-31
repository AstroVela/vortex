// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ScalarFnArray;
use crate::expr::Expression;
use crate::optimizer::ArrayOptimizer;
use crate::scalar_fn::fns::literal::Literal;

impl ArrayRef {
    /// Apply the expression to this array, producing a new array in constant time.
    pub fn apply(self, expr: &Expression) -> VortexResult<ArrayRef> {
        let scalar_fn = match expr {
            // Root evaluates to the scope, which is this array.
            Expression::Root => return Ok(self),
            Expression::Scalar { scalar_fn, .. } => scalar_fn,
        };

        // Manually convert literals to ConstantArray.
        if let Some(scalar) = expr.as_opt::<Literal>() {
            return Ok(ConstantArray::new(scalar.clone(), self.len()).into_array());
        }

        // Otherwise, collect the child arrays.
        let children: Vec<_> = expr
            .children()
            .iter()
            .map(|e| self.clone().apply(e))
            .try_collect()?;

        // And wrap the scalar function up in an array.
        let array =
            ScalarFnArray::try_new_with_len(scalar_fn.clone(), children, self.len())?.into_array();

        // Optimize the resulting array's root.
        array.optimize()
    }
}
