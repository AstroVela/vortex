// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_utils::aliases::hash_map::HashMap;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::expr::BoundLambda;
use crate::expr::VariableRef;
use crate::expression::ApplyCtx;

/// A closed lambda ready to be applied in a higher-order function's invocation domain.
///
/// Captures are ordinary arrays rather than slots in a dedicated higher-order-function array.
/// A higher-order function rebases them into its invocation domain before applying the scalar
/// expression body.
#[derive(Clone, Debug)]
pub struct LambdaCall {
    lambda: BoundLambda,
    captures: Vec<(VariableRef, ArrayRef)>,
}

impl LambdaCall {
    pub(crate) fn new(lambda: BoundLambda, captures: Vec<(VariableRef, ArrayRef)>) -> Self {
        Self { lambda, captures }
    }

    /// The bound lambda expression being invoked.
    pub fn lambda(&self) -> &BoundLambda {
        &self.lambda
    }

    /// Whether this lambda reads any enclosing lexical binding.
    pub fn has_captures(&self) -> bool {
        !self.captures.is_empty()
    }

    /// Apply the lambda to arrays in its invocation domain.
    ///
    /// When `parent_indices` is provided, it maps the invocation rows to the rows of the captured
    /// arrays. A list function, for example, can use it to map flattened element rows back to
    /// their parent list rows before applying the lambda body.
    pub fn apply(
        &self,
        root: ArrayRef,
        args: &[ArrayRef],
        parent_indices: Option<&ArrayRef>,
        execution_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        self.lambda.validate_arguments(&root, args)?;

        let mut bindings = HashMap::new();
        for (variable_ref, array) in &self.captures {
            let array = match parent_indices {
                Some(indices) => array.take(indices.clone())?,
                None => array.clone(),
            };
            bindings.insert(*variable_ref, array);
        }
        for (variable_ref, array) in self.lambda.param_refs().iter().zip(args) {
            bindings.insert(*variable_ref, array.clone());
        }

        self.lambda
            .body()
            .apply(&mut ApplyCtx::with_bindings(root, bindings, execution_ctx))
    }
}
