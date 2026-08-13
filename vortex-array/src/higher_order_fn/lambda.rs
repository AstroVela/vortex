// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_utils::aliases::hash_map::HashMap;

use crate::ArrayRef;
use crate::expr::ExprApplyCtx;
use crate::expr::TypedLambda;
use crate::expr::VariableRef;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Capture {
    variable_ref: VariableRef,
    slot: usize,
}

/// A type-checked lambda plus the locations of its captured array slots.
///
/// Captures are retained as slots of the enclosing
/// [`HigherOrderFnArray`](crate::arrays::HigherOrderFnArray), so normal array traversal,
/// execution, equality, and explain machinery sees every dependency of the closure.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LambdaClosure {
    lambda: TypedLambda,
    captures: Box<[Capture]>,
}

impl LambdaClosure {
    /// Construct a closure, moving each free lexical binding into an enclosing array slot.
    pub(crate) fn new(
        lambda: TypedLambda,
        captures: Vec<(VariableRef, ArrayRef)>,
        capture_slots: &mut Vec<ArrayRef>,
    ) -> Self {
        let captures = captures
            .into_iter()
            .map(|(variable_ref, array)| {
                let slot = capture_slots.len();
                capture_slots.push(array);
                Capture { variable_ref, slot }
            })
            .collect();
        Self { lambda, captures }
    }

    /// The typed lambda expression.
    pub fn lambda(&self) -> &TypedLambda {
        &self.lambda
    }

    pub(crate) fn validate(&self, capture_count: usize) -> VortexResult<()> {
        vortex_ensure!(
            self.captures
                .iter()
                .all(|capture| capture.slot < capture_count),
            "HigherOrderFnArray lambda capture references a missing slot"
        );
        Ok(())
    }

    pub(crate) fn call<'a>(&'a self, capture_slots: &[ArrayRef]) -> LambdaCall<'a> {
        let captures = self
            .captures
            .iter()
            .map(|capture| (capture.variable_ref, capture_slots[capture.slot].clone()))
            .collect();
        LambdaCall {
            lambda: &self.lambda,
            captures,
        }
    }
}

/// A short-lived invocation view of a [`LambdaClosure`].
///
/// The enclosing higher-order-function array resolves its capture slots before creating this
/// value. Calling [`Self::apply`] builds the lambda body array; it never executes that array.
#[derive(Debug)]
pub struct LambdaCall<'a> {
    lambda: &'a TypedLambda,
    captures: Vec<(VariableRef, ArrayRef)>,
}

impl LambdaCall<'_> {
    /// The typed lambda expression being invoked.
    pub fn lambda(&self) -> &TypedLambda {
        self.lambda
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
            .apply(&mut ExprApplyCtx::with_bindings(root, bindings))
    }
}
