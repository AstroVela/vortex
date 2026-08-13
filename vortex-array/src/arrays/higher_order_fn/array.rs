// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::ArraySlots;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::TypedArrayRef;
use crate::arrays::HigherOrderFn;
use crate::higher_order_fn::HigherOrderFunctionRef;
use crate::higher_order_fn::LambdaClosure;

/// Per-array data for [`HigherOrderFnArray`].
#[derive(Clone, Debug)]
pub struct HigherOrderFnData {
    pub(super) higher_order_fn: HigherOrderFunctionRef,
    pub(super) arg_count: usize,
    pub(super) lambdas: Box<[LambdaClosure]>,
}

impl Display for HigherOrderFnData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "higher_order_fn: {}", self.higher_order_fn)
    }
}

pub trait HigherOrderFnArrayExt: TypedArrayRef<HigherOrderFn> {
    /// The function represented by this array.
    fn higher_order_fn(&self) -> &HigherOrderFunctionRef {
        &self.higher_order_fn
    }

    /// The number of ordinary function arguments, before capture slots.
    fn arg_count(&self) -> usize {
        self.arg_count
    }

    /// The lambdas closed over the capture slots.
    fn lambdas(&self) -> &[LambdaClosure] {
        &self.lambdas
    }

    /// The function's ordinary argument arrays.
    fn args(&self) -> Vec<ArrayRef> {
        self.as_ref().slots()[..self.arg_count()]
            .iter()
            .map(|slot| {
                slot.as_ref()
                    .vortex_expect("HigherOrderFnArray argument slot")
                    .clone()
            })
            .collect()
    }

    /// Arrays captured by its lambdas, in closure-slot order.
    fn captures(&self) -> Vec<ArrayRef> {
        self.as_ref().slots()[self.arg_count()..]
            .iter()
            .map(|slot| {
                slot.as_ref()
                    .vortex_expect("HigherOrderFnArray capture slot")
                    .clone()
            })
            .collect()
    }
}
impl<T: TypedArrayRef<HigherOrderFn>> HigherOrderFnArrayExt for T {}

impl Array<HigherOrderFn> {
    /// Build a lazy higher-order-function array from ordinary arguments and lexical closures.
    pub(crate) fn try_new_with_len(
        higher_order_fn: HigherOrderFunctionRef,
        args: Vec<ArrayRef>,
        lambdas: Vec<LambdaClosure>,
        capture_slots: Vec<ArrayRef>,
        len: usize,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            args.iter().all(|arg| arg.len() == len),
            "HigherOrderFnArray arguments must have the array length"
        );
        vortex_ensure!(
            higher_order_fn.arity().matches(args.len()),
            "{} takes {} ordinary arguments, got {}",
            higher_order_fn,
            higher_order_fn.arity(),
            args.len()
        );
        vortex_ensure!(
            lambdas.len() == higher_order_fn.lambda_arity(),
            "{} takes {} lambda arguments, got {}",
            higher_order_fn,
            higher_order_fn.lambda_arity(),
            lambdas.len()
        );

        let arg_dtypes = args
            .iter()
            .map(|arg| arg.dtype().clone())
            .collect::<Vec<_>>();
        let arg_count = args.len();
        let lambdas = lambdas.into_boxed_slice();
        let mut slots = args;
        slots.extend(capture_slots);
        let bound_lambdas = lambdas
            .iter()
            .map(|lambda| lambda.lambda().clone())
            .collect::<Vec<_>>();
        let dtype = higher_order_fn.return_dtype(&arg_dtypes, &bound_lambdas)?;
        let data = HigherOrderFnData {
            higher_order_fn: higher_order_fn.clone(),
            arg_count,
            lambdas,
        };
        let vtable = HigherOrderFn {
            id: higher_order_fn.id(),
        };
        Ok(unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(vtable, dtype, len, data)
                    .with_slots(slots.into_iter().map(Some).collect::<ArraySlots>()),
            )
        })
    }
}
