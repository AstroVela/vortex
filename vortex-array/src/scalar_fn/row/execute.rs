// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The three pieces every row function is built from, whatever its `dispatch` chooses.
//!
//! These back the blanket impls in [`row_fn`](super::row_fn) and are deliberately not public:
//! [`RowFn`](crate::scalar_fn::RowFn) is the abstraction, these are its internals.

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::scalar_fn::ApplyResult;
use crate::scalar_fn::ElementTuple;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::NullHandling;
use crate::scalar_fn::OutputElement;

/// Validate the input dtypes of a row function and return its output element dtype.
///
/// [`StrictScalarFnVTable`](crate::scalar_fn::StrictScalarFnVTable) widens the result to nullable
/// iff any input is nullable, so this ignores nullability.
pub(super) fn validate_row_args<A: ElementTuple, R: ApplyResult>(
    args: &[DType],
) -> VortexResult<DType> {
    A::validate(args)?;
    Ok(R::Out::element_dtype())
}

/// The [`NullHandling`] a row function gets, from what its arguments and return type allow.
///
/// [`NullHandling::Dense`] is cheaper and the only option that leaves inputs at their original
/// encoding, so it is used whenever it is sound: every argument readable behind a null row, and a
/// row computation that cannot fail. Both conditions are read off the types, so a row function never
/// declares its null handling.
pub(super) const fn row_null_handling<A: ElementTuple, R: ApplyResult>() -> NullHandling {
    if A::DENSE_SAFE && !row_is_fallible::<A, R>() {
        NullHandling::Dense
    } else {
        NullHandling::Filter
    }
}

/// Whether a row function can fail on legal data, from either of the two places fallibility can
/// live: parsing an argument ([`InputElement::DECODE_FALLIBLE`]), or the row computation itself
/// returning a [`VortexResult`](vortex_error::VortexResult).
///
/// Deriving it from both is what keeps
/// [`is_fallible`](crate::scalar_fn::ScalarFnVTable::is_fallible) honest for an element that parses
/// its bytes: a geometry column is fallible however total the kernel over it is.
///
/// [`InputElement::DECODE_FALLIBLE`]: crate::scalar_fn::InputElement::DECODE_FALLIBLE
pub(super) const fn row_is_fallible<A: ElementTuple, R: ApplyResult>() -> bool {
    A::DECODE_FALLIBLE || R::FALLIBLE
}

/// Decode every input column once, then compute one output element per row.
///
/// Monomorphic in `A` and `R`, so `apply` inlines and the loop is free to auto-vectorize.
pub(super) fn execute_row_loop<A: ElementTuple, R: ApplyResult>(
    args: &dyn ExecutionArgs,
    ctx: &mut ExecutionCtx,
    apply: impl Fn(A::Elems<'_>) -> R,
) -> VortexResult<ArrayRef> {
    let columns = A::decode(args, ctx)?;
    let rows = 0..args.row_count();

    // `R::FALLIBLE` is a constant, so only one of these survives monomorphization. The infallible
    // arm must not collect through `Result`, which would cost the loop its unconditional shape;
    // there `into_result` inlines to `Ok(value)` and the unwrap folds away.
    let values = if R::FALLIBLE {
        rows.map(|index| apply(A::get(&columns, index)).into_result())
            .collect::<VortexResult<Vec<_>>>()?
    } else {
        rows.map(|index| {
            apply(A::get(&columns, index))
                .into_result()
                .vortex_expect("an infallible ApplyResult cannot be an error")
        })
        .collect()
    };

    Ok(R::Out::build(values))
}
