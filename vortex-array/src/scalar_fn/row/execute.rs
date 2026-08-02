// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The pieces every row function is built from, whatever its `dispatch` chooses.
//!
//! These back the blanket impls in [`row_fn`](super::row_fn) and are deliberately not public:
//! [`RowFn`](crate::scalar_fn::RowFn) is the abstraction, these are its internals.

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::scalar_fn::ApplyResult;
use crate::scalar_fn::ElementTuple;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::NullHandling;
use crate::scalar_fn::OutputElement;
use crate::scalar_fn::OutputSink;
use crate::scalar_fn::RowResult;
use crate::scalar_fn::SinkResult;

/// Validate the input dtypes of a row function and return its output element dtype.
///
/// The lifting widens the result to nullable iff any input is nullable, so this ignores
/// nullability.
pub(super) fn validate_row_args<A: ElementTuple, R: ApplyResult>(
    args: &[DType],
) -> VortexResult<DType> {
    A::validate(args)?;
    let dtype = R::Out::element_dtype();
    vortex_ensure!(
        !dtype.is_nullable(),
        "row output elements must declare a non-nullable dtype, got {dtype}",
    );
    Ok(dtype)
}

/// Validate the input dtypes of a sink-writing row function and return the dtype its sink builds.
///
/// Unlike [`validate_row_args`] the output dtype may be a function of the inputs. A sink can also
/// own a batch-wide builder, such as the shared byte and view buffers of a future string transform.
pub(super) fn validate_row_sink<A: ElementTuple, S: OutputSink>(
    args: &[DType],
) -> VortexResult<DType> {
    A::validate(args)?;
    let dtype = S::sink_dtype(args)?;
    vortex_ensure!(
        !dtype.is_nullable(),
        "row output sinks must declare a non-nullable dtype, got {dtype}",
    );
    Ok(dtype)
}

/// The [`NullHandling`] a row function gets, from what its arguments and return type allow.
///
/// [`NullHandling::Dense`] is cheaper and the only option that leaves inputs at their original
/// encoding, so it is used whenever it is sound: every argument readable behind a null row, and a
/// row computation that cannot fail. Both conditions are read off the types, so a row function never
/// declares its null handling.
pub(super) const fn row_null_handling<A: ElementTuple, R: RowResult>() -> NullHandling {
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
pub(super) const fn row_is_fallible<A: ElementTuple, R: RowResult>() -> bool {
    A::DECODE_FALLIBLE || R::FALLIBLE
}

/// Decode every input column once, run `prepare` over the batch-constant elements, then compute
/// one output element per row with the prepared state behind a shared reference.
///
/// The state lives here rather than in the closure, so `apply` stays [`Fn`] and the loop keeps the
/// unconditional shape that lets it vectorize, exactly as in [`execute_row_loop`]. `P` is chosen by
/// the dispatch and names no column lifetime, so the state owns its data and cannot alias the
/// columns the loop reads.
pub(super) fn execute_row_loop_prepared<A: ElementTuple, P, R: ApplyResult>(
    args: &dyn ExecutionArgs,
    ctx: &mut ExecutionCtx,
    prepare: impl FnOnce(A::ConstElems<'_>) -> P,
    apply: impl Fn(&P, A::Elems<'_>) -> R,
) -> VortexResult<ArrayRef> {
    let columns = A::decode(args, ctx)?;
    let state = prepare(A::constants(&columns));
    let rows = 0..args.row_count();

    // `R::FALLIBLE` is a constant, so only one of these survives monomorphization, exactly as in
    // [`execute_row_loop`].
    let values = if R::FALLIBLE {
        rows.map(|index| apply(&state, A::get(&columns, index)).into_result())
            .collect::<VortexResult<Vec<_>>>()?
    } else {
        rows.map(|index| {
            apply(&state, A::get(&columns, index))
                .into_result()
                .vortex_expect("an infallible ApplyResult cannot be an error")
        })
        .collect()
    };

    Ok(R::Out::build(values))
}

/// Decode every column null-tolerantly, then run the row loop only over the rows set in `valid`,
/// leaving a placeholder in every other output slot.
///
/// This is the branch-and-skip null strategy's row loop. The caller masks the built column with
/// `valid` afterwards, so the placeholders are never observed; what matters is that `apply` (and
/// any per-row fallible decode) never runs for an unset row, since those rows hold arbitrary
/// values and a fallible kernel would spuriously fail on them. Iteration is a word at a time via
/// [`BitBuffer::for_each_set_index`] rather than a per-row `valid.value(i)` branch.
///
/// Returns `Ok(None)` when some argument cannot decode null-tolerantly, in which case the lifting
/// falls back to the filter strategy.
///
/// [`BitBuffer::for_each_set_index`]: vortex_buffer::BitBuffer::for_each_set_index
pub(super) fn execute_row_loop_branch<A: ElementTuple, P, R: ApplyResult>(
    args: &dyn ExecutionArgs,
    valid: &Mask,
    ctx: &mut ExecutionCtx,
    prepare: impl FnOnce(A::ConstElems<'_>) -> P,
    apply: impl Fn(&P, A::Elems<'_>) -> R,
) -> VortexResult<Option<ArrayRef>> {
    let Some(columns) = A::decode_null_tolerant(args, ctx)? else {
        return Ok(None);
    };
    let state = prepare(A::constants(&columns));

    let AllOr::Some(valid) = valid.bit_buffer() else {
        // The lifting takes the all-true and all-false shortcuts before choosing a strategy, so a
        // degenerate mask here is a bug in the lifting.
        vortex_bail!("execute_row_loop_branch requires a mixed mask");
    };

    let mut values: Vec<R::Out> = Vec::with_capacity(args.row_count());
    values.resize_with(args.row_count(), R::Out::placeholder);

    // `R::FALLIBLE` is a constant, so only one of these survives monomorphization, exactly as in
    // [`execute_row_loop_prepared`].
    if R::FALLIBLE {
        // `for_each_set_index` cannot early-return, so the first error is remembered and the
        // remaining set rows are skipped cheaply; the success path pays only the `is_none` check.
        let mut error = None;
        valid.for_each_set_index(|index| {
            if error.is_none() {
                match apply(&state, A::get(&columns, index)).into_result() {
                    Ok(value) => values[index] = value,
                    Err(e) => error = Some(e),
                }
            }
        });

        if let Some(error) = error {
            return Err(error);
        }
    } else {
        valid.for_each_set_index(|index| {
            values[index] = apply(&state, A::get(&columns, index))
                .into_result()
                .vortex_expect("an infallible ApplyResult cannot be an error");
        });
    }

    Ok(Some(R::Out::build(values)))
}

/// Decode every input column once, allocate the sink once, then write one row at a time.
///
/// The sink lives here rather than in the closure, so `apply` stays [`Fn`] and the loop keeps the
/// unconditional shape that lets it vectorize. Monomorphic in `A`, `S` and `R`, so `apply` and
/// [`OutputSink::row`] both inline.
pub(super) fn execute_row_sink<A: ElementTuple, S: OutputSink, R: SinkResult>(
    args: &dyn ExecutionArgs,
    arg_dtypes: &[DType],
    ctx: &mut ExecutionCtx,
    apply: impl Fn(A::Elems<'_>, S::Row<'_>) -> R,
) -> VortexResult<ArrayRef> {
    let row_count = args.row_count();
    let mut sink = S::with_capacity(row_count, &S::sink_dtype(arg_dtypes)?)?;
    let columns = A::decode(args, ctx)?;

    // `R::FALLIBLE` is a constant, so only one of these survives monomorphization. There
    // `into_result` inlines to `Ok(())` and the unwrap folds away.
    if R::FALLIBLE {
        for index in 0..row_count {
            apply(A::get(&columns, index), sink.row(index)).into_result()?;
        }
    } else {
        for index in 0..row_count {
            apply(A::get(&columns, index), sink.row(index))
                .into_result()
                .vortex_expect("an infallible SinkResult cannot be an error");
        }
    }

    sink.finish()
}
