// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A fixed-signature shorthand for [`RowFn`].
//!
//! [`TypedRowFn`] states its row signature as associated types instead of choosing one inside
//! [`RowFn::dispatch`]. It suits a function whose element types do not depend on the input dtypes.
//! A function whose signature varies with its arguments — every operator that accepts more than
//! one [`PType`](crate::dtype::PType) — implements [`RowFn`] directly, because only `dispatch` can
//! hand a different concrete signature to each visit.

use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;

use vortex_error::VortexResult;

use super::RowFn;
use super::RowVisitor;
use crate::dtype::DType;
use crate::scalar_fn::ElementTuple;
use crate::scalar_fn::IndexedElementTuple;
use crate::scalar_fn::OutputElement;
use crate::scalar_fn::ScalarFnId;

/// A scalar function computed one row at a time, at one fixed row signature.
///
/// Implementing this implements [`RowFn`], and therefore
/// [`ScalarFnVTable`](crate::scalar_fn::ScalarFnVTable), through a blanket implementation. A type
/// cannot implement both this trait and [`RowFn`].
///
/// ```
/// # use vortex_array::scalar_fn::EmptyOptions;
/// # use vortex_array::scalar_fn::ScalarFnId;
/// # use vortex_array::scalar_fn::TypedRowFn;
/// # use vortex_session::registry::CachedId;
/// #[derive(Clone)]
/// struct Hypot;
///
/// impl TypedRowFn for Hypot {
///     type Options = EmptyOptions;
///     type Args = (f64, f64);
///     type Out = f64;
///
///     const ARG_NAMES: &'static [&'static str] = &["x", "y"];
///
///     fn id(&self) -> ScalarFnId {
///         static ID: CachedId = CachedId::new("example.hypot");
///         *ID
///     }
///
///     fn apply(_options: &EmptyOptions, (x, y): (f64, f64)) -> f64 {
///         x.hypot(y)
///     }
/// }
/// ```
pub trait TypedRowFn: 'static + Sized + Clone + Send + Sync {
    /// Options for this function, if any. See [`RowFn::Options`].
    type Options: 'static + Send + Sync + Clone + Debug + Display + PartialEq + Eq + Hash;

    /// The row signature's input elements. Its arity must match [`ARG_NAMES`](Self::ARG_NAMES).
    type Args: IndexedElementTuple;

    /// The owned value this function produces for one row.
    type Out: OutputElement;

    /// The arguments in display order. Its length is the function's exact arity.
    const ARG_NAMES: &'static [&'static str];

    /// Whether decoding an argument can fail. See [`RowFn::FALLIBLE`].
    const FALLIBLE: bool = false;

    /// Returns the ID of the scalar function.
    fn id(&self) -> ScalarFnId;

    /// Compute one row.
    ///
    /// This must be total over every stored element value: dense execution can pass unspecified
    /// values read from null rows, so it must not panic or have side effects.
    fn apply(options: &Self::Options, args: <Self::Args as ElementTuple>::Elems<'_>) -> Self::Out;
}

impl<F: TypedRowFn> RowFn for F {
    type Options = F::Options;

    const ARG_NAMES: &'static [&'static str] = F::ARG_NAMES;

    const FALLIBLE: bool = F::FALLIBLE;

    fn id(&self) -> ScalarFnId {
        TypedRowFn::id(self)
    }

    fn dispatch<V: RowVisitor>(
        &self,
        options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit::<F::Args, F::Out>(|args| F::apply(options, args))
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;
    use vortex_session::registry::CachedId;

    use super::TypedRowFn;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::scalar_fn::EmptyOptions;
    use crate::scalar_fn::ScalarFnId;
    use crate::scalar_fn::ScalarFnVTable;
    use crate::scalar_fn::VecExecutionArgs;

    /// The fixed-signature form: state the row types, compute one row.
    #[derive(Clone)]
    struct Plus;

    impl TypedRowFn for Plus {
        type Options = EmptyOptions;
        type Args = (f64, f64);
        type Out = f64;

        const ARG_NAMES: &'static [&'static str] = &["lhs", "rhs"];

        fn id(&self) -> ScalarFnId {
            static ID: CachedId = CachedId::new("example.plus_f64");
            *ID
        }

        fn apply(_options: &EmptyOptions, (lhs, rhs): (f64, f64)) -> f64 {
            lhs + rhs
        }
    }

    #[test]
    fn typed_row_fn_executes_over_a_batch() -> VortexResult<()> {
        let lhs = PrimitiveArray::from_iter([1.0f64, 2.0, 3.0]);
        let rhs = PrimitiveArray::from_iter([0.5f64, 0.25, 0.125]);
        let args = VecExecutionArgs::new(vec![lhs.into_array(), rhs.into_array()], 3);

        let mut ctx = array_session().create_execution_ctx();
        let actual = ScalarFnVTable::execute(&Plus, &EmptyOptions, &args, &mut ctx)?;

        let expected = PrimitiveArray::from_iter([1.5f64, 2.25, 3.125]);
        assert_arrays_eq!(&actual, &expected, &mut ctx);

        Ok(())
    }
}
