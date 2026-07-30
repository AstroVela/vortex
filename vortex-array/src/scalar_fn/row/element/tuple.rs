// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Argument lists built from [`InputElement`]s, and the per-argument decode behind them.

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::InputElement;

/// One decoded input column of an [`ElementTuple`], together with the stride used to read it.
///
/// A constant operand holds the same value in every row, so it is decoded once as a single row and
/// read at index 0 forever. That is what stops a constant argument costing one decode per row, which
/// matters whenever the decode is more than a buffer read: parsing a geometry from WKB, or
/// canonicalizing an extension row.
pub struct ArgColumn<T: InputElement> {
    /// The decoded column, holding either every row or the single row of a constant operand.
    column: T::Column,

    /// `1` for a real column and `0` for a constant operand, so `index * stride` pins a constant to
    /// its only row. A multiplier rather than a branch, to leave the row loop's shape unconditional.
    stride: usize,
}

impl<T: InputElement> ArgColumn<T> {
    /// Decode one input column, collapsing a constant operand to its single distinct row.
    fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        // An empty input has no row 0 to slice, and its row loop runs zero times either way.
        if array.as_constant().is_some() && !array.is_empty() {
            return Ok(Self {
                column: T::decode(array.slice(0..1)?, ctx)?,
                stride: 0,
            });
        }

        Ok(Self {
            column: T::decode(array, ctx)?,
            stride: 1,
        })
    }

    /// Read the element at `index`, which for a constant operand is always its single row.
    fn get(&self, index: usize) -> T::Elem<'_> {
        T::get(&self.column, index * self.stride)
    }
}

/// Tuples of [`InputElement`]s forming the typed argument list a [`RowFn`](crate::scalar_fn::RowFn)
/// visits with.
pub trait ElementTuple: 'static {
    /// The decoded column representations.
    type Columns;

    /// The borrowed row of element values.
    type Elems<'a>;

    /// The number of arguments.
    const ARITY: usize;

    /// Whether every argument is [`InputElement::DENSE_SAFE`].
    const DENSE_SAFE: bool;

    /// Whether *any* argument is [`InputElement::DECODE_FALLIBLE`].
    const DECODE_FALLIBLE: bool;

    /// Validate the input dtypes. `dtypes` has exactly `ARITY` entries (checked by the
    /// expression layer against [`Arity`](crate::scalar_fn::Arity)).
    fn validate(dtypes: &[DType]) -> VortexResult<()>;

    /// Decode every input column once. Called once per batch.
    fn decode(args: &dyn ExecutionArgs, ctx: &mut ExecutionCtx) -> VortexResult<Self::Columns>;

    /// Read the row of elements at `index`. Must be `O(1)`: it is called in the row loop.
    fn get(columns: &Self::Columns, index: usize) -> Self::Elems<'_>;
}

macro_rules! element_tuple {
    ($arity:literal; $($t:ident : $idx:tt),+) => {
        impl<$($t: InputElement),+> ElementTuple for ($($t,)+) {
            type Columns = ($(ArgColumn<$t>,)+);
            type Elems<'a> = ($($t::Elem<'a>,)+);

            const ARITY: usize = $arity;
            const DENSE_SAFE: bool = $($t::DENSE_SAFE &&)+ true;
            const DECODE_FALLIBLE: bool = $($t::DECODE_FALLIBLE ||)+ false;

            fn validate(dtypes: &[DType]) -> VortexResult<()> {
                $($t::validate(&dtypes[$idx])?;)+
                Ok(())
            }

            fn decode(
                args: &dyn ExecutionArgs,
                ctx: &mut ExecutionCtx,
            ) -> VortexResult<Self::Columns> {
                Ok(($(ArgColumn::<$t>::decode(args.get($idx)?, ctx)?,)+))
            }

            fn get(columns: &Self::Columns, index: usize) -> Self::Elems<'_> {
                ($(columns.$idx.get(index),)+)
            }
        }
    };
}

element_tuple!(1; A:0);
element_tuple!(2; A:0, B:1);
element_tuple!(3; A:0, B:1, C:2);
