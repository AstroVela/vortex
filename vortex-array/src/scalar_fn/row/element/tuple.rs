// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Argument lists built from [`InputElement`]s, and the per-argument decode behind them.

use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::arrays::Masked;
use crate::arrays::masked::MaskedArraySlotsExt;
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
        if let Some(constant) = batch_constant(&array)
            && !array.is_empty()
        {
            return Ok(Self {
                column: T::decode(constant.slice(0..1)?, ctx)?,
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

    /// The single decoded element of a constant operand, or `None` for a real column.
    ///
    /// `Some` exactly when [`decode`](Self::decode) collapsed the operand to its one distinct row,
    /// in which case the value returned is the element every row of the batch reads.
    fn constant(&self) -> Option<T::Elem<'_>> {
        (self.stride == 0).then(|| T::get(&self.column, 0))
    }
}

/// The array whose every row holds one distinct value, when `array` is constant for the batch.
///
/// Beyond the constant encoding itself this sees one level through [`Masked`], which is how the
/// compressor spells "the same value in every row, some rows null": the masked child carries the
/// value, the wrapper carries only validity. Reading the child's value for a null row is sound
/// here because the strict lifting owns validity entirely; the row loop's output behind a null
/// row is masked away (dense) or never computed (filter), so which value the loop read there
/// cannot be observed. An all-null constant never reaches decode at all, since the lifting
/// short-circuits it to an all-null result first.
fn batch_constant(array: &ArrayRef) -> Option<ArrayRef> {
    if array.as_constant().is_some() {
        return Some(array.clone());
    }

    array
        .as_opt::<Masked>()
        .map(|masked| masked.child().clone())
        .filter(|child| child.as_constant().is_some())
}

/// Tuples of [`InputElement`]s forming the typed argument list a [`RowFn`](crate::scalar_fn::RowFn)
/// visits with.
pub trait ElementTuple: 'static {
    /// The decoded column representations.
    type Columns;

    /// The borrowed row of element values.
    type Elems<'a>;

    /// The batch-constant element values: [`Elems`](Self::Elems) with every argument wrapped in
    /// `Option`.
    ///
    /// `Some` marks an argument whose operand is constant for the batch and carries the element
    /// every row reads; `None` marks one that varies by row. This is what
    /// [`visit_prepared`](crate::scalar_fn::RowVisitor::visit_prepared) hands to its prepare
    /// closure, so a kernel can hoist work that depends only on a constant argument out of the
    /// row loop.
    type ConstElems<'a>;

    /// The number of arguments.
    const ARITY: usize;

    /// Whether every argument is [`InputElement::DENSE_SAFE`].
    const DENSE_SAFE: bool;

    /// Whether *any* argument is [`InputElement::DECODE_FALLIBLE`].
    const DECODE_FALLIBLE: bool;

    /// Validate the input dtypes, including that `dtypes` has exactly `ARITY` entries.
    ///
    /// The expression layer checks the count against [`Arity`](crate::scalar_fn::Arity) before it
    /// builds a call, but this is also the entry point of the public
    /// [`return_element_dtype`](crate::scalar_fn::StrictScalarFnVTable::return_element_dtype), so
    /// the count is enforced here rather than assumed.
    fn validate(dtypes: &[DType]) -> VortexResult<()>;

    /// Decode every input column once. Called once per batch.
    fn decode(args: &dyn ExecutionArgs, ctx: &mut ExecutionCtx) -> VortexResult<Self::Columns>;

    /// Read the row of elements at `index`. Must be `O(1)`: it is called in the row loop.
    fn get(columns: &Self::Columns, index: usize) -> Self::Elems<'_>;

    /// Read the batch-constant elements out of the decoded columns. Called once per batch.
    fn constants(columns: &Self::Columns) -> Self::ConstElems<'_>;
}

macro_rules! element_tuple {
    ($arity:literal; $($t:ident : $idx:tt),+) => {
        impl<$($t: InputElement),+> ElementTuple for ($($t,)+) {
            type Columns = ($(ArgColumn<$t>,)+);
            type Elems<'a> = ($($t::Elem<'a>,)+);
            type ConstElems<'a> = ($(Option<$t::Elem<'a>>,)+);

            const ARITY: usize = $arity;
            const DENSE_SAFE: bool = $($t::DENSE_SAFE &&)+ true;
            const DECODE_FALLIBLE: bool = $($t::DECODE_FALLIBLE ||)+ false;

            fn validate(dtypes: &[DType]) -> VortexResult<()> {
                vortex_ensure_eq!(
                    dtypes.len(),
                    $arity,
                    "expected {} argument dtypes, got {}",
                    $arity,
                    dtypes.len(),
                );

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

            fn constants(columns: &Self::Columns) -> Self::ConstElems<'_> {
                ($(columns.$idx.constant(),)+)
            }
        }
    };
}

element_tuple!(1; A:0);
element_tuple!(2; A:0, B:1);
element_tuple!(3; A:0, B:1, C:2);
