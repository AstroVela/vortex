// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Typed infallible row kernels with an optional dense collector.
//!
//! [`RowKernel::eval`] defines the operation's scalar semantics. The executor uses that method for
//! validity-aware traversal and converts the values through the associated [`RowKernelOutput`].
//! Dense execution can override [`RowKernel::collect_dense`] with a representation-specific bulk
//! kernel.

use vortex_compute::lane_kernels::IndexedSourceExt;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::scalar_fn::unstable::row::IndexedElementTuple;
use crate::scalar_fn::unstable::row::OutputElement;
use crate::scalar_fn::unstable::row::RowKernelOutput;

/// A validated dense batch of typed row inputs.
///
/// The executor decodes the columns and retains the exact views whose lengths it validates before
/// constructing this value. Specialized collectors can inspect the typed decoded inputs through
/// [`inputs`](Self::inputs). This type does not expose the untyped execution arguments or output
/// arrays.
pub struct DenseRows<'a, Args: IndexedElementTuple> {
    inputs: &'a Args::Columns,
    views: Option<Args::Views<'a>>,
    row_count: usize,
}

impl<'a, Args: IndexedElementTuple> DenseRows<'a, Args> {
    pub(crate) fn new(inputs: &'a Args::Columns, row_count: usize) -> VortexResult<Self> {
        let views = Args::views_if_no_consts(inputs);

        if let Some(views) = &views {
            vortex_ensure!(
                Args::view_lens_match(views, row_count),
                "a decoded row input does not address exactly {row_count} rows",
            );
        } else {
            vortex_ensure!(
                Args::decoded_lens_match(inputs, row_count),
                "a decoded row input does not address exactly {row_count} rows",
            );
        }

        Ok(Self {
            inputs,
            views,
            row_count,
        })
    }

    /// Return the typed decoded inputs.
    ///
    /// A specialized collector that borrows a new view from these inputs must validate that view
    /// before using it for unchecked traversal. [`collect`](Self::collect) instead uses the exact
    /// views retained during construction.
    pub fn inputs(&self) -> &'a Args::Columns {
        self.inputs
    }

    /// Return the number of input rows.
    pub fn len(&self) -> usize {
        self.row_count
    }

    /// Return whether the batch contains no rows.
    pub fn is_empty(&self) -> bool {
        self.row_count == 0
    }

    /// Collect one initialized value per row with the framework's dense traversal.
    pub fn collect<Out>(self, apply: impl Fn(Args::Elems<'_>) -> Out) -> Vec<Out> {
        if let Some(views) = self.views {
            let mut values = Vec::<Out>::with_capacity(self.row_count);
            let output = &mut values.spare_capacity_mut()[..self.row_count];

            // SAFETY: `new` checked that these exact retained views address `row_count` rows. The
            // `InputElement` contract keeps their lengths stable while they exist.
            let source = unsafe { Args::indexed_source(views, self.row_count) };
            source.map_into(output, apply);

            // SAFETY: normal completion of `map_into` initializes every output slot exactly once.
            unsafe { values.set_len(self.row_count) };

            return values;
        }

        let mut values = Vec::with_capacity(self.row_count);
        for index in 0..self.row_count {
            values.push(apply(Args::get(self.inputs, index)));
        }

        values
    }
}

/// One semantic row operation with a selectable owned output representation.
pub trait RowKernel<Args>: Sized
where
    Args: IndexedElementTuple,
{
    /// The logical value produced for one row.
    type Element: OutputElement;

    /// The complete initialized output batch used by every execution policy.
    type Output: RowKernelOutput<Element = Self::Element>;

    /// Evaluate one row using the kernel's portable reference semantics.
    fn eval(&self, args: Args::Elems<'_>) -> Self::Element;

    /// Collect a validated dense batch.
    ///
    /// The default retains the framework's vectorizable scalar traversal, then converts the values
    /// into the associated output. Override this method to write a packed or otherwise specialized
    /// representation directly. The output must preserve row order and contain the same observable
    /// values as calling [`eval`](Self::eval) once for each row. Dense execution can pass
    /// unspecified payloads from null input rows; their output values can also be arbitrary because
    /// batch execution masks them before returning the array.
    fn collect_dense(&self, rows: DenseRows<'_, Args>) -> VortexResult<Self::Output> {
        Self::Output::from_values(rows.collect(|args| self.eval(args)))
    }
}
