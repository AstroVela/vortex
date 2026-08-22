// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execution for typed infallible row kernels.

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_mask::MaskValuesRef;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::unstable::row::DenseRows;
use crate::scalar_fn::unstable::row::IndexedElementTuple;
use crate::scalar_fn::unstable::row::RowKernel;
use crate::scalar_fn::unstable::row::RowKernelOutput;
use crate::scalar_fn::unstable::row::visitor::assert_owned_output_needs_no_drop;

/// Decode and validate the inputs, then delegate only dense collection to the kernel.
pub(in crate::scalar_fn::unstable::row) fn execute_kernel<Args, Kernel>(
    args: &dyn ExecutionArgs,
    ctx: &mut ExecutionCtx,
    kernel: Kernel,
) -> VortexResult<ArrayRef>
where
    Args: IndexedElementTuple,
    Kernel: RowKernel<Args>,
{
    let inputs = Args::decode(args, ctx)?;
    let rows = DenseRows::<Args>::new(&inputs, args.row_count())?;
    let output = kernel.collect_dense(rows)?;

    output.finish()
}

/// Decode nullable inputs, then store one kernel output for each valid row.
pub(in crate::scalar_fn::unstable::row) fn execute_kernel_valid_rows<Args, Kernel>(
    args: &dyn ExecutionArgs,
    valid: &MaskValuesRef,
    ctx: &mut ExecutionCtx,
    kernel: Kernel,
) -> VortexResult<Option<ArrayRef>>
where
    Args: IndexedElementTuple,
    Kernel: RowKernel<Args>,
{
    const { assert_owned_output_needs_no_drop::<Kernel::Element>() };

    let Some(columns) = Args::decode_null_tolerant(args, ctx)? else {
        return Ok(None);
    };

    let row_count = args.row_count();
    let valid_rows = valid.bit_buffer();
    vortex_ensure_eq!(
        valid_rows.len(),
        row_count,
        "the validity mask must address exactly {row_count} rows, got {}",
        valid_rows.len(),
    );

    let mut values: Vec<Kernel::Element> = std::iter::repeat_with(Kernel::Element::default)
        .take(row_count)
        .collect();

    if let Some(views) = Args::views_if_no_consts(&columns) {
        vortex_ensure!(
            Args::view_lens_match(&views, row_count),
            "a decoded row input does not address exactly {row_count} rows",
        );

        valid_rows.for_each_set_index(|index| {
            // SAFETY: the tuple-wide length check proved every view has `row_count` rows, and mask
            // indices are below `row_count`. Nullary tuples do not access an input view.
            let elements = unsafe { Args::get_from_views_unchecked(&views, index) };

            // SAFETY: the mask length check proved that every set index is below `row_count`.
            unsafe { *values.get_unchecked_mut(index) = kernel.eval(elements) };
        });
    } else {
        vortex_ensure!(
            Args::decoded_lens_match(&columns, row_count),
            "a decoded row input does not address exactly {row_count} rows",
        );

        valid_rows.for_each_set_index(|index| {
            let value = kernel.eval(Args::get(&columns, index));

            // SAFETY: the mask length check proved that every set index is below `row_count`.
            unsafe { *values.get_unchecked_mut(index) = value };
        });
    }

    let output = Kernel::Output::from_values(values)?;
    output.finish().map(Some)
}
