// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::ToPrimitive;
use num_traits::Zero;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::match_each_float_ptype;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::types::unit_vector::AnyUnitVector;
use crate::utils::extract_flat_elements;
use crate::utils::unit_norm_tolerance;

pub(super) fn validate_unit_vector_rows(
    array: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let metadata = array.dtype().as_extension().metadata::<AnyUnitVector>();
    let row_count = array.len();
    if row_count == 0 {
        return Ok(());
    }

    let array: ExtensionArray = array.clone().execute(ctx)?;
    let validity = array.as_ref().validity()?;
    let valid_rows = validity
        .nullability()
        .is_nullable()
        .then(|| validity.execute_mask(row_count, ctx))
        .transpose()?;
    let flat = extract_flat_elements(array.storage_array(), metadata.dimensions() as usize, ctx)?;
    let tolerance = unit_norm_tolerance(metadata.element_ptype(), metadata.dimensions() as usize);

    match_each_float_ptype!(metadata.element_ptype(), |T| {
        for row_idx in 0..row_count {
            if valid_rows
                .as_ref()
                .is_some_and(|valid_rows| !valid_rows.value(row_idx))
            {
                continue;
            }

            let (sum_squares, is_zero) = flat.row::<T>(row_idx).iter().fold(
                (0.0f64, true),
                |(sum_squares, is_zero), value| {
                    let value_f64 = ToPrimitive::to_f64(value)
                        .vortex_expect("UnitVector dtype validation established float elements");
                    (
                        sum_squares + value_f64 * value_f64,
                        is_zero && value.is_zero(),
                    )
                },
            );
            let norm = sum_squares.sqrt();

            vortex_ensure!(
                !is_zero && norm.is_finite() && (norm - 1.0).abs() <= tolerance,
                "UnitVector row must be finite, nonzero, and have L2 norm within {tolerance:.6} \
                 of 1.0, got row {row_idx} with norm {norm:.6}",
            );
        }
    });

    Ok(())
}
