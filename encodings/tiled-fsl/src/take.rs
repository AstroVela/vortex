// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::dtype::DType;
use vortex_array::dtype::IntegerPType;
use vortex_array::match_each_integer_ptype;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::TiledFixedSizeList;
use crate::TiledFixedSizeListArrayExt;
use crate::gather::gather_tiled_rows;

pub(crate) fn collect_checked_rows<I: IntegerPType>(
    indices: &[I],
    mask: &Mask,
    source_len: usize,
) -> VortexResult<Vec<Option<usize>>> {
    vortex_ensure!(
        indices.len() == mask.len(),
        InvalidArgument:
        "index count {} does not match validity mask length {}",
        indices.len(),
        mask.len()
    );
    let mut rows = Vec::with_capacity(indices.len());
    for (&index, is_valid) in indices.iter().zip(mask.iter()) {
        if !is_valid {
            rows.push(None);
            continue;
        }

        let row = index.to_usize().ok_or_else(
            || vortex_err!(InvalidArgument: "index {index} cannot be represented as usize"),
        )?;
        if row >= source_len {
            vortex_bail!(OutOfBounds: row, 0, source_len);
        }
        rows.push(Some(row));
    }
    Ok(rows)
}

impl TakeExecute for TiledFixedSizeList {
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let DType::Primitive(ptype, _) = indices.dtype() else {
            vortex_bail!("Invalid indices dtype: {}", indices.dtype());
        };
        if !ptype.is_int() {
            vortex_bail!("Invalid indices dtype: {}", indices.dtype());
        }

        let indices_ref = indices.clone();
        let indices_array = indices.clone().execute::<PrimitiveArray>(ctx)?;
        let mask = indices_array
            .validity()?
            .execute_mask(indices_array.len(), ctx)?;
        let rows = match_each_integer_ptype!(indices_array.ptype(), |I| {
            collect_checked_rows::<I>(indices_array.as_slice::<I>(), &mask, array.len())
        })?;
        let validity = array.array_validity().take(&indices_ref)?;
        Ok(Some(
            gather_tiled_rows(array, &rows, validity)?.into_array(),
        ))
    }
}
