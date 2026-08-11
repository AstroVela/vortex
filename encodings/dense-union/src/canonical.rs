// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::UnionArray;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_mask::AllOr;

use crate::array::DenseUnion;
use crate::array::DenseUnionArrayExt;
use crate::array::DenseUnionArraySlotsExt;

pub(crate) fn canonicalize(
    array: Array<DenseUnion>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let len = array.len();
    let variants = array.variants().clone();
    let type_ids = array.type_ids().as_::<Primitive>();
    let offsets = array.offsets().as_::<Primitive>();
    let type_id_values = type_ids.as_slice::<u8>();
    let offset_values = offsets.as_slice::<i32>();
    let valid_rows = type_ids.validity()?.execute_mask(len, ctx)?;
    let mut codes_by_child = vec![vec![0u32; len]; variants.len()];

    let mut assign_row = |row: usize| -> VortexResult<()> {
        let type_id = type_id_values[row];
        let child_index = variants
            .tag_to_child_index(type_id)
            .ok_or_else(|| vortex_err!("DenseUnion contains unknown type ID {type_id}"))?;
        let offset = usize::try_from(offset_values[row]).map_err(|_| {
            vortex_err!(
                "DenseUnion contains negative offset {} at row {row}",
                offset_values[row]
            )
        })?;
        let child_len = array
            .child(child_index)
            .ok_or_else(|| vortex_err!("DenseUnion is missing compact child {child_index}"))?
            .len();
        vortex_ensure!(
            offset < child_len,
            "DenseUnion offset {offset} is out of bounds for child {child_index} of length {child_len}"
        );
        codes_by_child[child_index][row] = u32::try_from(offset)
            .map_err(|_| vortex_err!("DenseUnion offset {offset} does not fit in u32"))?;
        Ok(())
    };

    match valid_rows.indices() {
        AllOr::All => {
            for row in 0..len {
                assign_row(row)?;
            }
        }
        AllOr::None => {}
        AllOr::Some(rows) => {
            for &row in rows {
                assign_row(row)?;
            }
        }
    }

    let sparse_children = array
        .iter_children()
        .zip(codes_by_child)
        .map(|(child, codes)| {
            let values = if child.is_empty() {
                ConstantArray::new(Scalar::default_value(child.dtype()), 1).into_array()
            } else {
                child.clone()
            };
            DictArray::try_new(PrimitiveArray::from_iter(codes).into_array(), values)
                .map(IntoArray::into_array)
        })
        .collect::<VortexResult<Vec<_>>>()?;

    UnionArray::try_new(type_ids.array().clone(), variants, sparse_children)
        .map(IntoArray::into_array)
}
