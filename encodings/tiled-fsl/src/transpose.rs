// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::match_each_native_ptype;
use vortex_array::validity::Validity;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::TileGeometry;
use crate::geometry::physical_offset;
use crate::geometry::tile_bounds;

pub(crate) enum TransposeDirection {
    CanonicalToTiled,
    TiledToCanonical,
}

#[expect(
    clippy::cognitive_complexity,
    reason = "complexity is attributed to native-type dispatch macro expansion"
)]
pub(crate) fn encode_elements(
    elements: ArrayView<'_, Primitive>,
    len: usize,
    list_size: usize,
    geometry: TileGeometry,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    match_each_native_ptype!(elements.ptype(), |T| {
        let source = elements.as_slice::<T>();
        let mut output = BufferMut::<T>::with_capacity(source.len());
        for dimension_tile in 0..list_size.div_ceil(usize::try_from(geometry.dimensions().get())?) {
            for row_tile in 0..len.div_ceil(usize::try_from(geometry.rows().get())?) {
                let bounds = tile_bounds(len, list_size, geometry, row_tile, dimension_tile)?;
                for dimension in bounds.dimension_range.clone() {
                    for row in bounds.row_range.clone() {
                        output.push(source[row * list_size + dimension]);
                    }
                }
            }
        }
        let validity = transpose_validity(
            elements.validity()?,
            len,
            list_size,
            geometry,
            TransposeDirection::CanonicalToTiled,
            ctx,
        )?;
        Ok(PrimitiveArray::new(output.freeze(), validity))
    })
}

#[expect(
    clippy::cognitive_complexity,
    reason = "complexity is attributed to native-type dispatch macro expansion"
)]
pub(crate) fn decode_elements(
    elements: ArrayView<'_, Primitive>,
    len: usize,
    list_size: usize,
    geometry: TileGeometry,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let expected_len = len.checked_mul(list_size);
    vortex_ensure!(
        expected_len == Some(elements.len()),
        InvalidArgument:
        "physical child length {} does not match logical extent ({len}, {list_size})",
        elements.len()
    );

    match_each_native_ptype!(elements.ptype(), |T| {
        let source = elements.as_slice::<T>();
        let mut output = BufferMut::<T>::zeroed(elements.len());
        for dimension_tile in 0..list_size.div_ceil(usize::try_from(geometry.dimensions().get())?) {
            for row_tile in 0..len.div_ceil(usize::try_from(geometry.rows().get())?) {
                let bounds = tile_bounds(len, list_size, geometry, row_tile, dimension_tile)?;
                let mut physical = bounds.physical_range.start;
                for dimension in bounds.dimension_range {
                    for row in bounds.row_range.clone() {
                        output[row * list_size + dimension] = source[physical];
                        physical += 1;
                    }
                }
            }
        }
        let validity = transpose_validity(
            elements.validity()?,
            len,
            list_size,
            geometry,
            TransposeDirection::TiledToCanonical,
            ctx,
        )?;
        Ok(PrimitiveArray::new(output.freeze(), validity))
    })
}

pub(crate) fn transpose_validity(
    validity: Validity,
    len: usize,
    list_size: usize,
    geometry: TileGeometry,
    direction: TransposeDirection,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Validity> {
    let Validity::Array(_) = validity else {
        return Ok(validity);
    };

    let element_count = len.checked_mul(list_size).ok_or_else(
        || vortex_err!(InvalidArgument: "logical extent ({len}, {list_size}) overflows usize"),
    )?;
    let mask = validity.execute_mask(element_count, ctx)?;
    let Mask::Values(values) = mask else {
        return Ok(Validity::from_mask(
            mask,
            vortex_array::dtype::Nullability::Nullable,
        ));
    };

    let mut output = BitBufferMut::new_unset(element_count);
    let mut mapping_result = Ok(());
    values.bit_buffer().for_each_set_index(|source| {
        if mapping_result.is_err() {
            return;
        }
        let destination = match direction {
            TransposeDirection::CanonicalToTiled => {
                let row = source / list_size;
                let dimension = source % list_size;
                physical_offset(len, list_size, geometry, row, dimension)
            }
            TransposeDirection::TiledToCanonical => {
                canonical_offset(len, list_size, geometry, source)
            }
        };
        match destination {
            Ok(destination) => output.set(destination),
            Err(error) => mapping_result = Err(error),
        }
    });
    mapping_result?;
    Ok(Validity::from(output.freeze()))
}

fn canonical_offset(
    len: usize,
    list_size: usize,
    geometry: TileGeometry,
    physical: usize,
) -> VortexResult<usize> {
    let element_count = len.checked_mul(list_size);
    vortex_ensure!(
        element_count.is_some_and(|count| physical < count),
        InvalidArgument:
        "physical position {physical} is outside logical extent ({len}, {list_size})"
    );

    let tile_rows = usize::try_from(geometry.rows().get())?;
    let tile_dimensions = usize::try_from(geometry.dimensions().get())?;
    let full_dimension_width = tile_dimensions.min(list_size);
    let dimension_tile_span = full_dimension_width.checked_mul(len).ok_or_else(
        || vortex_err!(InvalidArgument: "dimension tile span overflows logical extent"),
    )?;
    let dimension_tile = physical / dimension_tile_span;
    let dimension_start = dimension_tile.checked_mul(tile_dimensions).ok_or_else(
        || vortex_err!(InvalidArgument: "dimension tile start overflows logical extent"),
    )?;
    let dimension_width = tile_dimensions.min(list_size - dimension_start);
    let within_dimension_tile = physical - dimension_start * len;

    let full_row_height = tile_rows.min(len);
    let row_tile_span = full_row_height
        .checked_mul(dimension_width)
        .ok_or_else(|| vortex_err!(InvalidArgument: "row tile span overflows logical extent"))?;
    let row_tile = within_dimension_tile / row_tile_span;
    let row_start = row_tile
        .checked_mul(tile_rows)
        .ok_or_else(|| vortex_err!(InvalidArgument: "row tile start overflows logical extent"))?;
    let row_height = tile_rows.min(len - row_start);
    let within_tile = within_dimension_tile - row_start * dimension_width;
    let dimension = dimension_start + within_tile / row_height;
    let row = row_start + within_tile % row_height;

    Ok(row * list_size + dimension)
}
