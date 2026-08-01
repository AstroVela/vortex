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
use vortex_mask::Mask;

use crate::TileGeometry;
use crate::geometry::tile_bounds;

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
    let expected_len = len.checked_mul(list_size);
    vortex_ensure!(
        expected_len == Some(elements.len()),
        InvalidArgument:
        "physical child length {} does not match logical extent ({len}, {list_size})",
        elements.len()
    );

    match_each_native_ptype!(elements.ptype(), |T| {
        let source = elements.as_slice::<T>();
        let mut output = BufferMut::<T>::with_capacity(source.len());
        let (source_validity, preserved_validity) = match elements.validity()? {
            validity @ Validity::Array(_) => match validity.execute_mask(elements.len(), ctx)? {
                Mask::Values(values) => (Some(values), None),
                mask => (
                    None,
                    Some(Validity::from_mask(
                        mask,
                        vortex_array::dtype::Nullability::Nullable,
                    )),
                ),
            },
            validity => (None, Some(validity)),
        };
        let mut output_validity = source_validity
            .as_ref()
            .map(|_| BitBufferMut::new_unset(elements.len()));
        for dimension_tile in 0..list_size.div_ceil(usize::try_from(geometry.dimensions().get())?) {
            for row_tile in 0..len.div_ceil(usize::try_from(geometry.rows().get())?) {
                let bounds = tile_bounds(len, list_size, geometry, row_tile, dimension_tile)?;
                for dimension in bounds.dimension_range.clone() {
                    for row in bounds.row_range.clone() {
                        let canonical = row * list_size + dimension;
                        let physical = output.len();
                        output.push(source[canonical]);
                        if let (Some(source_validity), Some(output_validity)) =
                            (&source_validity, &mut output_validity)
                        {
                            output_validity.set_to(physical, source_validity.value(canonical));
                        }
                    }
                }
            }
        }
        let validity = output_validity
            .map(|validity| Validity::from(validity.freeze()))
            .or(preserved_validity)
            .expect("validity must be preserved or transposed");
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
        let (source_validity, preserved_validity) = match elements.validity()? {
            validity @ Validity::Array(_) => match validity.execute_mask(elements.len(), ctx)? {
                Mask::Values(values) => (Some(values), None),
                mask => (
                    None,
                    Some(Validity::from_mask(
                        mask,
                        vortex_array::dtype::Nullability::Nullable,
                    )),
                ),
            },
            validity => (None, Some(validity)),
        };
        let mut output_validity = source_validity
            .as_ref()
            .map(|_| BitBufferMut::new_unset(elements.len()));
        for dimension_tile in 0..list_size.div_ceil(usize::try_from(geometry.dimensions().get())?) {
            for row_tile in 0..len.div_ceil(usize::try_from(geometry.rows().get())?) {
                let bounds = tile_bounds(len, list_size, geometry, row_tile, dimension_tile)?;
                let mut physical = bounds.physical_range.start;
                for dimension in bounds.dimension_range {
                    for row in bounds.row_range.clone() {
                        let canonical = row * list_size + dimension;
                        output[canonical] = source[physical];
                        if let (Some(source_validity), Some(output_validity)) =
                            (&source_validity, &mut output_validity)
                        {
                            output_validity.set_to(canonical, source_validity.value(physical));
                        }
                        physical += 1;
                    }
                }
            }
        }
        let validity = output_validity
            .map(|validity| Validity::from(validity.freeze()))
            .or(preserved_validity)
            .expect("validity must be preserved or transposed");
        Ok(PrimitiveArray::new(output.freeze(), validity))
    })
}
