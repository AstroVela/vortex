// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! What the geo scalar functions add to the row-function machinery: an element type that decodes a
//! native geometry column into `geo_types` geometries.

use geo_types::Geometry;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::dtype::DType;
use vortex_array::scalar_fn::InputElement;
use vortex_error::VortexResult;

use crate::extension::geometries;
use crate::extension::validate_geometry_operands;

/// Marker for native geometry input elements: accepts any native geometry column and presents each
/// row as a decoded `geo_types` geometry.
///
/// The two operands of a binary geo function need not share a geometry type, since distance,
/// containment and intersection across types are all meaningful, so this validates only that the
/// column is *some* native geometry.
pub struct GeometryRow;

impl InputElement for GeometryRow {
    type Column = Vec<Geometry<f64>>;
    type Elem<'a> = &'a Geometry<f64>;

    // A geometry row is decoded from its coordinate storage, which behind a null row holds arbitrary
    // coordinates that need not describe a well-formed geometry.
    const DENSE_SAFE: bool = false;
    // Decoding builds a geometry from stored coordinates, and a malformed one in a *valid* row is a
    // domain error rather than an infrastructural failure.
    const DECODE_FALLIBLE: bool = true;

    fn validate(dtype: &DType) -> VortexResult<()> {
        validate_geometry_operands(std::slice::from_ref(dtype))
    }

    fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Column> {
        geometries(&array, ctx)
    }

    fn get(column: &Self::Column, index: usize) -> &Geometry<f64> {
        &column[index]
    }
}
