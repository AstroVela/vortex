// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Native geometry helpers and conversion to [`geo_types::Geometry`] for `geo` algorithms.

use geo_types::Geometry;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::list::ListArraySlotsExt;
use vortex_array::arrays::listview::ListViewArraySlotsExt;
use vortex_array::arrays::listview::list_from_list_view;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use super::LineString;
use super::MultiLineString;
use super::MultiPoint;
use super::MultiPolygon;
use super::Point;
use super::Polygon;
use super::Rect;
use super::linestring_geometries;
use super::multilinestring_geometries;
use super::multipoint_geometries;
use super::multipolygon_geometries;
use super::point_geometries;
use super::polygon_geometries;
use super::rect_geometries;

/// Whether `dtype` is a native geometry extension type.
pub(crate) fn is_native_geometry(dtype: &DType) -> bool {
    dtype.as_extension_opt().is_some_and(|ext| {
        ext.is::<Point>()
            || ext.is::<LineString>()
            || ext.is::<MultiPoint>()
            || ext.is::<Polygon>()
            || ext.is::<MultiLineString>()
            || ext.is::<MultiPolygon>()
            || ext.is::<Rect>()
    })
}

/// Flatten a native geometry column to its coordinates.
pub(crate) fn flatten_coordinates(
    array: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<StructArray> {
    if !is_native_geometry(array.dtype()) {
        vortex_bail!(
            "spatial: operand is not a native geometry extension type, was {}",
            array.dtype()
        );
    }
    let mut node = array
        .clone()
        .execute::<ExtensionArray>(ctx)?
        .storage_array()
        .clone();
    while node.dtype().is_list() {
        node = node.execute::<ListViewArray>(ctx)?.elements().clone();
    }
    node.execute::<StructArray>(ctx)
}

/// Flatten native geometry storage and return each row's coordinate offsets.
pub(crate) fn flatten_row_offsets(
    storage: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(Vec<usize>, StructArray)> {
    let mut row_offsets: Vec<usize> = (0..=storage.len()).collect();
    let mut level = storage;
    while level.dtype().is_list() {
        let list = list_from_list_view(level.execute::<ListViewArray>(ctx)?, ctx)?;
        let offsets = list
            .offsets()
            .clone()
            .cast(DType::Primitive(PType::U64, Nullability::NonNullable))?
            .execute::<Buffer<u64>>(ctx)?;
        for row_offset in &mut row_offsets {
            *row_offset = usize::try_from(offsets[*row_offset])
                .map_err(|_| vortex_err!("spatial: list offset exceeds usize"))?;
        }
        level = list.elements().clone();
    }
    Ok((row_offsets, level.execute::<StructArray>(ctx)?))
}

/// Decode a native geometry column to `geo_types`.
pub(crate) fn geometries(
    array: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Vec<Geometry<f64>>> {
    let Some(ext) = array.dtype().as_extension_opt() else {
        vortex_bail!(
            "spatial: operand is not a geometry extension type, was {}",
            array.dtype()
        );
    };
    let storage = array
        .clone()
        .execute::<ExtensionArray>(ctx)?
        .storage_array()
        .clone();
    if ext.is::<Point>() {
        point_geometries(&storage, ext.metadata::<Point>(), ctx)
    } else if ext.is::<LineString>() {
        linestring_geometries(&storage, ext.metadata::<LineString>(), ctx)
    } else if ext.is::<MultiPoint>() {
        multipoint_geometries(&storage, ext.metadata::<MultiPoint>(), ctx)
    } else if ext.is::<Polygon>() {
        polygon_geometries(&storage, ext.metadata::<Polygon>(), ctx)
    } else if ext.is::<MultiLineString>() {
        multilinestring_geometries(&storage, ext.metadata::<MultiLineString>(), ctx)
    } else if ext.is::<MultiPolygon>() {
        multipolygon_geometries(&storage, ext.metadata::<MultiPolygon>(), ctx)
    } else if ext.is::<Rect>() {
        rect_geometries(&storage, ext.metadata::<Rect>(), ctx)
    } else {
        vortex_bail!("spatial: unsupported geometry extension {}", array.dtype())
    }
}

/// Decode a constant operand to one geometry.
pub(crate) fn single_geometry(
    scalar: &Scalar,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Geometry<f64>> {
    let array = ConstantArray::new(scalar.clone(), 1).into_array();
    geometries(&array, ctx)?
        .pop()
        .ok_or_else(|| vortex_err!("spatial: constant operand decoded to no geometry"))
}
