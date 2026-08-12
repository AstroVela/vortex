// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub(crate) mod coordinate;
mod linestring;
mod multilinestring;
mod multipoint;
mod multipolygon;
mod point;
mod polygon;
mod rect;
mod wkb;

use std::fmt::Display;
use std::sync::Arc;

use ::wkb::reader::GeometryType;
use arrow_array::BinaryArray;
use geo_types::Geometry;
use geoarrow::array::GenericWkbArray;
use geoarrow::array::GeoArrowArray;
use geoarrow::datatypes::CoordType;
use geoarrow::datatypes::Crs;
use geoarrow::datatypes::Dimension;
use geoarrow::datatypes::GeoArrowType;
use geoarrow::datatypes::LineStringType;
use geoarrow::datatypes::Metadata;
use geoarrow::datatypes::MultiLineStringType;
use geoarrow::datatypes::MultiPointType;
use geoarrow::datatypes::MultiPolygonType;
use geoarrow::datatypes::PointType;
use geoarrow::datatypes::PolygonType;
use geoarrow::datatypes::WkbType;
use geoarrow_cast::cast::cast;
pub use linestring::*;
pub use multilinestring::*;
pub use multipoint::*;
pub use multipolygon::*;
pub use point::*;
pub use polygon::*;
pub use rect::*;
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
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::scalar::Scalar;
use vortex_arrow::ArrowSession;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
pub use wkb::*;

use crate::algorithms::Aabb;
use crate::algorithms::Coord;
use crate::algorithms::Coords;
use crate::algorithms::GeometryRef;
use crate::algorithms::MultiLineStringRef;
use crate::algorithms::MultiPolygonRef;
use crate::algorithms::PolygonRef;
use crate::extension::coordinate::ordinates;

/// Whether `dtype` is one of the native geometry extension types the spatial kernels operate on.
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

/// Flatten a native geometry column into a single coordinate `Struct<x, y, ...>` containing
/// every vertex of every geometry.
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

/// Flatten a native geometry `storage` array to its leaf coordinates, keeping track of which
/// row owns which coordinates.
///
/// Returns the flat coordinate `Struct` plus `row_offsets` (one entry per row plus a final
/// cap): row `r`'s coordinates are `coordinates[row_offsets[r]..row_offsets[r + 1]]`, an empty
/// range if the row has none.
///
/// Row boundaries are pushed down one `List` level at a time. A boundary at list `e` moves to
/// `offsets[e]`, where that list's children start. For example, a 2-row `MultiPolygon` column —
/// row 0 = one polygon of two rings (3 + 4 vertices), row 1 = one polygon of one ring (5
/// vertices):
///
/// ```text
/// level                offsets       row_offsets after the level
/// rows     → polygons  [0,1,2]       [0,1,2]    row 1 starts at polygon 1
/// polygons → rings     [0,2,3]       [0,2,3]    row 1 starts at ring 2
/// rings    → vertices  [0,3,7,12]    [0,7,12]   row 1 starts at vertex 7
/// ```
pub(crate) fn flatten_row_offsets(
    storage: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(Vec<usize>, StructArray)> {
    let len = storage.len();

    // At the outermost level, row `r` starts at element `r`; the extra entry caps the last row.
    let mut row_offsets: Vec<usize> = (0..=len).collect();
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

/// A native geometry column canonicalized for row-view access: the leaf ordinate buffers plus
/// every list level's offsets, outer to inner, each absolute into the next level. [`Self::row`]
/// borrows a [`GeometryRef`] without copying coordinates.
///
/// Row validity is not represented here: a null row's storage holds structurally valid
/// placeholder data, so callers compute over every row and mask the results separately (see
/// `ST_Length` for the pattern).
pub(crate) enum GeometryBatch {
    /// `Point` storage: one coordinate per row.
    Point { xs: Buffer<f64>, ys: Buffer<f64> },
    /// `LineString` storage: `rows` maps each row to its vertex run.
    LineString {
        rows: Vec<usize>,
        xs: Buffer<f64>,
        ys: Buffer<f64>,
    },
    /// `MultiPoint` storage: `rows` maps each row to its vertex run.
    MultiPoint {
        rows: Vec<usize>,
        xs: Buffer<f64>,
        ys: Buffer<f64>,
    },
    /// `Polygon` storage: `rows` maps each row to its rings, `rings` each ring to its vertices.
    Polygon {
        rows: Vec<usize>,
        rings: Vec<usize>,
        xs: Buffer<f64>,
        ys: Buffer<f64>,
    },
    /// `MultiLineString` storage: `rows` maps each row to its lines, `lines` each line to its
    /// vertices.
    MultiLineString {
        rows: Vec<usize>,
        lines: Vec<usize>,
        xs: Buffer<f64>,
        ys: Buffer<f64>,
    },
    /// `MultiPolygon` storage: `rows` maps each row to its polygons, `polygons` each polygon to
    /// its rings, and `rings` each ring to its vertices.
    MultiPolygon {
        rows: Vec<usize>,
        polygons: Vec<usize>,
        rings: Vec<usize>,
        xs: Buffer<f64>,
        ys: Buffer<f64>,
    },
    /// `Rect` storage: the four corner ordinates per row.
    Rect {
        xmin: Buffer<f64>,
        ymin: Buffer<f64>,
        xmax: Buffer<f64>,
        ymax: Buffer<f64>,
    },
}

impl GeometryBatch {
    /// Canonicalize a native geometry column for view access. A non-geometry operand is an
    /// error.
    pub(crate) fn try_from_array(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
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
            let (xs, ys) = leaf_ordinates(storage, ctx)?;
            Ok(GeometryBatch::Point { xs, ys })
        } else if ext.is::<LineString>() {
            let (rows, coords) = list_level(storage, ctx)?;
            let (xs, ys) = leaf_ordinates(coords, ctx)?;
            Ok(GeometryBatch::LineString { rows, xs, ys })
        } else if ext.is::<MultiPoint>() {
            let (rows, coords) = list_level(storage, ctx)?;
            let (xs, ys) = leaf_ordinates(coords, ctx)?;
            Ok(GeometryBatch::MultiPoint { rows, xs, ys })
        } else if ext.is::<Polygon>() {
            let (rows, ring_lists) = list_level(storage, ctx)?;
            let (rings, coords) = list_level(ring_lists, ctx)?;
            let (xs, ys) = leaf_ordinates(coords, ctx)?;
            Ok(GeometryBatch::Polygon {
                rows,
                rings,
                xs,
                ys,
            })
        } else if ext.is::<MultiLineString>() {
            let (rows, line_lists) = list_level(storage, ctx)?;
            let (lines, coords) = list_level(line_lists, ctx)?;
            let (xs, ys) = leaf_ordinates(coords, ctx)?;
            Ok(GeometryBatch::MultiLineString {
                rows,
                lines,
                xs,
                ys,
            })
        } else if ext.is::<MultiPolygon>() {
            let (rows, polygon_lists) = list_level(storage, ctx)?;
            let (polygons, ring_lists) = list_level(polygon_lists, ctx)?;
            let (rings, coords) = list_level(ring_lists, ctx)?;
            let (xs, ys) = leaf_ordinates(coords, ctx)?;
            Ok(GeometryBatch::MultiPolygon {
                rows,
                polygons,
                rings,
                xs,
                ys,
            })
        } else if ext.is::<Rect>() {
            let corners = storage.execute::<StructArray>(ctx)?;
            Ok(GeometryBatch::Rect {
                xmin: ordinates(&corners, "xmin", ctx)?,
                ymin: ordinates(&corners, "ymin", ctx)?,
                xmax: ordinates(&corners, "xmax", ctx)?,
                ymax: ordinates(&corners, "ymax", ctx)?,
            })
        } else {
            vortex_bail!("spatial: unsupported geometry extension {}", array.dtype())
        }
    }

    /// Canonicalize one non-null geometry scalar as a single-row batch (the constant-operand
    /// path).
    pub(crate) fn try_from_scalar(scalar: &Scalar, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        Self::try_from_array(&ConstantArray::new(scalar.clone(), 1).into_array(), ctx)
    }

    /// The number of rows.
    pub(crate) fn len(&self) -> usize {
        match self {
            GeometryBatch::Point { xs, .. } => xs.len(),
            GeometryBatch::LineString { rows, .. }
            | GeometryBatch::MultiPoint { rows, .. }
            | GeometryBatch::Polygon { rows, .. }
            | GeometryBatch::MultiLineString { rows, .. }
            | GeometryBatch::MultiPolygon { rows, .. } => rows.len() - 1,
            GeometryBatch::Rect { xmin, .. } => xmin.len(),
        }
    }

    /// Borrow row `index` as a [`GeometryRef`] view.
    pub(crate) fn row(&self, index: usize) -> GeometryRef<'_> {
        match self {
            GeometryBatch::Point { xs, ys } => GeometryRef::Point(Coord {
                x: xs[index],
                y: ys[index],
            }),
            GeometryBatch::LineString { rows, xs, ys } => {
                let (start, end) = (rows[index], rows[index + 1]);
                GeometryRef::LineString(Coords::new(&xs[start..end], &ys[start..end]))
            }
            GeometryBatch::MultiPoint { rows, xs, ys } => {
                let (start, end) = (rows[index], rows[index + 1]);
                GeometryRef::MultiPoint(Coords::new(&xs[start..end], &ys[start..end]))
            }
            GeometryBatch::Polygon {
                rows,
                rings,
                xs,
                ys,
            } => GeometryRef::Polygon(PolygonRef::new(
                xs,
                ys,
                &rings[rows[index]..=rows[index + 1]],
            )),
            GeometryBatch::MultiLineString {
                rows,
                lines,
                xs,
                ys,
            } => GeometryRef::MultiLineString(MultiLineStringRef::new(
                xs,
                ys,
                &lines[rows[index]..=rows[index + 1]],
            )),
            GeometryBatch::MultiPolygon {
                rows,
                polygons,
                rings,
                xs,
                ys,
            } => GeometryRef::MultiPolygon(MultiPolygonRef::new(
                xs,
                ys,
                &polygons[rows[index]..=rows[index + 1]],
                rings,
            )),
            GeometryBatch::Rect {
                xmin,
                ymin,
                xmax,
                ymax,
            } => GeometryRef::Rect(Aabb::new(
                xmin[index],
                ymin[index],
                xmax[index],
                ymax[index],
            )),
        }
    }
}

/// Convert one list level to canonical offsets, returning them with the elements underneath.
/// The offsets are position-aligned with the level's rows and absolute into the elements.
fn list_level(level: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<(Vec<usize>, ArrayRef)> {
    let list = list_from_list_view(level.execute::<ListViewArray>(ctx)?, ctx)?;
    let offsets = list
        .offsets()
        .clone()
        .cast(DType::Primitive(PType::U64, Nullability::NonNullable))?
        .execute::<Buffer<u64>>(ctx)?;
    debug_assert_eq!(offsets.len(), list.len() + 1);
    let offsets = offsets
        .iter()
        .map(|&offset| {
            usize::try_from(offset).map_err(|_| vortex_err!("spatial: list offset exceeds usize"))
        })
        .collect::<VortexResult<Vec<usize>>>()?;
    Ok((offsets, list.elements().clone()))
}

/// The `x`/`y` ordinate buffers of a coordinate leaf.
fn leaf_ordinates(
    leaf: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(Buffer<f64>, Buffer<f64>)> {
    let coords = leaf.execute::<StructArray>(ctx)?;
    Ok((ordinates(&coords, "x", ctx)?, ordinates(&coords, "y", ctx)?))
}

/// Decode a native geometry column to `geo_types`. A non-geometry operand is an error.
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
        point_geometries(&storage, ctx)
    } else if ext.is::<LineString>() {
        linestring_geometries(&storage, ctx)
    } else if ext.is::<MultiPoint>() {
        multipoint_geometries(&storage, ctx)
    } else if ext.is::<Polygon>() {
        polygon_geometries(&storage, ctx)
    } else if ext.is::<MultiLineString>() {
        multilinestring_geometries(&storage, ctx)
    } else if ext.is::<MultiPolygon>() {
        multipolygon_geometries(&storage, ctx)
    } else if ext.is::<Rect>() {
        rect_geometries(&storage, ctx)
    } else {
        vortex_bail!("spatial: unsupported geometry extension {}", array.dtype())
    }
}

/// Decode a constant operand scalar to one geometry, a constant of any
/// supported geometry type is decoded exactly like a column.
pub(crate) fn single_geometry(
    scalar: &Scalar,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Geometry<f64>> {
    let array = ConstantArray::new(scalar.clone(), 1).into_array();
    geometries(&array, ctx)?
        .pop()
        .ok_or_else(|| vortex_err!("spatial: constant operand decoded to no geometry"))
}

/// Decode a WKB geometry literal (DuckDB's wire form for `GEOMETRY` constants) to its native
/// `Point`/`Polygon`/`MultiPolygon` scalar. `None` for unsupported types. Plan-time, one value only.
pub fn native_geometry_scalar_from_wkb(
    bytes: &[u8],
    session: &ArrowSession,
) -> VortexResult<Option<Scalar>> {
    let metadata = geoarrow_metadata(&SpatialMetadata::default());
    let binary = BinaryArray::from(vec![Some(bytes)]);
    let wkb = GenericWkbArray::<i32>::try_from((
        &binary as &dyn arrow_array::Array,
        WkbType::new(Arc::clone(&metadata)),
    ))
    .map_err(|e| vortex_err!("failed to read WKB literal: {e}"))?;

    // Cast the WKB value to `target`, import its native storage as a Vortex array.
    let to_storage = |target: &GeoArrowType| -> VortexResult<ArrayRef> {
        let native =
            cast(&wkb, target).map_err(|e| vortex_err!("failed to cast WKB literal: {e}"))?;
        session.from_arrow_array(native.to_array_ref(), false)
    };

    let scalar = match Wkb::try_from_bytes(bytes)?.geometry_type() {
        GeometryType::Point => {
            let target = GeoArrowType::Point(
                PointType::new(Dimension::XY, metadata).with_coord_type(CoordType::Separated),
            );
            spatial_ext_scalar(Point, to_storage(&target)?)?
        }
        GeometryType::LineString => {
            let target = GeoArrowType::LineString(
                LineStringType::new(Dimension::XY, metadata).with_coord_type(CoordType::Separated),
            );
            spatial_ext_scalar(LineString, to_storage(&target)?)?
        }
        GeometryType::Polygon => {
            let target = GeoArrowType::Polygon(
                PolygonType::new(Dimension::XY, metadata).with_coord_type(CoordType::Separated),
            );
            spatial_ext_scalar(Polygon, to_storage(&target)?)?
        }
        GeometryType::MultiPoint => {
            let target = GeoArrowType::MultiPoint(
                MultiPointType::new(Dimension::XY, metadata).with_coord_type(CoordType::Separated),
            );
            spatial_ext_scalar(MultiPoint, to_storage(&target)?)?
        }
        GeometryType::MultiLineString => {
            let target = GeoArrowType::MultiLineString(
                MultiLineStringType::new(Dimension::XY, metadata)
                    .with_coord_type(CoordType::Separated),
            );
            spatial_ext_scalar(MultiLineString, to_storage(&target)?)?
        }
        GeometryType::MultiPolygon => {
            let target = GeoArrowType::MultiPolygon(
                MultiPolygonType::new(Dimension::XY, metadata)
                    .with_coord_type(CoordType::Separated),
            );
            spatial_ext_scalar(MultiPolygon, to_storage(&target)?)?
        }
        _ => return Ok(None),
    };
    Ok(Some(scalar))
}

/// Wrap cast-from-WKB `storage` in its `vtable` extension type and pull out the single scalar.
// `scalar_at` is deprecated for `execute_scalar`, but there is no execution context at plan time.
#[allow(deprecated)]
fn spatial_ext_scalar<V: ExtVTable<Metadata = SpatialMetadata>>(
    vtable: V,
    storage: ArrayRef,
) -> VortexResult<Scalar> {
    let ext =
        ExtDType::try_with_vtable(vtable, SpatialMetadata::default(), storage.dtype().clone())?
            .erased();
    ExtensionArray::try_new(ext, storage)?
        .into_array()
        .scalar_at(0)
}

/// Extension metadata that is common to all the spatial extension types.
///
/// Currently, this is just the coordinate reference system (CRS).
/// We may wish to add a second field for edges interpretation in the future similar to
/// the GeoArrow standard.
#[derive(Clone, PartialEq, Eq, Hash, prost::Message)]
pub struct SpatialMetadata {
    #[prost(optional, string, tag = "1")]
    pub crs: Option<String>,
}

impl Display for SpatialMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.crs.as_ref() {
            Some(crs) => write!(f, "Geometry(crs={crs})"),
            None => write!(f, "Geometry(unreferenced)"),
        }
    }
}

/// The GeoArrow [`Metadata`] equivalent of `spatial_metadata`.
pub(crate) fn geoarrow_metadata(spatial_metadata: &SpatialMetadata) -> Arc<Metadata> {
    Arc::new(Metadata::new(
        spatial_metadata
            .crs
            .as_ref()
            .map(|crs| Crs::from_unknown_crs_type(crs.to_string()))
            .unwrap_or_default(),
        None,
    ))
}

/// Serialize a native geometry array to WKB (a `WkbView` array) via geoarrow's cast.
/// Shared by the `to_wkb` methods on the geometry extension types.
pub(crate) fn geoarrow_to_wkb(
    geoarrow_array: &dyn GeoArrowArray,
    session: &ArrowSession,
) -> VortexResult<ArrayRef> {
    let wkb_type =
        GeoArrowType::WkbView(WkbType::new(geoarrow_metadata(&SpatialMetadata::default())));
    let wkb = cast(geoarrow_array, &wkb_type)
        .map_err(|e| vortex_err!("failed to cast geometry to WKB: {e}"))?;
    session.from_arrow_array(wkb.to_array_ref(), false)
}

/// Recover [`SpatialMetadata`] from GeoArrow metadata.
pub(crate) fn spatial_metadata_from_arrow(metadata: &Metadata) -> SpatialMetadata {
    let crs = metadata.crs().crs_value().map(|value| {
        // `Crs::from_unknown_crs_type` stores the user's string verbatim as a JSON string
        // value, so prefer the raw string when available to round-trip cleanly. For other
        // CRS encodings (PROJJSON object, etc.), fall back to the JSON-encoded form.
        value
            .as_str()
            .map(str::to_string)
            .unwrap_or_else(|| value.to_string())
    });
    SpatialMetadata { crs }
}

#[cfg(test)]
mod tests {
    use prost::Message;
    use vortex_array::dtype::DType;
    use vortex_arrow::ArrowSessionExt;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use super::LineString;
    use super::MultiLineString;
    use super::MultiPoint;
    use super::Point;
    use super::Polygon;
    use crate::extension::SpatialMetadata;

    /// Test shim: decode with an explicitly constructed session.
    fn native_geometry_scalar_from_wkb(
        bytes: &[u8],
    ) -> VortexResult<Option<vortex_array::scalar::Scalar>> {
        let session = vortex_array::array_session();
        super::native_geometry_scalar_from_wkb(bytes, &session.arrow())
    }

    #[test]
    fn test_metadata() {
        let meta = SpatialMetadata {
            crs: Some("EPSG:4326".to_string()),
        };

        assert_eq!(meta.to_string(), "Geometry(crs=EPSG:4326)");
        // round trip
        let bytes = meta.encode_to_vec();
        let decoded = SpatialMetadata::decode(bytes.as_slice()).unwrap();
        assert_eq!(decoded, meta);
    }

    /// A little-endian WKB `POINT` literal decodes to the native `Point` extension scalar.
    #[test]
    fn decodes_wkb_point_to_native() -> VortexResult<()> {
        let mut wkb = vec![1u8]; // little-endian byte order
        wkb.extend_from_slice(&1u32.to_le_bytes()); // geometry type: point
        wkb.extend_from_slice(&1.0f64.to_le_bytes()); // x
        wkb.extend_from_slice(&2.0f64.to_le_bytes()); // y

        let scalar = native_geometry_scalar_from_wkb(&wkb)?.expect("a point scalar");
        let DType::Extension(ext) = scalar.dtype() else {
            panic!("expected an extension dtype, got {}", scalar.dtype());
        };
        assert!(ext.is::<Point>());
        Ok(())
    }

    /// A little-endian WKB `POLYGON` literal decodes to the native `Polygon` extension scalar.
    #[test]
    fn decodes_wkb_polygon_to_native() -> VortexResult<()> {
        let ring = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)];
        let mut wkb = vec![1u8]; // little-endian byte order
        wkb.extend_from_slice(&3u32.to_le_bytes()); // geometry type: polygon
        wkb.extend_from_slice(&1u32.to_le_bytes()); // one ring
        let ring_len = u32::try_from(ring.len()).map_err(|e| vortex_err!("{e}"))?;
        wkb.extend_from_slice(&ring_len.to_le_bytes());
        for (x, y) in ring {
            wkb.extend_from_slice(&f64::to_le_bytes(x));
            wkb.extend_from_slice(&f64::to_le_bytes(y));
        }

        let scalar = native_geometry_scalar_from_wkb(&wkb)?.expect("a polygon scalar");
        let DType::Extension(ext) = scalar.dtype() else {
            panic!("expected an extension dtype, got {}", scalar.dtype());
        };
        assert!(ext.is::<Polygon>());
        Ok(())
    }

    /// A little-endian WKB `LINESTRING` literal decodes to the native `LineString` extension scalar.
    #[test]
    fn decodes_wkb_linestring_to_native() -> VortexResult<()> {
        let points = [(0.0, 0.0), (1.0, 1.0)];
        let mut wkb = vec![1u8]; // little-endian byte order
        wkb.extend_from_slice(&2u32.to_le_bytes()); // geometry type: linestring
        let len = u32::try_from(points.len()).map_err(|e| vortex_err!("{e}"))?;
        wkb.extend_from_slice(&len.to_le_bytes());
        for (x, y) in points {
            wkb.extend_from_slice(&f64::to_le_bytes(x));
            wkb.extend_from_slice(&f64::to_le_bytes(y));
        }

        let scalar = native_geometry_scalar_from_wkb(&wkb)?.expect("a linestring scalar");
        let DType::Extension(ext) = scalar.dtype() else {
            panic!("expected an extension dtype, got {}", scalar.dtype());
        };
        assert!(ext.is::<LineString>());
        Ok(())
    }

    /// A little-endian WKB `MULTIPOINT` literal decodes to the native `MultiPoint` extension scalar.
    #[test]
    fn decodes_wkb_multipoint_to_native() -> VortexResult<()> {
        let points = [(0.0, 0.0), (1.0, 1.0)];
        let mut wkb = vec![1u8]; // little-endian byte order
        wkb.extend_from_slice(&4u32.to_le_bytes()); // geometry type: multipoint
        let len = u32::try_from(points.len()).map_err(|e| vortex_err!("{e}"))?;
        wkb.extend_from_slice(&len.to_le_bytes());
        for (x, y) in points {
            // each member is a full WKB point
            wkb.push(1u8);
            wkb.extend_from_slice(&1u32.to_le_bytes());
            wkb.extend_from_slice(&f64::to_le_bytes(x));
            wkb.extend_from_slice(&f64::to_le_bytes(y));
        }

        let scalar = native_geometry_scalar_from_wkb(&wkb)?.expect("a multipoint scalar");
        let DType::Extension(ext) = scalar.dtype() else {
            panic!("expected an extension dtype, got {}", scalar.dtype());
        };
        assert!(ext.is::<MultiPoint>());
        Ok(())
    }

    /// A little-endian WKB `MULTILINESTRING` literal decodes to the native `MultiLineString` scalar.
    #[test]
    fn decodes_wkb_multilinestring_to_native() -> VortexResult<()> {
        let lines = [[(0.0, 0.0), (1.0, 1.0)], [(2.0, 2.0), (3.0, 3.0)]];
        let mut wkb = vec![1u8]; // little-endian byte order
        wkb.extend_from_slice(&5u32.to_le_bytes()); // geometry type: multilinestring
        let num_lines = u32::try_from(lines.len()).map_err(|e| vortex_err!("{e}"))?;
        wkb.extend_from_slice(&num_lines.to_le_bytes());
        for line in lines {
            // each member is a full WKB linestring
            wkb.push(1u8);
            wkb.extend_from_slice(&2u32.to_le_bytes());
            let len = u32::try_from(line.len()).map_err(|e| vortex_err!("{e}"))?;
            wkb.extend_from_slice(&len.to_le_bytes());
            for (x, y) in line {
                wkb.extend_from_slice(&f64::to_le_bytes(x));
                wkb.extend_from_slice(&f64::to_le_bytes(y));
            }
        }

        let scalar = native_geometry_scalar_from_wkb(&wkb)?.expect("a multilinestring scalar");
        let DType::Extension(ext) = scalar.dtype() else {
            panic!("expected an extension dtype, got {}", scalar.dtype());
        };
        assert!(ext.is::<MultiLineString>());
        Ok(())
    }
}
