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

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::fmt::Display;
use std::hash::Hash;
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
use vortex_array::arrays::Dict;
use vortex_array::arrays::Extension;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::dict::DictArraySlotsExt;
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
use vortex_arrow::FromArrowArray;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
pub use wkb::*;

/// Whether `dtype` is one of the native geometry extension types the geo kernels operate on.
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

/// Validate the operands of a geo scalar function: each must be a native geometry type so the
/// kernel can decode it. The two operands need not share a geometry type — e.g. a `Point` against
/// a `Polygon` is valid, since distance/containment/intersection across types is meaningful.
/// Nullable operands are allowed; the kernels propagate nulls (a null geometry input yields a null
/// result) rather than decoding null rows.
pub(crate) fn validate_geometry_operands(dtypes: &[DType]) -> VortexResult<()> {
    for dtype in dtypes {
        vortex_ensure!(
            is_native_geometry(dtype),
            "geo: operand {dtype} is not a native geometry type"
        );
    }
    Ok(())
}

/// Flatten a native geometry column into a single coordinate `Struct<x, y, ...>` containing
/// every vertex of every geometry.
pub(crate) fn flatten_coordinates(
    array: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<StructArray> {
    if !is_native_geometry(array.dtype()) {
        vortex_bail!(
            "geo: operand is not a native geometry extension type, was {}",
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
                .map_err(|_| vortex_err!("geo: list offset exceeds usize"))?;
        }
        level = list.elements().clone();
    }
    Ok((row_offsets, level.execute::<StructArray>(ctx)?))
}

/// Decode a native geometry column to `geo_types`. A non-geometry operand is an error.
pub(crate) fn geometries(
    array: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Vec<Geometry<f64>>> {
    let Some(ext) = array.dtype().as_extension_opt() else {
        vortex_bail!(
            "geo: operand is not a geometry extension type, was {}",
            array.dtype()
        );
    };
    // Preserve a reduced extension's lazy ListView storage. Executing the extension first would
    // compact duplicate list views and lose the shared row identity before we can deduplicate it.
    let storage = match array.as_opt::<Extension>() {
        Some(extension) => extension.storage_array().clone(),
        None => array
            .clone()
            .execute::<ExtensionArray>(ctx)?
            .storage_array()
            .clone(),
    };
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
        vortex_bail!("geo: unsupported geometry extension {}", array.dtype())
    }
}

/// Owned geometries decoded once per distinct native list row, plus the row mapping needed to
/// preserve the input's logical repetition.
///
/// A spatial join can `take` the same build geometry many times. Native list views preserve the
/// shared coordinate buffers, but flattening those views before decoding otherwise copies every
/// repeated row. Keeping this indirection lets binary kernels reuse one decoded geometry for all
/// candidate rows that reference the same list range.
pub(crate) struct GeometryRows {
    geometries: Vec<Geometry<f64>>,
    row_to_geometry: Option<Vec<usize>>,
}

impl GeometryRows {
    pub(crate) fn geometry(&self, row: usize) -> &Geometry<f64> {
        let geometry = self
            .row_to_geometry
            .as_ref()
            .map_or(row, |mapping| mapping[row]);
        &self.geometries[geometry]
    }
}

/// Decode a valid native geometry column while retaining duplicate top-level list views as shared
/// references. Non-list geometries and columns without duplicate rows use the ordinary decoder.
pub(crate) fn geometry_rows(
    array: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<GeometryRows> {
    if array.dtype().as_extension_opt().is_none() {
        vortex_bail!(
            "geo: operand is not a geometry extension type, was {}",
            array.dtype()
        );
    }
    let storage = match array.as_opt::<Extension>() {
        Some(extension) => extension.storage_array().clone(),
        None => array
            .clone()
            .execute::<ExtensionArray>(ctx)?
            .storage_array()
            .clone(),
    };
    if array.len() < 2 {
        return Ok(GeometryRows {
            geometries: geometries(array, ctx)?,
            row_to_geometry: None,
        });
    }

    let slot_dtype = DType::Primitive(PType::U64, Nullability::NonNullable);
    let deduplicated = if let Some(dict) = storage.as_opt::<Dict>() {
        let codes = dict
            .codes()
            .clone()
            .cast(slot_dtype.clone())?
            .execute::<Buffer<u64>>(ctx)?;
        distinct_geometry_rows(codes.iter().copied())?
    } else if let Some(list) = storage.as_opt::<vortex_array::arrays::ListView>() {
        let offsets = list
            .offsets()
            .clone()
            .cast(slot_dtype.clone())?
            .execute::<Buffer<u64>>(ctx)?;
        let sizes = list
            .sizes()
            .clone()
            .cast(slot_dtype)?
            .execute::<Buffer<u64>>(ctx)?;
        distinct_geometry_rows(offsets.iter().copied().zip(sizes.iter().copied()))?
    } else {
        None
    };

    let Some((representatives, row_to_geometry)) = deduplicated else {
        return Ok(GeometryRows {
            geometries: geometries(array, ctx)?,
            row_to_geometry: None,
        });
    };

    let representatives = PrimitiveArray::from_iter(representatives).into_array();
    let unique = array.clone().take(representatives)?;
    let geometries = geometries(&unique, ctx)?;
    vortex_ensure!(
        geometries.len()
            == row_to_geometry
                .iter()
                .copied()
                .max()
                .map_or(0, |last| last + 1),
        "geo: distinct geometry row count changed while decoding"
    );
    Ok(GeometryRows {
        geometries,
        row_to_geometry: Some(row_to_geometry),
    })
}

/// Return representative input rows and the row-to-representative mapping when `keys` contains
/// duplicates. `None` keeps the ordinary decoder's zero-indirection path for all-distinct input.
fn distinct_geometry_rows<K>(
    keys: impl ExactSizeIterator<Item = K>,
) -> VortexResult<Option<(Vec<u64>, Vec<usize>)>>
where
    K: Eq + Hash,
{
    let len = keys.len();
    let mut distinct = HashMap::with_capacity(len);
    let mut representatives = Vec::new();
    let mut row_to_geometry = Vec::with_capacity(len);
    for (row, key) in keys.enumerate() {
        let next = distinct.len();
        let geometry = match distinct.entry(key) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                representatives.push(
                    u64::try_from(row)
                        .map_err(|_| vortex_err!("geo: geometry row index exceeds u64"))?,
                );
                entry.insert(next);
                next
            }
        };
        row_to_geometry.push(geometry);
    }
    Ok((representatives.len() != len).then_some((representatives, row_to_geometry)))
}

/// Decode a constant operand scalar to one geo geometry, a constant of any
/// supported geometry type is decoded exactly like a column.
pub(crate) fn single_geometry(
    scalar: &Scalar,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Geometry<f64>> {
    let array = ConstantArray::new(scalar.clone(), 1).into_array();
    geometries(&array, ctx)?
        .pop()
        .ok_or_else(|| vortex_err!("geo: constant operand decoded to no geometry"))
}

/// Decode a WKB geometry literal (DuckDB's wire form for `GEOMETRY` constants) to its native
/// `Point`/`Polygon`/`MultiPolygon` scalar. `None` for unsupported types. Plan-time, one value only.
pub fn native_geometry_scalar_from_wkb(bytes: &[u8]) -> VortexResult<Option<Scalar>> {
    let metadata = geoarrow_metadata(&GeoMetadata::default());
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
        ArrayRef::from_arrow(native.to_array_ref().as_ref(), false)
    };

    let scalar = match Wkb::try_from_bytes(bytes)?.geometry_type() {
        GeometryType::Point => {
            let target = GeoArrowType::Point(
                PointType::new(Dimension::XY, metadata).with_coord_type(CoordType::Separated),
            );
            geo_ext_scalar(Point, to_storage(&target)?)?
        }
        GeometryType::LineString => {
            let target = GeoArrowType::LineString(
                LineStringType::new(Dimension::XY, metadata).with_coord_type(CoordType::Separated),
            );
            geo_ext_scalar(LineString, to_storage(&target)?)?
        }
        GeometryType::Polygon => {
            let target = GeoArrowType::Polygon(
                PolygonType::new(Dimension::XY, metadata).with_coord_type(CoordType::Separated),
            );
            geo_ext_scalar(Polygon, to_storage(&target)?)?
        }
        GeometryType::MultiPoint => {
            let target = GeoArrowType::MultiPoint(
                MultiPointType::new(Dimension::XY, metadata).with_coord_type(CoordType::Separated),
            );
            geo_ext_scalar(MultiPoint, to_storage(&target)?)?
        }
        GeometryType::MultiLineString => {
            let target = GeoArrowType::MultiLineString(
                MultiLineStringType::new(Dimension::XY, metadata)
                    .with_coord_type(CoordType::Separated),
            );
            geo_ext_scalar(MultiLineString, to_storage(&target)?)?
        }
        GeometryType::MultiPolygon => {
            let target = GeoArrowType::MultiPolygon(
                MultiPolygonType::new(Dimension::XY, metadata)
                    .with_coord_type(CoordType::Separated),
            );
            geo_ext_scalar(MultiPolygon, to_storage(&target)?)?
        }
        _ => return Ok(None),
    };
    Ok(Some(scalar))
}

/// Wrap cast-from-WKB `storage` in its `vtable` extension type and pull out the single scalar.
// `scalar_at` is deprecated for `execute_scalar`, but there is no execution context at plan time.
#[allow(deprecated)]
fn geo_ext_scalar<V: ExtVTable<Metadata = GeoMetadata>>(
    vtable: V,
    storage: ArrayRef,
) -> VortexResult<Scalar> {
    let ext = ExtDType::try_with_vtable(vtable, GeoMetadata::default(), storage.dtype().clone())?
        .erased();
    ExtensionArray::try_new(ext, storage)?
        .into_array()
        .scalar_at(0)
}

/// Extension metadata that is common to all the geospatial extension types.
///
/// Currently, this is just the coordinate reference system (CRS).
/// We may wish to add a second field for edges interpretation in the future similar to
/// the GeoArrow standard.
#[derive(Clone, PartialEq, Eq, Hash, prost::Message)]
pub struct GeoMetadata {
    #[prost(optional, string, tag = "1")]
    pub crs: Option<String>,
}

impl Display for GeoMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.crs.as_ref() {
            Some(crs) => write!(f, "Geometry(crs={crs})"),
            None => write!(f, "Geometry(unreferenced)"),
        }
    }
}

/// The GeoArrow [`Metadata`] equivalent of `geo_metadata`.
pub(crate) fn geoarrow_metadata(geo_metadata: &GeoMetadata) -> Arc<Metadata> {
    Arc::new(Metadata::new(
        geo_metadata
            .crs
            .as_ref()
            .map(|crs| Crs::from_unknown_crs_type(crs.to_string()))
            .unwrap_or_default(),
        None,
    ))
}

/// Serialize a native geometry array to WKB (a `WkbView` array) via geoarrow's cast.
/// Shared by the `to_wkb` methods on the geometry extension types.
pub(crate) fn geoarrow_to_wkb(geo_array: &dyn GeoArrowArray) -> VortexResult<ArrayRef> {
    let wkb_type = GeoArrowType::WkbView(WkbType::new(geoarrow_metadata(&GeoMetadata::default())));
    let wkb = cast(geo_array, &wkb_type)
        .map_err(|e| vortex_err!("failed to cast geometry to WKB: {e}"))?;
    ArrayRef::from_arrow(wkb.to_array_ref().as_ref(), false)
}

/// Recover [`GeoMetadata`] from GeoArrow metadata.
pub(crate) fn geo_metadata_from_arrow(metadata: &Metadata) -> GeoMetadata {
    let crs = metadata.crs().crs_value().map(|value| {
        // `Crs::from_unknown_crs_type` stores the user's string verbatim as a JSON string
        // value, so prefer the raw string when available to round-trip cleanly. For other
        // CRS encodings (PROJJSON object, etc.), fall back to the JSON-encoded form.
        value
            .as_str()
            .map(str::to_string)
            .unwrap_or_else(|| value.to_string())
    });
    GeoMetadata { crs }
}

#[cfg(test)]
mod tests {
    use prost::Message;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::DType;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use super::LineString;
    use super::MultiLineString;
    use super::MultiPoint;
    use super::Point;
    use super::Polygon;
    use super::geometry_rows;
    use super::native_geometry_scalar_from_wkb;
    use crate::test_harness::polygon_column;

    #[test]
    fn repeated_list_views_decode_once() -> VortexResult<()> {
        let geometries = polygon_column(vec![
            vec![vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)]],
            vec![vec![(10.0, 10.0), (11.0, 10.0), (10.0, 11.0), (10.0, 10.0)]],
        ])?;
        let indices = PrimitiveArray::from_iter([0u64, 0, 1, 0]).into_array();
        let repeated = geometries.take(indices)?;
        let mut ctx = vortex_array::array_session().create_execution_ctx();
        let decoded = geometry_rows(&repeated, &mut ctx)?;

        assert_eq!(decoded.geometries.len(), 2);
        assert_eq!(decoded.geometry(0), decoded.geometry(1));
        assert_eq!(decoded.geometry(0), decoded.geometry(3));
        assert_ne!(decoded.geometry(0), decoded.geometry(2));
        Ok(())
    }
    use crate::extension::GeoMetadata;

    #[test]
    fn test_metadata() {
        let meta = GeoMetadata {
            crs: Some("EPSG:4326".to_string()),
        };

        assert_eq!(meta.to_string(), "Geometry(crs=EPSG:4326)");
        // round trip
        let bytes = meta.encode_to_vec();
        let decoded = GeoMetadata::decode(bytes.as_slice()).unwrap();
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
