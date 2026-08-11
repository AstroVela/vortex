// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! WKB geometry literal conversion.

use std::sync::Arc;

use ::wkb::reader::GeometryType;
use arrow_array::BinaryArray;
use geoarrow::array::GenericWkbArray;
use geoarrow::datatypes::CoordType;
use geoarrow::datatypes::Dimension;
use geoarrow::datatypes::GeoArrowType;
use geoarrow::datatypes::LineStringType;
use geoarrow::datatypes::MultiLineStringType;
use geoarrow::datatypes::MultiPointType;
use geoarrow::datatypes::MultiPolygonType;
use geoarrow::datatypes::PointType;
use geoarrow::datatypes::PolygonType;
use geoarrow::datatypes::WkbType;
use geoarrow_cast::cast::cast;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::scalar::Scalar;
use vortex_arrow::FromArrowArray;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::LineString;
use super::MultiLineString;
use super::MultiPoint;
use super::MultiPolygon;
use super::Point;
use super::Polygon;
use super::SpatialMetadata;
use super::Wkb;
use super::metadata::to_geoarrow;

/// Decode a WKB literal to a native geometry scalar.
pub fn native_geometry_scalar_from_wkb(bytes: &[u8]) -> VortexResult<Option<Scalar>> {
    let metadata = to_geoarrow(&SpatialMetadata::default())?;
    let binary = BinaryArray::from(vec![Some(bytes)]);
    let wkb = GenericWkbArray::<i32>::try_from((
        &binary as &dyn arrow_array::Array,
        WkbType::new(Arc::clone(&metadata)),
    ))
    .map_err(|e| vortex_err!("failed to read WKB literal: {e}"))?;

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

// `scalar_at` is deprecated, but literal conversion has no execution context.
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

#[cfg(test)]
mod tests {
    use vortex_array::dtype::DType;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use super::LineString;
    use super::MultiLineString;
    use super::MultiPoint;
    use super::Point;
    use super::Polygon;
    use super::native_geometry_scalar_from_wkb;

    #[test]
    fn decodes_wkb_point_to_native() -> VortexResult<()> {
        let mut wkb = vec![1u8];
        wkb.extend_from_slice(&1u32.to_le_bytes());
        wkb.extend_from_slice(&1.0f64.to_le_bytes());
        wkb.extend_from_slice(&2.0f64.to_le_bytes());

        let scalar = native_geometry_scalar_from_wkb(&wkb)?.expect("a point scalar");
        let DType::Extension(ext) = scalar.dtype() else {
            panic!("expected an extension dtype, got {}", scalar.dtype());
        };
        assert!(ext.is::<Point>());
        Ok(())
    }

    #[test]
    fn decodes_wkb_polygon_to_native() -> VortexResult<()> {
        let ring = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)];
        let mut wkb = vec![1u8];
        wkb.extend_from_slice(&3u32.to_le_bytes());
        wkb.extend_from_slice(&1u32.to_le_bytes());
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

    #[test]
    fn decodes_wkb_linestring_to_native() -> VortexResult<()> {
        let points = [(0.0, 0.0), (1.0, 1.0)];
        let mut wkb = vec![1u8];
        wkb.extend_from_slice(&2u32.to_le_bytes());
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

    #[test]
    fn decodes_wkb_multipoint_to_native() -> VortexResult<()> {
        let points = [(0.0, 0.0), (1.0, 1.0)];
        let mut wkb = vec![1u8];
        wkb.extend_from_slice(&4u32.to_le_bytes());
        let len = u32::try_from(points.len()).map_err(|e| vortex_err!("{e}"))?;
        wkb.extend_from_slice(&len.to_le_bytes());
        for (x, y) in points {
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

    #[test]
    fn decodes_wkb_multilinestring_to_native() -> VortexResult<()> {
        let lines = [[(0.0, 0.0), (1.0, 1.0)], [(2.0, 2.0), (3.0, 3.0)]];
        let mut wkb = vec![1u8];
        wkb.extend_from_slice(&5u32.to_le_bytes());
        let num_lines = u32::try_from(lines.len()).map_err(|e| vortex_err!("{e}"))?;
        wkb.extend_from_slice(&num_lines.to_le_bytes());
        for line in lines {
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
