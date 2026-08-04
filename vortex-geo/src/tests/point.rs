// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow interop for the `vortex.geo.point` extension type (`geoarrow.point`).

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::Float64Array;
use arrow_array::StructArray as ArrowStructArray;
use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_buffer::NullBuffer;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Fields;
use arrow_schema::extension::ExtensionType as _;
use geoarrow::datatypes::CoordType;
use geoarrow::datatypes::Crs;
use geoarrow::datatypes::Dimension as GeoArrowDimension;
use geoarrow::datatypes::Metadata;
use geoarrow::datatypes::PointType;
use vortex_array::VortexSessionExecute;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_arrow::ArrowSessionExt;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::SESSION;
use crate::extension::Point;
use crate::extension::coordinate::Coordinate;
use crate::test_harness::coordinate_from_scalar;
use crate::test_harness::point_column;

/// A `geoarrow.point` Arrow field with separated (struct) XY coordinates.
fn point_field(name: &str, nullable: bool, crs: Option<&str>) -> Field {
    point_field_with_dimension(name, nullable, crs, GeoArrowDimension::XY)
}

/// A `geoarrow.point` Arrow field with separated coordinates in the requested dimension.
fn point_field_with_dimension(
    name: &str,
    nullable: bool,
    crs: Option<&str>,
    dimension: GeoArrowDimension,
) -> Field {
    let crs = crs
        .map(|crs| Crs::from_unknown_crs_type(crs.to_string()))
        .unwrap_or_default();
    let metadata = Arc::new(Metadata::new(crs, None));
    PointType::new(dimension, metadata).to_field(name, nullable)
}

/// An Arrow `Struct<x, y>` point array with non-nullable `Float64` children.
fn arrow_point_struct(xs: Vec<f64>, ys: Vec<f64>) -> ArrowStructArray {
    arrow_point_struct_with_ordinates([("x", xs), ("y", ys)])
}

/// An Arrow `Struct<x, y, z>` Point array with non-nullable `Float64` children.
fn arrow_point_struct_xyz(xs: Vec<f64>, ys: Vec<f64>, zs: Vec<f64>) -> ArrowStructArray {
    arrow_point_struct_with_ordinates([("x", xs), ("y", ys), ("z", zs)])
}

/// An Arrow `Struct<x, y, m>` Point array with non-nullable `Float64` children.
fn arrow_point_struct_xym(xs: Vec<f64>, ys: Vec<f64>, ms: Vec<f64>) -> ArrowStructArray {
    arrow_point_struct_with_ordinates([("x", xs), ("y", ys), ("m", ms)])
}

fn arrow_point_struct_with_ordinates<const N: usize>(
    ordinates: [(&str, Vec<f64>); N],
) -> ArrowStructArray {
    let fields: Fields = ordinates
        .iter()
        .map(|(name, _)| Field::new(*name, DataType::Float64, false))
        .collect::<Vec<_>>()
        .into();
    let columns = ordinates
        .into_iter()
        .map(|(_, values)| Arc::new(Float64Array::from(values)) as ArrowArrayRef)
        .collect();
    ArrowStructArray::new(fields, columns, None)
}

/// The exported Arrow field carries the `geoarrow.point` extension over the separated
/// `Struct<x, y>` coordinate layout.
#[test]
fn export_field_carries_extension() -> VortexResult<()> {
    let array = point_column(vec![1.0], vec![2.0])?;
    let field = SESSION.arrow().to_arrow_field("loc", array.dtype())?;

    assert_eq!(field.extension_type_name(), Some(PointType::NAME));
    let DataType::Struct(fields) = field.data_type() else {
        panic!("expected Struct, got {}", field.data_type());
    };
    assert_eq!(fields.len(), 2);
    assert_eq!(fields[0].name(), "x");
    assert_eq!(fields[0].data_type(), &DataType::Float64);
    assert_eq!(fields[1].name(), "y");
    assert_eq!(fields[1].data_type(), &DataType::Float64);
    Ok(())
}

/// Export materializes the point column as an Arrow struct with the original ordinates.
#[test]
fn exports_to_struct() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = point_column(vec![1.0, 3.0], vec![2.0, 4.0])?;

    let target = point_field("loc", false, Some("EPSG:4326"));
    let exported = SESSION
        .arrow()
        .execute_arrow(array, Some(&target), &mut ctx)?;

    let points = exported.as_struct();
    let ordinates = |name: &str| -> VortexResult<Vec<f64>> {
        Ok(points
            .column_by_name(name)
            .ok_or_else(|| vortex_err!("missing {name} column"))?
            .as_primitive::<Float64Type>()
            .values()
            .to_vec())
    };
    assert_eq!(ordinates("x")?, vec![1.0, 3.0]);
    assert_eq!(ordinates("y")?, vec![2.0, 4.0]);
    Ok(())
}

/// An imported `geoarrow.point` field maps to the Point extension dtype, recovering the
/// CRS, coordinate field names, and nullability.
#[test]
fn import_field_recovers_extension() -> VortexResult<()> {
    let field = point_field("loc", true, Some("EPSG:4326"));
    let dtype = SESSION.arrow().from_arrow_field(&field)?;

    let DType::Extension(ext) = &dtype else {
        panic!("expected Extension dtype, got {dtype}");
    };
    assert!(ext.is::<Point>());
    assert_eq!(ext.metadata::<Point>().crs.as_deref(), Some("EPSG:4326"));

    let DType::Struct(fields, nullability) = ext.storage_dtype() else {
        panic!("expected Struct storage, got {}", ext.storage_dtype());
    };
    assert_eq!(*nullability, Nullability::Nullable);
    let names: Vec<&str> = fields.names().iter().map(|n| n.as_ref()).collect();
    assert_eq!(names, vec!["x", "y"]);
    Ok(())
}

/// A field with interleaved (`FixedSizeList`) coordinates fails to import.
#[test]
fn import_interleaved_field_fails() {
    let point_type = PointType::new(GeoArrowDimension::XY, Default::default())
        .with_coord_type(CoordType::Interleaved);
    let field = point_type.to_field("loc", false);
    assert!(SESSION.arrow().from_arrow_field(&field).is_err());
}

/// Import wraps the Arrow struct's coordinate buffers into a Point column.
#[test]
fn imports_from_struct() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let arrow: ArrowArrayRef =
        Arc::new(arrow_point_struct(vec![1.0, -111.7610], vec![2.0, 34.8697]));
    let field = point_field("loc", false, Some("EPSG:4326"));

    let imported = SESSION.arrow().from_arrow_array(arrow, &field)?;
    assert!(
        imported
            .dtype()
            .as_extension_opt()
            .map(|ext| ext.is::<Point>())
            .unwrap_or(false)
    );

    assert_eq!(
        coordinate_from_scalar(&imported.execute_scalar(0, &mut ctx)?)?,
        Coordinate::xy(1.0, 2.0)
    );
    assert_eq!(
        coordinate_from_scalar(&imported.execute_scalar(1, &mut ctx)?)?,
        Coordinate::xy(-111.7610, 34.8697)
    );
    Ok(())
}

/// GeoArrow's all-NaN Point sentinel is an empty Point and remains importable.
#[test]
fn imports_empty_point_sentinel() -> VortexResult<()> {
    let arrow: ArrowArrayRef = Arc::new(arrow_point_struct(vec![f64::NAN], vec![f64::NAN]));
    let field = point_field("loc", false, Some("EPSG:4326"));

    SESSION.arrow().from_arrow_array(arrow, &field)?;
    Ok(())
}

/// A Point with only one non-finite XY ordinate is malformed, not GeoArrow's empty-Point
/// sentinel, and is rejected before it enters native Vortex storage.
#[test]
fn rejects_partial_nan_point() {
    let arrow: ArrowArrayRef = Arc::new(arrow_point_struct(vec![f64::NAN], vec![5.0]));
    let field = point_field("loc", false, Some("EPSG:4326"));
    assert!(SESSION.arrow().from_arrow_array(arrow, &field).is_err());
}

/// Child values of a null Point row are unspecified and must not be validated.
#[test]
fn ignores_invalid_point_under_null_row() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let field = point_field("loc", true, Some("EPSG:4326"));
    let source = point_column(vec![f64::NAN, 1.0], vec![0.0, 2.0])?;
    let arrow = SESSION
        .arrow()
        .execute_arrow(source, Some(&field), &mut ctx)?;
    let points = arrow.as_struct();
    let nullable = ArrowStructArray::try_new(
        points.fields().clone(),
        points.columns().to_vec(),
        Some(NullBuffer::from(vec![false, true])),
    )
    .map_err(|error| vortex_err!("failed to build nullable points: {error}"))?;

    SESSION
        .arrow()
        .from_arrow_array(Arc::new(nullable), &field)?;
    Ok(())
}

/// A higher-dimensional Point is empty only when every ordinate is NaN. A NaN Z value on a
/// finite XY coordinate remains an attribute value, because native geometry validity is 2-D.
#[test]
fn xyz_empty_point_uses_every_ordinate() -> VortexResult<()> {
    let field = point_field_with_dimension("loc", false, Some("EPSG:4326"), GeoArrowDimension::XYZ);

    let empty: ArrowArrayRef = Arc::new(arrow_point_struct_xyz(
        vec![f64::NAN],
        vec![f64::NAN],
        vec![f64::NAN],
    ));
    SESSION.arrow().from_arrow_array(empty, &field)?;

    let partial: ArrowArrayRef = Arc::new(arrow_point_struct_xyz(
        vec![f64::NAN],
        vec![f64::NAN],
        vec![1.0],
    ));
    assert!(SESSION.arrow().from_arrow_array(partial, &field).is_err());

    let finite_xy: ArrowArrayRef =
        Arc::new(arrow_point_struct_xyz(vec![1.0], vec![2.0], vec![f64::NAN]));
    SESSION.arrow().from_arrow_array(finite_xy, &field)?;
    Ok(())
}

/// M is carried as an attribute ordinate, so a NaN M value does not invalidate finite XY.
#[test]
fn xym_point_keeps_nan_measure() -> VortexResult<()> {
    let field = point_field_with_dimension("loc", false, Some("EPSG:4326"), GeoArrowDimension::XYM);
    let point: ArrowArrayRef =
        Arc::new(arrow_point_struct_xym(vec![1.0], vec![2.0], vec![f64::NAN]));

    SESSION.arrow().from_arrow_array(point, &field)?;
    Ok(())
}

/// A point column exported to Arrow and imported back is unchanged, including the CRS.
#[test]
fn roundtrips_through_arrow() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let original = point_column(vec![0.0, 3.0], vec![4.0, 0.0])?;

    let target = point_field("loc", false, Some("EPSG:4326"));
    let exported = SESSION
        .arrow()
        .execute_arrow(original, Some(&target), &mut ctx)?;
    let reimported = SESSION.arrow().from_arrow_array(exported, &target)?;

    let ext = reimported
        .dtype()
        .as_extension_opt()
        .ok_or_else(|| vortex_err!("expected Extension dtype"))?;
    assert_eq!(ext.metadata::<Point>().crs.as_deref(), Some("EPSG:4326"));

    assert_eq!(
        coordinate_from_scalar(&reimported.execute_scalar(0, &mut ctx)?)?,
        Coordinate::xy(0.0, 4.0)
    );
    assert_eq!(
        coordinate_from_scalar(&reimported.execute_scalar(1, &mut ctx)?)?,
        Coordinate::xy(3.0, 0.0)
    );
    Ok(())
}
