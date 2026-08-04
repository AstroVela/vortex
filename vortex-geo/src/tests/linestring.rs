// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow interop for the `vortex.geo.linestring` extension type (`geoarrow.linestring`).

use std::sync::Arc;

use arrow_array::Array;
use arrow_array::ListArray as ArrowListArray;
use arrow_buffer::NullBuffer;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::extension::ExtensionType as _;
use geoarrow::datatypes::CoordType;
use geoarrow::datatypes::Crs;
use geoarrow::datatypes::Dimension as GeoArrowDimension;
use geoarrow::datatypes::LineStringType;
use geoarrow::datatypes::Metadata;
use vortex_array::VortexSessionExecute;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_arrow::ArrowSessionExt;
use vortex_error::VortexResult;

use super::SESSION;
use crate::extension::LineString;
use crate::test_harness::linestring_column;

/// A `geoarrow.linestring` Arrow field with separated (struct) XY coordinates.
fn linestring_field(name: &str, nullable: bool, crs: Option<&str>) -> Field {
    let crs = crs
        .map(|crs| Crs::from_unknown_crs_type(crs.to_string()))
        .unwrap_or_default();
    let metadata = Arc::new(Metadata::new(crs, None));
    LineStringType::new(GeoArrowDimension::XY, metadata).to_field(name, nullable)
}

/// An imported `geoarrow.linestring` field maps to the LineString extension dtype, recovering the
/// CRS, the `List<Struct<x, y>>` storage, and nullability.
#[test]
fn import_field_recovers_extension() -> VortexResult<()> {
    let field = linestring_field("geom", true, Some("EPSG:4326"));
    let dtype = SESSION.arrow().from_arrow_field(&field)?;

    let DType::Extension(ext) = &dtype else {
        panic!("expected Extension dtype, got {dtype}");
    };
    assert!(ext.is::<LineString>());
    assert_eq!(
        ext.metadata::<LineString>().crs.as_deref(),
        Some("EPSG:4326")
    );

    // Storage peels one List layer (linestring → coordinates) to the coordinate struct.
    let DType::List(coords, nullability) = ext.storage_dtype() else {
        panic!("expected List storage, got {}", ext.storage_dtype());
    };
    assert_eq!(*nullability, Nullability::Nullable);
    let DType::Struct(fields, _) = coords.as_ref() else {
        panic!("expected coordinate Struct");
    };
    let names: Vec<&str> = fields.names().iter().map(|n| n.as_ref()).collect();
    assert_eq!(names, vec!["x", "y"]);
    Ok(())
}

/// A field with interleaved (`FixedSizeList`) coordinates fails to import.
#[test]
fn import_interleaved_field_fails() {
    let linestring_type = LineStringType::new(GeoArrowDimension::XY, Default::default())
        .with_coord_type(CoordType::Interleaved);
    let field = linestring_type.to_field("geom", false);
    assert!(SESSION.arrow().from_arrow_field(&field).is_err());
}

/// A field imported to the LineString dtype and exported back carries the `geoarrow.linestring`
/// extension over its `List` storage.
#[test]
fn export_field_carries_extension() -> VortexResult<()> {
    let imported =
        SESSION
            .arrow()
            .from_arrow_field(&linestring_field("geom", false, Some("EPSG:4326")))?;
    let field = SESSION.arrow().to_arrow_field("geom", &imported)?;

    assert_eq!(field.extension_type_name(), Some(LineStringType::NAME));
    assert!(
        matches!(field.data_type(), DataType::List(_)),
        "expected List storage, got {}",
        field.data_type()
    );
    Ok(())
}

/// A NaN coordinate in a non-Point geometry is invalid rather than an empty-geometry sentinel.
#[test]
fn rejects_non_finite_coordinate() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let field = linestring_field("geom", false, Some("EPSG:4326"));
    let source = linestring_column(vec![vec![(0.0, 0.0), (f64::NAN, 1.0)]])?;
    let arrow = SESSION
        .arrow()
        .execute_arrow(source, Some(&field), &mut ctx)?;

    assert!(SESSION.arrow().from_arrow_array(arrow, &field).is_err());
    Ok(())
}

/// Coordinates outside an Arrow slice are not part of the imported geometry.
#[test]
fn ignores_non_finite_coordinates_outside_slice() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let field = linestring_field("geom", false, Some("EPSG:4326"));
    let source = linestring_column(vec![
        vec![(f64::NAN, 0.0)],
        vec![(1.0, 2.0), (3.0, 4.0)],
        vec![(5.0, f64::NAN)],
    ])?;
    let arrow = SESSION
        .arrow()
        .execute_arrow(source, Some(&field), &mut ctx)?
        .slice(1, 1);

    SESSION.arrow().from_arrow_array(arrow, &field)?;
    Ok(())
}

/// Child values of a null geometry row are unspecified and must not be validated.
#[test]
fn ignores_non_finite_coordinates_under_null_row() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let field = linestring_field("geom", true, Some("EPSG:4326"));
    let source = linestring_column(vec![vec![(f64::NAN, 0.0)], vec![(1.0, 2.0)]])?;
    let arrow = SESSION
        .arrow()
        .execute_arrow(source, Some(&field), &mut ctx)?;
    let lists = arrow
        .as_any()
        .downcast_ref::<ArrowListArray>()
        .ok_or_else(|| vortex_error::vortex_err!("expected Arrow ListArray"))?;
    let DataType::List(element) = lists.data_type() else {
        vortex_error::vortex_bail!("expected Arrow ListArray")
    };
    let nullable = ArrowListArray::try_new(
        Arc::clone(element),
        lists.offsets().clone(),
        Arc::clone(lists.values()),
        Some(NullBuffer::from(vec![false, true])),
    )
    .map_err(|error| vortex_error::vortex_err!("failed to build nullable list: {error}"))?;

    SESSION
        .arrow()
        .from_arrow_array(Arc::new(nullable), &field)?;
    Ok(())
}
