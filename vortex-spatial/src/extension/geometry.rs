// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The [`Geometry`] extension type (`vortex.st.geometry`), represented logically as a union
//! of native geometry extension arrays and mapped to `geoarrow.geometry`. Arrow imports retain
//! the compact layout through the private DenseUnion physical encoding.

use std::sync::Arc;

use arrow_array::Array as _;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::UnionArray as ArrowUnionArray;
use arrow_array::new_empty_array;
use arrow_buffer::ScalarBuffer;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::UnionMode;
use arrow_schema::extension::ExtensionType;
use geo_traits::to_geo::ToGeoGeometry;
use geo_types::Geometry as GeoGeometry;
use geoarrow::array::GeoArrowArrayAccessor;
use geoarrow::array::GeometryArray as GeoArrowGeometryArray;
use geoarrow::datatypes::CoordType;
use geoarrow::datatypes::GeometryType as GeoArrowGeometryType;
use prost::Message;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::UnionArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::union::UnionArrayExt;
use vortex_array::arrays::union::UnionArraySlotsExt;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::UnionVariants;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtId;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_array::validity::Validity;
use vortex_arrow::ArrowExport;
use vortex_arrow::ArrowExportVTable;
use vortex_arrow::ArrowImport;
use vortex_arrow::ArrowImportVTable;
use vortex_arrow::ArrowSession;
use vortex_arrow::ArrowSessionExt;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::registry::CachedId;
use vortex_session::registry::Id;

use super::LineString;
use super::MultiLineString;
use super::MultiPoint;
use super::MultiPolygon;
use super::Point;
use super::Polygon;
use super::SpatialMetadata;
use super::coordinate::Dimension;
use super::coordinate::coordinate_dimension;
use super::geoarrow_metadata;
use super::geoarrow_to_wkb;
use super::linestring_dimension;
use super::multilinestring_dimension;
use super::multipoint_dimension;
use super::multipolygon_dimension;
use super::polygon_dimension;
use super::spatial_metadata_from_arrow;
use crate::dense_union::DenseUnion;
use crate::dense_union::DenseUnionArrayExt;
use crate::dense_union::DenseUnionArraySlotsExt;

/// A mixed native geometry column whose logical union variants are Point, LineString, Polygon,
/// MultiPoint, MultiLineString, and MultiPolygon extension arrays. GeoArrow GeometryCollection
/// fields may be present in the Arrow schema, but selected GeometryCollection values are not yet
/// supported.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Geometry;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GeoArrowGeometryKind {
    Point,
    LineString,
    Polygon,
    MultiPoint,
    MultiLineString,
    MultiPolygon,
    GeometryCollection,
}

fn geoarrow_type_id_parts(type_id: u8) -> VortexResult<(GeoArrowGeometryKind, Dimension)> {
    // GeoArrow assigns the ones digit to geometry kind and the tens digit to coordinate dimension.
    let dimension = match type_id / 10 {
        0 => Dimension::Xy,
        1 => Dimension::Xyz,
        2 => Dimension::Xym,
        3 => Dimension::Xyzm,
        _ => vortex_bail!("unsupported GeoArrow geometry type ID {type_id}"),
    };
    let kind = match type_id % 10 {
        1 => GeoArrowGeometryKind::Point,
        2 => GeoArrowGeometryKind::LineString,
        3 => GeoArrowGeometryKind::Polygon,
        4 => GeoArrowGeometryKind::MultiPoint,
        5 => GeoArrowGeometryKind::MultiLineString,
        6 => GeoArrowGeometryKind::MultiPolygon,
        7 => GeoArrowGeometryKind::GeometryCollection,
        _ => vortex_bail!("unsupported GeoArrow geometry type ID {type_id}"),
    };
    Ok((kind, dimension))
}

fn validate_variant_dtype(type_id: u8, dtype: &DType) -> VortexResult<()> {
    let (kind, expected_dimension) = geoarrow_type_id_parts(type_id)?;
    let ext = dtype
        .as_extension_opt()
        .ok_or_else(|| vortex_err!("GeoArrow geometry variant {type_id} must be an extension"))?;
    let actual_dimension = match kind {
        GeoArrowGeometryKind::Point => {
            vortex_ensure!(
                ext.is::<Point>(),
                "type ID {type_id} must contain Point values"
            );
            coordinate_dimension(ext.storage_dtype())?
        }
        GeoArrowGeometryKind::LineString => {
            vortex_ensure!(
                ext.is::<LineString>(),
                "type ID {type_id} must contain LineString values"
            );
            linestring_dimension(ext.storage_dtype())?
        }
        GeoArrowGeometryKind::Polygon => {
            vortex_ensure!(
                ext.is::<Polygon>(),
                "type ID {type_id} must contain Polygon values"
            );
            polygon_dimension(ext.storage_dtype())?
        }
        GeoArrowGeometryKind::MultiPoint => {
            vortex_ensure!(
                ext.is::<MultiPoint>(),
                "type ID {type_id} must contain MultiPoint values"
            );
            multipoint_dimension(ext.storage_dtype())?
        }
        GeoArrowGeometryKind::MultiLineString => {
            vortex_ensure!(
                ext.is::<MultiLineString>(),
                "type ID {type_id} must contain MultiLineString values"
            );
            multilinestring_dimension(ext.storage_dtype())?
        }
        GeoArrowGeometryKind::MultiPolygon => {
            vortex_ensure!(
                ext.is::<MultiPolygon>(),
                "type ID {type_id} must contain MultiPolygon values"
            );
            multipolygon_dimension(ext.storage_dtype())?
        }
        GeoArrowGeometryKind::GeometryCollection => {
            vortex_bail!("GeoArrow GeometryCollection values are not supported yet")
        }
    };
    vortex_ensure!(
        actual_dimension == expected_dimension,
        "GeoArrow geometry type ID {type_id} requires {expected_dimension:?} storage, got {actual_dimension:?}"
    );
    Ok(())
}

fn native_child_dtype(
    type_id: u8,
    metadata: &SpatialMetadata,
    storage_dtype: DType,
) -> VortexResult<DType> {
    let (kind, _) = geoarrow_type_id_parts(type_id)?;
    let dtype = match kind {
        GeoArrowGeometryKind::Point => {
            DType::Extension(ExtDType::<Point>::try_new(metadata.clone(), storage_dtype)?.erased())
        }
        GeoArrowGeometryKind::LineString => DType::Extension(
            ExtDType::<LineString>::try_new(metadata.clone(), storage_dtype)?.erased(),
        ),
        GeoArrowGeometryKind::Polygon => DType::Extension(
            ExtDType::<Polygon>::try_new(metadata.clone(), storage_dtype)?.erased(),
        ),
        GeoArrowGeometryKind::MultiPoint => DType::Extension(
            ExtDType::<MultiPoint>::try_new(metadata.clone(), storage_dtype)?.erased(),
        ),
        GeoArrowGeometryKind::MultiLineString => DType::Extension(
            ExtDType::<MultiLineString>::try_new(metadata.clone(), storage_dtype)?.erased(),
        ),
        GeoArrowGeometryKind::MultiPolygon => DType::Extension(
            ExtDType::<MultiPolygon>::try_new(metadata.clone(), storage_dtype)?.erased(),
        ),
        GeoArrowGeometryKind::GeometryCollection => {
            vortex_bail!("GeoArrow GeometryCollection values are not supported yet")
        }
    };
    validate_variant_dtype(type_id, &dtype)?;
    Ok(dtype)
}

fn geometry_variants(dtype: &DType) -> VortexResult<(&UnionVariants, Nullability)> {
    let DType::Union(variants, nullability) = dtype else {
        vortex_bail!("Geometry storage must be a union, got {dtype}");
    };
    Ok((variants, *nullability))
}

struct GeometryUnionParts {
    variants: UnionVariants,
    type_ids: PrimitiveArray,
    offsets: PrimitiveArray,
    children: Vec<ArrayRef>,
}

impl GeometryUnionParts {
    fn child(&self, type_id: u8) -> Option<&ArrayRef> {
        self.variants
            .tag_to_child_index(type_id)
            .and_then(|child_index| self.children.get(child_index))
    }
}

fn geometry_union_parts(
    storage: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<GeometryUnionParts> {
    if let Some(dense) = storage.as_opt::<DenseUnion>() {
        return Ok(GeometryUnionParts {
            variants: dense.variants().clone(),
            type_ids: dense.type_ids().clone().execute::<PrimitiveArray>(ctx)?,
            offsets: dense.offsets().clone().execute::<PrimitiveArray>(ctx)?,
            children: dense.iter_children().cloned().collect(),
        });
    }

    // Other physical encodings execute to the canonical sparse union. Its row-aligned children
    // can use row indices as dense-union offsets, so export does not need to compact them.
    let sparse = storage.execute::<UnionArray>(ctx)?;
    let offsets = (0..sparse.len())
        .map(|offset| {
            i32::try_from(offset)
                .map_err(|_| vortex_err!("Geometry row offset {offset} exceeds i32"))
        })
        .collect::<VortexResult<Vec<_>>>()?;
    let type_ids = sparse.type_ids().clone().execute::<PrimitiveArray>(ctx)?;
    let outer_validity = type_ids
        .validity()?
        .execute_mask(type_ids.len(), ctx)?
        .into_array();
    let children = sparse
        .iter_children()
        .cloned()
        .map(|child| child.mask(outer_validity.clone()))
        .collect::<VortexResult<Vec<_>>>()?;
    Ok(GeometryUnionParts {
        variants: sparse.variants().clone(),
        type_ids,
        offsets: PrimitiveArray::from_iter(offsets),
        children,
    })
}

impl ExtVTable for Geometry {
    type Metadata = SpatialMetadata;
    type NativeValue<'a> = Scalar;

    fn id(&self) -> ExtId {
        static ID: CachedId = CachedId::new("vortex.st.geometry");
        *ID
    }

    fn serialize_metadata(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(metadata.encode_to_vec())
    }

    fn deserialize_metadata(&self, metadata: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(SpatialMetadata::decode(metadata)?)
    }

    fn validate_dtype(ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
        let (variants, _) = geometry_variants(ext_dtype.storage_dtype())?;
        for (type_id, dtype) in variants.type_ids().iter().zip(variants.variants()) {
            validate_variant_dtype(*type_id, &dtype)?;
        }
        Ok(())
    }

    fn unpack_native<'a>(
        ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>> {
        Scalar::try_new(
            ext_dtype.storage_dtype().clone(),
            Some(storage_value.clone()),
        )
    }
}

static GEOARROW_GEOMETRY: CachedId = CachedId::new(GeoArrowGeometryType::NAME);

fn geoarrow_geometry_type(metadata: &SpatialMetadata) -> GeoArrowGeometryType {
    GeoArrowGeometryType::new(geoarrow_metadata(metadata)).with_coord_type(CoordType::Separated)
}

/// A materialized mixed geometry extension array.
pub struct GeometryData(ExtensionArray);

impl TryFrom<ExtensionArray> for GeometryData {
    type Error = VortexError;

    fn try_from(ext: ExtensionArray) -> Result<Self, Self::Error> {
        vortex_ensure!(
            ext.ext_dtype().is::<Geometry>(),
            "expected a Geometry extension array"
        );
        Ok(Self(ext))
    }
}

impl GeometryData {
    /// Serialize mixed geometries to WKB.
    pub fn to_wkb(&self, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        geoarrow_to_wkb(
            &geoarrow_geometry_array(&self.0.clone().into_array(), ctx)?,
            &ctx.session().arrow(),
        )
    }
}

fn geoarrow_geometry_array(
    array: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<GeoArrowGeometryArray> {
    let session = ctx.session().clone();
    let field = session.arrow().to_arrow_field("", array.dtype())?;
    let geometry_type = field
        .try_extension_type::<GeoArrowGeometryType>()
        .map_err(|e| vortex_err!("failed to construct GeoArrow GeometryType: {e}"))?;
    let arrow = session
        .arrow()
        .execute_arrow(array.clone(), Some(&field), ctx)?;
    GeoArrowGeometryArray::try_from((arrow.as_ref(), geometry_type))
        .map_err(|e| vortex_err!("failed to construct GeoArrow GeometryArray: {e}"))
}

pub(crate) fn decode_mixed_geometries(
    array: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Vec<GeoGeometry<f64>>> {
    geoarrow_geometry_array(array, ctx)?
        .iter()
        .map(|geometry| -> VortexResult<GeoGeometry<f64>> {
            Ok(geometry
                .ok_or_else(|| vortex_err!("spatial: null geometry is not supported"))?
                .map_err(|e| vortex_err!("spatial: geometry access failed: {e}"))?
                .to_geometry())
        })
        .collect()
}

impl ArrowExportVTable for Geometry {
    fn arrow_ext_id(&self) -> Id {
        *GEOARROW_GEOMETRY
    }

    fn vortex_id(&self) -> Id {
        self.id()
    }

    fn to_arrow_field(
        &self,
        name: &str,
        dtype: &DType,
        _session: &ArrowSession,
    ) -> VortexResult<Option<Field>> {
        let ext_dtype = dtype.as_extension();
        let metadata = ext_dtype.metadata::<Geometry>();
        let (_, nullability) = geometry_variants(ext_dtype.storage_dtype())?;
        Ok(Some(
            geoarrow_geometry_type(metadata).to_field(name, nullability.is_nullable()),
        ))
    }

    fn execute_arrow(
        &self,
        array: ArrayRef,
        target: &Field,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowExport> {
        let Some(ext_dtype) = array.dtype().as_extension_opt() else {
            return Ok(ArrowExport::Unsupported(array));
        };
        if !ext_dtype.is::<Geometry>() {
            return Ok(ArrowExport::Unsupported(array));
        }
        let Ok(target_type) = target.try_extension_type::<GeoArrowGeometryType>() else {
            return Ok(ArrowExport::Unsupported(array));
        };
        if target_type.coord_type() != CoordType::Separated {
            return Ok(ArrowExport::Unsupported(array));
        }
        let DataType::Union(target_fields, UnionMode::Dense) = target.data_type() else {
            return Ok(ArrowExport::Unsupported(array));
        };

        let extension = array.execute::<ExtensionArray>(ctx)?;
        let parts = geometry_union_parts(extension.storage_array().clone(), ctx)?;
        let mut known_type_ids = [false; 256];
        for source_id in parts.variants.type_ids() {
            known_type_ids[usize::from(*source_id)] = true;
            let source_id = i8::try_from(*source_id)
                .map_err(|_| vortex_err!("GeoArrow geometry type ID {source_id} exceeds i8"))?;
            vortex_ensure!(
                target_fields
                    .iter()
                    .any(|(target_id, _)| target_id == source_id),
                "target GeoArrow geometry union is missing type ID {source_id}"
            );
        }

        let validity = parts
            .type_ids
            .validity()?
            .execute_mask(parts.type_ids.len(), ctx)?;
        let fallback_type_id = parts
            .variants
            .type_ids()
            .first()
            .copied()
            .ok_or_else(|| vortex_err!("Geometry union has no variants"))?;
        let arrow_type_ids = parts
            .type_ids
            .as_slice::<u8>()
            .iter()
            .zip(validity.iter())
            .map(|(type_id, valid)| {
                let type_id = match (known_type_ids[usize::from(*type_id)], valid) {
                    (true, _) => *type_id,
                    (false, false) => fallback_type_id,
                    (false, true) => {
                        vortex_bail!("Geometry row has unknown type ID {type_id}")
                    }
                };
                i8::try_from(type_id)
                    .map_err(|_| vortex_err!("GeoArrow geometry type ID {type_id} exceeds i8"))
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let arrow_offsets = parts.offsets.as_slice::<i32>().to_vec();

        let session = ctx.session().clone();
        let mut arrow_children = Vec::with_capacity(target_fields.len());
        for (type_id, child_field) in target_fields.iter() {
            let source_id = u8::try_from(type_id)
                .map_err(|_| vortex_err!("GeoArrow geometry type ID {type_id} is negative"))?;
            let Some(child) = parts.child(source_id) else {
                arrow_children.push(new_empty_array(child_field.data_type()));
                continue;
            };
            let child = child.clone().execute::<ExtensionArray>(ctx)?;
            arrow_children.push(session.arrow().execute_arrow(
                child.storage_array().clone(),
                Some(child_field.as_ref()),
                ctx,
            )?);
        }

        let union = ArrowUnionArray::try_new(
            target_fields.clone(),
            ScalarBuffer::from(arrow_type_ids),
            Some(ScalarBuffer::from(arrow_offsets)),
            arrow_children,
        )
        .map_err(|e| vortex_err!("failed to construct Arrow dense union: {e}"))?;

        if !target.is_nullable() {
            vortex_ensure!(
                validity.all_true(),
                "cannot export nullable Geometry values to a non-nullable Arrow field"
            );
        }
        for (row, _) in validity.iter().enumerate().filter(|(_, valid)| !valid) {
            let type_id = union.type_id(row);
            let offset = union.value_offset(row);
            vortex_ensure!(
                union.child(type_id).is_null(offset),
                "GeoArrow Geometry null at row {row} is not represented by a null selected child"
            );
        }

        Ok(ArrowExport::Exported(Arc::new(union)))
    }
}

impl ArrowImportVTable for Geometry {
    fn arrow_ext_id(&self) -> Id {
        *GEOARROW_GEOMETRY
    }

    fn from_arrow_field(
        &self,
        field: &Field,
        session: &ArrowSession,
    ) -> VortexResult<Option<DType>> {
        let Ok(geometry_type) = field.try_extension_type::<GeoArrowGeometryType>() else {
            return Ok(None);
        };
        vortex_ensure!(
            geometry_type.coord_type() == CoordType::Separated,
            "geoarrow.geometry with interleaved coordinates is not supported; re-encode with separated coordinates"
        );
        let DataType::Union(fields, UnionMode::Dense) = field.data_type() else {
            vortex_bail!("geoarrow.geometry requires dense union storage");
        };
        let metadata = spatial_metadata_from_arrow(geometry_type.metadata());
        let mut names = Vec::new();
        let mut dtypes = Vec::new();
        let mut type_ids = Vec::new();
        for (type_id, child_field) in fields.iter() {
            let type_id = u8::try_from(type_id)
                .map_err(|_| vortex_err!("GeoArrow geometry type ID {type_id} is negative"))?;
            let (kind, _) = geoarrow_type_id_parts(type_id)?;
            if kind == GeoArrowGeometryKind::GeometryCollection {
                continue;
            }
            let storage_dtype = session
                .from_arrow_datatype(child_field.data_type(), child_field.is_nullable().into())?;
            names.push(child_field.name().as_str());
            dtypes.push(native_child_dtype(type_id, &metadata, storage_dtype)?);
            type_ids.push(type_id);
        }
        let variants = UnionVariants::try_new(FieldNames::from(names), dtypes, type_ids)?;
        let storage_dtype = DType::Union(variants, field.is_nullable().into());
        Ok(Some(DType::Extension(
            ExtDType::<Geometry>::try_new(metadata, storage_dtype)?.erased(),
        )))
    }

    fn from_arrow_array(
        &self,
        array: ArrowArrayRef,
        field: &Field,
        dtype: &DType,
        session: &ArrowSession,
    ) -> VortexResult<ArrowImport> {
        let Some(ext_dtype) = dtype.as_extension_opt() else {
            return Ok(ArrowImport::Unsupported(array));
        };
        if !ext_dtype.is::<Geometry>() {
            return Ok(ArrowImport::Unsupported(array));
        }
        let Some(union) = array.as_any().downcast_ref::<ArrowUnionArray>() else {
            return Ok(ArrowImport::Unsupported(array));
        };
        let DataType::Union(fields, UnionMode::Dense) = union.data_type() else {
            return Ok(ArrowImport::Unsupported(array));
        };
        let offsets = union
            .offsets()
            .ok_or_else(|| vortex_err!("geoarrow.geometry requires dense union offsets"))?;
        let type_ids = union
            .type_ids()
            .iter()
            .map(|type_id| {
                let type_id = u8::try_from(*type_id)
                    .map_err(|_| vortex_err!("GeoArrow geometry type ID {type_id} is negative"))?;
                let (kind, _) = geoarrow_type_id_parts(type_id)?;
                vortex_ensure!(
                    kind != GeoArrowGeometryKind::GeometryCollection,
                    "GeoArrow GeometryCollection values are not supported yet"
                );
                Ok(type_id)
            })
            .collect::<VortexResult<Vec<_>>>()?;

        let (variants, nullability) = geometry_variants(ext_dtype.storage_dtype())?;
        let mut children = Vec::with_capacity(variants.len());
        for (type_id, child_dtype) in variants.type_ids().iter().zip(variants.variants()) {
            let arrow_id = i8::try_from(*type_id)
                .map_err(|_| vortex_err!("GeoArrow geometry type ID {type_id} exceeds i8"))?;
            let child_field = fields
                .iter()
                .find_map(|(candidate, field)| (candidate == arrow_id).then_some(field))
                .ok_or_else(|| vortex_err!("missing GeoArrow geometry child {type_id}"))?;
            let storage = session
                .from_arrow_array(Arc::clone(union.child(arrow_id)), child_field.is_nullable())?;
            let child_ext = child_dtype.as_extension();
            children.push(ExtensionArray::try_new(child_ext.clone(), storage)?.into_array());
        }

        let row_validity = union
            .type_ids()
            .iter()
            .zip(offsets.iter())
            .enumerate()
            .map(|(row, (type_id, offset))| {
                let offset = usize::try_from(*offset)
                    .map_err(|_| vortex_err!("negative GeoArrow union offset at row {row}"))?;
                let child = union.child(*type_id);
                vortex_ensure!(
                    offset < child.len(),
                    "GeoArrow union offset {offset} is out of bounds at row {row}"
                );
                Ok(child.is_valid(offset))
            })
            .collect::<VortexResult<Vec<_>>>()?;
        if !field.is_nullable() {
            vortex_ensure!(
                row_validity.iter().all(|valid| *valid),
                "non-nullable geoarrow.geometry field contains null values"
            );
        }
        let validity = match nullability {
            Nullability::NonNullable => Validity::NonNullable,
            Nullability::Nullable => row_validity.into_iter().collect(),
        };
        let type_ids = PrimitiveArray::new(type_ids, validity).into_array();
        let offsets = PrimitiveArray::from_iter(offsets.iter().copied()).into_array();
        let dense =
            DenseUnion::try_new(type_ids, offsets, variants.clone(), children)?.into_array();
        Ok(ArrowImport::Imported(
            ExtensionArray::try_new(ext_dtype.clone(), dense)?.into_array(),
        ))
    }
}
