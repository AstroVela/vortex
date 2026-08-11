// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Spatial metadata and GeoArrow conversion.

use std::fmt::Display;
use std::sync::Arc;

use geoarrow::datatypes::Crs;
use geoarrow::datatypes::CrsType as ArrowCrsType;
use geoarrow::datatypes::Edges as ArrowEdges;
use geoarrow::datatypes::Metadata;
use prost::Message;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

/// CRS serialization format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, prost::Enumeration)]
#[repr(i32)]
pub enum CrsType {
    /// A PROJJSON object.
    Projjson = 0,
    /// A WKT2:2019 string.
    Wkt2_2019 = 1,
    /// An `AUTHORITY:CODE` identifier.
    AuthorityCode = 2,
    /// An opaque, producer-specific identifier.
    Srid = 3,
}

impl From<CrsType> for ArrowCrsType {
    fn from(value: CrsType) -> Self {
        match value {
            CrsType::Projjson => Self::Projjson,
            CrsType::Wkt2_2019 => Self::Wkt2_2019,
            CrsType::AuthorityCode => Self::AuthorityCode,
            CrsType::Srid => Self::Srid,
        }
    }
}

impl From<ArrowCrsType> for CrsType {
    fn from(value: ArrowCrsType) -> Self {
        match value {
            ArrowCrsType::Projjson => Self::Projjson,
            ArrowCrsType::Wkt2_2019 => Self::Wkt2_2019,
            ArrowCrsType::AuthorityCode => Self::AuthorityCode,
            ArrowCrsType::Srid => Self::Srid,
        }
    }
}

/// Edge interpretation. An omitted value means planar edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, prost::Enumeration)]
#[repr(i32)]
pub enum Edges {
    /// Andoyer geodesics on the CRS ellipsoid.
    Andoyer = 0,
    /// Karney geodesics on the CRS ellipsoid.
    Karney = 1,
    /// Great-circle paths.
    Spherical = 2,
    /// Thomas geodesics on the CRS ellipsoid.
    Thomas = 3,
    /// Vincenty geodesics on the CRS ellipsoid.
    Vincenty = 4,
}

impl From<Edges> for ArrowEdges {
    fn from(value: Edges) -> Self {
        match value {
            Edges::Andoyer => Self::Andoyer,
            Edges::Karney => Self::Karney,
            Edges::Spherical => Self::Spherical,
            Edges::Thomas => Self::Thomas,
            Edges::Vincenty => Self::Vincenty,
        }
    }
}

impl From<ArrowEdges> for Edges {
    fn from(value: ArrowEdges) -> Self {
        match value {
            ArrowEdges::Andoyer => Self::Andoyer,
            ArrowEdges::Karney => Self::Karney,
            ArrowEdges::Spherical => Self::Spherical,
            ArrowEdges::Thomas => Self::Thomas,
            ArrowEdges::Vincenty => Self::Vincenty,
        }
    }
}

/// Metadata shared by all spatial extension types.
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct SpatialMetadata {
    /// Coordinate reference system, if known.
    pub crs: Option<String>,
    /// Serialization format of [`Self::crs`].
    pub crs_type: Option<CrsType>,
    /// Edge interpretation. `None` means planar edges.
    pub edges: Option<Edges>,
}

/// Protobuf representation of [`SpatialMetadata`].
#[derive(Clone, PartialEq, Message)]
struct SpatialMetadataProto {
    #[prost(optional, string, tag = "1")]
    crs: Option<String>,
    #[prost(enumeration = "CrsType", optional, tag = "2")]
    crs_type: Option<i32>,
    #[prost(enumeration = "Edges", optional, tag = "3")]
    edges: Option<i32>,
}

impl From<&SpatialMetadata> for SpatialMetadataProto {
    fn from(metadata: &SpatialMetadata) -> Self {
        Self {
            crs: metadata.crs.clone(),
            crs_type: metadata.crs_type.map(Into::into),
            edges: metadata.edges.map(Into::into),
        }
    }
}

impl TryFrom<SpatialMetadataProto> for SpatialMetadata {
    type Error = VortexError;

    fn try_from(proto: SpatialMetadataProto) -> VortexResult<Self> {
        let crs_type = proto
            .crs_type
            .map(CrsType::try_from)
            .transpose()
            .map_err(|error| vortex_err!("spatial: invalid CRS type: {error}"))?;
        let edges = proto
            .edges
            .map(Edges::try_from)
            .transpose()
            .map_err(|error| vortex_err!("spatial: invalid edges value: {error}"))?;
        Ok(Self {
            crs: proto.crs,
            crs_type,
            edges,
        })
    }
}

impl SpatialMetadata {
    pub(super) fn serialize(&self) -> Vec<u8> {
        SpatialMetadataProto::from(self).encode_to_vec()
    }

    pub(super) fn deserialize(bytes: &[u8]) -> VortexResult<Self> {
        SpatialMetadataProto::decode(bytes)?.try_into()
    }
}

impl Display for SpatialMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.crs.as_ref() {
            Some(crs) => write!(f, "Geometry(crs={crs})"),
            None => write!(f, "Geometry(unreferenced)"),
        }
    }
}

/// Convert spatial metadata to GeoArrow.
pub(crate) fn to_geoarrow(metadata: &SpatialMetadata) -> VortexResult<Arc<Metadata>> {
    let crs = match (metadata.crs.as_deref(), metadata.crs_type) {
        (None, None) => Crs::default(),
        (None, Some(crs_type)) => {
            vortex_bail!("spatial: CRS type {crs_type:?} requires a CRS value")
        }
        (Some(crs), None) => Crs::from_unknown_crs_type(crs.to_owned()),
        (Some(crs), Some(CrsType::Projjson)) => {
            let value: serde_json::Value = serde_json::from_str(crs)
                .map_err(|error| vortex_err!("spatial: invalid PROJJSON CRS: {error}"))?;
            vortex_ensure!(
                value.is_object(),
                "spatial: PROJJSON CRS must be a JSON object"
            );
            Crs::from_projjson(value)
        }
        (Some(crs), Some(CrsType::Wkt2_2019)) => Crs::from_wkt2_2019(crs.to_owned()),
        (Some(crs), Some(CrsType::AuthorityCode)) => {
            vortex_ensure!(
                crs.contains(':'),
                "spatial: authority-code CRS must have the form AUTHORITY:CODE"
            );
            Crs::from_authority_code(crs.to_owned())
        }
        (Some(crs), Some(CrsType::Srid)) => Crs::from_srid(crs.to_owned()),
    };
    Ok(Arc::new(Metadata::new(crs, metadata.edges.map(Into::into))))
}

/// Convert GeoArrow metadata to [`SpatialMetadata`].
pub(crate) fn from_geoarrow(metadata: &Metadata) -> SpatialMetadata {
    let arrow_crs = metadata.crs();
    let value = arrow_crs.crs_value();
    let crs = value.map(|value| {
        value
            .as_str()
            .map_or_else(|| value.to_string(), str::to_owned)
    });
    // GeoArrow defines an object-valued CRS as PROJJSON, even when `crs_type` is omitted.
    let crs_type = arrow_crs.crs_type().map(Into::into).or_else(|| {
        value
            .filter(|value| value.is_object())
            .map(|_| CrsType::Projjson)
    });

    SpatialMetadata {
        crs,
        crs_type,
        edges: metadata.edges().map(Into::into),
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_error::VortexResult;

    use super::CrsType;
    use super::Edges;
    use super::SpatialMetadata;
    use super::from_geoarrow;
    use super::to_geoarrow;

    #[test]
    fn metadata_roundtrips_serialization() -> VortexResult<()> {
        let metadata = SpatialMetadata {
            crs: Some("EPSG:4326".to_string()),
            crs_type: Some(CrsType::AuthorityCode),
            edges: Some(Edges::Spherical),
        };

        assert_eq!(metadata.to_string(), "Geometry(crs=EPSG:4326)");
        let decoded = SpatialMetadata::deserialize(&metadata.serialize())?;
        assert_eq!(decoded, metadata);
        Ok(())
    }

    #[test]
    fn decodes_legacy_crs_only_bytes() -> VortexResult<()> {
        let legacy = b"\x0a\x09EPSG:4326";
        assert_eq!(
            SpatialMetadata::deserialize(legacy)?,
            SpatialMetadata {
                crs: Some("EPSG:4326".to_string()),
                crs_type: None,
                edges: None,
            }
        );
        Ok(())
    }

    #[rstest]
    #[case::projjson(CrsType::Projjson, r#"{"type":"GeographicCRS"}"#)]
    #[case::wkt(CrsType::Wkt2_2019, "GEOGCRS[\"WGS 84\"]")]
    #[case::authority_code(CrsType::AuthorityCode, "EPSG:4326")]
    #[case::srid(CrsType::Srid, "database-crs-42")]
    fn geoarrow_crs_type_roundtrips(
        #[case] crs_type: CrsType,
        #[case] crs: &str,
    ) -> VortexResult<()> {
        let metadata = SpatialMetadata {
            crs: Some(crs.to_string()),
            crs_type: Some(crs_type),
            edges: None,
        };
        let geoarrow = to_geoarrow(&metadata)?;
        assert_eq!(from_geoarrow(&geoarrow), metadata);
        Ok(())
    }

    #[rstest]
    #[case::andoyer(Edges::Andoyer)]
    #[case::karney(Edges::Karney)]
    #[case::spherical(Edges::Spherical)]
    #[case::thomas(Edges::Thomas)]
    #[case::vincenty(Edges::Vincenty)]
    fn geoarrow_edges_roundtrip(#[case] edges: Edges) -> VortexResult<()> {
        let metadata = SpatialMetadata {
            crs: None,
            crs_type: None,
            edges: Some(edges),
        };
        let geoarrow = to_geoarrow(&metadata)?;
        assert_eq!(from_geoarrow(&geoarrow), metadata);
        Ok(())
    }
}
