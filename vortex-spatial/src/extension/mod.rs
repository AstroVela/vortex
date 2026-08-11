// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub(crate) mod coordinate;
mod geometry;
mod linestring;
mod literal;
mod metadata;
mod multilinestring;
mod multipoint;
mod multipolygon;
mod point;
mod polygon;
mod rect;
mod wkb;

pub(crate) use geometry::flatten_coordinates;
pub(crate) use geometry::flatten_row_offsets;
pub(crate) use geometry::geometries;
pub(crate) use geometry::is_native_geometry;
pub(crate) use geometry::single_geometry;
pub use linestring::*;
pub use literal::native_geometry_scalar_from_wkb;
pub use metadata::CrsType;
pub use metadata::Edges;
pub use metadata::SpatialMetadata;
pub use multilinestring::*;
pub use multipoint::*;
pub use multipolygon::*;
pub use point::*;
pub use polygon::*;
pub use rect::*;
pub use wkb::*;
