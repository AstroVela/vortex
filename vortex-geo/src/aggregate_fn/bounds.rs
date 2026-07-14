// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A zone-map statistic: the 2D bounding box of a native geometry column.

use geo::Rect as GeoRect;
use vortex_array::ArrayRef;
use vortex_array::Columnar;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::AggregateFnId;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::scalar::Scalar;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::extension::GeoMetadata;
use crate::extension::Rect;
use crate::extension::box_storage_dtype;
use crate::extension::coordinate::Dimension;
use crate::extension::flatten_coordinates;
use crate::extension::is_native_geometry;

/// Aggregates a native geometry column's 2D minimum bounding box as the native `geoarrow.box` type.
/// Stored as a zone statistic, it lets spatial filters skip chunks whose bounding box cannot hold a
/// matching row.
#[derive(Clone, Debug)]
pub struct GeometryBounds;

/// Running union of geometry bounding boxes, or `None` until the first row. A transient
/// `geo::Rect` value - the persisted stat is the native box (see `to_scalar`).
pub struct BoundsPartial {
    rect: Option<GeoRect<f64>>,
}

impl BoundsPartial {
    /// Grow the accumulated box to also cover `other`.
    fn merge(&mut self, other: GeoRect<f64>) {
        self.rect = Some(match self.rect {
            Some(cur) => GeoRect::new(
                (
                    cur.min().x.min(other.min().x),
                    cur.min().y.min(other.min().y),
                ),
                (
                    cur.max().x.max(other.max().x),
                    cur.max().y.max(other.max().y),
                ),
            ),
            None => other,
        });
    }
}

/// The stat's type: the native `geoarrow.box` (2D), nullable so an empty group is a null box.
fn bounds_dtype() -> DType {
    DType::Extension(
        ExtDType::<Rect>::try_new(GeoMetadata::default(), bounds_storage_dtype())
            .vortex_expect("2D box storage is a valid Rect")
            .erased(),
    )
}

/// The `Rect` storage `Struct<xmin, ymin, xmax, ymax>` backing the zone statistic.
fn bounds_storage_dtype() -> DType {
    box_storage_dtype(Dimension::Xy, Nullability::Nullable)
}

/// The bounding box of the raw `x`/`y` slices, or `None` when empty.
fn bounds_of(xs: &[f64], ys: &[f64]) -> Option<GeoRect<f64>> {
    if xs.is_empty() {
        return None;
    }
    let min_max = |vals: &[f64]| {
        vals.iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                (lo.min(v), hi.max(v))
            })
    };
    let (xmin, xmax) = min_max(xs);
    let (ymin, ymax) = min_max(ys);
    Some(GeoRect::new((xmin, ymin), (xmax, ymax)))
}

/// Read a bounds stat scalar (a nullable native `geoarrow.box`) into a [`GeoRect`], or `None` when
/// the scalar is null (an empty group).
fn rect_from_storage(scalar: &Scalar) -> VortexResult<Option<GeoRect<f64>>> {
    if scalar.is_null() {
        return Ok(None);
    }
    let storage = scalar.as_extension().to_storage_scalar();
    let fields = storage.as_struct();
    let read = |name: &str| -> VortexResult<f64> {
        f64::try_from(
            &fields
                .field(name)
                .ok_or_else(|| vortex_err!("bounds missing {name}"))?,
        )
    };
    Ok(Some(GeoRect::new(
        (read("xmin")?, read("ymin")?),
        (read("xmax")?, read("ymax")?),
    )))
}

/// Serialize a [`GeoRect`] as a native `geoarrow.box` stat scalar (inverse of [`rect_from_storage`]).
fn rect_to_storage(rect: GeoRect<f64>) -> Scalar {
    let storage = Scalar::struct_(
        bounds_storage_dtype(),
        vec![
            Scalar::primitive(rect.min().x, Nullability::NonNullable),
            Scalar::primitive(rect.min().y, Nullability::NonNullable),
            Scalar::primitive(rect.max().x, Nullability::NonNullable),
            Scalar::primitive(rect.max().y, Nullability::NonNullable),
        ],
    );
    Scalar::extension::<Rect>(GeoMetadata::default(), storage)
}

impl AggregateFnVTable for GeometryBounds {
    type Options = EmptyOptions;
    type Partial = BoundsPartial;

    fn id(&self) -> AggregateFnId {
        static ID: CachedId = CachedId::new("vortex.geo.bounds");
        *ID
    }

    // Serializable so the zoned writer can persist this as a per-chunk stat. No options to encode.
    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        Ok(EmptyOptions)
    }

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        is_native_geometry(input_dtype).then(bounds_dtype)
    }

    fn zone_stat_default(&self, input_dtype: &DType) -> Option<AggregateFnRef> {
        is_native_geometry(input_dtype).then(|| self.bind(EmptyOptions))
    }

    fn partial_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        self.return_dtype(options, input_dtype)
    }

    fn empty_partial(
        &self,
        _options: &Self::Options,
        _input_dtype: &DType,
    ) -> VortexResult<Self::Partial> {
        Ok(BoundsPartial { rect: None })
    }

    fn combine_partials(&self, partial: &mut Self::Partial, other: Scalar) -> VortexResult<()> {
        if let Some(rect) = rect_from_storage(&other)? {
            partial.merge(rect);
        }
        Ok(())
    }

    fn to_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        Ok(match partial.rect {
            Some(rect) => rect_to_storage(rect),
            None => Scalar::null(bounds_dtype()),
        })
    }

    fn reset(&self, partial: &mut Self::Partial) {
        partial.rect = None;
    }

    fn is_saturated(&self, _partial: &Self::Partial) -> bool {
        // A bounding box can always grow, so it is never saturated.
        false
    }

    fn accumulate(
        &self,
        partial: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let array = match batch {
            Columnar::Canonical(canonical) => canonical.clone().into_array(),
            Columnar::Constant(constant) => constant.clone().into_array(),
        };
        // Min/max the raw x/y buffers directly - cheap, and avoids `to_geometry`'s panic on empty
        // points (which decoding each geometry would hit).
        let coords = flatten_coordinates(&array, ctx)?;
        let xs = coords
            .unmasked_field_by_name("x")?
            .clone()
            .execute::<PrimitiveArray>(ctx)?;
        let ys = coords
            .unmasked_field_by_name("y")?
            .clone()
            .execute::<PrimitiveArray>(ctx)?;
        if let Some(rect) = bounds_of(xs.as_slice::<f64>(), ys.as_slice::<f64>()) {
            partial.merge(rect);
        }
        Ok(())
    }

    fn finalize(&self, partials: ArrayRef) -> VortexResult<ArrayRef> {
        // The stored partial is already the MBR struct, so finalizing is the identity.
        Ok(partials)
    }

    fn finalize_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        self.to_scalar(partial)
    }
}

#[cfg(test)]
mod tests {
    use geo::Rect as GeoRect;
    use vortex_array::ArrayRef;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::Accumulator;
    use vortex_array::aggregate_fn::AggregateFnVTable;
    use vortex_array::aggregate_fn::DynAccumulator;
    use vortex_array::aggregate_fn::EmptyOptions;
    use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::scalar::Scalar;
    use vortex_error::VortexResult;

    use super::BoundsPartial;
    use super::GeometryBounds;
    use super::bounds_dtype;
    use super::rect_from_storage;
    use crate::test_harness::linestring_column;
    use crate::test_harness::multilinestring_column;
    use crate::test_harness::multipoint_column;
    use crate::test_harness::multipolygon_column;
    use crate::test_harness::point_column;
    use crate::test_harness::polygon_column;

    /// One column of every native geometry type over the same `(x, y)` vertex set.
    fn every_native_column(vertices: &[(f64, f64)]) -> VortexResult<Vec<ArrayRef>> {
        let (xs, ys): (Vec<f64>, Vec<f64>) = vertices.iter().copied().unzip();
        let flat = vertices.to_vec();
        Ok(vec![
            point_column(xs, ys)?,
            linestring_column(vec![flat.clone()])?,
            multipoint_column(vec![flat.clone()])?,
            polygon_column(vec![vec![flat.clone()]])?,
            multilinestring_column(vec![vec![flat.clone()]])?,
            multipolygon_column(vec![vec![vec![flat]]])?,
        ])
    }

    /// The aggregate must be serializable so the zoned writer can persist its zone-stat descriptor.
    #[test]
    fn serializes_for_zone_storage() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let metadata = GeometryBounds
            .serialize(&EmptyOptions)?
            .expect("GeometryBounds must be serializable to be stored as a zone statistic");
        GeometryBounds.deserialize(&metadata, &session)?;
        Ok(())
    }

    /// The MBR result's corners as `(xmin, ymin, xmax, ymax)`.
    fn mbr(result: &Scalar) -> VortexResult<(f64, f64, f64, f64)> {
        let rect = rect_from_storage(result)?.expect("non-null bounds");
        Ok((rect.min().x, rect.min().y, rect.max().x, rect.max().y))
    }

    /// The MBR of a Point column is the min/max of its coordinates, accumulated across batches.
    #[test]
    fn point_bounds_across_batches() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let dtype = point_column(vec![0.0], vec![0.0])?.dtype().clone();
        let mut acc = Accumulator::try_new(GeometryBounds, EmptyOptions, dtype)?;

        acc.accumulate(&point_column(vec![1.0, 3.0], vec![2.0, 4.0])?, &mut ctx)?;
        acc.accumulate(&point_column(vec![-1.0], vec![5.0])?, &mut ctx)?;

        assert_eq!(mbr(&acc.finish()?)?, (-1.0, 2.0, 3.0, 5.0));
        Ok(())
    }

    /// The MBR of a Polygon column unions every ring vertex - exercising the `List<List<Struct>>`
    /// unwrap, not just the bare Point struct.
    #[test]
    fn polygon_bounds_union_all_vertices() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        // Two rectangles: (0,0)-(2,3) and (5,5)-(7,8). The chunk MBR is their union: (0,0)-(7,8).
        let polygons = polygon_column(vec![
            vec![vec![(0.0, 0.0), (2.0, 0.0), (2.0, 3.0), (0.0, 3.0)]],
            vec![vec![(5.0, 5.0), (7.0, 5.0), (7.0, 8.0), (5.0, 8.0)]],
        ])?;
        let dtype = polygons.dtype().clone();
        let mut acc = Accumulator::try_new(GeometryBounds, EmptyOptions, dtype)?;
        acc.accumulate(&polygons, &mut ctx)?;

        assert_eq!(mbr(&acc.finish()?)?, (0.0, 0.0, 7.0, 8.0));
        Ok(())
    }

    /// Every native geometry type over the same vertex set yields the same MBR - the zone stat
    /// covers the whole type family.
    #[test]
    fn bounds_cover_every_native_geometry_type() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        for column in every_native_column(&[(1.0, 2.0), (-1.0, 5.0), (3.0, 4.0)])? {
            let mut acc =
                Accumulator::try_new(GeometryBounds, EmptyOptions, column.dtype().clone())?;
            acc.accumulate(&column, &mut ctx)?;
            assert_eq!(
                mbr(&acc.finish()?)?,
                (-1.0, 2.0, 3.0, 5.0),
                "MBR mismatch for {}",
                column.dtype()
            );
        }
        Ok(())
    }

    /// The MBR of a MultiPolygon column unions every vertex of every polygon's rings - exercising
    /// the triple-`List` unwrap.
    #[test]
    fn multipolygon_bounds_union_all_vertices() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        // Multipolygon 0: squares (0,0)-(1,1) and (4,4)-(5,5); multipolygon 1: square (-3,7)-(-2,9).
        let multipolygons = multipolygon_column(vec![
            vec![
                vec![vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]],
                vec![vec![(4.0, 4.0), (5.0, 4.0), (5.0, 5.0), (4.0, 5.0)]],
            ],
            vec![vec![vec![
                (-3.0, 7.0),
                (-2.0, 7.0),
                (-2.0, 9.0),
                (-3.0, 9.0),
            ]]],
        ])?;
        let dtype = multipolygons.dtype().clone();
        let mut acc = Accumulator::try_new(GeometryBounds, EmptyOptions, dtype)?;
        acc.accumulate(&multipolygons, &mut ctx)?;

        assert_eq!(mbr(&acc.finish()?)?, (-3.0, 0.0, 5.0, 9.0));
        Ok(())
    }

    /// `combine_partials` unions partial boxes - the path the zoned writer takes when a zone's
    /// array is chunked.
    #[test]
    fn combine_partials_unions_boxes() -> VortexResult<()> {
        let bbox = |xmin, ymin, xmax, ymax| BoundsPartial {
            rect: Some(GeoRect::new((xmin, ymin), (xmax, ymax))),
        };
        let mut partial = BoundsPartial { rect: None };
        GeometryBounds.combine_partials(
            &mut partial,
            GeometryBounds.to_scalar(&bbox(0.0, 0.0, 1.0, 1.0))?,
        )?;
        GeometryBounds.combine_partials(
            &mut partial,
            GeometryBounds.to_scalar(&bbox(5.0, -2.0, 7.0, 3.0))?,
        )?;
        assert_eq!(
            mbr(&GeometryBounds.to_scalar(&partial)?)?,
            (0.0, -2.0, 7.0, 3.0)
        );
        Ok(())
    }

    /// A null partial (an empty group's MBR) is a no-op in `combine_partials`.
    #[test]
    fn combine_partials_ignores_null() -> VortexResult<()> {
        let mut partial = BoundsPartial {
            rect: Some(GeoRect::new((0.0, 0.0), (1.0, 1.0))),
        };
        GeometryBounds.combine_partials(&mut partial, Scalar::null(bounds_dtype()))?;
        assert_eq!(
            mbr(&GeometryBounds.to_scalar(&partial)?)?,
            (0.0, 0.0, 1.0, 1.0)
        );
        Ok(())
    }

    /// All-NaN coordinates: `f64::min`/`max` skip the NaNs and `geo::Rect` normalizes the result to
    /// a valid (whole-plane) box, so such a chunk is always kept. Sound - NaN-coordinate rows can
    /// never satisfy `distance <= r` anyway.
    #[test]
    fn all_nan_coordinates_kept() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let column = point_column(vec![f64::NAN, f64::NAN], vec![f64::NAN, f64::NAN])?;
        let mut acc = Accumulator::try_new(GeometryBounds, EmptyOptions, column.dtype().clone())?;
        acc.accumulate(&column, &mut ctx)?;

        let (xmin, ymin, xmax, ymax) = mbr(&acc.finish()?)?;
        assert!(xmin <= xmax && ymin <= ymax);
        Ok(())
    }

    /// An empty group yields a null MBR.
    #[test]
    fn empty_group_is_null() -> VortexResult<()> {
        let dtype = point_column(vec![0.0], vec![0.0])?.dtype().clone();
        let mut acc = Accumulator::try_new(GeometryBounds, EmptyOptions, dtype)?;
        assert!(acc.finish()?.is_null());
        Ok(())
    }

    /// After `initialize`, the registry yields a default zone statistic for every native geometry
    /// type (so the zoned writer stores it) but none for ordinary numeric columns.
    #[test]
    fn registered_as_geometry_zone_default() -> VortexResult<()> {
        let session = vortex_array::array_session();
        crate::initialize(&session);

        for column in every_native_column(&[(0.0, 0.0), (1.0, 1.0)])? {
            assert!(
                !session
                    .aggregate_fns()
                    .zone_stat_defaults(column.dtype())
                    .is_empty(),
                "a geometry zone-stat default should be discovered for {}",
                column.dtype()
            );
        }
        let i32_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        assert!(
            session
                .aggregate_fns()
                .zone_stat_defaults(&i32_dtype)
                .is_empty(),
            "no geometry zone-stat default should apply to numeric columns"
        );
        Ok(())
    }
}
