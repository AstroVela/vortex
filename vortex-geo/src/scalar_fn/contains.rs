// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `ST_Contains`: OGC containment test between two native geometries.

use std::cell::OnceCell;

use geo::Contains;
use geo::PreparedGeometry;
use geo::Relate;
use geo_types::Geometry;
use vortex_array::ArrayRef;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::dtype::DType;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::RowFn;
use vortex_array::scalar_fn::RowVisitor;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::TypedScalarFnInstance;
use vortex_error::VortexResult;
use vortex_session::registry::CachedId;

use crate::scalar_fn::row::GeometryRow;
#[cfg(test)]
use crate::scalar_fn::row::probe;

/// OGC `ST_Contains` between two native geometry operands, each a column or a constant
/// literal: true where operand `b` lies completely inside operand `a` (boundary contact alone
/// does not count). Containment is not symmetric; the operand order is significant.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct GeoContains;

impl GeoContains {
    /// A lazy `ScalarFnArray` computing per-row whether operand `a` contains operand `b`;
    /// either may be constant. The output length is taken from `a`.
    pub fn try_new_array(a: ArrayRef, b: ArrayRef) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(
            TypedScalarFnInstance::new(GeoContains, EmptyOptions).erased(),
            vec![a, b],
        )
    }
}

impl RowFn for GeoContains {
    type Options = EmptyOptions;
    type ArgsWitness = (GeometryRow, GeometryRow);
    type RetWitness = bool;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.geo.contains");
        *ID
    }

    fn arg_name(&self, idx: usize) -> ChildName {
        ChildName::from(["a", "b"][idx])
    }

    /// Containment is not symmetric, so `a` is always the container and `b` the contained.
    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::Out> {
        visitor.visit_prepared::<(GeometryRow, GeometryRow), ConstOperands, bool>(
            |(a, b)| {
                #[cfg(test)]
                probe::record(a.is_some(), b.is_some());
                ConstOperands {
                    a: a.map(PreparedOperand::new),
                    b: b.map(PreparedOperand::new),
                }
            },
            |operands, (a, b)| contains_row_prepared(operands, a, b),
        )
    }
}

/// Per-batch state for the contains row kernel: the prepared form of whichever operand is
/// constant for the batch. `None` marks an operand that varies by row.
struct ConstOperands {
    /// Operand `a` (the container) when it is batch-constant.
    a: Option<PreparedOperand>,

    /// Operand `b` (the contained) when it is batch-constant.
    b: Option<PreparedOperand>,
}

/// One batch-constant operand: the geometry cloned out of its decoded column (the state must not
/// borrow from the columns), plus its [`PreparedGeometry`], built on the first row whose pairing
/// routes through relate.
///
/// The build is lazy because preparation (self-noding the topology graph plus an R*-tree over the
/// edges) costs `O(edges log edges)` and pays off only on relate-routed pairings; a batch of
/// point rows against a constant polygon never touches it, and preparing a large constant eagerly
/// would charge such a batch for nothing.
struct PreparedOperand {
    /// The constant's decoded geometry, owned so [`prepared`](Self::prepared) can be `'static`.
    geometry: Geometry<f64>,

    /// The lazily built prepared form of [`geometry`](Self::geometry).
    prepared: OnceCell<PreparedGeometry<'static, Geometry<f64>, f64>>,
}

impl PreparedOperand {
    fn new(geometry: &Geometry<f64>) -> Self {
        Self {
            geometry: geometry.clone(),
            prepared: OnceCell::new(),
        }
    }

    /// The prepared geometry, built on first use.
    fn get(&self) -> &PreparedGeometry<'static, Geometry<f64>, f64> {
        self.prepared
            .get_or_init(|| PreparedGeometry::from(self.geometry.clone()))
    }
}

/// How geo's `a.contains(b)` computes its verdict for a pairing.
enum ContainsRoute {
    /// `a.relate(b).is_contains()`.
    ForwardRelate,

    /// `b.relate(a).is_within()`, how geo phrases relate for `MultiPolygon` containers.
    ReversedRelate,

    /// A direct algorithm (coordinate position, point arithmetic); nothing to prepare.
    Direct,
}

/// The route geo 0.31's `Contains` dispatch takes for `a.contains(b)`.
///
/// The prepared substitution in [`contains_row_prepared`] **must** run relate exactly where geo
/// runs relate, with the same argument order, because geo's direct algorithms are not everywhere
/// bit-identical to a relate matrix query (they resolve degenerate and boundary cases with
/// different arithmetic). The relate rows below transcribe geo's `impl_contains_from_relate!`
/// lists per container type; everything else, notably every `Point`/`MultiPoint` contained side
/// and every `Point` container, is direct.
fn contains_route(a: &Geometry<f64>, b: &Geometry<f64>) -> ContainsRoute {
    use Geometry as G;

    match (a, b) {
        // Line contains [Polygon, MultiLineString, MultiPolygon, GeometryCollection, Rect,
        // Triangle].
        (
            G::Line(_),
            G::Polygon(_)
            | G::MultiLineString(_)
            | G::MultiPolygon(_)
            | G::GeometryCollection(_)
            | G::Rect(_)
            | G::Triangle(_),
        )
        // LineString contains [Polygon, MultiPoint, MultiLineString, MultiPolygon,
        // GeometryCollection, Rect, Triangle].
        | (
            G::LineString(_),
            G::Polygon(_)
            | G::MultiPoint(_)
            | G::MultiLineString(_)
            | G::MultiPolygon(_)
            | G::GeometryCollection(_)
            | G::Rect(_)
            | G::Triangle(_),
        )
        // MultiLineString contains everything except Point.
        | (
            G::MultiLineString(_),
            G::Line(_)
            | G::LineString(_)
            | G::Polygon(_)
            | G::MultiPoint(_)
            | G::MultiLineString(_)
            | G::MultiPolygon(_)
            | G::GeometryCollection(_)
            | G::Rect(_)
            | G::Triangle(_),
        )
        // MultiPoint contains [Line, LineString, Polygon, MultiLineString, MultiPolygon,
        // GeometryCollection, Rect, Triangle].
        | (
            G::MultiPoint(_),
            G::Line(_)
            | G::LineString(_)
            | G::Polygon(_)
            | G::MultiLineString(_)
            | G::MultiPolygon(_)
            | G::GeometryCollection(_)
            | G::Rect(_)
            | G::Triangle(_),
        )
        // Polygon contains everything except Point and MultiPoint.
        | (
            G::Polygon(_),
            G::Line(_)
            | G::LineString(_)
            | G::Polygon(_)
            | G::MultiLineString(_)
            | G::MultiPolygon(_)
            | G::GeometryCollection(_)
            | G::Rect(_)
            | G::Triangle(_),
        )
        // Rect contains [Line, LineString, MultiPoint, MultiLineString, MultiPolygon,
        // GeometryCollection, Triangle]; Rect contains Rect and Polygon are direct.
        | (
            G::Rect(_),
            G::Line(_)
            | G::LineString(_)
            | G::MultiPoint(_)
            | G::MultiLineString(_)
            | G::MultiPolygon(_)
            | G::GeometryCollection(_)
            | G::Triangle(_),
        )
        // Triangle and GeometryCollection contain everything except Point.
        | (
            G::Triangle(_) | G::GeometryCollection(_),
            G::Line(_)
            | G::LineString(_)
            | G::Polygon(_)
            | G::MultiPoint(_)
            | G::MultiLineString(_)
            | G::MultiPolygon(_)
            | G::GeometryCollection(_)
            | G::Rect(_)
            | G::Triangle(_),
        ) => ContainsRoute::ForwardRelate,

        // MultiPolygon contains everything except Point and MultiPoint, phrased reversed.
        (
            G::MultiPolygon(_),
            G::Line(_)
            | G::LineString(_)
            | G::Polygon(_)
            | G::MultiLineString(_)
            | G::MultiPolygon(_)
            | G::GeometryCollection(_)
            | G::Rect(_)
            | G::Triangle(_),
        ) => ContainsRoute::ReversedRelate,

        _ => ContainsRoute::Direct,
    }
}

/// Computes one row of contains, substituting a prepared graph for a constant operand on the
/// pairings geo itself answers through relate.
///
/// [`PreparedGeometry`] carries the operand's self-noded topology graph and edge R*-tree, so a
/// relate against it skips rebuilding both and reads its bounding rect from cache; geo asserts
/// the cached graph equal to a freshly built one (its `swap_arg_index` test), which is what makes
/// the substitution result-preserving. Direct pairings and the no-constant batch call the
/// unchanged `a.contains(b)`.
fn contains_row_prepared(operands: &ConstOperands, a: &Geometry<f64>, b: &Geometry<f64>) -> bool {
    if operands.a.is_none() && operands.b.is_none() {
        return a.contains(b);
    }

    match contains_route(a, b) {
        ContainsRoute::Direct => a.contains(b),
        ContainsRoute::ForwardRelate => match (&operands.a, &operands.b) {
            (Some(const_a), Some(const_b)) => const_a.get().relate(const_b.get()).is_contains(),
            (Some(const_a), None) => const_a.get().relate(b).is_contains(),
            (None, Some(const_b)) => a.relate(const_b.get()).is_contains(),
            (None, None) => a.contains(b),
        },
        ContainsRoute::ReversedRelate => match (&operands.a, &operands.b) {
            (Some(const_a), Some(const_b)) => const_b.get().relate(const_a.get()).is_within(),
            (Some(const_a), None) => b.relate(const_a.get()).is_within(),
            (None, Some(const_b)) => const_b.get().relate(a).is_within(),
            (None, None) => a.contains(b),
        },
    }
}

#[cfg(test)]
mod tests {
    use geo_types::Geometry;
    use geo_types::LineString;
    use geo_types::MultiPoint;
    use geo_types::MultiPolygon;
    use geo_types::Point;
    use geo_types::Polygon;
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::Canonical;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::MaskedArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::EmptyOptions;
    use vortex_array::scalar_fn::NullStrategy;
    use vortex_array::scalar_fn::ScalarFnVTable;
    use vortex_array::scalar_fn::execute_row_fn_with_strategy;
    use vortex_array::validity::Validity;
    use vortex_buffer::BitBuffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use wkb::writer::WriteOptions;

    use super::GeoContains;
    use crate::scalar_fn::row::probe::assert_prepared_agrees_with_columns;
    use crate::test_harness::linestring_column;
    use crate::test_harness::nullable_point_column;
    use crate::test_harness::point_column;
    use crate::test_harness::polygon_column;

    /// A rectangle polygon with corners `(x0, y0)` and `(x1, y1)`, no holes.
    fn rect_polygon(x0: f64, y0: f64, x1: f64, y1: f64) -> Polygon {
        Polygon::new(
            LineString::from(vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]),
            vec![],
        )
    }

    /// A constant column of length `len`, every row the native form of `geometry`.
    fn geometry_constant(geometry: &Geometry, len: usize) -> VortexResult<ArrayRef> {
        let mut buf = Vec::new();
        wkb::writer::write_geometry(&mut buf, geometry, &WriteOptions::default())
            .map_err(|e| vortex_err!("writing WKB failed: {e}"))?;
        let scalar = crate::extension::native_geometry_scalar_from_wkb(&buf)?
            .ok_or_else(|| vortex_err!("unsupported geometry type"))?;
        Ok(ConstantArray::new(scalar, len).into_array())
    }

    /// Materialize `array` so it is no longer a `Constant`, forcing the non-constant kernel
    /// paths.
    fn materialize(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        Ok(array.execute::<Canonical>(ctx)?.into_array())
    }

    /// Execute `GeoContains(a, b)` and assert the per-row verdicts equal `expected`.
    fn assert_contains(
        a: ArrayRef,
        b: ArrayRef,
        expected: impl IntoIterator<Item = bool>,
    ) -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();
        let contains = GeoContains::try_new_array(a, b)?.into_array();
        assert_arrays_eq!(contains, BoolArray::from_iter(expected), &mut ctx);
        Ok(())
    }

    // The tests cover each `execute` dispatch arm in match order, then the edge cases.

    /// Constant vs constant: a polygon contains a nested polygon but not a partially
    /// overlapping or disjoint one; every output row carries the same verdict.
    #[rstest]
    #[case::nested(rect_polygon(1.0, 1.0, 3.0, 3.0), true)]
    #[case::overlapping(rect_polygon(2.0, 2.0, 6.0, 6.0), false)]
    #[case::disjoint(rect_polygon(20.0, 20.0, 24.0, 24.0), false)]
    fn constant_vs_constant_polygons(
        #[case] other: Polygon,
        #[case] expected: bool,
    ) -> VortexResult<()> {
        let container = geometry_constant(&Geometry::Polygon(rect_polygon(0.0, 0.0, 4.0, 4.0)), 3)?;
        let other = geometry_constant(&Geometry::Polygon(other), 3)?;
        assert_contains(container, other, [expected; 3])
    }

    /// Partially overlapping polygons contain each other in neither direction.
    #[test]
    fn overlapping_polygons_contain_neither_way() -> VortexResult<()> {
        let a = geometry_constant(&Geometry::Polygon(rect_polygon(0.0, 0.0, 4.0, 4.0)), 2)?;
        let b = geometry_constant(&Geometry::Polygon(rect_polygon(2.0, 2.0, 6.0, 6.0)), 2)?;
        assert_contains(a.clone(), b.clone(), [false; 2])?;
        assert_contains(b, a, [false; 2])
    }

    /// Containment is not symmetric: a polygon contains an interior point, but the point does
    /// not contain the polygon.
    #[test]
    fn contains_is_asymmetric() -> VortexResult<()> {
        let polygon = geometry_constant(&Geometry::Polygon(rect_polygon(0.0, 0.0, 4.0, 4.0)), 2)?;
        let point = geometry_constant(&Geometry::Point(Point::new(2.0, 2.0)), 2)?;
        assert_contains(polygon.clone(), point.clone(), [true; 2])?;
        assert_contains(point, polygon, [false; 2])
    }

    /// Constant polygon vs point column: a strictly interior point is contained; points outside
    /// or exactly on the boundary are not (OGC contains excludes the boundary).
    #[test]
    fn constant_polygon_vs_point_column() -> VortexResult<()> {
        let container = geometry_constant(&Geometry::Polygon(rect_polygon(0.0, 0.0, 4.0, 4.0)), 3)?;
        let points = point_column(vec![2.0, 10.0, 0.0], vec![2.0, 10.0, 2.0])?;
        assert_contains(container, points, [true, false, false])
    }

    /// Polygon column vs constant point: only the polygon around the point contains it.
    #[test]
    fn polygon_column_vs_constant_point() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let around = materialize(
            geometry_constant(&Geometry::Polygon(rect_polygon(0.0, 0.0, 4.0, 4.0)), 2)?,
            &mut ctx,
        )?;
        let away = materialize(
            geometry_constant(&Geometry::Polygon(rect_polygon(20.0, 20.0, 24.0, 24.0)), 2)?,
            &mut ctx,
        )?;
        let point = geometry_constant(&Geometry::Point(Point::new(2.0, 2.0)), 2)?;

        assert_contains(around, point.clone(), [true; 2])?;
        assert_contains(away, point, [false; 2])
    }

    /// Column vs column pairs rows: each polygon row is tested against the point row at the
    /// same position.
    #[test]
    fn polygon_column_vs_point_column() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let polygons = materialize(
            geometry_constant(&Geometry::Polygon(rect_polygon(0.0, 0.0, 4.0, 4.0)), 2)?,
            &mut ctx,
        )?;
        let points = point_column(vec![2.0, 10.0], vec![2.0, 10.0])?;
        assert_contains(polygons, points, [true, false])
    }

    /// Output nullability mirrors the operands: nullable if any operand is nullable, otherwise
    /// non-nullable.
    #[test]
    fn output_nullability_mirrors_operands() -> VortexResult<()> {
        let dtype = point_column(vec![0.0], vec![0.0])?.dtype().clone();
        let non_nullable =
            GeoContains.return_dtype(&EmptyOptions, &[dtype.clone(), dtype.clone()])?;
        assert!(!non_nullable.is_nullable());
        let nullable = GeoContains.return_dtype(&EmptyOptions, &[dtype.as_nullable(), dtype])?;
        assert!(nullable.is_nullable());
        Ok(())
    }

    /// A null row in the contained operand yields a null verdict; valid rows keep their verdict
    /// (a strictly interior point is contained, an outside point is not).
    #[test]
    fn contains_propagates_null_rows() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let container = geometry_constant(&Geometry::Polygon(rect_polygon(0.0, 0.0, 4.0, 4.0)), 3)?;
        let points = nullable_point_column(vec![Some((2.0, 2.0)), None, Some((10.0, 10.0))])?;
        let contains = GeoContains::try_new_array(container, points)?.into_array();

        let expected = BoolArray::new(
            BitBuffer::from_iter([true, false, false]),
            Validity::from_iter([true, false, true]),
        )
        .into_array();
        assert_arrays_eq!(contains, expected, &mut ctx);
        Ok(())
    }

    /// A constant-null operand produces an all-null output.
    #[test]
    fn contains_constant_null_is_all_null() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let point_dtype = point_column(vec![0.0], vec![0.0])?.dtype().as_nullable();
        let null_const = ConstantArray::new(Scalar::null(point_dtype), 2).into_array();
        let points = point_column(vec![2.0, 10.0], vec![2.0, 10.0])?;
        let contains = GeoContains::try_new_array(null_const, points)?.into_array();

        let expected =
            BoolArray::new(BitBuffer::from_iter([false, false]), Validity::AllInvalid).into_array();
        assert_arrays_eq!(contains, expected, &mut ctx);
        Ok(())
    }

    /// Both operands nullable columns: containment (asymmetric) is null wherever either the
    /// container or the contained row is null, and computed on the rows valid in both.
    #[test]
    fn contains_propagates_column_pair_nulls() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        // A point contains another point only when they are equal.
        let container = nullable_point_column(vec![
            Some((1.0, 1.0)),
            None,
            Some((2.0, 2.0)),
            Some((3.0, 3.0)),
        ])?;
        let contained = nullable_point_column(vec![
            Some((1.0, 1.0)),
            Some((5.0, 5.0)),
            None,
            Some((4.0, 4.0)),
        ])?;
        let contains = GeoContains::try_new_array(container, contained)?.into_array();

        let expected = BoolArray::new(
            BitBuffer::from_iter([true, false, false, false]),
            Validity::from_iter([true, false, false, true]),
        )
        .into_array();
        assert_arrays_eq!(contains, expected, &mut ctx);
        Ok(())
    }

    /// An entirely-null geometry column yields an all-null output.
    #[test]
    fn contains_all_null_column_is_all_null() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let container = geometry_constant(&Geometry::Polygon(rect_polygon(0.0, 0.0, 4.0, 4.0)), 2)?;
        let points = nullable_point_column(vec![None, None])?;
        let contains = GeoContains::try_new_array(container, points)?.into_array();

        let expected =
            BoolArray::new(BitBuffer::from_iter([false, false]), Validity::AllInvalid).into_array();
        assert_arrays_eq!(contains, expected, &mut ctx);
        Ok(())
    }

    /// Two nullable columns whose nulls never line up: the combined mask is empty, so the output
    /// is all null.
    #[test]
    fn contains_column_pair_all_null() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let container = nullable_point_column(vec![Some((1.0, 1.0)), None])?;
        let contained = nullable_point_column(vec![None, Some((2.0, 2.0))])?;
        let contains = GeoContains::try_new_array(container, contained)?.into_array();

        let expected =
            BoolArray::new(BitBuffer::from_iter([false, false]), Validity::AllInvalid).into_array();
        assert_arrays_eq!(contains, expected, &mut ctx);
        Ok(())
    }

    /// A nullable polygon column: unit squares at `centers`, the rows where `nulls` is true
    /// masked out, spelled as `Masked` over non-nullable storage.
    fn nullable_squares(centers: &[(f64, f64)], nulls: &[bool]) -> VortexResult<ArrayRef> {
        let squares = centers
            .iter()
            .map(|&(x, y)| {
                vec![vec![
                    (x - 1.0, y - 1.0),
                    (x + 1.0, y - 1.0),
                    (x + 1.0, y + 1.0),
                    (x - 1.0, y + 1.0),
                    (x - 1.0, y - 1.0),
                ]]
            })
            .collect();
        let polygons = polygon_column(squares)?;

        Ok(
            MaskedArray::try_new(polygons, Validity::from_iter(nulls.iter().map(|n| !n)))?
                .into_array(),
        )
    }

    /// Executes `GeoContains(a, b)` with a forced null strategy, canonicalized.
    fn contains_forced(
        a: &ArrayRef,
        b: &ArrayRef,
        strategy: NullStrategy,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        Ok(execute_row_fn_with_strategy(
            &GeoContains,
            &EmptyOptions,
            vec![a.clone(), b.clone()],
            a.len(),
            strategy,
            ctx,
        )?
        .execute::<Canonical>(ctx)?
        .into_array())
    }

    /// The branch-and-skip and filter strategies, plus the automatic per-batch selection, must
    /// return identical arrays for nullable geometry operands: `Masked` polygons against nullable
    /// points, with independent nulls conjoined.
    #[test]
    fn branch_matches_filter_for_nullable_geometries() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let centers = [(0.0, 0.0), (5.0, 5.0), (0.5, -0.2), (9.0, 9.0), (0.0, 1.0)];
        let nulls = [false, true, false, false, true];
        let polygons = nullable_squares(&centers, &nulls)?;
        let points = nullable_point_column(vec![
            Some((0.0, 0.0)),
            Some((5.0, 5.0)),
            None,
            Some((0.0, 0.0)),
            Some((0.0, 1.0)),
        ])?;

        let filtered = contains_forced(&polygons, &points, NullStrategy::Filter, &mut ctx)?;
        let branched = contains_forced(&polygons, &points, NullStrategy::BranchAndSkip, &mut ctx)?;
        let auto = GeoContains::try_new_array(polygons, points)?
            .into_array()
            .execute::<Canonical>(&mut ctx)?
            .into_array();

        assert_arrays_eq!(branched, filtered, &mut ctx);
        assert_arrays_eq!(auto, filtered, &mut ctx);
        Ok(())
    }

    /// Geometry types without a null-tolerant decode refuse the branch strategy: forcing it is an
    /// error, and the automatic selection (which prefers branch at this density) silently falls
    /// back to filtering with the correct result.
    #[test]
    fn unsupported_geometry_falls_back_to_filter() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        // Four rows with one null: 75% surviving, so the selection prefers branch.
        let lines = MaskedArray::try_new(
            linestring_column(vec![
                vec![(0.0, 0.0), (4.0, 4.0)],
                vec![(0.0, 0.0), (1.0, 1.0)],
                vec![(2.0, 2.0), (3.0, 3.0)],
                vec![(0.0, 4.0), (4.0, 0.0)],
            ])?,
            Validity::from_iter([true, false, true, true]),
        )?
        .into_array();
        let point = geometry_constant(&Geometry::Point(Point::new(2.0, 2.0)), 4)?;

        let error = contains_forced(&lines, &point, NullStrategy::BranchAndSkip, &mut ctx)
            .expect_err("a linestring column with nulls has no branch decode");
        assert!(
            error.to_string().contains("branch-and-skip"),
            "unexpected error: {error}"
        );

        let filtered = contains_forced(&lines, &point, NullStrategy::Filter, &mut ctx)?;
        let auto = GeoContains::try_new_array(lines, point)?
            .into_array()
            .execute::<Canonical>(&mut ctx)?
            .into_array();

        assert_arrays_eq!(auto, filtered, &mut ctx);
        Ok(())
    }

    /// A non-geometry operand dtype is rejected up front, before execution.
    #[test]
    fn non_geometry_operand_is_rejected() -> VortexResult<()> {
        let geo = point_column(vec![0.0], vec![0.0])?.dtype().clone();
        let numeric = DType::Primitive(PType::I32, Nullability::NonNullable);
        let result = GeoContains.return_dtype(&EmptyOptions, &[geo, numeric]);
        assert!(result.is_err());
        Ok(())
    }

    // The prepared-vs-expanded agreement grid: every constant arrangement of a pairing must
    // return exactly what the fully expanded columns return.

    /// A point geometry.
    fn point(x: f64, y: f64) -> Geometry {
        Geometry::Point(Point::new(x, y))
    }

    /// A linestring geometry through `coords`.
    fn line(coords: Vec<(f64, f64)>) -> Geometry {
        Geometry::LineString(LineString::from(coords))
    }

    /// A multipoint geometry over `coords`.
    fn multipoint(coords: Vec<(f64, f64)>) -> Geometry {
        Geometry::MultiPoint(MultiPoint::from(coords))
    }

    /// A two-part multipolygon: `4x4` squares at the origin and at `(10, 10)`.
    fn two_part_multipolygon() -> Geometry {
        Geometry::MultiPolygon(MultiPolygon::new(vec![
            rect_polygon(0.0, 0.0, 4.0, 4.0),
            rect_polygon(10.0, 10.0, 14.0, 14.0),
        ]))
    }

    /// Constant arrangements agree with expanded columns across the routes the prepared kernel
    /// distinguishes: forward relate (polygon and linestring containers), reversed relate
    /// (multipolygon containers), and the direct pairings (a point on either side, polygon over
    /// multipoint), including boundary contact, crossing, disjoint and empty cases.
    #[rstest]
    #[case::polygon_nested_polygon(rect_polygon(0.0, 0.0, 8.0, 8.0).into(), rect_polygon(2.0, 2.0, 4.0, 4.0).into())]
    #[case::polygon_touching_from_inside(rect_polygon(0.0, 0.0, 8.0, 8.0).into(), rect_polygon(0.0, 2.0, 2.0, 4.0).into())]
    #[case::polygon_overlapping_polygon(rect_polygon(0.0, 0.0, 4.0, 4.0).into(), rect_polygon(2.0, 2.0, 6.0, 6.0).into())]
    #[case::polygon_disjoint_polygon(rect_polygon(0.0, 0.0, 4.0, 4.0).into(), rect_polygon(20.0, 20.0, 24.0, 24.0).into())]
    #[case::polygon_x_point_inside(rect_polygon(0.0, 0.0, 4.0, 4.0).into(), point(2.0, 2.0))]
    #[case::polygon_x_point_on_boundary(rect_polygon(0.0, 0.0, 4.0, 4.0).into(), point(0.0, 2.0))]
    #[case::polygon_x_point_outside(rect_polygon(0.0, 0.0, 4.0, 4.0).into(), point(20.0, 20.0))]
    #[case::polygon_x_linestring_inside(rect_polygon(0.0, 0.0, 4.0, 4.0).into(), line(vec![(1.0, 1.0), (2.0, 2.0)]))]
    #[case::polygon_x_linestring_on_boundary(rect_polygon(0.0, 0.0, 4.0, 4.0).into(), line(vec![(0.0, 1.0), (0.0, 3.0)]))]
    #[case::polygon_x_linestring_crossing(rect_polygon(0.0, 0.0, 4.0, 4.0).into(), line(vec![(-2.0, 2.0), (2.0, 2.0)]))]
    #[case::polygon_x_empty_linestring(rect_polygon(0.0, 0.0, 4.0, 4.0).into(), line(vec![]))]
    #[case::polygon_x_multipoint_inside(rect_polygon(0.0, 0.0, 4.0, 4.0).into(), multipoint(vec![(1.0, 1.0), (2.0, 2.0)]))]
    #[case::polygon_x_multipoint_on_boundary(rect_polygon(0.0, 0.0, 4.0, 4.0).into(), multipoint(vec![(0.0, 1.0), (0.0, 3.0)]))]
    #[case::linestring_x_multipoint_on_line(line(vec![(0.0, 0.0), (4.0, 4.0)]), multipoint(vec![(1.0, 1.0), (2.0, 2.0)]))]
    #[case::multipolygon_x_polygon_in_one_part(two_part_multipolygon(), rect_polygon(1.0, 1.0, 3.0, 3.0).into())]
    #[case::multipolygon_x_polygon_straddling(two_part_multipolygon(), rect_polygon(3.0, 3.0, 11.0, 11.0).into())]
    #[case::multipolygon_x_polygon_disjoint(two_part_multipolygon(), rect_polygon(20.0, 20.0, 24.0, 24.0).into())]
    #[case::multipolygon_x_point_inside(two_part_multipolygon(), point(11.0, 11.0))]
    #[case::point_x_point_equal(point(1.0, 1.0), point(1.0, 1.0))]
    #[case::point_x_polygon(point(2.0, 2.0), rect_polygon(0.0, 0.0, 4.0, 4.0).into())]
    fn constant_operands_agree_with_columns(
        #[case] a: Geometry,
        #[case] b: Geometry,
    ) -> VortexResult<()> {
        assert_prepared_agrees_with_columns(
            GeoContains::try_new_array,
            geometry_constant(&a, 3)?,
            geometry_constant(&b, 3)?,
        )
    }
}
