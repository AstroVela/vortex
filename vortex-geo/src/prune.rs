// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Chunk pruning for spatial filters, using the per-chunk [`GeometryAabb`] axis-aligned
//! bounding box (AABB).

use geo::BoundingRect;
use geo::Rect as GeoRect;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::expr::Expression;
use vortex_array::expr::case_when;
use vortex_array::expr::checked_add;
use vortex_array::expr::ext_storage;
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::gt_eq;
use vortex_array::expr::is_root;
use vortex_array::expr::lit;
use vortex_array::expr::lt;
use vortex_array::expr::lt_eq;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::literal::Literal;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::stats::rewrite::StatsRewriteCtx;
use vortex_array::stats::rewrite::StatsRewriteRule;
use vortex_array::stats::stat;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::aggregate_fn::GeometryAabb;
use crate::extension::is_native_geometry;
use crate::extension::single_geometry;
use crate::scalar_fn::distance::GeoDistance;

/// Prunes chunks for `ST_Distance(geom, const) <op> r` filters using the chunk's [`GeometryAabb`]
/// bounding box. Register it with `crate::initialize`; a chunk written without the `GeometryAabb`
/// statistic is scanned rather than skipped.
///
/// All four comparisons prune: `<= r` / `< r` skip a chunk whose box is wholly beyond `r` (box
/// min-distance); `>= r` / `> r` skip one wholly within `r` (box max-distance). `==` / `!=` don't
/// prune. To add another spatial predicate, write a sibling [`StatsRewriteRule`] from the
/// `geometry_and_constant` + `distance_prune_proof` helpers; no new statistic or file-format change
/// is needed.
#[derive(Debug)]
pub struct GeoDistancePrune;

impl StatsRewriteRule for GeoDistancePrune {
    fn scalar_fn_id(&self) -> ScalarFnId {
        // The predicate root is the comparison, not `GeoDistance`, so key on `Binary`.
        Binary.id()
    }

    fn falsify(
        &self,
        expr: &Expression,
        ctx: &StatsRewriteCtx<'_>,
    ) -> VortexResult<Option<Expression>> {
        // Only the ordered comparisons prune today. `== r` could prune in the future (a chunk is
        // provably empty when `r` lies outside its box's [min, max] distance interval), it's just
        // not implemented. `!= r` cannot: pruning would need every row's distance to equal `r`,
        // which an AABB can't prove.
        let op = *expr.as_::<Binary>();
        if !matches!(
            op,
            Operator::Lte | Operator::Lt | Operator::Gte | Operator::Gt
        ) {
            return Ok(None);
        }

        // The left operand must be `GeoDistance(geom, const)`; the right, the radius literal.
        let distance = expr.child(0);
        if distance.as_opt::<GeoDistance>().is_none() {
            return Ok(None);
        }
        let Some((geom, constant)) = geometry_and_constant(distance, ctx)? else {
            return Ok(None);
        };
        let Some(radius) = expr.child(1).as_opt::<Literal>() else {
            return Ok(None);
        };
        // Casts any primitive radius (integer literals included); it fails only for a null or
        // non-primitive literal, where falling through means "scan the chunk", which is always
        // sound.
        let Ok(radius) = f64::try_from(radius) else {
            return Ok(None);
        };
        // A NaN radius has no sound proof: `distance <op> NaN` is not a total order, so the chunk
        // must be scanned, not pruned.
        if radius.is_nan() {
            return Ok(None);
        }

        // Reduce `const` (any geometry type) to its AABB. Every row sits in the chunk AABB
        // and `const` in this box, so the box-to-box distance bounds the true distance soundly for
        // any geometry types.
        let mut exec = ctx.session().create_execution_ctx();
        let Some(query) = single_geometry(constant, &mut exec)?.bounding_rect() else {
            return Ok(None);
        };
        Ok(distance_prune_proof(geom, query, op, radius))
    }
}

/// Shared AABB-pruning helper: split a symmetric geo predicate's operands into the scope-rooted
/// geometry column and the constant's scalar, or `None` when the expression doesn't have that
/// shape or the geometry's dtype has no [`GeometryAabb`] support. Symmetric only - an asymmetric
/// predicate that needs to know *which* operand is the column must recover the role separately.
fn geometry_and_constant<'a>(
    expr: &'a Expression,
    ctx: &StatsRewriteCtx<'_>,
) -> VortexResult<Option<(&'a Expression, &'a Scalar)>> {
    // The predicate is symmetric, so the geometry column (scope root) and the constant may be on
    // either side.
    let (lhs, rhs) = (expr.child(0), expr.child(1));
    let (geom, constant) = if is_root(lhs) {
        (lhs, rhs)
    } else if is_root(rhs) {
        (rhs, lhs)
    } else {
        return Ok(None);
    };

    // A `GeometryAabb` stat reference only binds for dtypes it supports; anything else (e.g. a
    // WKB column) must fall through to the scan.
    if !is_native_geometry(&ctx.return_dtype(geom)?) {
        return Ok(None);
    }

    Ok(constant.as_opt::<Literal>().map(|scalar| (geom, scalar)))
}

/// Build the prune proof for `ST_Distance(geom, const) <op> radius` from the chunk's bounding-box
/// stat, or `None` when this operator/radius cannot prune. `<=` / `<` prune a chunk whose box is
/// wholly beyond `radius` (min box-distance); `>=` / `>` prune one wholly within it (max
/// box-distance). Every row sits in the box, so those bounds prove the whole chunk one-sided.
///
/// A distance is always `>= 0`, which decides the degenerate radii up front.
fn distance_prune_proof(
    geom: &Expression,
    query: GeoRect<f64>,
    op: Operator,
    radius: f64,
) -> Option<Expression> {
    // A distance is always non-negative, so degenerate radii resolve without touching the box.
    match op {
        // `<= r` / `< r` with a negative radius (or zero, for `<`) match nothing: prune every chunk.
        Operator::Lte if radius < 0.0 => return Some(lit(true)),
        Operator::Lt if radius <= 0.0 => return Some(lit(true)),
        // `>= r` / `> r` with a negative radius (or zero, for `>=`) match all rows: never prune.
        Operator::Gte if radius <= 0.0 => return None,
        Operator::Gt if radius < 0.0 => return None,
        _ => {}
    }
    // The stat is read through `ext_storage`/`get_item`, which propagate a missing stat (null) to
    // "keep the chunk". Compared squared to avoid a `sqrt`; all operands are `>= 0`.
    let aabb = ext_storage(stat(geom.clone(), GeometryAabb.bind(EmptyOptions)));
    let r2 = lit(radius * radius);
    Some(match op {
        // Beyond the threshold: even the nearest the box can be exceeds `r`.
        Operator::Lte => gt(min_dist_sq(&aabb, query), r2),
        Operator::Lt => gt_eq(min_dist_sq(&aabb, query), r2),
        // Within the threshold: even the farthest the box can be is below `r`.
        Operator::Gte => lt(max_dist_sq(&aabb, query), r2),
        Operator::Gt => lt_eq(max_dist_sq(&aabb, query), r2),
        _ => return None,
    })
}

/// Squared minimum distance between the chunk box `aabb` and the query box - a lower bound on every
/// row's distance. `dx^2 + dy^2`, each axis gap clamped at zero (zero when the intervals overlap).
fn min_dist_sq(aabb: &Expression, query: GeoRect<f64>) -> Expression {
    let field = |name: &str| get_item(name, aabb.clone());
    // max(0, q_lo - aabb_hi, aabb_lo - q_hi): positive only when the intervals are separated.
    let gap = |q_lo: f64, q_hi: f64, lo: Expression, hi: Expression| {
        maximum(
            lit(0.0),
            maximum(
                binop(Operator::Sub, lit(q_lo), hi),
                binop(Operator::Sub, lo, lit(q_hi)),
            ),
        )
    };
    let dx = gap(query.min().x, query.max().x, field("xmin"), field("xmax"));
    let dy = gap(query.min().y, query.max().y, field("ymin"), field("ymax"));
    checked_add(square(dx), square(dy))
}

/// Squared maximum distance between the chunk box `aabb` and the query box - an upper bound on every
/// row's distance. `Dx^2 + Dy^2`, each axis span the full extent of the two intervals' union.
fn max_dist_sq(aabb: &Expression, query: GeoRect<f64>) -> Expression {
    let field = |name: &str| get_item(name, aabb.clone());
    // max(q_hi, aabb_hi) - min(q_lo, aabb_lo): the farthest two points of the boxes can be on an axis.
    // The (nullable) AABB field is passed as the second arg so `case_when`'s else branch carries the
    // nullability - a missing stat then propagates null through to "keep the chunk".
    let span = |q_lo: f64, q_hi: f64, lo: Expression, hi: Expression| {
        binop(
            Operator::Sub,
            maximum(lit(q_hi), hi),
            minimum(lit(q_lo), lo),
        )
    };
    let dx = span(query.min().x, query.max().x, field("xmin"), field("xmax"));
    let dy = span(query.min().y, query.max().y, field("ymin"), field("ymax"));
    checked_add(square(dx), square(dy))
}

/// `a <op> b` as a binary-operator expression.
fn binop(op: Operator, a: Expression, b: Expression) -> Expression {
    Binary
        .try_new_expr(op, [a, b])
        .vortex_expect("binary expression")
}

/// `e * e` as an expression.
fn square(e: Expression) -> Expression {
    binop(Operator::Mul, e.clone(), e)
}

/// `max(a, b)` as an expression.
fn maximum(a: Expression, b: Expression) -> Expression {
    case_when(gt(a.clone(), b.clone()), a, b)
}

/// `min(a, b)` as an expression.
fn minimum(a: Expression, b: Expression) -> Expression {
    case_when(lt(a.clone(), b.clone()), a, b)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::AggregateFnVTableExt;
    use vortex_array::aggregate_fn::EmptyOptions as AggregateEmptyOptions;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::FieldNames;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::expr::Expression;
    use vortex_array::expr::gt_eq;
    use vortex_array::expr::lit;
    use vortex_array::expr::lt_eq;
    use vortex_array::expr::root;
    use vortex_array::scalar_fn::EmptyOptions;
    use vortex_array::scalar_fn::ScalarFnVTableExt;
    use vortex_array::scalar_fn::fns::binary::Binary;
    use vortex_array::scalar_fn::fns::operators::Operator;
    use vortex_array::stats::rewrite::StatsRewriteCtx;
    use vortex_array::stats::rewrite::StatsRewriteRule;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;
    use vortex_layout::layouts::zoned::zone_map::ZoneMap;

    use super::GeoDistancePrune;
    use crate::aggregate_fn::GeometryAabb;
    use crate::extension::GeoMetadata;
    use crate::extension::Rect;
    use crate::scalar_fn::distance::GeoDistance;
    use crate::test_harness::geo_session;
    use crate::test_harness::point_column;

    /// Run the rule against `GeoDistance(root, origin) <operator> radius`, operands swapped when
    /// `geom_first` is false.
    fn falsify_distance(
        operator: Operator,
        geom_first: bool,
        radius: f64,
    ) -> VortexResult<Option<Expression>> {
        let session = geo_session();
        let mut ctx = session.create_execution_ctx();

        let scope = point_column(vec![0.0], vec![0.0])?.dtype().clone();
        let origin = point_column(vec![0.0], vec![0.0])?.execute_scalar(0, &mut ctx)?;
        let operands = if geom_first {
            [root(), lit(origin)]
        } else {
            [lit(origin), root()]
        };
        let distance = GeoDistance.new_expr(EmptyOptions, operands);
        let predicate = Binary.new_expr(operator, [distance, lit(radius)]);

        GeoDistancePrune.falsify(&predicate, &StatsRewriteCtx::new(&session, &scope))
    }

    /// All four distance comparisons prune (`<=`/`<` via min-distance, `>=`/`>` via max-distance);
    /// `==`/`!=` are left to the scan.
    #[rstest]
    #[case(Operator::Lte, true)]
    #[case(Operator::Lt, true)]
    #[case(Operator::Gt, true)]
    #[case(Operator::Gte, true)]
    #[case(Operator::Eq, false)]
    #[case(Operator::NotEq, false)]
    fn prunes_distance_comparisons(
        #[case] operator: Operator,
        #[case] prunes: bool,
    ) -> VortexResult<()> {
        assert_eq!(falsify_distance(operator, true, 0.5)?.is_some(), prunes);
        Ok(())
    }

    /// Distance is symmetric: `GeoDistance(const, geom) <= r` falsifies just like the geom-first form.
    #[test]
    fn falsifies_with_constant_as_left_operand() -> VortexResult<()> {
        assert!(falsify_distance(Operator::Lte, false, 0.5)?.is_some());
        Ok(())
    }

    /// A NaN radius must not prune - the scan's total-order compare treats `dist <= NaN` as true.
    #[test]
    fn nan_radius_never_prunes() -> VortexResult<()> {
        assert!(falsify_distance(Operator::Lte, true, f64::NAN)?.is_none());
        Ok(())
    }

    /// A negative radius prunes every zone - vacuously sound: `distance <= r < 0` matches no row.
    #[test]
    fn negative_radius_prunes_vacuously() -> VortexResult<()> {
        assert!(falsify_distance(Operator::Lte, true, -0.5)?.is_some());
        Ok(())
    }

    /// A scope dtype without `GeometryAabb` support gets no proof - the stat reference would
    /// fail to bind at prune time.
    #[test]
    fn unsupported_scope_is_not_pruned() -> VortexResult<()> {
        let session = geo_session();
        let mut ctx = session.create_execution_ctx();

        let scope = DType::Primitive(PType::F64, Nullability::NonNullable);
        let origin = point_column(vec![0.0], vec![0.0])?.execute_scalar(0, &mut ctx)?;
        let distance = GeoDistance.new_expr(EmptyOptions, [root(), lit(origin)]);
        let predicate = lt_eq(distance, lit(0.5f64));

        let ctx = StatsRewriteCtx::new(&session, &scope);
        assert!(GeoDistancePrune.falsify(&predicate, &ctx)?.is_none());
        Ok(())
    }

    /// A comparison that does not wrap `GeoDistance` is left untouched.
    #[test]
    fn ignores_non_distance_comparison() -> VortexResult<()> {
        let session = geo_session();
        let scope = point_column(vec![0.0], vec![0.0])?.dtype().clone();

        let predicate = lt_eq(lit(1.0f64), lit(2.0f64));
        let ctx = StatsRewriteCtx::new(&session, &scope);
        assert!(GeoDistancePrune.falsify(&predicate, &ctx)?.is_none());
        Ok(())
    }

    /// End-to-end over a hand-built zone map: the far chunk is skipped, the near one kept.
    #[test]
    fn prunes_far_chunk_keeps_near() -> VortexResult<()> {
        let session = geo_session();
        let mut ctx = session.create_execution_ctx();

        let point_dtype = point_column(vec![0.0], vec![0.0])?.dtype().clone();
        let aabb_fn = GeometryAabb.bind(AggregateEmptyOptions);

        // Two chunks' AABBs, stored as the native `geoarrow.box` stat column with default
        // (unreferenced) metadata to match the aggregate's return dtype: chunk 0 near the origin
        // (0,0..1,1), chunk 1 far away (100,100..101,101).
        let ord = |a: f64, b: f64| PrimitiveArray::from_iter([a, b]).into_array();
        let boxes = StructArray::try_new(
            ["xmin", "ymin", "xmax", "ymax"].into(),
            vec![
                ord(0.0, 100.0),
                ord(0.0, 100.0),
                ord(1.0, 101.0),
                ord(1.0, 101.0),
            ],
            2,
            Validity::AllValid,
        )?
        .into_array();
        let box_dtype =
            ExtDType::<Rect>::try_new(GeoMetadata::default(), boxes.dtype().clone())?.erased();
        let aabbs = ExtensionArray::try_new(box_dtype, boxes)?.into_array();
        let zone_array = StructArray::from_fields(&[(aabb_fn.to_string().as_str(), aabbs)])?;
        let zone_map =
            ZoneMap::try_new(point_dtype.clone(), zone_array, Arc::new([aabb_fn]), 1, 2)?;

        let origin = point_column(vec![0.0], vec![0.0])?.execute_scalar(0, &mut ctx)?;
        let distance = GeoDistance.new_expr(EmptyOptions, [root(), lit(origin)]);
        let predicate = lt_eq(distance, lit(0.5f64));
        let proof = predicate
            .falsify(&point_dtype, &session)?
            .expect("distance filter should be falsifiable");

        // `true` means the zone is pruned: chunk 0 (near origin) is kept, chunk 1 (far) is skipped.
        let mask = zone_map.prune(&proof, &session)?;
        assert_eq!(mask.iter().collect::<Vec<bool>>(), vec![false, true]);
        Ok(())
    }

    /// The true-distance prune skips a chunk that is *diagonally* farther than `r`, even though
    /// neither axis alone exceeds `r` - the case a per-axis box-overlap test would wrongly keep.
    #[test]
    fn prunes_diagonally_distant_chunk() -> VortexResult<()> {
        let session = geo_session();
        let mut ctx = session.create_execution_ctx();

        let point_dtype = point_column(vec![0.0], vec![0.0])?.dtype().clone();
        let aabb_fn = GeometryAabb.bind(AggregateEmptyOptions);

        // One chunk, AABB (0.8,0.8)..(0.9,0.9): each axis is only 0.8 from the origin (<= r = 1), but
        // the near corner is sqrt(0.8^2 + 0.8^2) ~= 1.13 away (> 1), so no point in the box is within 1.
        let ord = |a: f64| PrimitiveArray::from_iter([a]).into_array();
        let boxes = StructArray::try_new(
            ["xmin", "ymin", "xmax", "ymax"].into(),
            vec![ord(0.8), ord(0.8), ord(0.9), ord(0.9)],
            1,
            Validity::AllValid,
        )?
        .into_array();
        let box_dtype =
            ExtDType::<Rect>::try_new(GeoMetadata::default(), boxes.dtype().clone())?.erased();
        let aabbs = ExtensionArray::try_new(box_dtype, boxes)?.into_array();
        let zone_array = StructArray::from_fields(&[(aabb_fn.to_string().as_str(), aabbs)])?;
        let zone_map =
            ZoneMap::try_new(point_dtype.clone(), zone_array, Arc::new([aabb_fn]), 1, 1)?;

        let origin = point_column(vec![0.0], vec![0.0])?.execute_scalar(0, &mut ctx)?;
        let distance = GeoDistance.new_expr(EmptyOptions, [root(), lit(origin)]);
        let predicate = lt_eq(distance, lit(1.0f64));
        let proof = predicate
            .falsify(&point_dtype, &session)?
            .expect("distance filter should be falsifiable");

        assert_eq!(
            zone_map
                .prune(&proof, &session)?
                .iter()
                .collect::<Vec<bool>>(),
            vec![true],
        );
        Ok(())
    }

    /// Backward compat: a zone map written without the `GeometryAabb` stat (an older file) keeps
    /// every zone - the missing stat binds to null and `null_as_false` retains the zone.
    #[test]
    fn missing_aabb_stat_keeps_all_zones() -> VortexResult<()> {
        let session = geo_session();
        let mut ctx = session.create_execution_ctx();

        let point_dtype = point_column(vec![0.0], vec![0.0])?.dtype().clone();
        let zone_map = ZoneMap::try_new(
            point_dtype.clone(),
            StructArray::try_new(FieldNames::empty(), vec![], 2, Validity::NonNullable)?,
            Arc::new([]),
            1,
            2,
        )?;

        let origin = point_column(vec![0.0], vec![0.0])?.execute_scalar(0, &mut ctx)?;
        let distance = GeoDistance.new_expr(EmptyOptions, [root(), lit(origin)]);
        let proof = lt_eq(distance, lit(0.5f64))
            .falsify(&point_dtype, &session)?
            .expect("distance filter should be falsifiable");

        let mask = zone_map.prune(&proof, &session)?;
        assert_eq!(mask.iter().collect::<Vec<bool>>(), vec![false, false]);
        Ok(())
    }

    /// A `>= r` filter prunes a chunk lying wholly *within* `r` (every row nearer than `r`, so none
    /// satisfy `>= r`) via the box max-distance, while a chunk beyond `r` is kept.
    #[test]
    fn prunes_within_chunk_for_far_filter() -> VortexResult<()> {
        let session = geo_session();
        let mut ctx = session.create_execution_ctx();

        let point_dtype = point_column(vec![0.0], vec![0.0])?.dtype().clone();
        let aabb_fn = GeometryAabb.bind(AggregateEmptyOptions);

        // Chunk 0 (AABB 0,0..0.5,0.5, farthest corner ~= 0.707) is entirely within 2 of the origin;
        // chunk 1 (100,100..101,101) is entirely beyond it.
        let ord = |a: f64, b: f64| PrimitiveArray::from_iter([a, b]).into_array();
        let boxes = StructArray::try_new(
            ["xmin", "ymin", "xmax", "ymax"].into(),
            vec![
                ord(0.0, 100.0),
                ord(0.0, 100.0),
                ord(0.5, 101.0),
                ord(0.5, 101.0),
            ],
            2,
            Validity::AllValid,
        )?
        .into_array();
        let box_dtype =
            ExtDType::<Rect>::try_new(GeoMetadata::default(), boxes.dtype().clone())?.erased();
        let aabbs = ExtensionArray::try_new(box_dtype, boxes)?.into_array();
        let zone_array = StructArray::from_fields(&[(aabb_fn.to_string().as_str(), aabbs)])?;
        let zone_map =
            ZoneMap::try_new(point_dtype.clone(), zone_array, Arc::new([aabb_fn]), 1, 2)?;

        let origin = point_column(vec![0.0], vec![0.0])?.execute_scalar(0, &mut ctx)?;
        let distance = GeoDistance.new_expr(EmptyOptions, [root(), lit(origin)]);
        let proof = gt_eq(distance, lit(2.0f64))
            .falsify(&point_dtype, &session)?
            .expect("distance filter should be falsifiable");

        // Chunk 0 (within 2) is pruned for `>= 2`; chunk 1 (beyond 2) is kept.
        let mask = zone_map.prune(&proof, &session)?;
        assert_eq!(mask.iter().collect::<Vec<bool>>(), vec![true, false]);
        Ok(())
    }
}
