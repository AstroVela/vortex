// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared execution for the binary geo scalar functions.
//!
//! [`execute_null_propagating`] runs a binary geo kernel (`ST_Distance`, `ST_Intersects`,
//! `ST_Contains`) over its two operands, decoding to `geo_types` and computing per row. Nulls
//! propagate as in SQL — the result is null wherever either operand is null — which the kernels
//! also expose via `vortex_array::expr::union_child_validities` as their `validity()`, so the
//! planner can derive the output null mask without executing them.
//!
//! When exactly one operand is a constant, a predicate kernel may pass a [`BboxReject`]
//! pre-check: the constant's bounding rect is fixed once per batch, and a row whose own rect
//! already proves the result skips the exact per-row test.

use geo::BoundingRect;
use geo_types::Geometry;
use geo_types::Rect;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::extension::geometries;
use crate::extension::single_geometry;

/// The result type a binary geo kernel produces. Today that is `f64` (for `ST_Distance`) and
/// `bool` (for the `ST_Intersects` / `ST_Contains` predicates), and the trait is implemented for
/// both. A kernel that returns some other type just adds its own `impl GeoOutput`.
pub(crate) trait GeoOutput: Copy {
    /// Convert this computed value into a Vortex [`Scalar`] (one typed, nullable value). Used
    /// only when both operands are constant: the kernel computes a single result, and this wraps
    /// it so a constant array can repeat that one value across every row.
    fn into_scalar(self, nullability: Nullability) -> Scalar;

    /// Assemble the `len`-row output: `values` (one per valid row, in row order) land at the set
    /// positions of `valid`, and every other row is null. With an empty `valid` this is the
    /// all-null output.
    fn build_array(
        len: usize,
        valid: &Mask,
        values: Vec<Self>,
        nullability: Nullability,
    ) -> ArrayRef;
}

impl GeoOutput for f64 {
    fn into_scalar(self, nullability: Nullability) -> Scalar {
        Scalar::primitive(self, nullability)
    }

    fn build_array(
        len: usize,
        valid: &Mask,
        values: Vec<Self>,
        nullability: Nullability,
    ) -> ArrayRef {
        let validity = Validity::from_mask(valid.clone(), nullability);
        match valid.indices() {
            // No nulls: `values` already lines up one-to-one with the rows.
            AllOr::All => PrimitiveArray::new(values, validity).into_array(),
            // No valid rows: the whole output is null.
            AllOr::None => PrimitiveArray::new(vec![0.0f64; len], validity).into_array(),
            // Some nulls: scatter each computed value back to the row it came from.
            AllOr::Some(rows) => {
                let mut data = vec![0.0f64; len];
                for (&row, value) in rows.iter().zip(values) {
                    data[row] = value;
                }
                PrimitiveArray::new(data, validity).into_array()
            }
        }
    }
}

impl GeoOutput for bool {
    fn into_scalar(self, nullability: Nullability) -> Scalar {
        Scalar::bool(self, nullability)
    }

    fn build_array(
        len: usize,
        valid: &Mask,
        values: Vec<Self>,
        nullability: Nullability,
    ) -> ArrayRef {
        let validity = Validity::from_mask(valid.clone(), nullability);
        match valid.indices() {
            // No nulls: `values` already lines up one-to-one with the rows.
            AllOr::All => BoolArray::new(BitBuffer::from_iter(values), validity).into_array(),
            // No valid rows: the whole output is null.
            AllOr::None => BoolArray::new(BitBuffer::new_unset(len), validity).into_array(),
            // Some nulls: scatter each computed value back to the row it came from.
            AllOr::Some(rows) => {
                let mut data = vec![false; len];
                for (&row, value) in rows.iter().zip(values) {
                    data[row] = value;
                }
                BoolArray::new(BitBuffer::from_iter(data), validity).into_array()
            }
        }
    }
}

/// A bounding-rect pre-check for [`execute_null_propagating`]'s one-constant arms.
///
/// Called per row with the operands' bounding rects in operand order (`a`'s, then `b`'s), it
/// returns `Some(result)` when the rects alone prove the kernel's result — the exact test is
/// skipped — and `None` when they cannot. The proof must be sound, never a guess: disjoint rects
/// prove `ST_Intersects` false, and a container rect not covering the contained rect proves
/// `ST_Contains` false. A kernel a rect cannot decide (`ST_Distance` produces a value, not a
/// verdict) passes `None` for the whole parameter.
pub(crate) type BboxReject<T> = fn(&Rect<f64>, &Rect<f64>) -> Option<T>;

/// Run a binary geo kernel over operands `a` and `b`, each a column or a constant literal.
///
/// The output is null wherever either operand is null, and its type is nullable if either operand
/// is: equivalently, the output validity is the intersection of the operands' validities.
///
/// The core idea: a geo kernel decodes each operand into a `geo_types` geometry, and a null row
/// has no geometry to decode, so it can't compute over every row and mask the nulls afterwards
/// (the way numeric kernels do). Instead it skips the nulls up front: keep the rows valid in both
/// operands, decode and compute only those, then scatter the results back to their rows and leave
/// every other row null.
///
/// With exactly one constant operand, `bbox_reject` short-circuits rows from bounding rects
/// alone: the constant's rect is fixed once per batch, each valid row's rect is offered to
/// `bbox_reject` before the exact test, and rows it decides never reach `compute`. An operand
/// without a rect (an empty geometry) always falls through to the exact test.
pub(crate) fn execute_null_propagating<T, F>(
    a: &ArrayRef,
    b: &ArrayRef,
    compute: F,
    bbox_reject: Option<BboxReject<T>>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    T: GeoOutput,
    F: Fn(&Geometry<f64>, &Geometry<f64>) -> T + Copy,
{
    let len = a.len();
    let nullability = Nullability::from(a.dtype().is_nullable() || b.dtype().is_nullable());

    // A null constant operand makes every row null (an empty mask builds the all-null output).
    for operand in [a, b] {
        if operand
            .as_opt::<Constant>()
            .is_some_and(|c| c.scalar().is_null())
        {
            return Ok(T::build_array(
                len,
                &Mask::new_false(len),
                vec![],
                nullability,
            ));
        }
    }

    match (a.as_opt::<Constant>(), b.as_opt::<Constant>()) {
        // Both constant: compute once and broadcast across every row.
        (Some(qa), Some(qb)) => {
            let ga = single_geometry(qa.scalar(), ctx)?;
            let gb = single_geometry(qb.scalar(), ctx)?;
            Ok(ConstantArray::new(compute(&ga, &gb).into_scalar(nullability), len).into_array())
        }
        // One constant, one column: fix the constant geometry and evaluate down the column. Its
        // bounding rect is also fixed once, so `bbox_reject` can prove rows from their rects
        // alone and skip the exact test; `zip` disables the pre-check when the kernel has none
        // or the constant has no rect (an empty geometry). The rects go to `bbox_reject` in
        // operand order, like the geometries to `compute`.
        (Some(qa), None) => {
            let ga = single_geometry(qa.scalar(), ctx)?;
            let prescreen = bbox_reject.zip(ga.bounding_rect());
            eval_column(
                b,
                |g| {
                    prescreen
                        .and_then(|(reject, fixed)| reject(&fixed, &g.bounding_rect()?))
                        .unwrap_or_else(|| compute(&ga, g))
                },
                nullability,
                ctx,
            )
        }
        (None, Some(qb)) => {
            let gb = single_geometry(qb.scalar(), ctx)?;
            let prescreen = bbox_reject.zip(gb.bounding_rect());
            eval_column(
                a,
                |g| {
                    prescreen
                        .and_then(|(reject, fixed)| reject(&g.bounding_rect()?, &fixed))
                        .unwrap_or_else(|| compute(g, &gb))
                },
                nullability,
                ctx,
            )
        }
        // Two columns: evaluate row by row.
        (None, None) => {
            vortex_ensure_eq!(
                a.len(),
                b.len(),
                "geo binary: operand length mismatch {} vs {}",
                a.len(),
                b.len()
            );
            eval_column_pair(a, b, compute, nullability, ctx)
        }
    }
}

/// Evaluate `f` over each valid row of one geometry `column`, propagating the column's nulls.
fn eval_column<T, F>(
    column: &ArrayRef,
    f: F,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    T: GeoOutput,
    F: Fn(&Geometry<f64>) -> T,
{
    let len = column.len();
    let valid = column.validity()?.execute_mask(len, ctx)?;
    // Drop the null rows before decoding, since a null row has no geometry to decode. `filter`
    // collapses an all-true mask, so an all-valid column passes through unchanged.
    let decoded = geometries(&column.filter(valid.clone())?, ctx)?;
    let values = decoded.iter().map(f).collect();
    Ok(T::build_array(len, &valid, values, nullability))
}

/// Evaluate `compute` over each row where both geometry columns are valid, propagating the nulls
/// of either column.
fn eval_column_pair<T, F>(
    a: &ArrayRef,
    b: &ArrayRef,
    compute: F,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    T: GeoOutput,
    F: Fn(&Geometry<f64>, &Geometry<f64>) -> T,
{
    let len = a.len();
    let a_present = a.validity()?.execute_mask(len, ctx)?;
    let b_present = b.validity()?.execute_mask(len, ctx)?;
    // A row survives only where both columns are present.
    let valid = &a_present & &b_present;
    // Keep only the rows valid in both columns, so decoding never sees a null geometry. `filter`
    // collapses an all-true mask, so all-valid columns pass through unchanged.
    let ag = geometries(&a.filter(valid.clone())?, ctx)?;
    let bg = geometries(&b.filter(valid.clone())?, ctx)?;
    let values = ag.iter().zip(&bg).map(|(x, y)| compute(x, y)).collect();
    Ok(T::build_array(len, &valid, values, nullability))
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use geo::Contains;
    use geo::Intersects;
    use geo_types::Geometry;
    use vortex_array::ArrayRef;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity;
    use vortex_buffer::BitBuffer;
    use vortex_error::VortexResult;

    use super::BboxReject;
    use super::execute_null_propagating;
    use crate::test_harness::linestring_column;
    use crate::test_harness::nullable_point_column;
    use crate::test_harness::point_column;
    use crate::test_harness::polygon_column;

    /// The `ST_Intersects` rejection: disjoint bounding rects prove no intersection.
    const DISJOINT_REJECTS: BboxReject<bool> = |ra, rb| (!ra.intersects(rb)).then_some(false);

    /// A constant column of length `len`, every row the right triangle `(0,0)-(10,0)-(0,10)`.
    /// Its bounding rect is the `[0, 10]` square, so points in the upper-right half of that
    /// square are inside the rect but outside the triangle.
    fn triangle_constant(len: usize, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        let ring = vec![(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (0.0, 0.0)];
        let single = polygon_column(vec![vec![ring]])?.execute_scalar(0, ctx)?;
        Ok(ConstantArray::new(single, len).into_array())
    }

    /// An intersects test that counts how many rows reach the exact per-row computation.
    fn counting_intersects(
        counter: &Cell<usize>,
    ) -> impl Fn(&Geometry<f64>, &Geometry<f64>) -> bool + Copy {
        move |x, y| {
            counter.set(counter.get() + 1);
            x.intersects(y)
        }
    }

    /// Probes against a constant triangle, one per tier: far outside the bounding rect
    /// (short-circuits to false), inside the rect but outside the triangle (exact test says
    /// false), and inside the triangle (exact test says true). Only the two in-rect probes
    /// reach the exact test.
    #[test]
    fn bbox_reject_skips_exact_test() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let triangle = triangle_constant(3, &mut ctx)?;
        let probes = point_column(vec![50.0, 8.0, 2.0], vec![50.0, 8.0, 2.0])?;

        let exact_runs = Cell::new(0);
        let result = execute_null_propagating(
            &triangle,
            &probes,
            counting_intersects(&exact_runs),
            Some(DISJOINT_REJECTS),
            &mut ctx,
        )?;

        assert_arrays_eq!(result, BoolArray::from_iter([false, false, true]), &mut ctx);
        assert_eq!(exact_runs.get(), 2);
        Ok(())
    }

    /// Null rows are untouched by the pre-check: they stay null and never reach the rect test
    /// or the exact test; valid rows keep their verdicts.
    #[test]
    fn bbox_reject_leaves_nulls_alone() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let triangle = triangle_constant(3, &mut ctx)?;
        let probes = nullable_point_column(vec![Some((50.0, 50.0)), None, Some((2.0, 2.0))])?;

        let exact_runs = Cell::new(0);
        let result = execute_null_propagating(
            &triangle,
            &probes,
            counting_intersects(&exact_runs),
            Some(DISJOINT_REJECTS),
            &mut ctx,
        )?;

        let expected = BoolArray::new(
            BitBuffer::from_iter([false, false, true]),
            Validity::from_iter([true, false, true]),
        )
        .into_array();
        assert_arrays_eq!(result, expected, &mut ctx);
        // The far probe short-circuits and the null row is filtered before decoding, so only
        // the in-triangle probe runs the exact test.
        assert_eq!(exact_runs.get(), 1);
        Ok(())
    }

    /// The rects reach the rejection in operand order even when the constant is the second
    /// operand: a point row's rect never contains the triangle's rect, so every row
    /// short-circuits; with the order flipped, the triangle's rect contains the in-rect
    /// point's and the exact test would run.
    #[test]
    fn bbox_reject_sees_rects_in_operand_order() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let probes = point_column(vec![2.0, 50.0], vec![2.0, 50.0])?;
        let triangle = triangle_constant(2, &mut ctx)?;

        let exact_runs = Cell::new(0);
        let counted = |x: &Geometry<f64>, y: &Geometry<f64>| {
            exact_runs.set(exact_runs.get() + 1);
            x.contains(y)
        };
        let result = execute_null_propagating(
            &probes,
            &triangle,
            counted,
            Some(|ra, rb| (!ra.contains(rb)).then_some(false)),
            &mut ctx,
        )?;

        assert_arrays_eq!(result, BoolArray::from_iter([false, false]), &mut ctx);
        assert_eq!(exact_runs.get(), 0);
        Ok(())
    }

    /// An empty constant geometry has no bounding rect, so the pre-check is disabled: every
    /// valid row falls through to the exact test and none is falsely rejected.
    #[test]
    fn empty_constant_falls_through_to_exact() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let single = linestring_column(vec![vec![]])?.execute_scalar(0, &mut ctx)?;
        let empty = ConstantArray::new(single, 2).into_array();
        let probes = point_column(vec![2.0, 50.0], vec![2.0, 50.0])?;

        let exact_runs = Cell::new(0);
        let result = execute_null_propagating(
            &empty,
            &probes,
            counting_intersects(&exact_runs),
            Some(DISJOINT_REJECTS),
            &mut ctx,
        )?;

        assert_arrays_eq!(result, BoolArray::from_iter([false, false]), &mut ctx);
        assert_eq!(exact_runs.get(), 2);
        Ok(())
    }

    /// Property: the pre-check never changes results — a mixed batch (far, in-rect-but-outside,
    /// inside, null, boundary, rect corner) computed with the rejection equals the same batch
    /// computed with the exact test alone.
    #[test]
    fn bbox_reject_matches_exact_results() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let triangle = triangle_constant(6, &mut ctx)?;
        let probes = nullable_point_column(vec![
            Some((50.0, 50.0)),
            Some((8.0, 8.0)),
            Some((2.0, 2.0)),
            None,
            Some((0.0, 0.0)),
            Some((10.0, 0.0)),
        ])?;
        let exact = |x: &Geometry<f64>, y: &Geometry<f64>| x.intersects(y);

        let with_reject =
            execute_null_propagating(&triangle, &probes, exact, Some(DISJOINT_REJECTS), &mut ctx)?;
        let exact_only = execute_null_propagating(&triangle, &probes, exact, None, &mut ctx)?;

        assert_arrays_eq!(with_reject, exact_only, &mut ctx);
        Ok(())
    }
}
