// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Geometry-value validation at the GeoArrow import boundary.
//!
//! The geometry importers validate the schema first, so coordinate and box columns are already in
//! canonical order here.

use std::ops::Range;

use arrow_array::Array;
use arrow_array::Float64Array;
use arrow_array::ListArray;
use arrow_array::StructArray;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

/// Validate Point coordinates before importing them into native storage.
///
/// - NaN in every ordinate is the empty-Point sentinel.
/// - Otherwise, X and Y must be finite.
/// - Z and M are attributes and are not part of this 2-D validity check.
/// - Child values of null Point rows are ignored.
pub fn validate_point(array: &dyn Array) -> VortexResult<()> {
    let coordinates = array
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| vortex_err!("geo: Point storage must be a coordinate Struct"))?;
    let ordinate_columns = coordinates
        .columns()
        .iter()
        .map(|column| {
            column
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| vortex_err!("geo: coordinate ordinates must be f64"))
        })
        .collect::<VortexResult<Vec<_>>>()?;
    let [x, y, ..] = ordinate_columns.as_slice() else {
        vortex_bail!("geo: coordinates must contain x and y ordinates");
    };

    for_each_non_null_run(coordinates, |rows| {
        let xs = &x.values()[rows.clone()];
        let ys = &y.values()[rows.clone()];

        let valid = if ordinate_columns.len() == 2 {
            // An XY Point is either the all-NaN empty sentinel or a pair of finite ordinates.
            xs.iter().zip(ys).fold(true, |all_valid, (&x, &y)| {
                let empty = x.is_nan() & y.is_nan();
                let finite = x.is_finite() & y.is_finite();
                all_valid & (empty | finite)
            })
        } else {
            rows.fold(true, |all_valid, index| {
                let empty = ordinate_columns.iter().fold(true, |all_nan, ordinate| {
                    all_nan & ordinate.value(index).is_nan()
                });
                let finite = x.value(index).is_finite() & y.value(index).is_finite();
                all_valid & (empty | finite)
            })
        };

        vortex_ensure!(valid, "geo: native Point contains an invalid coordinate");
        Ok(())
    })
}

/// Validate Rect bounds before importing them into native storage.
///
/// - A non-null Rect must have finite, ordered X/Y bounds.
/// - Z and M bounds are not part of this 2-D validity check.
/// - Child values of null Rect rows are ignored.
pub fn validate_rect(array: &dyn Array) -> VortexResult<()> {
    let bounds = array
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| vortex_err!("geo: Rect storage must be a Struct"))?;
    let columns = bounds.columns();
    vortex_ensure!(
        columns.len() >= 4 && columns.len().is_multiple_of(2),
        "geo: Rect storage must contain lower and upper bounds"
    );
    let dimensions = columns.len() / 2;
    let [xmin, ymin, ..] = columns else {
        vortex_bail!("geo: Rect storage must contain x/y bounds");
    };
    let (Some(xmin), Some(ymin), Some(xmax), Some(ymax)) = (
        xmin.as_any().downcast_ref::<Float64Array>(),
        ymin.as_any().downcast_ref::<Float64Array>(),
        columns[dimensions].as_any().downcast_ref::<Float64Array>(),
        columns[dimensions + 1]
            .as_any()
            .downcast_ref::<Float64Array>(),
    ) else {
        vortex_bail!("geo: Rect bounds must be f64");
    };

    for_each_non_null_run(bounds, |rows| {
        let xmin = &xmin.values()[rows.clone()];
        let ymin = &ymin.values()[rows.clone()];
        let xmax = &xmax.values()[rows.clone()];
        let ymax = &ymax.values()[rows];
        let x_bounds = xmin.iter().zip(xmax);
        let y_bounds = ymin.iter().zip(ymax);
        let valid =
            x_bounds
                .zip(y_bounds)
                .fold(true, |all_valid, ((&xmin, &xmax), (&ymin, &ymax))| {
                    all_valid
                        & xmin.is_finite()
                        & ymin.is_finite()
                        & xmax.is_finite()
                        & ymax.is_finite()
                        & (xmin <= xmax)
                        & (ymin <= ymax)
                });
        vortex_ensure!(valid, "geo: native Rect contains invalid x/y bounds");
        Ok(())
    })
}

/// Validate list-based geometry coordinates before importing them into native storage.
///
/// Empty geometries use empty lists, so every coordinate reachable from a non-null outer row must
/// have finite X and Y. Z and M are attributes and are not part of this 2-D validity check.
pub fn validate_list_geometry(array: &dyn Array) -> VortexResult<()> {
    let mut values = array;
    let mut list_levels = Vec::new();
    while let Some(list) = values.as_any().downcast_ref::<ListArray>() {
        list_levels.push(list);
        values = list.values().as_ref();
    }

    let coordinates = values
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| vortex_err!("geo: geometry storage must end with a coordinate Struct"))?;
    let [x, y, ..] = coordinates.columns() else {
        vortex_bail!("geo: coordinates must contain x and y ordinates");
    };
    let (Some(x), Some(y)) = (
        x.as_any().downcast_ref::<Float64Array>(),
        y.as_any().downcast_ref::<Float64Array>(),
    ) else {
        vortex_bail!("geo: coordinate ordinates must be f64");
    };
    let outer = list_levels
        .first()
        .ok_or_else(|| vortex_err!("geo: geometry storage must begin with a List"))?;

    for_each_non_null_run(*outer, |rows| {
        let mut coordinate_rows = rows;
        for list in &list_levels {
            let offsets = list.value_offsets();
            let start = usize::try_from(offsets[coordinate_rows.start])
                .map_err(|_| vortex_err!("geo: list offset exceeds usize"))?;
            let end = usize::try_from(offsets[coordinate_rows.end])
                .map_err(|_| vortex_err!("geo: list offset exceeds usize"))?;
            coordinate_rows = start..end;
        }

        let xs = &x.values()[coordinate_rows.clone()];
        let ys = &y.values()[coordinate_rows];

        // Bitwise boolean reduction keeps the buffer scan branch-free and vectorizable.
        let valid = xs.iter().zip(ys).fold(true, |all_valid, (&x, &y)| {
            all_valid & x.is_finite() & y.is_finite()
        });
        vortex_ensure!(
            valid,
            "geo: native geometry contains a non-finite x/y coordinate"
        );
        Ok(())
    })
}

/// Apply `validate` to contiguous runs of non-null rows.
///
/// Arrow child values beneath a null parent are unspecified, so they must not affect validation.
fn for_each_non_null_run(
    array: &dyn Array,
    mut validate: impl FnMut(Range<usize>) -> VortexResult<()>,
) -> VortexResult<()> {
    let Some(nulls) = array.nulls() else {
        return validate(0..array.len());
    };
    if nulls.null_count() == array.len() {
        return Ok(());
    }
    if nulls.null_count() == 0 {
        return validate(0..array.len());
    }

    nulls
        .valid_slices()
        .try_for_each(|(start, end)| validate(start..end))
}

#[cfg(test)]
mod tests {
    use vortex_array::VortexSessionExecute;
    use vortex_arrow::ArrowSessionExt;
    use vortex_error::VortexResult;

    use super::validate_list_geometry;
    use super::validate_rect;
    use crate::test_harness::geo_session;
    use crate::test_harness::linestring_column;
    use crate::test_harness::multilinestring_column;
    use crate::test_harness::multipoint_column;
    use crate::test_harness::multipolygon_column;
    use crate::test_harness::polygon_column;
    use crate::test_harness::rect_column;

    #[test]
    fn rejects_non_finite_xy_for_every_storage_shape() -> VortexResult<()> {
        let invalid = (f64::NAN, 0.0);
        let arrays = [
            ("LineString", linestring_column(vec![vec![invalid]])?),
            ("MultiPoint", multipoint_column(vec![vec![invalid]])?),
            ("Polygon", polygon_column(vec![vec![vec![invalid]]])?),
            (
                "MultiLineString",
                multilinestring_column(vec![vec![vec![invalid]]])?,
            ),
            (
                "MultiPolygon",
                multipolygon_column(vec![vec![vec![vec![invalid]]]])?,
            ),
        ];
        let session = geo_session();
        let mut ctx = session.create_execution_ctx();

        for (geometry_type, array) in arrays {
            let arrow = session.arrow().execute_arrow(array, None, &mut ctx)?;
            assert!(
                validate_list_geometry(arrow.as_ref()).is_err(),
                "{geometry_type} accepted a NaN x coordinate"
            );
        }

        let rects = rect_column(vec![(f64::NAN, 0.0, 1.0, 1.0)])?;
        let arrow = session.arrow().execute_arrow(rects, None, &mut ctx)?;
        assert!(validate_rect(arrow.as_ref()).is_err());
        Ok(())
    }

    #[test]
    fn rejects_inverted_rect_bounds() -> VortexResult<()> {
        let session = geo_session();
        let mut ctx = session.create_execution_ctx();
        let rects = rect_column(vec![(2.0, 0.0, 1.0, 1.0)])?;
        let arrow = session.arrow().execute_arrow(rects, None, &mut ctx)?;

        assert!(validate_rect(arrow.as_ref()).is_err());
        Ok(())
    }
}
