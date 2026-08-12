// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Differential harness for the spatial scalar functions.
//!
//! # The property
//!
//! Executing a scalar function over generated native geometry columns must equal the `geo`
//! crate applied row by row.
//!
//! The property names no implementation: it holds for today's `geo`-backed kernels and must
//! keep holding as native kernels replace them, so a failure that appears after a swap
//! implicates the new kernel alone.
//!
//! # Layout
//!
//! * [`fixture`]: the case data model, owned geometries materializable as both the native
//!   column and the per-row oracle values.
//! * [`generate`]: the proptest strategies, biased toward structural edge cases.
//! * [`check`]: run the function through a real session and compare every row exactly.
//!
//! # When a case fails
//!
//! proptest shrinks the failure to a minimal counterexample and persists it to
//! `proptest-regressions/`, which every future run replays.

mod check;
mod fixture;
mod generate;

use geo::Distance;
use geo::Euclidean;
use geo::Length;
use geo_types::Geometry;
use proptest::prelude::*;
use vortex_array::IntoArray;

use self::check::check_binary;
use self::check::check_unary;
use self::fixture::Family;
use self::generate::binary_input;
use self::generate::unary_input;
use crate::scalar_fn::distance::SpatialDistance;
use crate::scalar_fn::length::SpatialLength;

proptest! {
    /// `ST_Length` equals `geo`'s Euclidean length row by row. Line strings only, since the
    /// kernel rejects other families at planning time. The kernel is already native, so the
    /// two sides compute independently.
    #[test]
    fn st_length_matches_oracle((column, slice) in unary_input(Family::LineString)) {
        check_unary(
            &column,
            slice,
            |array| Ok(SpatialLength::try_new_array(array)?.into_array()),
            |g| match g {
                Geometry::LineString(line) => Euclidean.length(line),
                _ => unreachable!("generated length inputs are line strings"),
            },
        )
        .map_err(|e| TestCaseError::fail(e.to_string()))?;
    }

    /// `ST_Distance` equals `geo`'s Euclidean distance row by row, over every family pair.
    #[test]
    fn st_distance_matches_oracle(input in binary_input()) {
        check_binary(
            &input,
            |a, b| Ok(SpatialDistance::try_new_array(a, b)?.into_array()),
            |a, b| Euclidean.distance(a, b),
        )
        .map_err(|e| TestCaseError::fail(e.to_string()))?;
    }
}
