// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execution and comparison: run a scalar function over a generated case through a real
//! session, then assert the result column equals its `geo` oracle values. Comparison is
//! exact, with no epsilon: kernel and oracle run the same floating-point operations, so any
//! drift is a real behavior change and must fail.

use std::ops::Range;

use geo_types::Geometry;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_error::VortexResult;

use super::fixture::BinaryInput;
use super::fixture::ConstSide;
use super::fixture::GeometryColumn;
use crate::tests::SESSION;

/// Check one unary scalar function over a (possibly sliced) column against its oracle.
pub(super) fn check_unary(
    column: &GeometryColumn,
    slice: Range<usize>,
    build: impl Fn(ArrayRef) -> VortexResult<ArrayRef>,
    oracle: impl Fn(&Geometry<f64>) -> f64,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let result = build(column.to_array()?.slice(slice.clone())?)?;
    let expected =
        PrimitiveArray::from_iter(column.rows[slice].iter().map(|row| oracle(&row.oracle())))
            .into_array();
    assert_arrays_eq!(result, expected, &mut ctx);
    Ok(())
}

/// Check one binary scalar function against its oracle, exercising the column and constant
/// operand shapes.
pub(super) fn check_binary(
    input: &BinaryInput,
    build: impl Fn(ArrayRef, ArrayRef) -> VortexResult<ArrayRef>,
    oracle: impl Fn(&Geometry<f64>, &Geometry<f64>) -> f64,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let len = input.a.rows.len();
    let (a_array, b_array) = (input.a.to_array()?, input.b.to_array()?);

    // Collapse one side to a constant of one of its rows, keeping the substituted row index
    // for the expectation below.
    let (a_array, b_array, a_const, b_const) = match input.constant {
        ConstSide::Neither => (a_array, b_array, None, None),
        ConstSide::Left(row) => {
            let scalar = a_array.execute_scalar(row, &mut ctx)?;
            let constant = ConstantArray::new(scalar, len).into_array();
            (constant, b_array, Some(row), None)
        }
        ConstSide::Right(row) => {
            let scalar = b_array.execute_scalar(row, &mut ctx)?;
            let constant = ConstantArray::new(scalar, len).into_array();
            (a_array, constant, None, Some(row))
        }
    };
    let result = build(a_array, b_array)?;

    let (a_rows, b_rows) = (input.a.oracle_rows(), input.b.oracle_rows());
    let expected = PrimitiveArray::from_iter((0..len).map(|row| {
        oracle(
            &a_rows[a_const.unwrap_or(row)],
            &b_rows[b_const.unwrap_or(row)],
        )
    }))
    .into_array();
    assert_arrays_eq!(result, expected, &mut ctx);
    Ok(())
}
