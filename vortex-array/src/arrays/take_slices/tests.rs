// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::ConstantArray;
use crate::arrays::DictArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::arrays::TakeSlices;
use crate::arrays::TakeSlicesArray;
use crate::arrays::VarBinViewArray;
use crate::assert_arrays_eq;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;
use crate::validity::Validity;

#[test]
fn take_slices_preserves_order_duplicates_and_overlap() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let array = PrimitiveArray::from_iter(0i32..8).into_array();

    let actual = take_slices(&array, &[(4, 2), (1, 2), (4, 2), (2, 3)])?;
    let expected = PrimitiveArray::from_iter([4i32, 5, 1, 2, 4, 5, 2, 3, 4]);

    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

#[test]
fn take_slices_preserves_nullable_child_validity() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let array =
        PrimitiveArray::from_option_iter([Some(0i32), None, Some(2), Some(3), None, Some(5)])
            .into_array();

    let actual = take_slices(&array, &[(1, 3), (4, 2)])?;
    let expected = PrimitiveArray::from_option_iter([None, Some(2), Some(3), None, Some(5)]);

    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

#[test]
fn take_slices_lazy_scalar_and_validity_follow_runs() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let array =
        PrimitiveArray::from_option_iter([Some(0i32), None, Some(2), Some(3), None, Some(5)])
            .into_array();

    let actual = take_slices(&array, &[(1, 3), (4, 2)])?;
    let validity = actual.validity()?.execute_mask(actual.len(), &mut ctx)?;

    assert!(!validity.value(0));
    assert!(validity.value(1));
    assert_eq!(
        actual.execute_scalar(2, &mut ctx)?,
        array.execute_scalar(3, &mut ctx)?
    );
    assert!(!validity.value(3));
    assert_eq!(
        actual.execute_scalar(4, &mut ctx)?,
        array.execute_scalar(5, &mut ctx)?
    );
    Ok(())
}

#[test]
fn take_slices_size_one_child_can_repeat_the_only_range() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let array = PrimitiveArray::from_iter([7i32]).into_array();

    let actual = take_slices(&array, &[(0, 1), (0, 1)])?;
    let expected = PrimitiveArray::from_iter([7i32, 7]);

    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

#[test]
fn take_slices_size_one_nullable_child_can_repeat_null() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let array = PrimitiveArray::from_option_iter([None::<i32>]).into_array();

    let actual = take_slices(&array, &[(0, 1), (0, 1)])?;
    let expected = PrimitiveArray::from_option_iter([None::<i32>, None]);

    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

#[test]
fn take_slices_preserves_all_invalid_validity() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let array =
        PrimitiveArray::new(vortex_buffer::buffer![1i32, 2, 3], Validity::AllInvalid).into_array();

    let actual = take_slices(&array, &[(2, 1), (0, 2)])?;
    let expected = PrimitiveArray::from_option_iter([None::<i32>, None, None]);

    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

#[test]
fn take_slices_construction_defers_out_of_bounds_starts_to_execution() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let array = PrimitiveArray::from_iter(0i32..6).into_array();
    let starts = PrimitiveArray::from_iter([7u64]).into_array();
    let lengths = PrimitiveArray::from_iter([0u64]).into_array();

    let take_slices = TakeSlicesArray::try_new(array, starts, lengths, 0)?.into_array();

    assert!(take_slices.is::<TakeSlices>());
    assert_eq!(take_slices.len(), 0);
    assert!(take_slices.execute::<PrimitiveArray>(&mut ctx).is_err());
    Ok(())
}

#[test]
fn take_slices_accepts_constant_length() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let array = PrimitiveArray::from_iter(0i32..6).into_array();
    let starts = PrimitiveArray::from_iter([0u64, 1, 2]).into_array();
    let lengths = ConstantArray::new(3u64, 3).into_array();

    let actual = TakeSlicesArray::try_new(array, starts, lengths, 9)?.into_array();
    let expected = PrimitiveArray::from_iter([0i32, 1, 2, 1, 2, 3, 2, 3, 4]);

    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

#[test]
fn take_slices_accepts_constant_start() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let array = PrimitiveArray::from_iter(0i32..6).into_array();
    let starts = ConstantArray::new(1u64, 3).into_array();
    let lengths = PrimitiveArray::from_iter([1u64, 3, 5]).into_array();

    let actual = TakeSlicesArray::try_new(array, starts, lengths, 9)?.into_array();
    let expected = PrimitiveArray::from_iter([1i32, 1, 2, 3, 1, 2, 3, 4, 5]);

    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

#[test]
fn take_slices_accepts_constant_start_and_length() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let array = PrimitiveArray::from_iter(0i32..6).into_array();
    let starts = ConstantArray::new(0u64, 3).into_array();
    let lengths = ConstantArray::new(2u64, 3).into_array();

    let actual = TakeSlicesArray::try_new(array, starts, lengths, 6)?.into_array();
    let expected = PrimitiveArray::from_iter([0i32, 1, 0, 1, 0, 1]);

    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

#[test]
fn struct_take_slices_executes_generically() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let array = StructArray::try_new(
        FieldNames::from(["id", "name"]),
        vec![
            PrimitiveArray::from_iter(0i32..6).into_array(),
            VarBinViewArray::from_iter_str(["a", "b", "c", "d", "e", "f"]).into_array(),
        ],
        6,
        Validity::NonNullable,
    )?
    .into_array();

    let actual = take_slices(&array, &[(3, 2), (0, 2)])?;
    let expected = StructArray::try_new(
        FieldNames::from(["id", "name"]),
        vec![
            PrimitiveArray::from_iter([3i32, 4, 0, 1]).into_array(),
            VarBinViewArray::from_iter_str(["d", "e", "a", "b"]).into_array(),
        ],
        4,
        Validity::NonNullable,
    )?;

    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

#[test]
fn take_slices_empty_runs_return_empty_canonical() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let array = PrimitiveArray::from_iter(0i32..6).into_array();

    let actual = take_slices(&array, &[])?;
    let expected = PrimitiveArray::from_iter(std::iter::empty::<i32>());

    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

#[test]
fn take_slices_empty_child_accepts_empty_runs() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let array = PrimitiveArray::empty::<i32>(Nullability::NonNullable).into_array();

    let actual = take_slices(&array, &[])?;
    let expected = PrimitiveArray::from_iter(std::iter::empty::<i32>());

    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

#[test]
fn take_slices_rejects_invalid_ranges_at_the_right_layer() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let array = PrimitiveArray::from_iter(0i32..6).into_array();

    let out_of_bounds_empty = take_slices(&array, &[(7, 0)])?;
    assert!(
        out_of_bounds_empty
            .execute::<PrimitiveArray>(&mut ctx)
            .is_err()
    );

    let out_of_bounds_non_empty = take_slices(&array, &[(4, 3)])?;
    assert!(
        out_of_bounds_non_empty
            .execute::<PrimitiveArray>(&mut ctx)
            .is_err()
    );
    Ok(())
}

#[test]
fn take_slices_execution_rejects_declared_len_mismatch() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let array = PrimitiveArray::from_iter(0i32..6).into_array();
    let starts = PrimitiveArray::from_iter([0u64]).into_array();
    let lengths = PrimitiveArray::from_iter([1u64]).into_array();

    let take_slices = TakeSlicesArray::try_new(array, starts, lengths, 0)?.into_array();

    assert!(take_slices.execute::<PrimitiveArray>(&mut ctx).is_err());
    Ok(())
}

#[test]
fn take_slices_of_take_slices_executes_correctly() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let array = PrimitiveArray::from_iter(0i32..10).into_array();

    let inner = take_slices(&array, &[(2, 3), (7, 3)])?;
    let actual = take_slices(&inner, &[(1, 3), (4, 2)])?;

    assert_arrays_eq!(
        actual,
        PrimitiveArray::from_iter([3i32, 4, 7, 8, 9]),
        &mut ctx
    );
    Ok(())
}

#[test]
fn take_slices_generic_execution_handles_child_without_take_slices_kernel() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let dict = DictArray::try_new(
        vortex_buffer::buffer![2u8, 0, 1, 2, 0, 1].into_array(),
        PrimitiveArray::from_iter([10i32, 20, 30]).into_array(),
    )?
    .into_array();

    let actual = take_slices(&dict, &[(1, 3), (0, 2)])?;
    let expected = PrimitiveArray::from_iter([10i32, 20, 30, 30, 10]);

    assert!(actual.is::<TakeSlices>());
    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

#[test]
fn take_slices_generic_execution_preserves_nullable_encoded_child() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let dict = DictArray::try_new(
        vortex_buffer::buffer![0u8, 1, 2, 0].into_array(),
        PrimitiveArray::from_option_iter([Some(10i32), None, Some(30)]).into_array(),
    )?
    .into_array();

    let actual = take_slices(&dict, &[(1, 2), (0, 2)])?;
    let expected = PrimitiveArray::from_option_iter([None, Some(30), Some(10), None]);

    assert!(actual.is::<TakeSlices>());
    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

fn take_slices(array: &ArrayRef, runs: &[(usize, usize)]) -> VortexResult<ArrayRef> {
    let len = runs.iter().try_fold(0usize, |acc, &(_, length)| {
        acc.checked_add(length)
            .ok_or_else(|| vortex_err!("TakeSlicesArray length overflow"))
    })?;
    let starts = runs
        .iter()
        .map(|&(start, _)| start as u64)
        .collect::<Vec<_>>();
    let lengths = runs
        .iter()
        .map(|&(_, length)| length as u64)
        .collect::<Vec<_>>();
    TakeSlicesArray::try_new(
        array.clone(),
        PrimitiveArray::from_iter(starts).into_array(),
        PrimitiveArray::from_iter(lengths).into_array(),
        len,
    )
    .map(IntoArray::into_array)
}
