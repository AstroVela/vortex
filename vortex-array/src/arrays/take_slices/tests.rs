// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::DictArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::TakeSlices;
use crate::arrays::TakeSlicesArray;
use crate::arrays::take_slices::TakeSlicesArrayExt;
use crate::arrays::take_slices::TakeSlicesExecuteAdaptor;
use crate::assert_arrays_eq;
use crate::dtype::Nullability;
use crate::kernel::ExecuteParentKernel;
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
fn take_slices_one_slice_reduces_to_child_slice() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let array = PrimitiveArray::from_iter(0i32..6).into_array();

    let actual = take_slices(&array, &[(1, 4)])?;

    assert!(actual.is::<Primitive>());
    assert!(!actual.is::<TakeSlices>());
    assert_arrays_eq!(actual, PrimitiveArray::from_iter([1i32, 2, 3, 4]), &mut ctx);
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
    let array = PrimitiveArray::new(buffer![1i32, 2, 3], Validity::AllInvalid).into_array();

    let actual = take_slices(&array, &[(2, 1), (0, 2)])?;
    let expected = PrimitiveArray::from_option_iter([None::<i32>, None, None]);

    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

#[test]
fn primitive_take_slices_execute_parent_copies_runs_directly() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let array = PrimitiveArray::from_iter(0i32..8).into_array();
    let (starts, lengths) = selector_arrays(&[(3, 3), (0, 2)])?;
    let parent = TakeSlicesArray::try_new(array.clone(), starts, lengths)?.into_array();

    let actual = TakeSlicesExecuteAdaptor(Primitive)
        .execute_parent(
            array.as_::<Primitive>(),
            parent.as_::<TakeSlices>(),
            0,
            &mut ctx,
        )?
        .ok_or_else(|| vortex_err!("Primitive TakeSlicesExecute declined multi-slice take"))?;

    assert!(actual.is::<Primitive>());
    assert_arrays_eq!(
        actual,
        PrimitiveArray::from_iter([3i32, 4, 5, 0, 1]),
        &mut ctx
    );
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
fn take_slices_rejects_invalid_runs() -> VortexResult<()> {
    let array = PrimitiveArray::from_iter(0i32..6).into_array();

    assert!(take_slices(&array, &[(7, 0)]).is_err());
    assert!(take_slices(&array, &[(4, 3)]).is_err());
    let starts = PrimitiveArray::from_iter([0u64, 1]).into_array();
    let lengths = PrimitiveArray::from_iter([1u64]).into_array();
    assert!(array.take_slices(starts, lengths).is_err());
    Ok(())
}

#[test]
fn take_slices_of_take_slices_projects_to_original_child() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let array = PrimitiveArray::from_iter(0i32..10).into_array();

    let inner = take_slices(&array, &[(2, 3), (7, 3)])?;
    let actual = take_slices(&inner, &[(1, 3), (4, 2)])?;

    let actual_take_slices = actual.as_::<TakeSlices>();
    assert!(actual_take_slices.child().is::<Primitive>());
    assert_eq!(
        selector_values(actual_take_slices.starts(), &mut ctx)?,
        vec![3, 7, 8]
    );
    assert_eq!(
        selector_values(actual_take_slices.lengths(), &mut ctx)?,
        vec![2, 1, 2]
    );
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
        buffer![2u8, 0, 1, 2, 0, 1].into_array(),
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
        buffer![0u8, 1, 2, 0].into_array(),
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
    let (starts, lengths) = selector_arrays(runs)?;
    array.take_slices(starts, lengths)
}

fn selector_arrays(runs: &[(usize, usize)]) -> VortexResult<(ArrayRef, ArrayRef)> {
    let starts = runs
        .iter()
        .map(|&(start, _)| selector_value(start, "start"))
        .collect::<VortexResult<Vec<_>>>()?;
    let lengths = runs
        .iter()
        .map(|&(_, length)| selector_value(length, "length"))
        .collect::<VortexResult<Vec<_>>>()?;
    Ok((
        PrimitiveArray::from_iter(starts).into_array(),
        PrimitiveArray::from_iter(lengths).into_array(),
    ))
}

fn selector_value(value: usize, name: &str) -> VortexResult<u64> {
    u64::try_from(value)
        .map_err(|_| vortex_err!("test {name} selector {value} does not fit in u64"))
}

fn selector_values(selector: &ArrayRef, ctx: &mut crate::ExecutionCtx) -> VortexResult<Vec<u64>> {
    let selector = selector.clone().execute::<PrimitiveArray>(ctx)?;
    Ok(selector.as_slice::<u64>().to_vec())
}
