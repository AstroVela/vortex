// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use smallvec::smallvec;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use super::common::create_basic_fsl;
use super::common::create_empty_fsl;
use super::common::create_large_fsl;
use super::common::create_nullable_fsl;
use super::common::create_single_element_fsl;
use crate::ArrayRef;
use crate::ArrayView;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::EmptyArrayData;
use crate::array::OperationsVTable;
use crate::array::VTable;
use crate::array::ValidityVTable;
use crate::array::with_empty_buffers;
use crate::array_session;
use crate::arrays::FixedSizeList;
use crate::arrays::FixedSizeListArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::TakeSlices;
use crate::arrays::dict::TakeExecute;
use crate::arrays::fixed_size_list::FixedSizeListArrayExt;
use crate::assert_arrays_eq;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::builders::FixedSizeListBuilder;
use crate::compute::conformance::take::test_take_conformance;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionResult;
use crate::scalar::Scalar;
use crate::serde::ArrayChildren;
use crate::validity::Validity;

// Conformance tests for common take scenarios.
#[rstest]
#[case::basic(create_basic_fsl())]
#[case::nullable(create_nullable_fsl())]
#[case::large(create_large_fsl())]
#[case::single_element(create_single_element_fsl())]
#[case::empty(create_empty_fsl())]
fn test_take_fsl_conformance(#[case] fsl: FixedSizeListArray) {
    test_take_conformance(
        &fsl.into_array(),
        &mut array_session().create_execution_ctx(),
    );
}

// FSL-specific edge case tests that aren't covered by conformance.

#[test]
fn test_take_basic_smoke_test() {
    let mut ctx = array_session().create_execution_ctx();
    let elements = buffer![1i32, 2, 3, 4, 5, 6].into_array();
    let fsl = FixedSizeListArray::new(elements.into_array(), 2, Validity::NonNullable, 3);

    let indices = buffer![2u32, 0, 1].into_array();
    let result = fsl.take(indices).unwrap();

    // Expected: [[5,6], [1,2], [3,4]]
    let expected = FixedSizeListArray::new(
        buffer![5i32, 6, 1, 2, 3, 4].into_array(),
        2,
        Validity::NonNullable,
        3,
    );
    assert_arrays_eq!(expected, result, &mut ctx);
}

// Parameterized test for FSL-specific degenerate (list_size=0) cases.
#[rstest]
#[case::degenerate_non_null(
    Validity::NonNullable,
    vec![Some(3u32), Some(1), Some(4), Some(0), Some(2)],
    5,
    vec![false; 5]
)]
#[case::degenerate_with_nulls(
    Validity::from_iter([true, false, true, true, false]),
    vec![Some(1u32), Some(3), None, Some(0)],
    4,
    vec![true, false, true, false]
)]
#[case::degenerate_all_null(
    Validity::AllInvalid,
    vec![Some(2u32), Some(0), Some(1)],
    3,
    vec![true, true, true]
)]
fn test_take_degenerate_lists(
    #[case] validity: Validity,
    #[case] indices: Vec<Option<u32>>,
    #[case] expected_len: usize,
    #[case] expected_nulls: Vec<bool>,
) {
    // Create a degenerate FSL array with list_size = 0.
    // This is a specific edge case for FSL where lists have no elements.
    let elements = PrimitiveArray::empty::<i32>(Nullability::NonNullable);
    let fsl = FixedSizeListArray::new(elements.into_array(), 0, validity, 5);

    test_take_conformance(
        &fsl.clone().into_array(),
        &mut array_session().create_execution_ctx(),
    );

    // Also test the specific behavior.
    let indices_array = PrimitiveArray::from_option_iter(indices);
    let result = fsl.take(indices_array.into_array()).unwrap();

    assert_eq!(result.len(), expected_len);
    for (i, expected_null) in expected_nulls.iter().enumerate() {
        assert_eq!(
            result
                .execute_scalar(i, &mut array_session().create_execution_ctx())
                .unwrap()
                .is_null(),
            *expected_null
        );
    }
}

#[test]
fn test_take_degenerate_rejects_out_of_bounds_valid_index() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let elements = PrimitiveArray::empty::<i32>(Nullability::NonNullable).into_array();
    let fsl = FixedSizeListArray::new(elements, 0, Validity::NonNullable, 5);
    let indices = buffer![5u32].into_array();

    let result = <FixedSizeList as TakeExecute>::take(fsl.as_view(), &indices, &mut ctx);

    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_take_degenerate_ignores_out_of_bounds_null_index_payload() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let elements = PrimitiveArray::empty::<i32>(Nullability::NonNullable).into_array();
    let fsl = FixedSizeListArray::new(elements, 0, Validity::NonNullable, 5);
    let indices =
        PrimitiveArray::new(buffer![999u32, 1], Validity::from_iter([false, true])).into_array();

    let result = <FixedSizeList as TakeExecute>::take(fsl.as_view(), &indices, &mut ctx)?
        .ok_or_else(|| vortex_err!("FixedSizeList TakeExecute returned no result"))?;

    assert_eq!(result.len(), 2);
    assert!(result.execute_scalar(0, &mut ctx)?.is_null());
    assert!(!result.execute_scalar(1, &mut ctx)?.is_null());
    Ok(())
}

#[test]
fn test_take_large_list_size() {
    let mut ctx = array_session().create_execution_ctx();
    let elements = buffer![0i32..300].into_array();
    let fsl = FixedSizeListArray::new(elements, 100, Validity::NonNullable, 3);

    let indices = buffer![2u16, 0].into_array();
    let result = fsl.take(indices).unwrap();

    // Expected: [[200..300], [0..100]]
    let expected_elems = PrimitiveArray::from_iter((200i32..300).chain(0..100)).into_array();
    let expected = FixedSizeListArray::new(expected_elems, 100, Validity::NonNullable, 2);
    assert_arrays_eq!(expected, result, &mut ctx);
}

#[test]
fn test_take_range_path_large_list_size_non_nullable() {
    let mut ctx = array_session().create_execution_ctx();
    let elements = PrimitiveArray::from_iter(0i32..768).into_array();
    let fsl = FixedSizeListArray::new(elements, 256, Validity::NonNullable, 3);

    let indices = buffer![2u16, 0].into_array();
    let result = fsl.take(indices).unwrap();

    let expected_elems = PrimitiveArray::from_iter((512i32..768).chain(0..256)).into_array();
    let expected = FixedSizeListArray::new(expected_elems, 256, Validity::NonNullable, 2);
    assert_arrays_eq!(expected, result, &mut ctx);
}

#[test]
fn test_take_range_path_large_list_size_nullable() {
    let mut ctx = array_session().create_execution_ctx();
    let elements = PrimitiveArray::from_iter(0i32..768).into_array();
    let fsl = FixedSizeListArray::new(elements, 256, Validity::from_iter([true, false, true]), 3);

    let indices = buffer![2u16, 1, 0].into_array();
    let result = fsl.take(indices).unwrap();

    let expected_elems =
        PrimitiveArray::from_iter((512i32..768).chain((0..256).map(|_| 0)).chain(0..256))
            .into_array();
    let expected = FixedSizeListArray::new(
        expected_elems,
        256,
        Validity::from_iter([true, false, true]),
        3,
    );
    assert_arrays_eq!(expected, result, &mut ctx);
}

#[test]
fn test_take_fsl_with_null_indices_preserves_elements() {
    let mut ctx = array_session().create_execution_ctx();
    let elements = buffer![1i32, 2, 3, 4, 5, 6].into_array();
    let fsl = FixedSizeListArray::new(elements.into_array(), 2, Validity::NonNullable, 3);

    // Indices with nulls: [1, null, 0].
    let indices = PrimitiveArray::from_option_iter([Some(1u32), None, Some(0)]);
    let result = fsl.take(indices.into_array()).unwrap();

    // Expected: [[3,4], null, [1,2]]
    let expected = FixedSizeListArray::new(
        buffer![3i32, 4, 0, 0, 1, 2].into_array(),
        2,
        Validity::from_iter([true, false, true]),
        3,
    );
    assert_arrays_eq!(expected, result, &mut ctx);
}

#[test]
fn test_take_non_nullable_fsl_nullable_indices_makes_nullable_output() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let elements = buffer![1i32, 2, 3, 4, 5, 6].into_array();
    let fsl = FixedSizeListArray::new(elements.into_array(), 2, Validity::NonNullable, 3);

    let indices = PrimitiveArray::from_option_iter([Some(2u32), None, Some(0)]);
    let result = fsl.take(indices.into_array())?;

    assert_eq!(result.dtype().nullability(), Nullability::Nullable);
    assert!(!result.execute_scalar(0, &mut ctx)?.is_null());
    assert!(result.execute_scalar(1, &mut ctx)?.is_null());
    assert!(!result.execute_scalar(2, &mut ctx)?.is_null());
    Ok(())
}

// List offsets must not truncate when small index types select large lists.
#[rstest]
#[case::non_nullable(
    FixedSizeListArray::new(
        PrimitiveArray::from_iter(0u32..320).into_array(), 16, Validity::NonNullable, 20,
    ),
    buffer![0u8, 16, 5].into_array(),
    FixedSizeListArray::new(
        PrimitiveArray::from_iter((0u32..16).chain(256..272).chain(80..96)).into_array(),
        16, Validity::NonNullable, 3,
    ),
)]
#[case::nullable(
    FixedSizeListArray::new(
        PrimitiveArray::from_iter(0u32..320).into_array(), 16,
        Validity::from_iter((0..20).map(|i| i != 5)), 20,
    ),
    buffer![0u8, 16, 5].into_array(),
    FixedSizeListArray::new(
        PrimitiveArray::from_iter((0u32..16).chain(256..272).chain(80..96)).into_array(),
        16, Validity::from_iter([true, true, false]), 3,
    ),
)]
fn test_element_index_overflow(
    #[case] fsl: FixedSizeListArray,
    #[case] indices: ArrayRef,
    #[case] expected: FixedSizeListArray,
) {
    let mut ctx = array_session().create_execution_ctx();
    let result = fsl.take(indices).unwrap();
    assert_arrays_eq!(result, expected, &mut ctx);
}

#[test]
fn test_take_nullable_indices_ignores_out_of_bounds_null_value() {
    let mut ctx = array_session().create_execution_ctx();
    let elements = buffer![1i32, 2, 3, 4, 5, 6].into_array();
    let fsl = FixedSizeListArray::new(elements.into_array(), 2, Validity::NonNullable, 3);

    let indices = PrimitiveArray::new(
        buffer![1u64, 999, 0],
        Validity::from_iter([true, false, true]),
    );
    let result = fsl.take(indices.into_array()).unwrap();

    let expected = FixedSizeListArray::new(
        buffer![3i32, 4, 0, 0, 1, 2].into_array(),
        2,
        Validity::from_iter([true, false, true]),
        3,
    );
    assert_arrays_eq!(expected, result, &mut ctx);
}

#[test]
fn test_take_rejects_overflowing_valid_index() {
    let mut ctx = array_session().create_execution_ctx();
    let elements = buffer![1i32, 2, 3, 4].into_array();
    let fsl = FixedSizeListArray::new(elements.into_array(), 2, Validity::NonNullable, 2);
    let overflowing_index = (usize::MAX / 2 + 1) as u64;
    let indices = buffer![overflowing_index].into_array();

    let result = <FixedSizeList as TakeExecute>::take(fsl.as_view(), &indices, &mut ctx);

    assert!(result.is_err());
}

#[test]
fn test_take_nullable_fsl_with_nullable_indices() {
    let mut ctx = array_session().create_execution_ctx();
    let elements = buffer![1i32, 2, 3, 4, 5, 6].into_array();
    let fsl = FixedSizeListArray::new(
        elements.into_array(),
        2,
        Validity::from_iter([true, false, true]),
        3,
    );

    let indices = PrimitiveArray::new(
        buffer![2u64, 999, 1, 0],
        Validity::from_iter([true, false, true, true]),
    );
    let result = fsl.take(indices.into_array()).unwrap();

    let expected = FixedSizeListArray::new(
        buffer![5i32, 6, 0, 0, 0, 0, 1, 2].into_array(),
        2,
        Validity::from_iter([true, false, false, true]),
        4,
    );
    assert_arrays_eq!(expected, result, &mut ctx);
}

#[test]
fn test_take_empty_source_with_all_null_indices() {
    let fsl = create_empty_fsl();
    let indices = PrimitiveArray::new(buffer![999u64, 123], Validity::AllInvalid);

    let result = fsl.take(indices.into_array()).unwrap();

    assert_eq!(result.len(), 2);
    for idx in 0..result.len() {
        assert!(
            result
                .execute_scalar(idx, &mut array_session().create_execution_ctx())
                .unwrap()
                .is_null()
        );
    }
}

#[test]
fn test_take_execute_empty_source_all_null_indices_builds_default_elements() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let fsl = create_empty_fsl();
    let indices = PrimitiveArray::new(buffer![999u64, 123], Validity::AllInvalid).into_array();

    let result = <FixedSizeList as TakeExecute>::take(fsl.as_view(), &indices, &mut ctx)?
        .ok_or_else(|| vortex_err!("FixedSizeList TakeExecute returned no result"))?;
    let result_fsl = result.as_::<FixedSizeList>();

    assert_eq!(
        result_fsl.elements().len(),
        result.len() * result_fsl.list_size() as usize
    );
    for idx in 0..result.len() {
        assert!(result.execute_scalar(idx, &mut ctx)?.is_null());
    }
    Ok(())
}

#[test]
fn test_take_empty_source_rejects_valid_index_after_null_index() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let fsl = create_empty_fsl();
    let indices =
        PrimitiveArray::new(buffer![999u64, 0], Validity::from_iter([false, true])).into_array();

    let result = <FixedSizeList as TakeExecute>::take(fsl.as_view(), &indices, &mut ctx);

    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_take_uses_take_slices_for_encoded_elements_child() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let encoded_elements = NoTakeSlicesArray::wrap(buffer![10i32, 20, 30, 40, 20, 30].into_array());
    let fsl = FixedSizeListArray::new(encoded_elements, 2, Validity::NonNullable, 3);
    let indices = buffer![2u8, 0].into_array();

    let result = <FixedSizeList as TakeExecute>::take(fsl.as_view(), &indices, &mut ctx)?
        .ok_or_else(|| vortex_err!("FixedSizeList TakeExecute returned no result"))?;

    assert!(result.as_::<FixedSizeList>().elements().is::<TakeSlices>());
    let expected = FixedSizeListArray::new(
        PrimitiveArray::from_iter([20i32, 30, 10, 20]).into_array(),
        2,
        Validity::NonNullable,
        2,
    );
    assert_arrays_eq!(expected, result, &mut ctx);
    Ok(())
}

#[derive(Clone, Debug)]
struct NoTakeSlicesArray;

impl NoTakeSlicesArray {
    fn wrap(child: ArrayRef) -> ArrayRef {
        let dtype = child.dtype().clone();
        let len = child.len();

        // SAFETY: `NoTakeSlicesArray` has one child with matching dtype and length, and no
        // top-level metadata beyond `EmptyArrayData`.
        unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(NoTakeSlicesArray, dtype, len, EmptyArrayData)
                    .with_slots(smallvec![Some(child)]),
            )
        }
        .into_array()
    }
}

impl VTable for NoTakeSlicesArray {
    type TypedArrayData = EmptyArrayData;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.test.no_take_slices");
        *ID
    }

    fn validate(
        &self,
        _data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == 1,
            "NoTakeSlicesArray expected one child slot"
        );
        let child = slots[0]
            .as_ref()
            .ok_or_else(|| vortex_err!("NoTakeSlicesArray child slot must be present"))?;
        vortex_ensure!(
            child.dtype() == dtype,
            "NoTakeSlicesArray child dtype {} does not match outer dtype {}",
            child.dtype(),
            dtype
        );
        vortex_ensure!(
            child.len() == len,
            "NoTakeSlicesArray child length {} does not match outer length {}",
            child.len(),
            len
        );
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, _idx: usize) -> BufferHandle {
        vortex_panic!("NoTakeSlicesArray has no buffers")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        with_empty_buffers(self, array, buffers)
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        match idx {
            0 => "child".to_string(),
            _ => vortex_panic!("NoTakeSlicesArray slot index {idx} out of bounds"),
        }
    }

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        vortex_bail!("NoTakeSlicesArray is not serializable")
    }

    fn deserialize(
        &self,
        _dtype: &DType,
        _len: usize,
        _metadata: &[u8],
        _buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_bail!("NoTakeSlicesArray is not serializable")
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(array.slots()[0].clone().ok_or_else(
            || vortex_err!("NoTakeSlicesArray child slot must be present"),
        )?))
    }
}

impl OperationsVTable<NoTakeSlicesArray> for NoTakeSlicesArray {
    fn scalar_at(
        array: ArrayView<'_, NoTakeSlicesArray>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        array.as_ref().slots()[0]
            .as_ref()
            .ok_or_else(|| vortex_err!("NoTakeSlicesArray child slot must be present"))?
            .execute_scalar(index, ctx)
    }
}

impl ValidityVTable<NoTakeSlicesArray> for NoTakeSlicesArray {
    fn validity(array: ArrayView<'_, NoTakeSlicesArray>) -> VortexResult<Validity> {
        array.as_ref().slots()[0]
            .as_ref()
            .ok_or_else(|| vortex_err!("NoTakeSlicesArray child slot must be present"))?
            .validity()
    }
}

// Parameterized test for nullable array scenarios that are specific to FSL's implementation.
#[rstest]
#[case::nullable_mixed_elements(
    vec![Some(vec![1i32, 2]), None, Some(vec![5, 6])],
    vec![Some(2u32), Some(1), Some(0)],
    vec![Some(vec![5i32, 6]), None, Some(vec![1, 2])]
)]
#[case::nullable_with_null_indices(
    vec![Some(vec![1i32, 2]), None, Some(vec![5, 6])],
    vec![Some(0u32), None, Some(1), Some(2)],
    vec![Some(vec![1i32, 2]), None, None, Some(vec![5, 6])]
)]
fn test_take_nullable_arrays_fsl_specific(
    #[case] array_values: Vec<Option<Vec<i32>>>,
    #[case] indices: Vec<Option<u32>>,
    #[case] expected_values: Vec<Option<Vec<i32>>>,
) {
    let mut ctx = array_session().create_execution_ctx();
    let fsl = nullable_i32_fsl(array_values);

    let indices_array = PrimitiveArray::from_option_iter(indices);
    let result = fsl.take(indices_array.into_array()).unwrap();
    let expected = nullable_i32_fsl(expected_values);

    assert_arrays_eq!(expected, result, &mut ctx);
}

fn nullable_i32_fsl(array_values: Vec<Option<Vec<i32>>>) -> ArrayRef {
    let list_size = if let Some(Some(first)) = array_values.first() {
        u32::try_from(first.len()).unwrap()
    } else {
        2
    };

    let mut builder = FixedSizeListBuilder::with_capacity(
        DType::Primitive(PType::I32, Nullability::NonNullable).into(),
        list_size,
        Nullability::Nullable,
        array_values.len(),
    );

    for value in array_values {
        match value {
            Some(list) => {
                let scalars: Vec<Scalar> = list.into_iter().map(|v| v.into()).collect();
                builder
                    .append_value(
                        Scalar::list(
                            DType::Primitive(PType::I32, Nullability::NonNullable),
                            scalars,
                            Nullability::NonNullable,
                        )
                        .as_list(),
                    )
                    .unwrap();
            }
            None => builder.append_null(),
        }
    }

    builder.finish()
}
