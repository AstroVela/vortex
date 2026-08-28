// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::slice::SliceKernel;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
#[cfg(not(codspeed))]
use vortex_array::expr::list_contains;
#[cfg(not(codspeed))]
use vortex_array::expr::lit;
#[cfg(not(codspeed))]
use vortex_array::expr::root;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::list_contains::ListContainsElementKernel;
#[cfg(not(codspeed))]
use vortex_array::test_harness::trace::TraceOptions;
#[cfg(not(codspeed))]
use vortex_array::test_harness::trace::TraceResolution;
#[cfg(not(codspeed))]
use vortex_array::test_harness::trace::trace_op_with;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::BitPacked;
use crate::BitPackedArray;
use crate::BitPackedArrayExt;
use crate::BitPackedData;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    crate::initialize(&session);
    session
});

fn member_list<T>(
    values: impl IntoIterator<Item = Option<T>>,
    member_nullability: Nullability,
) -> Scalar
where
    T: NativePType + Into<PValue>,
{
    let member_dtype = DType::Primitive(T::PTYPE, member_nullability);
    let members = values
        .into_iter()
        .map(|value| {
            value
                .map(|value| Scalar::primitive(value, member_nullability))
                .unwrap_or_else(|| Scalar::null(member_dtype.clone()))
        })
        .collect();
    Scalar::list(Arc::new(member_dtype), members, Nullability::NonNullable)
}

fn list_array(list: Scalar, len: usize) -> ArrayRef {
    ConstantArray::new(list, len).into_array()
}

fn execute_direct(
    list: &ArrayRef,
    element: &BitPackedArray,
    ctx: &mut vortex_array::ExecutionCtx,
) -> VortexResult<BoolArray> {
    <BitPacked as ListContainsElementKernel>::list_contains(list, element.as_view(), ctx)?
        .ok_or_else(|| vortex_err!("BitPacked list_contains kernel declined a supported input"))?
        .execute::<BoolArray>(ctx)
}

macro_rules! integer_type_test {
    ($name:ident, $T:ty, $bit_width:expr) => {
        #[test]
        fn $name() -> VortexResult<()> {
            let mut ctx = SESSION.create_execution_ctx();
            let values = (0..2_048)
                .map(|value| (value % 64) as $T)
                .collect::<Vec<_>>();
            let members = [1 as $T, 3 as $T, 63 as $T];
            let primitive = PrimitiveArray::from_iter(values.iter().copied());
            let packed = BitPackedData::encode(&primitive.into_array(), $bit_width, &mut ctx)?;
            let list = list_array(
                member_list(members.into_iter().map(Some), Nullability::NonNullable),
                packed.len(),
            );

            let actual = execute_direct(&list, &packed, &mut ctx)?;
            let expected =
                BoolArray::from_iter(values.into_iter().map(|value| members.contains(&value)));
            assert_arrays_eq!(actual, expected, &mut ctx);
            Ok(())
        }
    };
}

integer_type_test!(test_integer_type_u8, u8, 6);
integer_type_test!(test_integer_type_u16, u16, 6);
integer_type_test!(test_integer_type_u32, u32, 6);
integer_type_test!(test_integer_type_u64, u64, 6);
integer_type_test!(test_integer_type_i8, i8, 6);
integer_type_test!(test_integer_type_i16, i16, 6);
integer_type_test!(test_integer_type_i32, i32, 6);
integer_type_test!(test_integer_type_i64, i64, 6);

#[rstest]
#[case::empty(vec![])]
#[case::one(vec![3])]
#[case::two(vec![3, 7])]
#[case::three(vec![3, 7, 11])]
#[case::four(vec![3, 7, 11, 15])]
#[case::larger((0..32).map(|value| value * 3).collect())]
#[case::sparse((0..32).map(|value| value * 10_000).collect())]
#[case::duplicates(vec![3, 3, 7, 7, 11, 11, 15, 15, 15])]
fn test_member_cardinalities(#[case] members: Vec<i32>) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = (0..2_048).map(|value| value % 128).collect::<Vec<_>>();
    let primitive = PrimitiveArray::from_iter(values.iter().copied());
    let packed = BitPackedData::encode(&primitive.into_array(), 7, &mut ctx)?;
    let list = list_array(
        member_list(members.iter().copied().map(Some), Nullability::NonNullable),
        packed.len(),
    );

    let actual = execute_direct(&list, &packed, &mut ctx)?;
    let expected = BoolArray::from_iter(values.into_iter().map(|value| members.contains(&value)));
    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

#[test]
fn test_patches() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = (0..2_048)
        .map(|index| {
            if index % 97 == 0 {
                100_000 + index
            } else {
                index % 100
            }
        })
        .collect::<Vec<i32>>();
    let primitive = PrimitiveArray::from_iter(values.iter().copied());
    let packed = BitPackedData::encode(&primitive.into_array(), 7, &mut ctx)?;
    assert!(packed.patches().is_some(), "test setup requires patches");
    let members = [3, 100_097];
    let list = list_array(
        member_list(members.into_iter().map(Some), Nullability::NonNullable),
        packed.len(),
    );

    let actual = execute_direct(&list, &packed, &mut ctx)?;
    let expected = BoolArray::from_iter(values.into_iter().map(|value| members.contains(&value)));
    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

#[test]
fn test_sliced_array() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = (0..5_000).map(|value| value % 128).collect::<Vec<u32>>();
    let primitive = PrimitiveArray::from_iter(values.iter().copied());
    let packed = BitPackedData::encode(&primitive.into_array(), 7, &mut ctx)?;
    let range = 333..4_333;
    let sliced = <BitPacked as SliceKernel>::slice(packed.as_view(), range.clone(), &mut ctx)?
        .ok_or_else(|| vortex_err!("BitPacked slice kernel declined a supported input"))?;
    let members = [1, 63, 127];
    let list = list_array(
        member_list(members.into_iter().map(Some), Nullability::NonNullable),
        sliced.len(),
    );

    let actual = <BitPacked as ListContainsElementKernel>::list_contains(
        &list,
        sliced.as_::<BitPacked>(),
        &mut ctx,
    )?
    .ok_or_else(|| vortex_err!("BitPacked list_contains kernel declined a sliced input"))?
    .execute::<BoolArray>(&mut ctx)?;
    let expected = BoolArray::from_iter(values[range].iter().map(|value| members.contains(value)));
    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

#[test]
fn test_null_needles() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = [Some(1i32), None, Some(2), Some(3), None];
    let primitive = PrimitiveArray::from_option_iter(values);
    let packed = BitPackedData::encode(&primitive.into_array(), 2, &mut ctx)?;
    let list = list_array(
        member_list([Some(1), Some(3)], Nullability::NonNullable),
        packed.len(),
    );

    let actual = execute_direct(&list, &packed, &mut ctx)?;
    let expected = BoolArray::from_iter([Some(true), None, Some(false), Some(true), None]);
    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

#[test]
fn test_null_list() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let primitive = PrimitiveArray::from_iter([1i32, 2, 3]);
    let packed = BitPackedData::encode(&primitive.into_array(), 2, &mut ctx)?;
    let list_dtype = DType::List(
        Arc::new(DType::Primitive(i32::PTYPE, Nullability::NonNullable)),
        Nullability::Nullable,
    );
    let list = list_array(Scalar::null(list_dtype), packed.len());

    let actual = execute_direct(&list, &packed, &mut ctx)?;
    let expected = BoolArray::new(
        [false, false, false].into_iter().collect(),
        Validity::AllInvalid,
    );
    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

#[test]
fn test_nullable_members_are_ignored() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let primitive = PrimitiveArray::from_iter([1i32, 2, 3, 4]);
    let packed = BitPackedData::encode(&primitive.into_array(), 3, &mut ctx)?;
    let list = list_array(
        member_list([Some(1), None, Some(3)], Nullability::Nullable),
        packed.len(),
    );

    let actual = execute_direct(&list, &packed, &mut ctx)?;
    let expected = BoolArray::from_iter([true, false, true, false]);
    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

#[test]
fn test_empty_and_all_null_members_with_null_needles() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = [Some(1i32), None, Some(2)];
    let primitive = PrimitiveArray::from_option_iter(values);
    let packed = BitPackedData::encode(&primitive.into_array(), 2, &mut ctx)?;

    let empty_list = list_array(
        member_list(std::iter::empty::<Option<i32>>(), Nullability::Nullable),
        packed.len(),
    );
    let actual = execute_direct(&empty_list, &packed, &mut ctx)?;
    let expected = BoolArray::from_iter([Some(false), Some(false), Some(false)]);
    assert_arrays_eq!(actual, expected, &mut ctx);

    let all_null_list = list_array(
        member_list([None::<i32>], Nullability::Nullable),
        packed.len(),
    );
    let actual = execute_direct(&all_null_list, &packed, &mut ctx)?;
    let expected = BoolArray::from_iter([Some(false), None, Some(false)]);
    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

#[test]
fn test_wrong_integer_type_declines_without_panic() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let primitive = PrimitiveArray::from_iter([1i32, 2, 3]);
    let packed = BitPackedData::encode(&primitive.into_array(), 2, &mut ctx)?;
    let list = list_array(
        member_list([Some(1i64), Some(3)], Nullability::NonNullable),
        packed.len(),
    );

    let result =
        <BitPacked as ListContainsElementKernel>::list_contains(&list, packed.as_view(), &mut ctx)?;
    assert!(result.is_none());
    Ok(())
}

#[test]
fn test_noninteger_list_declines_without_panic() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let primitive = PrimitiveArray::from_iter([1i32, 2, 3]);
    let packed = BitPackedData::encode(&primitive.into_array(), 2, &mut ctx)?;
    let list = list_array(
        Scalar::list(
            Arc::new(DType::Utf8(Nullability::NonNullable)),
            vec![Scalar::utf8("one", Nullability::NonNullable)],
            Nullability::NonNullable,
        ),
        packed.len(),
    );

    let result =
        <BitPacked as ListContainsElementKernel>::list_contains(&list, packed.as_view(), &mut ctx)?;
    assert!(result.is_none());
    Ok(())
}

#[test]
#[cfg(not(codspeed))]
fn test_registered_kernel_executes_through_expression() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = (0..2_048).map(|value| value % 128).collect::<Vec<i32>>();
    let primitive = PrimitiveArray::from_iter(values.iter().copied());
    let packed = BitPackedData::encode(&primitive.into_array(), 7, &mut ctx)?;
    let members = [0, 99];
    let expression = list_contains(
        lit(member_list(
            members.into_iter().map(Some),
            Nullability::NonNullable,
        )),
        root(),
    );
    let contains = packed.into_array().apply(&expression)?;

    let traced = trace_op_with(
        TraceOptions {
            resolution: TraceResolution::Attempts,
        },
        || contains.execute::<BoolArray>(&mut ctx),
    )?;
    let trace = traced.trace.to_string();
    assert!(trace.contains("parent=vortex.list.contains"), "{trace}");
    assert!(trace.contains("source=session"), "{trace}");

    let expected = BoolArray::from_iter(values.into_iter().map(|value| members.contains(&value)));
    assert_arrays_eq!(traced.output, expected, &mut ctx);
    Ok(())
}
