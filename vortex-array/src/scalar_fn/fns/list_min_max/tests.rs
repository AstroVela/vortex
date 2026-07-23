// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::cast_possible_truncation, clippy::cognitive_complexity)]

use std::sync::Arc;

use prost::Message;
use rstest::rstest;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_proto::expr as pb;

use super::ListMax;
use super::ListMin;
use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::array_session;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::DecimalArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::ListArray;
use crate::arrays::ListViewArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBinArray;
use crate::assert_arrays_eq;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::half::f16;
use crate::expr::Expression;
use crate::expr::list_max;
use crate::expr::list_max_opts;
use crate::expr::list_min;
use crate::expr::list_min_opts;
use crate::expr::proto::ExprSerializeProtoExt;
use crate::expr::root;
use crate::scalar::Scalar;
use crate::scalar_fn::ScalarFnVTable;
use crate::validity::Validity;

fn create_list_elements() -> ArrayRef {
    PrimitiveArray::from_option_iter::<i32, _>([
        Some(1),
        Some(2),
        Some(3),
        Some(4),
        Some(5),
        Some(6),
        None,
    ])
    .into_array()
}

fn assert_primitive_extrema(list: &ArrayRef) -> VortexResult<()> {
    let minimum = list.clone().apply(&list_min(root()))?;
    let maximum = list.clone().apply(&list_max(root()))?;
    let mut ctx = array_session().create_execution_ctx();

    let expected_min =
        PrimitiveArray::from_option_iter::<i32, _>([Some(1), Some(3), None, Some(6)]);
    let expected_max =
        PrimitiveArray::from_option_iter::<i32, _>([Some(2), Some(5), None, Some(6)]);
    assert_arrays_eq!(minimum, expected_min, &mut ctx);
    assert_arrays_eq!(maximum, expected_max, &mut ctx);
    Ok(())
}

#[rstest]
#[case(buffer![0u32, 2, 5, 5, 7].into_array())]
#[case(buffer![0u64, 2, 5, 5, 7].into_array())]
fn list_extrema(#[case] offsets: ArrayRef) -> VortexResult<()> {
    let list =
        ListArray::try_new(create_list_elements(), offsets, Validity::NonNullable)?.into_array();
    assert_primitive_extrema(&list)
}

#[test]
fn nullable_lists_and_elements() -> VortexResult<()> {
    let list = ListArray::try_new(
        create_list_elements(),
        buffer![0u32, 2, 5, 5, 7].into_array(),
        Validity::Array(BoolArray::from_iter([true, false, true, false]).into_array()),
    )?
    .into_array();
    let minimum = list.clone().apply(&list_min(root()))?;
    let maximum = list.apply(&list_max(root()))?;
    let mut ctx = array_session().create_execution_ctx();

    let expected = PrimitiveArray::from_option_iter::<i32, _>([Some(1), None, None, None]);
    assert_arrays_eq!(minimum, expected, &mut ctx);
    let expected = PrimitiveArray::from_option_iter::<i32, _>([Some(2), None, None, None]);
    assert_arrays_eq!(maximum, expected, &mut ctx);
    Ok(())
}

#[test]
fn listview_extrema() -> VortexResult<()> {
    let list_view = ListViewArray::new(
        create_list_elements(),
        buffer![5u32, 0, 4, 1].into_array(),
        buffer![2u32, 3, 0, 2].into_array(),
        Validity::NonNullable,
    )
    .into_array();
    let minimum = list_view.clone().apply(&list_min(root()))?;
    let maximum = list_view.apply(&list_max(root()))?;
    let mut ctx = array_session().create_execution_ctx();

    // The views are [6, null], [1, 2, 3], [], and [2, 3].
    let expected_min =
        PrimitiveArray::from_option_iter::<i32, _>([Some(6), Some(1), None, Some(2)]);
    let expected_max =
        PrimitiveArray::from_option_iter::<i32, _>([Some(6), Some(3), None, Some(3)]);
    assert_arrays_eq!(minimum, expected_min, &mut ctx);
    assert_arrays_eq!(maximum, expected_max, &mut ctx);
    Ok(())
}

#[test]
fn listview_extrema_across_validity_words() -> VortexResult<()> {
    let elements = PrimitiveArray::from_option_iter::<i32, _>(
        (0usize..160).map(|index| (!index.is_multiple_of(7)).then_some(index as i32)),
    )
    .into_array();
    let list_view = ListViewArray::new(
        elements,
        buffer![63u32, 1, 65, 126, 0].into_array(),
        buffer![70u32, 130, 64, 8, 0].into_array(),
        Validity::Array(BoolArray::from_iter([true, false, true, true, true]).into_array()),
    )
    .into_array();
    let minimum = list_view.clone().apply(&list_min(root()))?;
    let maximum = list_view.apply(&list_max(root()))?;
    let mut ctx = array_session().create_execution_ctx();

    let expected_min =
        PrimitiveArray::from_option_iter::<i32, _>([Some(64), None, Some(65), Some(127), None]);
    let expected_max =
        PrimitiveArray::from_option_iter::<i32, _>([Some(132), None, Some(128), Some(132), None]);
    assert_arrays_eq!(minimum, expected_min, &mut ctx);
    assert_arrays_eq!(maximum, expected_max, &mut ctx);
    Ok(())
}

#[test]
fn primitive_lane_widths() -> VortexResult<()> {
    macro_rules! assert_extrema {
        ($t:ty) => {{
            let elements = PrimitiveArray::from_option_iter::<$t, _>(
                (0usize..70).map(|index| (!index.is_multiple_of(7)).then_some(index as $t)),
            )
            .into_array();
            let list = ListArray::try_new(
                elements,
                buffer![0u32, 70].into_array(),
                Validity::NonNullable,
            )?
            .into_array();
            let minimum = list.clone().apply(&list_min(root()))?;
            let maximum = list.apply(&list_max(root()))?;
            let mut ctx = array_session().create_execution_ctx();

            let expected_min = PrimitiveArray::from_option_iter::<$t, _>([Some(1 as $t)]);
            let expected_max = PrimitiveArray::from_option_iter::<$t, _>([Some(69 as $t)]);
            assert_arrays_eq!(minimum, expected_min, &mut ctx);
            assert_arrays_eq!(maximum, expected_max, &mut ctx);
        }};
    }

    assert_extrema!(u8);
    assert_extrema!(u16);
    assert_extrema!(u32);
    assert_extrema!(u64);
    assert_extrema!(i8);
    assert_extrema!(i16);
    assert_extrema!(i32);
    assert_extrema!(i64);
    assert_extrema!(f32);
    assert_extrema!(f64);
    Ok(())
}

#[test]
fn primitive_wide_lane_widths() -> VortexResult<()> {
    macro_rules! assert_extrema {
        ($t:ty) => {{
            let elements = PrimitiveArray::from_option_iter::<$t, _>(
                (0usize..520)
                    .map(|index| (!index.is_multiple_of(7)).then_some((index % 100) as $t)),
            )
            .into_array();
            let list = ListArray::try_new(
                elements,
                buffer![0u32, 520].into_array(),
                Validity::NonNullable,
            )?
            .into_array();
            let minimum = list.clone().apply(&list_min(root()))?;
            let maximum = list.apply(&list_max(root()))?;
            let mut ctx = array_session().create_execution_ctx();

            let expected_min = PrimitiveArray::from_option_iter::<$t, _>([Some(0 as $t)]);
            let expected_max = PrimitiveArray::from_option_iter::<$t, _>([Some(99 as $t)]);
            assert_arrays_eq!(minimum, expected_min, &mut ctx);
            assert_arrays_eq!(maximum, expected_max, &mut ctx);
        }};
    }

    assert_extrema!(u8);
    assert_extrema!(u16);
    assert_extrema!(u32);
    assert_extrema!(u64);
    assert_extrema!(i8);
    assert_extrema!(i16);
    assert_extrema!(i32);
    assert_extrema!(i64);
    assert_extrema!(f32);
    assert_extrema!(f64);
    Ok(())
}

#[test]
fn f16_lane_width() -> VortexResult<()> {
    let elements = PrimitiveArray::from_option_iter::<f16, _>(
        (0usize..70).map(|index| (!index.is_multiple_of(7)).then_some(f16::from_f32(index as f32))),
    )
    .into_array();
    let list = ListArray::try_new(
        elements,
        buffer![0u32, 70].into_array(),
        Validity::NonNullable,
    )?
    .into_array();
    let minimum = list.clone().apply(&list_min(root()))?;
    let maximum = list.apply(&list_max(root()))?;
    let mut ctx = array_session().create_execution_ctx();

    let expected_min = PrimitiveArray::from_option_iter::<f16, _>([Some(f16::from_f32(1.0))]);
    let expected_max = PrimitiveArray::from_option_iter::<f16, _>([Some(f16::from_f32(69.0))]);
    assert_arrays_eq!(minimum, expected_min, &mut ctx);
    assert_arrays_eq!(maximum, expected_max, &mut ctx);
    Ok(())
}

#[test]
fn f16_wide_lane_width() -> VortexResult<()> {
    let elements =
        PrimitiveArray::from_option_iter::<f16, _>((0usize..520).map(|index| {
            (!index.is_multiple_of(7)).then_some(f16::from_f32((index % 100) as f32))
        }))
        .into_array();
    let list = ListArray::try_new(
        elements,
        buffer![0u32, 520].into_array(),
        Validity::NonNullable,
    )?
    .into_array();
    let minimum = list.clone().apply(&list_min(root()))?;
    let maximum = list.apply(&list_max(root()))?;
    let mut ctx = array_session().create_execution_ctx();

    let expected_min = PrimitiveArray::from_option_iter::<f16, _>([Some(f16::from_f32(0.0))]);
    let expected_max = PrimitiveArray::from_option_iter::<f16, _>([Some(f16::from_f32(99.0))]);
    assert_arrays_eq!(minimum, expected_min, &mut ctx);
    assert_arrays_eq!(maximum, expected_max, &mut ctx);
    Ok(())
}

fn create_fixed_size_list(validity: Validity) -> ArrayRef {
    let elements = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5, 6, 7, 8]).into_array();
    FixedSizeListArray::new(elements, 2, validity, 4).into_array()
}

#[test]
fn fixed_size_list_extrema() -> VortexResult<()> {
    let list = create_fixed_size_list(Validity::Array(
        BoolArray::from_iter([true, false, true, true]).into_array(),
    ));
    let minimum = list.clone().apply(&list_min(root()))?;
    let maximum = list.apply(&list_max(root()))?;
    let mut ctx = array_session().create_execution_ctx();

    let expected_min =
        PrimitiveArray::from_option_iter::<i32, _>([Some(1), None, Some(5), Some(7)]);
    let expected_max =
        PrimitiveArray::from_option_iter::<i32, _>([Some(2), None, Some(6), Some(8)]);
    assert_arrays_eq!(minimum, expected_min, &mut ctx);
    assert_arrays_eq!(maximum, expected_max, &mut ctx);
    Ok(())
}

#[test]
fn zero_width_fixed_size_lists_are_null() -> VortexResult<()> {
    let elements = PrimitiveArray::from_iter([0i32; 0]).into_array();
    let list = FixedSizeListArray::try_new(elements, 0, Validity::NonNullable, 3)?.into_array();
    let minimum = list.clone().apply(&list_min(root()))?;
    let maximum = list.apply(&list_max(root()))?;
    let mut ctx = array_session().create_execution_ctx();

    let expected = PrimitiveArray::from_option_iter::<i32, _>([None, None, None]);
    assert_arrays_eq!(minimum, expected, &mut ctx);
    assert_arrays_eq!(maximum, expected, &mut ctx);
    Ok(())
}

#[test]
fn string_extrema() -> VortexResult<()> {
    let elements = VarBinArray::from_iter(
        [Some("pear"), None, Some("apple"), Some("zebra")],
        DType::Utf8(Nullability::Nullable),
    )
    .into_array();
    let list = ListArray::try_new(
        elements,
        buffer![0u32, 3, 4].into_array(),
        Validity::NonNullable,
    )?
    .into_array();
    let minimum = list.clone().apply(&list_min(root()))?;
    let maximum = list.apply(&list_max(root()))?;
    let mut ctx = array_session().create_execution_ctx();

    let expected_min = VarBinArray::from_iter(
        [Some("apple"), Some("zebra")],
        DType::Utf8(Nullability::Nullable),
    );
    let expected_max = VarBinArray::from_iter(
        [Some("pear"), Some("zebra")],
        DType::Utf8(Nullability::Nullable),
    );
    assert_arrays_eq!(minimum, expected_min, &mut ctx);
    assert_arrays_eq!(maximum, expected_max, &mut ctx);
    Ok(())
}

#[test]
fn binary_extrema() -> VortexResult<()> {
    let elements = VarBinArray::from_iter(
        [
            Some(b"pear".as_slice()),
            None,
            Some(b"apple"),
            Some(b"zebra"),
        ],
        DType::Binary(Nullability::Nullable),
    )
    .into_array();
    let list = ListArray::try_new(
        elements,
        buffer![0u32, 3, 4].into_array(),
        Validity::NonNullable,
    )?
    .into_array();
    let minimum = list.clone().apply(&list_min(root()))?;
    let maximum = list.apply(&list_max(root()))?;
    let mut ctx = array_session().create_execution_ctx();

    let expected_min = VarBinArray::from_iter(
        [Some(b"apple".as_slice()), Some(b"zebra")],
        DType::Binary(Nullability::Nullable),
    );
    let expected_max = VarBinArray::from_iter(
        [Some(b"pear".as_slice()), Some(b"zebra")],
        DType::Binary(Nullability::Nullable),
    );
    assert_arrays_eq!(minimum, expected_min, &mut ctx);
    assert_arrays_eq!(maximum, expected_max, &mut ctx);
    Ok(())
}

#[test]
fn bool_extrema() -> VortexResult<()> {
    let elements = BoolArray::from_iter([true, true, false, true]);
    let list = ListArray::try_new(
        elements.into_array(),
        buffer![0u32, 3, 4].into_array(),
        Validity::NonNullable,
    )?
    .into_array();
    let minimum = list.clone().apply(&list_min(root()))?;
    let maximum = list.apply(&list_max(root()))?;
    let mut ctx = array_session().create_execution_ctx();

    let expected_min = BoolArray::from_iter([Some(false), Some(true)]);
    let expected_max = BoolArray::from_iter([Some(true), Some(true)]);
    assert_arrays_eq!(minimum, expected_min, &mut ctx);
    assert_arrays_eq!(maximum, expected_max, &mut ctx);
    Ok(())
}

#[test]
fn decimal_extrema() -> VortexResult<()> {
    let elements = DecimalArray::new(
        buffer![100i128, 250, 50, 999],
        DecimalDType::new(10, 2),
        Validity::NonNullable,
    )
    .into_array();
    let list = ListArray::try_new(
        elements,
        buffer![0u32, 3, 4].into_array(),
        Validity::NonNullable,
    )?
    .into_array();
    let minimum = list.clone().apply(&list_min(root()))?;
    let maximum = list.apply(&list_max(root()))?;
    let mut ctx = array_session().create_execution_ctx();

    let expected_min = DecimalArray::new(
        buffer![50i128, 999],
        DecimalDType::new(10, 2),
        Validity::AllValid,
    );
    let expected_max = DecimalArray::new(
        buffer![250i128, 999],
        DecimalDType::new(10, 2),
        Validity::AllValid,
    );
    assert_arrays_eq!(minimum, expected_min, &mut ctx);
    assert_arrays_eq!(maximum, expected_max, &mut ctx);
    Ok(())
}

#[test]
fn nan_options() -> VortexResult<()> {
    let elements = PrimitiveArray::from_iter([1.0f64, f64::NAN, 2.0, f64::NAN, f64::NAN]);
    let list = ListArray::try_new(
        elements.into_array(),
        buffer![0u32, 3, 5].into_array(),
        Validity::NonNullable,
    )?
    .into_array();
    let mut ctx = array_session().create_execution_ctx();

    let minimum = list.clone().apply(&list_min(root()))?;
    let maximum = list.clone().apply(&list_max(root()))?;
    let expected_min = PrimitiveArray::from_option_iter::<f64, _>([Some(1.0), None]);
    let expected_max = PrimitiveArray::from_option_iter::<f64, _>([Some(2.0), None]);
    assert_arrays_eq!(minimum, expected_min, &mut ctx);
    assert_arrays_eq!(maximum, expected_max, &mut ctx);

    for result in [
        list.clone().apply(&list_min_opts(
            root(),
            NumericalAggregateOpts::include_nans(),
        ))?,
        list.apply(&list_max_opts(
            root(),
            NumericalAggregateOpts::include_nans(),
        ))?,
    ] {
        let result = result.execute::<PrimitiveArray>(&mut ctx)?;
        assert!(result.as_slice::<f64>().iter().all(|value| value.is_nan()));
    }
    Ok(())
}

#[test]
fn nan_options_lane_path() -> VortexResult<()> {
    let elements = PrimitiveArray::from_option_iter::<f64, _>((0usize..520).map(|index| {
        if index == 411 {
            Some(f64::NAN)
        } else {
            (!index.is_multiple_of(11)).then_some(index as f64)
        }
    }));
    let list = ListArray::try_new(
        elements.into_array(),
        buffer![0u32, 520].into_array(),
        Validity::NonNullable,
    )?
    .into_array();
    let mut ctx = array_session().create_execution_ctx();

    let minimum = list.clone().apply(&list_min(root()))?;
    let maximum = list.clone().apply(&list_max(root()))?;
    let expected_min = PrimitiveArray::from_option_iter::<f64, _>([Some(1.0)]);
    let expected_max = PrimitiveArray::from_option_iter::<f64, _>([Some(519.0)]);
    assert_arrays_eq!(minimum, expected_min, &mut ctx);
    assert_arrays_eq!(maximum, expected_max, &mut ctx);

    for result in [
        list.clone().apply(&list_min_opts(
            root(),
            NumericalAggregateOpts::include_nans(),
        ))?,
        list.apply(&list_max_opts(
            root(),
            NumericalAggregateOpts::include_nans(),
        ))?,
    ] {
        let result = result.execute::<PrimitiveArray>(&mut ctx)?;
        assert!(result.as_slice::<f64>()[0].is_nan());
    }
    Ok(())
}

#[test]
fn constant_and_null_list_extrema() -> VortexResult<()> {
    let list = ListArray::try_new(
        create_list_elements(),
        buffer![0u32, 2, 5, 5, 7].into_array(),
        Validity::NonNullable,
    )?
    .into_array();
    let mut ctx = array_session().create_execution_ctx();
    let scalar = list.execute_scalar(1, &mut ctx)?;
    let constant = ConstantArray::new(scalar, 3).into_array();

    let minimum = constant.clone().apply(&list_min(root()))?;
    let maximum = constant.apply(&list_max(root()))?;
    let expected_min = PrimitiveArray::from_option_iter::<i32, _>([Some(3), Some(3), Some(3)]);
    let expected_max = PrimitiveArray::from_option_iter::<i32, _>([Some(5), Some(5), Some(5)]);
    assert_arrays_eq!(minimum, expected_min, &mut ctx);
    assert_arrays_eq!(maximum, expected_max, &mut ctx);

    let dtype = DType::List(
        Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
        Nullability::Nullable,
    );
    let nulls = ConstantArray::new(Scalar::null(dtype), 2).into_array();
    for result in [
        nulls.clone().apply(&list_min(root()))?,
        nulls.apply(&list_max(root()))?,
    ] {
        assert_eq!(result.valid_count(&mut ctx)?, 0);
    }
    Ok(())
}

#[test]
fn sliced_and_taken_extrema() -> VortexResult<()> {
    let list = ListArray::try_new(
        create_list_elements(),
        buffer![0u32, 2, 5, 5, 7].into_array(),
        Validity::NonNullable,
    )?
    .into_array();
    let mut ctx = array_session().create_execution_ctx();

    let sliced = list.slice(1..4)?;
    let minimum = sliced.apply(&list_min(root()))?;
    let expected_min = PrimitiveArray::from_option_iter::<i32, _>([Some(3), None, Some(6)]);
    assert_arrays_eq!(minimum, expected_min, &mut ctx);

    let taken = list.take(buffer![3u64, 0, 2].into_array())?;
    let maximum = taken.apply(&list_max(root()))?;
    let expected_max = PrimitiveArray::from_option_iter::<i32, _>([Some(6), Some(2), None]);
    assert_arrays_eq!(maximum, expected_max, &mut ctx);
    Ok(())
}

#[test]
fn empty_array_extrema() -> VortexResult<()> {
    let elements = PrimitiveArray::from_iter([0i32; 0]);
    let list = ListArray::try_new(
        elements.into_array(),
        buffer![0u32].into_array(),
        Validity::NonNullable,
    )?
    .into_array();
    let mut ctx = array_session().create_execution_ctx();

    for expression in [list_min(root()), list_max(root())] {
        let result = list
            .clone()
            .apply(&expression)?
            .execute::<PrimitiveArray>(&mut ctx)?;
        assert_eq!(result.len(), 0);
    }
    Ok(())
}

#[test]
fn dtype_validation() -> VortexResult<()> {
    let opts = NumericalAggregateOpts::default();
    let non_list = DType::Primitive(PType::I32, Nullability::NonNullable);
    assert!(
        ListMin
            .return_dtype(&opts, std::slice::from_ref(&non_list))
            .is_err()
    );
    assert!(ListMax.return_dtype(&opts, &[non_list]).is_err());

    let nested_element = DType::List(
        Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
        Nullability::NonNullable,
    );
    let nested_list = DType::List(Arc::new(nested_element), Nullability::NonNullable);
    assert!(
        ListMin
            .return_dtype(&opts, std::slice::from_ref(&nested_list))
            .is_err()
    );
    assert!(ListMax.return_dtype(&opts, &[nested_list]).is_err());

    let utf8_list = DType::List(
        Arc::new(DType::Utf8(Nullability::NonNullable)),
        Nullability::NonNullable,
    );
    assert_eq!(
        ListMin.return_dtype(&opts, std::slice::from_ref(&utf8_list))?,
        DType::Utf8(Nullability::Nullable)
    );
    assert_eq!(
        ListMax.return_dtype(&opts, &[utf8_list])?,
        DType::Utf8(Nullability::Nullable)
    );
    Ok(())
}

#[test]
fn display() {
    assert_eq!(list_min(root()).to_string(), "vortex.list.min($)");
    assert_eq!(list_max(root()).to_string(), "vortex.list.max($)");
    assert_eq!(
        list_min_opts(root(), NumericalAggregateOpts::include_nans()).to_string(),
        "vortex.list.min($, opts=skip_nans=false)"
    );
    assert_eq!(
        list_max_opts(root(), NumericalAggregateOpts::include_nans()).to_string(),
        "vortex.list.max($, opts=skip_nans=false)"
    );
}

#[test]
fn proto_round_trip() -> VortexResult<()> {
    for expr in [
        list_min(root()),
        list_min_opts(root(), NumericalAggregateOpts::include_nans()),
        list_max(root()),
        list_max_opts(root(), NumericalAggregateOpts::include_nans()),
    ] {
        let proto = expr.serialize_proto()?;
        let buf = proto.encode_to_vec();
        let decoded = pb::Expr::decode(buf.as_slice())?;
        let deserialized = Expression::from_proto(&decoded, &array_session())?;
        assert_eq!(expr, deserialized);
    }
    Ok(())
}
