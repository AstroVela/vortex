// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Extension;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::MaskedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayPlugin;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::extension::datetime::Date;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;

use crate::scalar_fns::l2_denorm::L2Denorm;
use crate::scalar_fns::l2_denorm::normalize_as_l2_denorm;
use crate::scalar_fns::l2_denorm::validate_l2_normalized_rows_against_norms;
use crate::tests::SESSION;
use crate::types::vector::Vector;
use crate::utils::test_helpers::assert_close;
use crate::utils::test_helpers::constant_tensor_array;
use crate::utils::test_helpers::tensor_array;
use crate::utils::test_helpers::vector_array;

/// Evaluates L2 denorm on a tensor/vector array and returns the executed array.
fn eval_l2_denorm(normalized: ArrayRef, norms: ArrayRef) -> VortexResult<ArrayRef> {
    let mut ctx = SESSION.create_execution_ctx();
    let result = L2Denorm::try_new_array(normalized, norms, &mut ctx)?;
    result.into_array().execute(&mut ctx)
}

fn non_tensor_extension_array() -> VortexResult<ArrayRef> {
    let storage = PrimitiveArray::from_iter([1i32, 2]).into_array();
    let ext_dtype = ExtDType::<Date>::try_new(TimeUnit::Days, storage.dtype().clone())?.erased();
    Ok(ExtensionArray::new(ext_dtype, storage).into_array())
}

fn tensor_snapshot(array: ArrayRef) -> VortexResult<(DType, Vec<bool>, Vec<f64>)> {
    let mut ctx = SESSION.create_execution_ctx();
    let ext: ExtensionArray = array.execute(&mut ctx)?;
    let validity = (0..ext.len())
        .map(|i| ext.is_valid(i, &mut ctx))
        .collect::<VortexResult<Vec<_>>>()?;
    let storage: FixedSizeListArray = ext.storage_array().clone().execute(&mut ctx)?;
    let elements: PrimitiveArray = storage.elements().clone().execute(&mut ctx)?;
    Ok((
        ext.dtype().clone(),
        validity,
        elements.as_slice::<f64>().to_vec(),
    ))
}

fn assert_tensor_arrays_eq(actual: ArrayRef, expected: ArrayRef) -> VortexResult<()> {
    let (actual_dtype, actual_validity, actual_elements) = tensor_snapshot(actual)?;
    let (expected_dtype, expected_validity, expected_elements) = tensor_snapshot(expected)?;

    assert_eq!(actual_dtype, expected_dtype);
    assert_eq!(actual_validity, expected_validity);
    assert_close(&actual_elements, &expected_elements);
    Ok(())
}

#[test]
fn l2_denorm_vectors() -> VortexResult<()> {
    let lhs = vector_array(3, &[0.6, 0.8, 0.0, 0.0, 0.0, 0.0])?;
    let rhs = PrimitiveArray::from_iter([5.0f64, 0.0]).into_array();
    let actual = eval_l2_denorm(lhs, rhs)?;
    let expected = vector_array(3, &[3.0, 4.0, 0.0, 0.0, 0.0, 0.0])?;

    assert_tensor_arrays_eq(actual, expected)?;
    Ok(())
}

#[test]
fn l2_denorm_fixed_shape_tensors() -> VortexResult<()> {
    let lhs = tensor_array(&[2, 2], &[0.5, 0.5, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0])?;
    let rhs = PrimitiveArray::from_iter([4.0f64, 2.0]).into_array();
    let actual = eval_l2_denorm(lhs, rhs)?;
    let expected = tensor_array(&[2, 2], &[2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0])?;

    assert_tensor_arrays_eq(actual, expected)?;
    Ok(())
}

#[test]
fn l2_denorm_null_propagation() -> VortexResult<()> {
    let lhs = vector_array(2, &[0.6, 0.8, 1.0, 0.0, 0.0, 0.0])?;
    let lhs = MaskedArray::try_new(lhs, Validity::from_iter([true, false, true]))?.into_array();

    let rhs = PrimitiveArray::from_option_iter([Some(5.0f64), Some(2.0), None]).into_array();
    let mut ctx = SESSION.create_execution_ctx();
    let actual: ExtensionArray = eval_l2_denorm(lhs, rhs)?.execute(&mut ctx)?;
    let storage: FixedSizeListArray = actual.storage_array().clone().execute(&mut ctx)?;
    let elements: PrimitiveArray = storage.elements().clone().execute(&mut ctx)?;

    assert!(actual.is_valid(0, &mut ctx)?);
    assert!(!actual.is_valid(1, &mut ctx)?);
    assert!(!actual.is_valid(2, &mut ctx)?);
    assert_close(&elements.as_slice::<f64>()[..2], &[3.0, 4.0]);
    Ok(())
}

#[test]
fn l2_denorm_rejects_non_extension_lhs() {
    let lhs = PrimitiveArray::from_iter([1.0f64, 2.0]).into_array();
    let rhs = PrimitiveArray::from_iter([1.0f64, 1.0]).into_array();

    let mut ctx = SESSION.create_execution_ctx();
    let result = L2Denorm::try_new_array(lhs, rhs, &mut ctx);
    assert!(result.is_err());
}

#[test]
fn l2_denorm_rejects_non_tensor_extension_lhs() -> VortexResult<()> {
    let lhs = non_tensor_extension_array()?;
    let rhs = PrimitiveArray::from_iter([1.0f64, 1.0]).into_array();

    let mut ctx = SESSION.create_execution_ctx();
    let result = L2Denorm::try_new_array(lhs, rhs, &mut ctx);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn l2_denorm_rejects_integer_tensor_lhs() -> VortexResult<()> {
    let lhs = tensor_array(&[2], &[1i32, 2, 3, 4])?;
    let rhs = PrimitiveArray::from_iter([1.0f64, 1.0]).into_array();

    let mut ctx = SESSION.create_execution_ctx();
    let result = L2Denorm::try_new_array(lhs, rhs, &mut ctx);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn l2_denorm_rejects_mismatched_rhs_ptype() -> VortexResult<()> {
    let lhs = vector_array(2, &[1.0, 0.0, 0.0, 1.0])?;
    let rhs = PrimitiveArray::from_iter([1.0f32, 1.0]).into_array();

    let mut ctx = SESSION.create_execution_ctx();
    let result = L2Denorm::try_new_array(lhs, rhs, &mut ctx);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn validate_l2_normalized_rows_accepts_normalized_f16_input() -> VortexResult<()> {
    let input = vector_array(2, &[3.0f32, 4.0, 0.0, 0.0].map(half::f16::from_f32))?;
    let mut ctx = SESSION.create_execution_ctx();
    let roundtrip = normalize_as_l2_denorm(input, &mut ctx)?;
    validate_l2_normalized_rows_against_norms(&roundtrip.child_at(0).clone(), None, &mut ctx)?;
    Ok(())
}

#[test]
fn validate_l2_normalized_rows_rejects_unnormalized_input() -> VortexResult<()> {
    let input = vector_array(2, &[3.0, 4.0, 1.0, 0.0])?;
    let mut ctx = SESSION.create_execution_ctx();
    let result = validate_l2_normalized_rows_against_norms(&input, None, &mut ctx);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn l2_denorm_try_new_array_rejects_unnormalized_child() -> VortexResult<()> {
    let normalized = vector_array(2, &[3.0, 4.0, 1.0, 0.0])?;
    let norms = PrimitiveArray::from_iter([5.0f64, 1.0]).into_array();
    let mut ctx = SESSION.create_execution_ctx();

    let result = L2Denorm::try_new_array(normalized, norms, &mut ctx);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn l2_denorm_try_new_array_rejects_nonzero_row_with_zero_norm() -> VortexResult<()> {
    let normalized = vector_array(2, &[1.0, 0.0, 0.0, 0.0])?;
    let norms = PrimitiveArray::from_iter([0.0f64, 0.0]).into_array();
    let mut ctx = SESSION.create_execution_ctx();

    let result = L2Denorm::try_new_array(normalized, norms, &mut ctx);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn l2_denorm_try_new_array_rejects_negative_norms() -> VortexResult<()> {
    let normalized = vector_array(2, &[1.0, 0.0, 0.0, 1.0])?;
    let norms = PrimitiveArray::from_iter([1.0f64, -1.0]).into_array();
    let mut ctx = SESSION.create_execution_ctx();

    let result = L2Denorm::try_new_array(normalized, norms, &mut ctx);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn l2_denorm_new_array_unchecked_accepts_unnormalized_child() -> VortexResult<()> {
    let normalized = vector_array(2, &[3.0, 4.0, 1.0, 0.0])?;
    let norms = PrimitiveArray::from_iter([5.0f64, 1.0]).into_array();

    let result = unsafe { L2Denorm::new_array_unchecked(normalized, norms) };
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn normalize_as_l2_denorm_roundtrips_vectors() -> VortexResult<()> {
    let input = vector_array(3, &[3.0, 4.0, 0.0, 0.0, 0.0, 0.0])?;
    let mut ctx = SESSION.create_execution_ctx();
    let roundtrip = normalize_as_l2_denorm(input.clone(), &mut ctx)?;
    let actual = roundtrip.into_array().execute(&mut ctx)?;

    assert_tensor_arrays_eq(actual, input)?;
    Ok(())
}

#[test]
fn normalize_as_l2_denorm_roundtrips_fixed_shape_tensors() -> VortexResult<()> {
    let input = tensor_array(&[2, 2], &[1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0])?;
    let mut ctx = SESSION.create_execution_ctx();
    let roundtrip = normalize_as_l2_denorm(input.clone(), &mut ctx)?;
    let actual = roundtrip.into_array().execute(&mut ctx)?;

    assert_tensor_arrays_eq(actual, input)?;
    Ok(())
}

#[test]
fn normalize_as_l2_denorm_supports_constant_tensors() -> VortexResult<()> {
    let input = constant_tensor_array(&[2], &[3.0, 4.0], 3)?;
    let mut ctx = SESSION.create_execution_ctx();
    let roundtrip = normalize_as_l2_denorm(input.clone(), &mut ctx)?;
    let actual = roundtrip.into_array().execute(&mut ctx)?;

    assert_tensor_arrays_eq(actual, input)?;
    Ok(())
}

#[test]
fn normalize_as_l2_denorm_supports_constant_vectors() -> VortexResult<()> {
    let input = Vector::constant_array(&[3.0, 4.0], 2)?;
    let mut ctx = SESSION.create_execution_ctx();
    let roundtrip = normalize_as_l2_denorm(input.clone(), &mut ctx)?;
    let actual = roundtrip.into_array().execute(&mut ctx)?;

    assert_tensor_arrays_eq(actual, input)?;
    Ok(())
}

#[test]
fn normalize_as_l2_denorm_constant_input_has_constant_children() -> VortexResult<()> {
    // The constant fast path in `normalize_as_l2_denorm` must produce an `L2Denorm` whose
    // normalized storage and norms child are both still `ConstantArray`s. This is what
    // allows downstream ops (cosine similarity, inner product) to short-circuit.
    let input = Vector::constant_array(&[3.0, 4.0], 16)?;
    let mut ctx = SESSION.create_execution_ctx();
    let roundtrip = normalize_as_l2_denorm(input, &mut ctx)?;

    // The normalized child must be an extension array whose storage is still constant.
    let normalized = roundtrip.child_at(0).clone();
    let normalized_ext = normalized
        .as_opt::<Extension>()
        .expect("normalized child should be an Extension array");
    assert!(
        normalized_ext
            .storage_array()
            .as_opt::<Constant>()
            .is_some(),
        "normalized storage should stay constant after the fast path"
    );

    // The norms child must itself be a ConstantArray with the exact precomputed norm.
    let norms = roundtrip.child_at(1).clone();
    let norms_const = norms
        .as_opt::<Constant>()
        .expect("norms child should be a ConstantArray");
    assert_close(
        &[norms_const
            .scalar()
            .as_primitive()
            .typed_value::<f64>()
            .expect("norms scalar")],
        &[5.0],
    );
    Ok(())
}

#[test]
fn normalize_as_l2_denorm_uses_zero_rows_for_zero_norms() -> VortexResult<()> {
    let input = vector_array(2, &[0.0, 0.0, 3.0, 4.0])?;
    let mut ctx = SESSION.create_execution_ctx();
    let roundtrip = normalize_as_l2_denorm(input.clone(), &mut ctx)?;
    let normalized: ExtensionArray = roundtrip.child_at(0).clone().execute(&mut ctx)?;
    let storage: FixedSizeListArray = normalized.storage_array().clone().execute(&mut ctx)?;
    let elements: PrimitiveArray = storage.elements().clone().execute(&mut ctx)?;
    let actual = roundtrip.into_array().execute(&mut ctx)?;

    assert_close(&elements.as_slice::<f64>()[..2], &[0.0, 0.0]);
    assert_tensor_arrays_eq(actual, input)?;
    Ok(())
}

/// Builds a non-nullable constant f64 norms array of length `len`.
fn constant_f64_norms(value: f64, len: usize) -> ArrayRef {
    ConstantArray::new(Scalar::primitive(value, Nullability::NonNullable), len).into_array()
}

#[test]
fn l2_denorm_constant_unit_norms_is_noop() -> VortexResult<()> {
    // Every stored norm is exactly 1.0, so the constant fast path must short-circuit and
    // return the normalized child unchanged.
    let normalized = vector_array(3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0])?;
    let norms = constant_f64_norms(1.0, 2);

    let actual = eval_l2_denorm(normalized.clone(), norms)?;
    assert_tensor_arrays_eq(actual, normalized)?;
    Ok(())
}

#[test]
fn l2_denorm_constant_near_unit_norms_is_noop() -> VortexResult<()> {
    // A norm that differs from 1.0 by less than the f64 unit-norm tolerance must still
    // hit the fast path and return the normalized child unchanged.
    let normalized = vector_array(3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0])?;
    let norms = constant_f64_norms(1.0 + 1e-12, 2);

    let actual = eval_l2_denorm(normalized.clone(), norms)?;
    assert_tensor_arrays_eq(actual, normalized)?;
    Ok(())
}

#[test]
fn l2_denorm_constant_nonunit_norms_scales_vectors() -> VortexResult<()> {
    // A constant norm that is not 1.0 must scale every element of every row by the same
    // factor via the backing elements multiplication path.
    let normalized = vector_array(3, &[0.6, 0.8, 0.0, 1.0, 0.0, 0.0])?;
    let norms = constant_f64_norms(5.0, 2);

    let actual = eval_l2_denorm(normalized, norms)?;
    let expected = vector_array(3, &[3.0, 4.0, 0.0, 5.0, 0.0, 0.0])?;
    assert_tensor_arrays_eq(actual, expected)?;
    Ok(())
}

#[test]
fn l2_denorm_nullable_constant_nonunit_norms_scales_vectors() -> VortexResult<()> {
    // A constant norm whose *dtype* is nullable, but whose value is not null, still reaches the
    // constant fast path. Tensor storage elements must stay non-nullable, so the norm has to be cast
    // to the element dtype before it multiplies the flat buffer; without that the rebuilt
    // `FixedSizeListArray` carries nullable elements and disagrees with the extension dtype.
    let normalized = vector_array(3, &[0.6, 0.8, 0.0, 1.0, 0.0, 0.0])?;
    let norms =
        ConstantArray::new(Scalar::primitive(5.0f64, Nullability::Nullable), 2).into_array();

    let actual = eval_l2_denorm(normalized, norms)?;

    let mut ctx = SESSION.create_execution_ctx();
    let ext: ExtensionArray = actual.execute(&mut ctx)?;
    let storage: FixedSizeListArray = ext.storage_array().clone().execute(&mut ctx)?;
    assert!(
        !storage.elements().dtype().is_nullable(),
        "tensor storage elements must stay non-nullable, got {}",
        storage.elements().dtype(),
    );

    // The nullable norms argument widens the result itself, which the strict lifting owns.
    assert!(ext.dtype().is_nullable());
    for i in 0..ext.len() {
        assert!(ext.is_valid(i, &mut ctx)?);
    }

    let elements: PrimitiveArray = storage.elements().clone().execute(&mut ctx)?;
    assert_close(elements.as_slice::<f64>(), &[3.0, 4.0, 0.0, 5.0, 0.0, 0.0]);
    Ok(())
}

#[test]
fn l2_denorm_constant_nonunit_norms_scales_fixed_shape_tensors() -> VortexResult<()> {
    // The same constant-scaling fast path must also cover multi-dimensional fixed-shape
    // tensors, where the backing elements buffer spans more than one slot per row.
    let normalized = tensor_array(&[2, 2], &[0.5, 0.5, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0])?;
    let norms = constant_f64_norms(4.0, 2);

    let actual = eval_l2_denorm(normalized, norms)?;
    let expected = tensor_array(&[2, 2], &[2.0, 2.0, 2.0, 2.0, 4.0, 0.0, 0.0, 0.0])?;
    assert_tensor_arrays_eq(actual, expected)?;
    Ok(())
}

/// Build an `L2Denorm` array from a raw input (which may have nullable storage) by running
/// `normalize_as_l2_denorm`. The normalized child ends up non-nullable, and the norms child
/// inherits the input's nullability, giving us two different per-child nullabilities to
/// round-trip.
#[rstest]
#[case::vector(l2_denorm_vector_input())]
#[case::fixed_shape_tensor(l2_denorm_tensor_input())]
fn serde_round_trip(#[case] input: ArrayRef) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let original = normalize_as_l2_denorm(input, &mut ctx)?.into_array();

    let scalar_fn_array = original.as_::<vortex_array::arrays::ScalarFn>();
    let children = scalar_fn_array.children();

    let plugin = ScalarFnArrayPlugin::new(L2Denorm);
    let metadata = plugin
        .serialize(&original, &SESSION)?
        .expect("L2Denorm serialize must produce metadata");

    let recovered = plugin.deserialize(
        original.dtype(),
        original.len(),
        &metadata,
        &[],
        &children,
        &SESSION,
    )?;

    assert_eq!(recovered.dtype(), original.dtype());
    assert_eq!(recovered.len(), original.len());
    assert_eq!(recovered.encoding_id(), original.encoding_id());
    Ok(())
}

fn l2_denorm_vector_input() -> ArrayRef {
    vector_array(3, &[3.0, 4.0, 0.0, 0.0, 0.0, 0.0]).expect("valid vector array")
}

fn l2_denorm_tensor_input() -> ArrayRef {
    tensor_array(&[2, 2], &[1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]).expect("valid tensor array")
}
