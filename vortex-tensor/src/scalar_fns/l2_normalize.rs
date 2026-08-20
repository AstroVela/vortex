// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! L2 normalization for vector columns.
//!
//! [`L2Normalize`] converts each finite, nonzero [`Vector`] row into a [`UnitVector`]. Exact-zero
//! rows become null because they have no direction, and input nulls remain null. An existing
//! [`UnitVector`] is returned unchanged.
//!
//! [`Vector`]: crate::vector::Vector
//! [`UnitVector`]: crate::unit_vector::UnitVector

use num_traits::ToPrimitive;
use num_traits::Zero;
use prost::Message;
use vortex_array::ArrayRef;
use vortex_array::EmptyMetadata;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFn as ScalarFnArrayEncoding;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex_array::arrays::scalar_fn::ScalarFnArrayView;
use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayParts;
use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayVTable;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::proto::dtype as pb;
use vortex_array::expr::Expression;
use vortex_array::match_each_float_ptype;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::types::unit_vector::UnitVector;
use crate::types::vector::AnyVector;
use crate::types::vector::Vector;
use crate::utils::extract_flat_elements;

/// Converts ordinary vector rows into nullable unit vectors.
#[derive(Clone)]
pub struct L2Normalize;

impl L2Normalize {
    /// Constructs a [`ScalarFnArray`] that lazily normalizes `child`.
    ///
    /// # Errors
    ///
    /// Returns an error if `child` is not a float [`Vector`] or [`UnitVector`], or if the scalar
    /// function array cannot be constructed.
    ///
    /// [`Vector`]: crate::vector::Vector
    /// [`UnitVector`]: crate::unit_vector::UnitVector
    pub fn try_new(child: ArrayRef) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(L2Normalize.bind(EmptyOptions), vec![child])
    }
}

impl ScalarFnVTable for L2Normalize {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.tensor.l2_normalize");
        *ID
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            _ => unreachable!("L2Normalize must have exactly one child"),
        }
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        normalized_dtype(&arg_dtypes[0])
    }

    fn execute(
        &self,
        _options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;
        if input.dtype().as_extension().is::<UnitVector>() {
            return Ok(input);
        }

        normalize_vector(input, ctx)
    }

    fn validity(
        &self,
        _options: &Self::Options,
        _expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        // A valid zero row becomes null, so validity requires evaluating the values.
        Ok(None)
    }

    fn is_strict(&self, _options: &Self::Options) -> bool {
        true
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        true
    }
}

#[derive(Clone, prost::Message)]
struct L2NormalizeMetadata {
    /// The child dtype required before deserializing the child array.
    #[prost(message, optional, tag = "1")]
    input_dtype: Option<pb::DType>,
}

impl ScalarFnArrayVTable for L2Normalize {
    fn serialize(
        &self,
        view: &ScalarFnArrayView<Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        let array = view.as_::<ScalarFnArrayEncoding>();
        let input_dtype = Some(array.child_at(0).dtype().try_into()?);

        Ok(Some(L2NormalizeMetadata { input_dtype }.encode_to_vec()))
    }

    fn deserialize(
        &self,
        _dtype: &DType,
        len: usize,
        metadata: &[u8],
        children: &dyn ArrayChildren,
        session: &VortexSession,
    ) -> VortexResult<ScalarFnArrayParts<Self>> {
        let metadata = L2NormalizeMetadata::decode(metadata)
            .map_err(|error| vortex_err!("Failed to decode L2Normalize metadata: {error}"))?;
        let input_dtype = metadata
            .input_dtype
            .as_ref()
            .ok_or_else(|| vortex_err!("L2Normalize metadata missing input_dtype"))?;
        let input_dtype = DType::from_proto(input_dtype, session)?;
        normalized_dtype(&input_dtype)?;
        let child = children.get(0, &input_dtype, len)?;

        Ok(ScalarFnArrayParts {
            options: EmptyOptions,
            children: vec![child],
        })
    }
}

fn normalized_dtype(input_dtype: &DType) -> VortexResult<DType> {
    let ext_dtype = input_dtype
        .as_extension_opt()
        .ok_or_else(|| vortex_err!("L2Normalize input must be a vector, got {input_dtype}"))?;
    let metadata = ext_dtype
        .metadata_opt::<AnyVector>()
        .ok_or_else(|| vortex_err!("L2Normalize input must be a vector, got {input_dtype}"))?;

    if ext_dtype.is::<UnitVector>() {
        return Ok(input_dtype.clone());
    }

    vortex_ensure!(
        ext_dtype.is::<Vector>(),
        "L2Normalize input must be a Vector or UnitVector, got {input_dtype}",
    );
    let storage_dtype = DType::FixedSizeList(
        DType::Primitive(metadata.element_ptype(), Nullability::NonNullable).into(),
        metadata.dimensions(),
        Nullability::Nullable,
    );
    let output_dtype = ExtDType::<UnitVector>::try_new(EmptyMetadata, storage_dtype)?;

    Ok(DType::Extension(output_dtype.erased()))
}

fn normalize_vector(input: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let row_count = input.len();
    let metadata = input.dtype().as_extension().metadata::<AnyVector>();
    let dimensions = metadata.dimensions() as usize;

    let input: ExtensionArray = input.execute(ctx)?;
    let input_validity = input.as_ref().validity()?;
    let valid_rows = input_validity
        .nullability()
        .is_nullable()
        .then(|| input_validity.execute_mask(row_count, ctx))
        .transpose()?;
    let flat = extract_flat_elements(input.storage_array(), dimensions, ctx)?;

    match_each_float_ptype!(metadata.element_ptype(), |T| {
        let mut elements = BufferMut::<T>::with_capacity(row_count * dimensions);
        let mut output_validity = Vec::with_capacity(row_count);

        for row_idx in 0..row_count {
            if valid_rows
                .as_ref()
                .is_some_and(|valid_rows| !valid_rows.value(row_idx))
            {
                // SAFETY: `elements` reserves `dimensions` values for each input row.
                unsafe { elements.push_n_unchecked(T::zero(), dimensions) };
                output_validity.push(false);
                continue;
            }

            // SAFETY: `elements` reserves `dimensions` values for each input row.
            let is_nonzero =
                unsafe { normalize_row_into(flat.row::<T>(row_idx), &mut elements, row_idx)? };
            output_validity.push(is_nonzero);
        }

        let validity = Validity::Array(BoolArray::from_iter(output_validity).into_array());
        // SAFETY: The loop writes exactly `row_count * dimensions` non-nullable elements.
        let elements =
            unsafe { PrimitiveArray::new_unchecked(elements.freeze(), Validity::NonNullable) };
        let storage = FixedSizeListArray::try_new(
            elements.into_array(),
            metadata.dimensions(),
            validity,
            row_count,
        )?;

        // SAFETY: `normalize_row_into` emits a finite unit direction for every valid row. Zero and
        // input-null rows are marked null, so their payloads do not participate in the invariant.
        unsafe { UnitVector::new_unchecked(storage.into_array()) }
    })
}

/// Writes one normalized row and returns whether its direction is defined.
///
/// # Safety
///
/// `output` must have spare capacity for every value in `row`.
unsafe fn normalize_row_into<T: NativePType>(
    row: &[T],
    output: &mut BufferMut<T>,
    row_idx: usize,
) -> VortexResult<bool> {
    let mut scale = 0.0f64;
    for value in row {
        let value = ToPrimitive::to_f64(value)
            .vortex_expect("float NativePType values must convert to f64");
        vortex_ensure!(
            value.is_finite(),
            InvalidArgument: "L2Normalize input row {row_idx} must be finite, got {value}",
        );
        scale = scale.max(value.abs());
    }

    if scale == 0.0 {
        // SAFETY: The caller reserves space for the entire row.
        unsafe { output.push_n_unchecked(T::zero(), row.len()) };
        return Ok(false);
    }

    let scaled_norm = row
        .iter()
        .map(|value| {
            ToPrimitive::to_f64(value).vortex_expect("float NativePType values must convert to f64")
                / scale
        })
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();

    for value in row {
        let value = ToPrimitive::to_f64(value)
            .vortex_expect("float NativePType values must convert to f64");
        let normalized = T::from_f64((value / scale) / scaled_norm)
            .vortex_expect("normalized float coordinates must fit their input ptype");

        // SAFETY: The caller reserves space for the entire row.
        unsafe { output.push_unchecked(normalized) };
    }

    Ok(true)
}

#[cfg(test)]
mod tests {
    use half::f16;
    use vortex_array::ArrayPlugin;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::arrays::MaskedArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::extension::ExtensionArrayExt;
    use vortex_array::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
    use vortex_array::arrays::scalar_fn::ExactScalarFn;
    use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayPlugin;
    use vortex_array::dtype::NativePType;
    use vortex_array::matcher::Matcher;
    use vortex_array::validity::Validity;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;

    use crate::scalar_fns::l2_normalize::L2Normalize;
    use crate::tests::SESSION;
    use crate::types::unit_vector::UnitVector;
    use crate::types::vector::Vector;
    use crate::utils::test_helpers::assert_close;
    use crate::utils::test_helpers::tensor_array;
    use crate::utils::test_helpers::unit_vector_array;
    use crate::utils::test_helpers::vector_array;

    fn evaluate(input: ArrayRef) -> VortexResult<ExtensionArray> {
        let mut ctx = SESSION.create_execution_ctx();
        L2Normalize::try_new(input)?.into_array().execute(&mut ctx)
    }

    fn values<T: NativePType>(array: &ExtensionArray) -> VortexResult<Vec<T>> {
        let mut ctx = SESSION.create_execution_ctx();
        let storage: FixedSizeListArray = array.storage_array().clone().execute(&mut ctx)?;
        let values: PrimitiveArray = storage.elements().clone().execute(&mut ctx)?;

        Ok(values.as_slice::<T>().to_vec())
    }

    #[test]
    fn normalizes_vector_rows() -> VortexResult<()> {
        let output = evaluate(vector_array(2, &[3.0f64, 4.0, 1.0, 0.0])?)?;

        assert!(output.dtype().as_extension().is::<UnitVector>());
        assert_close(&values::<f64>(&output)?, &[0.6, 0.8, 1.0, 0.0]);
        Ok(())
    }

    #[test]
    fn zero_rows_become_null() -> VortexResult<()> {
        let output = evaluate(vector_array(2, &[3.0f64, 4.0, 0.0, 0.0])?)?;
        let mut ctx = SESSION.create_execution_ctx();

        assert!(output.is_valid(0, &mut ctx)?);
        assert!(!output.is_valid(1, &mut ctx)?);
        assert_close(&values::<f64>(&output)?, &[0.6, 0.8, 0.0, 0.0]);
        Ok(())
    }

    #[test]
    fn input_nulls_remain_null() -> VortexResult<()> {
        let input = vector_array(2, &[3.0f64, 4.0, 1.0, 0.0])?;
        let input = MaskedArray::try_new(input, Validity::from_iter([false, true]))?.into_array();
        let output = evaluate(input)?;
        let mut ctx = SESSION.create_execution_ctx();

        assert!(!output.is_valid(0, &mut ctx)?);
        assert!(output.is_valid(1, &mut ctx)?);
        Ok(())
    }

    #[test]
    fn normalizes_constant_storage() -> VortexResult<()> {
        let input = Vector::constant_array(&[3.0f64, 4.0], 3)?;
        let output = evaluate(input)?;

        assert_close(&values::<f64>(&output)?, &[0.6, 0.8, 0.6, 0.8, 0.6, 0.8]);
        Ok(())
    }

    #[test]
    fn rejects_non_finite_rows() -> VortexResult<()> {
        let input = vector_array(2, &[f64::INFINITY, 1.0])?;
        let mut ctx = SESSION.create_execution_ctx();

        assert!(
            L2Normalize::try_new(input)?
                .into_array()
                .execute::<ExtensionArray>(&mut ctx)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn handles_extreme_f64_values() -> VortexResult<()> {
        let tiny = f64::from_bits(1);
        let output = evaluate(vector_array(2, &[f64::MAX, f64::MAX, tiny, 0.0])?)?;
        let values = values::<f64>(&output)?;

        assert_close(&values[..2], &[2.0f64.sqrt().recip(); 2]);
        assert_eq!(&values[2..], &[1.0, 0.0]);
        Ok(())
    }

    #[test]
    fn f16_output_satisfies_unit_vector_validation() -> VortexResult<()> {
        let dimensions = 768;
        let output = evaluate(vector_array(
            dimensions,
            &vec![f16::ONE; dimensions as usize],
        )?)?;
        let mut ctx = SESSION.create_execution_ctx();

        UnitVector::try_new_unit_vector_array(output.storage_array().clone(), &mut ctx)?;
        Ok(())
    }

    #[test]
    fn unit_vector_input_is_identity() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let input = unit_vector_array(2, &[0.6f64, 0.8], &mut ctx)?;
        let output: ExtensionArray = L2Normalize::try_new(input.clone())?
            .into_array()
            .execute(&mut ctx)?;

        vortex_array::assert_arrays_eq!(output.into_array(), input, &mut ctx);
        Ok(())
    }

    #[test]
    fn serde_round_trip() -> VortexResult<()> {
        let child = vector_array(2, &[3.0f64, 4.0])?;
        let original = L2Normalize::try_new(child.clone())?.into_array();
        let plugin = ScalarFnArrayPlugin::new(L2Normalize);
        let metadata = plugin
            .serialize(&original, &SESSION)?
            .vortex_expect("L2Normalize serialization must produce metadata");
        let recovered = plugin.deserialize(
            original.dtype(),
            original.len(),
            &metadata,
            &[],
            &[child],
            &SESSION,
        )?;

        assert!(ExactScalarFn::<L2Normalize>::try_match(&recovered).is_some());
        assert_eq!(recovered.dtype(), original.dtype());
        Ok(())
    }

    #[test]
    fn rejects_fixed_shape_tensor() -> VortexResult<()> {
        let input = tensor_array(&[2], &[1.0f64, 0.0])?;

        assert!(L2Normalize::try_new(input).is_err());
        Ok(())
    }

    #[test]
    fn return_dtype_is_nullable_for_ordinary_vectors() -> VortexResult<()> {
        let input = vector_array(2, &[1.0f64, 0.0])?;
        let output = L2Normalize::try_new(input)?.into_array();

        assert!(output.dtype().is_nullable());
        assert!(output.dtype().as_extension().is::<UnitVector>());
        assert!(!output.dtype().as_extension().is::<Vector>());
        Ok(())
    }
}
