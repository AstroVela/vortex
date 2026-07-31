// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! L2 denormalization: re-applying stored norms to a normalized tensor column.
//!
//! `L2Denorm(normalized, norms)` computes `normalized * norm` per row. It is the norm-splitting
//! wrapper the tensor crate compresses into, and downstream readthrough rules treat its stored
//! norms and normalized child as **authoritative**, even where that differs from recomputing over
//! fully decoded coordinates, which is what makes lossy normalized children usable.

use num_traits::Float;
use num_traits::ToPrimitive;
use num_traits::Zero;
use prost::Message;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Extension;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFn as ScalarFnArrayEncoding;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use vortex_array::arrays::scalar_fn::ExactScalarFn;
use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex_array::arrays::scalar_fn::ScalarFnArrayView;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayParts;
use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayVTable;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::proto::dtype as pb;
use vortex_array::match_each_float_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::RowFn;
use vortex_array::scalar_fn::RowVisitor;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::TypedScalarFnInstance;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::matcher::AnyTensor;
use crate::scalar_fns::l2_norm::L2Norm;
use crate::scalar_fns::row::TensorRow;
use crate::scalar_fns::row::TensorSink;
use crate::utils::extract_constant_flat_row;
use crate::utils::extract_flat_elements;
use crate::utils::unit_norm_tolerance;
use crate::utils::validate_tensor_float_input;

/// Re-applies authoritative L2 norms to a normalized tensor column.
///
/// Computes `normalized * norm` on each row over the flat backing buffer of each tensor-like type.
///
/// The normalized input must be a tensor-like extension array with a float element type and each
/// non-null row is semantically required to already be L2-normalized.
///
/// The norms input must be a primitive float column with the same element type as the normalized
/// tensor elements.
///
/// [`L2Denorm`] is the norm-splitting wrapper used throughout the tensor crate. Callers that build
/// it through [`try_new_array`](Self::try_new_array) get an exact unit-norm invariant on the
/// `normalized` child.
///
/// Advanced callers can also use [`new_array_unchecked`](Self::new_array_unchecked) to attach
/// authoritative stored norms to a lossy approximation of that child, such as quantized normalized
/// vectors.
///
/// Downstream readthrough rules intentionally treat the stored norms and normalized child as the
/// encoding contract, even when that differs slightly from recomputing over fully decoded
/// coordinates.
#[derive(Clone)]
pub struct L2Denorm;

impl L2Denorm {
    /// Creates a new [`TypedScalarFnInstance`] wrapping the L2 denormalization operation.
    ///
    /// This is a low-level scalar-function descriptor constructor. To build a semantically valid
    /// [`L2Denorm`] array, prefer [`try_new_array`](Self::try_new_array).
    pub fn new() -> TypedScalarFnInstance<L2Denorm> {
        TypedScalarFnInstance::new(L2Denorm, EmptyOptions)
    }

    /// Constructs a validated [`ScalarFnArray`] that lazily re-applies `norms` to `normalized`.
    ///
    /// This is the correct constructor for [`L2Denorm`] arrays. In addition to the structural
    /// checks performed by [`ScalarFnArray::try_new`], it validates that every valid row of the
    /// `normalized` child has L2 norm `1.0` (or `0.0` for zero rows), within the tolerance implied
    /// by the child element precision. It also validates that stored norms are non-negative, and
    /// that any row with stored norm `0.0` has an all-zero normalized row.
    ///
    /// # Errors
    ///
    /// Returns an error if the [`ScalarFnArray`] cannot be constructed (e.g. due to dtype
    /// mismatches) or if the `normalized` child is not row-wise L2-normalized.
    pub fn try_new_array(
        normalized: ArrayRef,
        norms: ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ScalarFnArray> {
        validate_l2_normalized_rows_against_norms(&normalized, Some(&norms), ctx)?;

        // SAFETY: We just validated that it is normalized.
        unsafe { Self::new_array_unchecked(normalized, norms) }
    }

    /// Constructs an [`L2Denorm`] array without validating that the `normalized` child is actually
    /// row-wise L2-normalized.
    ///
    /// This escape hatch is intended for advanced callers that already established, or
    /// intentionally relax, the normalized-child invariant. Structural validation still runs via
    /// [`ScalarFnArray::try_new`].
    ///
    /// # Safety
    ///
    /// The caller must ensure the `normalized` child is semantically suitable for L2
    /// denormalization. For exact wrappers, that means every valid row is unit-norm or zero.
    ///
    /// Lossy encodings may deliberately relax that invariant while still treating the stored norms
    /// as authoritative.
    ///
    /// Violating the intended contract will not cause memory unsafety, but may produce incorrect
    /// results.
    pub unsafe fn new_array_unchecked(
        normalized: ArrayRef,
        norms: ArrayRef,
    ) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(L2Denorm::new().erased(), vec![normalized, norms])
    }
}

impl RowFn for L2Denorm {
    type Options = EmptyOptions;
    type ArgsWitness = (TensorRow<f64>, f64);
    type RetWitness = ();

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.tensor.l2_denorm");
        *ID
    }

    fn arg_name(&self, idx: usize) -> ChildName {
        match idx {
            0 => ChildName::from("normalized"),
            1 => ChildName::from("norms"),
            _ => unreachable!("L2Denorm must have exactly two children"),
        }
    }

    /// The width comes from the normalized tensor, and pinning both element types to that one `T` is
    /// what enforces "norms match the tensor's element type": each argument validates against `T`
    /// separately, so agreement between them needs no cross-argument check of its own.
    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        args: &[DType],
        visitor: V,
    ) -> VortexResult<V::Out> {
        let normalized = args
            .first()
            .ok_or_else(|| vortex_err!("L2Denorm expects a normalized tensor argument"))?;

        match_each_float_ptype!(
            validate_tensor_float_input(normalized)?.element_ptype(),
            |T| {
                visitor.visit_into::<(TensorRow<T>, T), TensorSink<T>, ()>(|(row, norm), out| {
                    for (scaled, &x) in out.iter_mut().zip(row) {
                        *scaled = x * norm;
                    }
                })
            }
        )
    }

    /// Constant norms scale the whole flat buffer by one number, so the row loop has nothing to add.
    ///
    /// Unit norms make the function the identity and hand back the normalized child untouched, and
    /// any other constant rewrites the storage elements through a single multiply. Both beat one row
    /// at a time, and both survive a filtered batch, since filtering a constant yields a constant.
    fn reduce_encoded(
        &self,
        _options: &Self::Options,
        args: &[ArrayRef],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(const_norms) = args[1].as_opt::<Constant>() else {
            return Ok(None);
        };
        let norm_scalar = const_norms.scalar();
        vortex_ensure!(
            norm_scalar.dtype().is_float(),
            "L2Denorm constant norms must be a float scalar, got {}",
            norm_scalar.dtype(),
        );

        let Some(norm_value) = norm_scalar.value() else {
            // A null constant never reaches the kernel: the row lifting short-circuits it.
            return Ok(None);
        };

        execute_l2_denorm_constant_norms(args[0].clone(), norm_scalar, norm_value, ctx).map(Some)
    }
}

/// Metadata for a serialized [`L2Denorm`] array: both children's full [`DType`]s. The parent's
/// dtype is `normalized.union_nullability(norms.nullability())`, which loses both children's
/// individual nullabilities, so we persist them directly.
#[derive(Clone, prost::Message)]
pub(super) struct L2DenormMetadata {
    #[prost(message, optional, tag = "1")]
    normalized_dtype: Option<pb::DType>,
    #[prost(message, optional, tag = "2")]
    norms_dtype: Option<pb::DType>,
}

impl ScalarFnArrayVTable for L2Denorm {
    fn serialize(
        &self,
        view: &ScalarFnArrayView<Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        let scalar_fn_array = view.as_::<ScalarFnArrayEncoding>();
        let normalized_dtype = Some(scalar_fn_array.child_at(0).dtype().try_into()?);
        let norms_dtype = Some(scalar_fn_array.child_at(1).dtype().try_into()?);
        Ok(Some(
            L2DenormMetadata {
                normalized_dtype,
                norms_dtype,
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        _dtype: &DType,
        len: usize,
        metadata: &[u8],
        children: &dyn ArrayChildren,
        session: &VortexSession,
    ) -> VortexResult<ScalarFnArrayParts<Self>> {
        let metadata = L2DenormMetadata::decode(metadata)
            .map_err(|e| vortex_err!("Failed to decode L2DenormMetadata: {e}"))?;
        let normalized_pb = metadata
            .normalized_dtype
            .as_ref()
            .ok_or_else(|| vortex_err!("L2DenormMetadata missing normalized_dtype"))?;
        let norms_pb = metadata
            .norms_dtype
            .as_ref()
            .ok_or_else(|| vortex_err!("L2DenormMetadata missing norms_dtype"))?;
        let normalized_dtype = DType::from_proto(normalized_pb, session)?;
        let norms_dtype = DType::from_proto(norms_pb, session)?;
        let normalized = children.get(0, &normalized_dtype, len)?;
        let norms = children.get(1, &norms_dtype, len)?;
        Ok(ScalarFnArrayParts {
            options: EmptyOptions,
            children: vec![normalized, norms],
        })
    }
}

/// Optimized execution when the norms array is constant.
///
/// The result carries the normalized child's own dtype and validity, which the row lifting then
/// widens and masks exactly as it does the row loop's output.
fn execute_l2_denorm_constant_norms(
    normalized_ref: ArrayRef,
    norm_scalar: &Scalar,
    norm_value: &ScalarValue,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    // If the norms are all equal to 1 then we don't need to do anything.
    let err = norm_value
        .as_primitive()
        .as_f64()
        .vortex_expect("we know that this is a float, so it must fit in f64")
        - 1.0f64;

    let tensor_match = normalized_ref
        .dtype()
        .as_extension_opt()
        .and_then(|ext| ext.metadata_opt::<AnyTensor>())
        .ok_or_else(|| {
            vortex_err!(
                "L2Denorm normalized child must be a tensor-like extension, got {}",
                normalized_ref.dtype(),
            )
        })?;

    let tolerance = unit_norm_tolerance(
        norm_scalar.dtype().as_ptype(),
        tensor_match.list_size() as usize,
    );
    if err.abs() < tolerance {
        return Ok(normalized_ref);
    }

    // Even if the norms are not all 1, if they are all the same then we can multiply
    // the entire elements array by the same number.
    let ext_dtype = normalized_ref.dtype().as_extension().clone();
    let normalized: ExtensionArray = normalized_ref.execute(ctx)?;
    let storage_fsl: FixedSizeListArray = normalized.storage_array().clone().execute(ctx)?;

    // Replace the elements array with an array that multiplies it by the constant
    // norms array (with length multiplied by the dimensions of the vectors). The norm is cast to the
    // element dtype first so the product stays non-nullable, as tensor storage elements must be.
    let element_dtype = storage_fsl.elements().dtype().clone();
    let const_array = ConstantArray::new(
        norm_scalar.cast(&element_dtype)?,
        storage_fsl.elements().len(),
    )
    .into_array();
    let mult_elements = storage_fsl
        .elements()
        .clone()
        .binary(const_array, Operator::Mul)?;

    let new_fsl = FixedSizeListArray::try_new(
        mult_elements,
        storage_fsl.list_size(),
        storage_fsl.as_ref().validity()?,
        storage_fsl.len(),
    )?;

    Ok(ExtensionArray::new(ext_dtype, new_fsl.into_array()).into_array())
}

/// Validates that `normalized` and (when supplied) the matching `norms` jointly satisfy the
/// [`L2Denorm`] invariants:
///
/// - Every valid row of `normalized` has L2 norm `1.0` or `0.0` (within element-precision
///   tolerance).
/// - When `norms` is supplied, every stored norm is non-negative and any row whose stored norm is
///   `0.0` is exactly the zero vector in `normalized`.
pub fn validate_l2_normalized_rows_against_norms(
    normalized: &ArrayRef,
    norms: Option<&ArrayRef>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let row_count = normalized.len();
    if row_count == 0 {
        return Ok(());
    }

    let tensor_match = validate_tensor_float_input(normalized.dtype())?;
    let element_ptype = tensor_match.element_ptype();
    let tensor_flat_size = tensor_match.list_size() as usize;
    let tolerance = unit_norm_tolerance(element_ptype, tensor_flat_size);

    if let Some(norms) = norms {
        vortex_ensure_eq!(
            norms.dtype().as_ptype(),
            element_ptype,
            "L2Denorm norms ptype must match normalized element ptype"
        );
    }

    let normalized: ExtensionArray = normalized.clone().execute(ctx)?;
    let normalized_validity = normalized.as_ref().validity()?;

    let flat = extract_flat_elements(normalized.storage_array(), tensor_flat_size, ctx)?;
    let norms = norms
        .map(|norms| norms.clone().execute::<PrimitiveArray>(ctx))
        .transpose()?;

    let combined_validity = match &norms {
        Some(norms) => normalized_validity.and(norms.validity()?)?,
        None => normalized_validity,
    };
    // Resolve validity to a mask once rather than probing it per row.
    let combined_valid = combined_validity.execute_mask(row_count, ctx)?;

    match_each_float_ptype!(element_ptype, |T| {
        let stored_norms = norms.as_ref().map(|norms| norms.as_slice::<T>());

        for i in 0..row_count {
            if !combined_valid.value(i) {
                continue;
            }

            let (row_norm_sq, is_zero_row) =
                flat.row::<T>(i)
                    .iter()
                    .fold((0.0f64, true), |(sum_sq, is_zero), x| {
                        let value = ToPrimitive::to_f64(x).unwrap_or(f64::NAN);
                        (sum_sq + value * value, is_zero && value.abs() <= tolerance)
                    });
            let row_norm = row_norm_sq.sqrt();

            vortex_ensure!(
                row_norm == 0.0 || (row_norm - 1.0).abs() <= tolerance,
                "L2Denorm normalized child must have L2 norm 1.0 or 0.0, but row {i} has \
                 {row_norm:.6}",
            );

            if let Some(stored_norms) = stored_norms {
                let stored_norm_f64 = ToPrimitive::to_f64(&stored_norms[i]).unwrap_or(f64::NAN);
                vortex_ensure!(
                    stored_norm_f64 >= 0.0,
                    "L2Denorm norms must be non-negative, but row {i} has {stored_norm_f64:.6}",
                );

                if stored_norm_f64 == 0.0 {
                    vortex_ensure!(
                        is_zero_row,
                        "L2Denorm normalized child must be all zeros when norms row {i} is 0.0",
                    );
                }
            }
        }
    });

    Ok(())
}

/// Builds an unexecuted [`L2Denorm`] expression by normalizing `input` and reattaching the exact
/// norms as the norms child.
///
/// The returned array is a lazy `L2Denorm(normalized, norms)` scalar function array.
///
/// # Normalized child
///
/// The normalized child is always **non-nullable** with [`Validity::NonNullable`]. Every non-null
/// row with a positive L2 norm is divided by its norm to produce a unit-norm vector.
///
/// Rows that are null in the original input are **zeroed out** in the normalized output. This is
/// necessary because null rows may have undefined (garbage) physical storage values, and we do not
/// want to let those propagate into downstream lossy encodings.
///
/// # Nullability
///
/// Nullability is tracked entirely by the norms child. Null input rows produce null norms via
/// [`L2Norm`]'s validity propagation. When the [`L2Denorm`] wrapper is executed, its validity is
/// `and(normalized_validity, norms_validity)`, which correctly identifies originally-null rows
/// since the normalized child is all-valid and the norms child carries the original nulls.
///
/// Because this helper computes exact norms first and then divides by those norms, the returned
/// `normalized` child satisfies the strict unit-norm invariant required by [`L2Denorm`].
pub fn normalize_as_l2_denorm(
    input: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ScalarFnArray> {
    let row_count = input.len();
    let tensor_match = validate_tensor_float_input(input.dtype())?;
    let tensor_flat_size = tensor_match.list_size() as usize;

    // Constant fast path: if the input is a constant-backed extension, normalize the single
    // stored row once and return an `L2Denorm` whose children are both `ConstantArray`s.
    if let Some(wrapped) = try_build_constant_l2_denorm(&input, row_count, ctx)? {
        return Ok(wrapped);
    }

    // Calculate the norms of the vectors.
    let norms_array: ArrayRef = L2Norm
        .try_new_array(row_count, EmptyOptions, [input.clone()])?
        .execute(ctx)?;
    let primitive_norms: PrimitiveArray = norms_array.clone().execute(ctx)?;
    let norms_validity = primitive_norms.validity()?;

    let input: ExtensionArray = input.execute(ctx)?;
    let normalized_dtype = input.dtype().as_nonnullable();
    let flat = extract_flat_elements(input.storage_array(), tensor_flat_size, ctx)?;

    // Resolve validity to a mask once rather than probing it per row (each `Validity::is_valid`
    // executes a scalar for array-backed validity).
    let norms_valid = norms_validity.execute_mask(row_count, ctx)?;

    // Normalize all of the vectors.
    let normalized = match_each_float_ptype!(flat.ptype(), |T| {
        let norm_values = primitive_norms.as_slice::<T>();

        let total_elements = row_count * tensor_flat_size;
        let mut elements = BufferMut::<T>::with_capacity(total_elements);
        for i in 0..row_count {
            let is_valid = norms_valid.value(i);
            let norm = norm_values[i];

            // SAFETY: We allocated `row_count * tensor_flat_size` capacity and push exactly
            // `tensor_flat_size` elements per row.

            // Null rows must be explicitly zeroed out.
            if !is_valid || norm == T::zero() {
                unsafe { elements.push_n_unchecked(T::zero(), tensor_flat_size) };
            } else {
                for &x in flat.row::<T>(i) {
                    unsafe { elements.push_unchecked(x / norm) };
                }
            }
        }

        // Since L2Denorm's validity is the `and` of its child validities, we can make the
        // normalized array non-nullable.
        build_tensor_array(
            normalized_dtype,
            tensor_flat_size,
            row_count,
            Validity::NonNullable,
            elements.freeze(),
        )
    })?;

    // SAFETY:
    // - `norms_array` was produced by `L2Norm(input)`, so every stored norm is non-negative and
    //   null rows already carry null validity through that child.
    // - For every valid row, we either emit all zeros when the norm is zero or divide every
    //   element by the exact stored norm, so the normalized child is unit-norm (or zero) by
    //   construction.
    // - Null rows are zeroed out above to avoid propagating arbitrary physical storage values into
    //   downstream lossy encodings.
    unsafe { L2Denorm::new_array_unchecked(normalized, norms_array) }
}

/// Attempts to build an [`L2Denorm`] whose two children are both [`ConstantArray`]s by eagerly
/// normalizing `input`'s single stored row.
///
/// Returns `Ok(None)` when `input` is not a tensor-like extension array whose storage is a
/// [`ConstantArray`] with a non-null fixed-size-list scalar.
///
/// When `input` matches, the returned [`ScalarFnArray`] is equivalent to [`normalize_as_l2_denorm`]
/// but runs in `O(list_size)` time instead of `O(row_count * list_size)`.
///
/// This is helpful in some of the reduction steps for cosine similarity execution into inner
/// product execution.
pub(crate) fn try_build_constant_l2_denorm(
    input: &ArrayRef,
    len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ScalarFnArray>> {
    let Some(ext) = input.as_opt::<Extension>() else {
        return Ok(None);
    };
    let storage = ext.storage_array();
    let Some(const_storage) = storage.as_opt::<Constant>() else {
        return Ok(None);
    };
    if const_storage.scalar().is_null() {
        return Ok(None);
    }

    // The caller is expected to have already validated that `input` is an `AnyTensor`
    // extension dtype.
    let tensor_match = input
        .dtype()
        .as_extension()
        .metadata_opt::<AnyTensor>()
        .vortex_expect("caller validated input has AnyTensor metadata");
    let list_size = tensor_match.list_size() as usize;
    let original_nullability = input.dtype().nullability();
    let ext_dtype = input.dtype().as_extension().clone();
    let storage_fsl_nullability = storage.dtype().nullability();

    // Materialize just the single stored row; this does not expand the constant to the full
    // column length.
    let flat = extract_constant_flat_row(storage, ctx)?;

    let (normalized_fsl_scalar, norms_scalar) = match_each_float_ptype!(flat.ptype(), |T| {
        let row = flat.as_slice::<T>();

        let mut sum_sq = T::zero();
        for &x in row {
            sum_sq += x * x;
        }
        let norm_t: T = sum_sq.sqrt();

        // Zero-norm rows must be stored as all-zeros so [`L2Denorm`]'s unit-norm-or-zero
        // invariant holds. This mirrors the per-row logic in `normalize_as_l2_denorm`.
        let element_dtype = DType::Primitive(T::PTYPE, Nullability::NonNullable);
        let children: Vec<Scalar> = if norm_t == T::zero() {
            (0..list_size)
                .map(|_| Scalar::zero_value(&element_dtype))
                .collect()
        } else {
            row.iter()
                .map(|&v| Scalar::primitive(v / norm_t, Nullability::NonNullable))
                .collect()
        };

        // The rebuilt FSL scalar preserves the original storage FSL's nullability so the
        // resulting `ExtensionArray::new` call accepts the same extension dtype.
        let fsl_scalar = Scalar::fixed_size_list(element_dtype, children, storage_fsl_nullability);
        let norms_scalar = Scalar::primitive(norm_t, original_nullability);
        (fsl_scalar, norms_scalar)
    });

    let normalized_storage = ConstantArray::new(normalized_fsl_scalar, len).into_array();
    let normalized_ext = ExtensionArray::new(ext_dtype, normalized_storage).into_array();
    let norms_array = ConstantArray::new(norms_scalar, len).into_array();

    // SAFETY: Each row of `normalized_ext` is either `v / ||v||` (unit norm within floating
    // point tolerance) or all zeros when `||v|| == 0`. Stored norms are non-negative by
    // construction (`sqrt`). These are exactly the invariants required by
    // [`L2Denorm::new_array_unchecked`].
    let wrapped = unsafe { L2Denorm::new_array_unchecked(normalized_ext, norms_array)? };
    Ok(Some(wrapped))
}

/// Rebuilds a tensor-like extension array from flat primitive elements.
///
/// # Errors
///
/// Returns an error if the elements are invalid (have incorrect lengths for the
/// `FixedSizeListArray` storage array).
pub(crate) fn build_tensor_array<T: NativePType>(
    dtype: DType,
    tensor_flat_size: usize,
    row_count: usize,
    validity: Validity,
    elements: Buffer<T>,
) -> VortexResult<ArrayRef> {
    let list_size =
        u32::try_from(tensor_flat_size).vortex_expect("tensor flat size must fit into `u32`");

    let elements = PrimitiveArray::new(elements, Validity::NonNullable);

    let storage =
        FixedSizeListArray::try_new(elements.into_array(), list_size, validity, row_count)?;

    Ok(ExtensionArray::new(dtype.as_extension().clone(), storage.into_array()).into_array())
}

/// Classification of a binary operand pair by which side (if any) is wrapped in [`L2Denorm`].
///
/// Symmetric binary tensor operators (e.g. [`CosineSimilarity`], [`InnerProduct`]) have identical
/// fast paths for "only the lhs is denormalized" and "only the rhs is denormalized", and a separate
/// fast path for "both are denormalized". Rather than hand-rolling the commutative swap at every
/// call site, callers classify their operands with [`Self::classify`] and pattern-match on the
/// returned variant.
///
/// [`CosineSimilarity`]: crate::scalar_fns::cosine_similarity::CosineSimilarity
/// [`InnerProduct`]: crate::scalar_fns::inner_product::InnerProduct
pub(crate) enum DenormOrientation<'a> {
    /// Both operands are [`ExactScalarFn<L2Denorm>`] arrays.
    Both {
        lhs: &'a ArrayRef,
        rhs: &'a ArrayRef,
    },
    /// Exactly one operand is an [`ExactScalarFn<L2Denorm>`]; the other is plain.
    One {
        denorm: &'a ArrayRef,
        plain: &'a ArrayRef,
    },
    /// Neither operand is an [`ExactScalarFn<L2Denorm>`].
    Neither,
}

impl<'a> DenormOrientation<'a> {
    /// Classify `(lhs, rhs)` by which side (if any) is wrapped in [`L2Denorm`].
    pub(crate) fn classify(lhs: &'a ArrayRef, rhs: &'a ArrayRef) -> Self {
        let lhs_denorm = lhs.is::<ExactScalarFn<L2Denorm>>();
        let rhs_denorm = rhs.is::<ExactScalarFn<L2Denorm>>();
        match (lhs_denorm, rhs_denorm) {
            (true, true) => Self::Both { lhs, rhs },
            (true, false) => Self::One {
                denorm: lhs,
                plain: rhs,
            },
            (false, true) => Self::One {
                denorm: rhs,
                plain: lhs,
            },
            (false, false) => Self::Neither,
        }
    }
}
