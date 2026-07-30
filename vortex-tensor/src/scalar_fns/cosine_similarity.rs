// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cosine similarity between two tensor columns.

use num_traits::Float;
use num_traits::Zero;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_float_ptype;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::RowFn;
use vortex_array::scalar_fn::RowVisitor;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_session::registry::CachedId;

use crate::scalar_fns::inner_product::InnerProduct;
use crate::scalar_fns::l2_denorm::DenormOrientation;
use crate::scalar_fns::l2_denorm::try_build_constant_l2_denorm;
use crate::scalar_fns::l2_norm::L2Norm;
use crate::scalar_fns::row::TensorRow;
use crate::scalar_fns::row::tensor_element_ptype;
use crate::utils::extract_l2_denorm_children;

/// Cosine similarity between two columns.
///
/// Computes `dot(a, b) / (||a|| * ||b||)` over the flat backing buffer of each tensor or vector.
/// The shape and permutation do not affect the result because cosine similarity only depends on the
/// element values, not their logical arrangement. A zero norm on either side yields `0.0`.
///
/// Both inputs must be tensor-like extension arrays ([`FixedShapeTensor`] or [`Vector`]) with the
/// same dtype and a float element type. The output is a float column of the same float type.
///
/// When either input is wrapped in [`L2Denorm`], this operator treats the stored norms and
/// normalized children as authoritative. For lossy encodings, that means the
/// optimized readthrough path may intentionally differ slightly from decoding both sides to dense
/// coordinates and recomputing cosine from scratch.
///
/// [`FixedShapeTensor`]: crate::fixed_shape_tensor::FixedShapeTensor
/// [`Vector`]: crate::vector::Vector
/// [`L2Denorm`]: crate::scalar_fns::l2_denorm::L2Denorm
#[derive(Clone, Debug, Default)]
pub struct CosineSimilarity;

impl RowFn for CosineSimilarity {
    type Options = EmptyOptions;
    type ArgsWitness = (TensorRow<f64>, TensorRow<f64>);
    type RetWitness = f64;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.tensor.cosine_similarity");
        *ID
    }

    fn arg_name(&self, idx: usize) -> ChildName {
        ChildName::from(["lhs", "rhs"][idx])
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        args: &[DType],
        visitor: V,
    ) -> VortexResult<V::Out> {
        match_each_float_ptype!(tensor_element_ptype(args)?, |T| {
            visitor.visit::<(TensorRow<T>, TensorRow<T>), T>(|(lhs, rhs)| {
                cosine_similarity_row(lhs, rhs)
            })
        })
    }

    /// [`L2Denorm`]-wrapped operands make the *stored* norms and normalized children
    /// authoritative: `cos(D(x, s), D(y, t)) = dot(x, y)` and `cos(D(x, s), y) = dot(x, y) /
    /// ||y||`, in both cases forced to `0.0` on rows where any authoritative norm is `0.0` (even
    /// for lossy children whose decoded coordinates are nonzero). A constant plain operand is
    /// first normalized once and rewrapped so it takes the same path.
    ///
    /// [`L2Denorm`]: crate::scalar_fns::l2_denorm::L2Denorm
    fn reduce_encoded(
        &self,
        _options: &Self::Options,
        args: &[ArrayRef],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let mut lhs = args[0].clone();
        let mut rhs = args[1].clone();
        if let Some(sfn) = try_build_constant_l2_denorm(&lhs, lhs.len(), ctx)? {
            lhs = sfn.into_array();
        }
        if let Some(sfn) = try_build_constant_l2_denorm(&rhs, rhs.len(), ctx)? {
            rhs = sfn.into_array();
        }

        match DenormOrientation::classify(&lhs, &rhs) {
            DenormOrientation::Both { lhs, rhs } => cosine_both_denorm(lhs, rhs, ctx).map(Some),
            DenormOrientation::One { denorm, plain } => {
                cosine_one_denorm(denorm, plain, ctx).map(Some)
            }
            DenormOrientation::Neither => Ok(None),
        }
    }
}

/// Computes the cosine similarity of two equal-length float slices.
///
/// Returns `dot(a, b) / (||a|| * ||b||)`, or `0.0` when either norm is zero.
fn cosine_similarity_row<T: Float + NativePType>(a: &[T], b: &[T]) -> T {
    let mut dot = T::zero();
    let mut norm_sq_a = T::zero();
    let mut norm_sq_b = T::zero();
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot = dot + x * y;
        norm_sq_a = norm_sq_a + x * x;
        norm_sq_b = norm_sq_b + y * y;
    }

    let denom = norm_sq_a.sqrt() * norm_sq_b.sqrt();
    if denom == T::zero() {
        T::zero()
    } else {
        dot / denom
    }
}

/// Both sides are `L2Denorm`: the normalized children are authoritative, so their dot product is
/// the cosine similarity, except that a row with a zero *stored* norm is a zero vector.
fn cosine_both_denorm(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let len = lhs.len();
    let (normalized_l, norms_l) = extract_l2_denorm_children(lhs);
    let (normalized_r, norms_r) = extract_l2_denorm_children(rhs);

    let dot: PrimitiveArray = InnerProduct
        .try_new_array(len, EmptyOptions, [normalized_l, normalized_r])?
        .execute(ctx)?;
    let norms_l: PrimitiveArray = norms_l.execute(ctx)?;
    let norms_r: PrimitiveArray = norms_r.execute(ctx)?;

    match_each_float_ptype!(dot.ptype(), |T| {
        let dots = dot.as_slice::<T>();
        let norms_l = norms_l.as_slice::<T>();
        let norms_r = norms_r.as_slice::<T>();
        let buffer: Buffer<T> = (0..len)
            .map(|i| {
                if norms_l[i] == T::zero() || norms_r[i] == T::zero() {
                    T::zero()
                } else {
                    dots[i]
                }
            })
            .collect();

        Ok(PrimitiveArray::new(buffer, Validity::NonNullable).into_array())
    })
}

/// One side is `L2Denorm`: `cos = dot(normalized, plain) / ||plain||`, forced to `0.0` on rows
/// where the stored norm or the plain norm is `0.0`.
fn cosine_one_denorm(
    denorm: &ArrayRef,
    plain: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let len = denorm.len();
    let (normalized, denorm_norms) = extract_l2_denorm_children(denorm);

    let dot: PrimitiveArray = InnerProduct
        .try_new_array(len, EmptyOptions, [normalized, plain.clone()])?
        .execute(ctx)?;
    let denorm_norms: PrimitiveArray = denorm_norms.execute(ctx)?;
    let plain_norm: PrimitiveArray = L2Norm
        .try_new_array(len, EmptyOptions, [plain.clone()])?
        .execute(ctx)?;

    match_each_float_ptype!(dot.ptype(), |T| {
        let dots = dot.as_slice::<T>();
        let denorm_norms = denorm_norms.as_slice::<T>();
        let plain_norms = plain_norm.as_slice::<T>();
        let buffer: Buffer<T> = (0..len)
            .map(|i| {
                if denorm_norms[i] == T::zero() || plain_norms[i] == T::zero() {
                    T::zero()
                } else {
                    dots[i] / plain_norms[i]
                }
            })
            .collect();

        Ok(PrimitiveArray::new(buffer, Validity::NonNullable).into_array())
    })
}
