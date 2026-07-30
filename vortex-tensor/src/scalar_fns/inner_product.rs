// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Inner product expression for tensor-like types.

use num_traits::Float;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_float_ptype;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::RowFn;
use vortex_array::scalar_fn::RowVisitor;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_error::VortexResult;
use vortex_session::registry::CachedId;

use crate::scalar_fns::l2_denorm::DenormOrientation;
use crate::scalar_fns::row::TensorRow;
use crate::scalar_fns::row::tensor_element_ptype;
use crate::utils::extract_l2_denorm_children;

/// Inner product (dot product) between two columns.
///
/// Computes `sum(a_i * b_i)` over the flat backing buffer of each tensor or vector. For vectors
/// this is the standard dot product; for higher-rank ([`FixedShapeTensor`]) arrays this is the
/// Frobenius inner product.
///
/// Both inputs must be tensor-like extension arrays ([`FixedShapeTensor`] or [`Vector`]) with the
/// same dtype and a float element type. The output is a float column of the same float type.
///
/// [`FixedShapeTensor`]: crate::fixed_shape_tensor::FixedShapeTensor
/// [`Vector`]: crate::vector::Vector
#[derive(Clone, Debug, Default)]
pub struct InnerProduct;

impl RowFn for InnerProduct {
    type Options = EmptyOptions;
    type ArgsWitness = (TensorRow<f64>, TensorRow<f64>);
    type RetWitness = f64;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.tensor.inner_product");
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
            visitor
                .visit::<(TensorRow<T>, TensorRow<T>), T>(|(lhs, rhs)| inner_product_row(lhs, rhs))
        })
    }

    /// [`L2Denorm`]-wrapped operands factor through their stored norms: with `D(x, s)` denoting
    /// `x * s` rowwise, `dot(D(x, s), D(y, t)) = s * t * dot(x, y)` and
    /// `dot(D(x, s), y) = s * dot(x, y)`. The rewrite is expressed with lazy [`Operator::Mul`]
    /// arrays over the (much smaller) norm columns, so no denormalized coordinates are decoded.
    ///
    /// [`L2Denorm`]: crate::scalar_fns::l2_denorm::L2Denorm
    fn reduce_encoded(
        &self,
        _options: &Self::Options,
        args: &[ArrayRef],
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let len = args[0].len();

        Ok(match DenormOrientation::classify(&args[0], &args[1]) {
            DenormOrientation::Both { lhs, rhs } => {
                let (normalized_l, norms_l) = extract_l2_denorm_children(lhs);
                let (normalized_r, norms_r) = extract_l2_denorm_children(rhs);
                let dot =
                    InnerProduct.try_new_array(len, EmptyOptions, [normalized_l, normalized_r])?;
                Some(
                    dot.binary(norms_l, Operator::Mul)?
                        .binary(norms_r, Operator::Mul)?,
                )
            }
            DenormOrientation::One { denorm, plain } => {
                let (normalized, norms) = extract_l2_denorm_children(denorm);
                let dot =
                    InnerProduct.try_new_array(len, EmptyOptions, [normalized, plain.clone()])?;
                Some(dot.binary(norms, Operator::Mul)?)
            }
            DenormOrientation::Neither => None,
        })
    }
}

/// Computes the inner product (dot product) of two equal-length float slices.
///
/// Returns `sum(a_i * b_i)`.
fn inner_product_row<T: Float + NativePType>(a: &[T], b: &[T]) -> T {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x * y)
        .fold(T::zero(), |acc, v| acc + v)
}
