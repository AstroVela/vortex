// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The float-quantization join transform expressed as a scalar function.

use std::fmt::Display;
use std::fmt::Formatter;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::FloatQuantArray;
use crate::FloatQuantArrayExt;
use crate::FloatQuantArraySlotsExt;
use crate::array::join_f16;
use crate::array::join_f32;
use crate::array::join_f64;
use crate::array::precision_bits;

/// Scalar function joining quantized high bits and low-bit adjustments back into floats.
///
/// This is the element-wise inverse of the [`crate::FloatQuant`] split: the primary child
/// holds the ordered float bits shifted right by `k`, and the optional secondary child holds
/// the sign-normalized low `k` bits. A missing secondary child is an implicit all-zero child.
#[derive(Clone, Debug)]
pub struct FloatQuantJoin;

/// Options for [`FloatQuantJoin`]: the number of split low bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FloatQuantJoinOptions {
    /// The number of low bits held by the secondary child.
    pub k: u8,
}

impl Display for FloatQuantJoinOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "k: {}", self.k)
    }
}

impl FloatQuantJoin {
    /// Creates a lazy float-quantization join of one or two latent children.
    pub fn try_new(
        primary: ArrayRef,
        secondary: Option<ArrayRef>,
        k: u8,
    ) -> VortexResult<ScalarFnArray> {
        let mut children = vec![primary];
        children.extend(secondary);
        ScalarFnArray::try_new(FloatQuantJoin.bind(FloatQuantJoinOptions { k }), children)
    }
}

/// Express a [`FloatQuantArray`] as a scalar-function array over its latent children.
pub fn float_quant_as_scalar_fn(array: &FloatQuantArray) -> VortexResult<ScalarFnArray> {
    FloatQuantJoin::try_new(
        array.primary().clone(),
        array.secondary().cloned(),
        array.k(),
    )
}

fn joined_ptype(ptype: PType) -> VortexResult<PType> {
    match ptype {
        PType::U16 => Ok(PType::F16),
        PType::U32 => Ok(PType::F32),
        PType::U64 => Ok(PType::F64),
        _ => vortex_bail!("FloatQuantJoin requires u16, u32, or u64, got {ptype}"),
    }
}

impl ScalarFnVTable for FloatQuantJoin {
    type Options = FloatQuantJoinOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.float_quant.join");
        *ID
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![options.k]))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        let [k] = metadata else {
            vortex_bail!("FloatQuantJoin options require exactly one byte");
        };
        Ok(FloatQuantJoinOptions { k: *k })
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Variadic {
            min: 1,
            max: Some(2),
        }
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("primary"),
            1 => ChildName::from("secondary"),
            _ => unreachable!("Invalid child index {child_idx} for FloatQuantJoin"),
        }
    }

    fn return_dtype(&self, options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let DType::Primitive(ptype, nullability) = &arg_dtypes[0] else {
            vortex_bail!(
                "FloatQuantJoin expects a primitive primary child, got: {}",
                arg_dtypes[0]
            );
        };
        let float_ptype = joined_ptype(*ptype)?;
        vortex_ensure!(
            options.k >= 1 && options.k <= precision_bits(float_ptype)?,
            "FloatQuantJoin k {} is out of range for {float_ptype}",
            options.k
        );
        if let Some(secondary) = arg_dtypes.get(1) {
            vortex_ensure!(
                matches!(secondary, DType::Primitive(s, _) if s == ptype),
                "FloatQuantJoin secondary child must be {ptype}, got: {secondary}"
            );
        }
        Ok(DType::Primitive(float_ptype, *nullability))
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let k = options.k;
        let primary = args.get(0)?.execute::<PrimitiveArray>(ctx)?;
        let validity = primary.validity()?;
        let secondary = (args.num_inputs() > 1)
            .then(|| args.get(1)?.execute::<PrimitiveArray>(ctx))
            .transpose()?;

        macro_rules! join {
            ($latent:ty, $join:ident) => {{
                let primary = primary.into_buffer::<$latent>();
                let joined: Buffer<_> = match &secondary {
                    Some(secondary) => primary
                        .iter()
                        .zip(secondary.as_slice::<$latent>())
                        .map(|(&p, &s)| $join(p, s, k))
                        .collect(),
                    None => primary.iter().map(|&p| $join(p, 0, k)).collect(),
                };
                PrimitiveArray::new(joined, validity)
            }};
        }

        let joined = match primary.ptype() {
            PType::U16 => join!(u16, join_f16),
            PType::U32 => join!(u32, join_f32),
            PType::U64 => join!(u64, join_f64),
            ptype => vortex_bail!("FloatQuantJoin requires u16, u32, or u64, got {ptype}"),
        };
        Ok(joined.into_array())
    }

    fn is_strict(&self, _options: &Self::Options) -> bool {
        true
    }

    fn is_infallible(&self, _options: &Self::Options) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::half::f16;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::*;
    use crate::FloatQuant;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = array_session();
        crate::initialize(&session);
        session
    });

    #[test]
    fn joins_split_children() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = buffer![-3.5f32, -0.0, 0.0, 1.5, f32::INFINITY, 123.456];
        let array = PrimitiveArray::new(values, Validity::NonNullable);
        let encoded = FloatQuant::from_primitive(array.as_view(), 16)?;

        let lazy = float_quant_as_scalar_fn(&encoded)?;
        assert_arrays_eq!(lazy.into_array(), array, &mut ctx);
        Ok(())
    }

    #[test]
    fn joins_implicit_zero_secondary() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = buffer![f16::from_f32(1.5), f16::from_f32(-2.0), f16::ZERO];
        let array = PrimitiveArray::new(values, Validity::NonNullable);
        let encoded = FloatQuant::from_primitive_constant_secondary(array.as_view(), 4)?;

        let lazy = float_quant_as_scalar_fn(&encoded)?;
        assert_arrays_eq!(lazy.into_array(), array, &mut ctx);
        Ok(())
    }

    #[test]
    fn preserves_nulls() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = PrimitiveArray::from_option_iter([Some(1.5f64), None, Some(-2.25)]);
        let encoded = FloatQuant::from_primitive(array.as_view(), 20)?;

        let lazy = float_quant_as_scalar_fn(&encoded)?;
        assert_arrays_eq!(lazy.into_array(), array, &mut ctx);
        Ok(())
    }
}
