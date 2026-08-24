// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The ordered-float decode transform expressed as a scalar function.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::dtype::half::f16;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::OrderedFloatArray;
use crate::OrderedFloatArraySlotsExt;
use crate::ordered_float_array::unordered_u16;
use crate::ordered_float_array::unordered_u32;
use crate::ordered_float_array::unordered_u64;

/// Scalar function mapping order-preserving unsigned integers back to IEEE floats.
///
/// This is the element-wise inverse of the [`crate::OrderedFloat`] encode transform: the
/// latent child holds ordered unsigned bits and each output value is the float whose bit
/// pattern maps to them.
#[derive(Clone, Debug)]
pub struct OrderedFloatDecode;

impl OrderedFloatDecode {
    /// Creates a lazy ordered-bits-to-float decode of `encoded`.
    pub fn try_new(encoded: ArrayRef) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(OrderedFloatDecode.bind(EmptyOptions), vec![encoded])
    }
}

/// Express an [`OrderedFloatArray`] as a scalar-function array over its latent child.
pub fn ordered_float_as_scalar_fn(array: &OrderedFloatArray) -> VortexResult<ScalarFnArray> {
    OrderedFloatDecode::try_new(array.encoded().clone())
}

fn decoded_ptype(ptype: PType) -> VortexResult<PType> {
    match ptype {
        PType::U16 => Ok(PType::F16),
        PType::U32 => Ok(PType::F32),
        PType::U64 => Ok(PType::F64),
        _ => vortex_bail!("OrderedFloatDecode requires u16, u32, or u64, got {ptype}"),
    }
}

impl ScalarFnVTable for OrderedFloatDecode {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.ordered_float.decode");
        *ID
    }

    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        Ok(EmptyOptions)
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("encoded"),
            _ => unreachable!("Invalid child index {child_idx} for OrderedFloatDecode"),
        }
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let DType::Primitive(ptype, nullability) = &arg_dtypes[0] else {
            vortex_bail!(
                "OrderedFloatDecode expects a primitive child, got: {}",
                arg_dtypes[0]
            );
        };
        Ok(DType::Primitive(decoded_ptype(*ptype)?, *nullability))
    }

    fn execute(
        &self,
        _options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let encoded = args.get(0)?.execute::<PrimitiveArray>(ctx)?;
        let validity = encoded.validity()?;
        let decoded = match encoded.ptype() {
            PType::U16 => PrimitiveArray::new(
                encoded
                    .into_buffer::<u16>()
                    .map_each_in_place(|value| f16::from_bits(unordered_u16(value)))
                    .freeze(),
                validity,
            ),
            PType::U32 => PrimitiveArray::new(
                encoded
                    .into_buffer::<u32>()
                    .map_each_in_place(|value| f32::from_bits(unordered_u32(value)))
                    .freeze(),
                validity,
            ),
            PType::U64 => PrimitiveArray::new(
                encoded
                    .into_buffer::<u64>()
                    .map_each_in_place(|value| f64::from_bits(unordered_u64(value)))
                    .freeze(),
                validity,
            ),
            ptype => vortex_bail!("OrderedFloatDecode requires u16, u32, or u64, got {ptype}"),
        };
        Ok(decoded.into_array())
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
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::*;
    use crate::OrderedFloat;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = array_session();
        crate::initialize(&session);
        session
    });

    #[test]
    fn decodes_ordered_float_children() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = buffer![-3.5f32, -0.0, 0.0, 1.5, f32::INFINITY, f32::NEG_INFINITY];
        let array = PrimitiveArray::new(values, Validity::NonNullable);
        let encoded = OrderedFloat::from_primitive(array.as_view())?;

        let lazy = ordered_float_as_scalar_fn(&encoded)?;
        assert_arrays_eq!(lazy.into_array(), array, &mut ctx);
        Ok(())
    }

    #[test]
    fn preserves_nulls() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = PrimitiveArray::from_option_iter([Some(1.5f64), None, Some(-2.25)]);
        let encoded = OrderedFloat::from_primitive(array.as_view())?;

        let lazy = ordered_float_as_scalar_fn(&encoded)?;
        assert_arrays_eq!(lazy.into_array(), array, &mut ctx);
        Ok(())
    }
}
