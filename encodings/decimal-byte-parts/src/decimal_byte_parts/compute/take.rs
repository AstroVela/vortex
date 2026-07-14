// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::builtins::ArrayBuiltins;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::DecimalByteParts;
use crate::decimal_byte_parts::DecimalBytePartsArrayExt;

impl TakeExecute for DecimalByteParts {
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let decimal_dtype = *array
            .dtype()
            .as_decimal_opt()
            .vortex_expect("must be a decimal dtype");
        let msp = array.msp().take(indices.clone())?;
        let lower = array
            .lower()
            .map(|lower| {
                let taken = lower.take(indices.clone())?;
                // Nullable indices make the taken lower limb nullable, but validity is carried
                // solely by the msp (which is taken with the same indices): restore the
                // non-nullable u64 dtype the lower limb requires. Values at null positions are
                // unobservable.
                if taken.dtype().is_nullable() {
                    taken.fill_null(0u64)
                } else {
                    Ok(taken)
                }
            })
            .transpose()?;
        let taken = DecimalByteParts::try_new_parts(msp, lower, decimal_dtype)?;
        Ok(Some(taken.into_array()))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::assert_nth_scalar_is_null;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar::DecimalValue;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::fns::operators::Operator;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::super::two_limb::two_limb_array;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    /// Taking a two-limb array with nullable indices makes the taken lower limb nullable-typed;
    /// the kernel must restore its non-nullable dtype rather than fail validation.
    #[test]
    fn two_limb_take_nullable_indices() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = [0, (3i128 << 64) | 42, -(9i128 << 64) | 17, -1];
        let arr =
            two_limb_array(&values, Validity::NonNullable, DecimalDType::new(38, 0)).into_array();

        let indices = PrimitiveArray::from_option_iter([Some(2u64), None, Some(0)]).into_array();
        let taken = arr.take(indices)?;

        // Row values survive the round trip; the null index produces a null row.
        let expected = two_limb_array(
            &[values[2], 0, values[0]],
            Validity::Array(BoolArray::from_iter([true, false, true]).into_array()),
            DecimalDType::new(38, 0),
        )
        .into_array();
        assert_arrays_eq!(taken, expected, &mut ctx);
        assert_nth_scalar_is_null!(taken, 1, &mut ctx);

        // The taken array still supports the two-limb compare pushdown.
        let rhs = ConstantArray::new(
            Scalar::decimal(
                DecimalValue::I128(0),
                DecimalDType::new(38, 0),
                Nullability::NonNullable,
            ),
            taken.len(),
        )
        .into_array();
        let lt = taken.binary(rhs, Operator::Lt)?;
        assert_arrays_eq!(
            lt,
            BoolArray::from_iter([Some(true), None, Some(false)]),
            &mut ctx
        );

        Ok(())
    }
}
