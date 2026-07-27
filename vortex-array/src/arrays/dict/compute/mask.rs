// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Dict;
use crate::arrays::DictArray;
use crate::arrays::Primitive;
use crate::arrays::dict::DictArraySlotsExt;
use crate::arrays::scalar_fn::ScalarFnFactoryExt;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::fns::mask::Mask as MaskExpr;
use crate::scalar_fn::fns::mask::MaskKernel;
use crate::scalar_fn::fns::mask::MaskReduce;

impl MaskReduce for Dict {
    fn mask(array: ArrayView<'_, Dict>, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        let masked_codes = MaskExpr.try_new_array(
            array.codes().len(),
            EmptyOptions,
            [array.codes().clone(), mask.clone()],
        )?;
        // SAFETY: masking codes doesn't change dict invariants
        Ok(Some(unsafe {
            DictArray::new_unchecked(masked_codes, array.values().clone()).into_array()
        }))
    }
}

impl MaskKernel for Dict {
    fn mask(
        array: ArrayView<'_, Dict>,
        mask: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Keep Primitive codes probe-able: AND the mask into their validity instead of hiding
        // them behind a scalar-fn wrapper, which would defeat downstream fast paths that
        // downcast the codes (constant-code slicing, specialized primitive slicing, ...).
        if let Some(codes) = array.codes().as_typed::<Primitive>()
            && let Some(masked_codes) = <Primitive as MaskReduce>::mask(codes, mask)?
        {
            // SAFETY: masking codes doesn't change dict invariants
            return Ok(Some(unsafe {
                DictArray::new_unchecked(masked_codes, array.values().clone()).into_array()
            }));
        }

        <Dict as MaskReduce>::mask(array, mask)
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::BoolArray;
    use crate::arrays::Dict;
    use crate::arrays::DictArray;
    use crate::arrays::Primitive;
    use crate::arrays::VarBinViewArray;
    use crate::arrays::dict::DictArraySlotsExt;
    use crate::arrays::scalar_fn::ScalarFnFactoryExt;
    use crate::scalar_fn::EmptyOptions;
    use crate::scalar_fn::fns::mask::Mask;

    #[test]
    fn mask_kernel_keeps_primitive_codes_probeable() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let dict = DictArray::try_new(
            buffer![0u8, 1, 0].into_array(),
            VarBinViewArray::from_iter_str(["a", "b"]).into_array(),
        )?;
        let mask = BoolArray::from_iter([true, false, true]);
        let masked = Mask.try_new_array(3, EmptyOptions, [dict.into_array(), mask.into_array()])?;

        // The mask execute-parent kernel reveals a Dict whose codes remain Primitive (with
        // masked validity) rather than being hidden behind a scalar-fn wrapper.
        let revealed = masked.execute_until::<Dict>(&mut ctx)?;
        let dict = revealed
            .try_downcast::<Dict>()
            .map_err(|a| vortex_err!("expected Dict, got {}", a.encoding_id()))?;
        assert!(dict.codes().is::<Primitive>());
        Ok(())
    }
}
