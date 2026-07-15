// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::List;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::arrays::list::ListArrayExt;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::OnPair;
use crate::OnPairArrayExt;
use crate::OnPairArraySlotsExt;

impl TakeExecute for OnPair {
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let codes = unsafe {
            ListArray::new_unchecked(
                array.codes().clone(),
                array.codes_offsets().clone(),
                Validity::NonNullable,
            )
        };
        let taken_codes_ref = <List as TakeExecute>::take(codes.as_view(), indices, ctx)?
            .vortex_expect("List take kernel always returns Some");
        let taken_codes = taken_codes_ref
            .try_downcast::<List>()
            .ok()
            .vortex_expect("take for OnPair codes must return list array");

        let lengths = array
            .uncompressed_lengths()
            .take(indices.clone())?
            .fill_null(Scalar::zero_value(array.uncompressed_lengths().dtype()))?;
        let validity = array.array_validity().take(indices)?;

        Ok(Some(
            unsafe {
                OnPair::new_unchecked(
                    array
                        .dtype()
                        .clone()
                        .union_nullability(indices.dtype().nullability()),
                    array.dict_bytes_handle().clone(),
                    array.dict_offsets().clone(),
                    taken_codes.elements().clone(),
                    taken_codes.offsets().clone(),
                    lengths,
                    validity,
                )
            }
            .into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::compute::conformance::take::test_take_conformance;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::OnPair;
    use crate::compress::DEFAULT_DICT12_CONFIG;
    use crate::compress::onpair_compress;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    #[rstest]
    #[case(VarBinArray::from_iter(
        ["hello world", "testing onpair", "compression test", "data array", "vortex encoding"].map(Some),
        DType::Utf8(Nullability::NonNullable),
    ))]
    #[case(VarBinArray::from_iter(
        [Some("hello"), None, Some("world"), Some("test"), None],
        DType::Utf8(Nullability::Nullable),
    ))]
    #[case(VarBinArray::from_iter(
        ["single element"].map(Some),
        DType::Utf8(Nullability::NonNullable),
    ))]
    fn test_take_onpair_conformance(#[case] varbin: VarBinArray) -> VortexResult<()> {
        let array = varbin.into_array();
        let mut ctx = SESSION.create_execution_ctx();
        let onpair = onpair_compress(&array, DEFAULT_DICT12_CONFIG, &mut ctx)?;
        test_take_conformance(&onpair.into_array(), &mut ctx);
        Ok(())
    }

    /// `take` reuses the List take kernel over a synthesized `(codes,
    /// codes_offsets)` list. A sliced array's offsets do not start at zero, so
    /// run the conformance suite on a slice as well.
    #[test]
    fn test_take_sliced_onpair_conformance() -> VortexResult<()> {
        let varbin = VarBinArray::from_iter(
            [
                Some("hello world"),
                None,
                Some("testing onpair"),
                Some("compression test"),
                Some("data array"),
                None,
                Some("vortex encoding"),
                Some("tail row"),
            ],
            DType::Utf8(Nullability::Nullable),
        );
        let mut ctx = SESSION.create_execution_ctx();
        let onpair = onpair_compress(&varbin.into_array(), DEFAULT_DICT12_CONFIG, &mut ctx)?;
        let sliced = onpair.into_array().slice(2..7)?;
        assert!(sliced.is::<OnPair>(), "slice dropped OnPair encoding");
        test_take_conformance(&sliced, &mut ctx);
        Ok(())
    }
}
