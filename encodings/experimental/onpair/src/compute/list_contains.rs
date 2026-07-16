// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use onpair::CompactDictionaryView;
use onpair::search;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::dtype::DType;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::list_contains::ListContainsElementKernel;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::AllOr;

use crate::OnPair;
use crate::OnPairArraySlotsExt;
use crate::decode::collect_codes_window;
use crate::decode::collect_widened;

impl ListContainsElementKernel for OnPair {
    fn list_contains(
        list: &ArrayRef,
        element: ArrayView<'_, Self>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(list_scalar) = list.as_constant() else {
            return Ok(None);
        };

        let list_scalar = list_scalar.as_list();
        if !list_scalar
            .element_dtype()
            .eq_ignore_nullability(element.dtype())
        {
            return Ok(None);
        }
        let Some(elements) = list_scalar.elements() else {
            return Ok(Some(
                ConstantArray::new(null_bool(list, element), element.len()).into_array(),
            ));
        };

        let validity = element
            .array()
            .validity()?
            .execute_mask(element.len(), ctx)?;
        if matches!(validity.bit_buffer(), AllOr::None) {
            return Ok(Some(
                ConstantArray::new(null_bool(list, element), element.len()).into_array(),
            ));
        }

        let needles = elements.iter().filter_map(scalar_bytes).collect::<Vec<_>>();
        if needles.is_empty() {
            return Ok(Some(
                ConstantArray::new(false_bool(list, element), element.len()).into_array(),
            ));
        }

        let dict_offsets = collect_widened::<u32>(element.dict_offsets(), ctx)?;
        let dict = CompactDictionaryView::validate(
            element.dict_bytes().as_slice(),
            dict_offsets.as_slice(),
        )
        .map_err(|e| vortex_err!(InvalidArgument: "Invalid OnPair dictionary: {e}"))?;

        let mut queries = needles
            .iter()
            .map(|bytes| search::tokenize(bytes, dict))
            .collect::<Vec<_>>();
        queries.sort();
        queries.dedup();

        let window = collect_codes_window(element, ctx)?;

        let matches = |i| {
            let row = window.row(i);
            queries.iter().any(|query| search::equals(row, query))
        };
        let result = match validity.bit_buffer() {
            AllOr::All => BitBuffer::collect_bool(element.len(), matches),
            AllOr::None => unreachable!("all-invalid element handled above"),
            AllOr::Some(validity) => {
                let mut result = BitBufferMut::new_unset(element.len());
                validity.for_each_set_index(|i| {
                    if matches(i) {
                        result.set(i);
                    }
                });
                result.freeze()
            }
        };

        Ok(Some(
            BoolArray::new(
                result,
                Validity::from(list.dtype().nullability() | element.dtype().nullability()),
            )
            .into_array(),
        ))
    }
}

fn scalar_bytes(scalar: &Scalar) -> Option<&[u8]> {
    match scalar.dtype() {
        DType::Utf8(_) => scalar.as_utf8().value().map(|value| value.as_bytes()),
        DType::Binary(_) => scalar.as_binary().value().map(|value| value.as_slice()),
        _ => None,
    }
}

fn null_bool(list: &ArrayRef, element: ArrayView<'_, OnPair>) -> Scalar {
    Scalar::null(DType::Bool(
        list.dtype().nullability() | element.dtype().nullability(),
    ))
}

fn false_bool(list: &ArrayRef, element: ArrayView<'_, OnPair>) -> Scalar {
    Scalar::bool(
        false,
        list.dtype().nullability() | element.dtype().nullability(),
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::ListArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::expr::list_contains;
    use vortex_array::expr::lit;
    use vortex_array::expr::root;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::fns::list_contains::ListContainsElementKernel;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;

    use crate::OnPair;
    use crate::OnPairArray;
    use crate::compress::DEFAULT_DICT12_CONFIG;
    use crate::compress::onpair_compress;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    fn make_onpair(strings: &[Option<&str>], nullability: Nullability) -> OnPairArray {
        let array =
            VarBinArray::from_iter(strings.iter().copied(), DType::Utf8(nullability)).into_array();
        onpair_compress(
            &array,
            DEFAULT_DICT12_CONFIG,
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap()
    }

    fn string_list(values: Vec<Scalar>, nullability: Nullability) -> Scalar {
        Scalar::list(
            Arc::new(DType::Utf8(Nullability::Nullable)),
            values,
            nullability,
        )
    }

    fn assert_kernel_matches_canonical(
        strings: &[Option<&str>],
        nullability: Nullability,
        list: Scalar,
    ) -> VortexResult<()> {
        let canonical =
            VarBinArray::from_iter(strings.iter().copied(), DType::Utf8(nullability)).into_array();
        let mut ctx = SESSION.create_execution_ctx();
        let onpair = onpair_compress(&canonical, DEFAULT_DICT12_CONFIG, &mut ctx)?;
        let list_array = ConstantArray::new(list.clone(), canonical.len()).into_array();
        let actual = <OnPair as ListContainsElementKernel>::list_contains(
            &list_array,
            onpair.as_view(),
            &mut ctx,
        )?
        .expect("constant string list is handled by the OnPair kernel")
        .execute::<BoolArray>(&mut ctx)?;

        let expected = canonical
            .apply(&list_contains(lit(list), root()))?
            .execute::<BoolArray>(&mut ctx)?;
        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_list_contains_string_literals() -> VortexResult<()> {
        assert_kernel_matches_canonical(
            &[
                Some("alpha"),
                None,
                Some("beta"),
                Some("gamma"),
                Some("alpha"),
            ],
            Nullability::Nullable,
            string_list(
                vec![
                    Scalar::utf8("alpha", Nullability::Nullable),
                    Scalar::null(DType::Utf8(Nullability::Nullable)),
                    Scalar::utf8("gamma", Nullability::Nullable),
                ],
                Nullability::Nullable,
            ),
        )
    }

    #[test]
    fn test_list_contains_empty_list() -> VortexResult<()> {
        assert_kernel_matches_canonical(
            &[Some("alpha"), None, Some("beta")],
            Nullability::Nullable,
            string_list(Vec::new(), Nullability::Nullable),
        )
    }

    #[test]
    fn test_list_contains_only_null_list_elements() -> VortexResult<()> {
        assert_kernel_matches_canonical(
            &[Some("alpha"), None, Some("beta")],
            Nullability::Nullable,
            string_list(
                vec![Scalar::null(DType::Utf8(Nullability::Nullable))],
                Nullability::Nullable,
            ),
        )
    }

    #[test]
    fn test_list_contains_all_null_needles() -> VortexResult<()> {
        assert_kernel_matches_canonical(
            &[None, None],
            Nullability::Nullable,
            string_list(
                vec![Scalar::utf8("alpha", Nullability::Nullable)],
                Nullability::Nullable,
            ),
        )
    }

    #[test]
    fn test_list_contains_mismatched_string_dtype_falls_back() -> VortexResult<()> {
        let onpair = make_onpair(&[Some("alpha"), Some("beta")], Nullability::NonNullable);
        let list = Scalar::list(
            Arc::new(DType::Binary(Nullability::Nullable)),
            vec![Scalar::binary(b"alpha".to_vec(), Nullability::Nullable)],
            Nullability::Nullable,
        );
        let list = ConstantArray::new(list, onpair.len()).into_array();

        let result = <OnPair as ListContainsElementKernel>::list_contains(
            &list,
            onpair.as_view(),
            &mut SESSION.create_execution_ctx(),
        )?;

        assert!(result.is_none());
        Ok(())
    }

    #[test]
    fn test_list_contains_null_list() -> VortexResult<()> {
        let onpair = make_onpair(&[Some("alpha"), Some("beta")], Nullability::NonNullable);
        let list = Scalar::null(DType::List(
            Arc::new(DType::Utf8(Nullability::Nullable)),
            Nullability::Nullable,
        ));
        let pattern = ConstantArray::new(list, onpair.len()).into_array();
        let result = <OnPair as ListContainsElementKernel>::list_contains(
            &pattern,
            onpair.as_view(),
            &mut SESSION.create_execution_ctx(),
        )?
        .expect("constant null list is handled by the OnPair kernel")
        .execute::<BoolArray>(&mut SESSION.create_execution_ctx())?;

        assert_arrays_eq!(
            &result,
            &BoolArray::from_iter([None::<bool>, None]),
            &mut SESSION.create_execution_ctx()
        );
        Ok(())
    }

    /// The membership scan resolves row windows relative to `codes_offsets[0]`,
    /// which is nonzero for a sliced array. Exercise the kernel directly on a
    /// slice so the windowed row arithmetic is covered.
    #[test]
    fn test_list_contains_on_sliced_array() -> VortexResult<()> {
        let onpair = make_onpair(
            &[
                Some("aardvark"),
                Some("alpha"),
                Some("beta"),
                Some("gamma"),
                Some("zebra"),
            ],
            Nullability::NonNullable,
        );
        let sliced = onpair.into_array().slice(1..4)?;
        assert!(sliced.is::<OnPair>(), "slice dropped OnPair encoding");
        let sliced = sliced
            .try_downcast::<OnPair>()
            .map_err(|_| vortex_err!("sliced array was not OnPair"))?;

        let list = string_list(
            vec![
                Scalar::utf8("alpha", Nullability::Nullable),
                Scalar::utf8("gamma", Nullability::Nullable),
            ],
            Nullability::Nullable,
        );
        let pattern = ConstantArray::new(list, sliced.len()).into_array();
        let result = <OnPair as ListContainsElementKernel>::list_contains(
            &pattern,
            sliced.as_view(),
            &mut SESSION.create_execution_ctx(),
        )?
        .expect("constant string list is handled by the OnPair kernel")
        .execute::<BoolArray>(&mut SESSION.create_execution_ctx())?;

        assert_arrays_eq!(
            &result,
            &BoolArray::from_iter([Some(true), Some(false), Some(true)]),
            &mut SESSION.create_execution_ctx()
        );
        Ok(())
    }

    #[test]
    fn test_list_contains_non_constant_list_falls_back() -> VortexResult<()> {
        let onpair = make_onpair(&[Some("alpha"), Some("beta")], Nullability::NonNullable);
        let elements = VarBinArray::from_iter(
            [Some("alpha"), Some("beta")],
            DType::Utf8(Nullability::Nullable),
        )
        .into_array();
        let offsets = PrimitiveArray::from_iter([0u32, 1, 2]).into_array();
        let list = ListArray::try_new(elements, offsets, Validity::NonNullable)?.into_array();

        let result = <OnPair as ListContainsElementKernel>::list_contains(
            &list,
            onpair.as_view(),
            &mut SESSION.create_execution_ctx(),
        )?;

        assert!(result.is_none());
        Ok(())
    }
}
