// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use onpair::CompactDictionaryView;
use onpair::search;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::scalar_fn::fns::like::LikeKernel;
use vortex_array::scalar_fn::fns::like::LikeOptions;
use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::OnPair;
use crate::OnPairArraySlotsExt;
use crate::decode::collect_codes_window;
use crate::decode::collect_widened;

enum SearchPattern {
    Exact(Vec<u8>),
    Prefix(Vec<u8>),
    Contains(Vec<u8>),
}

enum PreparedPattern {
    Exact(Vec<u16>),
    Prefix(search::PrefixQuery),
    Contains(search::ContainsTable),
}

impl PreparedPattern {
    fn matches(&self, codes: &[u16]) -> bool {
        match self {
            Self::Exact(query) => search::equals(codes, query),
            Self::Prefix(query) => search::starts_with(codes, query),
            Self::Contains(table) => search::contains(codes, table),
        }
    }
}

impl LikeKernel for OnPair {
    fn like(
        array: ArrayView<'_, Self>,
        pattern: &ArrayRef,
        options: LikeOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if options.case_insensitive {
            return Ok(None);
        }

        let Some(pattern_scalar) = pattern.as_constant() else {
            return Ok(None);
        };
        let pattern_bytes: &[u8] = if let Some(s) = pattern_scalar.as_utf8_opt() {
            let Some(v) = s.value() else {
                return Ok(None);
            };
            v.as_ref()
        } else if let Some(b) = pattern_scalar.as_binary_opt() {
            let Some(v) = b.value() else {
                return Ok(None);
            };
            v
        } else {
            return Ok(None);
        };
        let Some(search_pattern) = classify_like_pattern(pattern_bytes) else {
            return Ok(None);
        };

        let dict_offsets = collect_widened::<u32>(array.dict_offsets(), ctx)?;
        let dict =
            CompactDictionaryView::validate(array.dict_bytes().as_slice(), dict_offsets.as_slice())
                .map_err(|e| vortex_err!(InvalidArgument: "Invalid OnPair dictionary: {e}"))?;
        let window = collect_codes_window(array, ctx)?;

        let prepared = match search_pattern {
            SearchPattern::Exact(needle) => PreparedPattern::Exact(search::tokenize(&needle, dict)),
            SearchPattern::Prefix(prefix) => {
                PreparedPattern::Prefix(search::PrefixQuery::new(&prefix, dict))
            }
            SearchPattern::Contains(pattern) => {
                if pattern.len() > u8::MAX as usize {
                    return Ok(None);
                }
                PreparedPattern::Contains(search::ContainsTable::new(&pattern, dict))
            }
        };

        let negated = options.negated;
        let result = BitBuffer::collect_bool(array.len(), |i| {
            let matched = prepared.matches(window.row(i));
            if negated { !matched } else { matched }
        });

        let validity = array
            .array()
            .validity()?
            .union_nullability(pattern_scalar.dtype().nullability());

        Ok(Some(BoolArray::new(result, validity).into_array()))
    }
}

fn classify_like_pattern(pattern: &[u8]) -> Option<SearchPattern> {
    let mut literal = Vec::with_capacity(pattern.len());
    let mut wildcards = Vec::new();
    let mut i = 0;
    while i < pattern.len() {
        match pattern[i] {
            b'\\' => {
                i += 1;
                if i < pattern.len() {
                    literal.push(pattern[i]);
                } else {
                    literal.push(b'\\');
                }
            }
            b'%' => wildcards.push(literal.len()),
            b'_' => return None,
            b => literal.push(b),
        }
        i += 1;
    }

    match wildcards.as_slice() {
        [] => Some(SearchPattern::Exact(literal)),
        [end] if *end == literal.len() => Some(SearchPattern::Prefix(literal)),
        [0, end] if *end == literal.len() => Some(SearchPattern::Contains(literal)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar_fn::fns::like::Like;
    use vortex_array::scalar_fn::fns::like::LikeKernel;
    use vortex_array::scalar_fn::fns::like::LikeOptions;
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

    fn run_like(array: OnPairArray, pattern: &str, opts: LikeOptions) -> VortexResult<BoolArray> {
        let len = array.len();
        let arr = array.into_array();
        let pattern = ConstantArray::new(pattern, len).into_array();
        let result = Like
            .try_new_array(len, opts, [arr, pattern])?
            .into_array()
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())?;
        Ok(result.into_bool())
    }

    fn like(array: OnPairArray, pattern: &str) -> VortexResult<BoolArray> {
        run_like(array, pattern, LikeOptions::default())
    }

    #[test]
    fn test_like_exact() -> VortexResult<()> {
        let onpair = make_onpair(
            &[Some("alpha"), Some("alphabet"), Some("beta"), Some("alpha")],
            Nullability::NonNullable,
        );
        let result = like(onpair, "alpha")?;
        assert_arrays_eq!(
            &result,
            &BoolArray::from_iter([true, false, false, true]),
            &mut SESSION.create_execution_ctx()
        );
        Ok(())
    }

    #[test]
    fn test_like_prefix() -> VortexResult<()> {
        let onpair = make_onpair(
            &[Some("http://a"), Some("ftp://b"), Some("http://c")],
            Nullability::NonNullable,
        );
        let result = like(onpair, "http%")?;
        assert_arrays_eq!(
            &result,
            &BoolArray::from_iter([true, false, true]),
            &mut SESSION.create_execution_ctx()
        );
        Ok(())
    }

    #[test]
    fn test_like_contains_with_nulls() -> VortexResult<()> {
        let onpair = make_onpair(
            &[
                Some("hello world"),
                None,
                Some("goodbye"),
                Some("say hello"),
            ],
            Nullability::Nullable,
        );
        let result = like(onpair, "%hello%")?;
        assert_arrays_eq!(
            &result,
            &BoolArray::from_iter([Some(true), None, Some(false), Some(true)]),
            &mut SESSION.create_execution_ctx()
        );
        Ok(())
    }

    #[test]
    fn test_not_like_contains() -> VortexResult<()> {
        let onpair = make_onpair(
            &[Some("foobar_sdf"), Some("sdf_start"), Some("nothing")],
            Nullability::NonNullable,
        );
        let opts = LikeOptions {
            negated: true,
            case_insensitive: false,
        };
        let result = run_like(onpair, "%sdf%", opts)?;
        assert_arrays_eq!(
            &result,
            &BoolArray::from_iter([false, false, true]),
            &mut SESSION.create_execution_ctx()
        );
        Ok(())
    }

    #[test]
    fn test_like_kernel_handles_contains() -> VortexResult<()> {
        let onpair = make_onpair(
            &[Some("hello world"), Some("goodbye")],
            Nullability::NonNullable,
        );
        let pattern = ConstantArray::new("%world%", onpair.len()).into_array();
        let result = <OnPair as LikeKernel>::like(
            onpair.as_view(),
            &pattern,
            LikeOptions::default(),
            &mut SESSION.create_execution_ctx(),
        )?;
        assert!(
            result.is_some(),
            "OnPair LikeKernel should handle %literal%"
        );
        Ok(())
    }

    /// The pattern scan resolves row windows relative to `codes_offsets[0]`,
    /// which is nonzero for a sliced array. Exercise the kernel directly on a
    /// slice so the windowed row arithmetic is covered.
    #[test]
    fn test_like_kernel_on_sliced_array() -> VortexResult<()> {
        let onpair = make_onpair(
            &[
                Some("nomatch"),
                Some("http://a"),
                Some("say hello"),
                Some("http://b"),
                Some("nomatch"),
            ],
            Nullability::NonNullable,
        );
        let sliced = onpair.into_array().slice(1..4)?;
        assert!(sliced.is::<OnPair>(), "slice dropped OnPair encoding");
        let sliced = sliced
            .try_downcast::<OnPair>()
            .map_err(|_| vortex_err!("sliced array was not OnPair"))?;
        let mut ctx = SESSION.create_execution_ctx();

        let prefix = ConstantArray::new("http%", sliced.len()).into_array();
        let result = <OnPair as LikeKernel>::like(
            sliced.as_view(),
            &prefix,
            LikeOptions::default(),
            &mut ctx,
        )?
        .expect("OnPair LikeKernel should handle prefix%");
        assert_arrays_eq!(result, BoolArray::from_iter([true, false, true]), &mut ctx);

        let contains = ConstantArray::new("%hello%", sliced.len()).into_array();
        let result = <OnPair as LikeKernel>::like(
            sliced.as_view(),
            &contains,
            LikeOptions::default(),
            &mut ctx,
        )?
        .expect("OnPair LikeKernel should handle %literal%");
        assert_arrays_eq!(result, BoolArray::from_iter([false, true, false]), &mut ctx);
        Ok(())
    }

    #[test]
    fn test_like_kernel_rejects_overlong_contains() -> VortexResult<()> {
        let onpair = make_onpair(
            &[Some("hello world"), Some("goodbye")],
            Nullability::NonNullable,
        );
        let pattern = format!("%{}%", "x".repeat(usize::from(u8::MAX) + 1));
        let pattern = ConstantArray::new(pattern.as_str(), onpair.len()).into_array();
        let result = <OnPair as LikeKernel>::like(
            onpair.as_view(),
            &pattern,
            LikeOptions::default(),
            &mut SESSION.create_execution_ctx(),
        )?;
        assert!(
            result.is_none(),
            "overlong contains should fall back instead of panicking"
        );
        Ok(())
    }
}
