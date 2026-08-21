// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fixed-bin range packing for ordered floating-point values.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::Dict;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::PType;
use vortex_array::dtype::half::f16;
use vortex_array::match_each_float_ptype;
use vortex_block_residual::BlockResidual;
use vortex_block_residual::OrderedFloat;
use vortex_block_residual::OrderedFloatArraySlotsExt;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::EstimateScore;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexResult;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::bitpack_compress::bitpack_encode_unchecked;
use vortex_int_mult::IntMult;
use vortex_range_packed::RangeDecomposition;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;
use crate::schemes::fixed_bins::CoarseModel;
use crate::schemes::fixed_bins::coarse_model;
use crate::schemes::sample_primitive_one_percent;

/// The default factor accounts for RangePacked decode cost during scheme selection.
const DEFAULT_DECODE_COST_FACTOR: f64 = 1.20;
const PREFILTER_MODEL_MARGIN: f64 = 1.50;
const MIN_PREFILTER_MODEL_RATIO: f64 = 1.15;
const MAX_BLOCK_TO_RANGE_RATIO: f64 = 1.10;

/// Compress floats through ordered IEEE bits, then use fixed-bin range packing.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct OrderedFloatRangePackedScheme {
    decode_cost_factor: f64,
}

impl OrderedFloatRangePackedScheme {
    /// Creates a scheme with the specified decode cost factor.
    pub const fn new(decode_cost_factor: f64) -> Self {
        Self { decode_cost_factor }
    }
}

impl Default for OrderedFloatRangePackedScheme {
    fn default() -> Self {
        Self::new(DEFAULT_DECODE_COST_FACTOR)
    }
}

impl Scheme for OrderedFloatRangePackedScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.ordered_float.range_packed"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_float()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![
            OrderedFloat.id(),
            IntMult.id(),
            Dict.id(),
            BitPacked.id(),
            BlockResidual.id(),
        ]
    }

    fn num_children(&self) -> usize {
        1
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        if compress_ctx.is_sample() {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        let decode_cost_factor = self.decode_cost_factor;
        let scheme_id = self.id();
        CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
            move |compressor, data, best_so_far, compress_ctx, exec_ctx| {
                let bit_width = data.array_as_primitive().ptype().bit_width() as f64;
                let maximum_adjusted_ratio = bit_width / decode_cost_factor;
                let best_ratio = best_so_far
                    .and_then(EstimateScore::finite_ratio)
                    .unwrap_or(1.0);
                if maximum_adjusted_ratio <= best_ratio {
                    return Ok(EstimateVerdict::Skip);
                }

                let sample = sample_primitive_one_percent(data.array_as_primitive(), exec_ctx)?;
                let sample = normalize_float_null_values(sample.as_view(), exec_ctx)?;
                let before_nbytes = sample.nbytes();
                let Some(after_nbytes) = estimate_ordered_float_if_promising(
                    sample.as_view(),
                    best_ratio,
                    decode_cost_factor,
                    exec_ctx,
                )?
                else {
                    return Ok(EstimateVerdict::Skip);
                };
                if after_nbytes == 0 {
                    return Ok(EstimateVerdict::Skip);
                }

                let adjusted_ratio =
                    before_nbytes as f64 / after_nbytes as f64 / decode_cost_factor;
                if adjusted_ratio <= best_ratio {
                    return Ok(EstimateVerdict::Skip);
                }

                let sample = sample.into_array();
                let incumbent = compressor.compress_sample_without_scheme(
                    &sample,
                    scheme_id,
                    compress_ctx,
                    exec_ctx,
                )?;
                if after_nbytes as f64 * decode_cost_factor >= incumbent.nbytes() as f64 {
                    return Ok(EstimateVerdict::Skip);
                }
                Ok(EstimateVerdict::Ratio(adjusted_ratio))
            },
        )))
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let primitive = normalize_float_null_values(data.array_as_primitive(), exec_ctx)?;
        encode_ordered_float(primitive.as_view(), exec_ctx)
    }
}

fn estimate_ordered_float_if_promising(
    primitive: vortex_array::ArrayView<'_, Primitive>,
    best_ratio: f64,
    decode_cost_factor: f64,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<Option<u64>> {
    let ordered_model = coarse_ordered_float_model(primitive);
    if model_prefers_blocks(ordered_model) {
        return Ok(None);
    }
    if !prefilter_passes(ordered_model, best_ratio, decode_cost_factor) {
        return Ok(None);
    }
    Ok(Some(estimate_ordered_float(primitive, exec_ctx)?))
}

fn prefilter_passes(model: CoarseModel, best_ratio: f64, decode_cost_factor: f64) -> bool {
    model.range_ratio.is_finite()
        && model.range_ratio >= MIN_PREFILTER_MODEL_RATIO
        && model.range_ratio / decode_cost_factor * PREFILTER_MODEL_MARGIN > best_ratio
        && !model_prefers_blocks(model)
}

fn model_prefers_blocks(model: CoarseModel) -> bool {
    model.block_ratio > model.range_ratio * MAX_BLOCK_TO_RANGE_RATIO
}

fn coarse_ordered_float_model(primitive: vortex_array::ArrayView<'_, Primitive>) -> CoarseModel {
    let values: Vec<u64> = match primitive.ptype() {
        PType::F16 => primitive
            .as_slice::<f16>()
            .iter()
            .map(|value| u64::from(ordered_u16(value.to_bits())))
            .collect(),
        PType::F32 => primitive
            .as_slice::<f32>()
            .iter()
            .map(|value| u64::from(ordered_u32(value.to_bits())))
            .collect(),
        PType::F64 => primitive
            .as_slice::<f64>()
            .iter()
            .map(|value| ordered_u64(value.to_bits()))
            .collect(),
        _ => {
            return CoarseModel {
                range_ratio: 0.0,
                block_ratio: 0.0,
            };
        }
    };
    coarse_model(&values, primitive.ptype().bit_width())
}

fn ordered_u16(bits: u16) -> u16 {
    if bits & (1_u16 << 15) == 0 {
        bits ^ (1_u16 << 15)
    } else {
        !bits
    }
}

fn ordered_u32(bits: u32) -> u32 {
    if bits & (1_u32 << 31) == 0 {
        bits ^ (1_u32 << 31)
    } else {
        !bits
    }
}

fn ordered_u64(bits: u64) -> u64 {
    if bits & (1_u64 << 63) == 0 {
        bits ^ (1_u64 << 63)
    } else {
        !bits
    }
}

fn estimate_ordered_float(
    primitive: vortex_array::ArrayView<'_, Primitive>,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<u64> {
    Ok(encode_ordered_float(primitive, exec_ctx)?.nbytes())
}

fn encode_ordered_float(
    primitive: vortex_array::ArrayView<'_, Primitive>,
    _exec_ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let ordered = OrderedFloat::from_primitive(primitive)?;
    let ordered_values = ordered.encoded().as_::<Primitive>();
    if ordered_values.is_empty() {
        return Ok(ordered.into_array());
    }

    let decomposition = match ordered_values.ptype() {
        PType::U16 => RangeDecomposition::encode(
            &ordered_values
                .as_slice::<u16>()
                .iter()
                .map(|&value| u64::from(value))
                .collect::<Vec<_>>(),
        )?,
        PType::U32 => RangeDecomposition::encode(
            &ordered_values
                .as_slice::<u32>()
                .iter()
                .map(|&value| u64::from(value))
                .collect::<Vec<_>>(),
        )?,
        PType::U64 => RangeDecomposition::encode(ordered_values.as_slice::<u64>())?,
        ptype => vortex_error::vortex_bail!(
            "range decomposition requires unsigned integers, got {ptype}"
        ),
    };
    let code_width = decomposition.code_width();
    let codes = PrimitiveArray::new(decomposition.codes().to_vec(), ordered_values.validity()?);
    // SAFETY: The decomposition computes the exact code width.
    let codes = unsafe { bitpack_encode_unchecked(codes, code_width) }?.into_array();
    let starts = narrow_unsigned(decomposition.bin_starts(), ordered_values.ptype())?.into_array();
    let references = DictArray::try_new(codes, starts)?.into_array();
    let offsets = narrow_unsigned(decomposition.offsets(), ordered_values.ptype())?;
    let offsets = BlockResidual::from_primitive(offsets.as_view())?.into_array();
    let reconstructed = IntMult::try_new(references, offsets, 1)?;
    Ok(OrderedFloat::try_new(reconstructed.into_array(), primitive.ptype())?.into_array())
}

fn narrow_unsigned(values: &[u64], ptype: PType) -> VortexResult<PrimitiveArray> {
    Ok(match ptype {
        PType::U16 => PrimitiveArray::from_iter(
            values
                .iter()
                .map(|&value| u16::try_from(value))
                .collect::<Result<Vec<_>, _>>()?,
        ),
        PType::U32 => PrimitiveArray::from_iter(
            values
                .iter()
                .map(|&value| u32::try_from(value))
                .collect::<Result<Vec<_>, _>>()?,
        ),
        PType::U64 => PrimitiveArray::from_iter(values.iter().copied()),
        _ => vortex_error::vortex_bail!(
            "range decomposition requires unsigned integers, got {ptype}"
        ),
    })
}

fn normalize_float_null_values(
    array: vortex_array::ArrayView<'_, Primitive>,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let validity = array.validity()?;
    if validity.definitely_no_nulls() {
        return Ok(array.into_owned());
    }

    let mask = validity.execute_mask(array.len(), exec_ctx)?;
    Ok(match_each_float_ptype!(array.ptype(), |T| {
        let values = array.as_slice::<T>();
        let replacement = mask
            .iter()
            .position(|valid| valid)
            .map(|index| values[index])
            .unwrap_or_default();
        array
            .into_owned()
            .map_each_with_validity::<T, T, _>(
                exec_ctx,
                |(value, valid)| if valid { value } else { replacement },
            )?
    }))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::half::f16;
    use vortex_error::VortexResult;

    use super::OrderedFloatRangePackedScheme;
    use super::encode_ordered_float;
    use super::estimate_ordered_float;
    use super::estimate_ordered_float_if_promising;
    use crate::BtrBlocksCompressorBuilder;
    use crate::CascadingCompressor;

    const TEST_SCHEME: OrderedFloatRangePackedScheme = OrderedFloatRangePackedScheme::new(1.0);

    #[rstest]
    #[case::f16(PrimitiveArray::from_iter((0_u16..4_096).map(|index| {
        f16::from_bits(0x3c00 + (index.wrapping_mul(101) % 1_024))
    })))]
    #[case::f32(PrimitiveArray::from_iter((0_u32..4_096).map(|index| {
        f32::from_bits(0x3f80_0000 + (index.wrapping_mul(7_919) % 65_536))
    })))]
    #[case::f64(PrimitiveArray::from_iter((0_u64..4_096).map(|index| {
        f64::from_bits(0x3ff0_0000_0000_0000 + (index.wrapping_mul(7_919) % 65_536))
    })))]
    fn ordered_float_tree_roundtrips(#[case] expected: PrimitiveArray) -> VortexResult<()> {
        let session = array_session();
        let mut ctx = session.create_execution_ctx();
        let compressed = CascadingCompressor::new(vec![&TEST_SCHEME])
            .compress(&expected.clone().into_array(), &mut ctx)?;

        assert_arrays_eq!(compressed, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn transform_estimate_matches_encoded_size() -> VortexResult<()> {
        let expected = PrimitiveArray::from_option_iter((0_u64..65_536).map(|index| {
            (index % 19 != 0).then(|| {
                let value = (index.wrapping_mul(7_919) % 100_000) as f64 / 100.0;
                if index % 101 == 0 {
                    f64::from_bits(value.to_bits() | 1)
                } else {
                    value
                }
            })
        }));
        let session = array_session();
        let mut ctx = session.create_execution_ctx();

        let estimate = estimate_ordered_float(expected.as_view(), &mut ctx)?;
        let encoded = encode_ordered_float(expected.as_view(), &mut ctx)?;

        assert_eq!(estimate, encoded.nbytes());
        Ok(())
    }

    #[test]
    fn prefilter_rejects_uniform_float_bits() -> VortexResult<()> {
        let mut state = 0x4d59_5df4_d0f3_3173_u64;
        let expected = PrimitiveArray::from_iter((0..65_536).map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = ((state >> 11) as f64 + 0.5) / (1_u64 << 53) as f64;
            unit * 2_000_000.0 - 1_000_000.0
        }));
        let session = array_session();
        let mut ctx = session.create_execution_ctx();

        let candidate =
            estimate_ordered_float_if_promising(expected.as_view(), 1.13, 1.20, &mut ctx)?;

        assert!(candidate.is_none());
        Ok(())
    }

    #[test]
    fn prefilter_rejects_constant_sample() -> VortexResult<()> {
        let expected = PrimitiveArray::from_iter(vec![1.0_f64; 1_024]);
        let session = array_session();
        let mut ctx = session.create_execution_ctx();

        let candidate =
            estimate_ordered_float_if_promising(expected.as_view(), 1.0, 1.20, &mut ctx)?;

        assert!(candidate.is_none());
        Ok(())
    }

    #[test]
    fn prefilter_retains_clustered_float_bits() -> VortexResult<()> {
        let expected = PrimitiveArray::from_iter((0_u64..65_536).map(|index| {
            let cluster = index % 8;
            let residual = index.wrapping_mul(7_919) % 1_024;
            f64::from_bits(0x3ff0_0000_0000_0000 + cluster * 0x1_0000_0000 + residual)
        }));
        let session = array_session();
        let mut ctx = session.create_execution_ctx();

        let candidate =
            estimate_ordered_float_if_promising(expected.as_view(), 1.36, 1.20, &mut ctx)?;

        assert!(candidate.is_some());
        Ok(())
    }

    #[test]
    fn prefilter_rejects_block_local_float_bits() -> VortexResult<()> {
        let expected = PrimitiveArray::from_iter((0_u64..65_536).map(|index| {
            let block = index / 64;
            let residual = index.wrapping_mul(7_919) % 1_024;
            f64::from_bits(0x3ff0_0000_0000_0000 + block * 0x10_0000 + residual)
        }));
        let session = array_session();
        let mut ctx = session.create_execution_ctx();

        let candidate =
            estimate_ordered_float_if_promising(expected.as_view(), 1.0, 1.20, &mut ctx)?;

        assert!(candidate.is_none());
        Ok(())
    }

    #[test]
    fn completed_incumbent_tree_beats_range_tree() -> VortexResult<()> {
        let expected = PrimitiveArray::from_iter((0_u64..65_536).map(|index| {
            let cluster = index % 8;
            f64::from_bits(0x3ff0_0000_0000_0000 + cluster * 0x1_0000_0000)
        }));
        let expected_array = expected.clone().into_array();
        let session = array_session();
        let mut ctx = session.create_execution_ctx();

        let retained = estimate_ordered_float_if_promising(
            expected.as_view(),
            1.0,
            TEST_SCHEME.decode_cost_factor,
            &mut ctx,
        )?;
        assert!(retained.is_some());

        let range_only =
            CascadingCompressor::new(vec![&TEST_SCHEME]).compress(&expected_array, &mut ctx)?;
        assert!(range_only.is::<vortex_block_residual::OrderedFloat>());
        assert!(range_only.children()[0].is::<vortex_int_mult::IntMult>());

        let selected = BtrBlocksCompressorBuilder::default()
            .with_new_scheme(&TEST_SCHEME)
            .build()
            .compress(&expected_array, &mut ctx)?;
        let selected_is_range = selected.is::<vortex_block_residual::OrderedFloat>()
            && selected
                .children()
                .first()
                .is_some_and(|child| child.is::<vortex_int_mult::IntMult>());
        assert!(!selected_is_range);
        assert!(selected.nbytes() < range_only.nbytes());
        assert_arrays_eq!(selected, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn nullable_tree_stores_full_positions() -> VortexResult<()> {
        let expected = PrimitiveArray::from_option_iter((0_u64..65_536).map(|index| {
            (index % 17 != 0)
                .then(|| f64::from_bits(0x3ff0_0000_0000_0000 + index.wrapping_mul(7_919) % 1_024))
        }));
        let expected_array = expected.clone().into_array();
        let session = array_session();
        let mut ctx = session.create_execution_ctx();
        let compressed =
            CascadingCompressor::new(vec![&TEST_SCHEME]).compress(&expected_array, &mut ctx)?;

        let int_mult = &compressed.children()[0];
        assert!(int_mult.is::<vortex_int_mult::IntMult>());
        let references = &int_mult.children()[0];
        let offsets = &int_mult.children()[1];
        assert!(references.is::<vortex_array::arrays::Dict>());
        assert!(references.children()[0].is::<vortex_fastlanes::BitPacked>());
        assert!(offsets.is::<vortex_block_residual::BlockResidual>());
        assert_eq!(references.len(), expected.len());
        assert_eq!(offsets.len(), expected.len());
        assert_arrays_eq!(compressed, expected, &mut ctx);
        Ok(())
    }
}
