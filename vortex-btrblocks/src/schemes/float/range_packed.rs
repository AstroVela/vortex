// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fixed-bin range packing for floating-point values.

use vortex_alp::ALP;
use vortex_alp::ALPArrayExt;
use vortex_alp::ALPArraySlotsExt;
use vortex_alp::alp_encode;
use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::match_each_float_ptype;
use vortex_block_residual::OrderedFloat;
use vortex_block_residual::OrderedFloatArraySlotsExt;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::EstimateScore;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexResult;
use vortex_range_packed::RangePacked;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::compress_patches;
use crate::schemes::sample_primitive_one_percent;

/// The default factor accounts for RangePacked decode cost during scheme selection.
const DEFAULT_DECODE_COST_FACTOR: f64 = 1.20;

/// Compress floats through ALP or ordered IEEE bits, then use fixed-bin range packing.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct FloatRangePackedScheme {
    decode_cost_factor: f64,
}

impl FloatRangePackedScheme {
    /// Creates a scheme with the specified decode cost factor.
    pub const fn new(decode_cost_factor: f64) -> Self {
        Self { decode_cost_factor }
    }
}

impl Default for FloatRangePackedScheme {
    fn default() -> Self {
        Self::new(DEFAULT_DECODE_COST_FACTOR)
    }
}

impl Scheme for FloatRangePackedScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.float.range_packed"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_float()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![RangePacked.id(), OrderedFloat.id(), ALP.id()]
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
        CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
            move |_compressor, data, best_so_far, _compress_ctx, exec_ctx| {
                let bit_width = data.array_as_primitive().ptype().bit_width() as f64;
                let maximum_adjusted_ratio = bit_width / decode_cost_factor;
                if best_so_far
                    .and_then(EstimateScore::finite_ratio)
                    .is_some_and(|best| maximum_adjusted_ratio <= best)
                {
                    return Ok(EstimateVerdict::Skip);
                }

                let sample = sample_primitive_one_percent(data.array_as_primitive(), exec_ctx)?;
                let sample = normalize_float_null_values(sample.as_view(), exec_ctx)?;
                let before_nbytes = sample.nbytes();
                let (_, after_nbytes) = choose_transform(sample.as_view(), exec_ctx)?;
                if after_nbytes == 0 {
                    return Ok(EstimateVerdict::Skip);
                }

                let adjusted_ratio =
                    before_nbytes as f64 / after_nbytes as f64 / decode_cost_factor;
                if adjusted_ratio <= 1.0 {
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
        let sample = sample_primitive_one_percent(data.array_as_primitive(), exec_ctx)?;
        let sample = normalize_float_null_values(sample.as_view(), exec_ctx)?;
        let (transform, _) = choose_transform(sample.as_view(), exec_ctx)?;

        let primitive = normalize_float_null_values(data.array_as_primitive(), exec_ctx)?;
        encode_transform(primitive.as_view(), transform, exec_ctx)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum FloatTransform {
    OrderedFloat,
    Alp,
}

fn choose_transform(
    primitive: vortex_array::ArrayView<'_, Primitive>,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<(FloatTransform, u64)> {
    let ordered = encode_transform(primitive, FloatTransform::OrderedFloat, exec_ctx)?;
    let mut best = (FloatTransform::OrderedFloat, ordered.nbytes());

    if primitive.ptype().is_float() && primitive.ptype().bit_width() > 16 {
        let alp = encode_transform(primitive, FloatTransform::Alp, exec_ctx)?;
        if alp.nbytes() < best.1 {
            best = (FloatTransform::Alp, alp.nbytes());
        }
    }

    Ok(best)
}

fn encode_transform(
    primitive: vortex_array::ArrayView<'_, Primitive>,
    transform: FloatTransform,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    match transform {
        FloatTransform::OrderedFloat => {
            let ordered = OrderedFloat::from_primitive(primitive)?;
            let packed = RangePacked::from_primitive_with_null_positions(
                ordered.encoded().as_::<Primitive>(),
                exec_ctx,
            )?;
            Ok(OrderedFloat::try_new(packed.into_array(), primitive.ptype())?.into_array())
        }
        FloatTransform::Alp => {
            let alp = alp_encode(primitive, None, exec_ctx)?;
            let packed = RangePacked::from_primitive_with_null_positions(
                alp.encoded().as_::<Primitive>(),
                exec_ctx,
            )?;
            let patches = alp
                .patches()
                .map(|patches| compress_patches(patches, exec_ctx))
                .transpose()?;
            Ok(ALP::try_new(packed.into_array(), alp.exponents(), patches)?.into_array())
        }
    }
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

    use super::FloatRangePackedScheme;
    use super::FloatTransform;
    use super::choose_transform;
    use crate::CascadingCompressor;

    const TEST_SCHEME: FloatRangePackedScheme = FloatRangePackedScheme::new(1.0);

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
        vortex_range_packed::initialize(&session);
        let mut ctx = session.create_execution_ctx();
        let compressed = CascadingCompressor::new(vec![&TEST_SCHEME])
            .compress(&expected.clone().into_array(), &mut ctx)?;

        assert_arrays_eq!(compressed, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn ordered_float_wins_for_close_bit_patterns() -> VortexResult<()> {
        let expected = PrimitiveArray::from_iter((0_u64..65_536).map(|index| {
            f64::from_bits(0x3ff0_0000_0000_0000 + index.wrapping_mul(7_919) % 1_024)
        }));
        let session = array_session();
        let mut ctx = session.create_execution_ctx();

        let (transform, _) = choose_transform(expected.as_view(), &mut ctx)?;

        assert_eq!(transform, FloatTransform::OrderedFloat);
        Ok(())
    }

    #[test]
    fn alp_wins_for_decimal_values() -> VortexResult<()> {
        let expected = PrimitiveArray::from_iter(
            (0_u64..65_536).map(|index| (index.wrapping_mul(7_919) % 100_000) as f64 / 100.0),
        );
        let session = array_session();
        let mut ctx = session.create_execution_ctx();

        let (transform, _) = choose_transform(expected.as_view(), &mut ctx)?;

        assert_eq!(transform, FloatTransform::Alp);
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
        vortex_range_packed::initialize(&session);
        let mut ctx = session.create_execution_ctx();
        let compressed =
            CascadingCompressor::new(vec![&TEST_SCHEME]).compress(&expected_array, &mut ctx)?;

        let packed = &compressed.children()[0];
        assert!(packed.is::<vortex_range_packed::RangePacked>());
        assert_eq!(packed.children().len(), 5);
        assert_arrays_eq!(compressed, expected, &mut ctx);
        Ok(())
    }
}
