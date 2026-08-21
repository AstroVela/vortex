// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Integer compression with fixed ranges and block-local offsets.

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
use vortex_block_residual::BlockResidual;
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
use crate::normalize_null_values;
use crate::schemes::fixed_bins::CoarseModel;
use crate::schemes::fixed_bins::coarse_model;
use crate::schemes::float::ALPScheme;
use crate::schemes::sample_primitive_one_percent;

const DEFAULT_DECODE_COST_FACTOR: f64 = 1.20;
const PREFILTER_MODEL_MARGIN: f64 = 1.50;
const MIN_PREFILTER_MODEL_RATIO: f64 = 1.15;
const MAX_BLOCK_TO_RANGE_RATIO: f64 = 1.10;

/// Compress integers with at most 64 range starts and block-local offsets.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct IntegerFixedBinsScheme {
    decode_cost_factor: f64,
}

impl IntegerFixedBinsScheme {
    /// Creates a scheme with the specified decode cost factor.
    pub const fn new(decode_cost_factor: f64) -> Self {
        Self { decode_cost_factor }
    }
}

impl Default for IntegerFixedBinsScheme {
    fn default() -> Self {
        Self::new(DEFAULT_DECODE_COST_FACTOR)
    }
}

impl Scheme for IntegerFixedBinsScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.fixed_bins"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![IntMult.id(), Dict.id(), BitPacked.id(), BlockResidual.id()]
    }

    fn num_children(&self) -> usize {
        2
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        let is_alp_child = compress_ctx
            .cascade_history()
            .last()
            .is_some_and(|(scheme, child)| *scheme == ALPScheme.id() && *child == 0);
        if compress_ctx.is_sample() || !is_alp_child {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        let decode_cost_factor = self.decode_cost_factor;
        let scheme_id = self.id();
        CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
            move |compressor, data, best_so_far, compress_ctx, exec_ctx| {
                let source_width = data.array_as_primitive().ptype().bit_width() as f64;
                let best_ratio = best_so_far
                    .and_then(EstimateScore::finite_ratio)
                    .unwrap_or(1.0);
                if source_width / decode_cost_factor <= best_ratio {
                    return Ok(EstimateVerdict::Skip);
                }

                let sample = sample_primitive_one_percent(data.array_as_primitive(), exec_ctx)?;
                let sample = normalize_null_values(sample.as_view(), exec_ctx)?;
                let ordered = ordered_integer_values(sample.as_view());
                let model = coarse_model(&ordered, sample.ptype().bit_width());
                if !prefilter_passes(model, best_ratio, decode_cost_factor) {
                    return Ok(EstimateVerdict::Skip);
                }

                let encoded = encode_fixed_bins(sample.as_view())?;
                let after_nbytes = encoded.nbytes();
                if after_nbytes == 0 {
                    return Ok(EstimateVerdict::Skip);
                }
                let adjusted_ratio =
                    sample.nbytes() as f64 / after_nbytes as f64 / decode_cost_factor;
                if adjusted_ratio <= best_ratio {
                    return Ok(EstimateVerdict::Skip);
                }

                let incumbent = compressor.compress_sample_without_scheme(
                    &sample.into_array(),
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
        let primitive = normalize_null_values(data.array_as_primitive(), exec_ctx)?;
        encode_fixed_bins(primitive.as_view())
    }
}

fn prefilter_passes(model: CoarseModel, best_ratio: f64, decode_cost_factor: f64) -> bool {
    model.range_ratio.is_finite()
        && model.range_ratio >= MIN_PREFILTER_MODEL_RATIO
        && model.range_ratio / decode_cost_factor * PREFILTER_MODEL_MARGIN > best_ratio
        && model.block_ratio <= model.range_ratio * MAX_BLOCK_TO_RANGE_RATIO
}

fn encode_fixed_bins(primitive: vortex_array::ArrayView<'_, Primitive>) -> VortexResult<ArrayRef> {
    let ordered = ordered_integer_values(primitive);
    let decomposition = RangeDecomposition::encode(&ordered)?;
    let code_width = decomposition.code_width();
    let codes = PrimitiveArray::new(decomposition.codes().to_vec(), primitive.validity()?);
    // SAFETY: The decomposition computes the exact code width.
    let codes = unsafe { bitpack_encode_unchecked(codes, code_width) }?.into_array();
    let starts = restore_starts(decomposition.bin_starts(), primitive.ptype())?.into_array();
    let references = DictArray::try_new(codes, starts)?.into_array();
    let offsets = restore_offsets(decomposition.offsets(), primitive.ptype())?;
    let offsets = BlockResidual::from_primitive(offsets.as_view())?.into_array();
    Ok(IntMult::try_new(references, offsets, 1)?.into_array())
}

fn ordered_integer_values(primitive: vortex_array::ArrayView<'_, Primitive>) -> Vec<u64> {
    match primitive.ptype() {
        PType::U8 => primitive
            .as_slice::<u8>()
            .iter()
            .map(|&value| u64::from(value))
            .collect(),
        PType::U16 => primitive
            .as_slice::<u16>()
            .iter()
            .map(|&value| u64::from(value))
            .collect(),
        PType::U32 => primitive
            .as_slice::<u32>()
            .iter()
            .map(|&value| u64::from(value))
            .collect(),
        PType::U64 => primitive.as_slice::<u64>().to_vec(),
        PType::I8 => primitive
            .as_slice::<i8>()
            .iter()
            .map(|&value| u64::from((value as u8) ^ (1 << 7)))
            .collect(),
        PType::I16 => primitive
            .as_slice::<i16>()
            .iter()
            .map(|&value| u64::from((value as u16) ^ (1 << 15)))
            .collect(),
        PType::I32 => primitive
            .as_slice::<i32>()
            .iter()
            .map(|&value| u64::from((value as u32) ^ (1 << 31)))
            .collect(),
        PType::I64 => primitive
            .as_slice::<i64>()
            .iter()
            .map(|&value| (value as u64) ^ (1 << 63))
            .collect(),
        ptype => unreachable!("fixed bins require integers, got {ptype}"),
    }
}

fn restore_starts(values: &[u64], ptype: PType) -> VortexResult<PrimitiveArray> {
    Ok(match ptype {
        PType::U8 => PrimitiveArray::from_iter(
            values
                .iter()
                .copied()
                .map(u8::try_from)
                .collect::<Result<Vec<_>, _>>()?,
        ),
        PType::U16 => PrimitiveArray::from_iter(
            values
                .iter()
                .copied()
                .map(u16::try_from)
                .collect::<Result<Vec<_>, _>>()?,
        ),
        PType::U32 => PrimitiveArray::from_iter(
            values
                .iter()
                .copied()
                .map(u32::try_from)
                .collect::<Result<Vec<_>, _>>()?,
        ),
        PType::U64 => PrimitiveArray::from_iter(values.iter().copied()),
        PType::I8 => PrimitiveArray::from_iter(
            values
                .iter()
                .copied()
                .map(u8::try_from)
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .map(|value| i8::from_le_bytes([(value ^ (1 << 7))])),
        ),
        PType::I16 => PrimitiveArray::from_iter(
            values
                .iter()
                .copied()
                .map(u16::try_from)
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .map(|value| i16::from_le_bytes((value ^ (1 << 15)).to_le_bytes())),
        ),
        PType::I32 => PrimitiveArray::from_iter(
            values
                .iter()
                .copied()
                .map(u32::try_from)
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .map(|value| i32::from_le_bytes((value ^ (1 << 31)).to_le_bytes())),
        ),
        PType::I64 => PrimitiveArray::from_iter(
            values
                .iter()
                .map(|&value| i64::from_le_bytes((value ^ (1 << 63)).to_le_bytes())),
        ),
        _ => vortex_error::vortex_bail!("fixed bins require integers, got {ptype}"),
    })
}

fn restore_offsets(values: &[u64], ptype: PType) -> VortexResult<PrimitiveArray> {
    Ok(match ptype {
        PType::U8 => PrimitiveArray::from_iter(
            values
                .iter()
                .copied()
                .map(u8::try_from)
                .collect::<Result<Vec<_>, _>>()?,
        ),
        PType::U16 => PrimitiveArray::from_iter(
            values
                .iter()
                .copied()
                .map(u16::try_from)
                .collect::<Result<Vec<_>, _>>()?,
        ),
        PType::U32 => PrimitiveArray::from_iter(
            values
                .iter()
                .copied()
                .map(u32::try_from)
                .collect::<Result<Vec<_>, _>>()?,
        ),
        PType::U64 => PrimitiveArray::from_iter(values.iter().copied()),
        PType::I8 => PrimitiveArray::from_iter(
            values
                .iter()
                .copied()
                .map(u8::try_from)
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .map(|value| i8::from_le_bytes([value])),
        ),
        PType::I16 => PrimitiveArray::from_iter(
            values
                .iter()
                .copied()
                .map(u16::try_from)
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .map(|value| i16::from_le_bytes(value.to_le_bytes())),
        ),
        PType::I32 => PrimitiveArray::from_iter(
            values
                .iter()
                .copied()
                .map(u32::try_from)
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .map(|value| i32::from_le_bytes(value.to_le_bytes())),
        ),
        PType::I64 => PrimitiveArray::from_iter(
            values
                .iter()
                .map(|&value| i64::from_le_bytes(value.to_le_bytes())),
        ),
        _ => vortex_error::vortex_bail!("fixed bins require integers, got {ptype}"),
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::cast_possible_truncation)]

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_error::VortexResult;

    use super::IntegerFixedBinsScheme;
    use super::encode_fixed_bins;
    use crate::CascadingCompressor;
    use crate::schemes::float::ALPScheme;
    use crate::schemes::integer::BitPackingScheme;
    use crate::schemes::integer::FoRScheme;

    const TEST_SCHEME: IntegerFixedBinsScheme = IntegerFixedBinsScheme::new(1.0);

    #[rstest]
    #[case::u8(PrimitiveArray::from_iter((0_u16..4_096).map(|index| {
        ((index % 4) * 64 + (index.wrapping_mul(37) % 8)) as u8
    })))]
    #[case::u16(PrimitiveArray::from_iter((0_u32..4_096).map(|index| {
        ((index % 4) * 16_000 + (index.wrapping_mul(37) % 32)) as u16
    })))]
    #[case::u32(PrimitiveArray::from_iter((0_u64..4_096).map(|index| {
        ((index % 4) * 1_000_000_000 + index.wrapping_mul(37) % 1_024) as u32
    })))]
    #[case::u64(PrimitiveArray::from_iter((0_u64..4_096).map(|index| {
        (index % 4) * 4_000_000_000_000_000_000 + index.wrapping_mul(37) % 1_024
    })))]
    #[case::i8(PrimitiveArray::from_iter((0_i16..4_096).map(|index| {
        (-120 + (index % 4) * 64 + (index.wrapping_mul(37) % 8)) as i8
    })))]
    #[case::i16(PrimitiveArray::from_iter((0_i32..4_096).map(|index| {
        (-30_000 + (index % 4) * 16_000 + (index.wrapping_mul(37) % 32)) as i16
    })))]
    #[case::i32(PrimitiveArray::from_iter((0_i64..4_096).map(|index| {
        (-2_000_000_000 + (index % 4) * 1_000_000_000 + index.wrapping_mul(37) % 1_024) as i32
    })))]
    #[case::i64(PrimitiveArray::from_iter((0_i64..4_096).map(|index| {
        [-8_000_000_000_000_000_000, -4_000_000_000_000_000_000, 0, 4_000_000_000_000_000_000]
            [usize::try_from(index % 4).unwrap_or_default()]
            + index.wrapping_mul(37) % 1_024
    })))]
    fn fixed_bin_tree_roundtrips(#[case] expected: PrimitiveArray) -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();
        let compressed = encode_fixed_bins(expected.as_view())?;

        assert_arrays_eq!(compressed, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn nullable_signed_values_roundtrip() -> VortexResult<()> {
        let expected = PrimitiveArray::from_option_iter((0_i64..65_536).map(|index| {
            (index % 17 != 0).then(|| {
                [
                    -8_000_000_000_000_000_000,
                    -4_000_000_000_000_000_000,
                    0,
                    4_000_000_000_000_000_000,
                ][usize::try_from(index % 4).unwrap_or_default()]
                    + index.wrapping_mul(37) % 1_024
            })
        }));
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();
        let compressed = encode_fixed_bins(expected.as_view())?;

        assert_arrays_eq!(compressed, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn selector_composes_below_alp() -> VortexResult<()> {
        let expected = PrimitiveArray::from_iter((0_i64..65_536).map(|index| {
            [0_i64, 1_000_000_000, 2_000_000_000, 3_000_000_000]
                [usize::try_from(index % 4).unwrap_or_default()] as f64
                + (index.wrapping_mul(37) % 1_024) as f64
        }));
        let expected_array = expected.clone().into_array();
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();
        let compressed = CascadingCompressor::new(vec![
            &ALPScheme,
            &TEST_SCHEME,
            &FoRScheme,
            &BitPackingScheme,
        ])
        .compress(&expected_array, &mut ctx)?;

        assert!(
            compressed.is::<vortex_alp::ALP>(),
            "expected ALP, got tree:\n{}",
            compressed.display_tree()
        );
        let encoded = &compressed.children()[0];
        assert!(
            encoded.is::<vortex_int_mult::IntMult>(),
            "expected IntMult child, got tree:\n{}",
            compressed.display_tree()
        );
        assert!(encoded.children()[0].is::<vortex_array::arrays::Dict>());
        assert!(encoded.children()[0].children()[0].is::<vortex_fastlanes::BitPacked>());
        assert!(encoded.children()[1].is::<vortex_block_residual::BlockResidual>());
        assert_arrays_eq!(compressed, expected, &mut ctx);
        Ok(())
    }
}
