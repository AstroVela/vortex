// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Lossless FloatMult split for exact-multiple f32 values.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::PType;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::EstimateScore;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexResult;
use vortex_float_quant::FloatMult;
use vortex_float_quant::FloatMultArraySlotsExt;
use vortex_float_quant::estimate_float_mult_constant_base;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;

const SAMPLE_RUNS: usize = 16;
const SAMPLE_RUN_LEN: usize = 128;
const SELECTION_PENALTY: f64 = 0.85;
const MIN_ESTIMATED_RATIO: f64 = 1.50;
const MIN_WIN_OVER_INCUMBENT: f64 = 1.05;
const MAX_PRIMARY_BIT_WIDTH: u32 = 17;

/// FloatMult split with normal BtrBlocks compression for an integer latent child.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FloatMultScheme;

impl Scheme for FloatMultScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.float.float_mult"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_float() && canonical.dtype().as_ptype() == PType::F32
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![FloatMult.id()]
    }

    fn num_children(&self) -> usize {
        1
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        let primitive = data.array_as_primitive();
        if compress_ctx.finished_cascading()
            || primitive.ptype() != PType::F32
            || primitive.len() < SAMPLE_RUN_LEN
        {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
            |compressor, data, best_so_far, compress_ctx, exec_ctx| {
                let sample = sample_primitive(data.array_as_primitive(), exec_ctx)?;
                let Some(base) = estimate_float_mult_constant_base(sample.as_view()) else {
                    return Ok(EstimateVerdict::Skip);
                };
                let Some(encoded) =
                    FloatMult::from_primitive_constant_secondary(sample.as_view(), base)?
                else {
                    return Ok(EstimateVerdict::Skip);
                };
                if primary_bit_width(encoded.primary().as_::<Primitive>()) > MAX_PRIMARY_BIT_WIDTH {
                    return Ok(EstimateVerdict::Skip);
                }
                let compressed_primary = compressor.compress_child(
                    encoded.primary(),
                    &compress_ctx,
                    FloatMultScheme.id(),
                    0,
                    exec_ctx,
                )?;
                let candidate = FloatMult::try_new(
                    compressed_primary,
                    encoded.secondary().cloned(),
                    sample.ptype(),
                    base,
                )?;
                let after_nbytes = candidate.nbytes();
                if after_nbytes == 0 {
                    return Ok(EstimateVerdict::Skip);
                }
                let ratio = sample.nbytes() as f64 / after_nbytes as f64 * SELECTION_PENALTY;
                if ratio < MIN_ESTIMATED_RATIO {
                    return Ok(EstimateVerdict::Skip);
                }
                let incumbent = best_so_far.and_then(EstimateScore::finite_ratio);
                if incumbent.is_some_and(|best| ratio < best * MIN_WIN_OVER_INCUMBENT) {
                    return Ok(EstimateVerdict::Skip);
                }
                Ok(EstimateVerdict::Ratio(ratio))
            },
        )))
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let primitive = data.array_as_primitive();
        let sample = sample_primitive(primitive, exec_ctx)?;
        let Some(base) = estimate_float_mult_constant_base(sample.as_view()) else {
            return Ok(primitive.array().clone());
        };
        let Some(encoded) = FloatMult::from_primitive_constant_secondary(primitive, base)? else {
            return Ok(primitive.array().clone());
        };
        let compressed_primary =
            compressor.compress_child(encoded.primary(), &compress_ctx, self.id(), 0, exec_ctx)?;
        Ok(FloatMult::try_new(
            compressed_primary,
            encoded.secondary().cloned(),
            primitive.ptype(),
            base,
        )?
        .into_array())
    }
}

fn primary_bit_width(primary: vortex_array::ArrayView<'_, Primitive>) -> u32 {
    match primary.ptype() {
        PType::U32 => {
            let (minimum, maximum) = primary
                .as_slice::<u32>()
                .iter()
                .copied()
                .fold((u32::MAX, u32::MIN), |(minimum, maximum), value| {
                    (minimum.min(value), maximum.max(value))
                });
            u32::BITS - (maximum - minimum).leading_zeros()
        }
        PType::U64 => {
            let (minimum, maximum) = primary
                .as_slice::<u64>()
                .iter()
                .copied()
                .fold((u64::MAX, u64::MIN), |(minimum, maximum), value| {
                    (minimum.min(value), maximum.max(value))
                });
            u64::BITS - (maximum - minimum).leading_zeros()
        }
        _ => unreachable!(),
    }
}

fn sample_primitive(
    primitive: vortex_array::ArrayView<'_, Primitive>,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let sample_len = SAMPLE_RUNS * SAMPLE_RUN_LEN;
    if primitive.len() <= sample_len {
        return primitive
            .array()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx);
    }

    let validity = primitive
        .validity()?
        .execute_mask(primitive.len(), exec_ctx)?;
    macro_rules! sample {
        ($T:ty) => {{
            let values = primitive.as_slice::<$T>();
            let mut sample = Vec::with_capacity(sample_len);
            for run_index in 0..SAMPLE_RUNS {
                let partition_start = run_index * primitive.len() / SAMPLE_RUNS;
                let next_partition = (run_index + 1) * primitive.len() / SAMPLE_RUNS;
                let start =
                    partition_start + (next_partition - partition_start - SAMPLE_RUN_LEN) / 2;
                sample.extend(
                    (start..start + SAMPLE_RUN_LEN)
                        .map(|index| validity.value(index).then_some(values[index])),
                );
            }
            PrimitiveArray::from_option_iter(sample)
        }};
    }
    Ok(match primitive.ptype() {
        PType::F32 => sample!(f32),
        PType::F64 => sample!(f64),
        _ => unreachable!(),
    })
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::*;
    use crate::BtrBlocksCompressorBuilder;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = array_session();
        vortex_float_quant::initialize(&session);
        session
    });

    #[test]
    fn candidate_never_increases_noisy_value_size() -> VortexResult<()> {
        let values = (0u32..65_536)
            .map(|index| f32::from_bits(0x3f80_0000 | index.wrapping_mul(7_919) & 0x007f_ffff))
            .collect::<Vec<_>>();
        let original = PrimitiveArray::from_iter(values).into_array();
        let baseline = BtrBlocksCompressorBuilder::default()
            .exclude_schemes([FloatMultScheme.id()])
            .build()
            .compress(&original, &mut SESSION.create_execution_ctx())?;
        let candidate = BtrBlocksCompressorBuilder::default()
            .build()
            .compress(&original, &mut SESSION.create_execution_ctx())?;
        assert!(
            candidate.nbytes() <= baseline.nbytes(),
            "FloatMult candidate uses {} bytes and baseline uses {} bytes",
            candidate.nbytes(),
            baseline.nbytes()
        );
        assert_arrays_eq!(candidate, original, &mut SESSION.create_execution_ctx());
        Ok(())
    }
}
