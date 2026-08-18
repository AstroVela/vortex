// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Ordered float bits with one block-local reference and packed residuals.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::PType;
use vortex_block_residual::BlockResidual;
use vortex_block_residual::OrderedFloat;
use vortex_block_residual::OrderedFloatArraySlotsExt;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::EstimateScore;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexResult;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;

const BLOCK_LEN: usize = 1024;
const ESTIMATE_BLOCKS: usize = 8;
const MIN_COMPRESSION_RATIO: f64 = 1.05;
const MIN_WIN_OVER_INCUMBENT: f64 = 1.02;

/// Compress `f64` values as block-local residuals of ordered IEEE bits.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct OrderedBlockResidualScheme;

impl Scheme for OrderedBlockResidualScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.float.ordered_block_residual"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_float() && canonical.dtype().as_ptype() == PType::F64
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![OrderedFloat.id()]
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
        if compress_ctx.finished_cascading() || data.array_as_primitive().ptype() != PType::F64 {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }
        CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
            |_compressor, data, best_so_far, _compress_ctx, exec_ctx| {
                let sample = locality_sample(data.array_as_primitive(), exec_ctx)?;
                let before_nbytes = sample.nbytes();
                let ordered = OrderedFloat::from_primitive(sample.as_view())?;
                let residuals =
                    BlockResidual::from_primitive(ordered.encoded().as_::<Primitive>())?;
                let after_nbytes = residuals.nbytes();
                if after_nbytes == 0 {
                    return Ok(EstimateVerdict::Skip);
                }

                let ratio = before_nbytes as f64 / after_nbytes as f64;
                if ratio < MIN_COMPRESSION_RATIO {
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
        _compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let ordered = OrderedFloat::from_primitive(data.array_as_primitive())?;
        let ordered_values = ordered.encoded().as_::<Primitive>();
        let residuals = BlockResidual::from_primitive(ordered_values)?;
        Ok(OrderedFloat::try_new(residuals.into_array(), PType::F64)?.into_array())
    }
}

fn locality_sample(
    primitive: vortex_array::ArrayView<'_, Primitive>,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let values = primitive.as_slice::<f64>();
    let validity = primitive
        .validity()?
        .execute_mask(primitive.len(), exec_ctx)?;
    let full_blocks = primitive.len() / BLOCK_LEN;

    if full_blocks <= ESTIMATE_BLOCKS {
        return Ok(PrimitiveArray::from_option_iter(
            values
                .iter()
                .copied()
                .enumerate()
                .map(|(index, value)| validity.value(index).then_some(value)),
        ));
    }

    let sample_blocks = ESTIMATE_BLOCKS.min(full_blocks);
    let mut sample = Vec::with_capacity(sample_blocks * BLOCK_LEN);
    for sample_index in 0..sample_blocks {
        let block_index = sample_index * full_blocks / sample_blocks;
        let start = block_index * BLOCK_LEN;
        sample.extend(
            (start..start + BLOCK_LEN).map(|index| validity.value(index).then_some(values[index])),
        );
    }
    Ok(PrimitiveArray::from_option_iter(sample))
}
