// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Integer compression with one block-local reference and packed residuals.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::match_each_integer_ptype;
use vortex_block_residual::BlockResidual;
use vortex_compressor::builtins::BinaryDictScheme;
use vortex_compressor::builtins::FloatDictScheme;
use vortex_compressor::builtins::IntDictScheme;
use vortex_compressor::builtins::StringDictScheme;
use vortex_compressor::scheme::AncestorExclusion;
use vortex_compressor::scheme::ChildSelection;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexResult;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;

const BLOCK_LEN: usize = 1024;
const ESTIMATE_BLOCKS: usize = 8;
const MIN_COMPRESSION_RATIO: f64 = 1.05;

/// Compress integers with one reference and packed residuals per 1,024-value block.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct BlockResidualScheme;

impl Scheme for BlockResidualScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.block_residual"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int() && canonical.dtype().as_ptype().bit_width() >= 32
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![BlockResidual.id()]
    }

    fn ancestor_exclusions(&self) -> Vec<AncestorExclusion> {
        vec![
            AncestorExclusion {
                ancestor: IntDictScheme.id(),
                children: ChildSelection::One(1),
            },
            AncestorExclusion {
                ancestor: FloatDictScheme.id(),
                children: ChildSelection::One(1),
            },
            AncestorExclusion {
                ancestor: StringDictScheme.id(),
                children: ChildSelection::One(1),
            },
            AncestorExclusion {
                ancestor: BinaryDictScheme.id(),
                children: ChildSelection::One(1),
            },
        ]
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        if compress_ctx.finished_cascading() || compress_ctx.is_sample() {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }
        CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
            |_compressor, data, _best_so_far, _compress_ctx, exec_ctx| {
                let sample = locality_sample(data.array_as_primitive(), exec_ctx)?;
                let before_nbytes = sample.nbytes();
                let residuals = BlockResidual::from_primitive(sample.as_view())?;
                let after_nbytes = residuals.nbytes();
                if after_nbytes == 0 {
                    return Ok(EstimateVerdict::Skip);
                }

                let ratio = before_nbytes as f64 / after_nbytes as f64;
                if ratio < MIN_COMPRESSION_RATIO {
                    return Ok(EstimateVerdict::Skip);
                }
                let speed_penalty = match sample.ptype().bit_width() {
                    32 => 1.10,
                    64 => 1.20,
                    _ => unreachable!("BlockResidual only matches 32-bit and 64-bit integers"),
                };
                let adjusted_ratio = ratio / speed_penalty;
                if adjusted_ratio < MIN_COMPRESSION_RATIO {
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
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        Ok(BlockResidual::from_primitive(data.array_as_primitive())?.into_array())
    }
}

fn locality_sample(
    primitive: vortex_array::ArrayView<'_, Primitive>,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let validity = primitive
        .validity()?
        .execute_mask(primitive.len(), exec_ctx)?;
    let full_blocks = primitive.len() / BLOCK_LEN;

    if full_blocks <= ESTIMATE_BLOCKS {
        return primitive
            .array()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx);
    }

    let sample_blocks = ESTIMATE_BLOCKS.min(full_blocks);
    Ok(match_each_integer_ptype!(primitive.ptype(), |T| {
        let values = primitive.as_slice::<T>();
        let mut sample = Vec::with_capacity(sample_blocks * BLOCK_LEN);
        for sample_index in 0..sample_blocks {
            let block_index = sample_index * full_blocks / sample_blocks;
            let start = block_index * BLOCK_LEN;
            sample.extend(
                (start..start + BLOCK_LEN)
                    .map(|index| validity.value(index).then_some(values[index])),
            );
        }
        PrimitiveArray::from_option_iter(sample)
    }))
}
