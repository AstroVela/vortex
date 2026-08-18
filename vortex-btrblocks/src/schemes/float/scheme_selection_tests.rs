// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests to verify that each float compression scheme produces the expected encoding.

use std::f64::consts::TAU;
use std::sync::LazyLock;

use vortex_alp::ALP;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Constant;
use vortex_array::arrays::Dict;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::PrimitiveBuilder;
use vortex_array::dtype::Nullability;
use vortex_array::validity::Validity;
use vortex_block_residual::BlockResidual;
use vortex_block_residual::OrderedFloat;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_float_quant::FloatQuant;
use vortex_session::VortexSession;

use crate::BtrBlocksCompressor;

static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

#[test]
fn test_constant_compressed() -> VortexResult<()> {
    let values: Vec<f64> = vec![42.5; 100];
    let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
    let btr = BtrBlocksCompressor::default();
    let compressed = btr.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
    assert!(compressed.is::<Constant>());
    Ok(())
}

#[test]
fn test_alp_compressed() -> VortexResult<()> {
    let values: Vec<f64> = (0..1000).map(|i| (i as f64) * 0.01).collect();
    let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
    let btr = BtrBlocksCompressor::default();
    let compressed = btr.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
    assert!(compressed.is::<ALP>());
    Ok(())
}

#[test]
fn test_dict_compressed() -> VortexResult<()> {
    let distinct_values = [1.1, 2.2, 3.3, 4.4, 5.5];
    let values: Vec<f64> = (0..1000)
        .map(|i| distinct_values[i % distinct_values.len()])
        .collect();
    let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
    let btr = BtrBlocksCompressor::default();
    let compressed = btr.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
    assert!(compressed.is::<ALP>());
    assert!(compressed.children()[0].is::<Dict>());
    Ok(())
}

#[test]
fn test_null_dominated_compressed() -> VortexResult<()> {
    let mut builder = PrimitiveBuilder::<f64>::with_capacity(Nullability::Nullable, 100);
    for i in 0..5 {
        builder.append_value(i as f64);
    }
    builder.append_nulls(95);
    let array = builder.finish_into_primitive();
    let btr = BtrBlocksCompressor::default();
    let compressed = btr.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
    // Verify the compressed array preserves values.
    assert_eq!(compressed.len(), 100);
    Ok(())
}

#[test]
fn test_widened_f32_uses_float_quant() -> VortexResult<()> {
    let values = (0u32..16_384)
        .map(|index| {
            let mantissa = index.wrapping_mul(7_919) & 0x007f_ffff;
            f64::from(f32::from_bits(0x3f80_0000 | mantissa))
        })
        .collect::<Vec<_>>();
    let array = PrimitiveArray::from_iter(values).into_array();
    let compressed =
        BtrBlocksCompressor::default().compress(&array, &mut SESSION.create_execution_ctx())?;
    assert!(compressed.is::<FloatQuant>());
    Ok(())
}

#[test]
fn test_f32_does_not_use_float_quant() -> VortexResult<()> {
    let values = (0u32..16_384)
        .map(|index| {
            let mantissa = index.wrapping_mul(7_919) & 0x007f_ffff;
            f32::from_bits(0x3f80_0000 | mantissa)
        })
        .collect::<Vec<_>>();
    let array = PrimitiveArray::from_iter(values).into_array();
    let compressed =
        BtrBlocksCompressor::default().compress(&array, &mut SESSION.create_execution_ctx())?;
    assert!(!compressed.is::<FloatQuant>());
    Ok(())
}

#[test]
fn test_repeated_f64_prefers_existing_scheme() -> VortexResult<()> {
    let values = (0u32..16_384)
        .map(|index| f64::from(index % 8))
        .collect::<Vec<_>>();
    let array = PrimitiveArray::from_iter(values).into_array();
    let compressed =
        BtrBlocksCompressor::default().compress(&array, &mut SESSION.create_execution_ctx())?;
    assert!(!compressed.is::<FloatQuant>());
    Ok(())
}

#[test]
fn test_random_walk_uses_ordered_block_residual() -> VortexResult<()> {
    fn uniform(state: &mut u64) -> f64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        ((*state >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
    }

    let mut state = 0x4d59_5df4_d0f3_3173_u64;
    let mut value = 0.0_f64;
    let values = (0..65_536)
        .map(|_| {
            let radius = (-2.0 * uniform(&mut state).ln()).sqrt();
            let normal = radius * (TAU * uniform(&mut state)).cos();
            value += normal * 0.01;
            value
        })
        .collect::<Vec<_>>();
    let array = PrimitiveArray::from_iter(values).into_array();
    let compressed =
        BtrBlocksCompressor::default().compress(&array, &mut SESSION.create_execution_ctx())?;
    assert!(compressed.is::<OrderedFloat>());
    assert!(compressed.children()[0].is::<BlockResidual>());
    Ok(())
}
