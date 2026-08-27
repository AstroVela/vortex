// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::SliceArray;
use vortex_array::arrays::slice::SliceKernel;
use vortex_array::assert_arrays_eq;
use vortex_array::assert_nth_scalar;
use vortex_array::compute::conformance::consistency::test_array_consistency;
use vortex_array::serde::SerializeOptions;
use vortex_array::serde::SerializedArray;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::BitPackedV2;
use crate::BitPackedV2Array;
use crate::BitPackedV2ArrayExt;
use crate::FL_CHUNK_SIZE;
use crate::bitpack_compress::bitpack_to_best_bit_width;
use crate::bitpack_v2_compress::bitpack_v2_encode;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    crate::initialize(&session);
    session
});

/// Values whose magnitude depends on which FastLanes chunk they land in: chunk `c` holds values
/// needing roughly `2 * c + 1` bits.
fn staircase(chunks: usize) -> Vec<u32> {
    (0..chunks * FL_CHUNK_SIZE)
        .map(|i| {
            let chunk = i / FL_CHUNK_SIZE;
            let width = 2 * chunk as u32 + 1;
            ((i as u32) % (1 << width)).max(1 << (width - 1))
        })
        .collect()
}

fn encode(values: &PrimitiveArray) -> VortexResult<BitPackedV2Array> {
    bitpack_v2_encode(values, &mut SESSION.create_execution_ctx())
}

fn assert_roundtrip(values: PrimitiveArray) -> VortexResult<BitPackedV2Array> {
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = encode(&values)?;
    let decoded = encoded
        .as_array()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?;
    assert_arrays_eq!(decoded, values, &mut ctx);
    Ok(encoded)
}

#[test]
fn each_chunk_gets_its_own_width() -> VortexResult<()> {
    let values = PrimitiveArray::from_iter(staircase(4));
    let encoded = assert_roundtrip(values)?;

    assert_eq!(encoded.bit_widths(), &[1, 3, 5, 7]);
    // The packed buffer holds exactly the sum of the per-chunk block sizes, which is far less
    // than packing the whole array at the widest chunk's width.
    assert_eq!(encoded.packed().len(), 128 * (1 + 3 + 5 + 7));
    Ok(())
}

#[test]
fn beats_single_width_bitpacking() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = PrimitiveArray::from_iter(staircase(8));

    let v1 = bitpack_to_best_bit_width(&values, &mut ctx)?;
    let v2 = encode(&values)?;

    assert!(
        v2.packed().len() < v1.packed().len(),
        "v2 packed {} should be smaller than v1 packed {}",
        v2.packed().len(),
        v1.packed().len(),
    );
    Ok(())
}

#[rstest]
// A partial trailing chunk.
#[case(PrimitiveArray::from_iter((0..2500u32).map(|i| i % 64)))]
// A single chunk that needs the full type width.
#[case(PrimitiveArray::from_iter([u32::MAX; FL_CHUNK_SIZE]))]
// Constant zeros, which pack to a zero-width chunk.
#[case(PrimitiveArray::from_iter([0u32; 3 * FL_CHUNK_SIZE]))]
// Zero-width and full-width chunks side by side.
#[case(PrimitiveArray::from_iter((0..2 * FL_CHUNK_SIZE).map(|i| if i < FL_CHUNK_SIZE { 0 } else { u32::MAX })))]
// Nulls, whose undefined values must not force a wider chunk.
#[case(PrimitiveArray::from_option_iter((0..2 * FL_CHUNK_SIZE).map(|i| (i % 3 != 0).then_some((i % 17) as u32))))]
// Signed values.
#[case(PrimitiveArray::from_iter((0..2 * FL_CHUNK_SIZE).map(|i| (i % 97) as i64)))]
// Narrow types.
#[case(PrimitiveArray::from_iter((0..3 * FL_CHUNK_SIZE).map(|i| (i % 7) as u8)))]
// Empty.
#[case(PrimitiveArray::from_iter(Vec::<u32>::new()))]
fn roundtrips(#[case] values: PrimitiveArray) -> VortexResult<()> {
    assert_roundtrip(values)?;
    Ok(())
}

#[test]
fn outliers_become_patches() -> VortexResult<()> {
    // Huge values in two chunks are cheaper to patch than to widen either whole chunk.
    let mut values: Vec<u64> = (0..3 * FL_CHUNK_SIZE).map(|i| (i % 32) as u64).collect();
    values[FL_CHUNK_SIZE + 7] = u64::MAX;
    values[2 * FL_CHUNK_SIZE + 9] = u64::MAX;

    let encoded = assert_roundtrip(PrimitiveArray::from_iter(values))?;
    let patches = encoded
        .patches()
        .ok_or_else(|| vortex_err!("outliers must be patched"))?;
    let mut ctx = SESSION.create_execution_ctx();
    let indices = patches
        .indices()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?;
    let offsets = patches
        .chunk_offsets()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?;
    assert_eq!(indices.as_slice::<u16>(), &[7, 9]);
    assert_eq!(offsets.as_slice::<u32>(), &[0, 0, 1, 2]);
    assert_eq!(encoded.bit_widths(), &[5, 5, 5]);
    Ok(())
}

#[test]
fn sliced_patches_v2_serde_round_trip() -> VortexResult<()> {
    let mut values: Vec<u64> = (0..3 * FL_CHUNK_SIZE).map(|i| (i % 32) as u64).collect();
    values[FL_CHUNK_SIZE + 7] = u64::MAX;
    let values = PrimitiveArray::from_iter(values);
    let range = 512..2500;

    let mut ctx = SESSION.create_execution_ctx();
    let encoded = BitPackedV2::slice(encode(&values)?.as_view(), range.clone(), &mut ctx)?
        .ok_or_else(|| vortex_err!("BitPackedV2 slice kernel did not return an array"))?;
    let expected = values.into_array().slice(range)?;
    let array_ctx = ArrayContext::empty();
    let serialized = encoded.serialize(&array_ctx, &SESSION, &SerializeOptions::default())?;
    let mut bytes = ByteBufferMut::empty();
    for buffer in serialized {
        bytes.extend_from_slice(buffer.as_ref());
    }
    let decoded = SerializedArray::try_from(bytes.freeze())?.decode(
        encoded.dtype(),
        encoded.len(),
        &ReadContext::new(array_ctx.to_ids()),
        &SESSION,
    )?;

    assert!(decoded.is::<BitPackedV2>());
    assert_arrays_eq!(decoded, expected, &mut ctx);
    Ok(())
}

#[test]
fn nulls_do_not_widen_a_chunk() -> VortexResult<()> {
    // Null slots hold a large value that must not be paid for, since it is never read back.
    let values = PrimitiveArray::from_option_iter(
        (0..2 * FL_CHUNK_SIZE)
            .map(|i| (i % 2 == 0).then_some(if i < FL_CHUNK_SIZE { 1u32 } else { 3 })),
    );
    let encoded = assert_roundtrip(values)?;
    assert_eq!(encoded.bit_widths(), &[1, 2]);
    Ok(())
}

#[rstest]
#[case(0..2048)]
#[case(1024..2048)]
#[case(512..1500)]
#[case(1..4095)]
#[case(2047..2048)]
#[case(1024..1024)]
fn slices_keep_chunk_widths(#[case] range: std::ops::Range<usize>) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = PrimitiveArray::from_iter(staircase(4));
    let encoded = encode(&values)?.into_array();

    let sliced = encoded.slice(range.clone())?;
    let expected = values.into_array().slice(range)?;
    assert_arrays_eq!(
        sliced.execute::<PrimitiveArray>(&mut ctx)?,
        expected.execute::<PrimitiveArray>(&mut ctx)?,
        &mut ctx
    );
    Ok(())
}

#[test]
fn slice_reduces_to_bitpacked_v2() -> VortexResult<()> {
    let values = PrimitiveArray::from_iter(staircase(4));
    let encoded = encode(&values)?.into_array();

    let reduced = encoded
        .reduce_parent(
            &SliceArray::new(encoded.clone(), 1500..3000).into_array(),
            0,
        )?
        .expect("expected the slice rule to fire");

    assert!(reduced.is::<BitPackedV2>());
    let reduced = reduced.as_::<BitPackedV2>();
    assert_eq!(reduced.offset(), 476);
    assert_eq!(reduced.bit_widths(), &[3, 5]);
    assert_eq!(reduced.as_ref().len(), 1500);
    Ok(())
}

#[test]
fn scalar_at_reads_the_right_chunk() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let raw = staircase(3);
    let encoded = encode(&PrimitiveArray::from_iter(raw.clone()))?;

    for index in [0usize, 1, 1023, 1024, 2047, 2048, 3071] {
        assert_nth_scalar!(encoded, index, raw[index], &mut ctx);
    }
    Ok(())
}

#[rstest]
#[case(PrimitiveArray::from_iter(staircase(3)))]
#[case(PrimitiveArray::from_option_iter((0..2 * FL_CHUNK_SIZE).map(|i| (i % 5 != 0).then_some((i % 33) as u32))))]
fn consistency(#[case] values: PrimitiveArray) -> VortexResult<()> {
    let encoded: ArrayRef = encode(&values)?.into_array();
    test_array_consistency(&encoded, &mut SESSION.create_execution_ctx());
    Ok(())
}
