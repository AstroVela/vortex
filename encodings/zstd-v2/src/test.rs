// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::assert_arrays_eq;
use vortex_array::compute::conformance::consistency::test_array_consistency;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_array::session::ArraySessionExt as _;
use vortex_session::VortexSession;

use crate::ZstdV2;
use crate::ZstdV2Data;

const N: usize = 512;
const VALUES_PER_FRAME: usize = 64;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = array_session();
    crate::initialize(&session);
    session
});

fn ctx() -> ExecutionCtx {
    SESSION.create_execution_ctx()
}

fn values(nullable: bool) -> VarBinViewArray {
    if nullable {
        VarBinViewArray::from_iter_nullable_str(
            (0..N).map(|i| (i % 5 != 0).then(|| format!("value-{i:05}-{}", "x".repeat(i % 17)))),
        )
    } else {
        VarBinViewArray::from_iter_str(
            (0..N).map(|i| format!("value-{i:05}-{}", "x".repeat(i % 17))),
        )
    }
}

fn encoded(nullable: bool, values_per_frame: usize) -> VortexResult<ArrayRef> {
    Ok(ZstdV2::from_var_bin_view(&values(nullable), 3, values_per_frame, &mut ctx())?.into_array())
}

#[rstest]
#[case::single_frame(false, 0)]
#[case::many_frames(false, VALUES_PER_FRAME)]
#[case::single_frame_nullable(true, 0)]
#[case::many_frames_nullable(true, VALUES_PER_FRAME)]
fn test_roundtrip(#[case] nullable: bool, #[case] values_per_frame: usize) -> VortexResult<()> {
    let mut ctx = ctx();
    let array = encoded(nullable, values_per_frame)?;
    assert_eq!(array.len(), N);
    assert_arrays_eq!(array, values(nullable).into_array(), &mut ctx);
    Ok(())
}

#[rstest]
#[case::single_frame(false, 0)]
#[case::many_frames(false, VALUES_PER_FRAME)]
#[case::many_frames_nullable(true, VALUES_PER_FRAME)]
fn test_slice_matches_canonical(
    #[case] nullable: bool,
    #[case] values_per_frame: usize,
) -> VortexResult<()> {
    let mut ctx = ctx();
    let array = encoded(nullable, values_per_frame)?;
    for range in [0..N, 0..1, 100..300, 511..512, 256..256] {
        let expected = values(nullable).into_array().slice(range.clone())?;
        assert_arrays_eq!(array.clone().slice(range)?, expected, &mut ctx);
    }
    Ok(())
}

fn assert_filter_matches_canonical(array: &ArrayRef, mask: Mask) -> VortexResult<()> {
    let mut ctx = ctx();
    let expected = array
        .clone()
        .execute::<Canonical>(&mut ctx)?
        .into_array()
        .filter(mask.clone())?;
    assert_arrays_eq!(array.clone().filter(mask)?, expected, &mut ctx);
    Ok(())
}

#[rstest]
#[case::single_frame(false, 0)]
#[case::many_frames(false, VALUES_PER_FRAME)]
#[case::many_frames_nullable(true, VALUES_PER_FRAME)]
fn test_filter_matches_canonical(
    #[case] nullable: bool,
    #[case] values_per_frame: usize,
) -> VortexResult<()> {
    let array = encoded(nullable, values_per_frame)?;
    assert_filter_matches_canonical(&array, Mask::from_indices(N, [N - 1]))?;
    assert_filter_matches_canonical(&array, Mask::from_indices(N, [0, 5, 200, 201, 511]))?;
    assert_filter_matches_canonical(&array, Mask::from_iter((0..N).map(|i| i % 3 == 0)))?;
    Ok(())
}

#[test]
fn test_filter_of_a_slice_matches_canonical() -> VortexResult<()> {
    let sliced = encoded(true, VALUES_PER_FRAME)?.slice(100..300)?;
    assert_filter_matches_canonical(&sliced, Mask::from_indices(200, [0, 150, 199]))?;
    assert_filter_matches_canonical(&sliced, Mask::from_iter((0..200).map(|i| i % 7 == 0)))?;
    Ok(())
}

/// Truncating a frame's compressed bytes leaves its header, and so its metadata, intact: the
/// frame only fails when something decompresses it.
fn with_truncated_frame(array: ArrayRef, frame: usize) -> VortexResult<ArrayRef> {
    let zstd = array.as_::<ZstdV2>();
    let dtype = zstd.dtype().clone();
    let validity = zstd.validity()?;
    let mut data: ZstdV2Data = zstd.data().clone();
    let bytes = data.frames[frame].as_slice();
    data.frames[frame] = ByteBuffer::from(bytes[..bytes.len() - 8].to_vec());
    Ok(ZstdV2::try_new(dtype, data, validity)?.into_array())
}

#[test]
fn test_filter_skips_frames_holding_no_selected_value() -> VortexResult<()> {
    let mut ctx = ctx();
    // Frame 1 holds values 64..128, and is the only unreadable frame.
    let array = with_truncated_frame(encoded(false, VALUES_PER_FRAME)?, 1)?;

    let kept = array
        .filter(Mask::from_indices(N, [5, 200, 511]))?
        .execute::<Canonical>(&mut ctx)?;
    assert_eq!(kept.into_array().len(), 3);

    assert!(
        array
            .filter(Mask::from_indices(N, [100]))?
            .execute::<Canonical>(&mut ctx)
            .is_err(),
        "a filter reading the truncated frame should fail"
    );
    Ok(())
}

#[test]
fn test_slice_skips_frames_outside_it() -> VortexResult<()> {
    let mut ctx = ctx();
    let array = with_truncated_frame(encoded(false, VALUES_PER_FRAME)?, 1)?;
    // Values 64..128 live in the broken frame, so a slice clear of them must still decode.
    let sliced = array.slice(200..300)?.execute::<Canonical>(&mut ctx)?;
    assert_eq!(sliced.into_array().len(), 100);

    assert!(
        array
            .slice(64..128)?
            .execute::<Canonical>(&mut ctx)
            .is_err(),
        "a slice covering the truncated frame should fail"
    );
    Ok(())
}

#[test]
fn test_empty_and_all_null() -> VortexResult<()> {
    let mut ctx = ctx();
    let empty = VarBinViewArray::from_iter_str(Vec::<String>::new());
    let array = ZstdV2::from_var_bin_view(&empty, 3, 0, &mut ctx)?;
    assert_eq!(array.len(), 0);

    let all_null = VarBinViewArray::from_iter_nullable_str((0..16).map(|_| None::<String>));
    let array = ZstdV2::from_var_bin_view(&all_null, 3, 4, &mut ctx)?.into_array();
    assert_arrays_eq!(array, all_null.into_array(), &mut ctx);
    Ok(())
}

/// The file writer serializes arrays and reads them back, which unit tests that keep an array in
/// memory never exercise.
#[test]
fn test_serde_roundtrip() -> VortexResult<()> {
    let mut ctx = ctx();
    let array = encoded(true, VALUES_PER_FRAME)?;
    let serialized = SESSION.array_serialize(&array)?;
    assert!(serialized.is_some(), "ZstdV2 must be serializable");
    assert_arrays_eq!(array, values(true).into_array(), &mut ctx);
    Ok(())
}

#[test]
fn test_consistency() {
    let mut ctx = ctx();
    let array = encoded(true, VALUES_PER_FRAME).unwrap_or_else(|e| panic!("encoding: {e}"));
    test_array_consistency(&array, &mut ctx);
}

/// Columns of empty strings are common in the wild, and make every frame zero bytes long.
#[rstest]
#[case::single_frame(0)]
#[case::many_frames(64)]
fn test_all_empty_values(#[case] values_per_frame: usize) -> VortexResult<()> {
    let mut ctx = ctx();
    let empty = VarBinViewArray::from_iter_str((0..N).map(|_| String::new()));
    let array = ZstdV2::from_var_bin_view(&empty, 3, values_per_frame, &mut ctx)?.into_array();
    assert_arrays_eq!(array.clone(), empty.into_array(), &mut ctx);
    assert_arrays_eq!(
        array.slice(10..20)?,
        VarBinViewArray::from_iter_str((0..10).map(|_| String::new())).into_array(),
        &mut ctx
    );
    Ok(())
}

/// A column that is mostly empty strings puts many values in a frame of very few bytes.
#[test]
fn test_mostly_empty_values() -> VortexResult<()> {
    let mut ctx = ctx();
    let sparse = VarBinViewArray::from_iter_str(
        (0..N).map(|i| if i % 97 == 0 { format!("value-{i}") } else { String::new() }),
    );
    let array = ZstdV2::from_var_bin_view(&sparse, 3, 64, &mut ctx)?.into_array();
    assert_arrays_eq!(array, sparse.into_array(), &mut ctx);
    Ok(())
}
