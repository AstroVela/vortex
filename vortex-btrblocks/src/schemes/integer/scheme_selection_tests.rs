// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests to verify that each integer compression scheme produces the expected encoding.

use std::iter;
use std::sync::LazyLock;

use rand::Rng;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Constant;
use vortex_array::arrays::Dict;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::expr::stats::Precision;
use vortex_array::expr::stats::Stat;
use vortex_array::expr::stats::StatsProviderExt;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::FoR;
use vortex_fastlanes::RLE;
use vortex_runend::RunEnd;
use vortex_sequence::Sequence;
use vortex_session::VortexSession;
use vortex_sparse::Sparse;

use crate::BtrBlocksCompressor;
use crate::BtrBlocksCompressorBuilder;
use crate::SchemeExt;
use crate::schemes::integer::RunEndScheme;
static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

#[test]
fn test_constant_compressed() -> VortexResult<()> {
    let values: Vec<i32> = iter::repeat_n(42, 100).collect();
    let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
    let btr = BtrBlocksCompressor::default();
    let compressed = btr.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
    assert!(compressed.is::<Constant>());
    Ok(())
}

#[test]
fn test_for_compressed() -> VortexResult<()> {
    let values: Vec<i32> = (0..1000).map(|i| 1_000_000 + ((i * 37) % 100)).collect();
    let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
    let btr = BtrBlocksCompressor::default();
    let compressed = btr.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
    assert!(compressed.is::<FoR>());
    Ok(())
}

#[test]
fn test_bitpacking_compressed() -> VortexResult<()> {
    let values: Vec<u32> = (0..1000).map(|i| i % 16).collect();
    let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
    let btr = BtrBlocksCompressor::default();
    let compressed = btr.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
    assert!(compressed.is::<BitPacked>());
    assert_eq!(
        compressed.statistics().get_as::<u64>(Stat::NullCount),
        Precision::exact(0u64)
    );
    assert_eq!(
        compressed.statistics().get_as::<u32>(Stat::Min),
        Precision::exact(0u32)
    );
    assert_eq!(
        compressed.statistics().get_as::<u32>(Stat::Max),
        Precision::exact(15u32)
    );
    Ok(())
}

#[test]
fn test_sparse_compressed() -> VortexResult<()> {
    let mut values: Vec<i32> = Vec::new();
    for i in 0..1000 {
        if i % 20 == 0 {
            values.push(2_000_000 + (i * 7) % 1000);
        } else {
            values.push(1_000_000);
        }
    }
    let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
    let btr = BtrBlocksCompressor::default();
    let compressed = btr.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
    assert!(compressed.is::<Sparse>());
    Ok(())
}

#[test]
fn test_dict_compressed() -> VortexResult<()> {
    let mut codes = Vec::with_capacity(65_535);
    let numbers: Vec<i32> = [0, 10, 50, 100, 1000, 3000]
        .into_iter()
        .map(|i| 12340 * i) // must be big enough to not prefer fastlanes.bitpacked
        .collect();

    let mut rng = StdRng::seed_from_u64(1u64);
    while codes.len() < 64000 {
        let run_length = rng.next_u32() % 5;
        let value = numbers[rng.next_u32() as usize % numbers.len()];
        for _ in 0..run_length {
            codes.push(value);
        }
    }

    let array = PrimitiveArray::new(Buffer::copy_from(&codes), Validity::NonNullable);
    let btr = BtrBlocksCompressor::default();
    let compressed = btr.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
    assert!(compressed.is::<Dict>());
    Ok(())
}

#[test]
fn test_runend_compressed() -> VortexResult<()> {
    let mut values: Vec<i32> = Vec::new();
    for i in 0..100 {
        values.extend(iter::repeat_n((i32::MAX - 50).wrapping_add(i), 10));
    }
    let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
    let btr = BtrBlocksCompressor::default();
    let compressed = btr.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
    assert!(compressed.is::<RunEnd>());
    Ok(())
}

#[test]
fn test_sequence_compressed() -> VortexResult<()> {
    let values: Vec<i32> = (0..1000).map(|i| i * 7).collect();
    let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
    let btr = BtrBlocksCompressor::default();
    let compressed = btr.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
    assert!(compressed.is::<Sequence>());
    Ok(())
}

#[test]
fn test_rle_compressed() -> VortexResult<()> {
    let mut values: Vec<i32> = Vec::new();
    for i in 0..1024 {
        // Scramble the per-run value so the data is run-length-dominant but not monotone: this
        // keeps RunEnd the winner instead of Delta (whose residuals would be small on a smooth
        // ramp).
        let v = (i as u32).wrapping_mul(2_654_435_761) as i32;
        values.extend(iter::repeat_n(v, 10));
    }
    let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
    let btr = BtrBlocksCompressor::default();
    let compressed = btr.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
    eprintln!("{}", compressed.display_tree());
    assert!(compressed.is::<RunEnd>());
    Ok(())
}

/// Wide values whose runs are only two elements long. RunEnd needs a run-end position per run,
/// which at this run length costs more than the values themselves, but RLE pays for position once
/// per element rather than once per run and still halves the array. The old fixed
/// `average_run_length >= 4` gate skipped RLE here and the column fell back to bit-packing,
/// which cannot shrink full-width scattered values at all.
#[test]
fn test_rle_compressed_short_runs_wide_values() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let mut values: Vec<i64> = Vec::new();
    for i in 0..32_768u64 {
        // Scatter the run values across the full 64-bit range so neither FoR nor BitPacking can
        // narrow them, and so Delta's residuals stay wide.
        let v = i.wrapping_mul(0x9E37_79B9_7F4A_7C15) as i64;
        values.extend(iter::repeat_n(v, 2));
    }
    let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
    let uncompressed_nbytes = array.clone().into_array().nbytes();

    // Force the RLE arm: on this data RunEnd is what the sampled estimate would otherwise pick.
    let btr = BtrBlocksCompressorBuilder::default()
        .exclude_schemes([RunEndScheme.id()])
        .build();
    let compressed = btr.compress(&array.clone().into_array(), &mut ctx)?;

    assert!(
        contains_rle(&compressed),
        "expected RLE, got tree:\n{}",
        compressed.display_tree()
    );
    // Halving is the floor: one dictionary entry per two elements. On top of that the index array
    // costs ~9 bits per element bit-packed, or ~1.4 bits once the unstable Delta cascade turns it
    // into a run-start bitmap.
    let bound = if cfg!(feature = "unstable_encodings") {
        uncompressed_nbytes * 6 / 10
    } else {
        uncompressed_nbytes * 7 / 10
    };
    assert!(
        compressed.nbytes() < bound,
        "expected < {bound} bytes, got {}",
        compressed.nbytes()
    );
    assert_arrays_eq!(compressed, array.into_array(), &mut ctx);
    Ok(())
}

/// A one-bit-wide column bit-packs to a single bit per element, which RLE's positional index can
/// never undercut however long the runs are. It must not be selected, no matter how run-heavy.
#[test]
fn test_rle_skipped_for_boolean_like_column() -> VortexResult<()> {
    let mut values: Vec<i32> = Vec::new();
    let mut rng = StdRng::seed_from_u64(11u64);
    let mut value = 0i32;
    while values.len() < 32_768 {
        let run_length = 20 + (rng.next_u32() % 40) as usize;
        values.extend(iter::repeat_n(value, run_length));
        value ^= 1;
    }
    let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
    let btr = BtrBlocksCompressor::default();
    let compressed = btr.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
    assert!(
        !contains_rle(&compressed),
        "RLE cannot beat a 1-bit bit-pack, got tree:\n{}",
        compressed.display_tree()
    );
    Ok(())
}

/// FastLanes Delta emits `1024 / bit_width` bases per chunk, so the RLE index array has to keep
/// its natural u16 width: narrowing to u8 doubles the bases for a byte-identical delta payload,
/// and a u8 index needing a full 8 bits cannot be bit-packed at all.
#[cfg(feature = "unstable_encodings")]
#[test]
fn test_rle_indices_are_not_narrowed_before_delta() -> VortexResult<()> {
    use vortex_array::dtype::PType;
    use vortex_error::VortexExpect;
    use vortex_fastlanes::RLEArraySlotsExt;

    let mut ctx = SESSION.create_execution_ctx();
    // Runs of exactly 4 put 256 runs in every 1024-element chunk, so the largest chunk-local run
    // index is 255: the widest value that still narrows to u8, and the one bit-packing refuses.
    let mut values: Vec<i64> = Vec::new();
    for i in 0..16_384u64 {
        values.extend(iter::repeat_n(
            i.wrapping_mul(0x9E37_79B9_7F4A_7C15) as i64,
            4,
        ));
    }
    let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);

    let btr = BtrBlocksCompressor::default();
    let compressed = btr.compress(&array.clone().into_array(), &mut ctx)?;

    let rle = find_rle(&compressed).unwrap_or_else(|| {
        panic!(
            "expected an RLE array, got tree:\n{}",
            compressed.display_tree()
        )
    });
    let rle = rle.as_opt::<RLE>().vortex_expect("checked by find_rle");
    assert_eq!(rle.indices().dtype().as_ptype(), PType::U16);
    assert_arrays_eq!(compressed, array.into_array(), &mut ctx);
    Ok(())
}

/// Returns the first `RLE` array in the tree, if any.
fn find_rle(array: &ArrayRef) -> Option<ArrayRef> {
    if array.is::<RLE>() {
        return Some(array.clone());
    }
    array.children().iter().find_map(find_rle)
}

/// Returns true if any `RLE` array appears in the tree.
fn contains_rle(array: &ArrayRef) -> bool {
    find_rle(array).is_some()
}

/// A strictly-increasing column with small, irregular steps: not a perfect arithmetic sequence
/// (so Sequence skips), all-unique with no runs (so RunEnd/Dict skip), and a wide absolute range.
/// Delta's residuals are far smaller than the FoR span, so Delta should win and round-trip, and
/// it must appear at most once in the tree.
#[cfg(feature = "unstable_encodings")]
#[test]
fn test_delta_compressed() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    use vortex_array::assert_arrays_eq;
    use vortex_fastlanes::Delta;

    let mut rng = StdRng::seed_from_u64(7u64);
    let mut value = 500_000i32;
    let values: Vec<i32> = (0..4096)
        .map(|_| {
            value += 1 + (rng.next_u32() % 6) as i32;
            value
        })
        .collect();
    let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);

    let btr = BtrBlocksCompressor::default();
    let compressed = btr.compress(
        &array.clone().into_array(),
        &mut SESSION.create_execution_ctx(),
    )?;
    assert!(
        compressed.is::<Delta>(),
        "expected Delta, got tree:\n{}",
        compressed.display_tree()
    );
    // Delta must appear at most once per tree: no Delta node may be nested under another.
    assert!(
        !has_nested_delta(&compressed, false),
        "Delta was applied more than once in the tree:\n{}",
        compressed.display_tree()
    );
    assert_arrays_eq!(compressed, array.into_array(), &mut ctx);
    Ok(())
}

/// Returns true if any `Delta` array appears below an ancestor `Delta` in the tree.
#[cfg(feature = "unstable_encodings")]
fn has_nested_delta(array: &ArrayRef, under_delta: bool) -> bool {
    use vortex_fastlanes::Delta;

    let is_delta = array.is::<Delta>();
    if is_delta && under_delta {
        return true;
    }
    array
        .children()
        .iter()
        .any(|child| has_nested_delta(child, under_delta || is_delta))
}

/// The RLE scheme delta-encodes its own indices by hand rather than letting the cascade select
/// `DeltaScheme`, so the hand-built Delta layer has to be recorded in the cascade history. If it
/// is not, the exclusion rules cannot see it and the cascade delta-encodes the Delta bases again,
/// producing `rle(indices=delta(bases=delta(..)))`.
#[cfg(feature = "unstable_encodings")]
#[test]
fn test_rle_indices_delta_is_not_nested() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let mut rng = StdRng::seed_from_u64(11u64);
    let mut values: Vec<i64> = Vec::with_capacity(1 << 15);
    while values.len() < (1 << 15) {
        let value = rng.random::<i64>();
        for _ in 0..rng.random_range(1..=20) {
            values.push(value);
        }
    }
    values.truncate(1 << 15);
    let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);

    // Force the RLE arm: on this data RunEnd is what the sampled estimate would otherwise pick.
    let btr = BtrBlocksCompressorBuilder::default()
        .exclude_schemes([RunEndScheme.id()])
        .build();
    let compressed = btr.compress(&array.clone().into_array(), &mut ctx)?;

    assert!(
        contains_rle(&compressed),
        "expected an RLE array in the tree:\n{}",
        compressed.display_tree()
    );
    assert!(
        !has_nested_delta(&compressed, false),
        "Delta was applied more than once in the tree:\n{}",
        compressed.display_tree()
    );
    assert_arrays_eq!(compressed, array.into_array(), &mut ctx);
    Ok(())
}
