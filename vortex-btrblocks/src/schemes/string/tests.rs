// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::VarBinViewBuilder;
use vortex_array::display::DisplayOptions;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::BtrBlocksCompressor;

static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

#[test]
fn test_strings() -> VortexResult<()> {
    let mut strings = Vec::new();
    for _ in 0..1024 {
        strings.push(Some("hello-world-1234"));
    }
    for _ in 0..1024 {
        strings.push(Some("hello-world-56789"));
    }
    let strings = VarBinViewArray::from_iter(strings, DType::Utf8(Nullability::NonNullable));

    let array_ref = strings.into_array();
    let btr = BtrBlocksCompressor::default();
    let compressed = btr.compress(&array_ref, &mut SESSION.create_execution_ctx())?;
    assert_eq!(compressed.len(), 2048);

    let display = compressed
        .display_as(DisplayOptions::MetadataOnly)
        .to_string()
        .to_lowercase();
    assert_eq!(display, "vortex.dict(utf8, len=2048)");

    Ok(())
}

#[test]
fn test_sparse_nulls() -> VortexResult<()> {
    let mut strings = VarBinViewBuilder::with_capacity(DType::Utf8(Nullability::Nullable), 100);
    strings.append_nulls(99);

    strings.append_value("one little string");

    let strings = strings.finish_into_varbinview();

    let array_ref = strings.into_array();
    let btr = BtrBlocksCompressor::default();
    let compressed = btr.compress(&array_ref, &mut SESSION.create_execution_ctx())?;
    assert_eq!(compressed.len(), 100);

    let display = compressed
        .display_as(DisplayOptions::MetadataOnly)
        .to_string()
        .to_lowercase();
    assert_eq!(display, "vortex.sparse(utf8?, len=100)");

    Ok(())
}

/// Strings no codec can usefully shrink should land as offset-addressed `varbin`, not as a codec
/// that barely beats canonical and not as the 16-byte-per-value `varbinview` it started as.
#[test]
fn incompressible_strings_fall_back_to_varbin() -> VortexResult<()> {
    // Deterministic pseudo-random ASCII: no repeated substrings for FSST to build symbols from.
    let values: Vec<String> = (0..8192)
        .map(|i| {
            let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
            (0..180)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    char::from(b'!' + (x % 90) as u8)
                })
                .collect()
        })
        .collect();
    let array = VarBinViewArray::from_iter_str(values.iter().map(String::as_str)).into_array();

    let compressed =
        BtrBlocksCompressor::default().compress(&array, &mut SESSION.create_execution_ctx())?;

    let display = compressed
        .display_as(DisplayOptions::MetadataOnly)
        .to_string()
        .to_lowercase();
    assert!(
        display.starts_with("vortex.varbin("),
        "expected varbin, got {display}"
    );
    assert!(
        compressed.nbytes() < array.nbytes(),
        "varbin ({}) should beat canonical views ({})",
        compressed.nbytes(),
        array.nbytes()
    );

    Ok(())
}

/// A column FSST genuinely compresses must still get FSST: the floor rejects marginal wins, not
/// real ones.
#[test]
fn compressible_strings_still_use_fsst() -> VortexResult<()> {
    let values: Vec<String> = (0..8192)
        .map(|i| format!("the quick brown fox jumps over the lazy dog number {i}"))
        .collect();
    let array = VarBinViewArray::from_iter_str(values.iter().map(String::as_str)).into_array();

    let compressed =
        BtrBlocksCompressor::default().compress(&array, &mut SESSION.create_execution_ctx())?;

    let display = compressed
        .display_as(DisplayOptions::MetadataOnly)
        .to_string()
        .to_lowercase();
    assert!(
        display.contains("fsst") || display.contains("dict"),
        "expected a real codec, got {display}"
    );
    Ok(())
}
