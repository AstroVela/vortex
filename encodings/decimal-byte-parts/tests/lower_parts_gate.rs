// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The lower-parts write gate, checked from outside the crate.
//!
//! The gate lets the crate's own unit tests through so the multi-part paths stay covered by a
//! default `cargo test`. That bypass keys off `cfg!(test)`, which is false for the library
//! when it is compiled as a dependency of this integration test — so this is the only place
//! the gate's real behaviour can be observed.

#![expect(clippy::tests_outside_test_module)]

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::dtype::DecimalDType;
use vortex_buffer::buffer;
use vortex_decimal_byte_parts::DecimalByteParts;

fn msp() -> ArrayRef {
    buffer![1i64, 2, 3].into_array()
}

fn lower_part() -> ArrayRef {
    buffer![1u64, 2, 3].into_array()
}

/// A single-child array is the stable shape and is always constructible.
#[test]
fn single_child_is_always_allowed() {
    assert!(DecimalByteParts::try_new(msp(), DecimalDType::new(19, 2)).is_ok());
    assert!(
        DecimalByteParts::try_new_with_lower_parts(msp(), vec![], DecimalDType::new(19, 2)).is_ok()
    );
}

#[cfg(not(feature = "unstable_encodings"))]
#[test]
fn lower_parts_rejected_without_the_feature() {
    let result = DecimalByteParts::try_new_with_lower_parts(
        msp(),
        vec![lower_part()],
        DecimalDType::new(38, 2),
    );
    let err = result.expect_err("expected the write gate to reject");
    assert!(
        err.to_string().contains("unstable_encodings"),
        "error should name the feature, got: {err}"
    );
}

#[cfg(feature = "unstable_encodings")]
#[test]
fn lower_parts_allowed_with_the_feature() {
    assert!(
        DecimalByteParts::try_new_with_lower_parts(
            msp(),
            vec![lower_part()],
            DecimalDType::new(38, 2),
        )
        .is_ok()
    );
}

/// The gate must hold for *every* way of getting an array with more than one limb, not just
/// the public constructor. `ArrayParts` is public and `DecimalBytePartsData` is a public unit
/// struct, so a caller can assemble slots by hand and go straight to `Array::try_from_parts`,
/// bypassing `try_new_with_lower_parts` entirely.
///
/// That back door is left open on purpose — it is the same path `deserialize` uses, and
/// closing it would stop a build without the feature reading a file written by one with it.
/// What must hold is that such an array can never be turned back into bytes.
#[cfg(not(feature = "unstable_encodings"))]
#[test]
fn hand_assembled_lower_parts_cannot_be_serialized() {
    use vortex_array::Array;
    use vortex_array::ArrayContext;
    use vortex_array::ArrayParts;
    use vortex_array::ArraySlots;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::serde::SerializeOptions;
    use vortex_array::session::ArraySessionExt;
    use vortex_decimal_byte_parts::DecimalBytePartsData;
    use vortex_error::VortexExpect;

    let session = vortex_array::array_session();
    session.arrays().register(DecimalByteParts);

    let mut slots = ArraySlots::with_capacity(2);
    slots.push(Some(msp()));
    slots.push(Some(lower_part()));

    // Assembling the array by hand succeeds: this is the shape a file read produces.
    let array = Array::try_from_parts(
        ArrayParts::new(
            DecimalByteParts,
            DType::Decimal(DecimalDType::new(38, 2), Nullability::NonNullable),
            3,
            DecimalBytePartsData,
        )
        .with_slots(slots),
    )
    .vortex_expect("hand assembly is not gated")
    .into_array();
    assert_eq!(array.nchildren(), 2, "expected two limbs");

    // Writing it out does not.
    let err = array
        .serialize(
            &ArrayContext::empty(),
            &session,
            &SerializeOptions::default(),
        )
        .expect_err("expected the write gate to refuse");
    assert!(
        err.to_string().contains("unstable_encodings"),
        "error should name the feature, got: {err}"
    );
}
