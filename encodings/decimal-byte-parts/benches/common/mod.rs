// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Data and session scaffolding shared by the decimal comparison benchmarks, so the
//! cross-benchmark comparison always runs over identical inputs.

// Each bench target compiles this module independently and uses a subset of it.
#![allow(dead_code)]

use std::sync::LazyLock;

use arrow_array::Decimal128Array;
use arrow_array::Scalar as ArrowScalar;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::DecimalValue;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_decimal_byte_parts::DecimalByteParts;
use vortex_session::VortexSession;

pub static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_decimal_byte_parts::initialize(&session);
    session
});

pub const LENGTHS: &[usize] = &[1 << 16, 1 << 17];

/// Logical decimal values in [0, 1000): precision 9 scale 2 fits an i32 MSP, precision 18 an i64.
pub fn values(len: usize) -> Vec<i64> {
    let mut rng = StdRng::seed_from_u64(0x5eed);
    (0..len).map(|_| rng.random_range(0..1000i64)).collect()
}

/// Values genuinely occupying the i128 range (the high 64-bit limb varies), so neither Vortex nor
/// arrow can keep them in a narrow integer.
#[allow(clippy::cast_possible_truncation)]
pub fn wide_values(len: usize) -> Vec<i128> {
    let mut rng = StdRng::seed_from_u64(0x5eed);
    (0..len)
        .map(|_| {
            let high = i128::from(rng.random_range(0..1000i64));
            let low = i128::from(rng.random_range(0..u64::MAX));
            (high << 64) | low
        })
        .collect()
}

pub fn narrow_dtype() -> DecimalDType {
    DecimalDType::new(9, 2)
}

pub fn i64_dtype() -> DecimalDType {
    DecimalDType::new(18, 2)
}

pub fn wide_dtype() -> DecimalDType {
    DecimalDType::new(38, 2)
}

/// `values(len)` in a `DecimalByteParts` with an i32 most-significant part.
#[allow(clippy::cast_possible_truncation)]
pub fn byteparts_i32(len: usize) -> ArrayRef {
    let msp = PrimitiveArray::from_iter(values(len).into_iter().map(|v| v as i32)).into_array();
    DecimalByteParts::try_new(msp, narrow_dtype())
        .unwrap()
        .into_array()
}

/// `values(len)` in a `DecimalByteParts` with an i64 most-significant part.
pub fn byteparts_i64(len: usize) -> ArrayRef {
    let msp = PrimitiveArray::from_iter(values(len)).into_array();
    DecimalByteParts::try_new(msp, i64_dtype())
        .unwrap()
        .into_array()
}

/// `values(len)` in the canonical i128 `DecimalArray`.
pub fn canonical_i128(len: usize) -> ArrayRef {
    DecimalArray::new(
        values(len).into_iter().map(i128::from).collect(),
        narrow_dtype(),
        Validity::NonNullable,
    )
    .into_array()
}

/// `wide_values(len)` in a two-limb `DecimalByteParts` (signed i64 high limb, unsigned u64 low
/// limb).
#[allow(clippy::cast_possible_truncation)]
pub fn byteparts_two_limb(len: usize) -> ArrayRef {
    let values = wide_values(len);
    let highs = PrimitiveArray::from_iter(values.iter().map(|v| (v >> 64) as i64)).into_array();
    let lows = PrimitiveArray::from_iter(values.iter().map(|v| *v as u64)).into_array();
    DecimalByteParts::try_new_with_lower(highs, lows, wide_dtype())
        .unwrap()
        .into_array()
}

/// `values(len)` in an arrow `Decimal128Array`.
pub fn arrow_decimal128(len: usize) -> Decimal128Array {
    Decimal128Array::from_iter_values(values(len).into_iter().map(i128::from))
        .with_precision_and_scale(9, 2)
        .unwrap()
}

/// A non-nullable decimal constant array of length `len`.
pub fn decimal_const(value: DecimalValue, dt: DecimalDType, len: usize) -> ArrayRef {
    ConstantArray::new(Scalar::decimal(value, dt, Nullability::NonNullable), len).into_array()
}

/// An arrow decimal scalar matching `arrow_decimal128`'s precision and scale.
pub fn arrow_const(value: i128) -> ArrowScalar<Decimal128Array> {
    ArrowScalar::new(
        Decimal128Array::from_iter_values([value])
            .with_precision_and_scale(9, 2)
            .unwrap(),
    )
}
