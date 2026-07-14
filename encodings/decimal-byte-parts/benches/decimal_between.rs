// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compares `between` (`lower <= x <= upper`) over decimal data for:
//!   * Vortex `DecimalByteParts` with an i32 most-significant-part (pushdown kernel),
//!   * Vortex `DecimalByteParts` with an i64 most-significant-part (pushdown kernel),
//!   * Vortex canonical `DecimalArray` (i128 storage),
//!   * arrow-rs `Decimal128Array` via `gt_eq` + `lt_eq` + `and`.
//!
//! arrow-rs has no decimal storage narrower than 128 bits, so logically-small decimals that
//! Vortex keeps in an i32/i64 MSP must be materialised as i128 in Arrow. This benchmark
//! measures the resulting throughput difference.

#![allow(clippy::unwrap_used, clippy::cast_possible_truncation)]

mod common;

use arrow_arith::boolean::and;
use arrow_ord::cmp;
use divan::Bencher;
use divan::black_box;
use vortex_array::ArrayRef;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar::DecimalValue;
use vortex_array::scalar_fn::fns::between::BetweenOptions;
use vortex_array::scalar_fn::fns::between::StrictComparison;

use crate::common::LENGTHS;
use crate::common::SESSION;
use crate::common::arrow_const;
use crate::common::decimal_const;

fn main() {
    divan::main();
}

// Logical decimal range [0, 1000); bounds chosen so ~half the rows pass.
const LOWER: i64 = 250;
const UPPER: i64 = 750;

const OPTIONS: BetweenOptions = BetweenOptions {
    lower_strict: StrictComparison::NonStrict,
    upper_strict: StrictComparison::NonStrict,
};

fn bench_between(bencher: Bencher, arr: ArrayRef, lower: ArrayRef, upper: ArrayRef) {
    bencher
        .with_inputs(|| {
            (
                arr.clone(),
                lower.clone(),
                upper.clone(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(arr, lower, upper, mut ctx)| {
            black_box(
                arr.between(lower, upper, OPTIONS)
                    .unwrap()
                    .execute::<BoolArray>(&mut ctx)
                    .unwrap(),
            )
        });
}

#[divan::bench(args = LENGTHS)]
fn vortex_byteparts_i32(bencher: Bencher, len: usize) {
    let dt = common::narrow_dtype();
    bench_between(
        bencher,
        common::byteparts_i32(len),
        decimal_const(DecimalValue::I32(LOWER as i32), dt, len),
        decimal_const(DecimalValue::I32(UPPER as i32), dt, len),
    );
}

#[divan::bench(args = LENGTHS)]
fn vortex_byteparts_i64(bencher: Bencher, len: usize) {
    let dt = common::i64_dtype();
    bench_between(
        bencher,
        common::byteparts_i64(len),
        decimal_const(DecimalValue::I64(LOWER), dt, len),
        decimal_const(DecimalValue::I64(UPPER), dt, len),
    );
}

#[divan::bench(args = LENGTHS)]
fn vortex_canonical_i128(bencher: Bencher, len: usize) {
    let dt = common::narrow_dtype();
    bench_between(
        bencher,
        common::canonical_i128(len),
        decimal_const(DecimalValue::I128(i128::from(LOWER)), dt, len),
        decimal_const(DecimalValue::I128(i128::from(UPPER)), dt, len),
    );
}

// ---- arrow-rs Decimal128 (gt_eq + lt_eq + and) ----

#[divan::bench(args = LENGTHS)]
fn arrow_decimal128(bencher: Bencher, len: usize) {
    let arr = common::arrow_decimal128(len);
    let lower = arrow_const(i128::from(LOWER));
    let upper = arrow_const(i128::from(UPPER));

    bencher
        .with_inputs(|| (arr.clone(), lower.clone(), upper.clone()))
        .bench_values(|(arr, lower, upper)| {
            let ge = cmp::gt_eq(&arr, &lower).unwrap();
            let le = cmp::lt_eq(&arr, &upper).unwrap();
            black_box(and(&ge, &le).unwrap())
        });
}

// ---- Wide i128 decimals: two-limb ----
//
// The two-limb representation splits each value into a signed i64 high limb and an unsigned u64
// low limb, compared limb-wise (AVX-512 when available) instead of arrow's 128-bit comparison.
//
// The i128 baselines for this comparison are `arrow_decimal128` and `vortex_canonical_i128` above:
// an i128 comparison's cost is independent of the values and the declared precision/scale, so those
// benches measure the same kernel regardless of whether the data is logically narrow or wide.

// Bounds with non-zero low limbs so the low-limb tie-break is exercised at the high-limb edges.
const WIDE_LOWER: i128 = (250i128 << 64) | 0x1234_5678;
const WIDE_UPPER: i128 = (750i128 << 64) | 0x90ab_cdef;

#[divan::bench(args = LENGTHS)]
fn vortex_byteparts_twolimb(bencher: Bencher, len: usize) {
    let dt = common::wide_dtype();
    bench_between(
        bencher,
        common::byteparts_two_limb(len),
        decimal_const(DecimalValue::I128(WIDE_LOWER), dt, len),
        decimal_const(DecimalValue::I128(WIDE_UPPER), dt, len),
    );
}
