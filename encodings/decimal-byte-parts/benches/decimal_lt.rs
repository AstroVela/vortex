// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compares `x < threshold` (less-than against a constant) over decimal data for:
//!   * Vortex `DecimalByteParts` with an i32 most-significant-part (pushdown kernel),
//!   * Vortex `DecimalByteParts` with an i64 most-significant-part (pushdown kernel),
//!   * Vortex `DecimalByteParts` two-limb i128 (signed-high / unsigned-low limbs, fused kernel),
//!   * Vortex canonical `DecimalArray` (i128 storage),
//!   * arrow-rs `Decimal128Array` via `cmp::lt`.
//!
//! Unlike `between`, arrow evaluates `lt` in a single pass, so this isolates a one-sided
//! comparison. arrow-rs has no decimal storage narrower than 128 bits, so logically-small or
//! limb-split decimals that Vortex keeps narrower must be materialised as i128 in arrow.

#![allow(clippy::unwrap_used, clippy::cast_possible_truncation)]

mod common;

use arrow_ord::cmp;
use divan::Bencher;
use divan::black_box;
use vortex_array::ArrayRef;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar::DecimalValue;
use vortex_array::scalar_fn::fns::operators::Operator;

use crate::common::LENGTHS;
use crate::common::SESSION;
use crate::common::arrow_const;
use crate::common::decimal_const;

fn main() {
    divan::main();
}

// Logical decimal range [0, 1000); threshold in the middle so ~half the rows pass.
const THRESHOLD: i64 = 500;

fn bench_lt(bencher: Bencher, arr: ArrayRef, rhs: ArrayRef) {
    bencher
        .with_inputs(|| (arr.clone(), rhs.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(arr, rhs, mut ctx)| {
            black_box(
                arr.binary(rhs, Operator::Lt)
                    .unwrap()
                    .execute::<BoolArray>(&mut ctx)
                    .unwrap(),
            )
        });
}

#[divan::bench(args = LENGTHS)]
fn vortex_byteparts_i32(bencher: Bencher, len: usize) {
    bench_lt(
        bencher,
        common::byteparts_i32(len),
        decimal_const(
            DecimalValue::I32(THRESHOLD as i32),
            common::narrow_dtype(),
            len,
        ),
    );
}

#[divan::bench(args = LENGTHS)]
fn vortex_byteparts_i64(bencher: Bencher, len: usize) {
    bench_lt(
        bencher,
        common::byteparts_i64(len),
        decimal_const(DecimalValue::I64(THRESHOLD), common::i64_dtype(), len),
    );
}

#[divan::bench(args = LENGTHS)]
fn vortex_canonical_i128(bencher: Bencher, len: usize) {
    bench_lt(
        bencher,
        common::canonical_i128(len),
        decimal_const(
            DecimalValue::I128(i128::from(THRESHOLD)),
            common::narrow_dtype(),
            len,
        ),
    );
}

// ---- arrow-rs Decimal128 (cmp::lt) ----

#[divan::bench(args = LENGTHS)]
fn arrow_decimal128(bencher: Bencher, len: usize) {
    let arr = common::arrow_decimal128(len);
    let rhs = arrow_const(i128::from(THRESHOLD));

    bencher
        .with_inputs(|| (arr.clone(), rhs.clone()))
        .bench_values(|(arr, rhs)| black_box(cmp::lt(&arr, &rhs).unwrap()));
}

// ---- Wide i128 decimals: two-limb ----
//
// The two-limb representation splits each value into a signed i64 high limb and an unsigned u64
// low limb, compared limb-wise (AVX-512 when available) instead of arrow's 128-bit comparison.
//
// The i128 baselines for this comparison are `arrow_decimal128` and `vortex_canonical_i128` above:
// an i128 comparison's cost is independent of the values and the declared precision/scale, so those
// benches measure the same kernel regardless of whether the data is logically narrow or wide.

// Threshold with a non-zero low limb so the low-limb tie-break is exercised at the high-limb edge.
const WIDE_THRESHOLD: i128 = (500i128 << 64) | 0x90ab_cdef;

#[divan::bench(args = LENGTHS)]
fn vortex_byteparts_twolimb(bencher: Bencher, len: usize) {
    bench_lt(
        bencher,
        common::byteparts_two_limb(len),
        decimal_const(
            DecimalValue::I128(WIDE_THRESHOLD),
            common::wide_dtype(),
            len,
        ),
    );
}
