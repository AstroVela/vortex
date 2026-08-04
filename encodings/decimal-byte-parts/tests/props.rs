// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Property tests for splitting decimals into byte parts and putting them back together.
//!
//! Every property here is the same shape: whatever the encoding does must be indistinguishable
//! from doing it to the canonical `DecimalArray`. Round tripping covers the split/assemble
//! pair directly; the compute properties cover it indirectly, since each one canonicalizes an
//! encoded array at the end.
//!
//! The generators deliberately reach the cases hand-written tests tend to miss: values that
//! straddle a 64-bit word boundary, negative values whose sign extension fills the words above
//! the most significant part, and null rows whose lower parts hold arbitrary bits.

#![expect(clippy::tests_outside_test_module)]

use hegel::TestCase;
use hegel::generators as gs;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::i256;
use vortex_array::serde::SerializeOptions;
use vortex_array::serde::SerializedArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBufferMut;
use vortex_decimal_byte_parts::DecimalByteParts;
use vortex_decimal_byte_parts::DecimalBytePartsArray;
use vortex_decimal_byte_parts::split_decimal;
use vortex_error::VortexExpect;
use vortex_mask::Mask;
use vortex_session::registry::ReadContext;

/// Largest magnitude a `Decimal(38, _)` can hold: 38 nines.
const MAX_I128: i128 = 10i128.pow(38) - 1;

/// Bound on the high `i128` half of an `i256` draw. `10^37 * 2^128` is about `3.4e75`, so any
/// value built from it stays inside the 76 digits a `Decimal(76, _)` can hold.
const MAX_I256_HIGH: i128 = 10i128.pow(37);

/// Rows per generated array. Small enough to shrink usefully, large enough that a chunked or
/// vectorized path is not trivially degenerate.
const MAX_LEN: usize = 48;

fn ctx() -> ExecutionCtx {
    let session = array_session();
    vortex_decimal_byte_parts::initialize(&session);
    session.create_execution_ctx()
}

/// Encode a canonical decimal as byte parts, splitting wide values into lower parts.
fn encode(decimal: &DecimalArray) -> DecimalBytePartsArray {
    let parts = split_decimal(decimal).vortex_expect("split");
    DecimalByteParts::try_new_with_lower_parts(
        parts.msp,
        parts.lower_parts,
        decimal.decimal_dtype(),
    )
    .vortex_expect("valid byte parts")
}

/// A validity mask of exactly `len` entries, so null rows exercise lower parts holding bits
/// that must never be read.
fn draw_validity(tc: &TestCase, len: usize) -> Validity {
    let valid: Vec<bool> = tc.draw(gs::vecs(gs::booleans()).min_size(len).max_size(len));
    Validity::from_iter(valid)
}

/// An `i128`-backed decimal. The bounds keep values inside `Decimal(38, 2)` while still
/// reaching both sides of the 64-bit word boundary the encoding splits on.
fn draw_i128_decimal(tc: &TestCase) -> DecimalArray {
    let values: Vec<i128> = tc.draw(
        gs::vecs(
            gs::integers::<i128>()
                .min_value(-MAX_I128)
                .max_value(MAX_I128),
        )
        .min_size(1)
        .max_size(MAX_LEN),
    );
    let validity = draw_validity(tc, values.len());
    DecimalArray::new(Buffer::from(values), DecimalDType::new(38, 2), validity)
}

/// An `i256`-backed decimal, built from a signed high half and an unsigned low half so the
/// draw covers sign extension above the most significant part.
fn draw_i256_decimal(tc: &TestCase) -> DecimalArray {
    let halves: Vec<(i128, u128)> = tc.draw(
        gs::vecs(gs::tuples2(
            gs::integers::<i128>()
                .min_value(-MAX_I256_HIGH)
                .max_value(MAX_I256_HIGH),
            gs::integers::<u128>(),
        ))
        .min_size(1)
        .max_size(MAX_LEN),
    );
    let values: Vec<i256> = halves
        .into_iter()
        .map(|(high, low)| i256::from_parts(low, high))
        .collect();
    let validity = draw_validity(tc, values.len());
    DecimalArray::new(Buffer::from(values), DecimalDType::new(76, 2), validity)
}

fn draw_decimal(tc: &TestCase) -> DecimalArray {
    if tc.draw(gs::booleans()) {
        draw_i128_decimal(tc)
    } else {
        draw_i256_decimal(tc)
    }
}

/// Canonicalize an encoded array back to a `DecimalArray`.
fn canonicalize(array: ArrayRef, ctx: &mut ExecutionCtx) -> DecimalArray {
    array.execute::<DecimalArray>(ctx).vortex_expect("execute")
}

// ---------------------------------------------------------------------------------------
// Split / assemble
// ---------------------------------------------------------------------------------------

/// Splitting a decimal into byte parts and reassembling it must reproduce it exactly,
/// including null rows and the sign extension above a narrow most significant part.
#[hegel::test]
fn split_then_assemble_is_identity(tc: TestCase) {
    let decimal = draw_decimal(&tc);
    let mut ctx = ctx();

    let round_tripped = canonicalize(encode(&decimal).into_array(), &mut ctx);

    assert_eq!(round_tripped.values_type(), decimal.values_type());
    assert_arrays_eq!(decimal, round_tripped, &mut ctx);
}

/// Serializing an encoded array and reading it back must preserve it. This is the path a file
/// takes, so it covers the metadata carrying the lower part count as well as the buffers.
#[hegel::test]
fn serde_round_trip_preserves_values(tc: TestCase) {
    let decimal = draw_decimal(&tc);
    let session = array_session();
    vortex_decimal_byte_parts::initialize(&session);
    let mut ctx = session.create_execution_ctx();

    let encoded = encode(&decimal).into_array();
    let dtype = encoded.dtype().clone();
    let len = encoded.len();

    let array_ctx = ArrayContext::empty();
    let serialized = encoded
        .serialize(&array_ctx, &session, &SerializeOptions::default())
        .vortex_expect("serialize");
    let mut concat = ByteBufferMut::empty();
    for buf in serialized {
        concat.extend_from_slice(buf.as_ref());
    }
    let parts = SerializedArray::try_from(concat.freeze()).vortex_expect("serialized array");
    let decoded = parts
        .decode(&dtype, len, &ReadContext::new(array_ctx.to_ids()), &session)
        .vortex_expect("decode");

    assert_arrays_eq!(decimal, canonicalize(decoded, &mut ctx), &mut ctx);
}

/// Reading one row at a time must agree with canonicalizing the whole array. These are
/// separate implementations — `combine_*` per row against the bulk assembly loops — so they
/// can disagree without any test noticing.
#[hegel::test]
fn scalar_at_agrees_with_canonical(tc: TestCase) {
    let decimal = draw_decimal(&tc);
    let mut ctx = ctx();

    let encoded = encode(&decimal).into_array();
    let canonical = decimal.into_array();

    for index in 0..encoded.len() {
        let from_parts = encoded
            .execute_scalar(index, &mut ctx)
            .vortex_expect("scalar from byte parts");
        let from_canonical = canonical
            .execute_scalar(index, &mut ctx)
            .vortex_expect("scalar from canonical");
        assert_eq!(from_parts, from_canonical, "row {index}");
    }
}

// ---------------------------------------------------------------------------------------
// Compute
// ---------------------------------------------------------------------------------------

/// Filtering the encoded array must match filtering the canonical one. The encoding pushes the
/// filter into every part, so dropping or misaligning one shows up here.
#[hegel::test]
fn filter_matches_canonical(tc: TestCase) {
    let decimal = draw_decimal(&tc);
    let len = decimal.len();
    let keep: Vec<bool> = tc.draw(gs::vecs(gs::booleans()).min_size(len).max_size(len));
    let mut ctx = ctx();

    let mask = Mask::from_iter(keep);
    let expected = canonicalize(
        decimal
            .clone()
            .into_array()
            .filter(mask.clone())
            .vortex_expect("filter canonical"),
        &mut ctx,
    );
    let actual = canonicalize(
        encode(&decimal)
            .into_array()
            .filter(mask)
            .vortex_expect("filter byte parts"),
        &mut ctx,
    );

    assert_arrays_eq!(expected, actual, &mut ctx);
}

/// Slicing must match, including slices that start partway through the array — the offsets of
/// every part have to move together.
#[hegel::test]
fn slice_matches_canonical(tc: TestCase) {
    let decimal = draw_decimal(&tc);
    let len = decimal.len();
    let a = tc.draw(gs::integers::<usize>().min_value(0).max_value(len));
    let b = tc.draw(gs::integers::<usize>().min_value(0).max_value(len));
    let (start, stop) = if a <= b { (a, b) } else { (b, a) };
    tc.assume(start < stop);
    let mut ctx = ctx();

    let expected = canonicalize(
        decimal
            .clone()
            .into_array()
            .slice(start..stop)
            .vortex_expect("slice canonical"),
        &mut ctx,
    );
    let actual = canonicalize(
        encode(&decimal)
            .into_array()
            .slice(start..stop)
            .vortex_expect("slice byte parts"),
        &mut ctx,
    );

    assert_arrays_eq!(expected, actual, &mut ctx);
}

/// Taking arbitrary indices must match, including repeats and out-of-order indices.
#[hegel::test]
fn take_matches_canonical(tc: TestCase) {
    let decimal = draw_decimal(&tc);
    let len = decimal.len();
    let indices: Vec<u64> = tc.draw(
        gs::vecs(
            gs::integers::<u64>()
                .min_value(0)
                .max_value((len - 1) as u64),
        )
        .min_size(1)
        .max_size(MAX_LEN),
    );
    let mut ctx = ctx();

    let indices = PrimitiveArray::new(Buffer::from(indices), Validity::NonNullable).into_array();
    let expected = canonicalize(
        decimal
            .clone()
            .into_array()
            .take(indices.clone())
            .vortex_expect("take canonical"),
        &mut ctx,
    );
    let actual = canonicalize(
        encode(&decimal)
            .into_array()
            .take(indices)
            .vortex_expect("take byte parts"),
        &mut ctx,
    );

    assert_arrays_eq!(expected, actual, &mut ctx);
}

/// A most significant part sitting below the top word must sign-extend into the words above
/// it. `split_decimal` never produces this shape — it always fills all three lower parts, so
/// every word is written and the sign fill is dead — which means the round-trip properties
/// above cannot see it. Only a directly constructed array reaches it.
///
/// The expectation is computed independently of the assembly loop: with two lower parts the
/// MSP occupies bits 191..128, which is exactly the low half of an `i256`'s signed `i128`
/// half, so widening it with `i128::from` performs the sign extension the encoding must.
#[hegel::test]
fn msp_below_the_top_word_sign_extends(tc: TestCase) {
    let msp: Vec<i64> = tc.draw(
        gs::vecs(gs::integers::<i64>())
            .min_size(1)
            .max_size(MAX_LEN),
    );
    let len = msp.len();
    let high: Vec<u64> = tc.draw(gs::vecs(gs::integers::<u64>()).min_size(len).max_size(len));
    let low: Vec<u64> = tc.draw(gs::vecs(gs::integers::<u64>()).min_size(len).max_size(len));
    let mut ctx = ctx();

    let array = DecimalByteParts::try_new_with_lower_parts(
        PrimitiveArray::new(Buffer::from(msp.clone()), Validity::NonNullable).into_array(),
        vec![
            PrimitiveArray::new(Buffer::from(high.clone()), Validity::NonNullable).into_array(),
            PrimitiveArray::new(Buffer::from(low.clone()), Validity::NonNullable).into_array(),
        ],
        DecimalDType::new(76, 2),
    )
    .vortex_expect("two lower parts under an i64 msp");

    let canonical = canonicalize(array.into_array(), &mut ctx);
    let actual = canonical.buffer::<i256>();

    for row in 0..len {
        let expected = i256::from_parts(
            u128::from(low[row]) | (u128::from(high[row]) << 64),
            i128::from(msp[row]),
        );
        assert_eq!(actual[row], expected, "row {row}, msp {}", msp[row]);
    }
}
