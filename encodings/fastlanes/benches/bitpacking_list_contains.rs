// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Measures the linear-search and binary-search crossover for constant-list membership.
//!
//! The strategy benchmarks isolate lookup cost across mostly-missing and mixed probes. The kernel
//! benchmark includes scalar extraction, sorting, FastLanes decoding, and result construction.
//! The isolated lookup benchmark excludes sorting, so it favors binary search. Use the forced
//! full-kernel results to select the production strategy.
//!
//! Run with `cargo bench -p vortex-fastlanes --bench bitpacking_list_contains`.

#![expect(clippy::cast_possible_truncation)]
#![expect(clippy::unwrap_used)]

use std::hint::black_box;
use std::sync::Arc;

use divan::Bencher;
use divan::counter::ItemsCount;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::list_contains::ListContainsElementKernel;
use vortex_array::validity::Validity;
use vortex_buffer::Alignment;
use vortex_buffer::BufferMut;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::BitPackedArray;
use vortex_fastlanes::BitPackedData;
use vortex_fastlanes::list_contains_test_harness;
use vortex_fastlanes::list_contains_test_harness::MembershipSearch;

const LEN: usize = 64 * 1_024;
const MEMBER_COUNTS: &[usize] = &[1, 2, 3, 4, 6, 8, 9, 12, 16, 24, 32, 64];
const STRATEGY_MEMBER_COUNTS: &[usize] = &[3, 4, 6, 8, 9, 12, 16, 24, 32, 64];

fn main() {
    divan::main();
}

fn members(count: usize) -> Vec<u32> {
    (0..count).map(|index| index as u32 * 2).collect()
}

fn mostly_missing_values() -> Vec<u32> {
    (0..LEN)
        .map(|index| ((index as u32 * 17) % 4_096) | 1)
        .collect()
}

fn mixed_values(members: &[u32]) -> Vec<u32> {
    (0..LEN)
        .map(|index| {
            if index.is_multiple_of(2) {
                members[(index / 2) % members.len()]
            } else {
                ((index as u32 * 17) % 4_096) | 1
            }
        })
        .collect()
}

fn count_linear(values: &[u32], members: &[u32]) -> usize {
    values
        .iter()
        .filter(|value| members.contains(black_box(value)))
        .count()
}

fn count_binary(values: &[u32], members: &[u32]) -> usize {
    values
        .iter()
        .filter(|value| members.binary_search(black_box(value)).is_ok())
        .count()
}

#[divan::bench(args = MEMBER_COUNTS)]
fn linear_mostly_missing(bencher: Bencher, member_count: usize) {
    let values = mostly_missing_values();
    let members = members(member_count);
    bencher
        .counter(ItemsCount::new(LEN))
        .bench_local(|| black_box(count_linear(&values, &members)));
}

#[divan::bench(args = MEMBER_COUNTS)]
fn binary_mostly_missing(bencher: Bencher, member_count: usize) {
    let values = mostly_missing_values();
    let members = members(member_count);
    bencher
        .counter(ItemsCount::new(LEN))
        .bench_local(|| black_box(count_binary(&values, &members)));
}

#[divan::bench(args = MEMBER_COUNTS)]
fn linear_mixed(bencher: Bencher, member_count: usize) {
    let members = members(member_count);
    let values = mixed_values(&members);
    bencher
        .counter(ItemsCount::new(LEN))
        .bench_local(|| black_box(count_linear(&values, &members)));
}

#[divan::bench(args = MEMBER_COUNTS)]
fn binary_mixed(bencher: Bencher, member_count: usize) {
    let members = members(member_count);
    let values = mixed_values(&members);
    bencher
        .counter(ItemsCount::new(LEN))
        .bench_local(|| black_box(count_binary(&values, &members)));
}

fn page_aligned(array: BitPackedArray) -> BitPackedArray {
    let ptype = array.dtype().as_ptype();
    let parts = BitPacked::into_parts(array);
    BitPacked::try_new(
        parts.packed.ensure_aligned(Alignment::new(4_096)).unwrap(),
        ptype,
        parts.validity,
        parts.patches,
        parts.bit_width,
        parts.len,
        parts.offset,
    )
    .unwrap()
}

fn kernel_inputs(member_count: usize) -> (BitPackedArray, ArrayRef) {
    let mut ctx = array_session().create_execution_ctx();
    let values: BufferMut<u32> = (0..LEN).map(|index| (index as u32 * 17) % 1_024).collect();
    let packed = page_aligned(
        BitPackedData::encode(
            &PrimitiveArray::new(values.freeze(), Validity::NonNullable).into_array(),
            10,
            &mut ctx,
        )
        .unwrap(),
    );
    let member_scalars = members(member_count)
        .into_iter()
        .map(|value| Scalar::primitive(value, Nullability::NonNullable))
        .collect();
    let list = ConstantArray::new(
        Scalar::list(
            Arc::new(DType::Primitive(PType::U32, Nullability::NonNullable)),
            member_scalars,
            Nullability::NonNullable,
        ),
        LEN,
    )
    .into_array();
    (packed, list)
}

#[divan::bench(args = MEMBER_COUNTS)]
fn bitpacked_kernel(bencher: Bencher, member_count: usize) {
    let (packed, list) = kernel_inputs(member_count);
    let mut ctx = array_session().create_execution_ctx();
    bencher.counter(ItemsCount::new(LEN)).bench_local(|| {
        black_box(
            <BitPacked as ListContainsElementKernel>::list_contains(
                &list,
                packed.as_view(),
                &mut ctx,
            )
            .unwrap()
            .unwrap(),
        )
    });
}

#[divan::bench(args = STRATEGY_MEMBER_COUNTS)]
fn bitpacked_linear(bencher: Bencher, member_count: usize) {
    let (packed, list) = kernel_inputs(member_count);
    let mut ctx = array_session().create_execution_ctx();
    bencher.counter(ItemsCount::new(LEN)).bench_local(|| {
        black_box(
            list_contains_test_harness::list_contains(
                &list,
                packed.as_view(),
                MembershipSearch::Linear,
                &mut ctx,
            )
            .unwrap()
            .unwrap(),
        )
    });
}

#[divan::bench(args = STRATEGY_MEMBER_COUNTS)]
fn bitpacked_binary(bencher: Bencher, member_count: usize) {
    let (packed, list) = kernel_inputs(member_count);
    let mut ctx = array_session().create_execution_ctx();
    bencher.counter(ItemsCount::new(LEN)).bench_local(|| {
        black_box(
            list_contains_test_harness::list_contains(
                &list,
                packed.as_view(),
                MembershipSearch::Binary,
                &mut ctx,
            )
            .unwrap()
            .unwrap(),
        )
    });
}
