// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for primitive `list_min` and `list_max` over `List`, `ListView`, and
//! `FixedSizeList` inputs.

#![expect(clippy::cast_possible_truncation)]
#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::expr::list_max;
use vortex_array::expr::list_min;
use vortex_array::expr::root;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

const BENCH_ARGS: &[(usize, usize)] = &[(8_192, 10), (8_192, 100), (8_192, 1_000)];

fn elements(num_lists: usize, list_size: usize) -> ArrayRef {
    PrimitiveArray::from_option_iter(
        (0..num_lists * list_size)
            .map(|index| (index % 10 != 0).then_some(((index * 31) % 10_007) as i32)),
    )
    .into_array()
}

fn list_validity(num_lists: usize) -> Validity {
    Validity::Array(BoolArray::from_iter((0..num_lists).map(|index| index % 10 != 0)).into_array())
}

fn make_list(num_lists: usize, list_size: usize) -> ArrayRef {
    let offsets: Buffer<i32> = (0..=num_lists)
        .map(|index| (index * list_size) as i32)
        .collect();
    ListArray::try_new(
        elements(num_lists, list_size),
        offsets.into_array(),
        list_validity(num_lists),
    )
    .unwrap()
    .into_array()
}

fn make_listview(num_lists: usize, list_size: usize) -> ArrayRef {
    let offsets: Buffer<i32> = (0..num_lists)
        .map(|index| (((index * 17) % num_lists) * (list_size - 1)) as i32)
        .collect();
    let sizes: Buffer<i32> = std::iter::repeat_n(list_size as i32, num_lists).collect();
    ListViewArray::new(
        elements(num_lists, list_size),
        offsets.into_array(),
        sizes.into_array(),
        list_validity(num_lists),
    )
    .into_array()
}

fn make_fixed_size_list(num_lists: usize, list_size: usize) -> ArrayRef {
    FixedSizeListArray::new(
        elements(num_lists, list_size),
        list_size as u32,
        list_validity(num_lists),
        num_lists,
    )
    .into_array()
}

fn run_vortex(bencher: Bencher, array: ArrayRef, minimum: bool) {
    let expression = if minimum {
        list_min(root())
    } else {
        list_max(root())
    };
    bencher
        .with_inputs(|| (&array, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| {
            array
                .clone()
                .apply(&expression)
                .unwrap()
                .execute::<Canonical>(ctx)
                .unwrap()
        });
}

#[divan::bench(args = BENCH_ARGS)]
fn vortex_list_min(bencher: Bencher, (num_lists, list_size): (usize, usize)) {
    run_vortex(bencher, make_list(num_lists, list_size), true);
}

#[divan::bench(args = BENCH_ARGS)]
fn vortex_list_max(bencher: Bencher, (num_lists, list_size): (usize, usize)) {
    run_vortex(bencher, make_list(num_lists, list_size), false);
}

#[divan::bench(args = BENCH_ARGS)]
fn vortex_listview_min(bencher: Bencher, (num_lists, list_size): (usize, usize)) {
    run_vortex(bencher, make_listview(num_lists, list_size), true);
}

#[divan::bench(args = BENCH_ARGS)]
fn vortex_listview_max(bencher: Bencher, (num_lists, list_size): (usize, usize)) {
    run_vortex(bencher, make_listview(num_lists, list_size), false);
}

#[divan::bench(args = BENCH_ARGS)]
fn vortex_fixed_size_list_min(bencher: Bencher, (num_lists, list_size): (usize, usize)) {
    run_vortex(bencher, make_fixed_size_list(num_lists, list_size), true);
}

#[divan::bench(args = BENCH_ARGS)]
fn vortex_fixed_size_list_max(bencher: Bencher, (num_lists, list_size): (usize, usize)) {
    run_vortex(bencher, make_fixed_size_list(num_lists, list_size), false);
}
