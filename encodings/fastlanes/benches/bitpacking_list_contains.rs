// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compares compressed list membership with the canonical fallback.
//!
//! The specialized session evaluates membership while it decodes FastLanes lanes. The fallback
//! session decodes the complete array before the generic membership operation.
//! Density cases stress the 4 KiB lookup-table boundary. Sparse cases exceed that boundary.
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
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::list_contains;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::scalar::Scalar;
use vortex_array::session::ArraySessionExt;
use vortex_array::validity::Validity;
use vortex_buffer::Alignment;
use vortex_buffer::BufferMut;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::BitPackedArray;
use vortex_fastlanes::BitPackedData;
use vortex_session::VortexSession;

const DENSE_CASES: &[(usize, usize)] = &[
    (64, 1),
    (64, 4),
    (64, 8),
    (64, 32),
    (64, 64),
    (1_024, 1),
    (1_024, 4),
    (1_024, 8),
    (1_024, 32),
    (1_024, 64),
    (65_536, 1),
    (65_536, 4),
    (65_536, 8),
    (65_536, 32),
    (65_536, 64),
];
const SPARSE_CASES: &[(usize, usize)] = &[(1_024, 8), (1_024, 64), (65_536, 8), (65_536, 64)];
const DENSITY_CASES: &[(usize, usize, u32)] = &[
    (64, 5, 1_000),
    (64, 8, 512),
    (64, 64, 64),
    (65_536, 5, 1_000),
    (65_536, 8, 512),
    (65_536, 64, 64),
];

fn main() {
    divan::main();
}

fn members(count: usize, stride: u32) -> Vec<u32> {
    (0..count).map(|index| index as u32 * stride).collect()
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

fn benchmark_input(
    len: usize,
    member_count: usize,
    member_stride: u32,
    specialized: bool,
) -> (ArrayRef, VortexSession) {
    let session = array_session();
    if specialized {
        vortex_fastlanes::initialize(&session);
    } else {
        session.arrays().register(BitPacked);
    }

    let mut ctx = session.create_execution_ctx();
    let values: BufferMut<u32> = (0..len).map(|index| (index as u32 * 17) % 1_024).collect();
    let packed = page_aligned(
        BitPackedData::encode(
            &PrimitiveArray::new(values.freeze(), Validity::NonNullable).into_array(),
            10,
            &mut ctx,
        )
        .unwrap(),
    );
    let member_scalars = members(member_count, member_stride)
        .into_iter()
        .map(|value| Scalar::primitive(value, Nullability::NonNullable))
        .collect();
    let list = Scalar::list(
        Arc::new(DType::Primitive(PType::U32, Nullability::NonNullable)),
        member_scalars,
        Nullability::NonNullable,
    );
    let contains = packed
        .into_array()
        .apply(&list_contains(lit(list), root()))
        .unwrap();
    (contains, session)
}

fn bench_contains(
    bencher: Bencher,
    len: usize,
    member_count: usize,
    member_stride: u32,
    specialized: bool,
) {
    let (contains, session) = benchmark_input(len, member_count, member_stride, specialized);
    let mut ctx = session.create_execution_ctx();
    bencher
        .counter(ItemsCount::new(len))
        .bench_local(|| black_box(contains.clone().execute::<BoolArray>(&mut ctx).unwrap()));
}

#[divan::bench(args = DENSE_CASES)]
fn compressed_dense(bencher: Bencher, (len, member_count): (usize, usize)) {
    bench_contains(bencher, len, member_count, 2, true);
}

#[divan::bench(args = DENSE_CASES)]
fn canonical_dense(bencher: Bencher, (len, member_count): (usize, usize)) {
    bench_contains(bencher, len, member_count, 2, false);
}

#[divan::bench(args = SPARSE_CASES)]
fn compressed_sparse(bencher: Bencher, (len, member_count): (usize, usize)) {
    bench_contains(bencher, len, member_count, 10_000, true);
}

#[divan::bench(args = SPARSE_CASES)]
fn canonical_sparse(bencher: Bencher, (len, member_count): (usize, usize)) {
    bench_contains(bencher, len, member_count, 10_000, false);
}

#[divan::bench(args = DENSITY_CASES)]
fn compressed_density(bencher: Bencher, (len, member_count, member_stride): (usize, usize, u32)) {
    bench_contains(bencher, len, member_count, member_stride, true);
}

#[divan::bench(args = DENSITY_CASES)]
fn canonical_density(bencher: Bencher, (len, member_count, member_stride): (usize, usize, u32)) {
    bench_contains(bencher, len, member_count, member_stride, false);
}
