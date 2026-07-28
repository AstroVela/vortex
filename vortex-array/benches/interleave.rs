// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks the Vortex [`Interleave`](vortex_array::arrays::Interleave) execute path on a focused
//! set of configurations:
//!
//! - `round_robin`, 2 children: a merge — `array_index = i % N`, `row_index = i / N`.
//! - `random`, 2 children: fully random `(array_index, row_index)` per output row.
//! - `random`, 64 children: the same random gather spread over many value arrays.
//!
//! The boolean cases are run nullable and non-nullable. Primitive cases cover four-child random
//! gathers.

#![expect(clippy::unwrap_used)]

use std::fmt::Display;
use std::fmt::Formatter;

use divan::Bencher;
use half::f16;
use rand::RngExt;
use rand::SeedableRng;
use rand::distr::Uniform;
use rand::prelude::StdRng;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::InterleaveArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;

fn main() {
    divan::main();
}

const ARRAY_SIZE: usize = 8_192;

/// The access pattern used to generate the `(array_index, row_index)` selectors.
#[derive(Clone, Copy)]
enum Pattern {
    /// A merge: `array_index = i % N`, `row_index = i / N`.
    RoundRobin,
    /// Fully random `(array_index, row_index)` per output row.
    Random,
}

impl Display for Pattern {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Pattern::RoundRobin => "round_robin",
            Pattern::Random => "random",
        })
    }
}

/// A single benchmark configuration: data type, access pattern, and branch count.
#[derive(Clone)]
struct Combo {
    dtype: DType,
    pattern: Pattern,
    branches: usize,
}

impl Display for Combo {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/n{}/{}", self.pattern, self.branches, self.dtype)
    }
}

/// The configurations the benchmark covers.
fn combos() -> Vec<Combo> {
    let mut out = Vec::new();
    for nullability in [Nullability::NonNullable, Nullability::Nullable] {
        for (pattern, branches) in [
            (Pattern::RoundRobin, 2),
            (Pattern::Random, 2),
            (Pattern::Random, 64),
        ] {
            out.push(Combo {
                dtype: DType::Bool(nullability),
                pattern,
                branches,
            });
        }
    }
    for ptype in [PType::F16, PType::U64] {
        out.push(Combo {
            dtype: ptype.into(),
            pattern: Pattern::Random,
            branches: 4,
        });
    }
    out
}

/// Builds the Vortex value arrays and the `u32` selector buffers for a [`Combo`].
///
/// Seeded only by the combo so a run is deterministic and comparable across revisions.
fn vortex_inputs(combo: &Combo) -> (Vec<ArrayRef>, Buffer<u32>, Buffer<u32>) {
    let mut rng = StdRng::seed_from_u64(0);

    let values = (0..combo.branches)
        .map(|_| match &combo.dtype {
            DType::Bool(Nullability::Nullable) => {
                let bit = Uniform::new(0u8, 2).unwrap();
                BoolArray::from_iter(
                    (0..ARRAY_SIZE).map(|_| (rng.sample(bit) == 0).then_some(rng.sample(bit) == 0)),
                )
                .into_array()
            }
            DType::Bool(Nullability::NonNullable) => {
                let bit = Uniform::new(0u8, 2).unwrap();
                BoolArray::from_iter((0..ARRAY_SIZE).map(|_| rng.sample(bit) == 0)).into_array()
            }
            DType::Primitive(PType::F16, Nullability::NonNullable) => PrimitiveArray::from_iter(
                (0..ARRAY_SIZE).map(|_| f16::from_bits(rng.random::<u16>())),
            )
            .into_array(),
            DType::Primitive(PType::U64, Nullability::NonNullable) => {
                PrimitiveArray::from_iter((0..ARRAY_SIZE).map(|_| rng.random::<u64>())).into_array()
            }
            dtype => unreachable!("unsupported interleave benchmark dtype: {dtype}"),
        })
        .collect();

    let branch = Uniform::new(0u32, u32::try_from(combo.branches).unwrap()).unwrap();
    let row = Uniform::new(0u32, u32::try_from(ARRAY_SIZE).unwrap()).unwrap();
    let array_indices: Buffer<u32> = (0..ARRAY_SIZE)
        .map(|i| match combo.pattern {
            Pattern::Random => rng.sample(branch),
            Pattern::RoundRobin => u32::try_from(i % combo.branches).unwrap(),
        })
        .collect();
    let row_indices: Buffer<u32> = (0..ARRAY_SIZE)
        .map(|i| match combo.pattern {
            Pattern::Random => rng.sample(row),
            Pattern::RoundRobin => u32::try_from((i / combo.branches) % ARRAY_SIZE).unwrap(),
        })
        .collect();
    (values, array_indices, row_indices)
}

#[divan::bench(args = combos())]
fn vortex(bencher: Bencher, combo: &Combo) {
    let (values, array_indices, row_indices) = vortex_inputs(combo);
    let session = array_session();
    bencher
        .with_inputs(|| {
            (
                InterleaveArray::try_new(
                    values.clone(),
                    array_indices.clone().into_array(),
                    row_indices.clone().into_array(),
                )
                .unwrap()
                .into_array(),
                session.create_execution_ctx(),
            )
        })
        .bench_refs(|(array, ctx)| array.clone().execute::<Canonical>(ctx));
}

fn shuffle_chunk(chunk: usize) -> ArrayRef {
    const ROWS: usize = 4;
    const DIMENSIONS: usize = 4_096;

    let ids = PrimitiveArray::from_iter((0..ROWS).map(|row| (chunk * ROWS + row) as u64));
    let vectors = PrimitiveArray::from_iter(
        (0..ROWS * DIMENSIONS)
            .map(|value| f16::from_f32((chunk * ROWS * DIMENSIONS + value) as f32)),
    );
    let vectors = FixedSizeListArray::new(
        vectors.into_array(),
        u32::try_from(DIMENSIONS).unwrap(),
        Validity::NonNullable,
        ROWS,
    );
    StructArray::new(
        ["id", "vector"].into(),
        [ids.into_array(), vectors.into_array()],
        ROWS,
        Validity::NonNullable,
    )
    .into_array()
}

#[divan::bench]
fn shuffle_struct_fsl(bencher: Bencher) {
    let values = (0..4).map(shuffle_chunk).collect::<Vec<_>>();
    let array_indices = Buffer::from_iter([3u32, 0, 2, 1]);
    let row_indices = Buffer::from_iter([1u32, 3, 0, 2]);
    let session = array_session();
    bencher
        .with_inputs(|| {
            (
                InterleaveArray::try_new(
                    values.clone(),
                    array_indices.clone().into_array(),
                    row_indices.clone().into_array(),
                )
                .unwrap()
                .into_array(),
                session.create_execution_ctx(),
            )
        })
        .bench_refs(|(array, ctx)| array.clone().execute::<RecursiveCanonical>(ctx));
}
