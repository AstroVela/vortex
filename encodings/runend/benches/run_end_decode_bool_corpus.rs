// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Run-end bool decode over fixed corpora of arrays.
//!
//! `run_end_decode`'s bool cases give every run the same length and derive its value from the
//! run index, then decode that one array over and over. Both properties flatter a decoder: the
//! run length is a loop-invariant constant, and the per-run branch on the run's value — the
//! branch an adaptive decode strategy turns on — is perfectly learnable after one pass. Real
//! columns are neither.
//!
//! So these cases decode a *set* of arrays per sample, each array independently generated from
//! a fixed seed, with every run length drawn at random around the target average. Nothing about
//! run boundaries or run values repeats across a sample, and the working set is large enough
//! that the input does not sit in L1. The seeds are fixed, so the corpus is identical on every
//! run and in CI.
//!
//! Two shapes, sized to bracket what a scan sees:
//!
//! * `128x16k` — 128 arrays of 16,384 elements (2M elements per sample), the many-chunks case
//! * `4x100k` — 4 arrays of 100,000 elements (400k elements per sample), the few-large case
//!
//! Average run lengths of 4, 16 and 64 span the regimes a bool decoder cares about: shorter than
//! a machine word, around a word, and longer than a word.

#![expect(clippy::cast_possible_truncation)]
#![expect(clippy::unwrap_used)]

use std::fmt;
use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::BufferMut;
use vortex_runend::decompress_bool::runend_decode_bools;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_runend::initialize(&session);
    session
});

/// Base seed for every corpus. Fixed so a given case is byte-identical on every run.
const SEED: u64 = 0x5EED_A55E_7B1E_0001;

/// Odd multiplier used to spread per-array seeds.
const SEED_STRIDE: u64 = 0x9E37_79B9_7F4A_7C15;

/// Fraction of runs whose value is `true`.
const EVEN: f64 = 0.5;
const SKEWED: f64 = 0.9;

/// Fraction of runs that are valid, in the nullable cases.
const VALID: f64 = 0.9;

/// How many arrays of what length make up one sample.
#[derive(Clone, Copy)]
struct Shape {
    arrays: usize,
    length: usize,
    name: &'static str,
}

const MANY_SMALL: Shape = Shape {
    arrays: 128,
    length: 16 * 1024,
    name: "128x16k",
};

const FEW_LARGE: Shape = Shape {
    arrays: 4,
    length: 100_000,
    name: "4x100k",
};

/// The mix of run values and validity a corpus is generated with.
#[derive(Clone, Copy)]
enum Mix {
    /// 50/50 true/false runs, all valid. No majority for a decoder to exploit.
    Even,
    /// 90/10 true/false runs, all valid. The shape a majority-prefill strategy is built for.
    Skewed,
    /// 50/50 true/false runs, 90% valid, so validity has to be decoded alongside the values.
    EvenNullable,
}

impl Mix {
    fn true_probability(self) -> f64 {
        match self {
            Mix::Even | Mix::EvenNullable => EVEN,
            Mix::Skewed => SKEWED,
        }
    }

    fn valid_probability(self) -> Option<f64> {
        match self {
            Mix::Even | Mix::Skewed => None,
            Mix::EvenNullable => Some(VALID),
        }
    }
}

impl fmt::Display for Mix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Mix::Even => write!(f, "even"),
            Mix::Skewed => write!(f, "skewed"),
            Mix::EvenNullable => write!(f, "even_nullable"),
        }
    }
}

#[derive(Clone, Copy)]
struct Case {
    shape: Shape,
    avg_run_length: usize,
    mix: Mix,
}

impl fmt::Display for Case {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}_run{}_{}",
            self.shape.name, self.avg_run_length, self.mix
        )
    }
}

/// Every combination of shape, average run length and value mix.
const CASES: &[Case] = &{
    let mut cases = [Case {
        shape: MANY_SMALL,
        avg_run_length: 4,
        mix: Mix::Even,
    }; 18];

    let shapes = [MANY_SMALL, FEW_LARGE];
    let run_lengths = [4usize, 16, 64];
    let mixes = [Mix::Even, Mix::Skewed, Mix::EvenNullable];

    let mut i = 0;
    let mut s = 0;
    while s < shapes.len() {
        let mut r = 0;
        while r < run_lengths.len() {
            let mut m = 0;
            while m < mixes.len() {
                cases[i] = Case {
                    shape: shapes[s],
                    avg_run_length: run_lengths[r],
                    mix: mixes[m],
                };
                i += 1;
                m += 1;
            }
            r += 1;
        }
        s += 1;
    }
    cases
};

/// One run-end encoded bool array: run ends plus the value (and validity) of each run.
type Encoded = (PrimitiveArray, BoolArray);

/// Builds one array whose runs average `avg_run_length` but are individually random.
fn encoded_array(seed: u64, length: usize, avg_run_length: usize, mix: Mix) -> Encoded {
    let mut rng = StdRng::seed_from_u64(seed);
    // Uniform over `1..=2*avg - 1`, so the mean is `avg` but no two runs need be alike and the
    // decoder cannot hoist the run length out of its loop.
    let max_run = (2 * avg_run_length - 1).max(1);
    let true_probability = mix.true_probability();

    let mut ends = BufferMut::<u32>::with_capacity(length / avg_run_length + 1);
    let mut values = Vec::with_capacity(length / avg_run_length + 1);
    let mut validity = Vec::new();

    let mut pos = 0usize;
    while pos < length {
        pos += rng.random_range(1..=max_run).min(length - pos);
        ends.push(pos as u32);
        values.push(rng.random_bool(true_probability));
        if let Some(valid) = mix.valid_probability() {
            validity.push(rng.random_bool(valid));
        }
    }

    let values = match mix.valid_probability() {
        Some(_) => BoolArray::new(
            BitBuffer::from(values),
            Validity::from(BitBuffer::from(validity)),
        ),
        None => BoolArray::from(BitBuffer::from(values)),
    };
    (
        PrimitiveArray::new(ends.freeze(), Validity::NonNullable),
        values,
    )
}

/// Builds the whole corpus for a case. Each array gets its own seed, so no two arrays in a
/// sample share a run-boundary or value sequence.
fn corpus(case: Case) -> Vec<Encoded> {
    (0..case.shape.arrays as u64)
        .map(|k| {
            encoded_array(
                SEED.wrapping_add(k.wrapping_mul(SEED_STRIDE)),
                case.shape.length,
                case.avg_run_length,
                case.mix,
            )
        })
        .collect()
}

#[divan::bench(args = CASES)]
fn decode_bool_corpus(bencher: Bencher, case: Case) {
    let corpus = corpus(case);
    let length = case.shape.length;

    bencher
        .counter(ItemsCount::new(case.shape.arrays * length))
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_local_refs(|ctx| {
            for (ends, values) in &corpus {
                divan::black_box(
                    runend_decode_bools(ends.clone(), values.clone(), 0, length, ctx).unwrap(),
                );
            }
        });
}
