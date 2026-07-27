// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::cast_possible_truncation)]

use std::fmt;
use std::sync::LazyLock;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use divan::Bencher;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::BufferMut;
use vortex_runend::compress::runend_decode_primitive;
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

/// Distribution types for bool benchmarks
#[derive(Clone, Copy)]
enum BoolDistribution {
    /// Alternating true/false (50/50)
    Alternating,
    /// Mostly true (90% true runs)
    MostlyTrue,
    /// Mostly false (90% false runs)
    MostlyFalse,
    /// All true
    AllTrue,
    /// All false
    AllFalse,
}

impl fmt::Display for BoolDistribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BoolDistribution::Alternating => write!(f, "alternating"),
            BoolDistribution::MostlyTrue => write!(f, "mostly_true"),
            BoolDistribution::MostlyFalse => write!(f, "mostly_false"),
            BoolDistribution::AllTrue => write!(f, "all_true"),
            BoolDistribution::AllFalse => write!(f, "all_false"),
        }
    }
}

#[derive(Clone, Copy)]
struct BoolBenchArgs {
    total_length: usize,
    avg_run_length: usize,
    distribution: BoolDistribution,
}

impl fmt::Display for BoolBenchArgs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}_{}_{}",
            self.total_length, self.avg_run_length, self.distribution
        )
    }
}

/// Number of independently-seeded datasets cycled across iterations, so the branch predictor
/// cannot memorise one run-length/value sequence. See `decode_bool_periodic` for what a single
/// fixed dataset is worth here.
const BOOL_DATASETS: u64 = 16;

/// Creates bool test data with *varied* run lengths and *randomised* values.
///
/// The periodic generator below fixes every run to the same length and derives values from
/// `run_index`, which makes the decoder's per-run `value != prefill` branch -- the branch the
/// whole adaptive-prefill strategy turns on -- perfectly predictable. Real columns are not.
fn create_bool_test_data_varied(
    seed: u64,
    total_length: usize,
    avg_run_length: usize,
    distribution: BoolDistribution,
    validity_density: Option<f64>,
) -> (PrimitiveArray, BoolArray) {
    let mut rng = StdRng::seed_from_u64(seed);
    let max_run = (2 * avg_run_length - 1).max(1);
    let mut ends = BufferMut::<u32>::empty();
    let mut values = Vec::new();
    let mut validity_bits = Vec::new();

    let mut pos = 0usize;
    while pos < total_length {
        let run_len = rng.random_range(1..=max_run).min(total_length - pos);
        pos += run_len;
        ends.push(pos as u32);
        values.push(match distribution {
            BoolDistribution::Alternating => rng.random_bool(0.5),
            BoolDistribution::MostlyTrue => rng.random_bool(0.9),
            BoolDistribution::MostlyFalse => rng.random_bool(0.1),
            BoolDistribution::AllTrue => true,
            BoolDistribution::AllFalse => false,
        });
        if let Some(d) = validity_density {
            validity_bits.push(rng.random_bool(d));
        }
    }

    let bools = match validity_density {
        Some(_) => BoolArray::new(
            BitBuffer::from(values),
            Validity::from(BitBuffer::from(validity_bits)),
        ),
        None => BoolArray::from(BitBuffer::from(values)),
    };
    (
        PrimitiveArray::new(ends.freeze(), Validity::NonNullable),
        bools,
    )
}

/// Input provider cycling `BOOL_DATASETS` independently-seeded varied datasets.
fn bool_rotating(
    total_length: usize,
    avg_run_length: usize,
    distribution: BoolDistribution,
    validity_density: Option<f64>,
) -> impl Fn() -> (PrimitiveArray, BoolArray) {
    let sets: Vec<_> = (0..BOOL_DATASETS)
        .map(|k| {
            create_bool_test_data_varied(
                0x5eed_u64.wrapping_add(k.wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                total_length,
                avg_run_length,
                distribution,
                validity_density,
            )
        })
        .collect();
    let next = AtomicUsize::new(0);
    move || {
        let i = next.fetch_add(1, Ordering::Relaxed);
        sets[i % sets.len()].clone()
    }
}

/// Creates bool test data with configurable distribution
fn create_bool_test_data(
    total_length: usize,
    avg_run_length: usize,
    distribution: BoolDistribution,
) -> (PrimitiveArray, BoolArray) {
    let mut ends = BufferMut::<u32>::with_capacity(total_length / avg_run_length + 1);
    let mut values = Vec::with_capacity(total_length / avg_run_length + 1);

    let mut pos = 0usize;
    let mut run_index = 0usize;

    while pos < total_length {
        let run_len = avg_run_length.min(total_length - pos);
        pos += run_len;
        ends.push(pos as u32);

        let val = match distribution {
            BoolDistribution::Alternating => run_index.is_multiple_of(2),
            BoolDistribution::MostlyTrue => !run_index.is_multiple_of(10), // 90% true
            BoolDistribution::MostlyFalse => run_index.is_multiple_of(10), // 10% true (90% false)
            BoolDistribution::AllTrue => true,
            BoolDistribution::AllFalse => false,
        };
        values.push(val);
        run_index += 1;
    }

    (
        PrimitiveArray::new(ends.freeze(), Validity::NonNullable),
        BoolArray::from(BitBuffer::from(values)),
    )
}

// Medium size: 10k elements with various run lengths and distributions
const BOOL_ARGS: &[BoolBenchArgs] = &[
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 2,
        distribution: BoolDistribution::Alternating,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 10,
        distribution: BoolDistribution::Alternating,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 100,
        distribution: BoolDistribution::Alternating,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 1000,
        distribution: BoolDistribution::Alternating,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 2,
        distribution: BoolDistribution::MostlyTrue,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 10,
        distribution: BoolDistribution::MostlyTrue,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 100,
        distribution: BoolDistribution::MostlyTrue,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 1000,
        distribution: BoolDistribution::MostlyTrue,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 2,
        distribution: BoolDistribution::MostlyFalse,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 10,
        distribution: BoolDistribution::MostlyFalse,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 100,
        distribution: BoolDistribution::MostlyFalse,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 1000,
        distribution: BoolDistribution::MostlyFalse,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 2,
        distribution: BoolDistribution::AllTrue,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 10,
        distribution: BoolDistribution::AllTrue,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 100,
        distribution: BoolDistribution::AllTrue,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 1000,
        distribution: BoolDistribution::AllTrue,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 2,
        distribution: BoolDistribution::AllFalse,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 10,
        distribution: BoolDistribution::AllFalse,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 100,
        distribution: BoolDistribution::AllFalse,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 1000,
        distribution: BoolDistribution::AllFalse,
    },
];

/// Run lengths for the data-realism control, matching the primitive sweep.
const BOOL_CONTROL_ARGS: &[BoolBenchArgs] = &[
    BoolBenchArgs {
        total_length: 65_536,
        avg_run_length: 32,
        distribution: BoolDistribution::Alternating,
    },
    BoolBenchArgs {
        total_length: 65_536,
        avg_run_length: 64,
        distribution: BoolDistribution::Alternating,
    },
    BoolBenchArgs {
        total_length: 65_536,
        avg_run_length: 128,
        distribution: BoolDistribution::Alternating,
    },
    BoolBenchArgs {
        total_length: 65_536,
        avg_run_length: 256,
        distribution: BoolDistribution::Alternating,
    },
    BoolBenchArgs {
        total_length: 65_536,
        avg_run_length: 512,
        distribution: BoolDistribution::Alternating,
    },
    BoolBenchArgs {
        total_length: 65_536,
        avg_run_length: 32,
        distribution: BoolDistribution::MostlyTrue,
    },
    BoolBenchArgs {
        total_length: 65_536,
        avg_run_length: 128,
        distribution: BoolDistribution::MostlyTrue,
    },
    BoolBenchArgs {
        total_length: 65_536,
        avg_run_length: 512,
        distribution: BoolDistribution::MostlyTrue,
    },
];

/// Data-realism control: identical kernel, periodic fixed-length data (one dataset).
///
/// Every run is the same length and values come from `run_index`, so the decoder's per-run
/// `value != prefill` branch is perfectly predictable and the run-length sequence is memorable.
#[divan::bench(args = BOOL_CONTROL_ARGS)]
fn decode_bool_periodic(bencher: Bencher, args: BoolBenchArgs) {
    let (ends, values) =
        create_bool_test_data(args.total_length, args.avg_run_length, args.distribution);
    bencher
        .with_inputs(|| (ends.clone(), values.clone(), SESSION.create_execution_ctx()))
        .bench_refs(|(ends, values, ctx)| {
            runend_decode_bools(ends.clone(), values.clone(), 0, args.total_length, ctx)
        });
}

/// Data-realism control: identical kernel, varied run lengths and randomised values, rotating
/// over `BOOL_DATASETS` seeds. The gap to `decode_bool_periodic` is what the periodic generator
/// was worth -- i.e. how much of the reported bool decode speed is branch prediction.
#[divan::bench(args = BOOL_CONTROL_ARGS)]
fn decode_bool_varied(bencher: Bencher, args: BoolBenchArgs) {
    let next_dataset = bool_rotating(
        args.total_length,
        args.avg_run_length,
        args.distribution,
        None,
    );
    bencher
        .with_inputs(|| {
            let (ends, values) = next_dataset();
            (ends, values, SESSION.create_execution_ctx())
        })
        .bench_refs(|(ends, values, ctx)| {
            runend_decode_bools(ends.clone(), values.clone(), 0, args.total_length, ctx)
        });
}

#[divan::bench(args = BOOL_ARGS)]
fn decode_bool(bencher: Bencher, args: BoolBenchArgs) {
    let BoolBenchArgs {
        total_length,
        avg_run_length,
        distribution,
    } = args;
    let (ends, values) = create_bool_test_data(total_length, avg_run_length, distribution);
    bencher
        .with_inputs(|| (ends.clone(), values.clone(), SESSION.create_execution_ctx()))
        .bench_refs(|(ends, values, ctx)| {
            runend_decode_bools(ends.clone(), values.clone(), 0, total_length, ctx)
        });
}

/// Validity distribution for nullable benchmarks
#[derive(Clone, Copy)]
enum ValidityDistribution {
    /// 90% valid
    MostlyValid,
    /// 50% valid
    HalfValid,
    /// 10% valid
    MostlyNull,
}

impl fmt::Display for ValidityDistribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidityDistribution::MostlyValid => write!(f, "mostly_valid"),
            ValidityDistribution::HalfValid => write!(f, "half_valid"),
            ValidityDistribution::MostlyNull => write!(f, "mostly_null"),
        }
    }
}

#[derive(Clone, Copy)]
struct NullableBoolBenchArgs {
    total_length: usize,
    avg_run_length: usize,
    distribution: BoolDistribution,
    validity: ValidityDistribution,
}

impl fmt::Display for NullableBoolBenchArgs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}_{}_{}_{}",
            self.total_length, self.avg_run_length, self.distribution, self.validity
        )
    }
}

/// Creates nullable bool test data with configurable distribution and validity
fn create_nullable_bool_test_data(
    total_length: usize,
    avg_run_length: usize,
    distribution: BoolDistribution,
    validity: ValidityDistribution,
) -> (PrimitiveArray, BoolArray) {
    let mut ends = BufferMut::<u32>::with_capacity(total_length / avg_run_length + 1);
    let mut values = Vec::with_capacity(total_length / avg_run_length + 1);
    let mut validity_bits = Vec::with_capacity(total_length / avg_run_length + 1);

    let mut pos = 0usize;
    let mut run_index = 0usize;

    while pos < total_length {
        let run_len = avg_run_length.min(total_length - pos);
        pos += run_len;
        ends.push(pos as u32);

        let val = match distribution {
            BoolDistribution::Alternating => run_index.is_multiple_of(2),
            BoolDistribution::MostlyTrue => !run_index.is_multiple_of(10),
            BoolDistribution::MostlyFalse => run_index.is_multiple_of(10),
            BoolDistribution::AllTrue => true,
            BoolDistribution::AllFalse => false,
        };
        values.push(val);

        let is_valid = match validity {
            ValidityDistribution::MostlyValid => !run_index.is_multiple_of(10),
            ValidityDistribution::HalfValid => run_index.is_multiple_of(2),
            ValidityDistribution::MostlyNull => run_index.is_multiple_of(10),
        };
        validity_bits.push(is_valid);

        run_index += 1;
    }

    (
        PrimitiveArray::new(ends.freeze(), Validity::NonNullable),
        BoolArray::new(
            BitBuffer::from(values),
            Validity::from(BitBuffer::from(validity_bits)),
        ),
    )
}

const NULLABLE_BOOL_ARGS: &[NullableBoolBenchArgs] = &[
    // Alternating with different validity
    NullableBoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 10,
        distribution: BoolDistribution::Alternating,
        validity: ValidityDistribution::MostlyValid,
    },
    NullableBoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 10,
        distribution: BoolDistribution::Alternating,
        validity: ValidityDistribution::HalfValid,
    },
    NullableBoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 10,
        distribution: BoolDistribution::Alternating,
        validity: ValidityDistribution::MostlyNull,
    },
    // MostlyTrue with different validity
    NullableBoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 10,
        distribution: BoolDistribution::MostlyTrue,
        validity: ValidityDistribution::MostlyValid,
    },
    NullableBoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 10,
        distribution: BoolDistribution::MostlyTrue,
        validity: ValidityDistribution::HalfValid,
    },
    NullableBoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 10,
        distribution: BoolDistribution::MostlyTrue,
        validity: ValidityDistribution::MostlyNull,
    },
    // Different run lengths with MostlyValid
    NullableBoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 2,
        distribution: BoolDistribution::Alternating,
        validity: ValidityDistribution::MostlyValid,
    },
    NullableBoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 100,
        distribution: BoolDistribution::Alternating,
        validity: ValidityDistribution::MostlyValid,
    },
    NullableBoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 1000,
        distribution: BoolDistribution::Alternating,
        validity: ValidityDistribution::MostlyValid,
    },
];

#[derive(Clone, Copy)]
struct PrimitiveBenchArgs {
    total_length: usize,
    avg_run_length: usize,
}

impl fmt::Display for PrimitiveBenchArgs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}_{}", self.total_length, self.avg_run_length)
    }
}

/// Creates primitive test data with random run lengths (uniform with the requested average),
/// random values, and random 90%-valid run validity, so branch-heavy decode strategies are
/// measured against unpredictable inputs rather than a fixed pattern.
fn create_primitive_test_data<T: NativePType + From<u8>>(
    total_length: usize,
    avg_run_length: usize,
    nullable: bool,
) -> (PrimitiveArray, PrimitiveArray) {
    let mut rng = StdRng::seed_from_u64(0x5eed);
    let num_runs = total_length / avg_run_length + 1;
    let mut ends = BufferMut::<u32>::with_capacity(num_runs);
    let mut values = BufferMut::<T>::with_capacity(num_runs);
    let mut validity_bits = Vec::with_capacity(num_runs);

    let max_run_len = (2 * avg_run_length).saturating_sub(1).max(1);
    let mut pos = 0usize;
    while pos < total_length {
        let run_len = rng.random_range(1..=max_run_len).min(total_length - pos);
        pos += run_len;
        ends.push(pos as u32);
        values.push(<T as From<u8>>::from(rng.random::<u8>()));
        validity_bits.push(rng.random_bool(0.9));
    }

    let validity = if nullable {
        Validity::from(BitBuffer::from(validity_bits))
    } else {
        Validity::NonNullable
    };
    (
        PrimitiveArray::new(ends.freeze(), Validity::NonNullable),
        PrimitiveArray::new(values.freeze(), validity),
    )
}

const PRIMITIVE_ARGS: &[PrimitiveBenchArgs] = &[
    PrimitiveBenchArgs {
        total_length: 65_536,
        avg_run_length: 2,
    },
    PrimitiveBenchArgs {
        total_length: 65_536,
        avg_run_length: 4,
    },
    PrimitiveBenchArgs {
        total_length: 65_536,
        avg_run_length: 8,
    },
    PrimitiveBenchArgs {
        total_length: 65_536,
        avg_run_length: 16,
    },
    PrimitiveBenchArgs {
        total_length: 65_536,
        avg_run_length: 64,
    },
    PrimitiveBenchArgs {
        total_length: 65_536,
        avg_run_length: 1024,
    },
];

#[divan::bench(types = [u8, u16, u32, u64], args = PRIMITIVE_ARGS)]
fn decode_primitive<T: NativePType + From<u8>>(bencher: Bencher, args: PrimitiveBenchArgs) {
    let PrimitiveBenchArgs {
        total_length,
        avg_run_length,
    } = args;
    let (ends, values) = create_primitive_test_data::<T>(total_length, avg_run_length, false);
    bencher
        .with_inputs(|| (ends.clone(), values.clone(), SESSION.create_execution_ctx()))
        .bench_refs(|(ends, values, ctx)| {
            runend_decode_primitive(ends.clone(), values.clone(), 0, total_length, ctx)
        });
}

#[divan::bench(types = [u8, u32, u64], args = PRIMITIVE_ARGS)]
fn decode_primitive_nullable<T: NativePType + From<u8>>(
    bencher: Bencher,
    args: PrimitiveBenchArgs,
) {
    let PrimitiveBenchArgs {
        total_length,
        avg_run_length,
    } = args;
    let (ends, values) = create_primitive_test_data::<T>(total_length, avg_run_length, true);
    bencher
        .with_inputs(|| (ends.clone(), values.clone(), SESSION.create_execution_ctx()))
        .bench_refs(|(ends, values, ctx)| {
            runend_decode_primitive(ends.clone(), values.clone(), 0, total_length, ctx)
        });
}

#[divan::bench(args = NULLABLE_BOOL_ARGS)]
fn decode_bool_nullable(bencher: Bencher, args: NullableBoolBenchArgs) {
    let NullableBoolBenchArgs {
        total_length,
        avg_run_length,
        distribution,
        validity,
    } = args;
    let (ends, values) =
        create_nullable_bool_test_data(total_length, avg_run_length, distribution, validity);
    bencher
        .with_inputs(|| (ends.clone(), values.clone(), SESSION.create_execution_ctx()))
        .bench_refs(|(ends, values, ctx)| {
            runend_decode_bools(ends.clone(), values.clone(), 0, total_length, ctx)
        });
}
