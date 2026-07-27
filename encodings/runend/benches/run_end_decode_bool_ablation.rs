// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A/B between the two structures for run-end bool decode, in one process on one dataset.
//!
//! * `prefill` is a bench-local mirror of the kernel this crate shipped before the splat: memset
//!   the output with the majority value, then `fill_range` every run whose value differs.
//! * `shipped` is `runend_decode_typed_bool`, which picks between that same prefill and the
//!   splat — accumulate runs into a 64-bit word, store each output word exactly once.
//!
//! Run lengths and values are randomised and datasets are rotated across iterations, so the
//! per-run value branch the prefill kernel depends on cannot be memorised by the predictor.
//!
//! The dispatch in `prefer_splat` comes from running this grid against a build pinned to each
//! kernel (return a constant from `prefer_splat` to re-derive it). Splat over prefill, by
//! fastest sample:
//!
//! ```text
//! run length      1     2     8    32    64   128  1024
//! non-nullable
//!   50/50      3.70  3.46  2.86  1.25  1.26  0.87  0.61
//!   90/10         -  1.28  0.93  0.44  0.48  0.49  0.61
//! nullable
//!   50/50      1.97  2.09  1.92  1.21  0.74  0.50  0.63
//!   90/10         -  1.20  1.14  0.64  0.46  0.47  0.63
//! ```
//!
//! The splat is worth up to 3.7x while runs are short. The crossover moves in with skew, since
//! a skewed prefill patches few runs, and in again for nullable input, where the splat builds
//! two buffers but the prefill still skips both at once. `prefer_splat` reproduces this table's
//! sign everywhere except non-nullable 8/90-10, where it gives up 7%.

#![expect(clippy::cast_possible_truncation)]

use std::fmt;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use divan::Bencher;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::dtype::Nullability;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_mask::Mask;
use vortex_runend::decompress_bool::runend_decode_typed_bool;
use vortex_runend::trimmed_ends_iter;

fn main() {
    divan::main();
}

const TOTAL_LENGTH: usize = 65_536;

/// Number of independently-seeded datasets cycled across iterations.
const DATASETS: usize = 8;

/// Fraction of runs whose value is `true`.
#[derive(Clone, Copy)]
enum Skew {
    /// 50/50, so the prefill kernel's `value != prefill` branch is unpredictable.
    Even,
    /// 90% true, the shape the adaptive prefill is designed for.
    Skewed,
}

impl fmt::Display for Skew {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Skew::Even => write!(f, "even"),
            Skew::Skewed => write!(f, "skewed"),
        }
    }
}

impl Skew {
    fn true_probability(self) -> f64 {
        match self {
            Skew::Even => 0.5,
            Skew::Skewed => 0.9,
        }
    }
}

#[derive(Clone, Copy)]
struct Args {
    avg_run_length: usize,
    skew: Skew,
}

impl fmt::Display for Args {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "run{}_{}", self.avg_run_length, self.skew)
    }
}

const ARGS: &[Args] = &[
    Args {
        avg_run_length: 1,
        skew: Skew::Even,
    },
    Args {
        avg_run_length: 2,
        skew: Skew::Even,
    },
    Args {
        avg_run_length: 8,
        skew: Skew::Even,
    },
    Args {
        avg_run_length: 32,
        skew: Skew::Even,
    },
    Args {
        avg_run_length: 64,
        skew: Skew::Even,
    },
    Args {
        avg_run_length: 128,
        skew: Skew::Even,
    },
    Args {
        avg_run_length: 1024,
        skew: Skew::Even,
    },
    Args {
        avg_run_length: 2,
        skew: Skew::Skewed,
    },
    Args {
        avg_run_length: 8,
        skew: Skew::Skewed,
    },
    Args {
        avg_run_length: 32,
        skew: Skew::Skewed,
    },
    Args {
        avg_run_length: 64,
        skew: Skew::Skewed,
    },
    Args {
        avg_run_length: 128,
        skew: Skew::Skewed,
    },
    Args {
        avg_run_length: 1024,
        skew: Skew::Skewed,
    },
];

/// One decode input: run ends plus the run values, and optionally per-run validity.
struct Dataset {
    ends: Buffer<u32>,
    values: BitBuffer,
    validity: Mask,
}

fn dataset(seed: u64, avg_run_length: usize, skew: Skew, nullable: bool) -> Dataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let max_run = (2 * avg_run_length - 1).max(1);
    let p = skew.true_probability();

    let mut ends = BufferMut::<u32>::empty();
    let mut values = Vec::new();
    let mut validity = Vec::new();

    let mut pos = 0usize;
    while pos < TOTAL_LENGTH {
        let run = rng.random_range(1..=max_run).min(TOTAL_LENGTH - pos);
        pos += run;
        ends.push(pos as u32);
        values.push(rng.random_bool(p));
        if nullable {
            validity.push(rng.random_bool(0.9));
        }
    }

    Dataset {
        ends: ends.freeze(),
        values: BitBuffer::from(values),
        validity: if nullable {
            Mask::from(BitBuffer::from(validity))
        } else {
            Mask::AllTrue(pos)
        },
    }
}

/// Cycles `DATASETS` independently-seeded datasets across iterations.
fn rotating(avg_run_length: usize, skew: Skew, nullable: bool) -> impl Fn() -> &'static Dataset {
    let sets: &'static [Dataset] = Box::leak(
        (0..DATASETS as u64)
            .map(|k| {
                dataset(
                    0x5eed_u64.wrapping_add(k.wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                    avg_run_length,
                    skew,
                    nullable,
                )
            })
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    );
    let next = AtomicUsize::new(0);
    move || &sets[next.fetch_add(1, Ordering::Relaxed) % DATASETS]
}

#[divan::bench(args = ARGS)]
fn shipped(bencher: Bencher, args: Args) {
    let next = rotating(args.avg_run_length, args.skew, false);
    bencher.with_inputs(&next).bench_values(|d| {
        runend_decode_typed_bool(
            d.ends.as_slice(),
            0,
            &d.values,
            d.validity.clone(),
            Nullability::NonNullable,
            TOTAL_LENGTH,
        )
    });
}

#[divan::bench(args = ARGS)]
fn prefill(bencher: Bencher, args: Args) {
    let next = rotating(args.avg_run_length, args.skew, false);
    bencher.with_inputs(&next).bench_values(|d| {
        prefill_non_nullable(
            trimmed_ends_iter(d.ends.as_slice(), 0, TOTAL_LENGTH),
            &d.values,
            Nullability::NonNullable,
            TOTAL_LENGTH,
        )
        .into_array()
    });
}

#[divan::bench(args = ARGS)]
fn shipped_nullable(bencher: Bencher, args: Args) {
    let next = rotating(args.avg_run_length, args.skew, true);
    bencher.with_inputs(&next).bench_values(|d| {
        runend_decode_typed_bool(
            d.ends.as_slice(),
            0,
            &d.values,
            d.validity.clone(),
            Nullability::Nullable,
            TOTAL_LENGTH,
        )
    });
}

#[divan::bench(args = ARGS)]
fn prefill_nullable(bencher: Bencher, args: Args) {
    let next = rotating(args.avg_run_length, args.skew, true);
    bencher.with_inputs(&next).bench_values(|d| {
        let Mask::Values(mask) = &d.validity else {
            unreachable!("nullable dataset")
        };
        prefill_nullable_kernel(
            trimmed_ends_iter(d.ends.as_slice(), 0, TOTAL_LENGTH),
            &d.values,
            mask.bit_buffer(),
            TOTAL_LENGTH,
        )
        .into_array()
    });
}

/// Threshold below which the mirrored kernel appends sequentially instead of prefilling.
const PREFILL_RUN_THRESHOLD: usize = 32;

/// Mirror of the previously shipped non-nullable kernel.
fn prefill_non_nullable(
    run_ends: impl Iterator<Item = usize>,
    values: &BitBuffer,
    nullability: Nullability,
    length: usize,
) -> BoolArray {
    let num_runs = values.len();

    if num_runs < PREFILL_RUN_THRESHOLD {
        let mut decoded = BitBufferMut::with_capacity(length);
        for (end, value) in run_ends.zip(values.iter()) {
            decoded.append_n(value, end - decoded.len());
        }
        return BoolArray::new(decoded.freeze(), nullability.into());
    }

    let prefill = values.true_count() > num_runs - values.true_count();
    let mut decoded = BitBufferMut::full(prefill, length);
    let mut current_pos = 0usize;

    for (end, value) in run_ends.zip(values.iter()) {
        if end > current_pos && value != prefill {
            // SAFETY: current_pos < end <= length == decoded.len()
            unsafe { decoded.fill_range_unchecked(current_pos, end, value) };
        }
        current_pos = end;
    }
    BoolArray::new(decoded.freeze(), nullability.into())
}

/// Mirror of the previously shipped nullable kernel.
fn prefill_nullable_kernel(
    run_ends: impl Iterator<Item = usize>,
    values: &BitBuffer,
    validity_mask: &BitBuffer,
    length: usize,
) -> BoolArray {
    let num_runs = values.len();

    if num_runs < PREFILL_RUN_THRESHOLD {
        let mut decoded = BitBufferMut::with_capacity(length);
        let mut decoded_validity = BitBufferMut::with_capacity(length);
        for (end, (value, is_valid)) in run_ends.zip(values.iter().zip(validity_mask.iter())) {
            let run_len = end - decoded.len();
            decoded_validity.append_n(is_valid, run_len);
            decoded.append_n(is_valid && value, run_len);
        }
        return BoolArray::new(decoded.freeze(), Validity::from(decoded_validity.freeze()));
    }

    let prefill_decoded = values.true_count() > num_runs - values.true_count();
    let prefill_valid = validity_mask.true_count() > num_runs - validity_mask.true_count();

    let mut decoded = BitBufferMut::full(prefill_decoded, length);
    let mut decoded_validity = BitBufferMut::full(prefill_valid, length);
    let mut current_pos = 0usize;

    for (end, (value, is_valid)) in run_ends.zip(values.iter().zip(validity_mask.iter())) {
        if end > current_pos {
            // SAFETY: current_pos < end <= length == decoded.len() == decoded_validity.len()
            if is_valid != prefill_valid {
                unsafe { decoded_validity.fill_range_unchecked(current_pos, end, is_valid) };
            }
            let want_decoded = is_valid && value;
            if want_decoded != prefill_decoded {
                unsafe { decoded.fill_range_unchecked(current_pos, end, want_decoded) };
            }
            current_pos = end;
        }
    }
    BoolArray::new(decoded.freeze(), Validity::from(decoded_validity.freeze()))
}
