// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A/B between the two structures for run-end bool decode, in one process on one dataset.
//!
//! * `prefill` is a bench-local mirror of the kernel this crate shipped before the splat: memset
//!   the output with the majority value, then `fill_range` every run whose value differs.
//! * `push_splat` is the splat as first written, appending words to a `BufferMut<u64>`.
//! * `shipped` is `runend_decode_typed_bool`, which picks between that same prefill and the
//!   splat — accumulate runs into a 64-bit word, store each output word exactly once, into a
//!   buffer sized up front and overwritten by index — and between the splat's two forms.
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
//!   50/50      3.49  3.59  3.22  1.95  2.13  0.92  1.06
//!   90/10         -  1.37  1.13  0.66  0.77  0.51  0.78
//! nullable
//!   50/50      2.41  2.54  2.40  1.58  1.20  0.96  0.73
//!   90/10         -  1.46  1.35  0.96  0.73  0.60  0.84
//! ```
//!
//! The splat is worth up to 3.6x while runs are short. The crossover moves in with skew, since a
//! skewed prefill patches few runs. `prefer_splat` reproduces this table's sign at 25 of the 26
//! points; the miss is non-nullable 1024/50-50, where it gives up 6%.
//!
//! `prefer_blend` picks between the splat's two forms, and was derived the same way — pin the
//! form and run the grid. Blend over branching, by fastest sample, splat forced:
//!
//! ```text
//! run length      1     2     8    16    32    48    64   128  1024
//! non-nullable 0.75  0.79  1.02  1.35  1.71  1.39  1.15  0.86  0.74
//! nullable     0.82  0.88  1.15  1.37  1.71  1.79  1.74  1.89  1.16
//! ```
//!
//! The peak is at half a word, where a run's chance of crossing a word boundary is closest to a
//! coin flip and the branching form's test mispredicts most.

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

/// Bits per accumulator word, mirroring the kernel.
const WORD_BITS: usize = u64::BITS as usize;

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
        avg_run_length: 16,
        skew: Skew::Even,
    },
    Args {
        avg_run_length: 32,
        skew: Skew::Even,
    },
    Args {
        avg_run_length: 48,
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

#[divan::bench(args = ARGS)]
fn push_splat(bencher: Bencher, args: Args) {
    let next = rotating(args.avg_run_length, args.skew, false);
    bencher.with_inputs(&next).bench_values(|d| {
        BoolArray::new(
            push_splat_non_nullable(d.ends.as_slice(), &d.values, TOTAL_LENGTH).freeze(),
            Nullability::NonNullable.into(),
        )
        .into_array()
    });
}

#[divan::bench(args = ARGS)]
fn push_splat_nullable(bencher: Bencher, args: Args) {
    let next = rotating(args.avg_run_length, args.skew, true);
    bencher.with_inputs(&next).bench_values(|d| {
        let Mask::Values(mask) = &d.validity else {
            unreachable!("nullable dataset")
        };
        let (decoded, validity) = push_splat_nullable_kernel(
            d.ends.as_slice(),
            &d.values,
            mask.bit_buffer(),
            TOTAL_LENGTH,
        );
        BoolArray::new(decoded.freeze(), Validity::from(validity.freeze())).into_array()
    });
}

/// The splat as it was first written, appending to a `BufferMut<u64>` with `push`/`push_n`
/// instead of overwriting a pre-sized buffer.
///
/// This is the variant the shipped kernel replaced. The interesting part is not the append
/// itself: `push_n` and `slice::fill` lower to the same inlined vector store loop, and the
/// per-word capacity check merely trades places with a slice bounds check. What the append costs
/// is inlining — `reserve`'s allocation slow path keeps `append_spanning` above LLVM's inline
/// budget, so `&mut self` escapes into a call and the accumulator lives on the stack for the
/// whole loop, at a read-modify-write per short run.
struct PushSplat {
    words: BufferMut<u64>,
    word: u64,
    bits: usize,
}

impl PushSplat {
    fn with_capacity(length: usize) -> Self {
        Self {
            words: BufferMut::with_capacity(length.div_ceil(WORD_BITS)),
            word: 0,
            bits: 0,
        }
    }

    #[inline(always)]
    fn append_run(&mut self, value: bool, n: usize) {
        let splat = u64::from(value).wrapping_neg();
        if self.bits + n < WORD_BITS {
            self.word |= splat & (((1u64 << n) - 1) << self.bits);
            self.bits += n;
            return;
        }
        self.append_spanning(splat, n);
    }

    fn append_spanning(&mut self, splat: u64, n: usize) {
        self.words.push(self.word | (splat << self.bits));
        let rest = n - (WORD_BITS - self.bits);

        let whole = rest / WORD_BITS;
        if whole > 0 {
            self.words.push_n(splat, whole);
        }

        let tail = rest % WORD_BITS;
        self.word = splat & ((1u64 << tail) - 1);
        self.bits = tail;
    }

    fn finish(mut self, length: usize) -> BitBufferMut {
        if self.bits > 0 {
            self.words.push(self.word);
        }
        let words = length.div_ceil(WORD_BITS);
        if self.words.len() < words {
            self.words.push_n(0, words - self.words.len());
        }
        self.words.truncate(words);

        let mut bytes = self.words.into_byte_buffer();
        bytes.truncate(length.div_ceil(8));
        BitBufferMut::from_buffer(bytes, 0, length)
    }
}

fn push_splat_non_nullable(run_ends: &[u32], values: &BitBuffer, length: usize) -> BitBufferMut {
    let mut decoded = PushSplat::with_capacity(length);
    let mut pos = 0usize;
    for (i, &end) in run_ends.iter().enumerate() {
        let end = (end as usize).min(length);
        decoded.append_run(values.value(i), end.saturating_sub(pos));
        pos = end;
    }
    decoded.finish(length)
}

fn push_splat_nullable_kernel(
    run_ends: &[u32],
    values: &BitBuffer,
    validity_mask: &BitBuffer,
    length: usize,
) -> (BitBufferMut, BitBufferMut) {
    let mut decoded = PushSplat::with_capacity(length);
    let mut decoded_validity = PushSplat::with_capacity(length);
    let mut pos = 0usize;
    for (i, &end) in run_ends.iter().enumerate() {
        let end = (end as usize).min(length);
        let run = end.saturating_sub(pos);
        pos = end;

        let is_valid = validity_mask.value(i);
        decoded.append_run(is_valid && values.value(i), run);
        decoded_validity.append_run(is_valid, run);
    }
    (decoded.finish(length), decoded_validity.finish(length))
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
