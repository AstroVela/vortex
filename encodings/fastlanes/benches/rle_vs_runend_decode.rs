// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Head-to-head decode benchmark for the two run-oriented encodings.
//!
//! Both encodings are built over the *same* input and with plain primitive children, so what
//! is timed is the decode kernel itself rather than whatever child encoding chain a real
//! compressor would have chosen underneath.
//!
//! The two kernels have different shapes:
//!
//! * `RunEnd` walks one `push_n` (a splat fill) per run, so its loop trip count is the number
//!   of runs and it reads nothing proportional to the logical length.
//! * `RLE` (FastLanes) stores one `u16` index per element and decodes with a per-element
//!   gather, so its trip count is the logical length and it reads `2 * len` extra bytes.

#![expect(clippy::cast_possible_truncation)]

use std::sync::LazyLock;

use divan::Bencher;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_fastlanes::RLEData;
use vortex_runend::RunEnd;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_fastlanes::initialize(&session);
    vortex_runend::initialize(&session);
    session
});

const LEN: usize = 1 << 20;

/// Average run length. 1 is the degenerate "no runs at all" case; 1024 fills a whole
/// FastLanes chunk with a single run.
const AVG_RUN_LENS: &[usize] = &[1, 2, 4, 8, 16, 64, 256, 1024];

/// Deterministic xorshift, so the bench needs no RNG dependency and is reproducible.
struct Rand(u64);

impl Rand {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Build `LEN` i32s whose runs are uniformly distributed around `avg_run_len`.
fn make_input(avg_run_len: usize) -> PrimitiveArray {
    let mut rng = Rand(0x2545_F491_4F6C_DD1D);
    let mut values = BufferMut::<i32>::with_capacity(LEN);
    while values.len() < LEN {
        let value = rng.next() as i32;
        let run = if avg_run_len == 1 {
            1
        } else {
            1 + (rng.next() as usize) % (2 * avg_run_len - 1)
        };
        for _ in 0..run.min(LEN - values.len()) {
            values.push(value);
        }
    }
    PrimitiveArray::new(values.freeze(), Validity::NonNullable)
}

fn encode_runend(input: &PrimitiveArray) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    RunEnd::encode(input.clone().into_array(), &mut ctx)
        .vortex_expect("run-end encode")
        .into_array()
}

fn encode_rle(input: &PrimitiveArray) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    RLEData::encode(input.as_view(), &mut ctx)
        .vortex_expect("rle encode")
        .into_array()
}

#[divan::bench(args = AVG_RUN_LENS)]
fn runend_decode(bencher: Bencher<'_, '_>, avg_run_len: usize) {
    let encoded = encode_runend(&make_input(avg_run_len));
    bencher
        .with_inputs(|| (encoded.clone(), SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| array.clone().execute::<PrimitiveArray>(ctx));
}

#[divan::bench(args = AVG_RUN_LENS)]
fn rle_decode(bencher: Bencher<'_, '_>, avg_run_len: usize) {
    let encoded = encode_rle(&make_input(avg_run_len));
    bencher
        .with_inputs(|| (encoded.clone(), SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| array.clone().execute::<PrimitiveArray>(ctx));
}
