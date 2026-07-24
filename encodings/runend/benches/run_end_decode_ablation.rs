// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Ablation benchmark: isolates each change to the run-end decode kernel so every change is
//! justified by its own numbers, all measured in one binary run over identical randomized
//! inputs at a spread of average run lengths.
//!
//! The intermediate kernels live in `shared/decode_variants.rs`; see that file for what each
//! stage (`v0`..`v3`, `n0`..`n3`) contains. For a sweep of the *data distributions* each
//! change is sensitive to (run length, element width, validity density) see
//! `run_end_decode_distribution`.

#![expect(clippy::cast_possible_truncation)]

use std::fmt;

use divan::Bencher;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_buffer::Buffer;
use vortex_mask::Mask;
use vortex_runend::compress::runend_decode_typed_primitive;
use vortex_runend::trimmed_ends_iter;

#[path = "shared/decode_variants.rs"]
mod decode_variants;

use decode_variants::decode_n2;
use decode_variants::decode_v0;
use decode_variants::decode_v1;
use decode_variants::decode_v2;

fn main() {
    divan::main();
}

const SEED: u64 = 0x5eed;
const DENSITY: f64 = 0.9;

#[derive(Clone, Copy)]
struct Args {
    total_length: usize,
    avg_run_length: usize,
}

impl fmt::Display for Args {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}_{}", self.total_length, self.avg_run_length)
    }
}

const ARGS: &[Args] = &[
    Args {
        total_length: 65_536,
        avg_run_length: 2,
    },
    Args {
        total_length: 65_536,
        avg_run_length: 8,
    },
    Args {
        total_length: 65_536,
        avg_run_length: 64,
    },
    Args {
        total_length: 65_536,
        avg_run_length: 1024,
    },
];

fn data<T: NativePType + From<u8>>(
    args: Args,
) -> (Buffer<u32>, Buffer<T>, vortex_buffer::BitBuffer) {
    // Uniform 1..=(2*avg-1) has mean `avg`; matches the distribution bench's convention.
    decode_variants::make_data::<T>(
        SEED,
        args.total_length,
        2 * args.avg_run_length - 1,
        DENSITY,
    )
}

#[divan::bench(types = [u8, u32, u64], args = ARGS)]
fn v0_original<T: NativePType + From<u8>>(bencher: Bencher, args: Args) {
    let (ends, values, _) = data::<T>(args);
    bencher
        .with_inputs(|| (ends.clone(), values.clone()))
        .bench_refs(|(ends, values)| {
            let (buf, validity) = decode_v0(
                trimmed_ends_iter(ends.as_slice(), 0, args.total_length),
                values.as_slice(),
                Mask::new_true(values.len()),
                args.total_length,
            );
            PrimitiveArray::new(buf, validity)
        });
}

#[divan::bench(types = [u8, u32, u64], args = ARGS)]
fn v1_slice_loop<T: NativePType + From<u8>>(bencher: Bencher, args: Args) {
    let (ends, values, _) = data::<T>(args);
    bencher
        .with_inputs(|| (ends.clone(), values.clone()))
        .bench_refs(|(ends, values)| {
            let (buf, validity) = decode_v1(ends.as_slice(), values.as_slice(), args.total_length);
            PrimitiveArray::new(buf, validity)
        });
}

#[divan::bench(types = [u8, u32, u64], args = ARGS)]
fn v2_chunk_stores<T: NativePType + From<u8>>(bencher: Bencher, args: Args) {
    let (ends, values, _) = data::<T>(args);
    bencher
        .with_inputs(|| (ends.clone(), values.clone()))
        .bench_refs(|(ends, values)| {
            let (buf, validity) = decode_v2(ends.as_slice(), values.as_slice(), args.total_length);
            PrimitiveArray::new(buf, validity)
        });
}

#[divan::bench(types = [u8, u32, u64], args = ARGS)]
fn v3_shipped<T: NativePType + From<u8>>(bencher: Bencher, args: Args) {
    let (ends, values, _) = data::<T>(args);
    bencher
        .with_inputs(|| (ends.clone(), values.clone()))
        .bench_refs(|(ends, values)| {
            runend_decode_typed_primitive(
                ends.as_slice(),
                0,
                values.as_slice(),
                Mask::new_true(values.len()),
                Nullability::NonNullable,
                args.total_length,
            )
        });
}

#[divan::bench(types = [u8, u32, u64], args = ARGS)]
fn n0_original<T: NativePType + From<u8>>(bencher: Bencher, args: Args) {
    let (ends, values, run_validity) = data::<T>(args);
    bencher
        .with_inputs(|| (ends.clone(), values.clone(), run_validity.clone()))
        .bench_refs(|(ends, values, run_validity)| {
            let (buf, validity) = decode_v0(
                trimmed_ends_iter(ends.as_slice(), 0, args.total_length),
                values.as_slice(),
                Mask::from_buffer(run_validity.clone()),
                args.total_length,
            );
            PrimitiveArray::new(buf, validity)
        });
}

#[divan::bench(types = [u8, u32, u64], args = ARGS)]
fn n2_chunk_stores<T: NativePType + From<u8>>(bencher: Bencher, args: Args) {
    let (ends, values, run_validity) = data::<T>(args);
    bencher
        .with_inputs(|| (ends.clone(), values.clone(), run_validity.clone()))
        .bench_refs(|(ends, values, run_validity)| {
            let (buf, validity) = decode_n2(
                ends.as_slice(),
                values.as_slice(),
                run_validity,
                args.total_length,
            );
            PrimitiveArray::new(buf, validity)
        });
}

#[divan::bench(types = [u8, u32, u64], args = ARGS)]
fn n3_shipped<T: NativePType + From<u8>>(bencher: Bencher, args: Args) {
    let (ends, values, run_validity) = data::<T>(args);
    bencher
        .with_inputs(|| (ends.clone(), values.clone(), run_validity.clone()))
        .bench_refs(|(ends, values, run_validity)| {
            runend_decode_typed_primitive(
                ends.as_slice(),
                0,
                values.as_slice(),
                Mask::from_buffer(run_validity.clone()),
                Nullability::Nullable,
                args.total_length,
            )
        });
}
