<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# `for_decode.rs`

FoR decode written as a `RowFn` row body, in two loop shapes: a grid-stride loop, and the
block-tiled 16-bytes-per-iteration shape that `scalar_kernel.cuh` uses.

```rust
//! FoR decode expressed as a RowFn row body, compiled for nvptx64.
//!
//! `vortex-cuda/kernels/src/for.cu` is:
//!
//! ```cpp
//! template <typename T> struct ForOp {
//!     T reference;
//!     __device__ inline T operator()(T value) const { return value + reference; }
//! };
//! scalar_kernel(input, output, array_len, ForOp<Type>{reference});
//! ```
//!
//! which is `prepare` producing `reference` from the constant operand, and `apply` doing
//! `value + reference`. The kernels below are that same split written as Rust generics, in two
//! loop shapes: a grid-stride loop, and the block-tiled 16-bytes-per-iteration shape that
//! `scalar_kernel.cuh` uses.

use core::ops::Add;

/// The row body. Generic over the element type, exactly as `ForOp<T>` is generic over `T`.
trait RowOp<T> {
    fn apply(state: &T, value: T) -> T;
}

struct ForOp;

impl<T: Copy + Add<Output = T>> RowOp<T> for ForOp {
    #[inline(always)]
    fn apply(reference: &T, value: T) -> T {
        value + *reference
    }
}

/// Grid-stride executor, generic over element type and row op.
#[inline(always)]
unsafe fn grid_stride_exec<T: Copy, Op: RowOp<T>>(
    input: *const T,
    output: *mut T,
    len: usize,
    state: &T,
) {
    let mut i = thread_index();
    let stride = grid_stride();
    while i < len {
        *output.add(i) = Op::apply(state, *input.add(i));
        i += stride;
    }
}

/// Block-tiled executor mirroring `scalar_kernel.cuh`: 2048 elements per block, 64 threads,
/// `VALUES_PER_LOOP = 16 / size_of::<T>()` per iteration.
#[inline(always)]
unsafe fn tiled_exec<T: Copy, Op: RowOp<T>, const VALUES_PER_LOOP: usize>(
    input: *const T,
    output: *mut T,
    len: usize,
    state: &T,
) {
    const ELEMENTS_PER_BLOCK: usize = 2048;

    let block_start = block_index() * ELEMENTS_PER_BLOCK;
    let block_end = if block_start + ELEMENTS_PER_BLOCK < len {
        block_start + ELEMENTS_PER_BLOCK
    } else {
        len
    };

    let block_start_vec = block_start / VALUES_PER_LOOP;
    let block_end_vec = block_end / VALUES_PER_LOOP;

    let mut idx = block_start_vec + thread_in_block();
    while idx < block_end_vec {
        let base = idx * VALUES_PER_LOOP;
        let mut staged = [*input.add(base); VALUES_PER_LOOP];
        let mut k = 0;
        while k < VALUES_PER_LOOP {
            staged[k] = Op::apply(state, *input.add(base + k));
            k += 1;
        }
        k = 0;
        while k < VALUES_PER_LOOP {
            *output.add(base + k) = staged[k];
            k += 1;
        }
        idx += block_dim();
    }

    let mut rem = block_end_vec * VALUES_PER_LOOP + thread_in_block();
    while rem < block_end {
        *output.add(rem) = Op::apply(state, *input.add(rem));
        rem += block_dim();
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn for_grid_stride_i64(
    input: *const i64,
    output: *mut i64,
    reference: i64,
    len: usize,
) {
    grid_stride_exec::<i64, ForOp>(input, output, len, &reference);
}

#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn for_tiled_i64(
    input: *const i64,
    output: *mut i64,
    reference: i64,
    len: usize,
) {
    tiled_exec::<i64, ForOp, 2>(input, output, len, &reference);
}

#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn for_tiled_i32(
    input: *const i32,
    output: *mut i32,
    reference: i32,
    len: usize,
) {
    tiled_exec::<i32, ForOp, 4>(input, output, len, &reference);
}

#[inline(always)]
fn thread_index() -> usize {
    block_index() * block_dim() + thread_in_block()
}

#[inline(always)]
fn grid_stride() -> usize {
    grid_dim() * block_dim()
}

macro_rules! sreg {
    ($name:ident, $reg:literal) => {
        #[inline(always)]
        fn $name() -> usize {
            unsafe {
                let v: u32;
                core::arch::asm!(concat!("mov.u32 {}, %", $reg, ";"), out(reg32) v);
                v as usize
            }
        }
    };
}

sreg!(block_index, "ctaid.x");
sreg!(block_dim, "ntid.x");
sreg!(thread_in_block, "tid.x");
sreg!(grid_dim, "nctaid.x");
```
