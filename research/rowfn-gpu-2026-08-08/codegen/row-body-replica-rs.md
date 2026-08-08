<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# `row_body_replica.rs`

A replica of the `visit_prepared_deferred` shape, compiled for `nvptx64-nvidia-cuda`. It mirrors
`vortex-array/src/scalar_fn/fns/binary/numeric/row.rs`: a trait with an associated `Failure` type,
generic monomorphization over the primitive width, a `prepare` step producing batch state, and an
`apply` closure passed as `impl Fn` to a generic row loop.

```rust
//! A faithful replica of the RowFn `visit_prepared_deferred` shape, compiled for nvptx64.
//!
//! Mirrors `vortex-array/src/scalar_fn/fns/binary/numeric/row.rs`:
//!   - a trait with an associated `Failure` type and a `(O, F)` row apply
//!   - generic monomorphization over the primitive width
//!   - a `prepare` step producing batch state `P`
//!   - an `apply` closure taking `&P` and the element tuple
//!   - an OR-reduced deferred failure word

#![no_std]
#![allow(internal_features)]
#![feature(abi_ptx)]
#![feature(asm_experimental_arch)]

use core::ops::BitOrAssign;

/// Replica of `CheckedPrimitiveOp`.
trait CheckedPrimitiveOp<T> {
    type Failure: Copy + Default + BitOrAssign + PartialEq;
    const ERROR: &'static str;
    fn apply(lhs: T, rhs: T) -> (T, Self::Failure);
}

struct CheckedAdd;

impl CheckedPrimitiveOp<i64> for CheckedAdd {
    type Failure = u64;
    const ERROR: &'static str = "integer addition overflowed";

    #[inline(always)]
    fn apply(lhs: i64, rhs: i64) -> (i64, u64) {
        // Branchless overflow evidence, exactly as the CPU path does it: the
        // failure word is OR-reducible so no per-row branch is needed.
        let sum = lhs.wrapping_add(rhs);
        let overflow = ((lhs ^ sum) & (rhs ^ sum)) as u64;
        (sum, overflow >> 63)
    }
}

struct CheckedMul;

impl CheckedPrimitiveOp<f32> for CheckedMul {
    type Failure = u32;
    const ERROR: &'static str = "unreachable for floats";

    #[inline(always)]
    fn apply(lhs: f32, rhs: f32) -> (f32, u32) {
        (lhs * rhs, 0)
    }
}

/// Replica of the generic row executor: `prepare` once, `apply` per row,
/// OR-reduce the failure word. Generic over element type, output type,
/// prepared state and failure word, with `apply` an `impl Fn` closure —
/// the same signature shape as `RowVisitor::visit_prepared_deferred`.
#[inline(always)]
unsafe fn row_loop<T: Copy, O: Copy, P, F: Copy + Default + BitOrAssign>(
    lhs: *const T,
    rhs: *const T,
    out: *mut O,
    len: usize,
    tid: usize,
    stride: usize,
    state: &P,
    apply: impl Fn(&P, (T, T)) -> (O, F),
) -> F {
    let mut failed = F::default();
    let mut i = tid;
    while i < len {
        let (value, failure) = apply(state, (*lhs.add(i), *rhs.add(i)));
        *out.add(i) = value;
        failed |= failure;
        i += stride;
    }
    failed
}

/// A monomorphized entry point over `(i64, i64) -> i64` with a deferred failure word.
///
/// This is the direct GPU analogue of the `NumericBinary` RowFn dispatching
/// `match_each_native_ptype!` to `CheckedAdd` at `i64`.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn numeric_add_i64(
    lhs: *const i64,
    rhs: *const i64,
    out: *mut i64,
    failure_out: *mut u64,
    len: usize,
) {
    let tid = thread_index();
    let stride = grid_stride();
    // `prepare` with no constant operands: the `|_| ()` case.
    let state = ();
    let failed = row_loop(lhs, rhs, out, len, tid, stride, &state, |&(), (l, r)| {
        <CheckedAdd as CheckedPrimitiveOp<i64>>::apply(l, r)
    });
    if failed != 0 {
        *failure_out = 1;
    }
}

/// The same executor at a different width with a non-trivial prepared state —
/// the `column x constant` shape, where `prepare` hoists the constant operand.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn scale_f32_by_constant(
    lhs: *const f32,
    rhs: *const f32,
    out: *mut f32,
    _failure_out: *mut u64,
    len: usize,
) {
    let tid = thread_index();
    let stride = grid_stride();
    // A batch constant computed once, as `prepare(A::ConstElems)` does.
    let state: f32 = 2.0 * core::f32::consts::PI;
    let failed = row_loop(lhs, rhs, out, len, tid, stride, &state, |scale, (l, r)| {
        let (product, f) = <CheckedMul as CheckedPrimitiveOp<f32>>::apply(l, r);
        (product * *scale, f)
    });
    core::hint::black_box(failed);
}

#[inline(always)]
fn thread_index() -> usize {
    unsafe {
        let ctaid: u32;
        let ntid: u32;
        let tid: u32;
        core::arch::asm!("mov.u32 {}, %ctaid.x;", out(reg32) ctaid);
        core::arch::asm!("mov.u32 {}, %ntid.x;", out(reg32) ntid);
        core::arch::asm!("mov.u32 {}, %tid.x;", out(reg32) tid);
        (ctaid * ntid + tid) as usize
    }
}

#[inline(always)]
fn grid_stride() -> usize {
    unsafe {
        let nctaid: u32;
        let ntid: u32;
        core::arch::asm!("mov.u32 {}, %nctaid.x;", out(reg32) nctaid);
        core::arch::asm!("mov.u32 {}, %ntid.x;", out(reg32) ntid);
        (nctaid * ntid) as usize
    }
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
```
