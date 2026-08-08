<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# The real `bool` failure word

The first replica used `type Failure = u64` where the production `CheckedAdd` in
`vortex-array/src/scalar_fn/fns/binary/numeric/primitive.rs` declares `type Failure = bool` through
a sign test. This probe compiles the production shape. The loop stays branchless: the evidence
lowers to `setp.lt.s64` and accumulates with `or.pred` in a predicate register, so the clean-loop
result holds for the failure type the real function uses, not only for the replica's wider word.

## Emitted loop

```ptx
$L__BB1_2:
	add.s64 	%rd12, %rd1, %rd21;
	ld.global.b64 	%rd13, [%rd12];
	add.s64 	%rd14, %rd2, %rd21;
	ld.global.b64 	%rd15, [%rd14];
	add.s64 	%rd16, %rd15, %rd13;
	xor.b64 	%rd17, %rd16, %rd13;
	xor.b64 	%rd18, %rd16, %rd15;
	and.b64 	%rd19, %rd17, %rd18;
	add.s64 	%rd20, %rd3, %rd21;
	st.global.b64 	[%rd20], %rd16;
	setp.lt.s64 	%p2, %rd19, 0;
	or.pred 	%p4, %p4, %p2;
	add.s64 	%rd22, %rd22, %rd5;
	add.s64 	%rd21, %rd21, %rd6;
	setp.lt.u64 	%p3, %rd22, %rd7;
	@%p3 bra 	$L__BB1_2;
```

## Source

```rust
#![no_std]
#![allow(internal_features)]
#![feature(abi_ptx)]
#![feature(asm_experimental_arch)]

use core::ops::BitOrAssign;

// The real CheckedAdd shape from primitive.rs: bool failure via sign test.
trait CheckedPrimitiveOp<T> {
    type Failure: Copy + Default + BitOrAssign + PartialEq;
    fn apply(lhs: T, rhs: T) -> (T, Self::Failure);
}

struct CheckedAdd;

impl CheckedPrimitiveOp<i64> for CheckedAdd {
    type Failure = bool;

    #[inline(always)]
    fn apply(lhs: i64, rhs: i64) -> (i64, bool) {
        let sum = lhs.wrapping_add(rhs);
        (sum, ((lhs ^ sum) & (rhs ^ sum)) < 0)
    }
}

#[inline(always)]
unsafe fn row_loop<T: Copy, O: Copy, F: Copy + Default + BitOrAssign>(
    lhs: *const T,
    rhs: *const T,
    out: *mut O,
    len: usize,
    tid: usize,
    stride: usize,
    apply: impl Fn((T, T)) -> (O, F),
) -> F {
    let mut failed = F::default();
    let mut i = tid;
    while i < len {
        let (value, failure) = apply((*lhs.add(i), *rhs.add(i)));
        *out.add(i) = value;
        failed |= failure;
        i += stride;
    }
    failed
}

#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn numeric_add_i64_bool_failure(
    lhs: *const i64,
    rhs: *const i64,
    out: *mut i64,
    failure_out: *mut u32,
    len: usize,
) {
    let tid = tid();
    let stride = stride();
    let failed = row_loop(lhs, rhs, out, len, tid, stride, |(l, r)| {
        <CheckedAdd as CheckedPrimitiveOp<i64>>::apply(l, r)
    });
    if failed {
        *failure_out = 1;
    }
}

#[inline(always)]
fn tid() -> usize {
    unsafe {
        let c: u32; let n: u32; let t: u32;
        core::arch::asm!("mov.u32 {}, %ctaid.x;", out(reg32) c, options(pure, nomem, nostack));
        core::arch::asm!("mov.u32 {}, %ntid.x;", out(reg32) n, options(pure, nomem, nostack));
        core::arch::asm!("mov.u32 {}, %tid.x;", out(reg32) t, options(pure, nomem, nostack));
        (c * n + t) as usize
    }
}
#[inline(always)]
fn stride() -> usize {
    unsafe {
        let g: u32; let n: u32;
        core::arch::asm!("mov.u32 {}, %nctaid.x;", out(reg32) g, options(pure, nomem, nostack));
        core::arch::asm!("mov.u32 {}, %ntid.x;", out(reg32) n, options(pure, nomem, nostack));
        (g * n) as usize
    }
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
```
