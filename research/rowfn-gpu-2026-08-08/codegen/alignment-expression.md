<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Expressing alignment from a plain pointer

Three ways to reach a 16-byte access starting from `*const i64`, which is what a flat
`InputElement::Varying` hands the executor today. All three pair adjacent elements in one
iteration, so they differ only in how alignment is claimed.

| Approach | Vector ops emitted |
| --- | ---: |
| adjacency only, no alignment claim | 0 |
| adjacency plus `ptr::align_offset(16)` runtime check | 0 |
| adjacency plus cast to a `#[repr(align(16))]` chunk type | 2 |

Adjacency alone is not enough, and neither is a runtime `align_offset` check. Rust can only
communicate the alignment through a type. The `align_offset` check is still required for soundness,
since casting an unaligned pointer to an over-aligned type and dereferencing it is undefined
behavior, but it does no work for code generation.

The cast is executor-internal. The public element API still hands out `*const i64`.

## Source

```rust
// Three ways to get 16-byte alignment from a plain *const i64, which is what
// InputElement::Varying gives the executor today.

// A: pair up adjacent elements, no alignment claim at all.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn a_adjacent_only(
    input: *const i64, output: *mut i64, r: i64, len: usize,
) {
    let mut i = tid() * 2;
    while i + 1 < len {
        let x = *input.add(i);
        let y = *input.add(i + 1);
        *output.add(i) = x + r;
        *output.add(i + 1) = y + r;
        i += stride() * 2;
    }
}

// B: adjacency plus a runtime align_offset check, the idiomatic Rust way to
// tell LLVM a pointer is aligned without a newtype.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn b_align_offset(
    input: *const i64, output: *mut i64, r: i64, len: usize,
) {
    if input.align_offset(16) != 0 || output.align_offset(16) != 0 {
        return;
    }
    let mut i = tid() * 2;
    while i + 1 < len {
        let x = *input.add(i);
        let y = *input.add(i + 1);
        *output.add(i) = x + r;
        *output.add(i + 1) = y + r;
        i += stride() * 2;
    }
}

#[repr(align(16))]
#[derive(Clone, Copy)]
struct A16<T>([T; 2]);

// C: executor-side reinterpret to an over-aligned chunk type. The public API
// still hands out *const i64; only the executor knows about A16.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn c_executor_cast(
    input: *const i64, output: *mut i64, r: i64, len: usize,
) {
    if input.align_offset(16) != 0 || output.align_offset(16) != 0 {
        return;
    }
    let input = input as *const A16<i64>;
    let output = output as *mut A16<i64>;
    let mut i = tid();
    while i < len / 2 {
        let c = *input.add(i);
        *output.add(i) = A16([c.0[0] + r, c.0[1] + r]);
        i += stride();
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
```
