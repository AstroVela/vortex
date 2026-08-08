<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Can alignment live in the element rather than the trait signature?

Keeps today's `get_varying(&Varying, row_index) -> Elem` signature and puts a `#[repr(align(16))]`
pair behind the pointer, so the element carries the alignment and the executor only changes its
loop. Two loop shapes, both compiled for `nvptx64-nvidia-cuda` at `-O`.

Neither merges. Both emit adjacent, correct, unmerged scalar accesses:

```ptx
	ld.global.b64 	%rd11, [%rd10];
	ld.global.b64 	%rd12, [%rd10+8];
```

With `i = tid() * 2` the accessor's `i / 2` and `i % 2` fold correctly, but LLVM flattens the
address to `base + i * 8` and cannot prove `i` is even, because `stride()` is opaque inline
assembly. With `i = c * 2`, which is provably even at the source level, LLVM strength-reduces the
loop to a running byte offset incremented by an opaque stride, and the evenness is gone before
instruction selection.

The shape that does merge indexes a pointer whose pointee is the 16-byte chunk, so `ptr.add(c)` has
a type-derived stride of 16 and the alignment holds for any index. That is a property of access
granularity, which the element owns, not of the loop, which the executor owns.

## Source

Only the chunk-indexed variant is recorded below. The row-indexed variant differed only in the
executor loop, using `i = tid() * 2` advanced by `stride() * 2`, and emitted the same unmerged
pair of scalar accesses.

```rust
// Can the alignment live inside the element's Varying type, leaving the trait signature
// `get_varying(&Varying, index) -> Elem` exactly as it is today?

#[repr(align(16))]
#[derive(Clone, Copy)]
struct Pair<T>([T; 2]);

/// The element trait, with today's signature. Only `Varying` changed.
trait InputElement: Copy {
    type Varying: Copy;
    fn get_varying(v: &Self::Varying, i: usize) -> Self;
    fn set_varying(v: &Self::VaryingMut, i: usize, value: Self);
    type VaryingMut: Copy;
}

/// A varying view that carries 16-byte alignment in its pointer type.
#[derive(Clone, Copy)]
struct Aligned<T: 'static>(*const Pair<T>);
#[derive(Clone, Copy)]
struct AlignedMut<T: 'static>(*mut Pair<T>);

impl InputElement for i64 {
    type Varying = Aligned<i64>;
    type VaryingMut = AlignedMut<i64>;

    #[inline(always)]
    fn get_varying(v: &Aligned<i64>, i: usize) -> i64 {
        // SAFETY: caller guarantees i is in bounds; the base is 16-byte aligned by construction.
        unsafe { (*v.0.add(i / 2)).0[i % 2] }
    }

    #[inline(always)]
    fn set_varying(v: &AlignedMut<i64>, i: usize, value: i64) {
        unsafe { (*v.0.add(i / 2)).0[i % 2] = value }
    }
}

/// The executor, generic over the element, stepping two rows per iteration.
/// `apply` is an ordinary scalar row body.
#[inline(always)]
fn exec<E: InputElement, P>(
    input: E::Varying,
    output: E::VaryingMut,
    rows: usize,
    state: &P,
    apply: impl Fn(&P, E) -> E,
) {
    // Iterate chunks so the row index is provably even.
    let chunks = rows / 2;
    let mut c = tid();
    while c < chunks {
        let i = c * 2;
        let a = E::get_varying(&input, i);
        let b = E::get_varying(&input, i + 1);
        E::set_varying(&output, i, apply(state, a));
        E::set_varying(&output, i + 1, apply(state, b));
        c += stride();
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn api_shape_for_i64(
    input: *const i64, output: *mut i64, r: i64, rows: usize,
) {
    exec::<i64, i64>(
        Aligned(input.cast()),
        AlignedMut(output.cast()),
        rows,
        &r,
        |r, v| v + *r,
    );
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
