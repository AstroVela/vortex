<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Generic chunked executor

The proposed `InputElement` addition. The element type names how many values it reads as one
aligned aggregate, the executor loads a chunk and applies the row body once per lane over
registers, and the row body stays an ordinary scalar closure.

## Emitted loop, `chunked_for_i64`

```ptx
$L__BB2_2:
	add.s64 	%rd10, %rd1, %rd16;
	ld.global.v2.b64 	{%rd11, %rd12}, [%rd10];
	add.s64 	%rd13, %rd11, %rd6;
	add.s64 	%rd14, %rd12, %rd6;
	add.s64 	%rd15, %rd2, %rd16;
	st.global.v2.b64 	[%rd15], {%rd13, %rd14};
	add.s64 	%rd17, %rd17, %rd4;
	add.s64 	%rd16, %rd16, %rd5;
	setp.lt.u64 	%p2, %rd17, %rd3;
	@%p2 bra 	$L__BB2_2;
```

## Emitted loop, `chunked_affine_i32`

The row body is `|s, v| v * *s + 7`, which LLVM fuses into `mad.lo.s32`.

```ptx
$L__BB1_2:
	add.s64 	%rd10, %rd6, %rd12;
	ld.global.v4.b32 	{%r8, %r9, %r10, %r11}, [%rd10+-12];
	mad.lo.s32 	%r12, %r8, %r1, 7;
	mad.lo.s32 	%r13, %r9, %r1, 7;
	mad.lo.s32 	%r14, %r10, %r1, 7;
	mad.lo.s32 	%r15, %r11, %r1, 7;
	add.s64 	%rd11, %rd2, %rd12;
	st.global.v4.b32 	[%rd11], {%r12, %r13, %r14, %r15};
	add.s64 	%rd13, %rd13, %rd4;
	add.s64 	%rd12, %rd12, %rd5;
	setp.lt.u64 	%p2, %rd13, %rd3;
	@%p2 bra 	$L__BB1_2;
```

## Source

```rust
//! Does a generic chunked executor vectorize while the row body stays a scalar closure?
//!
//! This models the proposed `InputElement` addition: the element type names how many values it
//! reads as one aligned aggregate, the executor loads a chunk and applies the row body N times
//! over registers, and the row body itself is unchanged.

/// The proposed addition. `Chunk` is the aligned aggregate actually loaded.
trait ChunkedElement: Copy {
    const LANES: usize;
    type Chunk: Copy;

    fn lane(chunk: &Self::Chunk, i: usize) -> Self;
    fn from_lanes(lanes: [Self; 8]) -> Self::Chunk;
}

#[repr(align(16))]
#[derive(Clone, Copy)]
struct I64x2([i64; 2]);

impl ChunkedElement for i64 {
    const LANES: usize = 2;
    type Chunk = I64x2;

    #[inline(always)]
    fn lane(chunk: &I64x2, i: usize) -> i64 {
        chunk.0[i]
    }

    #[inline(always)]
    fn from_lanes(lanes: [i64; 8]) -> I64x2 {
        I64x2([lanes[0], lanes[1]])
    }
}

#[repr(align(16))]
#[derive(Clone, Copy)]
struct I32x4([i32; 4]);

impl ChunkedElement for i32 {
    const LANES: usize = 4;
    type Chunk = I32x4;

    #[inline(always)]
    fn lane(chunk: &I32x4, i: usize) -> i32 {
        chunk.0[i]
    }

    #[inline(always)]
    fn from_lanes(lanes: [i32; 8]) -> I32x4 {
        I32x4([lanes[0], lanes[1], lanes[2], lanes[3]])
    }
}

/// The generic executor. `apply` is an ordinary scalar row body, exactly what a `RowFn` author
/// writes today. The executor is the only thing that knows about chunking.
#[inline(always)]
unsafe fn chunked_exec<T: ChunkedElement, P>(
    input: *const T::Chunk,
    output: *mut T::Chunk,
    chunks: usize,
    state: &P,
    apply: impl Fn(&P, T) -> T,
) {
    let mut i = tid();
    while i < chunks {
        let chunk = *input.add(i);
        let mut lanes = [T::lane(&chunk, 0); 8];
        let mut k = 0;
        while k < T::LANES {
            lanes[k] = apply(state, T::lane(&chunk, k));
            k += 1;
        }
        *output.add(i) = T::from_lanes(lanes);
        i += stride();
    }
}

/// FoR at i64 through the generic chunked executor, with a scalar row body.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn chunked_for_i64(
    input: *const i64,
    output: *mut i64,
    reference: i64,
    len: usize,
) {
    chunked_exec::<i64, i64>(
        input.cast(),
        output.cast(),
        len / 2,
        &reference,
        |r, v| v + *r,
    );
}

/// The same executor at i32 with a different row body, to confirm it is not one lucky shape.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn chunked_affine_i32(
    input: *const i32,
    output: *mut i32,
    scale: i32,
    len: usize,
) {
    chunked_exec::<i32, i32>(
        input.cast(),
        output.cast(),
        len / 4,
        &scale,
        |s, v| v * *s + 7,
    );
}

#[inline(always)]
fn tid() -> usize {
    unsafe {
        let ctaid: u32;
        let ntid: u32;
        let t: u32;
        core::arch::asm!("mov.u32 {}, %ctaid.x;", out(reg32) ctaid, options(pure, nomem, nostack));
        core::arch::asm!("mov.u32 {}, %ntid.x;", out(reg32) ntid, options(pure, nomem, nostack));
        core::arch::asm!("mov.u32 {}, %tid.x;", out(reg32) t, options(pure, nomem, nostack));
        (ctaid * ntid + t) as usize
    }
}

#[inline(always)]
fn stride() -> usize {
    unsafe {
        let nctaid: u32;
        let ntid: u32;
        core::arch::asm!("mov.u32 {}, %nctaid.x;", out(reg32) nctaid, options(pure, nomem, nostack));
        core::arch::asm!("mov.u32 {}, %ntid.x;", out(reg32) ntid, options(pure, nomem, nostack));
        (nctaid * ntid) as usize
    }
}
```
