<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Vectorization probes

Four kernels doing the same FoR row body over adjacent `i64` elements, differing only in what the
compiler is told about the access. Compiled for `nvptx64-nvidia-cuda` at `-O`.

## Result

| Probe | Emitted |
| --- | --- |
| plain `*const i64` | `ld.global.b64` x2 |
| `assert_unchecked(ptr % 16 == 0)` | `ld.global.b64` x2 |
| `#[repr(align(16))] Pair([i64; 2])` | `ld.global.v2.b64` |
| `#[repr(align(16))] Quad([i32; 4])` | `ld.global.v4.b32` |
| scalar field reads through `#[repr(align(16))] Pair` | `ld.global.v2.b64` |

The last row is the decisive one. The LoadStoreVectorizer *does* merge adjacent scalar accesses, so
long as alignment is provable from the type. `assert_unchecked` emits an `llvm.assume`, which does
not reach that analysis; `#[repr(align(16))]` does.

Flags do not change the plain result. `opt-level` 2 and 3 crossed with default, `sm_80`, and `sm_90`
all emit zero vector loads.

## Source

```rust
//! Probes for why the NVPTX backend does not merge adjacent loads.
//!
//! Each kernel does the same FoR row body over two adjacent i64 elements. They differ only in
//! what the compiler is told about alignment.

/// Baseline: plain `*const i64`, 8-byte alignment guaranteed. This is what the executor has today.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn probe_plain(
    input: *const i64,
    output: *mut i64,
    reference: i64,
    len: usize,
) {
    let mut i = tid() * 2;
    while i + 1 < len {
        *output.add(i) = *input.add(i) + reference;
        *output.add(i + 1) = *input.add(i + 1) + reference;
        i += stride() * 2;
    }
}

/// Tell LLVM the base pointers are 16-byte aligned, keeping the scalar element accesses.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn probe_assume_aligned(
    input: *const i64,
    output: *mut i64,
    reference: i64,
    len: usize,
) {
    core::hint::assert_unchecked(input as usize % 16 == 0);
    core::hint::assert_unchecked(output as usize % 16 == 0);

    let mut i = tid() * 2;
    while i + 1 < len {
        *output.add(i) = *input.add(i) + reference;
        *output.add(i + 1) = *input.add(i + 1) + reference;
        i += stride() * 2;
    }
}

/// A 16-byte aligned pair, read and written as one value.
#[repr(align(16))]
#[derive(Clone, Copy)]
struct Pair([i64; 2]);

/// Access through the aligned pair type instead of through element pointers.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn probe_aligned_chunk(
    input: *const i64,
    output: *mut i64,
    reference: i64,
    len: usize,
) {
    let input = input as *const Pair;
    let output = output as *mut Pair;
    let pairs = len / 2;

    let mut i = tid();
    while i < pairs {
        let Pair([a, b]) = *input.add(i);
        *output.add(i) = Pair([a + reference, b + reference]);
        i += stride();
    }
}

/// Four-wide aligned chunk, the 32-byte case.
#[repr(align(16))]
#[derive(Clone, Copy)]
struct Quad([i32; 4]);

#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn probe_aligned_quad_i32(
    input: *const i32,
    output: *mut i32,
    reference: i32,
    len: usize,
) {
    let input = input as *const Quad;
    let output = output as *mut Quad;
    let quads = len / 4;

    let mut i = tid();
    while i < quads {
        let Quad([a, b, c, d]) = *input.add(i);
        *output.add(i) = Quad([a + reference, b + reference, c + reference, d + reference]);
        i += stride();
    }
}

#[inline(always)]
fn tid() -> usize {
    unsafe {
        let ctaid: u32;
        let ntid: u32;
        let t: u32;
        core::arch::asm!("mov.u32 {}, %ctaid.x;", out(reg32) ctaid, options(nomem, nostack));
        core::arch::asm!("mov.u32 {}, %ntid.x;", out(reg32) ntid, options(nomem, nostack));
        core::arch::asm!("mov.u32 {}, %tid.x;", out(reg32) t, options(nomem, nostack));
        (ctaid * ntid + t) as usize
    }
}

#[inline(always)]
fn stride() -> usize {
    unsafe {
        let nctaid: u32;
        let ntid: u32;
        core::arch::asm!("mov.u32 {}, %nctaid.x;", out(reg32) nctaid, options(nomem, nostack));
        core::arch::asm!("mov.u32 {}, %ntid.x;", out(reg32) ntid, options(nomem, nostack));
        (nctaid * ntid) as usize
    }
}
```

## Provable alignment with scalar accesses

```rust
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn provably_aligned_scalar_access(
    input: *const Pair, output: *mut Pair, reference: i64, len: usize,
) {
    let mut i = tid();
    while i < len {
        let a = (*input.add(i)).0[0];
        let b = (*input.add(i)).0[1];
        (*output.add(i)).0[0] = a + reference;
        (*output.add(i)).0[1] = b + reference;
        i += stride();
    }
}
```

```ptx
$L__BB7_2:
	add.s64 	%rd9, %rd1, %rd15;
	ld.global.v2.b64 	{%rd10, %rd11}, [%rd9];
	add.s64 	%rd12, %rd2, %rd15;
	add.s64 	%rd13, %rd10, %rd5;
	add.s64 	%rd14, %rd11, %rd5;
	st.global.v2.b64 	[%rd12], {%rd13, %rd14};
	setp.lt.u64 	%p2, %rd16, %rd6;
	@%p2 bra 	$L__BB7_2;
```
