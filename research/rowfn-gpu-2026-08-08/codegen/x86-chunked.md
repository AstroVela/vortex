<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Chunked access on x86

The same row bodies under the current one-lane-at-a-time shape and under an aligned-chunk shape,
compiled for x86-64. Vector register counts are occurrences in the emitted function body.

This is code generation only. There are no timings, and LLVM preferring `ymm` over `zmm` is often a
deliberate heuristic rather than a defect.

| Target | Function | zmm | ymm | imulq |
| --- | --- | ---: | ---: | ---: |
| x86-64-v3 | `add_scalar` | 0 | 15 | 0 |
| x86-64-v3 | `add_chunked` | 0 | 18 | 0 |
| x86-64-v3 | `checked_mul_scalar` | 0 | 0 | 5 |
| x86-64-v3 | `checked_mul_chunked` | 0 | 0 | 8 |
| x86-64-v4 | `add_scalar` | 0 | 15 | 0 |
| x86-64-v4 | `add_chunked` | 6 | 6 | 0 |
| x86-64-v4 | `checked_mul_scalar` | 0 | 0 | 5 |
| x86-64-v4 | `checked_mul_chunked` | 0 | 0 | 8 |

Two results. Under AVX-512 the chunked shape reaches `zmm` where the scalar loop stays on `ymm`.
The checked `i64` multiply does not vectorize under either shape, because x86 has no vector 64-bit
multiply that yields overflow evidence, which is consistent with `mul_i64` and `mul_u64` being the
problem shapes in the x86 record.

The chunked loads are also `vmovdqa` rather than `vmovdqu`.

## Source

```rust
//! Does the chunked access path change x86 codegen, or does the autovectorizer already capture it?
//!
//! Each pair is the same row body under the current one-lane-at-a-time shape and under an
//! aligned-chunk shape.

#[repr(align(64))]
#[derive(Clone, Copy)]
pub struct Chunk8<T>(pub [T; 8]);

/// Current shape: one lane at a time, as `IndexedSource::get_unchecked` provides.
#[unsafe(no_mangle)]
pub fn add_scalar(lhs: &[i64], rhs: &[i64], out: &mut [i64]) {
    let n = lhs.len().min(rhs.len()).min(out.len());
    for i in 0..n {
        out[i] = lhs[i].wrapping_add(rhs[i]);
    }
}

/// Chunked shape: one aligned aggregate per iteration, row body applied per lane.
#[unsafe(no_mangle)]
pub fn add_chunked(lhs: &[Chunk8<i64>], rhs: &[Chunk8<i64>], out: &mut [Chunk8<i64>]) {
    let n = lhs.len().min(rhs.len()).min(out.len());
    for i in 0..n {
        let mut v = [0i64; 8];
        for k in 0..8 {
            v[k] = lhs[i].0[k].wrapping_add(rhs[i].0[k]);
        }
        out[i] = Chunk8(v);
    }
}

/// The epic's hard shape: checked multiply with an OR-reduced failure word, one lane at a time.
#[unsafe(no_mangle)]
pub fn checked_mul_scalar(lhs: &[i64], rhs: &[i64], out: &mut [i64]) -> u64 {
    let n = lhs.len().min(rhs.len()).min(out.len());
    let mut failed = 0u64;
    for i in 0..n {
        let (v, o) = lhs[i].overflowing_mul(rhs[i]);
        out[i] = v;
        failed |= o as u64;
    }
    failed
}

/// The same checked multiply under the chunked shape.
#[unsafe(no_mangle)]
pub fn checked_mul_chunked(lhs: &[Chunk8<i64>], rhs: &[Chunk8<i64>], out: &mut [Chunk8<i64>]) -> u64 {
    let n = lhs.len().min(rhs.len()).min(out.len());
    let mut failed = 0u64;
    for i in 0..n {
        let mut v = [0i64; 8];
        let mut f = 0u64;
        for k in 0..8 {
            let (value, o) = lhs[i].0[k].overflowing_mul(rhs[i].0[k]);
            v[k] = value;
            f |= o as u64;
        }
        out[i] = Chunk8(v);
        failed |= f;
    }
    failed
}
```
