<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# `stage1-candidate-1`

```text
binary_ops           fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ add_i64_constant  9.149 µs      │ 1.038 ms      │ 9.279 µs      │ 19.64 µs      │ 100     │ 100
│                    3.581 Gitem/s │ 31.54 Mitem/s │ 3.531 Gitem/s │ 1.667 Gitem/s │         │
├─ add_i64_nonnull   9.399 µs      │ 23.57 µs      │ 9.459 µs      │ 9.66 µs       │ 100     │ 100
│                    3.486 Gitem/s │ 1.389 Gitem/s │ 3.463 Gitem/s │ 3.391 Gitem/s │         │
├─ div_i64_nonnull   45.03 µs      │ 63.07 µs      │ 45.1 µs       │ 45.48 µs      │ 100     │ 100
│                    727.5 Mitem/s │ 519.4 Mitem/s │ 726.4 Mitem/s │ 720.4 Mitem/s │         │
├─ mul_i8_nonnull    4.629 µs      │ 65.15 µs      │ 4.689 µs      │ 5.344 µs      │ 100     │ 100
│                    7.077 Gitem/s │ 502.8 Mitem/s │ 6.987 Gitem/s │ 6.131 Gitem/s │         │
├─ mul_i16_nonnull   4.209 µs      │ 71.32 µs      │ 4.259 µs      │ 4.934 µs      │ 100     │ 100
│                    7.783 Gitem/s │ 459.3 Mitem/s │ 7.692 Gitem/s │ 6.64 Gitem/s  │         │
├─ mul_i32_constant  18.71 µs      │ 73.84 µs      │ 18.82 µs      │ 19.43 µs      │ 100     │ 100
│                    1.75 Gitem/s  │ 443.7 Mitem/s │ 1.74 Gitem/s  │ 1.685 Gitem/s │         │
├─ mul_i32_nonnull   28.22 µs      │ 32.07 µs      │ 28.34 µs      │ 28.45 µs      │ 100     │ 100
│                    1.16 Gitem/s  │ 1.021 Gitem/s │ 1.155 Gitem/s │ 1.151 Gitem/s │         │
├─ mul_i32_nullable  29.04 µs      │ 237.4 µs      │ 29.16 µs      │ 31.4 µs       │ 100     │ 100
│                    1.127 Gitem/s │ 138 Mitem/s   │ 1.123 Gitem/s │ 1.043 Gitem/s │         │
├─ mul_i64_nonnull   29.72 µs      │ 54.86 µs      │ 30.07 µs      │ 30.53 µs      │ 100     │ 100
│                    1.102 Gitem/s │ 597.1 Mitem/s │ 1.089 Gitem/s │ 1.073 Gitem/s │         │
├─ mul_u8_nonnull    3.469 µs      │ 15.4 µs       │ 3.529 µs      │ 3.658 µs      │ 100     │ 100
│                    9.443 Gitem/s │ 2.126 Gitem/s │ 9.283 Gitem/s │ 8.956 Gitem/s │         │
├─ mul_u16_nonnull   2.339 µs      │ 13.45 µs      │ 2.419 µs      │ 2.574 µs      │ 100     │ 100
│                    14 Gitem/s    │ 2.434 Gitem/s │ 13.54 Gitem/s │ 12.72 Gitem/s │         │
├─ mul_u32_nonnull   6.969 µs      │ 19.59 µs      │ 7.049 µs      │ 7.223 µs      │ 100     │ 100
│                    4.701 Gitem/s │ 1.671 Gitem/s │ 4.648 Gitem/s │ 4.536 Gitem/s │         │
├─ mul_u64_nonnull   30.36 µs      │ 42.47 µs      │ 30.45 µs      │ 30.69 µs      │ 100     │ 100
│                    1.079 Gitem/s │ 771.3 Mitem/s │ 1.075 Gitem/s │ 1.067 Gitem/s │         │
╰─ sub_i64_constant  9.009 µs      │ 31.68 µs      │ 9.119 µs      │ 9.431 µs      │ 100     │ 100
                     3.636 Gitem/s │ 1.034 Gitem/s │ 3.593 Gitem/s │ 3.474 Gitem/s │         │


```
