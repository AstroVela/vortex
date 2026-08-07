<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# `stage0-candidate-2`

```text
binary_ops           fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ add_i64_constant  9.159 µs      │ 88.36 µs      │ 9.29 µs       │ 10.41 µs      │ 100     │ 100
│                    3.577 Gitem/s │ 370.8 Mitem/s │ 3.527 Gitem/s │ 3.147 Gitem/s │         │
├─ add_i64_nonnull   9.389 µs      │ 12.37 µs      │ 9.49 µs       │ 9.622 µs      │ 100     │ 100
│                    3.489 Gitem/s │ 2.646 Gitem/s │ 3.452 Gitem/s │ 3.405 Gitem/s │         │
├─ div_i64_nonnull   45.1 µs       │ 48.73 µs      │ 45.16 µs      │ 45.34 µs      │ 100     │ 100
│                    726.4 Mitem/s │ 672.3 Mitem/s │ 725.5 Mitem/s │ 722.6 Mitem/s │         │
├─ mul_i8_nonnull    4.639 µs      │ 7.529 µs      │ 4.699 µs      │ 4.751 µs      │ 100     │ 100
│                    7.062 Gitem/s │ 4.351 Gitem/s │ 6.972 Gitem/s │ 6.897 Gitem/s │         │
├─ mul_i16_nonnull   4.199 µs      │ 6.379 µs      │ 4.269 µs      │ 4.295 µs      │ 100     │ 100
│                    7.802 Gitem/s │ 5.136 Gitem/s │ 7.674 Gitem/s │ 7.629 Gitem/s │         │
├─ mul_i32_constant  18.77 µs      │ 24.05 µs      │ 18.87 µs      │ 19 µs         │ 100     │ 100
│                    1.744 Gitem/s │ 1.361 Gitem/s │ 1.736 Gitem/s │ 1.724 Gitem/s │         │
├─ mul_i32_nonnull   28.19 µs      │ 31.61 µs      │ 28.35 µs      │ 28.45 µs      │ 100     │ 100
│                    1.161 Gitem/s │ 1.036 Gitem/s │ 1.155 Gitem/s │ 1.151 Gitem/s │         │
├─ mul_i32_nullable  29 µs         │ 50.06 µs      │ 29.15 µs      │ 29.47 µs      │ 100     │ 100
│                    1.129 Gitem/s │ 654.5 Mitem/s │ 1.123 Gitem/s │ 1.111 Gitem/s │         │
├─ mul_i64_nonnull   29.82 µs      │ 33.7 µs       │ 30.08 µs      │ 30.21 µs      │ 100     │ 100
│                    1.098 Gitem/s │ 972 Mitem/s   │ 1.089 Gitem/s │ 1.084 Gitem/s │         │
├─ mul_u8_nonnull    3.469 µs      │ 9.249 µs      │ 3.529 µs      │ 3.592 µs      │ 100     │ 100
│                    9.443 Gitem/s │ 3.542 Gitem/s │ 9.283 Gitem/s │ 9.119 Gitem/s │         │
├─ mul_u16_nonnull   2.369 µs      │ 3.699 µs      │ 2.429 µs      │ 2.448 µs      │ 100     │ 100
│                    13.82 Gitem/s │ 8.856 Gitem/s │ 13.48 Gitem/s │ 13.38 Gitem/s │         │
├─ mul_u32_nonnull   6.989 µs      │ 9.659 µs      │ 7.059 µs      │ 7.111 µs      │ 100     │ 100
│                    4.687 Gitem/s │ 3.392 Gitem/s │ 4.641 Gitem/s │ 4.607 Gitem/s │         │
├─ mul_u64_nonnull   30.41 µs      │ 33.95 µs      │ 30.49 µs      │ 30.63 µs      │ 100     │ 100
│                    1.077 Gitem/s │ 964.9 Mitem/s │ 1.074 Gitem/s │ 1.069 Gitem/s │         │
╰─ sub_i64_constant  8.989 µs      │ 11.76 µs      │ 9.099 µs      │ 9.169 µs      │ 100     │ 100
                     3.645 Gitem/s │ 2.784 Gitem/s │ 3.6 Gitem/s   │ 3.573 Gitem/s │         │


```
