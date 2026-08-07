<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# `stage2-candidate-1`

```text
Timer precision: 10 ns
binary_ops           fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ add_i64_constant  9.099 µs      │ 1.015 ms      │ 9.269 µs      │ 19.41 µs      │ 100     │ 100
│                    3.6 Gitem/s   │ 32.26 Mitem/s │ 3.534 Gitem/s │ 1.687 Gitem/s │         │
├─ add_i64_nonnull   9.319 µs      │ 12.12 µs      │ 9.429 µs      │ 9.48 µs       │ 100     │ 100
│                    3.515 Gitem/s │ 2.701 Gitem/s │ 3.474 Gitem/s │ 3.456 Gitem/s │         │
├─ div_i64_nonnull   44.99 µs      │ 61.85 µs      │ 45.07 µs      │ 45.6 µs       │ 100     │ 100
│                    728.1 Mitem/s │ 529.7 Mitem/s │ 726.8 Mitem/s │ 718.4 Mitem/s │         │
├─ mul_i8_nonnull    4.639 µs      │ 61.66 µs      │ 4.689 µs      │ 5.268 µs      │ 100     │ 100
│                    7.062 Gitem/s │ 531.3 Mitem/s │ 6.987 Gitem/s │ 6.219 Gitem/s │         │
├─ mul_i16_nonnull   4.209 µs      │ 66.06 µs      │ 4.279 µs      │ 4.931 µs      │ 100     │ 100
│                    7.783 Gitem/s │ 495.9 Mitem/s │ 7.656 Gitem/s │ 6.644 Gitem/s │         │
├─ mul_i32_constant  18.77 µs      │ 71.76 µs      │ 18.88 µs      │ 19.64 µs      │ 100     │ 100
│                    1.744 Gitem/s │ 456.5 Mitem/s │ 1.734 Gitem/s │ 1.667 Gitem/s │         │
├─ mul_i32_nonnull   28.21 µs      │ 33.28 µs      │ 28.34 µs      │ 28.48 µs      │ 100     │ 100
│                    1.161 Gitem/s │ 984.3 Mitem/s │ 1.155 Gitem/s │ 1.15 Gitem/s  │         │
├─ mul_i32_nullable  29.02 µs      │ 233.9 µs      │ 29.2 µs       │ 31.37 µs      │ 100     │ 100
│                    1.128 Gitem/s │ 140 Mitem/s   │ 1.121 Gitem/s │ 1.044 Gitem/s │         │
├─ mul_i64_nonnull   29.73 µs      │ 52.85 µs      │ 30.02 µs      │ 30.37 µs      │ 100     │ 100
│                    1.101 Gitem/s │ 619.9 Mitem/s │ 1.091 Gitem/s │ 1.078 Gitem/s │         │
├─ mul_u8_nonnull    3.449 µs      │ 14.5 µs       │ 3.529 µs      │ 3.679 µs      │ 100     │ 100
│                    9.498 Gitem/s │ 2.258 Gitem/s │ 9.283 Gitem/s │ 8.906 Gitem/s │         │
├─ mul_u16_nonnull   2.349 µs      │ 13.44 µs      │ 2.419 µs      │ 2.529 µs      │ 100     │ 100
│                    13.94 Gitem/s │ 2.436 Gitem/s │ 13.54 Gitem/s │ 12.95 Gitem/s │         │
├─ mul_u32_nonnull   6.979 µs      │ 18.54 µs      │ 7.059 µs      │ 7.228 µs      │ 100     │ 100
│                    4.694 Gitem/s │ 1.766 Gitem/s │ 4.641 Gitem/s │ 4.532 Gitem/s │         │
├─ mul_u64_nonnull   30.31 µs      │ 42.57 µs      │ 30.41 µs      │ 30.68 µs      │ 100     │ 100
│                    1.08 Gitem/s  │ 769.5 Mitem/s │ 1.077 Gitem/s │ 1.067 Gitem/s │         │
╰─ sub_i64_constant  8.979 µs      │ 32.68 µs      │ 9.064 µs      │ 9.358 µs      │ 100     │ 100
                     3.649 Gitem/s │ 1.002 Gitem/s │ 3.614 Gitem/s │ 3.501 Gitem/s │         │


```
