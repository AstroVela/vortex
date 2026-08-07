<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# `stage0-candidate-1`

```text
binary_ops           fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ add_i64_constant  9.13 µs       │ 93.77 µs      │ 9.269 µs      │ 10.17 µs      │ 100     │ 100
│                    3.588 Gitem/s │ 349.4 Mitem/s │ 3.534 Gitem/s │ 3.22 Gitem/s  │         │
├─ add_i64_nonnull   9.369 µs      │ 12.45 µs      │ 9.455 µs      │ 9.51 µs       │ 100     │ 100
│                    3.497 Gitem/s │ 2.629 Gitem/s │ 3.465 Gitem/s │ 3.445 Gitem/s │         │
├─ div_i64_nonnull   45.01 µs      │ 54.4 µs       │ 45.09 µs      │ 45.42 µs      │ 100     │ 100
│                    727.8 Mitem/s │ 602.2 Mitem/s │ 726.5 Mitem/s │ 721.4 Mitem/s │         │
├─ mul_i8_nonnull    4.639 µs      │ 12.25 µs      │ 4.694 µs      │ 4.777 µs      │ 100     │ 100
│                    7.062 Gitem/s │ 2.672 Gitem/s │ 6.979 Gitem/s │ 6.858 Gitem/s │         │
├─ mul_i16_nonnull   4.219 µs      │ 6.999 µs      │ 4.269 µs      │ 4.33 µs       │ 100     │ 100
│                    7.765 Gitem/s │ 4.681 Gitem/s │ 7.674 Gitem/s │ 7.567 Gitem/s │         │
├─ mul_i32_constant  18.77 µs      │ 21.99 µs      │ 18.88 µs      │ 19 µs         │ 100     │ 100
│                    1.744 Gitem/s │ 1.489 Gitem/s │ 1.734 Gitem/s │ 1.724 Gitem/s │         │
├─ mul_i32_nonnull   28.23 µs      │ 31.75 µs      │ 28.39 µs      │ 28.48 µs      │ 100     │ 100
│                    1.16 Gitem/s  │ 1.031 Gitem/s │ 1.153 Gitem/s │ 1.15 Gitem/s  │         │
├─ mul_i32_nullable  29.04 µs      │ 44.74 µs      │ 29.18 µs      │ 29.42 µs      │ 100     │ 100
│                    1.128 Gitem/s │ 732.2 Mitem/s │ 1.122 Gitem/s │ 1.113 Gitem/s │         │
├─ mul_i64_nonnull   29.7 µs       │ 34.12 µs      │ 30.02 µs      │ 30.14 µs      │ 100     │ 100
│                    1.102 Gitem/s │ 960.1 Mitem/s │ 1.091 Gitem/s │ 1.087 Gitem/s │         │
├─ mul_u8_nonnull    3.489 µs      │ 7.519 µs      │ 3.539 µs      │ 3.602 µs      │ 100     │ 100
│                    9.389 Gitem/s │ 4.357 Gitem/s │ 9.257 Gitem/s │ 9.095 Gitem/s │         │
├─ mul_u16_nonnull   2.349 µs      │ 3.849 µs      │ 2.429 µs      │ 2.446 µs      │ 100     │ 100
│                    13.94 Gitem/s │ 8.511 Gitem/s │ 13.48 Gitem/s │ 13.39 Gitem/s │         │
├─ mul_u32_nonnull   6.999 µs      │ 9.889 µs      │ 7.069 µs      │ 7.11 µs       │ 100     │ 100
│                    4.681 Gitem/s │ 3.313 Gitem/s │ 4.634 Gitem/s │ 4.608 Gitem/s │         │
├─ mul_u64_nonnull   30.36 µs      │ 33.89 µs      │ 30.43 µs      │ 30.55 µs      │ 100     │ 100
│                    1.078 Gitem/s │ 966.6 Mitem/s │ 1.076 Gitem/s │ 1.072 Gitem/s │         │
╰─ sub_i64_constant  8.979 µs      │ 12.15 µs      │ 9.099 µs      │ 9.159 µs      │ 100     │ 100
                     3.649 Gitem/s │ 2.696 Gitem/s │ 3.6 Gitem/s   │ 3.577 Gitem/s │         │


```
