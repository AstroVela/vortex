<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# `land-final-2`

```text
Timer precision: 10 ns
binary_ops           fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ add_i64_constant  9.149 µs      │ 73.57 µs      │ 9.279 µs      │ 9.987 µs      │ 100     │ 100
│                    3.581 Gitem/s │ 445.3 Mitem/s │ 3.531 Gitem/s │ 3.28 Gitem/s  │         │
├─ add_i64_nonnull   9.319 µs      │ 14.28 µs      │ 9.389 µs      │ 9.475 µs      │ 100     │ 100
│                    3.515 Gitem/s │ 2.293 Gitem/s │ 3.489 Gitem/s │ 3.458 Gitem/s │         │
├─ div_i64_nonnull   44.96 µs      │ 62.15 µs      │ 45.06 µs      │ 45.46 µs      │ 100     │ 100
│                    728.6 Mitem/s │ 527.2 Mitem/s │ 727 Mitem/s   │ 720.7 Mitem/s │         │
├─ mul_i8_nonnull    5.979 µs      │ 41.01 µs      │ 6.409 µs      │ 6.853 µs      │ 100     │ 100
│                    5.479 Gitem/s │ 798.8 Mitem/s │ 5.112 Gitem/s │ 4.78 Gitem/s  │         │
├─ mul_i16_nonnull   4.229 µs      │ 5.739 µs      │ 4.299 µs      │ 4.314 µs      │ 100     │ 100
│                    7.746 Gitem/s │ 5.708 Gitem/s │ 7.62 Gitem/s  │ 7.595 Gitem/s │         │
├─ mul_i32_constant  18.6 µs       │ 23.91 µs      │ 18.7 µs       │ 18.79 µs      │ 100     │ 100
│                    1.76 Gitem/s  │ 1.369 Gitem/s │ 1.751 Gitem/s │ 1.743 Gitem/s │         │
├─ mul_i32_nonnull   26.56 µs      │ 30.13 µs      │ 26.64 µs      │ 26.74 µs      │ 100     │ 100
│                    1.233 Gitem/s │ 1.087 Gitem/s │ 1.229 Gitem/s │ 1.225 Gitem/s │         │
├─ mul_i32_nullable  27.35 µs      │ 42.74 µs      │ 27.44 µs      │ 27.69 µs      │ 100     │ 100
│                    1.197 Gitem/s │ 766.5 Mitem/s │ 1.193 Gitem/s │ 1.183 Gitem/s │         │
├─ mul_i64_nonnull   23.31 µs      │ 27.31 µs      │ 23.43 µs      │ 23.56 µs      │ 100     │ 100
│                    1.405 Gitem/s │ 1.199 Gitem/s │ 1.398 Gitem/s │ 1.39 Gitem/s  │         │
├─ mul_u8_nonnull    3.479 µs      │ 52.76 µs      │ 3.549 µs      │ 4.103 µs      │ 100     │ 100
│                    9.416 Gitem/s │ 620.9 Mitem/s │ 9.231 Gitem/s │ 7.984 Gitem/s │         │
├─ mul_u16_nonnull   2.739 µs      │ 3.799 µs      │ 2.81 µs       │ 2.824 µs      │ 100     │ 100
│                    11.96 Gitem/s │ 8.623 Gitem/s │ 11.66 Gitem/s │ 11.6 Gitem/s  │         │
├─ mul_u32_nonnull   7.089 µs      │ 10.28 µs      │ 7.149 µs      │ 7.207 µs      │ 100     │ 100
│                    4.621 Gitem/s │ 3.184 Gitem/s │ 4.583 Gitem/s │ 4.546 Gitem/s │         │
├─ mul_u64_nonnull   19.32 µs      │ 23.55 µs      │ 19.38 µs      │ 19.47 µs      │ 100     │ 100
│                    1.695 Gitem/s │ 1.391 Gitem/s │ 1.689 Gitem/s │ 1.682 Gitem/s │         │
╰─ sub_i64_constant  8.939 µs      │ 30.07 µs      │ 9.159 µs      │ 9.386 µs      │ 100     │ 100
                     3.665 Gitem/s │ 1.089 Gitem/s │ 3.577 Gitem/s │ 3.49 Gitem/s  │         │


```
