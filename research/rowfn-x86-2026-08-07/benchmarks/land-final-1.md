<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# `land-final-1`

```text
Timer precision: 10 ns
binary_ops           fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ add_i64_constant  8.979 µs      │ 88.65 µs      │ 9.269 µs      │ 10.12 µs      │ 100     │ 100
│                    3.649 Gitem/s │ 369.6 Mitem/s │ 3.534 Gitem/s │ 3.237 Gitem/s │         │
├─ add_i64_nonnull   9.299 µs      │ 13.4 µs       │ 9.379 µs      │ 9.444 µs      │ 100     │ 100
│                    3.523 Gitem/s │ 2.443 Gitem/s │ 3.493 Gitem/s │ 3.469 Gitem/s │         │
├─ div_i64_nonnull   44.96 µs      │ 55 µs         │ 45.02 µs      │ 45.48 µs      │ 100     │ 100
│                    728.8 Mitem/s │ 595.6 Mitem/s │ 727.6 Mitem/s │ 720.4 Mitem/s │         │
├─ mul_i8_nonnull    5.959 µs      │ 44.56 µs      │ 6.389 µs      │ 6.855 µs      │ 100     │ 100
│                    5.498 Gitem/s │ 735.3 Mitem/s │ 5.128 Gitem/s │ 4.78 Gitem/s  │         │
├─ mul_i16_nonnull   4.199 µs      │ 7.599 µs      │ 4.265 µs      │ 4.321 µs      │ 100     │ 100
│                    7.802 Gitem/s │ 4.311 Gitem/s │ 7.682 Gitem/s │ 7.581 Gitem/s │         │
├─ mul_i32_constant  18.6 µs       │ 22.22 µs      │ 18.69 µs      │ 18.81 µs      │ 100     │ 100
│                    1.76 Gitem/s  │ 1.474 Gitem/s │ 1.753 Gitem/s │ 1.741 Gitem/s │         │
├─ mul_i32_nonnull   26.52 µs      │ 34.76 µs      │ 26.59 µs      │ 26.77 µs      │ 100     │ 100
│                    1.235 Gitem/s │ 942.6 Mitem/s │ 1.231 Gitem/s │ 1.223 Gitem/s │         │
├─ mul_i32_nullable  27.3 µs       │ 51.11 µs      │ 27.4 µs       │ 27.77 µs      │ 100     │ 100
│                    1.199 Gitem/s │ 641 Mitem/s   │ 1.195 Gitem/s │ 1.179 Gitem/s │         │
├─ mul_i64_nonnull   23.37 µs      │ 27.17 µs      │ 23.46 µs      │ 23.56 µs      │ 100     │ 100
│                    1.401 Gitem/s │ 1.206 Gitem/s │ 1.396 Gitem/s │ 1.39 Gitem/s  │         │
├─ mul_u8_nonnull    3.459 µs      │ 59.52 µs      │ 3.514 µs      │ 4.079 µs      │ 100     │ 100
│                    9.471 Gitem/s │ 550.5 Mitem/s │ 9.322 Gitem/s │ 8.033 Gitem/s │         │
├─ mul_u16_nonnull   2.709 µs      │ 3.799 µs      │ 2.789 µs      │ 2.798 µs      │ 100     │ 100
│                    12.09 Gitem/s │ 8.623 Gitem/s │ 11.74 Gitem/s │ 11.71 Gitem/s │         │
├─ mul_u32_nonnull   7.049 µs      │ 10.9 µs       │ 7.129 µs      │ 7.19 µs       │ 100     │ 100
│                    4.648 Gitem/s │ 3.003 Gitem/s │ 4.595 Gitem/s │ 4.556 Gitem/s │         │
├─ mul_u64_nonnull   19.26 µs      │ 22.46 µs      │ 19.37 µs      │ 19.45 µs      │ 100     │ 100
│                    1.7 Gitem/s   │ 1.458 Gitem/s │ 1.691 Gitem/s │ 1.683 Gitem/s │         │
╰─ sub_i64_constant  9.009 µs      │ 13.94 µs      │ 9.149 µs      │ 9.215 µs      │ 100     │ 100
                     3.636 Gitem/s │ 2.348 Gitem/s │ 3.581 Gitem/s │ 3.555 Gitem/s │         │


```
