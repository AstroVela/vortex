<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# `stage2-base-2`

```text
Timer precision: 10 ns
binary_ops           fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ add_i64_constant  8.299 µs      │ 47.08 µs      │ 8.429 µs      │ 8.916 µs      │ 100     │ 100
│                    3.948 Gitem/s │ 695.8 Mitem/s │ 3.887 Gitem/s │ 3.675 Gitem/s │         │
├─ add_i64_nonnull   9.139 µs      │ 12.35 µs      │ 9.209 µs      │ 9.286 µs      │ 100     │ 100
│                    3.585 Gitem/s │ 2.651 Gitem/s │ 3.557 Gitem/s │ 3.528 Gitem/s │         │
├─ div_i64_nonnull   44.8 µs       │ 49.64 µs      │ 44.87 µs      │ 45.09 µs      │ 100     │ 100
│                    731.2 Mitem/s │ 659.9 Mitem/s │ 730.1 Mitem/s │ 726.6 Mitem/s │         │
├─ mul_i8_nonnull    5.869 µs      │ 9.279 µs      │ 6.174 µs      │ 6.295 µs      │ 100     │ 100
│                    5.582 Gitem/s │ 3.531 Gitem/s │ 5.306 Gitem/s │ 5.204 Gitem/s │         │
├─ mul_i16_nonnull   4.039 µs      │ 7.909 µs      │ 4.104 µs      │ 4.153 µs      │ 100     │ 100
│                    8.111 Gitem/s │ 4.142 Gitem/s │ 7.982 Gitem/s │ 7.888 Gitem/s │         │
├─ mul_i32_constant  26.37 µs      │ 29.56 µs      │ 26.43 µs      │ 26.52 µs      │ 100     │ 100
│                    1.242 Gitem/s │ 1.108 Gitem/s │ 1.239 Gitem/s │ 1.235 Gitem/s │         │
├─ mul_i32_nonnull   26.36 µs      │ 30.91 µs      │ 26.41 µs      │ 26.53 µs      │ 100     │ 100
│                    1.242 Gitem/s │ 1.059 Gitem/s │ 1.24 Gitem/s  │ 1.234 Gitem/s │         │
├─ mul_i32_nullable  27.27 µs      │ 42.76 µs      │ 27.38 µs      │ 27.63 µs      │ 100     │ 100
│                    1.201 Gitem/s │ 766.1 Mitem/s │ 1.196 Gitem/s │ 1.185 Gitem/s │         │
├─ mul_i64_nonnull   23.14 µs      │ 26.52 µs      │ 23.24 µs      │ 23.33 µs      │ 100     │ 100
│                    1.415 Gitem/s │ 1.235 Gitem/s │ 1.409 Gitem/s │ 1.404 Gitem/s │         │
├─ mul_u8_nonnull    3.279 µs      │ 6.329 µs      │ 3.329 µs      │ 3.378 µs      │ 100     │ 100
│                    9.99 Gitem/s  │ 5.176 Gitem/s │ 9.84 Gitem/s  │ 9.698 Gitem/s │         │
├─ mul_u16_nonnull   2.549 µs      │ 3.489 µs      │ 2.599 µs      │ 2.615 µs      │ 100     │ 100
│                    12.85 Gitem/s │ 9.389 Gitem/s │ 12.6 Gitem/s  │ 12.52 Gitem/s │         │
├─ mul_u32_nonnull   6.879 µs      │ 9.609 µs      │ 6.939 µs      │ 7.003 µs      │ 100     │ 100
│                    4.762 Gitem/s │ 3.409 Gitem/s │ 4.721 Gitem/s │ 4.678 Gitem/s │         │
├─ mul_u64_nonnull   19.15 µs      │ 22.82 µs      │ 19.21 µs      │ 19.33 µs      │ 100     │ 100
│                    1.71 Gitem/s  │ 1.435 Gitem/s │ 1.704 Gitem/s │ 1.694 Gitem/s │         │
╰─ sub_i64_constant  8.119 µs      │ 11.08 µs      │ 8.249 µs      │ 8.293 µs      │ 100     │ 100
                     4.035 Gitem/s │ 2.954 Gitem/s │ 3.971 Gitem/s │ 3.951 Gitem/s │         │


```
