<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# `land-base-2`

```text
Timer precision: 10 ns
binary_ops           fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ add_i64_constant  8.319 µs      │ 39.44 µs      │ 8.449 µs      │ 8.813 µs      │ 100     │ 100
│                    3.938 Gitem/s │ 830.8 Mitem/s │ 3.877 Gitem/s │ 3.718 Gitem/s │         │
├─ add_i64_nonnull   9.169 µs      │ 12.56 µs      │ 9.239 µs      │ 9.304 µs      │ 100     │ 100
│                    3.573 Gitem/s │ 2.608 Gitem/s │ 3.546 Gitem/s │ 3.521 Gitem/s │         │
├─ div_i64_nonnull   44.77 µs      │ 50.37 µs      │ 44.86 µs      │ 45.15 µs      │ 100     │ 100
│                    731.7 Mitem/s │ 650.4 Mitem/s │ 730.2 Mitem/s │ 725.7 Mitem/s │         │
├─ mul_i8_nonnull    5.829 µs      │ 8.179 µs      │ 6.199 µs      │ 6.265 µs      │ 100     │ 100
│                    5.62 Gitem/s  │ 4.005 Gitem/s │ 5.285 Gitem/s │ 5.23 Gitem/s  │         │
├─ mul_i16_nonnull   4.049 µs      │ 8.029 µs      │ 4.109 µs      │ 4.15 µs       │ 100     │ 100
│                    8.091 Gitem/s │ 4.08 Gitem/s  │ 7.973 Gitem/s │ 7.895 Gitem/s │         │
├─ mul_i32_constant  26.37 µs      │ 35.43 µs      │ 26.44 µs      │ 26.77 µs      │ 100     │ 100
│                    1.242 Gitem/s │ 924.6 Mitem/s │ 1.239 Gitem/s │ 1.223 Gitem/s │         │
├─ mul_i32_nonnull   26.37 µs      │ 35.11 µs      │ 26.42 µs      │ 26.84 µs      │ 100     │ 100
│                    1.242 Gitem/s │ 933 Mitem/s   │ 1.239 Gitem/s │ 1.22 Gitem/s  │         │
├─ mul_i32_nullable  27.26 µs      │ 40.16 µs      │ 27.36 µs      │ 27.78 µs      │ 100     │ 100
│                    1.201 Gitem/s │ 815.9 Mitem/s │ 1.197 Gitem/s │ 1.179 Gitem/s │         │
├─ mul_i64_nonnull   23.24 µs      │ 32.01 µs      │ 23.35 µs      │ 23.63 µs      │ 100     │ 100
│                    1.409 Gitem/s │ 1.023 Gitem/s │ 1.402 Gitem/s │ 1.386 Gitem/s │         │
├─ mul_u8_nonnull    3.26 µs       │ 4.759 µs      │ 3.319 µs      │ 3.349 µs      │ 100     │ 100
│                    10.04 Gitem/s │ 6.884 Gitem/s │ 9.87 Gitem/s  │ 9.783 Gitem/s │         │
├─ mul_u16_nonnull   2.53 µs       │ 7.289 µs      │ 2.609 µs      │ 2.664 µs      │ 100     │ 100
│                    12.94 Gitem/s │ 4.495 Gitem/s │ 12.55 Gitem/s │ 12.29 Gitem/s │         │
├─ mul_u32_nonnull   6.889 µs      │ 12.89 µs      │ 6.959 µs      │ 7.027 µs      │ 100     │ 100
│                    4.756 Gitem/s │ 2.54 Gitem/s  │ 4.708 Gitem/s │ 4.663 Gitem/s │         │
├─ mul_u64_nonnull   19.13 µs      │ 22.83 µs      │ 19.21 µs      │ 19.29 µs      │ 100     │ 100
│                    1.712 Gitem/s │ 1.435 Gitem/s │ 1.704 Gitem/s │ 1.698 Gitem/s │         │
╰─ sub_i64_constant  8.139 µs      │ 11.06 µs      │ 8.259 µs      │ 8.297 µs      │ 100     │ 100
                     4.025 Gitem/s │ 2.96 Gitem/s  │ 3.967 Gitem/s │ 3.949 Gitem/s │         │


```
