<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# `stage2-base-1`

```text
Timer precision: 10 ns
binary_ops           fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ add_i64_constant  8.099 µs      │ 766.1 µs      │ 8.419 µs      │ 16.03 µs      │ 100     │ 100
│                    4.045 Gitem/s │ 42.76 Mitem/s │ 3.891 Gitem/s │ 2.043 Gitem/s │         │
├─ add_i64_nonnull   9.099 µs      │ 30.45 µs      │ 9.179 µs      │ 9.474 µs      │ 100     │ 100
│                    3.6 Gitem/s   │ 1.075 Gitem/s │ 3.569 Gitem/s │ 3.458 Gitem/s │         │
├─ div_i64_nonnull   44.76 µs      │ 75.04 µs      │ 44.84 µs      │ 45.32 µs      │ 100     │ 100
│                    731.9 Mitem/s │ 436.6 Mitem/s │ 730.6 Mitem/s │ 722.9 Mitem/s │         │
├─ mul_i8_nonnull    5.809 µs      │ 69.77 µs      │ 6.209 µs      │ 6.952 µs      │ 100     │ 100
│                    5.64 Gitem/s  │ 469.5 Mitem/s │ 5.276 Gitem/s │ 4.713 Gitem/s │         │
├─ mul_i16_nonnull   4.029 µs      │ 66.19 µs      │ 4.099 µs      │ 4.725 µs      │ 100     │ 100
│                    8.131 Gitem/s │ 494.9 Mitem/s │ 7.992 Gitem/s │ 6.934 Gitem/s │         │
├─ mul_i32_constant  26.36 µs      │ 55.21 µs      │ 26.42 µs      │ 26.88 µs      │ 100     │ 100
│                    1.242 Gitem/s │ 593.4 Mitem/s │ 1.239 Gitem/s │ 1.218 Gitem/s │         │
├─ mul_i32_nonnull   26.34 µs      │ 39.19 µs      │ 26.39 µs      │ 26.64 µs      │ 100     │ 100
│                    1.243 Gitem/s │ 835.9 Mitem/s │ 1.241 Gitem/s │ 1.229 Gitem/s │         │
├─ mul_i32_nullable  27.21 µs      │ 333.7 µs      │ 27.37 µs      │ 30.56 µs      │ 100     │ 100
│                    1.203 Gitem/s │ 98.17 Mitem/s │ 1.196 Gitem/s │ 1.072 Gitem/s │         │
├─ mul_i64_nonnull   23.14 µs      │ 43.91 µs      │ 23.22 µs      │ 23.6 µs       │ 100     │ 100
│                    1.415 Gitem/s │ 746 Mitem/s   │ 1.41 Gitem/s  │ 1.388 Gitem/s │         │
├─ mul_u8_nonnull    3.259 µs      │ 52.38 µs      │ 3.329 µs      │ 3.817 µs      │ 100     │ 100
│                    10.05 Gitem/s │ 625.4 Mitem/s │ 9.84 Gitem/s  │ 8.582 Gitem/s │         │
├─ mul_u16_nonnull   2.549 µs      │ 30.18 µs      │ 2.609 µs      │ 2.929 µs      │ 100     │ 100
│                    12.85 Gitem/s │ 1.085 Gitem/s │ 12.55 Gitem/s │ 11.18 Gitem/s │         │
├─ mul_u32_nonnull   6.869 µs      │ 27.32 µs      │ 6.939 µs      │ 7.147 µs      │ 100     │ 100
│                    4.769 Gitem/s │ 1.198 Gitem/s │ 4.721 Gitem/s │ 4.584 Gitem/s │         │
├─ mul_u64_nonnull   19.14 µs      │ 41.82 µs      │ 19.22 µs      │ 19.57 µs      │ 100     │ 100
│                    1.711 Gitem/s │ 783.3 Mitem/s │ 1.704 Gitem/s │ 1.674 Gitem/s │         │
╰─ sub_i64_constant  8.159 µs      │ 40.98 µs      │ 8.249 µs      │ 8.63 µs       │ 100     │ 100
                     4.015 Gitem/s │ 799.4 Mitem/s │ 3.971 Gitem/s │ 3.796 Gitem/s │         │


```
