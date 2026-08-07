<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# `stage0-base-1`

```text
binary_ops           fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ add_i64_constant  8.319 µs      │ 62.83 µs      │ 8.449 µs      │ 9.036 µs      │ 100     │ 100
│                    3.938 Gitem/s │ 521.5 Mitem/s │ 3.877 Gitem/s │ 3.626 Gitem/s │         │
├─ add_i64_nonnull   9.139 µs      │ 13.11 µs      │ 9.205 µs      │ 9.316 µs      │ 100     │ 100
│                    3.585 Gitem/s │ 2.497 Gitem/s │ 3.559 Gitem/s │ 3.517 Gitem/s │         │
├─ div_i64_nonnull   44.78 µs      │ 66.09 µs      │ 44.85 µs      │ 45.23 µs      │ 100     │ 100
│                    731.5 Mitem/s │ 495.8 Mitem/s │ 730.4 Mitem/s │ 724.4 Mitem/s │         │
├─ mul_i8_nonnull    5.799 µs      │ 17.72 µs      │ 6.184 µs      │ 6.39 µs       │ 100     │ 100
│                    5.649 Gitem/s │ 1.848 Gitem/s │ 5.298 Gitem/s │ 5.127 Gitem/s │         │
├─ mul_i16_nonnull   4.009 µs      │ 10.13 µs      │ 4.099 µs      │ 4.214 µs      │ 100     │ 100
│                    8.172 Gitem/s │ 3.234 Gitem/s │ 7.992 Gitem/s │ 7.774 Gitem/s │         │
├─ mul_i32_constant  26.35 µs      │ 30.02 µs      │ 26.42 µs      │ 26.53 µs      │ 100     │ 100
│                    1.243 Gitem/s │ 1.091 Gitem/s │ 1.239 Gitem/s │ 1.234 Gitem/s │         │
├─ mul_i32_nonnull   26.36 µs      │ 30.26 µs      │ 26.41 µs      │ 26.51 µs      │ 100     │ 100
│                    1.242 Gitem/s │ 1.082 Gitem/s │ 1.24 Gitem/s  │ 1.235 Gitem/s │         │
├─ mul_i32_nullable  27.26 µs      │ 48.92 µs      │ 27.35 µs      │ 27.7 µs       │ 100     │ 100
│                    1.202 Gitem/s │ 669.8 Mitem/s │ 1.197 Gitem/s │ 1.182 Gitem/s │         │
├─ mul_i64_nonnull   23.13 µs      │ 28.03 µs      │ 23.22 µs      │ 23.37 µs      │ 100     │ 100
│                    1.416 Gitem/s │ 1.168 Gitem/s │ 1.41 Gitem/s  │ 1.402 Gitem/s │         │
├─ mul_u8_nonnull    3.259 µs      │ 6.989 µs      │ 3.319 µs      │ 3.365 µs      │ 100     │ 100
│                    10.05 Gitem/s │ 4.687 Gitem/s │ 9.87 Gitem/s  │ 9.735 Gitem/s │         │
├─ mul_u16_nonnull   2.539 µs      │ 3.489 µs      │ 2.599 µs      │ 2.613 µs      │ 100     │ 100
│                    12.9 Gitem/s  │ 9.389 Gitem/s │ 12.6 Gitem/s  │ 12.53 Gitem/s │         │
├─ mul_u32_nonnull   6.879 µs      │ 12.13 µs      │ 6.939 µs      │ 7.009 µs      │ 100     │ 100
│                    4.762 Gitem/s │ 2.699 Gitem/s │ 4.721 Gitem/s │ 4.674 Gitem/s │         │
├─ mul_u64_nonnull   19.14 µs      │ 23.82 µs      │ 19.21 µs      │ 19.32 µs      │ 100     │ 100
│                    1.711 Gitem/s │ 1.375 Gitem/s │ 1.704 Gitem/s │ 1.695 Gitem/s │         │
╰─ sub_i64_constant  8.129 µs      │ 12.18 µs      │ 8.255 µs      │ 8.34 µs       │ 100     │ 100
                     4.03 Gitem/s  │ 2.688 Gitem/s │ 3.969 Gitem/s │ 3.928 Gitem/s │         │


```
