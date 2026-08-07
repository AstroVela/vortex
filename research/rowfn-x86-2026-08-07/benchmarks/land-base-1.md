<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# `land-base-1`

```text
Timer precision: 10 ns
binary_ops           fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ add_i64_constant  8.059 µs      │ 781.5 µs      │ 8.399 µs      │ 16.16 µs      │ 100     │ 100
│                    4.065 Gitem/s │ 41.92 Mitem/s │ 3.901 Gitem/s │ 2.027 Gitem/s │         │
├─ add_i64_nonnull   9.079 µs      │ 29.55 µs      │ 9.159 µs      │ 9.466 µs      │ 100     │ 100
│                    3.608 Gitem/s │ 1.108 Gitem/s │ 3.577 Gitem/s │ 3.461 Gitem/s │         │
├─ div_i64_nonnull   44.73 µs      │ 78.44 µs      │ 44.82 µs      │ 45.35 µs      │ 100     │ 100
│                    732.4 Mitem/s │ 417.6 Mitem/s │ 731 Mitem/s   │ 722.5 Mitem/s │         │
├─ mul_i8_nonnull    5.819 µs      │ 72.73 µs      │ 6.209 µs      │ 6.939 µs      │ 100     │ 100
│                    5.63 Gitem/s  │ 450.5 Mitem/s │ 5.276 Gitem/s │ 4.721 Gitem/s │         │
├─ mul_i16_nonnull   4.039 µs      │ 66.44 µs      │ 4.099 µs      │ 4.731 µs      │ 100     │ 100
│                    8.111 Gitem/s │ 493.1 Mitem/s │ 7.992 Gitem/s │ 6.926 Gitem/s │         │
├─ mul_i32_constant  26.37 µs      │ 56.27 µs      │ 26.44 µs      │ 26.97 µs      │ 100     │ 100
│                    1.242 Gitem/s │ 582.2 Mitem/s │ 1.238 Gitem/s │ 1.214 Gitem/s │         │
├─ mul_i32_nonnull   26.35 µs      │ 38.26 µs      │ 26.41 µs      │ 26.64 µs      │ 100     │ 100
│                    1.243 Gitem/s │ 856.4 Mitem/s │ 1.24 Gitem/s  │ 1.229 Gitem/s │         │
├─ mul_i32_nullable  27.28 µs      │ 340.1 µs      │ 27.38 µs      │ 30.62 µs      │ 100     │ 100
│                    1.2 Gitem/s   │ 96.34 Mitem/s │ 1.196 Gitem/s │ 1.069 Gitem/s │         │
├─ mul_i64_nonnull   23.11 µs      │ 45.96 µs      │ 23.2 µs       │ 23.55 µs      │ 100     │ 100
│                    1.417 Gitem/s │ 712.9 Mitem/s │ 1.411 Gitem/s │ 1.391 Gitem/s │         │
├─ mul_u8_nonnull    3.259 µs      │ 52.07 µs      │ 3.319 µs      │ 3.838 µs      │ 100     │ 100
│                    10.05 Gitem/s │ 629.1 Mitem/s │ 9.87 Gitem/s  │ 8.535 Gitem/s │         │
├─ mul_u16_nonnull   2.539 µs      │ 30.81 µs      │ 2.609 µs      │ 2.888 µs      │ 100     │ 100
│                    12.9 Gitem/s  │ 1.063 Gitem/s │ 12.55 Gitem/s │ 11.34 Gitem/s │         │
├─ mul_u32_nonnull   6.879 µs      │ 26.44 µs      │ 6.949 µs      │ 7.183 µs      │ 100     │ 100
│                    4.762 Gitem/s │ 1.238 Gitem/s │ 4.714 Gitem/s │ 4.561 Gitem/s │         │
├─ mul_u64_nonnull   19.11 µs      │ 41.4 µs       │ 19.18 µs      │ 19.53 µs      │ 100     │ 100
│                    1.713 Gitem/s │ 791.4 Mitem/s │ 1.707 Gitem/s │ 1.677 Gitem/s │         │
╰─ sub_i64_constant  8.109 µs      │ 40.38 µs      │ 8.239 µs      │ 8.604 µs      │ 100     │ 100
                     4.04 Gitem/s  │ 811.2 Mitem/s │ 3.976 Gitem/s │ 3.808 Gitem/s │         │


```
