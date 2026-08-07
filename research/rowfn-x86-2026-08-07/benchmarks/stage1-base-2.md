<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# `stage1-base-2`

```text
binary_ops           fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ add_i64_constant  8.309 µs      │ 51.73 µs      │ 8.459 µs      │ 8.93 µs       │ 100     │ 100
│                    3.943 Gitem/s │ 633.3 Mitem/s │ 3.873 Gitem/s │ 3.669 Gitem/s │         │
├─ add_i64_nonnull   9.119 µs      │ 19.9 µs       │ 9.199 µs      │ 9.345 µs      │ 100     │ 100
│                    3.593 Gitem/s │ 1.646 Gitem/s │ 3.561 Gitem/s │ 3.506 Gitem/s │         │
├─ div_i64_nonnull   44.77 µs      │ 49.58 µs      │ 44.85 µs      │ 45.06 µs      │ 100     │ 100
│                    731.7 Mitem/s │ 660.7 Mitem/s │ 730.5 Mitem/s │ 727.1 Mitem/s │         │
├─ mul_i8_nonnull    5.779 µs      │ 10.19 µs      │ 6.15 µs       │ 6.267 µs      │ 100     │ 100
│                    5.669 Gitem/s │ 3.212 Gitem/s │ 5.327 Gitem/s │ 5.228 Gitem/s │         │
├─ mul_i16_nonnull   4.059 µs      │ 8.189 µs      │ 4.109 µs      │ 4.198 µs      │ 100     │ 100
│                    8.071 Gitem/s │ 4.001 Gitem/s │ 7.973 Gitem/s │ 7.804 Gitem/s │         │
├─ mul_i32_constant  26.37 µs      │ 30.04 µs      │ 26.44 µs      │ 26.54 µs      │ 100     │ 100
│                    1.242 Gitem/s │ 1.09 Gitem/s  │ 1.238 Gitem/s │ 1.234 Gitem/s │         │
├─ mul_i32_nonnull   26.36 µs      │ 29.99 µs      │ 26.41 µs      │ 26.54 µs      │ 100     │ 100
│                    1.242 Gitem/s │ 1.092 Gitem/s │ 1.24 Gitem/s  │ 1.234 Gitem/s │         │
├─ mul_i32_nullable  27.23 µs      │ 42.3 µs       │ 27.35 µs      │ 27.61 µs      │ 100     │ 100
│                    1.202 Gitem/s │ 774.4 Mitem/s │ 1.197 Gitem/s │ 1.186 Gitem/s │         │
├─ mul_i64_nonnull   23.14 µs      │ 27.63 µs      │ 23.26 µs      │ 23.41 µs      │ 100     │ 100
│                    1.415 Gitem/s │ 1.185 Gitem/s │ 1.408 Gitem/s │ 1.399 Gitem/s │         │
├─ mul_u8_nonnull    3.269 µs      │ 4.809 µs      │ 3.319 µs      │ 3.339 µs      │ 100     │ 100
│                    10.02 Gitem/s │ 6.812 Gitem/s │ 9.87 Gitem/s  │ 9.812 Gitem/s │         │
├─ mul_u16_nonnull   2.549 µs      │ 3.519 µs      │ 2.609 µs      │ 2.618 µs      │ 100     │ 100
│                    12.85 Gitem/s │ 9.309 Gitem/s │ 12.55 Gitem/s │ 12.51 Gitem/s │         │
├─ mul_u32_nonnull   6.879 µs      │ 11.37 µs      │ 6.95 µs       │ 7.026 µs      │ 100     │ 100
│                    4.762 Gitem/s │ 2.879 Gitem/s │ 4.714 Gitem/s │ 4.663 Gitem/s │         │
├─ mul_u64_nonnull   19.16 µs      │ 22.33 µs      │ 19.21 µs      │ 19.3 µs       │ 100     │ 100
│                    1.709 Gitem/s │ 1.466 Gitem/s │ 1.705 Gitem/s │ 1.697 Gitem/s │         │
╰─ sub_i64_constant  8.139 µs      │ 11.6 µs       │ 8.259 µs      │ 8.319 µs      │ 100     │ 100
                     4.025 Gitem/s │ 2.822 Gitem/s │ 3.967 Gitem/s │ 3.938 Gitem/s │         │


```
