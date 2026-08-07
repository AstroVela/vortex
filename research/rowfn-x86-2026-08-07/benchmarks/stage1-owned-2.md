<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# `stage1-owned-2`

```text
binary_ops           fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ add_i64_constant  8.919 µs      │ 99.93 µs      │ 9.229 µs      │ 10.19 µs      │ 100     │ 100
│                    3.673 Gitem/s │ 327.8 Mitem/s │ 3.55 Gitem/s  │ 3.214 Gitem/s │         │
├─ add_i64_nonnull   9.279 µs      │ 12.53 µs      │ 9.339 µs      │ 9.417 µs      │ 100     │ 100
│                    3.531 Gitem/s │ 2.613 Gitem/s │ 3.508 Gitem/s │ 3.479 Gitem/s │         │
├─ div_i64_nonnull   44.91 µs      │ 54.36 µs      │ 45.01 µs      │ 45.32 µs      │ 100     │ 100
│                    729.4 Mitem/s │ 602.6 Mitem/s │ 727.8 Mitem/s │ 722.9 Mitem/s │         │
├─ mul_i8_nonnull    4.579 µs      │ 61.3 µs       │ 4.649 µs      │ 5.229 µs      │ 100     │ 100
│                    7.154 Gitem/s │ 534.5 Mitem/s │ 7.047 Gitem/s │ 6.265 Gitem/s │         │
├─ mul_i16_nonnull   4.169 µs      │ 7.419 µs      │ 4.229 µs      │ 4.276 µs      │ 100     │ 100
│                    7.858 Gitem/s │ 4.416 Gitem/s │ 7.746 Gitem/s │ 7.661 Gitem/s │         │
├─ mul_i32_constant  32.27 µs      │ 35.87 µs      │ 32.38 µs      │ 32.49 µs      │ 100     │ 100
│                    1.015 Gitem/s │ 913.5 Mitem/s │ 1.011 Gitem/s │ 1.008 Gitem/s │         │
├─ mul_i32_nonnull   27.77 µs      │ 32.39 µs      │ 31.23 µs      │ 30.31 µs      │ 100     │ 100
│                    1.179 Gitem/s │ 1.011 Gitem/s │ 1.048 Gitem/s │ 1.08 Gitem/s  │         │
├─ mul_i32_nullable  28.53 µs      │ 49.46 µs      │ 32.04 µs      │ 31.34 µs      │ 100     │ 100
│                    1.148 Gitem/s │ 662.3 Mitem/s │ 1.022 Gitem/s │ 1.045 Gitem/s │         │
├─ mul_i64_nonnull   25.25 µs      │ 29.06 µs      │ 25.59 µs      │ 25.68 µs      │ 100     │ 100
│                    1.297 Gitem/s │ 1.127 Gitem/s │ 1.28 Gitem/s  │ 1.275 Gitem/s │         │
├─ mul_u8_nonnull    3.429 µs      │ 60.44 µs      │ 3.479 µs      │ 4.062 µs      │ 100     │ 100
│                    9.553 Gitem/s │ 542 Mitem/s   │ 9.416 Gitem/s │ 8.065 Gitem/s │         │
├─ mul_u16_nonnull   2.319 µs      │ 7.879 µs      │ 2.389 µs      │ 2.446 µs      │ 100     │ 100
│                    14.12 Gitem/s │ 4.158 Gitem/s │ 13.71 Gitem/s │ 13.39 Gitem/s │         │
├─ mul_u32_nonnull   6.949 µs      │ 10.35 µs      │ 7.019 µs      │ 7.069 µs      │ 100     │ 100
│                    4.714 Gitem/s │ 3.163 Gitem/s │ 4.667 Gitem/s │ 4.635 Gitem/s │         │
├─ mul_u64_nonnull   19.34 µs      │ 22.75 µs      │ 19.41 µs      │ 19.49 µs      │ 100     │ 100
│                    1.693 Gitem/s │ 1.44 Gitem/s  │ 1.687 Gitem/s │ 1.68 Gitem/s  │         │
╰─ sub_i64_constant  9.439 µs      │ 12.1 µs       │ 9.569 µs      │ 9.636 µs      │ 100     │ 100
                     3.471 Gitem/s │ 2.705 Gitem/s │ 3.424 Gitem/s │ 3.4 Gitem/s   │         │


```
