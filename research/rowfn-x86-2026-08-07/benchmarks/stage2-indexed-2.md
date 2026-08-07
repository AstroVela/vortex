<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# `stage2-indexed-2`

```text
Timer precision: 10 ns
binary_ops           fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ add_i64_constant  9.009 µs      │ 93.1 µs       │ 9.259 µs      │ 10.14 µs      │ 100     │ 100
│                    3.636 Gitem/s │ 351.9 Mitem/s │ 3.538 Gitem/s │ 3.23 Gitem/s  │         │
├─ add_i64_nonnull   9.309 µs      │ 12.56 µs      │ 9.379 µs      │ 9.426 µs      │ 100     │ 100
│                    3.519 Gitem/s │ 2.606 Gitem/s │ 3.493 Gitem/s │ 3.476 Gitem/s │         │
├─ div_i64_nonnull   44.96 µs      │ 48.24 µs      │ 45.03 µs      │ 45.22 µs      │ 100     │ 100
│                    728.6 Mitem/s │ 679.1 Mitem/s │ 727.5 Mitem/s │ 724.5 Mitem/s │         │
├─ mul_i8_nonnull    5.969 µs      │ 50.06 µs      │ 6.359 µs      │ 6.902 µs      │ 100     │ 100
│                    5.488 Gitem/s │ 654.4 Mitem/s │ 5.152 Gitem/s │ 4.747 Gitem/s │         │
├─ mul_i16_nonnull   4.199 µs      │ 15.43 µs      │ 4.284 µs      │ 4.418 µs      │ 100     │ 100
│                    7.802 Gitem/s │ 2.122 Gitem/s │ 7.647 Gitem/s │ 7.415 Gitem/s │         │
├─ mul_i32_constant  32.25 µs      │ 35.51 µs      │ 32.39 µs      │ 32.51 µs      │ 100     │ 100
│                    1.015 Gitem/s │ 922.5 Mitem/s │ 1.011 Gitem/s │ 1.007 Gitem/s │         │
├─ mul_i32_nonnull   26.54 µs      │ 29.82 µs      │ 26.6 µs       │ 26.7 µs       │ 100     │ 100
│                    1.234 Gitem/s │ 1.098 Gitem/s │ 1.231 Gitem/s │ 1.226 Gitem/s │         │
├─ mul_i32_nullable  27.32 µs      │ 40.06 µs      │ 27.43 µs      │ 27.68 µs      │ 100     │ 100
│                    1.198 Gitem/s │ 817.7 Mitem/s │ 1.194 Gitem/s │ 1.183 Gitem/s │         │
├─ mul_i64_nonnull   23.33 µs      │ 26.7 µs       │ 23.44 µs      │ 23.53 µs      │ 100     │ 100
│                    1.403 Gitem/s │ 1.226 Gitem/s │ 1.397 Gitem/s │ 1.392 Gitem/s │         │
├─ mul_u8_nonnull    3.459 µs      │ 61.29 µs      │ 3.519 µs      │ 4.131 µs      │ 100     │ 100
│                    9.471 Gitem/s │ 534.5 Mitem/s │ 9.309 Gitem/s │ 7.931 Gitem/s │         │
├─ mul_u16_nonnull   2.709 µs      │ 6.939 µs      │ 2.789 µs      │ 2.828 µs      │ 100     │ 100
│                    12.09 Gitem/s │ 4.721 Gitem/s │ 11.74 Gitem/s │ 11.58 Gitem/s │         │
├─ mul_u32_nonnull   7.039 µs      │ 10.56 µs      │ 7.119 µs      │ 7.18 µs       │ 100     │ 100
│                    4.654 Gitem/s │ 3.1 Gitem/s   │ 4.602 Gitem/s │ 4.563 Gitem/s │         │
├─ mul_u64_nonnull   19.34 µs      │ 23.2 µs       │ 19.42 µs      │ 19.48 µs      │ 100     │ 100
│                    1.693 Gitem/s │ 1.411 Gitem/s │ 1.686 Gitem/s │ 1.681 Gitem/s │         │
╰─ sub_i64_constant  9.509 µs      │ 12.69 µs      │ 9.619 µs      │ 9.668 µs      │ 100     │ 100
                     3.445 Gitem/s │ 2.58 Gitem/s  │ 3.406 Gitem/s │ 3.389 Gitem/s │         │


```
