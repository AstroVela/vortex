<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# `stage2-candidate-2`

```text
Timer precision: 10 ns
binary_ops           fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ add_i64_constant  9.159 µs      │ 57.71 µs      │ 9.269 µs      │ 9.826 µs      │ 100     │ 100
│                    3.577 Gitem/s │ 567.7 Mitem/s │ 3.534 Gitem/s │ 3.334 Gitem/s │         │
├─ add_i64_nonnull   9.359 µs      │ 18.58 µs      │ 9.454 µs      │ 9.765 µs      │ 100     │ 100
│                    3.5 Gitem/s   │ 1.762 Gitem/s │ 3.465 Gitem/s │ 3.355 Gitem/s │         │
├─ div_i64_nonnull   45.03 µs      │ 49.11 µs      │ 45.12 µs      │ 45.32 µs      │ 100     │ 100
│                    727.5 Mitem/s │ 667.1 Mitem/s │ 726 Mitem/s   │ 722.9 Mitem/s │         │
├─ mul_i8_nonnull    4.649 µs      │ 8.699 µs      │ 4.729 µs      │ 4.798 µs      │ 100     │ 100
│                    7.047 Gitem/s │ 3.766 Gitem/s │ 6.928 Gitem/s │ 6.828 Gitem/s │         │
├─ mul_i16_nonnull   4.219 µs      │ 7.469 µs      │ 4.309 µs      │ 4.374 µs      │ 100     │ 100
│                    7.765 Gitem/s │ 4.386 Gitem/s │ 7.603 Gitem/s │ 7.491 Gitem/s │         │
├─ mul_i32_constant  18.75 µs      │ 22.67 µs      │ 18.88 µs      │ 18.95 µs      │ 100     │ 100
│                    1.746 Gitem/s │ 1.444 Gitem/s │ 1.734 Gitem/s │ 1.728 Gitem/s │         │
├─ mul_i32_nonnull   28.2 µs       │ 40.32 µs      │ 28.36 µs      │ 28.69 µs      │ 100     │ 100
│                    1.161 Gitem/s │ 812.5 Mitem/s │ 1.155 Gitem/s │ 1.141 Gitem/s │         │
├─ mul_i32_nullable  29.02 µs      │ 40.92 µs      │ 29.17 µs      │ 29.4 µs       │ 100     │ 100
│                    1.128 Gitem/s │ 800.5 Mitem/s │ 1.123 Gitem/s │ 1.114 Gitem/s │         │
├─ mul_i64_nonnull   29.63 µs      │ 33.89 µs      │ 30.1 µs       │ 30.22 µs      │ 100     │ 100
│                    1.105 Gitem/s │ 966.6 Mitem/s │ 1.088 Gitem/s │ 1.084 Gitem/s │         │
├─ mul_u8_nonnull    3.489 µs      │ 4.839 µs      │ 3.559 µs      │ 3.576 µs      │ 100     │ 100
│                    9.389 Gitem/s │ 6.77 Gitem/s  │ 9.205 Gitem/s │ 9.163 Gitem/s │         │
├─ mul_u16_nonnull   2.379 µs      │ 5.529 µs      │ 2.439 µs      │ 2.489 µs      │ 100     │ 100
│                    13.76 Gitem/s │ 5.925 Gitem/s │ 13.43 Gitem/s │ 13.16 Gitem/s │         │
├─ mul_u32_nonnull   6.989 µs      │ 8.019 µs      │ 7.079 µs      │ 7.089 µs      │ 100     │ 100
│                    4.687 Gitem/s │ 4.085 Gitem/s │ 4.628 Gitem/s │ 4.621 Gitem/s │         │
├─ mul_u64_nonnull   30.35 µs      │ 34.77 µs      │ 30.43 µs      │ 30.57 µs      │ 100     │ 100
│                    1.079 Gitem/s │ 942.1 Mitem/s │ 1.076 Gitem/s │ 1.071 Gitem/s │         │
╰─ sub_i64_constant  8.949 µs      │ 10.5 µs       │ 9.089 µs      │ 9.109 µs      │ 100     │ 100
                     3.661 Gitem/s │ 3.117 Gitem/s │ 3.604 Gitem/s │ 3.597 Gitem/s │         │


```
