<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# `land-candidate-2`

```text
Timer precision: 10 ns
binary_ops           fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ add_i64_constant  9.129 µs      │ 48.92 µs      │ 9.289 µs      │ 9.744 µs      │ 100     │ 100
│                    3.589 Gitem/s │ 669.8 Mitem/s │ 3.527 Gitem/s │ 3.362 Gitem/s │         │
├─ add_i64_nonnull   9.349 µs      │ 10.43 µs      │ 9.449 µs      │ 9.46 µs       │ 100     │ 100
│                    3.504 Gitem/s │ 3.141 Gitem/s │ 3.467 Gitem/s │ 3.463 Gitem/s │         │
├─ div_i64_nonnull   44.99 µs      │ 50.41 µs      │ 45.08 µs      │ 45.29 µs      │ 100     │ 100
│                    728.1 Mitem/s │ 650 Mitem/s   │ 726.8 Mitem/s │ 723.5 Mitem/s │         │
├─ mul_i8_nonnull    4.639 µs      │ 7.969 µs      │ 4.699 µs      │ 4.761 µs      │ 100     │ 100
│                    7.062 Gitem/s │ 4.111 Gitem/s │ 6.972 Gitem/s │ 6.881 Gitem/s │         │
├─ mul_i16_nonnull   4.209 µs      │ 7.669 µs      │ 4.269 µs      │ 4.308 µs      │ 100     │ 100
│                    7.783 Gitem/s │ 4.272 Gitem/s │ 7.674 Gitem/s │ 7.605 Gitem/s │         │
├─ mul_i32_constant  18.75 µs      │ 22.77 µs      │ 18.84 µs      │ 18.95 µs      │ 100     │ 100
│                    1.746 Gitem/s │ 1.439 Gitem/s │ 1.738 Gitem/s │ 1.728 Gitem/s │         │
├─ mul_i32_nonnull   28.27 µs      │ 45.57 µs      │ 28.39 µs      │ 28.65 µs      │ 100     │ 100
│                    1.158 Gitem/s │ 718.9 Mitem/s │ 1.154 Gitem/s │ 1.143 Gitem/s │         │
├─ mul_i32_nullable  29.04 µs      │ 43.06 µs      │ 29.17 µs      │ 29.41 µs      │ 100     │ 100
│                    1.128 Gitem/s │ 760.8 Mitem/s │ 1.122 Gitem/s │ 1.113 Gitem/s │         │
├─ mul_i64_nonnull   29.7 µs       │ 35.65 µs      │ 30.05 µs      │ 30.18 µs      │ 100     │ 100
│                    1.103 Gitem/s │ 918.9 Mitem/s │ 1.09 Gitem/s  │ 1.085 Gitem/s │         │
├─ mul_u8_nonnull    3.459 µs      │ 4.579 µs      │ 3.519 µs      │ 3.532 µs      │ 100     │ 100
│                    9.471 Gitem/s │ 7.154 Gitem/s │ 9.309 Gitem/s │ 9.275 Gitem/s │         │
├─ mul_u16_nonnull   2.359 µs      │ 3.269 µs      │ 2.429 µs      │ 2.441 µs      │ 100     │ 100
│                    13.88 Gitem/s │ 10.02 Gitem/s │ 13.48 Gitem/s │ 13.42 Gitem/s │         │
├─ mul_u32_nonnull   6.999 µs      │ 11.21 µs      │ 7.059 µs      │ 7.105 µs      │ 100     │ 100
│                    4.681 Gitem/s │ 2.92 Gitem/s  │ 4.641 Gitem/s │ 4.611 Gitem/s │         │
├─ mul_u64_nonnull   30.33 µs      │ 34.65 µs      │ 30.4 µs       │ 30.53 µs      │ 100     │ 100
│                    1.08 Gitem/s  │ 945.4 Mitem/s │ 1.077 Gitem/s │ 1.073 Gitem/s │         │
╰─ sub_i64_constant  8.949 µs      │ 12.59 µs      │ 9.079 µs      │ 9.155 µs      │ 100     │ 100
                     3.661 Gitem/s │ 2.6 Gitem/s   │ 3.608 Gitem/s │ 3.578 Gitem/s │         │


```
