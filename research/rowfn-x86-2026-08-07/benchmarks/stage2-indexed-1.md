<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# `stage2-indexed-1`

```text
Timer precision: 10 ns
binary_ops           fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ add_i64_constant  8.889 µs      │ 89.9 µs       │ 9.244 µs      │ 10.11 µs      │ 100     │ 100
│                    3.686 Gitem/s │ 364.4 Mitem/s │ 3.544 Gitem/s │ 3.24 Gitem/s  │         │
├─ add_i64_nonnull   9.279 µs      │ 18.28 µs      │ 9.399 µs      │ 9.605 µs      │ 100     │ 100
│                    3.531 Gitem/s │ 1.791 Gitem/s │ 3.486 Gitem/s │ 3.411 Gitem/s │         │
├─ div_i64_nonnull   44.92 µs      │ 52.66 µs      │ 45.07 µs      │ 45.39 µs      │ 100     │ 100
│                    729.3 Mitem/s │ 622.1 Mitem/s │ 726.8 Mitem/s │ 721.9 Mitem/s │         │
├─ mul_i8_nonnull    6.069 µs      │ 69.59 µs      │ 6.339 µs      │ 7.085 µs      │ 100     │ 100
│                    5.398 Gitem/s │ 470.8 Mitem/s │ 5.168 Gitem/s │ 4.624 Gitem/s │         │
├─ mul_i16_nonnull   4.209 µs      │ 5.439 µs      │ 4.259 µs      │ 4.274 µs      │ 100     │ 100
│                    7.783 Gitem/s │ 6.023 Gitem/s │ 7.692 Gitem/s │ 7.665 Gitem/s │         │
├─ mul_i32_constant  32.23 µs      │ 36.63 µs      │ 32.38 µs      │ 32.54 µs      │ 100     │ 100
│                    1.016 Gitem/s │ 894.3 Mitem/s │ 1.011 Gitem/s │ 1.006 Gitem/s │         │
├─ mul_i32_nonnull   26.52 µs      │ 31.34 µs      │ 26.58 µs      │ 26.69 µs      │ 100     │ 100
│                    1.235 Gitem/s │ 1.045 Gitem/s │ 1.232 Gitem/s │ 1.227 Gitem/s │         │
├─ mul_i32_nullable  27.31 µs      │ 47.02 µs      │ 27.41 µs      │ 27.83 µs      │ 100     │ 100
│                    1.199 Gitem/s │ 696.7 Mitem/s │ 1.195 Gitem/s │ 1.177 Gitem/s │         │
├─ mul_i64_nonnull   23.33 µs      │ 32.13 µs      │ 23.43 µs      │ 23.74 µs      │ 100     │ 100
│                    1.403 Gitem/s │ 1.019 Gitem/s │ 1.397 Gitem/s │ 1.379 Gitem/s │         │
├─ mul_u8_nonnull    3.439 µs      │ 61.51 µs      │ 3.509 µs      │ 4.121 µs      │ 100     │ 100
│                    9.526 Gitem/s │ 532.6 Mitem/s │ 9.336 Gitem/s │ 7.951 Gitem/s │         │
├─ mul_u16_nonnull   2.699 µs      │ 6.979 µs      │ 2.769 µs      │ 2.813 µs      │ 100     │ 100
│                    12.13 Gitem/s │ 4.694 Gitem/s │ 11.83 Gitem/s │ 11.64 Gitem/s │         │
├─ mul_u32_nonnull   7.029 µs      │ 9.939 µs      │ 7.109 µs      │ 7.16 µs       │ 100     │ 100
│                    4.661 Gitem/s │ 3.296 Gitem/s │ 4.608 Gitem/s │ 4.576 Gitem/s │         │
├─ mul_u64_nonnull   19.35 µs      │ 22.92 µs      │ 19.41 µs      │ 19.51 µs      │ 100     │ 100
│                    1.692 Gitem/s │ 1.429 Gitem/s │ 1.687 Gitem/s │ 1.679 Gitem/s │         │
╰─ sub_i64_constant  9.499 µs      │ 12.48 µs      │ 9.609 µs      │ 9.668 µs      │ 100     │ 100
                     3.449 Gitem/s │ 2.623 Gitem/s │ 3.409 Gitem/s │ 3.389 Gitem/s │         │


```
