<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# `land-candidate-1`

```text
Timer precision: 10 ns
binary_ops           fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ add_i64_constant  8.989 µs      │ 1.016 ms      │ 9.31 µs       │ 19.44 µs      │ 100     │ 100
│                    3.645 Gitem/s │ 32.25 Mitem/s │ 3.519 Gitem/s │ 1.684 Gitem/s │         │
├─ add_i64_nonnull   9.279 µs      │ 13.06 µs      │ 9.374 µs      │ 9.443 µs      │ 100     │ 100
│                    3.531 Gitem/s │ 2.508 Gitem/s │ 3.495 Gitem/s │ 3.469 Gitem/s │         │
├─ div_i64_nonnull   44.97 µs      │ 63.03 µs      │ 45.04 µs      │ 45.41 µs      │ 100     │ 100
│                    728.5 Mitem/s │ 519.7 Mitem/s │ 727.3 Mitem/s │ 721.5 Mitem/s │         │
├─ mul_i8_nonnull    4.649 µs      │ 60.73 µs      │ 4.719 µs      │ 5.455 µs      │ 100     │ 100
│                    7.047 Gitem/s │ 539.4 Mitem/s │ 6.942 Gitem/s │ 6.006 Gitem/s │         │
├─ mul_i16_nonnull   4.219 µs      │ 71.24 µs      │ 4.269 µs      │ 4.959 µs      │ 100     │ 100
│                    7.765 Gitem/s │ 459.9 Mitem/s │ 7.674 Gitem/s │ 6.607 Gitem/s │         │
├─ mul_i32_constant  18.79 µs      │ 72.37 µs      │ 18.89 µs      │ 19.53 µs      │ 100     │ 100
│                    1.742 Gitem/s │ 452.7 Mitem/s │ 1.734 Gitem/s │ 1.677 Gitem/s │         │
├─ mul_i32_nonnull   28.24 µs      │ 33.43 µs      │ 28.35 µs      │ 28.48 µs      │ 100     │ 100
│                    1.159 Gitem/s │ 980.1 Mitem/s │ 1.155 Gitem/s │ 1.15 Gitem/s  │         │
├─ mul_i32_nullable  29.01 µs      │ 234.9 µs      │ 29.17 µs      │ 31.37 µs      │ 100     │ 100
│                    1.129 Gitem/s │ 139.4 Mitem/s │ 1.122 Gitem/s │ 1.044 Gitem/s │         │
├─ mul_i64_nonnull   29.63 µs      │ 52.67 µs      │ 30.01 µs      │ 30.5 µs       │ 100     │ 100
│                    1.105 Gitem/s │ 622 Mitem/s   │ 1.091 Gitem/s │ 1.074 Gitem/s │         │
├─ mul_u8_nonnull    3.459 µs      │ 14.69 µs      │ 3.545 µs      │ 3.659 µs      │ 100     │ 100
│                    9.471 Gitem/s │ 2.229 Gitem/s │ 9.242 Gitem/s │ 8.953 Gitem/s │         │
├─ mul_u16_nonnull   2.369 µs      │ 13.01 µs      │ 2.429 µs      │ 2.615 µs      │ 100     │ 100
│                    13.82 Gitem/s │ 2.516 Gitem/s │ 13.48 Gitem/s │ 12.52 Gitem/s │         │
├─ mul_u32_nonnull   6.999 µs      │ 18.45 µs      │ 7.06 µs       │ 7.181 µs      │ 100     │ 100
│                    4.681 Gitem/s │ 1.775 Gitem/s │ 4.641 Gitem/s │ 4.562 Gitem/s │         │
├─ mul_u64_nonnull   30.27 µs      │ 43.99 µs      │ 30.37 µs      │ 30.65 µs      │ 100     │ 100
│                    1.082 Gitem/s │ 744.8 Mitem/s │ 1.078 Gitem/s │ 1.068 Gitem/s │         │
╰─ sub_i64_constant  8.959 µs      │ 31.53 µs      │ 9.114 µs      │ 9.385 µs      │ 100     │ 100
                     3.657 Gitem/s │ 1.038 Gitem/s │ 3.595 Gitem/s │ 3.491 Gitem/s │         │


```
