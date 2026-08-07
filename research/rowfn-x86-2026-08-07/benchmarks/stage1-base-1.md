<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# `stage1-base-1`

```text
binary_ops           fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ add_i64_constant  8.089 µs      │ 783.5 µs      │ 8.419 µs      │ 16.21 µs      │ 100     │ 100
│                    4.05 Gitem/s  │ 41.81 Mitem/s │ 3.891 Gitem/s │ 2.02 Gitem/s  │         │
├─ add_i64_nonnull   9.089 µs      │ 32.99 µs      │ 9.189 µs      │ 9.53 µs       │ 100     │ 100
│                    3.604 Gitem/s │ 992.9 Mitem/s │ 3.565 Gitem/s │ 3.438 Gitem/s │         │
├─ div_i64_nonnull   44.76 µs      │ 76.23 µs      │ 44.84 µs      │ 45.51 µs      │ 100     │ 100
│                    731.9 Mitem/s │ 429.8 Mitem/s │ 730.6 Mitem/s │ 719.8 Mitem/s │         │
├─ mul_i8_nonnull    5.929 µs      │ 72.34 µs      │ 6.239 µs      │ 7.053 µs      │ 100     │ 100
│                    5.526 Gitem/s │ 452.9 Mitem/s │ 5.251 Gitem/s │ 4.645 Gitem/s │         │
├─ mul_i16_nonnull   4.049 µs      │ 61.69 µs      │ 4.114 µs      │ 4.692 µs      │ 100     │ 100
│                    8.091 Gitem/s │ 531.1 Mitem/s │ 7.963 Gitem/s │ 6.983 Gitem/s │         │
├─ mul_i32_constant  26.36 µs      │ 55.98 µs      │ 26.43 µs      │ 26.85 µs      │ 100     │ 100
│                    1.242 Gitem/s │ 585.2 Mitem/s │ 1.239 Gitem/s │ 1.219 Gitem/s │         │
├─ mul_i32_nonnull   26.38 µs      │ 37.56 µs      │ 26.42 µs      │ 26.65 µs      │ 100     │ 100
│                    1.241 Gitem/s │ 872.3 Mitem/s │ 1.239 Gitem/s │ 1.229 Gitem/s │         │
├─ mul_i32_nullable  27.23 µs      │ 340.3 µs      │ 27.36 µs      │ 30.62 µs      │ 100     │ 100
│                    1.202 Gitem/s │ 96.26 Mitem/s │ 1.197 Gitem/s │ 1.07 Gitem/s  │         │
├─ mul_i64_nonnull   23.11 µs      │ 44.46 µs      │ 23.2 µs       │ 23.5 µs       │ 100     │ 100
│                    1.417 Gitem/s │ 736.8 Mitem/s │ 1.411 Gitem/s │ 1.394 Gitem/s │         │
├─ mul_u8_nonnull    3.269 µs      │ 51.96 µs      │ 3.329 µs      │ 3.829 µs      │ 100     │ 100
│                    10.02 Gitem/s │ 630.6 Mitem/s │ 9.84 Gitem/s  │ 8.556 Gitem/s │         │
├─ mul_u16_nonnull   2.559 µs      │ 30.97 µs      │ 2.609 µs      │ 2.898 µs      │ 100     │ 100
│                    12.8 Gitem/s  │ 1.057 Gitem/s │ 12.55 Gitem/s │ 11.3 Gitem/s  │         │
├─ mul_u32_nonnull   6.88 µs       │ 26.3 µs       │ 6.959 µs      │ 7.202 µs      │ 100     │ 100
│                    4.762 Gitem/s │ 1.245 Gitem/s │ 4.708 Gitem/s │ 4.549 Gitem/s │         │
├─ mul_u64_nonnull   19.11 µs      │ 40.79 µs      │ 19.18 µs      │ 19.48 µs      │ 100     │ 100
│                    1.713 Gitem/s │ 803.3 Mitem/s │ 1.707 Gitem/s │ 1.681 Gitem/s │         │
╰─ sub_i64_constant  8.109 µs      │ 41.21 µs      │ 8.219 µs      │ 8.589 µs      │ 100     │ 100
                     4.04 Gitem/s  │ 794.9 Mitem/s │ 3.986 Gitem/s │ 3.814 Gitem/s │         │


```
