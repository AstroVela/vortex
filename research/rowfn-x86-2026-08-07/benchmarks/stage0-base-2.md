<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# `stage0-base-2`

```text
binary_ops           fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ add_i64_constant  8.089 µs      │ 63.25 µs      │ 8.399 µs      │ 8.966 µs      │ 100     │ 100
│                    4.05 Gitem/s  │ 518 Mitem/s   │ 3.901 Gitem/s │ 3.654 Gitem/s │         │
├─ add_i64_nonnull   9.069 µs      │ 13.16 µs      │ 9.149 µs      │ 9.232 µs      │ 100     │ 100
│                    3.612 Gitem/s │ 2.488 Gitem/s │ 3.581 Gitem/s │ 3.549 Gitem/s │         │
├─ div_i64_nonnull   44.73 µs      │ 51.02 µs      │ 44.8 µs       │ 45.03 µs      │ 100     │ 100
│                    732.4 Mitem/s │ 642.1 Mitem/s │ 731.2 Mitem/s │ 727.6 Mitem/s │         │
├─ mul_i8_nonnull    5.979 µs      │ 11.05 µs      │ 6.199 µs      │ 6.323 µs      │ 100     │ 100
│                    5.479 Gitem/s │ 2.962 Gitem/s │ 5.285 Gitem/s │ 5.182 Gitem/s │         │
├─ mul_i16_nonnull   4.049 µs      │ 8.709 µs      │ 4.119 µs      │ 4.205 µs      │ 100     │ 100
│                    8.091 Gitem/s │ 3.762 Gitem/s │ 7.953 Gitem/s │ 7.791 Gitem/s │         │
├─ mul_i32_constant  26.36 µs      │ 29.65 µs      │ 26.43 µs      │ 26.51 µs      │ 100     │ 100
│                    1.242 Gitem/s │ 1.104 Gitem/s │ 1.239 Gitem/s │ 1.235 Gitem/s │         │
├─ mul_i32_nonnull   26.37 µs      │ 36.95 µs      │ 26.42 µs      │ 26.61 µs      │ 100     │ 100
│                    1.242 Gitem/s │ 886.8 Mitem/s │ 1.239 Gitem/s │ 1.231 Gitem/s │         │
├─ mul_i32_nullable  27.3 µs       │ 49.11 µs      │ 27.4 µs       │ 27.76 µs      │ 100     │ 100
│                    1.199 Gitem/s │ 667.1 Mitem/s │ 1.195 Gitem/s │ 1.18 Gitem/s  │         │
├─ mul_i64_nonnull   23.11 µs      │ 27.98 µs      │ 23.2 µs       │ 23.32 µs      │ 100     │ 100
│                    1.417 Gitem/s │ 1.171 Gitem/s │ 1.411 Gitem/s │ 1.404 Gitem/s │         │
├─ mul_u8_nonnull    3.259 µs      │ 4.809 µs      │ 3.329 µs      │ 3.345 µs      │ 100     │ 100
│                    10.05 Gitem/s │ 6.812 Gitem/s │ 9.84 Gitem/s  │ 9.794 Gitem/s │         │
├─ mul_u16_nonnull   2.549 µs      │ 3.649 µs      │ 2.599 µs      │ 2.611 µs      │ 100     │ 100
│                    12.85 Gitem/s │ 8.978 Gitem/s │ 12.6 Gitem/s  │ 12.54 Gitem/s │         │
├─ mul_u32_nonnull   6.889 µs      │ 10.15 µs      │ 6.949 µs      │ 6.999 µs      │ 100     │ 100
│                    4.756 Gitem/s │ 3.228 Gitem/s │ 4.714 Gitem/s │ 4.681 Gitem/s │         │
├─ mul_u64_nonnull   19.12 µs      │ 24.11 µs      │ 19.19 µs      │ 19.3 µs       │ 100     │ 100
│                    1.712 Gitem/s │ 1.358 Gitem/s │ 1.706 Gitem/s │ 1.697 Gitem/s │         │
╰─ sub_i64_constant  8.119 µs      │ 12.58 µs      │ 8.239 µs      │ 8.323 µs      │ 100     │ 100
                     4.035 Gitem/s │ 2.602 Gitem/s │ 3.976 Gitem/s │ 3.936 Gitem/s │         │


```
