<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# `stage1-candidate-2`

```text
binary_ops           fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ add_i64_constant  9.149 µs      │ 65.44 µs      │ 9.309 µs      │ 9.916 µs      │ 100     │ 100
│                    3.581 Gitem/s │ 500.6 Mitem/s │ 3.519 Gitem/s │ 3.304 Gitem/s │         │
├─ add_i64_nonnull   9.429 µs      │ 18.82 µs      │ 9.529 µs      │ 9.64 µs       │ 100     │ 100
│                    3.474 Gitem/s │ 1.74 Gitem/s  │ 3.438 Gitem/s │ 3.399 Gitem/s │         │
├─ div_i64_nonnull   45.09 µs      │ 51.42 µs      │ 45.16 µs      │ 45.37 µs      │ 100     │ 100
│                    726.5 Mitem/s │ 637.2 Mitem/s │ 725.4 Mitem/s │ 722.1 Mitem/s │         │
├─ mul_i8_nonnull    4.669 µs      │ 7.159 µs      │ 4.729 µs      │ 4.781 µs      │ 100     │ 100
│                    7.017 Gitem/s │ 4.576 Gitem/s │ 6.928 Gitem/s │ 6.853 Gitem/s │         │
├─ mul_i16_nonnull   4.249 µs      │ 8.269 µs      │ 4.319 µs      │ 4.391 µs      │ 100     │ 100
│                    7.71 Gitem/s  │ 3.962 Gitem/s │ 7.585 Gitem/s │ 7.461 Gitem/s │         │
├─ mul_i32_constant  18.75 µs      │ 22.35 µs      │ 18.85 µs      │ 18.93 µs      │ 100     │ 100
│                    1.746 Gitem/s │ 1.465 Gitem/s │ 1.737 Gitem/s │ 1.73 Gitem/s  │         │
├─ mul_i32_nonnull   28.25 µs      │ 32.31 µs      │ 28.39 µs      │ 28.46 µs      │ 100     │ 100
│                    1.159 Gitem/s │ 1.013 Gitem/s │ 1.153 Gitem/s │ 1.151 Gitem/s │         │
├─ mul_i32_nullable  29.04 µs      │ 38.59 µs      │ 29.18 µs      │ 29.37 µs      │ 100     │ 100
│                    1.127 Gitem/s │ 848.9 Mitem/s │ 1.122 Gitem/s │ 1.115 Gitem/s │         │
├─ mul_i64_nonnull   29.84 µs      │ 33.8 µs       │ 30.16 µs      │ 30.24 µs      │ 100     │ 100
│                    1.097 Gitem/s │ 969.1 Mitem/s │ 1.086 Gitem/s │ 1.083 Gitem/s │         │
├─ mul_u8_nonnull    3.509 µs      │ 6.339 µs      │ 3.579 µs      │ 3.604 µs      │ 100     │ 100
│                    9.336 Gitem/s │ 5.168 Gitem/s │ 9.153 Gitem/s │ 9.091 Gitem/s │         │
├─ mul_u16_nonnull   2.389 µs      │ 38.12 µs      │ 2.474 µs      │ 2.857 µs      │ 100     │ 100
│                    13.71 Gitem/s │ 859.3 Mitem/s │ 13.24 Gitem/s │ 11.46 Gitem/s │         │
├─ mul_u32_nonnull   7.019 µs      │ 8.19 µs       │ 7.109 µs      │ 7.121 µs      │ 100     │ 100
│                    4.667 Gitem/s │ 4 Gitem/s     │ 4.608 Gitem/s │ 4.601 Gitem/s │         │
├─ mul_u64_nonnull   30.4 µs       │ 33.46 µs      │ 30.49 µs      │ 30.63 µs      │ 100     │ 100
│                    1.077 Gitem/s │ 979 Mitem/s   │ 1.074 Gitem/s │ 1.069 Gitem/s │         │
╰─ sub_i64_constant  9.009 µs      │ 10.35 µs      │ 9.119 µs      │ 9.142 µs      │ 100     │ 100
                     3.636 Gitem/s │ 3.163 Gitem/s │ 3.593 Gitem/s │ 3.584 Gitem/s │         │


```
