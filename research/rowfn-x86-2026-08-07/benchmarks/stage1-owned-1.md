<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# `stage1-owned-1`

```text
binary_ops           fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ add_i64_constant  8.869 µs      │ 97.56 µs      │ 9.224 µs      │ 10.16 µs      │ 100     │ 100
│                    3.694 Gitem/s │ 335.8 Mitem/s │ 3.552 Gitem/s │ 3.225 Gitem/s │         │
├─ add_i64_nonnull   9.299 µs      │ 13.77 µs      │ 9.389 µs      │ 9.495 µs      │ 100     │ 100
│                    3.523 Gitem/s │ 2.377 Gitem/s │ 3.489 Gitem/s │ 3.45 Gitem/s  │         │
├─ div_i64_nonnull   44.96 µs      │ 54.06 µs      │ 45.04 µs      │ 45.34 µs      │ 100     │ 100
│                    728.8 Mitem/s │ 606.1 Mitem/s │ 727.5 Mitem/s │ 722.7 Mitem/s │         │
├─ mul_i8_nonnull    4.559 µs      │ 58.61 µs      │ 4.619 µs      │ 5.202 µs      │ 100     │ 100
│                    7.186 Gitem/s │ 558.9 Mitem/s │ 7.092 Gitem/s │ 6.298 Gitem/s │         │
├─ mul_i16_nonnull   4.159 µs      │ 5.809 µs      │ 4.229 µs      │ 4.244 µs      │ 100     │ 100
│                    7.877 Gitem/s │ 5.64 Gitem/s  │ 7.746 Gitem/s │ 7.72 Gitem/s  │         │
├─ mul_i32_constant  32.23 µs      │ 36.15 µs      │ 32.36 µs      │ 32.49 µs      │ 100     │ 100
│                    1.016 Gitem/s │ 906.2 Mitem/s │ 1.012 Gitem/s │ 1.008 Gitem/s │         │
├─ mul_i32_nonnull   27.81 µs      │ 43.9 µs       │ 31.24 µs      │ 31.22 µs      │ 100     │ 100
│                    1.177 Gitem/s │ 746.2 Mitem/s │ 1.048 Gitem/s │ 1.049 Gitem/s │         │
├─ mul_i32_nullable  28.55 µs      │ 50.24 µs      │ 32.04 µs      │ 31.62 µs      │ 100     │ 100
│                    1.147 Gitem/s │ 652.1 Mitem/s │ 1.022 Gitem/s │ 1.036 Gitem/s │         │
├─ mul_i64_nonnull   25.31 µs      │ 29.26 µs      │ 25.65 µs      │ 25.77 µs      │ 100     │ 100
│                    1.294 Gitem/s │ 1.119 Gitem/s │ 1.277 Gitem/s │ 1.271 Gitem/s │         │
├─ mul_u8_nonnull    3.399 µs      │ 55.89 µs      │ 3.469 µs      │ 3.999 µs      │ 100     │ 100
│                    9.638 Gitem/s │ 586.1 Mitem/s │ 9.443 Gitem/s │ 8.193 Gitem/s │         │
├─ mul_u16_nonnull   2.289 µs      │ 6.769 µs      │ 2.369 µs      │ 2.415 µs      │ 100     │ 100
│                    14.31 Gitem/s │ 4.84 Gitem/s  │ 13.82 Gitem/s │ 13.56 Gitem/s │         │
├─ mul_u32_nonnull   6.919 µs      │ 9.879 µs      │ 7.009 µs      │ 7.059 µs      │ 100     │ 100
│                    4.735 Gitem/s │ 3.316 Gitem/s │ 4.674 Gitem/s │ 4.641 Gitem/s │         │
├─ mul_u64_nonnull   19.34 µs      │ 22.38 µs      │ 19.41 µs      │ 19.5 µs       │ 100     │ 100
│                    1.693 Gitem/s │ 1.463 Gitem/s │ 1.687 Gitem/s │ 1.68 Gitem/s  │         │
╰─ sub_i64_constant  9.419 µs      │ 12.37 µs      │ 9.554 µs      │ 9.607 µs      │ 100     │ 100
                     3.478 Gitem/s │ 2.646 Gitem/s │ 3.429 Gitem/s │ 3.41 Gitem/s  │         │


```
