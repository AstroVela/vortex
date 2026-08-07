<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# `Copy`-bound compiler ablation

This note records the compact evidence behind the no-drop assertion in owned RowFn execution.

## Timings

All public `binary_ops` runs used CPU 8 and the same default repository flags.

```text
no Copy bound                 mul_i32_constant  18.77 / 18.72 us
inert private marker bound    mul_i32_constant  18.77 / 18.72 us
Output: Copy                  mul_i32_constant  29.94 / 29.93 us
Output: Copy, CGU=1           mul_i32_constant  29.87 / 29.89 us
i64/u64 controls              unchanged
```

The inert marker rules out a generic “any where-clause/source change perturbs codegen” explanation.
`codegen-units=1` rules out the default partitioning choice as a repair.

## Exact production functions

Default-CGU DWARF identified the measured `i32 CheckedMul` monomorphs:

```text
Copy:     0xe58c90..0xe59adc, size 0xe4c
no-Copy:  0xe7a1c0..0xe7b6d0, size 0x1510

Copy hot loop:     0xe58f90, 16-byte but not 32-byte aligned
no-Copy hot loop:  0xe7b260, 32-byte aligned
```

The Copy loop schedules the low `imul` before the widened-product chain. No-Copy schedules the
widened chain first and delays the low multiply. LLVM-MCA predicts Copy slightly better at 2.5
cycles versus 2.7, contradicting scheduling as the cause of its 1.6x wall-time loss.

Fresh Copy-plus-CGU1 optimized IR contains store-before-OR and still runs at 29.9 microseconds.
Therefore store-before-OR is not sufficient. Exact isolated loops also contradict causality:

```text
LLVM-MCA:          both orders 2.7 cycles
OR before store:   0.75-0.77 ns/row
store before OR:   0.823-0.825 ns/row
```

The standalone generic `MaybeUninit` loop produces identical Copy/no-Copy IR and assembly. The
remaining hypothesis is phase-order or code-quality sensitivity requiring the real trait, closure,
`Vec`, and monomorphization context. Do not label this a correctness bug or assign it to a specific
rustc/LLVM pass without a reduced reproducer. Reduce the real monomorph while retaining timing and
whole-function changes, then bisect MIR/LLVM passes and compiler versions.
