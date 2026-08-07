<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Merge-base production numeric multiply code generation

Revision: `19f771f2a426103aa7d1bf7153a258bb1bab1e19`

Command:

```text
CARGO_TARGET_DIR=/tmp/rowfn-x86.ccCdz5/target-base-codegen \
  cargo rustc -p vortex-array --lib --profile bench -- \
  --emit=llvm-ir,asm -C codegen-units=1
```

Artifacts:

```text
/tmp/rowfn-x86.ccCdz5/target-base-codegen/release/deps/vortex_array-4e5fe3dd7af89793.ll
/tmp/rowfn-x86.ccCdz5/target-base-codegen/release/deps/vortex_array-4e5fe3dd7af89793.s
```

## Production symbols

```text
i64 execute_checked_typed: 648ec4b22808a2d4
i64 checked_op_lanes (varying x varying): df33f84e66e75a91
u64 execute_checked_typed: 8da1eac40a9b0934
u64 checked_op_lanes (varying x varying): 84edf83ddcd3fe05
```

## i64 hot loop

Assembly source begins at line 6,698,561 in the `.s` artifact. The loop is
`.LBB6227_8`:

```asm
movq    (%rdi,%rsi,8), %rax
imulq   (%r15,%rsi,8)
movq    %rax, (%r13,%rsi,8)
incq    %rsi
sarq    $63, %rax
xorq    %rdx, %rax
orq     %rax, %rcx
cmpq    %rsi, %rbx
jne     .LBB6227_8
```

This is one lane per backedge. The one-operand `imulq` produces the signed
128-bit product in `RDX:RAX`; the low half is stored and the high half is
compared with the low-half sign extension through `sarq`/`xorq`. Failure stays
in register `%rcx`. There is no `vector.body`, unroll, or separate remainder.

## u64 hot loop

Assembly source begins at line 6,646,461 in the `.s` artifact. The loop is
`.LBB6175_10`:

```asm
movq    (%rbx,%rdi,8), %rax
mulq    (%r11,%rdi,8)
movq    %rdx, %rsi
movq    %rax, -8(%r9,%rdi,8)
movq    8(%rbx,%rdi,8), %rax
mulq    8(%r11,%rdi,8)
orq     %rcx, %rsi
movq    %rax, (%r9,%rdi,8)
addq    $2, %rdi
movq    %rdx, %rcx
orq     %rsi, %rcx
cmpq    %r10, %rdi
jne     .LBB6175_10
```

This is scalar unsigned high-half multiplication unrolled by two, followed by
a one-lane remainder when the row count is odd. The two loads, multiplies, and
stores are independent except for the register OR reduction. There is no
`vector.body` in this fast value loop.

## IR facts

The all-varying functions are internal and take the source structure through a
`noalias readonly` pointer and return storage through a `noalias writeonly`
pointer. The allocated output stores carry a distinct `!alias.scope` and
`!noalias`; both input loads carry input-side `!noalias`. The second input
length check has become `llvm.assume`, so no panic branch remains in either hot
loop. A slice/assert failure edge exists before the loop at the output-length
validation boundary.

The u64 IR loop is unrolled by two and reduces two i128 high halves through
scalar `or i64`; it has a one-lane epilogue. The i64 IR loop is scalar and uses
an i128 signed multiply, truncation, arithmetic sign extraction, XOR, and a
loop-carried register OR. Neither fast loop contains a call.

Both monomorphs have this parameter-level ownership shape (metadata IDs differ
between them):

```llvm
define internal fastcc void @checked_op_lanes(
    ptr noalias writable writeonly %output,
    ptr noalias readonly %source,
    i64 %valid_rows_tag,
    ptr readonly %valid_rows_data)
```

The relevant u64 body is structurally:

```llvm
%failed = phi i64 [ 0, %preheader ], [ %failed_2, %loop ]
%lhs_0 = load i64, ptr %lhs_ptr_0, !noalias !input_scope
%rhs_0 = load i64, ptr %rhs_ptr_0, !noalias !input_scope
%low_0 = mul i64 %rhs_0, %lhs_0
%wide_0 = mul nuw i128 (zext i64 %rhs_0), (zext i64 %lhs_0)
%high_0 = trunc i128 (lshr i128 %wide_0, 64) to i64
%failed_1 = or i64 %failed, %high_0
store i64 %low_0, ptr %output_0, !alias.scope !output_scope, !noalias !output_noalias

%lhs_1 = load i64, ptr %lhs_ptr_1, !noalias !input_scope
%rhs_1 = load i64, ptr %rhs_ptr_1, !noalias !input_scope
%low_1 = mul i64 %rhs_1, %lhs_1
%wide_1 = mul nuw i128 (zext i64 %rhs_1), (zext i64 %lhs_1)
%high_1 = trunc i128 (lshr i128 %wide_1, 64) to i64
%failed_2 = or i64 %failed_1, %high_1
store i64 %low_1, ptr %output_1, !alias.scope !output_scope, !noalias !output_noalias
```

The relevant i64 body is structurally:

```llvm
%failed = phi i64 [ 0, %preheader ], [ %failed_next, %loop ]
%lhs = load i64, ptr %lhs_ptr, !noalias !input_scope
%rhs = load i64, ptr %rhs_ptr, !noalias !input_scope
%wide = mul nsw i128 (sext i64 %rhs), (sext i64 %lhs)
%low = trunc i128 %wide to i64
%high = trunc i128 (lshr i128 %wide, 64) to i64
%discarded_mismatch = xor i64 (ashr i64 %low, 63), %high
%failed_next = or i64 %discarded_mismatch, %failed
store i64 %low, ptr %output, !alias.scope !output_scope, !noalias !output_noalias
```
