<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# `owned-i64-mul-dense.s`

```s
	.loc	562 112 26 is_stmt 1
	je	.LBB1685_70
.Ltmp114190:
	.loc	562 0 26 is_stmt 0
	movq	-128(%rbp), %r13
.Ltmp114191:
	xorl	%esi, %esi
.Ltmp114192:
	xorl	%ecx, %ecx
	movq	-48(%rbp), %rdi
.Ltmp114193:
	.p2align	4
.LBB1685_25:
	.loc	564 62 9 is_stmt 1
	movq	(%r13,%rsi,8), %rax
.Ltmp114194:
	.loc	565 194 24
	imulq	(%r12,%rsi,8)
.Ltmp114195:
	.loc	207 475 9
	movq	%rax, (%rdi,%rsi,8)
.Ltmp114196:
	.loc	565 198 26
	sarq	$63, %rax
.Ltmp114197:
	.loc	565 198 13 is_stmt 0
	xorq	%rdx, %rax
.Ltmp114198:
	.loc	566 821 53 is_stmt 1
	orq	%rax, %rcx
.Ltmp114199:
	.loc	182 1904 50
	incq	%rsi
.Ltmp114200:
	cmpq	%rsi, %r9
	jne	.LBB1685_25
	jmp	.LBB1685_60
.Ltmp114201:

```
