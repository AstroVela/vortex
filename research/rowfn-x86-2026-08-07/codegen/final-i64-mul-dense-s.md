<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# `final-i64-mul-dense.s`

```s
.LBB1675_25:
	.loc	567 39 18 is_stmt 1
	movq	(%r13,%rsi,8), %rax
.Ltmp109477:
	.loc	565 194 24
	imulq	(%r12,%rsi,8)
.Ltmp109478:
	.loc	207 475 9
	movq	%rax, (%rdi,%rsi,8)
.Ltmp109479:
	.loc	156 717 17
	incq	%rsi
.Ltmp109480:
	.loc	565 198 26
	sarq	$63, %rax
.Ltmp109481:
	.loc	565 198 13 is_stmt 0
	xorq	%rdx, %rax
.Ltmp109482:
	.loc	566 821 53 is_stmt 1
	orq	%rax, %rcx
.Ltmp109483:
	.loc	182 1904 50
	cmpq	%rsi, %r9
	jne	.LBB1675_25
	jmp	.LBB1675_60

```
