<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# `indexed-i64-mul-dense.s`

```s
	.p2align	4
.LBB1685_25:
	.loc	566 39 18 is_stmt 1
	movq	(%r13,%rsi,8), %rax
.Ltmp113564:
	.loc	565 194 24
	imulq	(%r12,%rsi,8)
.Ltmp113565:
	.loc	207 475 9
	movq	%rax, (%rdi,%rsi,8)
.Ltmp113566:
	.loc	156 717 17
	incq	%rsi
.Ltmp113567:
	.loc	565 198 26
	sarq	$63, %rax
.Ltmp113568:
	.loc	565 198 13 is_stmt 0
	xorq	%rdx, %rax
.Ltmp113569:
	.loc	568 821 53 is_stmt 1
	orq	%rax, %rcx
.Ltmp113570:
	.loc	182 1904 50
	cmpq	%rsi, %r9
	jne	.LBB1685_25
	jmp	.LBB1685_60

```
