<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# `candidate-i64-mul-dense.s`

```s
	movq	-376(%rbp), %r15
.Ltmp108913:
	.loc	524 86 32
	testq	%r15, %r15
	.loc	524 86 16 is_stmt 0
	je	.LBB1673_26
.Ltmp108914:
	.loc	563 318 19 is_stmt 1
	xorq	%r14, %rdi
.Ltmp108915:
	.loc	563 0 19 is_stmt 0
	xorq	%r14, %r10
.Ltmp108916:
	.loc	524 88 17 is_stmt 1
	orq	%rdi, %r10
.Ltmp108917:
	jne	.LBB1673_42
.Ltmp108918:
	.loc	182 1904 50
	testq	%r14, %r14
.Ltmp108919:
	.loc	524 92 26
	je	.LBB1673_41
.Ltmp108920:
	.loc	524 0 26 is_stmt 0
	movq	-320(%rbp), %rsi
.Ltmp108921:
	xorl	%edi, %edi
	xorl	%ecx, %ecx
.Ltmp108922:
	.p2align	4
.LBB1673_25:
	.loc	564 62 9 is_stmt 1
	movq	(%r15,%rdi,8), %rax
.Ltmp108923:
	.loc	565 193 24
	imulq	(%rsi,%rdi,8)
.Ltmp108924:
	.loc	207 475 9
	movq	%rax, (%r9,%rdi,8)
.Ltmp108925:
	.loc	565 197 26
	sarq	$63, %rax
.Ltmp108926:
	.loc	565 197 13 is_stmt 0
	xorq	%rdx, %rax
.Ltmp108927:
	.loc	566 109 21 is_stmt 1
	orq	%rax, %rcx
.Ltmp108928:
	.loc	182 1904 50
	incq	%rdi
.Ltmp108929:
	cmpq	%rdi, %r14
	jne	.LBB1673_25
	jmp	.LBB1673_62
.Ltmp108930:
.LBB1673_26:
	.loc	563 90 47
	cmpq	%r14, %rdi
	sete	%cl

```
