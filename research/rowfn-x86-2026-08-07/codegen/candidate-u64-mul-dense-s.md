<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# `candidate-u64-mul-dense.s`

```s
	.loc	524 92 26
	andq	$-2, %r15
	leaq	(%r12,%rdx), %r11
	addq	$8, %r11
	xorl	%ecx, %ecx
	xorl	%r10d, %r10d
.Ltmp117915:
.LBB1693_70:
	.loc	564 62 9
	movq	(%r14,%r10,8), %rax
.Ltmp117916:
	.loc	565 175 44
	mulq	(%r8,%r10,8)
.Ltmp117917:
	movq	%rdx, %rdi
.Ltmp117918:
	.loc	207 475 9
	movq	%rax, -8(%r11,%r10,8)
.Ltmp117919:
	.loc	564 62 9
	movq	8(%r14,%r10,8), %rax
.Ltmp117920:
	.loc	565 175 44
	mulq	8(%r8,%r10,8)
.Ltmp117921:
	.loc	566 109 21
	orq	%rcx, %rdi
.Ltmp117922:
	.loc	207 475 9
	movq	%rax, (%r11,%r10,8)
.Ltmp117923:
	.loc	565 175 44
	movq	%rdx, %rcx
.Ltmp117924:
	.loc	566 109 21
	orq	%rdi, %rcx
.Ltmp117925:
	.loc	524 92 26
	addq	$2, %r10
	cmpq	%r10, %r15
	jne	.LBB1693_70
.Ltmp117926:
.LBB1693_71:
	testb	$1, %sil
	je	.LBB1693_87
.Ltmp117927:
	.loc	564 62 9
	movq	(%r14,%r10,8), %rax
.Ltmp117928:
	.loc	565 175 44
	mulq	(%r8,%r10,8)
.Ltmp117929:
.LBB1693_73:
	.loc	207 475 9
	movq	%rax, (%r9,%r10,8)
.Ltmp117930:
	.loc	566 109 21
	orq	%rdx, %rcx
.Ltmp117931:
	.loc	566 0 21 is_stmt 0
	jmp	.LBB1693_87
.Ltmp117932:

```
