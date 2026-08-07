<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# `owned-u64-mul-dense.s`

```s
.LBB1688_67:
	.loc	562 112 26
	andq	$-2, %r10
	xorl	%ecx, %ecx
	xorl	%edi, %edi
	movq	-48(%rbp), %r8
.Ltmp115437:
.LBB1688_68:
	.loc	564 62 9
	movq	(%r13,%rdi,8), %rax
.Ltmp115438:
	.loc	565 176 44
	mulq	(%r12,%rdi,8)
.Ltmp115439:
	movq	%rdx, %rsi
.Ltmp115440:
	.loc	207 475 9
	movq	%rax, (%r8,%rdi,8)
.Ltmp115441:
	.loc	564 62 9
	movq	8(%r13,%rdi,8), %rax
.Ltmp115442:
	.loc	565 176 44
	mulq	8(%r12,%rdi,8)
.Ltmp115443:
	.loc	566 821 53
	orq	%rcx, %rsi
.Ltmp115444:
	.loc	565 176 44
	movq	%rdx, %rcx
.Ltmp115445:
	.loc	566 821 53
	orq	%rsi, %rcx
.Ltmp115446:
	.loc	207 475 9
	movq	%rax, 8(%r8,%rdi,8)
.Ltmp115447:
	.loc	562 112 26
	addq	$2, %rdi
	cmpq	%rdi, %r10
	jne	.LBB1688_68
.Ltmp115448:
.LBB1688_69:
	testb	$1, %r9b
	je	.LBB1688_84
.Ltmp115449:
	.loc	564 62 9
	movq	(%r13,%rdi,8), %rax
.Ltmp115450:
	.loc	565 176 44
	mulq	(%r12,%rdi,8)
.Ltmp115451:
	.loc	566 821 53
	orq	%rdx, %rcx
.Ltmp115452:
	.loc	566 0 53 is_stmt 0
	movq	-48(%rbp), %rdx
.Ltmp115453:
	.loc	207 475 9 is_stmt 1
	movq	%rax, (%rdx,%rdi,8)
.Ltmp115454:
.LBB1688_84:
	.loc	182 1868 54
	testq	%rcx, %rcx

```
