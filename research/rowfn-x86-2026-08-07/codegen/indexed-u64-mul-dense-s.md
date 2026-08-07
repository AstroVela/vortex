<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# `indexed-u64-mul-dense.s`

```s
.LBB1688_67:
	.loc	181 765 12
	andq	$-2, %r10
	xorl	%edi, %edi
	xorl	%ecx, %ecx
	movq	-48(%rbp), %r8
.Ltmp114798:
.LBB1688_68:
	.loc	566 39 18
	movq	(%r13,%rdi,8), %rax
.Ltmp114799:
	.loc	565 176 44
	mulq	(%r12,%rdi,8)
.Ltmp114800:
	movq	%rdx, %rsi
.Ltmp114801:
	.loc	207 475 9
	movq	%rax, (%r8,%rdi,8)
.Ltmp114802:
	.loc	566 39 18
	movq	8(%r13,%rdi,8), %rax
.Ltmp114803:
	.loc	565 176 44
	mulq	8(%r12,%rdi,8)
.Ltmp114804:
	.loc	568 821 53
	orq	%rcx, %rsi
.Ltmp114805:
	.loc	207 475 9
	movq	%rax, 8(%r8,%rdi,8)
.Ltmp114806:
	.loc	156 717 17
	addq	$2, %rdi
.Ltmp114807:
	.loc	565 176 44
	movq	%rdx, %rcx
.Ltmp114808:
	.loc	568 821 53
	orq	%rsi, %rcx
.Ltmp114809:
	.loc	181 765 12
	cmpq	%r10, %rdi
	jne	.LBB1688_68
.Ltmp114810:
.LBB1688_69:
	testb	$1, %r9b
	je	.LBB1688_84
.Ltmp114811:
	.loc	566 39 18
	movq	(%r13,%rdi,8), %rax
.Ltmp114812:
	.loc	565 176 44
	mulq	(%r12,%rdi,8)
.Ltmp114813:
	.loc	568 821 53
	orq	%rdx, %rcx
.Ltmp114814:
	.loc	568 0 53 is_stmt 0
	movq	-48(%rbp), %rdx
.Ltmp114815:
	.loc	207 475 9 is_stmt 1
	movq	%rax, (%rdx,%rdi,8)
.Ltmp114816:

```
