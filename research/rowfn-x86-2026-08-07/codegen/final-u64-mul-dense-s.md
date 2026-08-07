<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# `final-u64-mul-dense.s`

```s
.LBB1679_67:
	.loc	181 765 12
	andq	$-2, %r10
	xorl	%edi, %edi
	xorl	%ecx, %ecx
	movq	-48(%rbp), %r8
.Ltmp111183:
.LBB1679_68:
	.loc	567 39 18
	movq	(%r13,%rdi,8), %rax
.Ltmp111184:
	.loc	565 176 44
	mulq	(%r12,%rdi,8)
.Ltmp111185:
	movq	%rdx, %rsi
.Ltmp111186:
	.loc	207 475 9
	movq	%rax, (%r8,%rdi,8)
.Ltmp111187:
	.loc	567 39 18
	movq	8(%r13,%rdi,8), %rax
.Ltmp111188:
	.loc	565 176 44
	mulq	8(%r12,%rdi,8)
.Ltmp111189:
	.loc	566 821 53
	orq	%rcx, %rsi
.Ltmp111190:
	.loc	207 475 9
	movq	%rax, 8(%r8,%rdi,8)
.Ltmp111191:
	.loc	156 717 17
	addq	$2, %rdi
.Ltmp111192:
	.loc	565 176 44
	movq	%rdx, %rcx
.Ltmp111193:
	.loc	566 821 53
	orq	%rsi, %rcx
.Ltmp111194:
	.loc	181 765 12
	cmpq	%r10, %rdi
	jne	.LBB1679_68
.Ltmp111195:
.LBB1679_69:
	testb	$1, %r9b
	je	.LBB1679_84
.Ltmp111196:
	.loc	567 39 18
	movq	(%r13,%rdi,8), %rax
.Ltmp111197:
	.loc	565 176 44
	mulq	(%r12,%rdi,8)
.Ltmp111198:
	.loc	566 821 53
	orq	%rdx, %rcx
.Ltmp111199:
	.loc	566 0 53 is_stmt 0
	movq	-48(%rbp), %rdx
.Ltmp111200:
	.loc	207 475 9 is_stmt 1
	movq	%rax, (%rdx,%rdi,8)
.Ltmp111201:

```
