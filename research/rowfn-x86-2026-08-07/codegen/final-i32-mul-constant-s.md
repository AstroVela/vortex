<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# `final-i32-mul-constant.s`

```s
.LBB1677_31:
	.loc	564 47 9 is_stmt 1
	cmpq	%rdi, %rdx
	je	.LBB1677_89
.Ltmp110238:
	.loc	564 47 9 is_stmt 0
	movslq	(%r13,%rdi,4), %rcx
.Ltmp110239:
	.loc	564 47 9
	movslq	(%r12), %r8
.Ltmp110240:
	.loc	462 2133 13 is_stmt 1
	movl	%r8d, %r10d
	imull	%ecx, %r10d
.Ltmp110241:
	.loc	565 185 27
	imulq	%rcx, %r8
.Ltmp110242:
	.loc	565 185 35 is_stmt 0
	addq	$-2147483648, %r8
.Ltmp110243:
	cmpq	%rax, %r8
	setb	%cl
.Ltmp110244:
	.loc	565 0 35
	movq	-48(%rbp), %r8
.Ltmp110245:
	.loc	207 475 9 is_stmt 1
	movl	%r10d, (%r8,%rdi,4)
.Ltmp110246:
	.loc	566 821 53
	orb	%cl, %r9b
.Ltmp110247:
	.loc	182 1904 50
	incq	%rdi
.Ltmp110248:
	cmpq	%rdi, %rdx
.Ltmp110249:
	.loc	562 124 26
	jne	.LBB1677_31
	jmp	.LBB1677_64

```
