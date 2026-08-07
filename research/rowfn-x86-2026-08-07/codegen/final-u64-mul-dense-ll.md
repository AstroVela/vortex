<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# `final-u64-mul-dense.ll`

```ll
bb15.i.i:                                         ; preds = %bb15.i.i, %bb15.i.i.preheader.new
  %iter.sroa.0.012.i.i = phi i64 [ 0, %bb15.i.i.preheader.new ], [ %_36.i.i.1, %bb15.i.i ]
  %failed.sroa.0.011.i.i = phi i64 [ 0, %bb15.i.i.preheader.new ], [ %116, %bb15.i.i ]
  %niter = phi i64 [ 0, %bb15.i.i.preheader.new ], [ %niter.next.1, %bb15.i.i ]
    #dbg_value(i64 %iter.sroa.0.012.i.i, !569875, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !569928)
    #dbg_value(i64 %failed.sroa.0.011.i.i, !569873, !DIExpression(), !569927)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !569912, !DIExpression(), !570143)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !569905, !DIExpression(), !569906)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !569922, !DIExpression(), !569923)
  %_36.i.i = or disjoint i64 %iter.sroa.0.012.i.i, 1, !dbg !570144
    #dbg_value(i64 %_36.i.i, !569875, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !569928)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !569877, !DIExpression(), !570145)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !569899, !DIExpression(), !569900)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !570146, !DIExpression(), !570150)
    #dbg_value(ptr undef, !551471, !DIExpression(), !569894)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !551477, !DIExpression(), !569894)
    #dbg_value(ptr poison, !551549, !DIExpression(), !570152)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !551550, !DIExpression(), !570152)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !551542, !DIExpression(), !570154)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !551534, !DIExpression(), !570156)
    #dbg_value(ptr %column.val.i.i, !551541, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !570154)
    #dbg_value(ptr %column.val.i.i, !551535, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !570156)
    #dbg_value(i64 %len3.i.i.i, !551541, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !570154)
    #dbg_value(i64 %len3.i.i.i, !551535, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !570156)
  %_4.i.i.i.i = getelementptr inbounds nuw i64, ptr %column.val.i.i, i64 %iter.sroa.0.012.i.i, !dbg !570158
  %_0.i.i.i.i = load i64, ptr %_4.i.i.i.i, align 8, !dbg !570159, !noalias !570160, !noundef !23
    #dbg_value(ptr poison, !551549, !DIExpression(), !570164)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !551550, !DIExpression(), !570164)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !551542, !DIExpression(), !570166)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !551534, !DIExpression(), !570168)
    #dbg_value(ptr %column5.val.i.i, !551541, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !570166)
    #dbg_value(ptr %column5.val.i.i, !551535, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !570168)
    #dbg_value(i64 %len3.i.i.i, !551541, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !570166)
    #dbg_value(i64 %len3.i.i.i, !551535, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !570168)
  %_5.i3.i.i.i = icmp ult i64 %iter.sroa.0.012.i.i, %len3.i.i.i, !dbg !570170
  tail call void @llvm.assume(i1 %_5.i3.i.i.i), !dbg !570171
  %_4.i4.i.i.i = getelementptr inbounds nuw i64, ptr %column5.val.i.i, i64 %iter.sroa.0.012.i.i, !dbg !570172
  %_0.i5.i.i.i = load i64, ptr %_4.i4.i.i.i, align 8, !dbg !570173, !noalias !570160, !noundef !23
    #dbg_value(i64 %_0.i.i.i.i, !569879, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !570174)
    #dbg_value(i64 %_0.i5.i.i.i, !569879, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !570174)
    #dbg_value(i64 %_0.i.i.i.i, !570175, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !570183)
    #dbg_value(i64 %_0.i5.i.i.i, !570175, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !570183)
    #dbg_value(ptr poison, !569758, !DIExpression(), !570185)
    #dbg_value(ptr poison, !569759, !DIExpression(), !570185)
    #dbg_value(i64 %_0.i.i.i.i, !569760, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !570185)
    #dbg_value(i64 %_0.i5.i.i.i, !569760, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !570185)
    #dbg_value(i64 %_0.i.i.i.i, !569756, !DIExpression(), !570187)
    #dbg_value(i64 %_0.i5.i.i.i, !569757, !DIExpression(), !570187)
    #dbg_value(i64 %_0.i.i.i.i, !569747, !DIExpression(), !570188)
    #dbg_value(i64 %_0.i5.i.i.i, !569748, !DIExpression(), !570188)
    #dbg_value(i64 %_0.i.i.i.i, !569740, !DIExpression(), !570190)
    #dbg_value(i64 %_0.i.i.i.i, !569735, !DIExpression(), !570192)
    #dbg_value(i64 %_0.i5.i.i.i, !569741, !DIExpression(), !570190)
    #dbg_value(i64 %_0.i5.i.i.i, !569736, !DIExpression(), !570192)
  %_0.i3.i.i.i.i = mul i64 %_0.i5.i.i.i, %_0.i.i.i.i, !dbg !570194
    #dbg_value(i64 %_0.i.i.i.i, !569766, !DIExpression(), !570195)
    #dbg_value(i64 %_0.i.i.i.i, !569768, !DIExpression(), !570197)
    #dbg_value(i64 %_0.i5.i.i.i, !569767, !DIExpression(), !570195)
    #dbg_value(i64 %_0.i5.i.i.i, !569769, !DIExpression(), !570197)
  %_5.i.i.i.i.i = zext i64 %_0.i.i.i.i to i128, !dbg !570198
  %_6.i.i.i.i.i = zext i64 %_0.i5.i.i.i to i128, !dbg !570199
  %_4.i1.i.i.i.i = mul nuw i128 %_6.i.i.i.i.i, %_5.i.i.i.i.i, !dbg !570200
  %_3.i2.i.i.i.i = lshr i128 %_4.i1.i.i.i.i, 64, !dbg !570201
  %_0.i.i.i.i.i = trunc nuw i128 %_3.i2.i.i.i.i to i64, !dbg !570202
    #dbg_value(i64 poison, !569881, !DIExpression(), !570203)
    #dbg_value(i64 %_0.i.i.i.i.i, !569883, !DIExpression(), !570203)
    #dbg_value(ptr undef, !564034, !DIExpression(), !569892)
    #dbg_value(i64 %_0.i.i.i.i.i, !564040, !DIExpression(), !569892)
  %115 = or i64 %failed.sroa.0.011.i.i, %_0.i.i.i.i.i, !dbg !570204
    #dbg_value(i64 %115, !569873, !DIExpression(), !569927)
    #dbg_value(i64 %_0.i3.i.i.i.i, !569881, !DIExpression(), !570203)
    #dbg_value(ptr %_4.sroa.10.0.i.i, !570149, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !570150)
    #dbg_value(i64 %index.i, !570149, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !570150)
  %self4.i.i = getelementptr inbounds nuw i64, ptr %_4.sroa.10.0.i.i, i64 %iter.sroa.0.012.i.i, !dbg !570205
    #dbg_value(ptr %self4.i.i, !570206, !DIExpression(), !570210)
    #dbg_value(i64 %_0.i3.i.i.i.i, !570209, !DIExpression(), !570210)
  store i64 %_0.i3.i.i.i.i, ptr %self4.i.i, align 8, !dbg !570212, !alias.scope !569888, !noalias !570213
    #dbg_value(ptr undef, !569916, !DIExpression(), !569929)
    #dbg_value(ptr undef, !569911, !DIExpression(), !569930)
    #dbg_value(ptr undef, !569931, !DIExpression(), !569935)
    #dbg_value(ptr poison, !569934, !DIExpression(), !569935)
    #dbg_value(i64 %_36.i.i, !569912, !DIExpression(), !570143)
    #dbg_value(i64 %_36.i.i, !569905, !DIExpression(), !569906)
    #dbg_value(i64 %_36.i.i, !569922, !DIExpression(), !569923)
  %_36.i.i.1 = add nuw i64 %iter.sroa.0.012.i.i, 2, !dbg !570144
    #dbg_value(i64 %_36.i.i.1, !569875, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !569928)
    #dbg_value(i64 %_36.i.i, !569877, !DIExpression(), !570145)
    #dbg_value(i64 %_36.i.i, !569899, !DIExpression(), !569900)
    #dbg_value(i64 %_36.i.i, !570146, !DIExpression(), !570150)
    #dbg_value(i64 %_36.i.i, !551477, !DIExpression(), !569894)
    #dbg_value(i64 %_36.i.i, !551550, !DIExpression(), !570152)
    #dbg_value(i64 %_36.i.i, !551542, !DIExpression(), !570154)
    #dbg_value(i64 %_36.i.i, !551534, !DIExpression(), !570156)
    #dbg_value(ptr %column.val.i.i, !551541, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !570154)
    #dbg_value(ptr %column.val.i.i, !551535, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !570156)
    #dbg_value(i64 %len3.i.i.i, !551541, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !570154)
    #dbg_value(i64 %len3.i.i.i, !551535, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !570156)
  %_4.i.i.i.i.1 = getelementptr inbounds nuw i64, ptr %column.val.i.i, i64 %_36.i.i, !dbg !570158
  %_0.i.i.i.i.1 = load i64, ptr %_4.i.i.i.i.1, align 8, !dbg !570159, !noalias !570160, !noundef !23
    #dbg_value(i64 %_36.i.i, !551550, !DIExpression(), !570164)
    #dbg_value(i64 %_36.i.i, !551542, !DIExpression(), !570166)
    #dbg_value(i64 %_36.i.i, !551534, !DIExpression(), !570168)
    #dbg_value(ptr %column5.val.i.i, !551541, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !570166)
    #dbg_value(ptr %column5.val.i.i, !551535, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !570168)
    #dbg_value(i64 %len3.i.i.i, !551541, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !570166)
    #dbg_value(i64 %len3.i.i.i, !551535, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !570168)
  %_5.i3.i.i.i.1 = icmp ult i64 %_36.i.i, %len3.i.i.i, !dbg !570170
  tail call void @llvm.assume(i1 %_5.i3.i.i.i.1), !dbg !570171
  %_4.i4.i.i.i.1 = getelementptr inbounds nuw i64, ptr %column5.val.i.i, i64 %_36.i.i, !dbg !570172
  %_0.i5.i.i.i.1 = load i64, ptr %_4.i4.i.i.i.1, align 8, !dbg !570173, !noalias !570160, !noundef !23
    #dbg_value(i64 %_0.i.i.i.i.1, !569879, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !570174)
    #dbg_value(i64 %_0.i5.i.i.i.1, !569879, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !570174)
    #dbg_value(i64 %_0.i.i.i.i.1, !570175, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !570183)
    #dbg_value(i64 %_0.i5.i.i.i.1, !570175, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !570183)
    #dbg_value(i64 %_0.i.i.i.i.1, !569760, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !570185)
    #dbg_value(i64 %_0.i5.i.i.i.1, !569760, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !570185)
    #dbg_value(i64 %_0.i.i.i.i.1, !569756, !DIExpression(), !570187)
    #dbg_value(i64 %_0.i5.i.i.i.1, !569757, !DIExpression(), !570187)
    #dbg_value(i64 %_0.i.i.i.i.1, !569747, !DIExpression(), !570188)
    #dbg_value(i64 %_0.i5.i.i.i.1, !569748, !DIExpression(), !570188)
    #dbg_value(i64 %_0.i.i.i.i.1, !569740, !DIExpression(), !570190)
    #dbg_value(i64 %_0.i.i.i.i.1, !569735, !DIExpression(), !570192)
    #dbg_value(i64 %_0.i5.i.i.i.1, !569741, !DIExpression(), !570190)
    #dbg_value(i64 %_0.i5.i.i.i.1, !569736, !DIExpression(), !570192)
  %_0.i3.i.i.i.i.1 = mul i64 %_0.i5.i.i.i.1, %_0.i.i.i.i.1, !dbg !570194
    #dbg_value(i64 %_0.i.i.i.i.1, !569766, !DIExpression(), !570195)
    #dbg_value(i64 %_0.i.i.i.i.1, !569768, !DIExpression(), !570197)
    #dbg_value(i64 %_0.i5.i.i.i.1, !569767, !DIExpression(), !570195)
    #dbg_value(i64 %_0.i5.i.i.i.1, !569769, !DIExpression(), !570197)
  %_5.i.i.i.i.i.1 = zext i64 %_0.i.i.i.i.1 to i128, !dbg !570198
  %_6.i.i.i.i.i.1 = zext i64 %_0.i5.i.i.i.1 to i128, !dbg !570199
  %_4.i1.i.i.i.i.1 = mul nuw i128 %_6.i.i.i.i.i.1, %_5.i.i.i.i.i.1, !dbg !570200
  %_3.i2.i.i.i.i.1 = lshr i128 %_4.i1.i.i.i.i.1, 64, !dbg !570201
  %_0.i.i.i.i.i.1 = trunc nuw i128 %_3.i2.i.i.i.i.1 to i64, !dbg !570202
    #dbg_value(i64 poison, !569881, !DIExpression(), !570203)
    #dbg_value(i64 %_0.i.i.i.i.i.1, !569883, !DIExpression(), !570203)
    #dbg_value(i64 %_0.i.i.i.i.i.1, !564040, !DIExpression(), !569892)
  %116 = or i64 %115, %_0.i.i.i.i.i.1, !dbg !570204
    #dbg_value(i64 %116, !569873, !DIExpression(), !569927)
    #dbg_value(i64 %_0.i3.i.i.i.i.1, !569881, !DIExpression(), !570203)
    #dbg_value(ptr %_4.sroa.10.0.i.i, !570149, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !570150)
    #dbg_value(i64 %index.i, !570149, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !570150)
  %self4.i.i.1 = getelementptr inbounds nuw i64, ptr %_4.sroa.10.0.i.i, i64 %_36.i.i, !dbg !570205
    #dbg_value(ptr %self4.i.i.1, !570206, !DIExpression(), !570210)
    #dbg_value(i64 %_0.i3.i.i.i.i.1, !570209, !DIExpression(), !570210)
  store i64 %_0.i3.i.i.i.i.1, ptr %self4.i.i.1, align 8, !dbg !570212, !alias.scope !569888, !noalias !570213
  %niter.next.1 = add i64 %niter, 2, !dbg !569937
  %niter.ncmp.1 = icmp eq i64 %niter.next.1, %unroll_iter, !dbg !569937
  br i1 %niter.ncmp.1, label %bb33.i.loopexit144.unr-lcssa, label %bb15.i.i, !dbg !569937

bb33.thread.i:                                    ; preds = %bb26.preheader.i.thread, %bb11.i, %bb26.preheader.i
    #dbg_value(i64 0, !569430, !DIExpression(), !570214)
    #dbg_value(i64 %index.i, !569423, !DIExpression(DW_OP_LLVM_fragment, 128, 64), !569651)
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %value.i.i), !dbg !570215, !noalias !569589
    #dbg_value(i64 0, !569401, !DIExpression(), !570217)
    #dbg_declare(ptr poison, !569405, !DIExpression(), !570218)
    #dbg_declare(ptr %value.i.i, !570219, !DIExpression(), !570222)
    #dbg_value(ptr undef, !564775, !DIExpression(), !570225)
    #dbg_value(ptr undef, !564776, !DIExpression(), !570225)
  br label %bb36.i, !dbg !570226

bb33.i.loopexit.unr-lcssa:                        ; preds = %bb27.us.i.us, %bb27.us.i.us.preheader
  %.lcssa.ph = phi i64 [ poison, %bb27.us.i.us.preheader ], [ %88, %bb27.us.i.us ]
  %iter.sroa.0.046.us.i.us.unr = phi i64 [ 0, %bb27.us.i.us.preheader ], [ %_158.us.i.us, %bb27.us.i.us ]
  %accumulated.sroa.0.045.us.i.us.unr = phi i64 [ 0, %bb27.us.i.us.preheader ], [ %88, %bb27.us.i.us ]
  %lcmp.mod148.not = icmp eq i64 %xtraiter147, 0, !dbg !569720
  br i1 %lcmp.mod148.not, label %bb33.i, label %bb27.us.i.us.epil, !dbg !569720

bb27.us.i.us.epil:                                ; preds = %bb33.i.loopexit.unr-lcssa
    #dbg_value(i64 %iter.sroa.0.046.us.i.us.unr, !569449, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !569718)
    #dbg_value(i64 %accumulated.sroa.0.045.us.i.us.unr, !569447, !DIExpression(), !569717)
    #dbg_value(i64 %iter.sroa.0.046.us.i.us.unr, !569451, !DIExpression(), !569793)
    #dbg_value(ptr %columns.i, !551276, !DIExpression(), !569794)
    #dbg_value(i64 %iter.sroa.0.046.us.i.us.unr, !551277, !DIExpression(), !569794)
    #dbg_value(ptr %columns.i, !551266, !DIExpression(), !569795)
    #dbg_value(i64 %iter.sroa.0.046.us.i.us.unr, !551267, !DIExpression(), !569795)
    #dbg_value(ptr %columns.i, !551137, !DIExpression(DW_OP_plus_uconst, 8, DW_OP_stack_value), !569798)
    #dbg_value(i64 0, !551142, !DIExpression(), !569796)
    #dbg_value(ptr %columns.i, !551269, !DIExpression(DW_OP_plus_uconst, 8, DW_OP_stack_value), !569827)
    #dbg_value(ptr %columns.i, !551137, !DIExpression(DW_OP_plus_uconst, 8, DW_OP_stack_value), !569796)
  %_0.sroa.0.0.i.i.us.i.us.epil = load i64, ptr %data.i.i.i.us.i, align 8, !dbg !569725, !noalias !569509, !noundef !23
    #dbg_value(ptr %14, !551266, !DIExpression(), !569800)
    #dbg_value(i64 %iter.sroa.0.046.us.i.us.unr, !551267, !DIExpression(), !569800)
    #dbg_value(ptr %14, !551137, !DIExpression(DW_OP_plus_uconst, 8, DW_OP_stack_value), !569801)
    #dbg_value(ptr %14, !551137, !DIExpression(DW_OP_plus_uconst, 8, DW_OP_stack_value), !569803)
    #dbg_value(ptr %14, !551269, !DIExpression(DW_OP_plus_uconst, 8, DW_OP_stack_value), !569804)
    #dbg_value(i64 0, !551142, !DIExpression(), !569801)
  %_0.sroa.0.0.i9.i.us.i.us.epil = load i64, ptr %data.i6.i7.i.us.i, align 8, !dbg !569730, !noalias !569509, !noundef !23
    #dbg_value(ptr poison, !569758, !DIExpression(), !569805)
    #dbg_value(ptr poison, !569759, !DIExpression(), !569805)
    #dbg_value(i64 %_0.sroa.0.0.i.i.us.i.us.epil, !569760, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !569805)
    #dbg_value(i64 %_0.sroa.0.0.i9.i.us.i.us.epil, !569760, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !569805)
    #dbg_value(i64 %_0.sroa.0.0.i.i.us.i.us.epil, !569756, !DIExpression(), !569806)
    #dbg_value(i64 %_0.sroa.0.0.i9.i.us.i.us.epil, !569757, !DIExpression(), !569806)
    #dbg_value(i64 %_0.sroa.0.0.i.i.us.i.us.epil, !569747, !DIExpression(), !569807)
    #dbg_value(i64 %_0.sroa.0.0.i9.i.us.i.us.epil, !569748, !DIExpression(), !569807)
    #dbg_value(i64 %_0.sroa.0.0.i.i.us.i.us.epil, !569740, !DIExpression(), !569808)
    #dbg_value(i64 %_0.sroa.0.0.i.i.us.i.us.epil, !569735, !DIExpression(), !569809)
    #dbg_value(i64 %_0.sroa.0.0.i9.i.us.i.us.epil, !569741, !DIExpression(), !569808)
    #dbg_value(i64 %_0.sroa.0.0.i9.i.us.i.us.epil, !569736, !DIExpression(), !569809)
  %_0.i3.i.us.i.us.epil = mul i64 %_0.sroa.0.0.i9.i.us.i.us.epil, %_0.sroa.0.0.i.i.us.i.us.epil, !dbg !569732
    #dbg_value(i64 %_0.sroa.0.0.i.i.us.i.us.epil, !569766, !DIExpression(), !569810)
    #dbg_value(i64 %_0.sroa.0.0.i.i.us.i.us.epil, !569768, !DIExpression(), !569811)
    #dbg_value(i64 %_0.sroa.0.0.i9.i.us.i.us.epil, !569767, !DIExpression(), !569810)
    #dbg_value(i64 %_0.sroa.0.0.i9.i.us.i.us.epil, !569769, !DIExpression(), !569811)
  %_5.i.i.us.i.us.epil = zext i64 %_0.sroa.0.0.i.i.us.i.us.epil to i128, !dbg !569762
  %_6.i.i.us.i.us.epil = zext i64 %_0.sroa.0.0.i9.i.us.i.us.epil to i128, !dbg !569771
  %_4.i1.i.us.i.us.epil = mul nuw i128 %_6.i.i.us.i.us.epil, %_5.i.i.us.i.us.epil, !dbg !569772
  %_3.i2.i.us.i.us.epil = lshr i128 %_4.i1.i.us.i.us.epil, 64, !dbg !569773
  %_0.i.i159.us.i.us.epil = trunc nuw i128 %_3.i2.i.us.i.us.epil to i64, !dbg !569774
    #dbg_value(i64 %_0.i.i159.us.i.us.epil, !569455, !DIExpression(), !569812)
    #dbg_value(i64 %_0.i3.i.us.i.us.epil, !569453, !DIExpression(), !569812)
  %self34.us.i.us.epil = getelementptr inbounds nuw i64, ptr %_4.sroa.10.0.i.i, i64 %iter.sroa.0.046.us.i.us.unr, !dbg !569775
    #dbg_value(ptr %self34.us.i.us.epil, !569779, !DIExpression(), !569813)
    #dbg_value(i64 %_0.i3.i.us.i.us.epil, !569780, !DIExpression(), !569813)
  store i64 %_0.i3.i.us.i.us.epil, ptr %self34.us.i.us.epil, align 8, !dbg !569776, !noalias !569509
    #dbg_value(ptr undef, !564034, !DIExpression(), !569482)
    #dbg_value(i64 %_0.i.i159.us.i.us.epil, !564040, !DIExpression(), !569482)
  %117 = or i64 %accumulated.sroa.0.045.us.i.us.unr, %_0.i.i159.us.i.us.epil, !dbg !569782
    #dbg_value(i64 %117, !569447, !DIExpression(), !569717)
    #dbg_value(i64 poison, !569449, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !569718)
    #dbg_value(ptr undef, !569472, !DIExpression(), !569475)
    #dbg_value(ptr undef, !569463, !DIExpression(), !569468)
    #dbg_value(ptr undef, !569476, !DIExpression(), !569480)
    #dbg_value(ptr poison, !569479, !DIExpression(), !569480)
    #dbg_value(i64 poison, !569449, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !569718)
  br label %bb33.i, !dbg !570227

bb33.i.loopexit144.unr-lcssa:                     ; preds = %bb15.i.i, %bb15.i.i.preheader
  %.lcssa145.ph = phi i64 [ poison, %bb15.i.i.preheader ], [ %116, %bb15.i.i ]
  %iter.sroa.0.012.i.i.unr = phi i64 [ 0, %bb15.i.i.preheader ], [ %_36.i.i.1, %bb15.i.i ]
  %failed.sroa.0.011.i.i.unr = phi i64 [ 0, %bb15.i.i.preheader ], [ %116, %bb15.i.i ]
  %lcmp.mod.not = icmp eq i64 %xtraiter, 0, !dbg !569937
  br i1 %lcmp.mod.not, label %bb33.i, label %bb15.i.i.epil, !dbg !569937

bb15.i.i.epil:                                    ; preds = %bb33.i.loopexit144.unr-lcssa
    #dbg_value(i64 %iter.sroa.0.012.i.i.unr, !569875, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !569928)
    #dbg_value(i64 %failed.sroa.0.011.i.i.unr, !569873, !DIExpression(), !569927)
    #dbg_value(i64 %iter.sroa.0.012.i.i.unr, !569912, !DIExpression(), !570143)
    #dbg_value(i64 %iter.sroa.0.012.i.i.unr, !569905, !DIExpression(), !569906)
    #dbg_value(i64 %iter.sroa.0.012.i.i.unr, !569922, !DIExpression(), !569923)
    #dbg_value(i64 %iter.sroa.0.012.i.i.unr, !569875, !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value, DW_OP_LLVM_fragment, 0, 64), !569928)
    #dbg_value(i64 %iter.sroa.0.012.i.i.unr, !569877, !DIExpression(), !570145)
    #dbg_value(i64 %iter.sroa.0.012.i.i.unr, !569899, !DIExpression(), !569900)
    #dbg_value(i64 %iter.sroa.0.012.i.i.unr, !570146, !DIExpression(), !570150)
    #dbg_value(ptr undef, !551471, !DIExpression(), !569894)
    #dbg_value(i64 %iter.sroa.0.012.i.i.unr, !551477, !DIExpression(), !569894)
    #dbg_value(ptr poison, !551549, !DIExpression(), !570152)
    #dbg_value(i64 %iter.sroa.0.012.i.i.unr, !551550, !DIExpression(), !570152)
    #dbg_value(i64 %iter.sroa.0.012.i.i.unr, !551542, !DIExpression(), !570154)
    #dbg_value(i64 %iter.sroa.0.012.i.i.unr, !551534, !DIExpression(), !570156)
    #dbg_value(ptr %column.val.i.i, !551541, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !570154)
    #dbg_value(ptr %column.val.i.i, !551535, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !570156)
    #dbg_value(i64 %len3.i.i.i, !551541, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !570154)
    #dbg_value(i64 %len3.i.i.i, !551535, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !570156)
  %_4.i.i.i.i.epil = getelementptr inbounds nuw i64, ptr %column.val.i.i, i64 %iter.sroa.0.012.i.i.unr, !dbg !570158
  %_0.i.i.i.i.epil = load i64, ptr %_4.i.i.i.i.epil, align 8, !dbg !570159, !noalias !570160, !noundef !23
    #dbg_value(ptr poison, !551549, !DIExpression(), !570164)
    #dbg_value(i64 %iter.sroa.0.012.i.i.unr, !551550, !DIExpression(), !570164)
    #dbg_value(i64 %iter.sroa.0.012.i.i.unr, !551542, !DIExpression(), !570166)
    #dbg_value(i64 %iter.sroa.0.012.i.i.unr, !551534, !DIExpression(), !570168)
    #dbg_value(ptr %column5.val.i.i, !551541, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !570166)
    #dbg_value(ptr %column5.val.i.i, !551535, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !570168)
    #dbg_value(i64 %len3.i.i.i, !551541, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !570166)
    #dbg_value(i64 %len3.i.i.i, !551535, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !570168)
  %_5.i3.i.i.i.epil = icmp ult i64 %iter.sroa.0.012.i.i.unr, %len3.i.i.i, !dbg !570170
  tail call void @llvm.assume(i1 %_5.i3.i.i.i.epil), !dbg !570171
  %_4.i4.i.i.i.epil = getelementptr inbounds nuw i64, ptr %column5.val.i.i, i64 %iter.sroa.0.012.i.i.unr, !dbg !570172
  %_0.i5.i.i.i.epil = load i64, ptr %_4.i4.i.i.i.epil, align 8, !dbg !570173, !noalias !570160, !noundef !23
    #dbg_value(i64 %_0.i.i.i.i.epil, !569879, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !570174)
    #dbg_value(i64 %_0.i5.i.i.i.epil, !569879, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !570174)
    #dbg_value(i64 %_0.i.i.i.i.epil, !570175, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !570183)
    #dbg_value(i64 %_0.i5.i.i.i.epil, !570175, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !570183)
    #dbg_value(ptr poison, !569758, !DIExpression(), !570185)
    #dbg_value(ptr poison, !569759, !DIExpression(), !570185)
    #dbg_value(i64 %_0.i.i.i.i.epil, !569760, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !570185)
    #dbg_value(i64 %_0.i5.i.i.i.epil, !569760, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !570185)
    #dbg_value(i64 %_0.i.i.i.i.epil, !569756, !DIExpression(), !570187)
    #dbg_value(i64 %_0.i5.i.i.i.epil, !569757, !DIExpression(), !570187)
    #dbg_value(i64 %_0.i.i.i.i.epil, !569747, !DIExpression(), !570188)
    #dbg_value(i64 %_0.i5.i.i.i.epil, !569748, !DIExpression(), !570188)
    #dbg_value(i64 %_0.i.i.i.i.epil, !569740, !DIExpression(), !570190)
    #dbg_value(i64 %_0.i.i.i.i.epil, !569735, !DIExpression(), !570192)
    #dbg_value(i64 %_0.i5.i.i.i.epil, !569741, !DIExpression(), !570190)
    #dbg_value(i64 %_0.i5.i.i.i.epil, !569736, !DIExpression(), !570192)
  %_0.i3.i.i.i.i.epil = mul i64 %_0.i5.i.i.i.epil, %_0.i.i.i.i.epil, !dbg !570194
    #dbg_value(i64 %_0.i.i.i.i.epil, !569766, !DIExpression(), !570195)
    #dbg_value(i64 %_0.i.i.i.i.epil, !569768, !DIExpression(), !570197)
    #dbg_value(i64 %_0.i5.i.i.i.epil, !569767, !DIExpression(), !570195)
    #dbg_value(i64 %_0.i5.i.i.i.epil, !569769, !DIExpression(), !570197)
  %_5.i.i.i.i.i.epil = zext i64 %_0.i.i.i.i.epil to i128, !dbg !570198
  %_6.i.i.i.i.i.epil = zext i64 %_0.i5.i.i.i.epil to i128, !dbg !570199
  %_4.i1.i.i.i.i.epil = mul nuw i128 %_6.i.i.i.i.i.epil, %_5.i.i.i.i.i.epil, !dbg !570200
  %_3.i2.i.i.i.i.epil = lshr i128 %_4.i1.i.i.i.i.epil, 64, !dbg !570201
  %_0.i.i.i.i.i.epil = trunc nuw i128 %_3.i2.i.i.i.i.epil to i64, !dbg !570202
    #dbg_value(i64 poison, !569881, !DIExpression(), !570203)
    #dbg_value(i64 %_0.i.i.i.i.i.epil, !569883, !DIExpression(), !570203)
    #dbg_value(ptr undef, !564034, !DIExpression(), !569892)
    #dbg_value(i64 %_0.i.i.i.i.i.epil, !564040, !DIExpression(), !569892)
  %118 = or i64 %failed.sroa.0.011.i.i.unr, %_0.i.i.i.i.i.epil, !dbg !570204
    #dbg_value(i64 %118, !569873, !DIExpression(), !569927)
    #dbg_value(i64 %_0.i3.i.i.i.i.epil, !569881, !DIExpression(), !570203)
    #dbg_value(ptr %_4.sroa.10.0.i.i, !570149, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !570150)
    #dbg_value(i64 %index.i, !570149, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !570150)
  %self4.i.i.epil = getelementptr inbounds nuw i64, ptr %_4.sroa.10.0.i.i, i64 %iter.sroa.0.012.i.i.unr, !dbg !570205
    #dbg_value(ptr %self4.i.i.epil, !570206, !DIExpression(), !570210)
    #dbg_value(i64 %_0.i3.i.i.i.i.epil, !570209, !DIExpression(), !570210)
  store i64 %_0.i3.i.i.i.i.epil, ptr %self4.i.i.epil, align 8, !dbg !570212, !alias.scope !569888, !noalias !570213
    #dbg_value(ptr undef, !569916, !DIExpression(), !569929)
    #dbg_value(ptr undef, !569911, !DIExpression(), !569930)
    #dbg_value(ptr undef, !569931, !DIExpression(), !569935)
    #dbg_value(ptr poison, !569934, !DIExpression(), !569935)
  br label %bb33.i, !dbg !570227


```
