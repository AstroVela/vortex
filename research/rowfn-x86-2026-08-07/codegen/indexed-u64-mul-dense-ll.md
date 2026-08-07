<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# `indexed-u64-mul-dense.ll`

```ll
bb15.i.i:                                         ; preds = %bb15.i.i, %bb15.i.i.preheader.new
  %iter.sroa.0.012.i.i = phi i64 [ 0, %bb15.i.i.preheader.new ], [ %_36.i.i.1, %bb15.i.i ]
  %failed.sroa.0.011.i.i = phi i64 [ 0, %bb15.i.i.preheader.new ], [ %116, %bb15.i.i ]
  %niter = phi i64 [ 0, %bb15.i.i.preheader.new ], [ %niter.next.1, %bb15.i.i ]
    #dbg_value(i64 %iter.sroa.0.012.i.i, !580573, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !580626)
    #dbg_value(i64 %failed.sroa.0.011.i.i, !580571, !DIExpression(), !580625)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !580610, !DIExpression(), !580841)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !580603, !DIExpression(), !580604)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !580620, !DIExpression(), !580621)
  %_36.i.i = or disjoint i64 %iter.sroa.0.012.i.i, 1, !dbg !580842
    #dbg_value(i64 %_36.i.i, !580573, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !580626)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !580575, !DIExpression(), !580843)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !580597, !DIExpression(), !580598)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !580844, !DIExpression(), !580848)
    #dbg_value(ptr undef, !563670, !DIExpression(), !580592)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !563676, !DIExpression(), !580592)
    #dbg_value(ptr poison, !563748, !DIExpression(), !580850)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !563749, !DIExpression(), !580850)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !563741, !DIExpression(), !580852)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !563733, !DIExpression(), !580854)
    #dbg_value(ptr %column.val.i.i, !563740, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !580852)
    #dbg_value(ptr %column.val.i.i, !563734, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !580854)
    #dbg_value(i64 %len3.i.i.i, !563740, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !580852)
    #dbg_value(i64 %len3.i.i.i, !563734, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !580854)
  %_4.i.i.i.i = getelementptr inbounds nuw i64, ptr %column.val.i.i, i64 %iter.sroa.0.012.i.i, !dbg !580856
  %_0.i.i.i.i = load i64, ptr %_4.i.i.i.i, align 8, !dbg !580857, !noalias !580858, !noundef !23
    #dbg_value(ptr poison, !563748, !DIExpression(), !580862)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !563749, !DIExpression(), !580862)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !563741, !DIExpression(), !580864)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !563733, !DIExpression(), !580866)
    #dbg_value(ptr %column5.val.i.i, !563740, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !580864)
    #dbg_value(ptr %column5.val.i.i, !563734, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !580866)
    #dbg_value(i64 %len3.i.i.i, !563740, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !580864)
    #dbg_value(i64 %len3.i.i.i, !563734, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !580866)
  %_5.i3.i.i.i = icmp ult i64 %iter.sroa.0.012.i.i, %len3.i.i.i, !dbg !580868
  tail call void @llvm.assume(i1 %_5.i3.i.i.i), !dbg !580869
  %_4.i4.i.i.i = getelementptr inbounds nuw i64, ptr %column5.val.i.i, i64 %iter.sroa.0.012.i.i, !dbg !580870
  %_0.i5.i.i.i = load i64, ptr %_4.i4.i.i.i, align 8, !dbg !580871, !noalias !580858, !noundef !23
    #dbg_value(i64 %_0.i.i.i.i, !580577, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !580872)
    #dbg_value(i64 %_0.i5.i.i.i, !580577, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !580872)
    #dbg_value(i64 %_0.i.i.i.i, !580873, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !580881)
    #dbg_value(i64 %_0.i5.i.i.i, !580873, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !580881)
    #dbg_value(ptr poison, !580456, !DIExpression(), !580883)
    #dbg_value(ptr poison, !580457, !DIExpression(), !580883)
    #dbg_value(i64 %_0.i.i.i.i, !580458, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !580883)
    #dbg_value(i64 %_0.i5.i.i.i, !580458, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !580883)
    #dbg_value(i64 %_0.i.i.i.i, !580454, !DIExpression(), !580885)
    #dbg_value(i64 %_0.i5.i.i.i, !580455, !DIExpression(), !580885)
    #dbg_value(i64 %_0.i.i.i.i, !580445, !DIExpression(), !580886)
    #dbg_value(i64 %_0.i5.i.i.i, !580446, !DIExpression(), !580886)
    #dbg_value(i64 %_0.i.i.i.i, !580438, !DIExpression(), !580888)
    #dbg_value(i64 %_0.i.i.i.i, !580433, !DIExpression(), !580890)
    #dbg_value(i64 %_0.i5.i.i.i, !580439, !DIExpression(), !580888)
    #dbg_value(i64 %_0.i5.i.i.i, !580434, !DIExpression(), !580890)
  %_0.i3.i.i.i.i = mul i64 %_0.i5.i.i.i, %_0.i.i.i.i, !dbg !580892
    #dbg_value(i64 %_0.i.i.i.i, !580464, !DIExpression(), !580893)
    #dbg_value(i64 %_0.i.i.i.i, !580466, !DIExpression(), !580895)
    #dbg_value(i64 %_0.i5.i.i.i, !580465, !DIExpression(), !580893)
    #dbg_value(i64 %_0.i5.i.i.i, !580467, !DIExpression(), !580895)
  %_5.i.i.i.i.i = zext i64 %_0.i.i.i.i to i128, !dbg !580896
  %_6.i.i.i.i.i = zext i64 %_0.i5.i.i.i to i128, !dbg !580897
  %_4.i1.i.i.i.i = mul nuw i128 %_6.i.i.i.i.i, %_5.i.i.i.i.i, !dbg !580898
  %_3.i2.i.i.i.i = lshr i128 %_4.i1.i.i.i.i, 64, !dbg !580899
  %_0.i.i.i.i.i = trunc nuw i128 %_3.i2.i.i.i.i to i64, !dbg !580900
    #dbg_value(i64 poison, !580579, !DIExpression(), !580901)
    #dbg_value(i64 %_0.i.i.i.i.i, !580581, !DIExpression(), !580901)
    #dbg_value(ptr undef, !576390, !DIExpression(), !580590)
    #dbg_value(i64 %_0.i.i.i.i.i, !576396, !DIExpression(), !580590)
  %115 = or i64 %failed.sroa.0.011.i.i, %_0.i.i.i.i.i, !dbg !580902
    #dbg_value(i64 %115, !580571, !DIExpression(), !580625)
    #dbg_value(i64 %_0.i3.i.i.i.i, !580579, !DIExpression(), !580901)
    #dbg_value(ptr %_4.sroa.10.0.i.i, !580847, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !580848)
    #dbg_value(i64 %index.i, !580847, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !580848)
  %self4.i.i = getelementptr inbounds nuw i64, ptr %_4.sroa.10.0.i.i, i64 %iter.sroa.0.012.i.i, !dbg !580903
    #dbg_value(ptr %self4.i.i, !580904, !DIExpression(), !580908)
    #dbg_value(i64 %_0.i3.i.i.i.i, !580907, !DIExpression(), !580908)
  store i64 %_0.i3.i.i.i.i, ptr %self4.i.i, align 8, !dbg !580910, !alias.scope !580586, !noalias !580911
    #dbg_value(ptr undef, !580614, !DIExpression(), !580627)
    #dbg_value(ptr undef, !580609, !DIExpression(), !580628)
    #dbg_value(ptr undef, !580629, !DIExpression(), !580633)
    #dbg_value(ptr poison, !580632, !DIExpression(), !580633)
    #dbg_value(i64 %_36.i.i, !580610, !DIExpression(), !580841)
    #dbg_value(i64 %_36.i.i, !580603, !DIExpression(), !580604)
    #dbg_value(i64 %_36.i.i, !580620, !DIExpression(), !580621)
  %_36.i.i.1 = add nuw i64 %iter.sroa.0.012.i.i, 2, !dbg !580842
    #dbg_value(i64 %_36.i.i.1, !580573, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !580626)
    #dbg_value(i64 %_36.i.i, !580575, !DIExpression(), !580843)
    #dbg_value(i64 %_36.i.i, !580597, !DIExpression(), !580598)
    #dbg_value(i64 %_36.i.i, !580844, !DIExpression(), !580848)
    #dbg_value(i64 %_36.i.i, !563676, !DIExpression(), !580592)
    #dbg_value(i64 %_36.i.i, !563749, !DIExpression(), !580850)
    #dbg_value(i64 %_36.i.i, !563741, !DIExpression(), !580852)
    #dbg_value(i64 %_36.i.i, !563733, !DIExpression(), !580854)
    #dbg_value(ptr %column.val.i.i, !563740, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !580852)
    #dbg_value(ptr %column.val.i.i, !563734, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !580854)
    #dbg_value(i64 %len3.i.i.i, !563740, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !580852)
    #dbg_value(i64 %len3.i.i.i, !563734, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !580854)
  %_4.i.i.i.i.1 = getelementptr inbounds nuw i64, ptr %column.val.i.i, i64 %_36.i.i, !dbg !580856
  %_0.i.i.i.i.1 = load i64, ptr %_4.i.i.i.i.1, align 8, !dbg !580857, !noalias !580858, !noundef !23
    #dbg_value(i64 %_36.i.i, !563749, !DIExpression(), !580862)
    #dbg_value(i64 %_36.i.i, !563741, !DIExpression(), !580864)
    #dbg_value(i64 %_36.i.i, !563733, !DIExpression(), !580866)
    #dbg_value(ptr %column5.val.i.i, !563740, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !580864)
    #dbg_value(ptr %column5.val.i.i, !563734, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !580866)
    #dbg_value(i64 %len3.i.i.i, !563740, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !580864)
    #dbg_value(i64 %len3.i.i.i, !563734, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !580866)
  %_5.i3.i.i.i.1 = icmp ult i64 %_36.i.i, %len3.i.i.i, !dbg !580868
  tail call void @llvm.assume(i1 %_5.i3.i.i.i.1), !dbg !580869
  %_4.i4.i.i.i.1 = getelementptr inbounds nuw i64, ptr %column5.val.i.i, i64 %_36.i.i, !dbg !580870
  %_0.i5.i.i.i.1 = load i64, ptr %_4.i4.i.i.i.1, align 8, !dbg !580871, !noalias !580858, !noundef !23
    #dbg_value(i64 %_0.i.i.i.i.1, !580577, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !580872)
    #dbg_value(i64 %_0.i5.i.i.i.1, !580577, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !580872)
    #dbg_value(i64 %_0.i.i.i.i.1, !580873, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !580881)
    #dbg_value(i64 %_0.i5.i.i.i.1, !580873, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !580881)
    #dbg_value(i64 %_0.i.i.i.i.1, !580458, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !580883)
    #dbg_value(i64 %_0.i5.i.i.i.1, !580458, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !580883)
    #dbg_value(i64 %_0.i.i.i.i.1, !580454, !DIExpression(), !580885)
    #dbg_value(i64 %_0.i5.i.i.i.1, !580455, !DIExpression(), !580885)
    #dbg_value(i64 %_0.i.i.i.i.1, !580445, !DIExpression(), !580886)
    #dbg_value(i64 %_0.i5.i.i.i.1, !580446, !DIExpression(), !580886)
    #dbg_value(i64 %_0.i.i.i.i.1, !580438, !DIExpression(), !580888)
    #dbg_value(i64 %_0.i.i.i.i.1, !580433, !DIExpression(), !580890)
    #dbg_value(i64 %_0.i5.i.i.i.1, !580439, !DIExpression(), !580888)
    #dbg_value(i64 %_0.i5.i.i.i.1, !580434, !DIExpression(), !580890)
  %_0.i3.i.i.i.i.1 = mul i64 %_0.i5.i.i.i.1, %_0.i.i.i.i.1, !dbg !580892
    #dbg_value(i64 %_0.i.i.i.i.1, !580464, !DIExpression(), !580893)
    #dbg_value(i64 %_0.i.i.i.i.1, !580466, !DIExpression(), !580895)
    #dbg_value(i64 %_0.i5.i.i.i.1, !580465, !DIExpression(), !580893)
    #dbg_value(i64 %_0.i5.i.i.i.1, !580467, !DIExpression(), !580895)
  %_5.i.i.i.i.i.1 = zext i64 %_0.i.i.i.i.1 to i128, !dbg !580896
  %_6.i.i.i.i.i.1 = zext i64 %_0.i5.i.i.i.1 to i128, !dbg !580897
  %_4.i1.i.i.i.i.1 = mul nuw i128 %_6.i.i.i.i.i.1, %_5.i.i.i.i.i.1, !dbg !580898
  %_3.i2.i.i.i.i.1 = lshr i128 %_4.i1.i.i.i.i.1, 64, !dbg !580899
  %_0.i.i.i.i.i.1 = trunc nuw i128 %_3.i2.i.i.i.i.1 to i64, !dbg !580900
    #dbg_value(i64 poison, !580579, !DIExpression(), !580901)
    #dbg_value(i64 %_0.i.i.i.i.i.1, !580581, !DIExpression(), !580901)
    #dbg_value(i64 %_0.i.i.i.i.i.1, !576396, !DIExpression(), !580590)
  %116 = or i64 %115, %_0.i.i.i.i.i.1, !dbg !580902
    #dbg_value(i64 %116, !580571, !DIExpression(), !580625)
    #dbg_value(i64 %_0.i3.i.i.i.i.1, !580579, !DIExpression(), !580901)
    #dbg_value(ptr %_4.sroa.10.0.i.i, !580847, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !580848)
    #dbg_value(i64 %index.i, !580847, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !580848)
  %self4.i.i.1 = getelementptr inbounds nuw i64, ptr %_4.sroa.10.0.i.i, i64 %_36.i.i, !dbg !580903
    #dbg_value(ptr %self4.i.i.1, !580904, !DIExpression(), !580908)
    #dbg_value(i64 %_0.i3.i.i.i.i.1, !580907, !DIExpression(), !580908)
  store i64 %_0.i3.i.i.i.i.1, ptr %self4.i.i.1, align 8, !dbg !580910, !alias.scope !580586, !noalias !580911
  %niter.next.1 = add i64 %niter, 2, !dbg !580635
  %niter.ncmp.1 = icmp eq i64 %niter.next.1, %unroll_iter, !dbg !580635
  br i1 %niter.ncmp.1, label %bb33.i.loopexit144.unr-lcssa, label %bb15.i.i, !dbg !580635

bb33.thread.i:                                    ; preds = %bb26.preheader.i.thread, %bb11.i, %bb26.preheader.i
    #dbg_value(i64 0, !580128, !DIExpression(), !580912)
    #dbg_value(i64 %index.i, !580121, !DIExpression(DW_OP_LLVM_fragment, 128, 64), !580349)
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %value.i.i), !dbg !580913, !noalias !580287
    #dbg_value(i64 0, !580099, !DIExpression(), !580915)
    #dbg_declare(ptr poison, !580103, !DIExpression(), !580916)
    #dbg_declare(ptr %value.i.i, !580917, !DIExpression(), !580920)
    #dbg_value(ptr undef, !577131, !DIExpression(), !580923)
    #dbg_value(ptr undef, !577132, !DIExpression(), !580923)
  br label %bb36.i, !dbg !580924

bb33.i.loopexit.unr-lcssa:                        ; preds = %bb27.us.i.us, %bb27.us.i.us.preheader
  %.lcssa.ph = phi i64 [ poison, %bb27.us.i.us.preheader ], [ %88, %bb27.us.i.us ]
  %iter.sroa.0.046.us.i.us.unr = phi i64 [ 0, %bb27.us.i.us.preheader ], [ %_157.us.i.us, %bb27.us.i.us ]
  %accumulated.sroa.0.045.us.i.us.unr = phi i64 [ 0, %bb27.us.i.us.preheader ], [ %88, %bb27.us.i.us ]
  %lcmp.mod148.not = icmp eq i64 %xtraiter147, 0, !dbg !580418
  br i1 %lcmp.mod148.not, label %bb33.i, label %bb27.us.i.us.epil, !dbg !580418

bb27.us.i.us.epil:                                ; preds = %bb33.i.loopexit.unr-lcssa
    #dbg_value(i64 %iter.sroa.0.046.us.i.us.unr, !580147, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !580416)
    #dbg_value(i64 %accumulated.sroa.0.045.us.i.us.unr, !580145, !DIExpression(), !580415)
    #dbg_value(i64 %iter.sroa.0.046.us.i.us.unr, !580149, !DIExpression(), !580491)
    #dbg_value(ptr %columns.i, !563475, !DIExpression(), !580492)
    #dbg_value(i64 %iter.sroa.0.046.us.i.us.unr, !563476, !DIExpression(), !580492)
    #dbg_value(ptr %columns.i, !563465, !DIExpression(), !580493)
    #dbg_value(i64 %iter.sroa.0.046.us.i.us.unr, !563466, !DIExpression(), !580493)
    #dbg_value(ptr %columns.i, !563336, !DIExpression(DW_OP_plus_uconst, 8, DW_OP_stack_value), !580496)
    #dbg_value(i64 0, !563341, !DIExpression(), !580494)
    #dbg_value(ptr %columns.i, !563468, !DIExpression(DW_OP_plus_uconst, 8, DW_OP_stack_value), !580525)
    #dbg_value(ptr %columns.i, !563336, !DIExpression(DW_OP_plus_uconst, 8, DW_OP_stack_value), !580494)
  %_0.sroa.0.0.i.i.us.i.us.epil = load i64, ptr %data.i.i.i.us.i, align 8, !dbg !580423, !noalias !580207, !noundef !23
    #dbg_value(ptr %14, !563465, !DIExpression(), !580498)
    #dbg_value(i64 %iter.sroa.0.046.us.i.us.unr, !563466, !DIExpression(), !580498)
    #dbg_value(ptr %14, !563336, !DIExpression(DW_OP_plus_uconst, 8, DW_OP_stack_value), !580499)
    #dbg_value(ptr %14, !563336, !DIExpression(DW_OP_plus_uconst, 8, DW_OP_stack_value), !580501)
    #dbg_value(ptr %14, !563468, !DIExpression(DW_OP_plus_uconst, 8, DW_OP_stack_value), !580502)
    #dbg_value(i64 0, !563341, !DIExpression(), !580499)
  %_0.sroa.0.0.i9.i.us.i.us.epil = load i64, ptr %data.i6.i7.i.us.i, align 8, !dbg !580428, !noalias !580207, !noundef !23
    #dbg_value(ptr poison, !580456, !DIExpression(), !580503)
    #dbg_value(ptr poison, !580457, !DIExpression(), !580503)
    #dbg_value(i64 %_0.sroa.0.0.i.i.us.i.us.epil, !580458, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !580503)
    #dbg_value(i64 %_0.sroa.0.0.i9.i.us.i.us.epil, !580458, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !580503)
    #dbg_value(i64 %_0.sroa.0.0.i.i.us.i.us.epil, !580454, !DIExpression(), !580504)
    #dbg_value(i64 %_0.sroa.0.0.i9.i.us.i.us.epil, !580455, !DIExpression(), !580504)
    #dbg_value(i64 %_0.sroa.0.0.i.i.us.i.us.epil, !580445, !DIExpression(), !580505)
    #dbg_value(i64 %_0.sroa.0.0.i9.i.us.i.us.epil, !580446, !DIExpression(), !580505)
    #dbg_value(i64 %_0.sroa.0.0.i.i.us.i.us.epil, !580438, !DIExpression(), !580506)
    #dbg_value(i64 %_0.sroa.0.0.i.i.us.i.us.epil, !580433, !DIExpression(), !580507)
    #dbg_value(i64 %_0.sroa.0.0.i9.i.us.i.us.epil, !580439, !DIExpression(), !580506)
    #dbg_value(i64 %_0.sroa.0.0.i9.i.us.i.us.epil, !580434, !DIExpression(), !580507)
  %_0.i3.i.us.i.us.epil = mul i64 %_0.sroa.0.0.i9.i.us.i.us.epil, %_0.sroa.0.0.i.i.us.i.us.epil, !dbg !580430
    #dbg_value(i64 %_0.sroa.0.0.i.i.us.i.us.epil, !580464, !DIExpression(), !580508)
    #dbg_value(i64 %_0.sroa.0.0.i.i.us.i.us.epil, !580466, !DIExpression(), !580509)
    #dbg_value(i64 %_0.sroa.0.0.i9.i.us.i.us.epil, !580465, !DIExpression(), !580508)
    #dbg_value(i64 %_0.sroa.0.0.i9.i.us.i.us.epil, !580467, !DIExpression(), !580509)
  %_5.i.i.us.i.us.epil = zext i64 %_0.sroa.0.0.i.i.us.i.us.epil to i128, !dbg !580460
  %_6.i.i.us.i.us.epil = zext i64 %_0.sroa.0.0.i9.i.us.i.us.epil to i128, !dbg !580469
  %_4.i1.i.us.i.us.epil = mul nuw i128 %_6.i.i.us.i.us.epil, %_5.i.i.us.i.us.epil, !dbg !580470
  %_3.i2.i.us.i.us.epil = lshr i128 %_4.i1.i.us.i.us.epil, 64, !dbg !580471
  %_0.i.i159.us.i.us.epil = trunc nuw i128 %_3.i2.i.us.i.us.epil to i64, !dbg !580472
    #dbg_value(i64 %_0.i3.i.us.i.us.epil, !580151, !DIExpression(), !580510)
    #dbg_value(i64 %_0.i.i159.us.i.us.epil, !580153, !DIExpression(), !580510)
    #dbg_value(ptr undef, !576390, !DIExpression(), !580180)
    #dbg_value(i64 %_0.i.i159.us.i.us.epil, !576396, !DIExpression(), !580180)
  %117 = or i64 %accumulated.sroa.0.045.us.i.us.unr, %_0.i.i159.us.i.us.epil, !dbg !580473
    #dbg_value(i64 %117, !580145, !DIExpression(), !580415)
  %self34.us.i.us.epil = getelementptr inbounds nuw i64, ptr %_4.sroa.10.0.i.i, i64 %iter.sroa.0.046.us.i.us.unr, !dbg !580474
    #dbg_value(ptr %self34.us.i.us.epil, !580478, !DIExpression(), !580511)
    #dbg_value(i64 %_0.i3.i.us.i.us.epil, !580479, !DIExpression(), !580511)
  store i64 %_0.i3.i.us.i.us.epil, ptr %self34.us.i.us.epil, align 8, !dbg !580475, !noalias !580207
    #dbg_value(i64 poison, !580147, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !580416)
    #dbg_value(ptr undef, !580170, !DIExpression(), !580173)
    #dbg_value(ptr undef, !580161, !DIExpression(), !580166)
    #dbg_value(ptr undef, !580174, !DIExpression(), !580178)
    #dbg_value(ptr poison, !580177, !DIExpression(), !580178)
    #dbg_value(i64 poison, !580147, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !580416)
  br label %bb33.i, !dbg !580925

bb33.i.loopexit144.unr-lcssa:                     ; preds = %bb15.i.i, %bb15.i.i.preheader
  %.lcssa145.ph = phi i64 [ poison, %bb15.i.i.preheader ], [ %116, %bb15.i.i ]
  %iter.sroa.0.012.i.i.unr = phi i64 [ 0, %bb15.i.i.preheader ], [ %_36.i.i.1, %bb15.i.i ]
  %failed.sroa.0.011.i.i.unr = phi i64 [ 0, %bb15.i.i.preheader ], [ %116, %bb15.i.i ]
  %lcmp.mod.not = icmp eq i64 %xtraiter, 0, !dbg !580635
  br i1 %lcmp.mod.not, label %bb33.i, label %bb15.i.i.epil, !dbg !580635

bb15.i.i.epil:                                    ; preds = %bb33.i.loopexit144.unr-lcssa
    #dbg_value(i64 %iter.sroa.0.012.i.i.unr, !580573, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !580626)
    #dbg_value(i64 %failed.sroa.0.011.i.i.unr, !580571, !DIExpression(), !580625)
    #dbg_value(i64 %iter.sroa.0.012.i.i.unr, !580610, !DIExpression(), !580841)
    #dbg_value(i64 %iter.sroa.0.012.i.i.unr, !580603, !DIExpression(), !580604)
    #dbg_value(i64 %iter.sroa.0.012.i.i.unr, !580620, !DIExpression(), !580621)
    #dbg_value(i64 %iter.sroa.0.012.i.i.unr, !580573, !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value, DW_OP_LLVM_fragment, 0, 64), !580626)
    #dbg_value(i64 %iter.sroa.0.012.i.i.unr, !580575, !DIExpression(), !580843)
    #dbg_value(i64 %iter.sroa.0.012.i.i.unr, !580597, !DIExpression(), !580598)
    #dbg_value(i64 %iter.sroa.0.012.i.i.unr, !580844, !DIExpression(), !580848)
    #dbg_value(ptr undef, !563670, !DIExpression(), !580592)
    #dbg_value(i64 %iter.sroa.0.012.i.i.unr, !563676, !DIExpression(), !580592)
    #dbg_value(ptr poison, !563748, !DIExpression(), !580850)
    #dbg_value(i64 %iter.sroa.0.012.i.i.unr, !563749, !DIExpression(), !580850)
    #dbg_value(i64 %iter.sroa.0.012.i.i.unr, !563741, !DIExpression(), !580852)
    #dbg_value(i64 %iter.sroa.0.012.i.i.unr, !563733, !DIExpression(), !580854)
    #dbg_value(ptr %column.val.i.i, !563740, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !580852)
    #dbg_value(ptr %column.val.i.i, !563734, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !580854)
    #dbg_value(i64 %len3.i.i.i, !563740, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !580852)
    #dbg_value(i64 %len3.i.i.i, !563734, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !580854)
  %_4.i.i.i.i.epil = getelementptr inbounds nuw i64, ptr %column.val.i.i, i64 %iter.sroa.0.012.i.i.unr, !dbg !580856
  %_0.i.i.i.i.epil = load i64, ptr %_4.i.i.i.i.epil, align 8, !dbg !580857, !noalias !580858, !noundef !23
    #dbg_value(ptr poison, !563748, !DIExpression(), !580862)
    #dbg_value(i64 %iter.sroa.0.012.i.i.unr, !563749, !DIExpression(), !580862)
    #dbg_value(i64 %iter.sroa.0.012.i.i.unr, !563741, !DIExpression(), !580864)
    #dbg_value(i64 %iter.sroa.0.012.i.i.unr, !563733, !DIExpression(), !580866)
    #dbg_value(ptr %column5.val.i.i, !563740, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !580864)
    #dbg_value(ptr %column5.val.i.i, !563734, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !580866)
    #dbg_value(i64 %len3.i.i.i, !563740, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !580864)
    #dbg_value(i64 %len3.i.i.i, !563734, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !580866)
  %_5.i3.i.i.i.epil = icmp ult i64 %iter.sroa.0.012.i.i.unr, %len3.i.i.i, !dbg !580868
  tail call void @llvm.assume(i1 %_5.i3.i.i.i.epil), !dbg !580869
  %_4.i4.i.i.i.epil = getelementptr inbounds nuw i64, ptr %column5.val.i.i, i64 %iter.sroa.0.012.i.i.unr, !dbg !580870
  %_0.i5.i.i.i.epil = load i64, ptr %_4.i4.i.i.i.epil, align 8, !dbg !580871, !noalias !580858, !noundef !23
    #dbg_value(i64 %_0.i.i.i.i.epil, !580577, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !580872)
    #dbg_value(i64 %_0.i5.i.i.i.epil, !580577, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !580872)
    #dbg_value(i64 %_0.i.i.i.i.epil, !580873, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !580881)
    #dbg_value(i64 %_0.i5.i.i.i.epil, !580873, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !580881)
    #dbg_value(ptr poison, !580456, !DIExpression(), !580883)
    #dbg_value(ptr poison, !580457, !DIExpression(), !580883)
    #dbg_value(i64 %_0.i.i.i.i.epil, !580458, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !580883)
    #dbg_value(i64 %_0.i5.i.i.i.epil, !580458, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !580883)
    #dbg_value(i64 %_0.i.i.i.i.epil, !580454, !DIExpression(), !580885)
    #dbg_value(i64 %_0.i5.i.i.i.epil, !580455, !DIExpression(), !580885)
    #dbg_value(i64 %_0.i.i.i.i.epil, !580445, !DIExpression(), !580886)
    #dbg_value(i64 %_0.i5.i.i.i.epil, !580446, !DIExpression(), !580886)
    #dbg_value(i64 %_0.i.i.i.i.epil, !580438, !DIExpression(), !580888)
    #dbg_value(i64 %_0.i.i.i.i.epil, !580433, !DIExpression(), !580890)
    #dbg_value(i64 %_0.i5.i.i.i.epil, !580439, !DIExpression(), !580888)
    #dbg_value(i64 %_0.i5.i.i.i.epil, !580434, !DIExpression(), !580890)
  %_0.i3.i.i.i.i.epil = mul i64 %_0.i5.i.i.i.epil, %_0.i.i.i.i.epil, !dbg !580892
    #dbg_value(i64 %_0.i.i.i.i.epil, !580464, !DIExpression(), !580893)
    #dbg_value(i64 %_0.i.i.i.i.epil, !580466, !DIExpression(), !580895)
    #dbg_value(i64 %_0.i5.i.i.i.epil, !580465, !DIExpression(), !580893)
    #dbg_value(i64 %_0.i5.i.i.i.epil, !580467, !DIExpression(), !580895)
  %_5.i.i.i.i.i.epil = zext i64 %_0.i.i.i.i.epil to i128, !dbg !580896
  %_6.i.i.i.i.i.epil = zext i64 %_0.i5.i.i.i.epil to i128, !dbg !580897
  %_4.i1.i.i.i.i.epil = mul nuw i128 %_6.i.i.i.i.i.epil, %_5.i.i.i.i.i.epil, !dbg !580898
  %_3.i2.i.i.i.i.epil = lshr i128 %_4.i1.i.i.i.i.epil, 64, !dbg !580899
  %_0.i.i.i.i.i.epil = trunc nuw i128 %_3.i2.i.i.i.i.epil to i64, !dbg !580900
    #dbg_value(i64 poison, !580579, !DIExpression(), !580901)
    #dbg_value(i64 %_0.i.i.i.i.i.epil, !580581, !DIExpression(), !580901)
    #dbg_value(ptr undef, !576390, !DIExpression(), !580590)
    #dbg_value(i64 %_0.i.i.i.i.i.epil, !576396, !DIExpression(), !580590)
  %118 = or i64 %failed.sroa.0.011.i.i.unr, %_0.i.i.i.i.i.epil, !dbg !580902
    #dbg_value(i64 %118, !580571, !DIExpression(), !580625)
    #dbg_value(i64 %_0.i3.i.i.i.i.epil, !580579, !DIExpression(), !580901)
    #dbg_value(ptr %_4.sroa.10.0.i.i, !580847, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !580848)
    #dbg_value(i64 %index.i, !580847, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !580848)
  %self4.i.i.epil = getelementptr inbounds nuw i64, ptr %_4.sroa.10.0.i.i, i64 %iter.sroa.0.012.i.i.unr, !dbg !580903
    #dbg_value(ptr %self4.i.i.epil, !580904, !DIExpression(), !580908)
    #dbg_value(i64 %_0.i3.i.i.i.i.epil, !580907, !DIExpression(), !580908)
  store i64 %_0.i3.i.i.i.i.epil, ptr %self4.i.i.epil, align 8, !dbg !580910, !alias.scope !580586, !noalias !580911
    #dbg_value(ptr undef, !580614, !DIExpression(), !580627)
    #dbg_value(ptr undef, !580609, !DIExpression(), !580628)
    #dbg_value(ptr undef, !580629, !DIExpression(), !580633)
    #dbg_value(ptr poison, !580632, !DIExpression(), !580633)
  br label %bb33.i, !dbg !580925


```
