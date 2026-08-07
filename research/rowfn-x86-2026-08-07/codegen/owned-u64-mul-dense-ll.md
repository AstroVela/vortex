<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# `owned-u64-mul-dense.ll`

```ll
bb18.i:                                           ; preds = %bb18.i, %bb18.lr.ph.i.new
  %_15552.i = phi i64 [ 1, %bb18.lr.ph.i.new ], [ %_155.i.1, %bb18.i ]
  %iter.sroa.0.051.i = phi i64 [ 0, %bb18.lr.ph.i.new ], [ %_155.i, %bb18.i ]
  %failed.sroa.0.050.i = phi i64 [ 0, %bb18.lr.ph.i.new ], [ %120, %bb18.i ]
  %niter = phi i64 [ 0, %bb18.lr.ph.i.new ], [ %niter.next.1, %bb18.i ]
    #dbg_value(i64 %iter.sroa.0.051.i, !576964, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !577378)
    #dbg_value(i64 %failed.sroa.0.050.i, !576953, !DIExpression(), !577231)
    #dbg_value(i64 %iter.sroa.0.051.i, !576966, !DIExpression(), !577601)
    #dbg_value(ptr undef, !561311, !DIExpression(), !577032)
    #dbg_value(i64 %iter.sroa.0.051.i, !561317, !DIExpression(), !577032)
    #dbg_value(ptr poison, !561836, !DIExpression(), !577602)
    #dbg_value(i64 %iter.sroa.0.051.i, !561837, !DIExpression(), !577602)
    #dbg_value(ptr poison, !561836, !DIExpression(), !577604)
    #dbg_value(i64 %iter.sroa.0.051.i, !561837, !DIExpression(), !577604)
  %115 = getelementptr inbounds nuw i64, ptr %column.val.i.i, i64 %iter.sroa.0.051.i, !dbg !577606
  %_0.i.i92.i = load i64, ptr %115, align 8, !dbg !577606, !noalias !577607, !noundef !23
  %116 = getelementptr inbounds nuw i64, ptr %column5.val.i.i, i64 %iter.sroa.0.051.i, !dbg !577610
  %_0.i5.i.i = load i64, ptr %116, align 8, !dbg !577610, !noalias !577607, !noundef !23
    #dbg_value(ptr poison, !577301, !DIExpression(), !577611)
    #dbg_value(ptr poison, !577302, !DIExpression(), !577611)
    #dbg_value(i64 %_0.i.i92.i, !577303, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !577611)
    #dbg_value(i64 %_0.i5.i.i, !577303, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !577611)
    #dbg_value(i64 %_0.i.i92.i, !577299, !DIExpression(), !577613)
    #dbg_value(i64 %_0.i5.i.i, !577300, !DIExpression(), !577613)
    #dbg_value(i64 %_0.i.i92.i, !577290, !DIExpression(), !577614)
    #dbg_value(i64 %_0.i5.i.i, !577291, !DIExpression(), !577614)
    #dbg_value(i64 %_0.i.i92.i, !577283, !DIExpression(), !577616)
    #dbg_value(i64 %_0.i.i92.i, !577278, !DIExpression(), !577618)
    #dbg_value(i64 %_0.i5.i.i, !577284, !DIExpression(), !577616)
    #dbg_value(i64 %_0.i5.i.i, !577279, !DIExpression(), !577618)
  %_0.i3.i.i = mul i64 %_0.i5.i.i, %_0.i.i92.i, !dbg !577620
    #dbg_value(i64 %_0.i.i92.i, !577309, !DIExpression(), !577621)
    #dbg_value(i64 %_0.i.i92.i, !577311, !DIExpression(), !577623)
    #dbg_value(i64 %_0.i5.i.i, !577310, !DIExpression(), !577621)
    #dbg_value(i64 %_0.i5.i.i, !577312, !DIExpression(), !577623)
  %_5.i.i.i = zext i64 %_0.i.i92.i to i128, !dbg !577624
  %_6.i.i.i = zext i64 %_0.i5.i.i to i128, !dbg !577625
  %_4.i1.i.i = mul nuw i128 %_6.i.i.i, %_5.i.i.i, !dbg !577626
  %_3.i2.i.i = lshr i128 %_4.i1.i.i, 64, !dbg !577627
  %_0.i.i106.i = trunc nuw i128 %_3.i2.i.i to i64, !dbg !577628
    #dbg_value(i64 %_0.i3.i.i, !576968, !DIExpression(), !577629)
    #dbg_value(i64 %_0.i.i106.i, !576970, !DIExpression(), !577629)
    #dbg_value(ptr undef, !573548, !DIExpression(), !577036)
    #dbg_value(i64 %_0.i.i106.i, !573554, !DIExpression(), !577036)
  %117 = or i64 %failed.sroa.0.050.i, %_0.i.i106.i, !dbg !577630
    #dbg_value(i64 %117, !576953, !DIExpression(), !577231)
  %self32.i = getelementptr inbounds nuw i64, ptr %_4.sroa.10.0.i.i, i64 %iter.sroa.0.051.i, !dbg !577631
    #dbg_value(ptr %self32.i, !577323, !DIExpression(), !577632)
    #dbg_value(i64 %_0.i3.i.i, !577324, !DIExpression(), !577632)
  store i64 %_0.i3.i.i, ptr %self32.i, align 8, !dbg !577634, !noalias !577052
    #dbg_value(i64 %_15552.i, !576964, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !577378)
    #dbg_value(ptr undef, !577003, !DIExpression(), !577025)
    #dbg_value(ptr undef, !576991, !DIExpression(), !577021)
    #dbg_value(ptr undef, !577007, !DIExpression(), !577026)
    #dbg_value(ptr poison, !577010, !DIExpression(), !577026)
  %_155.i = add i64 %_15552.i, 1, !dbg !577635
    #dbg_value(i64 poison, !576964, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !577378)
    #dbg_value(i64 %_15552.i, !576964, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !577378)
    #dbg_value(i64 %_15552.i, !576966, !DIExpression(), !577601)
    #dbg_value(i64 %_15552.i, !561317, !DIExpression(), !577032)
    #dbg_value(i64 %_15552.i, !561837, !DIExpression(), !577602)
    #dbg_value(i64 %_15552.i, !561837, !DIExpression(), !577604)
  %118 = getelementptr inbounds nuw i64, ptr %column.val.i.i, i64 %_15552.i, !dbg !577606
  %_0.i.i92.i.1 = load i64, ptr %118, align 8, !dbg !577606, !noalias !577607, !noundef !23
  %119 = getelementptr inbounds nuw i64, ptr %column5.val.i.i, i64 %_15552.i, !dbg !577610
  %_0.i5.i.i.1 = load i64, ptr %119, align 8, !dbg !577610, !noalias !577607, !noundef !23
    #dbg_value(i64 %_0.i.i92.i.1, !577303, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !577611)
    #dbg_value(i64 %_0.i5.i.i.1, !577303, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !577611)
    #dbg_value(i64 %_0.i.i92.i.1, !577299, !DIExpression(), !577613)
    #dbg_value(i64 %_0.i5.i.i.1, !577300, !DIExpression(), !577613)
    #dbg_value(i64 %_0.i.i92.i.1, !577290, !DIExpression(), !577614)
    #dbg_value(i64 %_0.i5.i.i.1, !577291, !DIExpression(), !577614)
    #dbg_value(i64 %_0.i.i92.i.1, !577283, !DIExpression(), !577616)
    #dbg_value(i64 %_0.i.i92.i.1, !577278, !DIExpression(), !577618)
    #dbg_value(i64 %_0.i5.i.i.1, !577284, !DIExpression(), !577616)
    #dbg_value(i64 %_0.i5.i.i.1, !577279, !DIExpression(), !577618)
  %_0.i3.i.i.1 = mul i64 %_0.i5.i.i.1, %_0.i.i92.i.1, !dbg !577620
    #dbg_value(i64 %_0.i.i92.i.1, !577309, !DIExpression(), !577621)
    #dbg_value(i64 %_0.i.i92.i.1, !577311, !DIExpression(), !577623)
    #dbg_value(i64 %_0.i5.i.i.1, !577310, !DIExpression(), !577621)
    #dbg_value(i64 %_0.i5.i.i.1, !577312, !DIExpression(), !577623)
  %_5.i.i.i.1 = zext i64 %_0.i.i92.i.1 to i128, !dbg !577624
  %_6.i.i.i.1 = zext i64 %_0.i5.i.i.1 to i128, !dbg !577625
  %_4.i1.i.i.1 = mul nuw i128 %_6.i.i.i.1, %_5.i.i.i.1, !dbg !577626
  %_3.i2.i.i.1 = lshr i128 %_4.i1.i.i.1, 64, !dbg !577627
  %_0.i.i106.i.1 = trunc nuw i128 %_3.i2.i.i.1 to i64, !dbg !577628
    #dbg_value(i64 %_0.i3.i.i.1, !576968, !DIExpression(), !577629)
    #dbg_value(i64 %_0.i.i106.i.1, !576970, !DIExpression(), !577629)
    #dbg_value(i64 %_0.i.i106.i.1, !573554, !DIExpression(), !577036)
  %120 = or i64 %117, %_0.i.i106.i.1, !dbg !577630
    #dbg_value(i64 %120, !576953, !DIExpression(), !577231)
  %self32.i.1 = getelementptr inbounds nuw i64, ptr %_4.sroa.10.0.i.i, i64 %_15552.i, !dbg !577631
    #dbg_value(ptr %self32.i.1, !577323, !DIExpression(), !577632)
    #dbg_value(i64 %_0.i3.i.i.1, !577324, !DIExpression(), !577632)
  store i64 %_0.i3.i.i.1, ptr %self32.i.1, align 8, !dbg !577634, !noalias !577052
    #dbg_value(i64 %_155.i, !576964, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !577378)
  %_155.i.1 = add i64 %_15552.i, 2, !dbg !577635
    #dbg_value(i64 poison, !576964, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !577378)
  %niter.next.1 = add i64 %niter, 2, !dbg !577379
  %niter.ncmp.1 = icmp eq i64 %niter.next.1, %unroll_iter, !dbg !577379
  br i1 %niter.ncmp.1, label %bb38.i.loopexit144.unr-lcssa, label %bb18.i, !dbg !577379

bb38.thread.i:                                    ; preds = %bb31.preheader.i.thread, %bb17.preheader.split.i, %bb31.preheader.i
    #dbg_value(i64 0, !576953, !DIExpression(), !577231)
    #dbg_value(i64 %index.i, !576946, !DIExpression(DW_OP_LLVM_fragment, 128, 64), !577194)
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %value.i.i), !dbg !577636, !noalias !577132
    #dbg_value(i64 0, !576924, !DIExpression(), !577638)
    #dbg_declare(ptr poison, !576928, !DIExpression(), !577639)
    #dbg_declare(ptr %value.i.i, !577640, !DIExpression(), !577643)
    #dbg_value(ptr undef, !574157, !DIExpression(), !577646)
    #dbg_value(ptr undef, !574158, !DIExpression(), !577646)
  br label %bb41.i, !dbg !577647


```
