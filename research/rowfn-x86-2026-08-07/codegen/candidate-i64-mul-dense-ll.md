<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# `candidate-i64-mul-dense.ll`

```ll
  %_16456.i = phi i64 [ 1, %bb28.lr.ph.i ], [ %_164.i, %bb28.i ]
  %iter.sroa.0.055.i = phi i64 [ 0, %bb28.lr.ph.i ], [ %_16456.i, %bb28.i ]
  %accumulated.sroa.0.054.i = phi i64 [ 0, %bb28.lr.ph.i ], [ %77, %bb28.i ]
    #dbg_value(i64 %iter.sroa.0.055.i, !561376, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !561961)
    #dbg_value(i64 %accumulated.sroa.0.054.i, !561355, !DIExpression(), !561631)
    #dbg_value(i64 %iter.sroa.0.055.i, !561378, !DIExpression(), !562028)
    #dbg_value(ptr undef, !558643, !DIExpression(), !561458)
    #dbg_value(i64 %iter.sroa.0.055.i, !558649, !DIExpression(), !561458)
    #dbg_value(ptr poison, !559346, !DIExpression(), !562029)
    #dbg_value(i64 %iter.sroa.0.055.i, !559351, !DIExpression(), !562029)
    #dbg_value(ptr poison, !559346, !DIExpression(), !562031)
    #dbg_value(i64 %iter.sroa.0.055.i, !559351, !DIExpression(), !562031)
    #dbg_value(ptr poison, !561841, !DIExpression(), !562033)
    #dbg_value(i64 %iter.sroa.0.055.i, !561851, !DIExpression(), !562033)
  %75 = getelementptr inbounds nuw i64, ptr %column5.val.i.i, i64 %iter.sroa.0.055.i, !dbg !562035
  %_0.i5.i.i = load i64, ptr %75, align 8, !dbg !562035, !noalias !562036, !noundef !23
  %76 = getelementptr inbounds nuw i64, ptr %column.val.i.i, i64 %iter.sroa.0.055.i, !dbg !562039
  %_0.i.i123.i = load i64, ptr %76, align 8, !dbg !562039, !noalias !562036, !noundef !23
  %_3.i126.i = getelementptr inbounds nuw i64, ptr %ptr.i.i, i64 %iter.sroa.0.055.i, !dbg !562040
    #dbg_value(ptr poison, !561857, !DIExpression(), !562041)
    #dbg_value(ptr poison, !561867, !DIExpression(), !562041)
    #dbg_value(i64 %_0.i.i123.i, !561868, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !562041)
    #dbg_value(i64 %_0.i5.i.i, !561868, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !562041)
    #dbg_value(ptr %_3.i126.i, !561863, !DIExpression(), !562041)
    #dbg_value(i64 %_0.i.i123.i, !561864, !DIExpression(), !562043)
    #dbg_value(i64 %_0.i.i123.i, !561873, !DIExpression(), !562044)
    #dbg_value(i64 %_0.i5.i.i, !561866, !DIExpression(), !562043)
    #dbg_value(i64 %_0.i5.i.i, !561880, !DIExpression(), !562044)
    #dbg_value(ptr %_3.i126.i, !561879, !DIExpression(), !562044)
    #dbg_value(ptr %_3.i126.i, !561886, !DIExpression(), !562046)
    #dbg_value(i64 %_0.i.i123.i, !561892, !DIExpression(), !562048)
    #dbg_value(i64 %_0.i5.i.i, !561901, !DIExpression(), !562048)
    #dbg_value(i64 %_0.i.i123.i, !561904, !DIExpression(), !562050)
    #dbg_value(i64 %_0.i.i123.i, !561910, !DIExpression(), !562052)
    #dbg_value(i64 %_0.i5.i.i, !561907, !DIExpression(), !562050)
    #dbg_value(i64 %_0.i5.i.i, !561913, !DIExpression(), !562052)
  %_0.i.i128.i = mul i64 %_0.i.i123.i, %_0.i5.i.i, !dbg !562054
    #dbg_value(i64 %_0.i.i123.i, !561917, !DIExpression(), !562055)
    #dbg_value(i64 %_0.i.i123.i, !561923, !DIExpression(), !562057)
    #dbg_value(i64 %_0.i5.i.i, !561922, !DIExpression(), !562055)
    #dbg_value(i64 %_0.i5.i.i, !561925, !DIExpression(), !562057)
  %_4.i1.i.i = sext i64 %_0.i.i123.i to i128, !dbg !562058
  %_5.i.i.i = sext i64 %_0.i5.i.i to i128, !dbg !562059
  %wide.i.i.i = mul nsw i128 %_4.i1.i.i, %_5.i.i.i, !dbg !562058
    #dbg_value(i128 %wide.i.i.i, !561926, !DIExpression(), !562060)
  %kept.i.i.i = trunc i128 %wide.i.i.i to i64, !dbg !562061
    #dbg_value(i64 %kept.i.i.i, !561928, !DIExpression(), !562062)
  %_8.i.i.i = lshr i128 %wide.i.i.i, 64, !dbg !562063
  %discarded.i.i.i = trunc nuw i128 %_8.i.i.i to i64, !dbg !562064
    #dbg_value(i64 %discarded.i.i.i, !561930, !DIExpression(), !562065)
  %_10.i.i.i = ashr i64 %kept.i.i.i, 63, !dbg !562066
  %_9.i.i.i = xor i64 %_10.i.i.i, %discarded.i.i.i, !dbg !562067
    #dbg_value(i64 %_0.i.i128.i, !561881, !DIExpression(), !562068)
    #dbg_value(i64 %_0.i.i128.i, !561889, !DIExpression(), !562046)
    #dbg_value(i64 %_9.i.i.i, !561883, !DIExpression(), !562068)
  store i64 %_0.i.i128.i, ptr %_3.i126.i, align 8, !dbg !562069, !alias.scope !562070, !noalias !561484
    #dbg_value(i64 %_9.i.i.i, !561469, !DIExpression(), !561472)
    #dbg_value(ptr undef, !561463, !DIExpression(), !561472)
  %77 = or i64 %_9.i.i.i, %accumulated.sroa.0.054.i, !dbg !562073
    #dbg_value(i64 %77, !561355, !DIExpression(), !561631)
    #dbg_value(i64 %_16456.i, !561376, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !561961)
    #dbg_value(ptr undef, !561426, !DIExpression(), !561451)
    #dbg_value(ptr undef, !561414, !DIExpression(), !561447)
    #dbg_value(ptr undef, !561430, !DIExpression(), !561452)
    #dbg_value(ptr poison, !561433, !DIExpression(), !561452)
  %_164.i = add i64 %_16456.i, 1, !dbg !562074
    #dbg_value(i64 poison, !561376, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !561961)
  %exitcond.not.i = icmp eq i64 %_16456.i, %4, !dbg !561962
  br i1 %exitcond.not.i, label %bb54.i, label %bb28.i, !dbg !561963

bb59.i:                                           ; preds = %bb24.i
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(48) %71, ptr noundef nonnull align 8 dereferenceable(48) %_59.i, i64 48, i1 false), !dbg !562075, !noalias !561484
  call void @llvm.lifetime.end.p0(i64 48, ptr nonnull %_59.i), !dbg !561556, !noalias !561565
  %_49.sroa.4.0..sroa_idx.i = getelementptr inbounds nuw i8, ptr %_0, i64 16, !dbg !561964
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) %_49.sroa.4.0..sroa_idx.i, ptr noundef nonnull align 8 dereferenceable(24) %_57.i, i64 24, i1 false), !dbg !561556, !noalias !561605
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %_57.i), !dbg !561556, !noalias !561565

```
