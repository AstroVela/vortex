<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# `owned-i64-mul-dense.ll`

```ll
terminate.i:                                      ; preds = %bb57.i
  %78 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
; call core::panicking::panic_in_cleanup
  call void @_ZN4core9panicking16panic_in_cleanup17h8f68387bb6cbbf54E() #88, !dbg !573712, !noalias !573567
  unreachable, !dbg !573712

bb18.i:                                           ; preds = %bb18.i, %bb18.lr.ph.i
  %_15552.i = phi i64 [ 1, %bb18.lr.ph.i ], [ %_155.i, %bb18.i ]
  %iter.sroa.0.051.i = phi i64 [ 0, %bb18.lr.ph.i ], [ %_15552.i, %bb18.i ]
  %failed.sroa.0.050.i = phi i64 [ 0, %bb18.lr.ph.i ], [ %81, %bb18.i ]
    #dbg_value(i64 %iter.sroa.0.051.i, !573477, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !573890)
    #dbg_value(i64 %failed.sroa.0.050.i, !573466, !DIExpression(), !573746)
    #dbg_value(i64 %iter.sroa.0.051.i, !573479, !DIExpression(), !574113)
    #dbg_value(ptr undef, !552190, !DIExpression(), !573546)
    #dbg_value(i64 %iter.sroa.0.051.i, !552196, !DIExpression(), !573546)
    #dbg_value(ptr poison, !552716, !DIExpression(), !574114)
    #dbg_value(i64 %iter.sroa.0.051.i, !552717, !DIExpression(), !574114)
    #dbg_value(ptr poison, !552716, !DIExpression(), !574116)
    #dbg_value(i64 %iter.sroa.0.051.i, !552717, !DIExpression(), !574116)
  %79 = getelementptr inbounds nuw i64, ptr %column.val.i.i, i64 %iter.sroa.0.051.i, !dbg !574118
  %_0.i.i97.i = load i64, ptr %79, align 8, !dbg !574118, !noalias !574119, !noundef !23
  %80 = getelementptr inbounds nuw i64, ptr %column5.val.i.i, i64 %iter.sroa.0.051.i, !dbg !574122
  %_0.i5.i.i = load i64, ptr %80, align 8, !dbg !574122, !noalias !574119, !noundef !23
    #dbg_value(ptr poison, !573820, !DIExpression(), !574123)
    #dbg_value(ptr poison, !573821, !DIExpression(), !574123)
    #dbg_value(i64 %_0.i.i97.i, !573822, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !574123)
    #dbg_value(i64 %_0.i5.i.i, !573822, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !574123)
    #dbg_value(i64 %_0.i.i97.i, !573818, !DIExpression(), !574125)
    #dbg_value(i64 %_0.i5.i.i, !573819, !DIExpression(), !574125)
    #dbg_value(i64 %_0.i.i97.i, !573809, !DIExpression(), !574126)
    #dbg_value(i64 %_0.i5.i.i, !573810, !DIExpression(), !574126)
    #dbg_value(i64 %_0.i.i97.i, !573798, !DIExpression(), !574128)
    #dbg_value(i64 %_0.i.i97.i, !573793, !DIExpression(), !574130)
    #dbg_value(i64 %_0.i5.i.i, !573799, !DIExpression(), !574128)
    #dbg_value(i64 %_0.i5.i.i, !573794, !DIExpression(), !574130)
  %_0.i.i111.i = mul i64 %_0.i5.i.i, %_0.i.i97.i, !dbg !574132
    #dbg_value(i64 %_0.i.i97.i, !573830, !DIExpression(), !574133)
    #dbg_value(i64 %_0.i.i97.i, !573832, !DIExpression(), !574135)
    #dbg_value(i64 %_0.i5.i.i, !573831, !DIExpression(), !574133)
    #dbg_value(i64 %_0.i5.i.i, !573833, !DIExpression(), !574135)
  %_4.i1.i.i = sext i64 %_0.i.i97.i to i128, !dbg !574136
  %_5.i.i.i = sext i64 %_0.i5.i.i to i128, !dbg !574137
  %wide.i.i.i = mul nsw i128 %_5.i.i.i, %_4.i1.i.i, !dbg !574136
    #dbg_value(i128 %wide.i.i.i, !573834, !DIExpression(), !574138)
  %kept.i.i.i = trunc i128 %wide.i.i.i to i64, !dbg !574139
    #dbg_value(i64 %kept.i.i.i, !573836, !DIExpression(), !574140)
  %_8.i.i.i = lshr i128 %wide.i.i.i, 64, !dbg !574141
  %discarded.i.i.i = trunc nuw i128 %_8.i.i.i to i64, !dbg !574142
    #dbg_value(i64 %discarded.i.i.i, !573838, !DIExpression(), !574143)
  %_10.i.i.i = ashr i64 %kept.i.i.i, 63, !dbg !574144
  %_9.i.i.i = xor i64 %_10.i.i.i, %discarded.i.i.i, !dbg !574145
    #dbg_value(i64 %_0.i.i111.i, !573481, !DIExpression(), !574146)
    #dbg_value(i64 %_9.i.i.i, !573483, !DIExpression(), !574146)
    #dbg_value(ptr undef, !573548, !DIExpression(), !573557)
    #dbg_value(i64 %_9.i.i.i, !573554, !DIExpression(), !573557)
  %81 = or i64 %_9.i.i.i, %failed.sroa.0.050.i, !dbg !574147
    #dbg_value(i64 %81, !573466, !DIExpression(), !573746)
  %self32.i = getelementptr inbounds nuw i64, ptr %_4.sroa.10.0.i.i, i64 %iter.sroa.0.051.i, !dbg !574148
    #dbg_value(ptr %self32.i, !573852, !DIExpression(), !574149)
    #dbg_value(i64 %_0.i.i111.i, !573853, !DIExpression(), !574149)
  store i64 %_0.i.i111.i, ptr %self32.i, align 8, !dbg !574151, !noalias !573567
    #dbg_value(i64 %_15552.i, !573477, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !573890)
    #dbg_value(ptr undef, !573517, !DIExpression(), !573539)
    #dbg_value(ptr undef, !573505, !DIExpression(), !573535)
    #dbg_value(ptr undef, !573521, !DIExpression(), !573540)
    #dbg_value(ptr poison, !573524, !DIExpression(), !573540)
  %_155.i = add i64 %_15552.i, 1, !dbg !574152
    #dbg_value(i64 poison, !573477, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !573890)
  %exitcond.not.i = icmp eq i64 %_15552.i, %len3.i4.i.i.fr, !dbg !574153
  br i1 %exitcond.not.i, label %bb38.i, label %bb18.i, !dbg !573891

bb38.thread.i:                                    ; preds = %bb31.preheader.i.thread, %bb17.preheader.split.i, %bb31.preheader.i
    #dbg_value(i64 0, !573466, !DIExpression(), !573746)
    #dbg_value(i64 %index.i, !573459, !DIExpression(DW_OP_LLVM_fragment, 128, 64), !573709)
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %value.i.i), !dbg !574154, !noalias !573647

```
