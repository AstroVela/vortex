<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# `indexed-i64-mul-dense.ll`

```ll
bb15.i.i:                                         ; preds = %bb11.i, %bb15.i.i
  %iter.sroa.0.012.i.i = phi i64 [ %_36.i.i, %bb15.i.i ], [ 0, %bb11.i ]
  %failed.sroa.0.011.i.i = phi i64 [ %79, %bb15.i.i ], [ 0, %bb11.i ]
    #dbg_value(i64 %iter.sroa.0.012.i.i, !576781, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !576835)
    #dbg_value(i64 %failed.sroa.0.011.i.i, !576779, !DIExpression(), !576834)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !576819, !DIExpression(), !577050)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !576812, !DIExpression(), !576813)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !576829, !DIExpression(), !576830)
  %_36.i.i = add nuw i64 %iter.sroa.0.012.i.i, 1, !dbg !577051
    #dbg_value(i64 %_36.i.i, !576781, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !576835)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !576783, !DIExpression(), !577052)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !576806, !DIExpression(), !576807)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !577053, !DIExpression(), !577057)
    #dbg_value(ptr undef, !553722, !DIExpression(), !576801)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !553728, !DIExpression(), !576801)
    #dbg_value(ptr poison, !553800, !DIExpression(), !577059)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !553801, !DIExpression(), !577059)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !553793, !DIExpression(), !577061)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !553785, !DIExpression(), !577063)
    #dbg_value(ptr %column.val.i.i, !553792, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !577061)
    #dbg_value(ptr %column.val.i.i, !553786, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !577063)
    #dbg_value(i64 %len3.i.i.i, !553792, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !577061)
    #dbg_value(i64 %len3.i.i.i, !553786, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !577063)
  %_4.i.i.i.i = getelementptr inbounds nuw i64, ptr %column.val.i.i, i64 %iter.sroa.0.012.i.i, !dbg !577065
  %_0.i.i.i.i = load i64, ptr %_4.i.i.i.i, align 8, !dbg !577066, !noalias !577067, !noundef !23
    #dbg_value(ptr poison, !553800, !DIExpression(), !577071)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !553801, !DIExpression(), !577071)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !553793, !DIExpression(), !577073)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !553785, !DIExpression(), !577075)
    #dbg_value(ptr %column5.val.i.i, !553792, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !577073)
    #dbg_value(ptr %column5.val.i.i, !553786, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !577075)
    #dbg_value(i64 %len3.i.i.i, !553792, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !577073)
    #dbg_value(i64 %len3.i.i.i, !553786, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !577075)
  %_5.i3.i.i.i = icmp ult i64 %iter.sroa.0.012.i.i, %len3.i.i.i, !dbg !577077
  tail call void @llvm.assume(i1 %_5.i3.i.i.i), !dbg !577078
  %_4.i4.i.i.i = getelementptr inbounds nuw i64, ptr %column5.val.i.i, i64 %iter.sroa.0.012.i.i, !dbg !577079
  %_0.i5.i.i.i = load i64, ptr %_4.i4.i.i.i, align 8, !dbg !577080, !noalias !577067, !noundef !23
    #dbg_value(i64 %_0.i.i.i.i, !576785, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !577081)
    #dbg_value(i64 %_0.i5.i.i.i, !576785, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !577081)
    #dbg_value(i64 %_0.i.i.i.i, !577082, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !577090)
    #dbg_value(i64 %_0.i5.i.i.i, !577082, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !577090)
    #dbg_value(ptr poison, !576671, !DIExpression(), !577092)
    #dbg_value(ptr poison, !576672, !DIExpression(), !577092)
    #dbg_value(i64 %_0.i.i.i.i, !576673, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !577092)
    #dbg_value(i64 %_0.i5.i.i.i, !576673, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !577092)
    #dbg_value(i64 %_0.i.i.i.i, !576669, !DIExpression(), !577094)
    #dbg_value(i64 %_0.i5.i.i.i, !576670, !DIExpression(), !577094)
    #dbg_value(i64 %_0.i.i.i.i, !576660, !DIExpression(), !577095)
    #dbg_value(i64 %_0.i5.i.i.i, !576661, !DIExpression(), !577095)
    #dbg_value(i64 %_0.i.i.i.i, !576649, !DIExpression(), !577097)
    #dbg_value(i64 %_0.i.i.i.i, !576644, !DIExpression(), !577099)
    #dbg_value(i64 %_0.i5.i.i.i, !576650, !DIExpression(), !577097)
    #dbg_value(i64 %_0.i5.i.i.i, !576645, !DIExpression(), !577099)
  %_0.i.i.i.i.i = mul i64 %_0.i5.i.i.i, %_0.i.i.i.i, !dbg !577101
    #dbg_value(i64 %_0.i.i.i.i, !576681, !DIExpression(), !577102)
    #dbg_value(i64 %_0.i.i.i.i, !576683, !DIExpression(), !577104)
    #dbg_value(i64 %_0.i5.i.i.i, !576682, !DIExpression(), !577102)
    #dbg_value(i64 %_0.i5.i.i.i, !576684, !DIExpression(), !577104)
  %_4.i1.i.i.i.i = sext i64 %_0.i.i.i.i to i128, !dbg !577105
  %_5.i.i.i.i.i = sext i64 %_0.i5.i.i.i to i128, !dbg !577106
  %wide.i.i.i.i.i = mul nsw i128 %_5.i.i.i.i.i, %_4.i1.i.i.i.i, !dbg !577105
    #dbg_value(i128 %wide.i.i.i.i.i, !576685, !DIExpression(), !577107)
  %kept.i.i.i.i.i = trunc i128 %wide.i.i.i.i.i to i64, !dbg !577108
    #dbg_value(i64 %kept.i.i.i.i.i, !576687, !DIExpression(), !577109)
  %_8.i.i.i.i.i = lshr i128 %wide.i.i.i.i.i, 64, !dbg !577110
  %discarded.i.i.i.i.i = trunc nuw i128 %_8.i.i.i.i.i to i64, !dbg !577111
    #dbg_value(i64 %discarded.i.i.i.i.i, !576689, !DIExpression(), !577112)
  %_10.i.i.i.i.i = ashr i64 %kept.i.i.i.i.i, 63, !dbg !577113
  %_9.i.i.i.i.i = xor i64 %_10.i.i.i.i.i, %discarded.i.i.i.i.i, !dbg !577114
    #dbg_value(i64 poison, !576787, !DIExpression(), !577115)
    #dbg_value(i64 %_9.i.i.i.i.i, !576789, !DIExpression(), !577115)
    #dbg_value(ptr undef, !576390, !DIExpression(), !576799)
    #dbg_value(i64 %_9.i.i.i.i.i, !576396, !DIExpression(), !576799)
  %79 = or i64 %_9.i.i.i.i.i, %failed.sroa.0.011.i.i, !dbg !577116
    #dbg_value(i64 %79, !576779, !DIExpression(), !576834)
    #dbg_value(i64 %_0.i.i.i.i.i, !576787, !DIExpression(), !577115)
    #dbg_value(ptr %_4.sroa.10.0.i.i, !577056, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !577057)
    #dbg_value(i64 %index.i, !577056, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !577057)
  %self4.i.i = getelementptr inbounds nuw i64, ptr %_4.sroa.10.0.i.i, i64 %iter.sroa.0.012.i.i, !dbg !577117
    #dbg_value(ptr %self4.i.i, !577118, !DIExpression(), !577122)
    #dbg_value(i64 %_0.i.i.i.i.i, !577121, !DIExpression(), !577122)
  store i64 %_0.i.i.i.i.i, ptr %self4.i.i, align 8, !dbg !577124, !alias.scope !576795, !noalias !577125
    #dbg_value(ptr undef, !576823, !DIExpression(), !576836)
    #dbg_value(ptr undef, !576818, !DIExpression(), !576837)
    #dbg_value(ptr undef, !576838, !DIExpression(), !576842)
    #dbg_value(ptr poison, !576841, !DIExpression(), !576842)
  %exitcond.not.i.i = icmp eq i64 %_36.i.i, %len3.i4.i.i.fr, !dbg !577126
  br i1 %exitcond.not.i.i, label %bb33.i, label %bb15.i.i, !dbg !576844

```
