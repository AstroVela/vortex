<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# `final-i64-mul-dense.ll`

```ll
bb15.i.i:                                         ; preds = %bb11.i, %bb15.i.i
  %iter.sroa.0.012.i.i = phi i64 [ %_36.i.i, %bb15.i.i ], [ 0, %bb11.i ]
  %failed.sroa.0.011.i.i = phi i64 [ %79, %bb15.i.i ], [ 0, %bb11.i ]
    #dbg_value(i64 %iter.sroa.0.012.i.i, !564425, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !564479)
    #dbg_value(i64 %failed.sroa.0.011.i.i, !564423, !DIExpression(), !564478)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !564463, !DIExpression(), !564694)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !564456, !DIExpression(), !564457)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !564473, !DIExpression(), !564474)
  %_36.i.i = add nuw i64 %iter.sroa.0.012.i.i, 1, !dbg !564695
    #dbg_value(i64 %_36.i.i, !564425, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !564479)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !564427, !DIExpression(), !564696)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !564450, !DIExpression(), !564451)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !564697, !DIExpression(), !564701)
    #dbg_value(ptr undef, !547076, !DIExpression(), !564445)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !547082, !DIExpression(), !564445)
    #dbg_value(ptr poison, !547154, !DIExpression(), !564703)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !547155, !DIExpression(), !564703)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !547147, !DIExpression(), !564705)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !547139, !DIExpression(), !564707)
    #dbg_value(ptr %column.val.i.i, !547146, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !564705)
    #dbg_value(ptr %column.val.i.i, !547140, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !564707)
    #dbg_value(i64 %len3.i.i.i, !547146, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !564705)
    #dbg_value(i64 %len3.i.i.i, !547140, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !564707)
  %_4.i.i.i.i = getelementptr inbounds nuw i64, ptr %column.val.i.i, i64 %iter.sroa.0.012.i.i, !dbg !564709
  %_0.i.i.i.i = load i64, ptr %_4.i.i.i.i, align 8, !dbg !564710, !noalias !564711, !noundef !23
    #dbg_value(ptr poison, !547154, !DIExpression(), !564715)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !547155, !DIExpression(), !564715)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !547147, !DIExpression(), !564717)
    #dbg_value(i64 %iter.sroa.0.012.i.i, !547139, !DIExpression(), !564719)
    #dbg_value(ptr %column5.val.i.i, !547146, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !564717)
    #dbg_value(ptr %column5.val.i.i, !547140, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !564719)
    #dbg_value(i64 %len3.i.i.i, !547146, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !564717)
    #dbg_value(i64 %len3.i.i.i, !547140, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !564719)
  %_5.i3.i.i.i = icmp ult i64 %iter.sroa.0.012.i.i, %len3.i.i.i, !dbg !564721
  tail call void @llvm.assume(i1 %_5.i3.i.i.i), !dbg !564722
  %_4.i4.i.i.i = getelementptr inbounds nuw i64, ptr %column5.val.i.i, i64 %iter.sroa.0.012.i.i, !dbg !564723
  %_0.i5.i.i.i = load i64, ptr %_4.i4.i.i.i, align 8, !dbg !564724, !noalias !564711, !noundef !23
    #dbg_value(i64 %_0.i.i.i.i, !564429, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !564725)
    #dbg_value(i64 %_0.i5.i.i.i, !564429, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !564725)
    #dbg_value(i64 %_0.i.i.i.i, !564726, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !564734)
    #dbg_value(i64 %_0.i5.i.i.i, !564726, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !564734)
    #dbg_value(ptr poison, !564315, !DIExpression(), !564736)
    #dbg_value(ptr poison, !564316, !DIExpression(), !564736)
    #dbg_value(i64 %_0.i.i.i.i, !564317, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !564736)
    #dbg_value(i64 %_0.i5.i.i.i, !564317, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !564736)
    #dbg_value(i64 %_0.i.i.i.i, !564313, !DIExpression(), !564738)
    #dbg_value(i64 %_0.i5.i.i.i, !564314, !DIExpression(), !564738)
    #dbg_value(i64 %_0.i.i.i.i, !564304, !DIExpression(), !564739)
    #dbg_value(i64 %_0.i5.i.i.i, !564305, !DIExpression(), !564739)
    #dbg_value(i64 %_0.i.i.i.i, !564293, !DIExpression(), !564741)
    #dbg_value(i64 %_0.i.i.i.i, !564288, !DIExpression(), !564743)
    #dbg_value(i64 %_0.i5.i.i.i, !564294, !DIExpression(), !564741)
    #dbg_value(i64 %_0.i5.i.i.i, !564289, !DIExpression(), !564743)
  %_0.i.i.i.i.i = mul i64 %_0.i5.i.i.i, %_0.i.i.i.i, !dbg !564745
    #dbg_value(i64 %_0.i.i.i.i, !564325, !DIExpression(), !564746)
    #dbg_value(i64 %_0.i.i.i.i, !564327, !DIExpression(), !564748)
    #dbg_value(i64 %_0.i5.i.i.i, !564326, !DIExpression(), !564746)
    #dbg_value(i64 %_0.i5.i.i.i, !564328, !DIExpression(), !564748)
  %_4.i1.i.i.i.i = sext i64 %_0.i.i.i.i to i128, !dbg !564749
  %_5.i.i.i.i.i = sext i64 %_0.i5.i.i.i to i128, !dbg !564750
  %wide.i.i.i.i.i = mul nsw i128 %_5.i.i.i.i.i, %_4.i1.i.i.i.i, !dbg !564749
    #dbg_value(i128 %wide.i.i.i.i.i, !564329, !DIExpression(), !564751)
  %kept.i.i.i.i.i = trunc i128 %wide.i.i.i.i.i to i64, !dbg !564752
    #dbg_value(i64 %kept.i.i.i.i.i, !564331, !DIExpression(), !564753)
  %_8.i.i.i.i.i = lshr i128 %wide.i.i.i.i.i, 64, !dbg !564754
  %discarded.i.i.i.i.i = trunc nuw i128 %_8.i.i.i.i.i to i64, !dbg !564755
    #dbg_value(i64 %discarded.i.i.i.i.i, !564333, !DIExpression(), !564756)
  %_10.i.i.i.i.i = ashr i64 %kept.i.i.i.i.i, 63, !dbg !564757
  %_9.i.i.i.i.i = xor i64 %_10.i.i.i.i.i, %discarded.i.i.i.i.i, !dbg !564758
    #dbg_value(i64 poison, !564431, !DIExpression(), !564759)
    #dbg_value(i64 %_9.i.i.i.i.i, !564433, !DIExpression(), !564759)
    #dbg_value(ptr undef, !564034, !DIExpression(), !564443)
    #dbg_value(i64 %_9.i.i.i.i.i, !564040, !DIExpression(), !564443)
  %79 = or i64 %_9.i.i.i.i.i, %failed.sroa.0.011.i.i, !dbg !564760
    #dbg_value(i64 %79, !564423, !DIExpression(), !564478)
    #dbg_value(i64 %_0.i.i.i.i.i, !564431, !DIExpression(), !564759)
    #dbg_value(ptr %_4.sroa.10.0.i.i, !564700, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !564701)
    #dbg_value(i64 %index.i, !564700, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !564701)
  %self4.i.i = getelementptr inbounds nuw i64, ptr %_4.sroa.10.0.i.i, i64 %iter.sroa.0.012.i.i, !dbg !564761
    #dbg_value(ptr %self4.i.i, !564762, !DIExpression(), !564766)
    #dbg_value(i64 %_0.i.i.i.i.i, !564765, !DIExpression(), !564766)
  store i64 %_0.i.i.i.i.i, ptr %self4.i.i, align 8, !dbg !564768, !alias.scope !564439, !noalias !564769
    #dbg_value(ptr undef, !564467, !DIExpression(), !564480)
    #dbg_value(ptr undef, !564462, !DIExpression(), !564481)
    #dbg_value(ptr undef, !564482, !DIExpression(), !564486)
    #dbg_value(ptr poison, !564485, !DIExpression(), !564486)
  %exitcond.not.i.i = icmp eq i64 %_36.i.i, %len3.i4.i.i.fr, !dbg !564770
  br i1 %exitcond.not.i.i, label %bb33.i, label %bb15.i.i, !dbg !564488

```
