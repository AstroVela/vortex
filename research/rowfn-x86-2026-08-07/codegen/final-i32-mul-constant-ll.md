<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# `final-i32-mul-constant.ll`

```ll
bb27.preheader.i.split.us:                        ; preds = %bb27.preheader.i
  br i1 %_3.i5.not.i.i, label %panic.i5.i5.i.invoke.i, label %bb27.i.us.preheader

bb27.i.us.preheader:                              ; preds = %bb27.preheader.i.split.us
  %57 = add nuw nsw i64 %len3.i.i.i, 1, !dbg !566996
  br label %bb27.i.us, !dbg !566996

bb27.i.us:                                        ; preds = %bb27.i.us.preheader, %"_ZN12vortex_array9scalar_fn3row7element9primitive83_$LT$impl$u20$vortex_array..scalar_fn..row..element..InputElement$u20$for$u20$T$GT$3get17h2f14351c5788fe60E.exit8.i.i.i.us"
  %_15854.i.us = phi i64 [ %_158.i.us, %"_ZN12vortex_array9scalar_fn3row7element9primitive83_$LT$impl$u20$vortex_array..scalar_fn..row..element..InputElement$u20$for$u20$T$GT$3get17h2f14351c5788fe60E.exit8.i.i.i.us" ], [ 1, %bb27.i.us.preheader ]
  %iter.sroa.0.053.i.us = phi i64 [ %_15854.i.us, %"_ZN12vortex_array9scalar_fn3row7element9primitive83_$LT$impl$u20$vortex_array..scalar_fn..row..element..InputElement$u20$for$u20$T$GT$3get17h2f14351c5788fe60E.exit8.i.i.i.us" ], [ 0, %bb27.i.us.preheader ]
  %accumulated.sroa.0.052.i.us = phi i1 [ %60, %"_ZN12vortex_array9scalar_fn3row7element9primitive83_$LT$impl$u20$vortex_array..scalar_fn..row..element..InputElement$u20$for$u20$T$GT$3get17h2f14351c5788fe60E.exit8.i.i.i.us" ], [ false, %bb27.i.us.preheader ]
    #dbg_value(i64 %iter.sroa.0.053.i.us, !566730, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !566993)
    #dbg_value(i64 %iter.sroa.0.053.i.us, !566732, !DIExpression(), !567057)
    #dbg_value(ptr %columns.i, !545259, !DIExpression(), !567058)
    #dbg_value(i64 %iter.sroa.0.053.i.us, !545260, !DIExpression(), !567058)
    #dbg_value(ptr %columns.i, !545249, !DIExpression(), !567059)
    #dbg_value(i64 %iter.sroa.0.053.i.us, !545250, !DIExpression(), !567059)
    #dbg_value(ptr %columns.i, !545120, !DIExpression(DW_OP_plus_uconst, 8, DW_OP_stack_value), !567060)
    #dbg_value(ptr %columns.i, !545120, !DIExpression(DW_OP_plus_uconst, 8, DW_OP_stack_value), !567062)
    #dbg_value(ptr %columns.i, !545251, !DIExpression(DW_OP_plus_uconst, 8, DW_OP_stack_value), !567063)
    #dbg_value(i64 %iter.sroa.0.053.i.us, !545125, !DIExpression(), !567062)
  %exitcond35.not = icmp eq i64 %_15854.i.us, %57, !dbg !566996
  br i1 %exitcond35.not, label %panic.i5.i5.i.invoke.i, label %"_ZN12vortex_array9scalar_fn3row7element9primitive83_$LT$impl$u20$vortex_array..scalar_fn..row..element..InputElement$u20$for$u20$T$GT$3get17h2f14351c5788fe60E.exit8.i.i.i.us", !dbg !566996

"_ZN12vortex_array9scalar_fn3row7element9primitive83_$LT$impl$u20$vortex_array..scalar_fn..row..element..InputElement$u20$for$u20$T$GT$3get17h2f14351c5788fe60E.exit8.i.i.i.us": ; preds = %bb27.i.us
    #dbg_value(ptr %columns.i, !545251, !DIExpression(DW_OP_plus_uconst, 8, DW_OP_stack_value), !567063)
    #dbg_value(ptr %columns.i, !545120, !DIExpression(DW_OP_plus_uconst, 8, DW_OP_stack_value), !567062)
  %58 = getelementptr inbounds nuw i32, ptr %data.i6.i.i.i, i64 %iter.sroa.0.053.i.us, !dbg !566996
  %_0.sroa.0.0.i.i.i.us = load i32, ptr %58, align 4, !dbg !567000, !noalias !566784, !noundef !23
    #dbg_value(ptr %14, !545249, !DIExpression(), !567064)
    #dbg_value(i64 %iter.sroa.0.053.i.us, !545250, !DIExpression(), !567064)
    #dbg_value(ptr %14, !545120, !DIExpression(DW_OP_plus_uconst, 8, DW_OP_stack_value), !567065)
    #dbg_value(ptr %14, !545120, !DIExpression(DW_OP_plus_uconst, 8, DW_OP_stack_value), !567067)
    #dbg_value(ptr %14, !545252, !DIExpression(DW_OP_plus_uconst, 8, DW_OP_stack_value), !567068)
    #dbg_value(i64 0, !545125, !DIExpression(), !567065)
  %_0.sroa.0.0.i9.i.i.us = load i32, ptr %data.i6.i7.i.i, align 4, !dbg !567005, !noalias !566784, !noundef !23
    #dbg_value(ptr poison, !567031, !DIExpression(), !567069)
    #dbg_value(ptr poison, !567032, !DIExpression(), !567069)
    #dbg_value(i32 %_0.sroa.0.0.i.i.i.us, !567033, !DIExpression(DW_OP_LLVM_fragment, 0, 32), !567069)
    #dbg_value(i32 %_0.sroa.0.0.i9.i.i.us, !567033, !DIExpression(DW_OP_LLVM_fragment, 32, 32), !567069)
    #dbg_value(i32 %_0.sroa.0.0.i.i.i.us, !567029, !DIExpression(), !567070)
    #dbg_value(i32 %_0.sroa.0.0.i9.i.i.us, !567030, !DIExpression(), !567070)
    #dbg_value(i32 %_0.sroa.0.0.i.i.i.us, !567020, !DIExpression(), !567071)
    #dbg_value(i32 %_0.sroa.0.0.i9.i.i.us, !567021, !DIExpression(), !567071)
    #dbg_value(i32 %_0.sroa.0.0.i.i.i.us, !567015, !DIExpression(), !567072)
    #dbg_value(i32 %_0.sroa.0.0.i.i.i.us, !567010, !DIExpression(), !567073)
    #dbg_value(i32 %_0.sroa.0.0.i9.i.i.us, !567016, !DIExpression(), !567072)
    #dbg_value(i32 %_0.sroa.0.0.i9.i.i.us, !567011, !DIExpression(), !567073)
  %_0.i.i160.i.us = mul i32 %_0.sroa.0.0.i9.i.i.us, %_0.sroa.0.0.i.i.i.us, !dbg !567007
    #dbg_value(i32 %_0.sroa.0.0.i.i.i.us, !567039, !DIExpression(), !567074)
    #dbg_value(i32 %_0.sroa.0.0.i.i.i.us, !567041, !DIExpression(), !567075)
    #dbg_value(i32 %_0.sroa.0.0.i9.i.i.us, !567040, !DIExpression(), !567074)
    #dbg_value(i32 %_0.sroa.0.0.i9.i.i.us, !567042, !DIExpression(), !567075)
  %_4.i1.i.i.us = sext i32 %_0.sroa.0.0.i.i.i.us to i64, !dbg !567035
  %_5.i.i.i.us = sext i32 %_0.sroa.0.0.i9.i.i.us to i64, !dbg !567046
  %product.i.i.i.us = mul nsw i64 %_5.i.i.i.us, %_4.i1.i.i.us, !dbg !567035
    #dbg_value(i64 %product.i.i.i.us, !567043, !DIExpression(), !567076)
  %59 = add nsw i64 %product.i.i.i.us, -2147483648, !dbg !567047
  %_0.sroa.0.0.i.i161.i.us = icmp ult i64 %59, -4294967296, !dbg !567047
    #dbg_value(i1 %_0.sroa.0.0.i.i161.i.us, !566736, !DIExpression(DW_OP_LLVM_convert, 1, DW_ATE_unsigned, DW_OP_LLVM_convert, 8, DW_ATE_unsigned, DW_OP_stack_value), !567077)
    #dbg_value(i32 %_0.i.i160.i.us, !566734, !DIExpression(), !567077)
  %self34.i.us = getelementptr inbounds nuw i32, ptr %_4.sroa.10.0.i.i, i64 %iter.sroa.0.053.i.us, !dbg !567048
    #dbg_value(ptr %self34.i.us, !567052, !DIExpression(), !567078)
    #dbg_value(i32 %_0.i.i160.i.us, !567053, !DIExpression(), !567078)
  store i32 %_0.i.i160.i.us, ptr %self34.i.us, align 4, !dbg !567049, !noalias !566784
    #dbg_value(ptr undef, !541560, !DIExpression(), !566763)
    #dbg_value(i1 %_0.sroa.0.0.i.i161.i.us, !541568, !DIExpression(DW_OP_LLVM_convert, 1, DW_ATE_unsigned, DW_OP_LLVM_convert, 8, DW_ATE_unsigned, DW_OP_stack_value), !566763)
  %60 = or i1 %accumulated.sroa.0.052.i.us, %_0.sroa.0.0.i.i161.i.us, !dbg !567055
    #dbg_value(i8 poison, !566728, !DIExpression(), !566992)
    #dbg_value(i64 %_15854.i.us, !566730, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !566993)
    #dbg_value(ptr undef, !566753, !DIExpression(), !566756)
    #dbg_value(ptr undef, !566744, !DIExpression(), !566749)
    #dbg_value(ptr undef, !566757, !DIExpression(), !566761)
    #dbg_value(ptr poison, !566760, !DIExpression(), !566761)
  %_158.i.us = add i64 %_15854.i.us, 1, !dbg !567079
    #dbg_value(i64 poison, !566730, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !566993)
  %exitcond.not.i.us = icmp eq i64 %_15854.i.us, %len3.i.i.i, !dbg !566994
  br i1 %exitcond.not.i.us, label %bb33.i, label %bb27.i.us, !dbg !566995

```
